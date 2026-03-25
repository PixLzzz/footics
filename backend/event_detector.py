"""
Conservative event detection for five-a-side futsal.

Camera-angle agnostic: works with side view, behind-goal, or any angle.

Philosophy: FEWER events but ACCURATE. Every detected event should be real.
Ball detection is unreliable — we require high confidence and corroborating signals.

Detected events:
  - pass: Possession switches between two assigned teammates with ball movement
  - shot: Ball accelerates significantly while a player has possession
  - goal: Multi-signal (shot + ball disappearance + team convergence + play stoppage)
  - assist: Last pass before a goal
  - dribble: Player maintains possession while covering distance with direction changes

NOT auto-detected (too unreliable):
  - interception, tackle, foul, key_pass
"""

import logging
import bisect
import math
from sqlalchemy.orm import Session
from models import (
    Match, Player, TrackingFrame, BallFrame,
    TrackAssignment, MatchEvent,
)

logger = logging.getLogger(__name__)

# ── Thresholds ───────────────────────────────────────────────────────────────

# Ball confidence: only trust detections above this
BALL_MIN_CONFIDENCE = 0.30

# Possession
POSSESSION_RADIUS = 0.065
POSSESSION_HYSTERESIS = 5
POSSESSION_STICKY_BONUS = 0.025
POSSESSION_MIN_SECONDS = 0.5

# Pass
PASS_MIN_BALL_TRAVEL = 0.05
PASS_MAX_BALL_TRAVEL = 0.50       # futsal court is small — reject huge jumps (noise)
PASS_MAX_PLAYER_DIST = 0.50       # tighter for futsal
PASS_COOLDOWN = 2.0

# Shot: ball speed spike (camera-angle agnostic — just uses total speed)
SHOT_MIN_SPEED = 0.06
SHOT_SPEED_RATIO = 2.5
SHOT_COOLDOWN = 4.0

# Goal: multi-signal detection
GOAL_DISAPPEAR_MIN = 0.8          # ball vanishes 0.8s+ after shot
GOAL_CONVERGENCE_RADIUS = 0.15    # players converge within this radius
GOAL_CONVERGENCE_MIN = 3          # at least 3 teammates converge
GOAL_STOPPAGE_SPEED = 0.005       # avg team speed below this = stoppage
GOAL_SIGNAL_MIN = 2               # require at least 2 signals out of 3
GOAL_COOLDOWN = 15.0

# Dribble — futsal-specific (shorter distances, faster game)
DRIBBLE_MIN_DURATION = 1.5        # shorter for futsal (was 2.5)
DRIBBLE_MIN_TRAVEL = 0.06         # shorter for futsal (was 0.10)
DRIBBLE_MIN_DIRECTION_CHANGES = 1 # require at least 1 direction change
DRIBBLE_DIRECTION_ANGLE = 45      # degrees — change > this counts
DRIBBLE_COOLDOWN = 3.0

# Assist
ASSIST_WINDOW = 8.0

# Ball interpolation
INTERP_LINEAR_MAX_GAP = 1.5       # linear interpolation for short gaps
INTERP_SPLINE_MAX_GAP = 3.5       # spline interpolation for medium gaps


def detect_events(match_id: int, db: Session, **kwargs) -> list[dict]:
    """Detect match events from tracking + ball data. Camera-angle agnostic."""
    match = db.query(Match).filter_by(id=match_id).first()
    if not match:
        raise ValueError("Match not found")

    assignments = _load_assignments(match_id, db)
    if not assignments:
        raise ValueError("Aucun joueur assigné aux tracks.")

    player_by_ts, ball_raw = _load_tracking_data(match_id, db)
    all_ts = sorted(player_by_ts.keys())

    if not all_ts:
        return []

    # Filter ball by confidence and interpolate gaps
    ball_hq = {t: pos for t, (pos, conf) in ball_raw.items() if conf >= BALL_MIN_CONFIDENCE}
    ball = _interpolate_ball(all_ts, ball_hq)

    n_hq = len(ball_hq)
    n_interp = len(ball) - n_hq
    logger.info(
        f"Match {match_id}: {len(all_ts)} frames, "
        f"{n_hq} HQ ball, {n_interp} interpolated, "
        f"{len(all_ts) - len(ball)} without ball"
    )

    # Build possession timeline with direction tracking
    possession = _build_possession(all_ts, player_by_ts, ball, assignments)

    # Ball velocity
    ball_ts = sorted(ball.keys())
    velocity = _compute_velocity(ball_ts, ball)

    # Rolling average speed for shot detection
    avg_speed = _rolling_avg_speed(ball_ts, velocity, window=2.0)

    # Ball detection timestamps (original, for disappearance detection)
    ball_original_ts = set(ball_hq.keys())

    # ── Detect events ────────────────────────────────────────────────────
    events = []
    last_events = {}
    pass_history = []
    shot_history = []

    prev_poss = None
    for entry in possession:
        t = entry["t"]
        tid = entry["possessor"]
        bx, by = entry["bx"], entry["by"]
        has_ball = entry["has_ball"]
        poss_dur = entry["poss_duration"]
        travel = entry["player_travel"]

        if tid is None or tid not in assignments:
            prev_poss = entry
            continue

        pid = assignments[tid]["player_id"]

        # ── Pass detection ───────────────────────────────────────────
        if (prev_poss and has_ball and prev_poss["has_ball"]
                and prev_poss["possessor"] is not None
                and prev_poss["possessor"] != tid
                and prev_poss["possessor"] in assignments):

            old_pid = assignments[prev_poss["possessor"]]["player_id"]
            ball_travel = _dist(bx, by, prev_poss["bx"], prev_poss["by"])

            if (PASS_MIN_BALL_TRAVEL <= ball_travel <= PASS_MAX_BALL_TRAVEL
                    and poss_dur >= POSSESSION_MIN_SECONDS):
                passer_pos = _find_player_pos(prev_poss["t"], prev_poss["possessor"], player_by_ts)
                receiver_pos = _find_player_pos(t, tid, player_by_ts)
                if passer_pos and receiver_pos:
                    player_dist = _dist(*passer_pos, *receiver_pos)
                    if player_dist <= PASS_MAX_PLAYER_DIST:
                        added = _add_event(events, last_events, match_id,
                                           old_pid, "pass", prev_poss["t"],
                                           PASS_COOLDOWN, x=bx, y=by)
                        if added:
                            pass_history.append((prev_poss["t"], old_pid, bx, by))

        # ── Shot detection (camera-angle agnostic) ───────────────────
        if has_ball and t in velocity:
            speed = velocity[t]
            avg = avg_speed.get(t, 0.02)

            if (speed >= SHOT_MIN_SPEED
                    and (avg < 0.01 or speed >= avg * SHOT_SPEED_RATIO)):
                added = _add_event(events, last_events, match_id,
                                   pid, "shot", t, SHOT_COOLDOWN, x=bx, y=by)
                if added:
                    shot_history.append((t, pid, bx, by))

        # ── Dribble detection (futsal-specific) ─────────────────────
        if (has_ball
                and poss_dur >= DRIBBLE_MIN_DURATION
                and travel >= DRIBBLE_MIN_TRAVEL):
            dir_changes = entry.get("direction_changes", 0)
            if dir_changes >= DRIBBLE_MIN_DIRECTION_CHANGES:
                _add_event(events, last_events, match_id,
                           pid, "dribble", t, DRIBBLE_COOLDOWN, x=bx, y=by)

        prev_poss = entry

    # ── Goal detection: multi-signal ───────────────────────────────────
    assigned_tids = set(assignments.keys())
    for shot_t, shot_pid, shot_bx, shot_by in shot_history:
        signals = 0

        # Signal 1: ball disappears after shot
        disappear = _measure_disappearance_after(shot_t, all_ts, ball_original_ts)
        if disappear >= GOAL_DISAPPEAR_MIN:
            signals += 1

        # Signal 2: team convergence (celebration) within 5s after shot
        convergence = _measure_team_convergence(
            shot_t, all_ts, player_by_ts, assigned_tids,
            window=5.0, radius=GOAL_CONVERGENCE_RADIUS,
            min_players=GOAL_CONVERGENCE_MIN,
        )
        if convergence:
            signals += 1

        # Signal 3: play stoppage (team stops moving) within 3s after shot
        stoppage = _measure_play_stoppage(
            shot_t, all_ts, player_by_ts, assigned_tids,
            window=3.0, speed_threshold=GOAL_STOPPAGE_SPEED,
        )
        if stoppage:
            signals += 1

        if signals >= GOAL_SIGNAL_MIN:
            added = _add_event(events, last_events, match_id,
                               shot_pid, "goal", shot_t, GOAL_COOLDOWN,
                               x=shot_bx, y=shot_by)
            if added:
                # Assist: last pass before goal by a different player
                for pt, pp_id, px, py in reversed(pass_history):
                    if pp_id != shot_pid and shot_t - pt < ASSIST_WINDOW:
                        _add_event(events, last_events, match_id,
                                   pp_id, "assist", pt, cooldown=10.0, x=px, y=py)
                        break

    # ── Post-processing ──────────────────────────────────────────────────
    # Remove shots that became goals
    goal_set = {(e["player_id"], e["timestamp_seconds"]) for e in events if e["event_type"] == "goal"}
    events[:] = [
        e for e in events
        if not (e["event_type"] == "shot"
                and (e["player_id"], e["timestamp_seconds"]) in goal_set)
    ]

    # Sort and persist
    events.sort(key=lambda e: e["timestamp_seconds"])

    db.query(MatchEvent).filter_by(match_id=match_id).delete()
    for ev in events:
        db.add(MatchEvent(**ev))
    db.commit()

    summary = {}
    for e in events:
        summary[e["event_type"]] = summary.get(e["event_type"], 0) + 1
    logger.info(f"Match {match_id}: detected {len(events)} events — {summary}")

    return events


# ── Data loading ─────────────────────────────────────────────────────────────

def _load_assignments(match_id: int, db: Session) -> dict:
    assignments = {}
    for a in db.query(TrackAssignment).filter_by(match_id=match_id).all():
        if a.is_referee or not a.player_id:
            continue
        player = db.query(Player).filter_by(id=a.player_id).first()
        if player:
            assignments[a.track_id] = {
                "player_id": player.id,
                "team_id": player.team_id,
                "name": player.name,
            }
    return assignments


def _load_tracking_data(match_id: int, db: Session):
    tracking = (
        db.query(TrackingFrame)
        .filter_by(match_id=match_id)
        .order_by(TrackingFrame.timestamp_seconds)
        .all()
    )
    player_by_ts = {}
    for tf in tracking:
        t = tf.timestamp_seconds
        if t not in player_by_ts:
            player_by_ts[t] = []
        player_by_ts[t].append({
            "track_id": tf.track_id,
            "cx": tf.bbox_x + tf.bbox_w / 2,
            "cy": tf.bbox_y + tf.bbox_h / 2,
            "foot_y": tf.bbox_y + tf.bbox_h,
        })

    ball_frames = (
        db.query(BallFrame)
        .filter_by(match_id=match_id)
        .order_by(BallFrame.timestamp_seconds)
        .all()
    )
    ball_raw = {bf.timestamp_seconds: ((bf.x, bf.y), bf.confidence) for bf in ball_frames}

    return player_by_ts, ball_raw


# ── Ball interpolation (linear + spline) ─────────────────────────────────────

def _interpolate_ball(all_ts, ball_hq):
    """Interpolate ball positions in gaps.

    - Gaps <= INTERP_LINEAR_MAX_GAP: linear interpolation
    - Gaps <= INTERP_SPLINE_MAX_GAP: Catmull-Rom spline (uses 2 points before/after)
    """
    result = dict(ball_hq)
    if len(ball_hq) < 2:
        return result

    known_ts = sorted(ball_hq.keys())

    # Build gap list: [(gap_start_idx, gap_end_idx)] in known_ts
    # For each timestamp not in ball_hq, find surrounding known points
    ki = 0
    for t in all_ts:
        if t in result:
            continue
        # Advance ki
        while ki < len(known_ts) - 1 and known_ts[ki + 1] <= t:
            ki += 1

        prev_t = known_ts[ki] if ki < len(known_ts) and known_ts[ki] <= t else None
        next_idx = None
        for j in range(max(ki, 0), len(known_ts)):
            if known_ts[j] > t:
                next_idx = j
                break

        if prev_t is None or next_idx is None:
            continue

        next_t = known_ts[next_idx]
        gap = next_t - prev_t
        prev_idx = ki

        if gap <= INTERP_LINEAR_MAX_GAP:
            # Linear interpolation
            alpha = (t - prev_t) / gap if gap > 0 else 0
            x1, y1 = ball_hq[prev_t]
            x2, y2 = ball_hq[next_t]
            result[t] = (x1 + alpha * (x2 - x1), y1 + alpha * (y2 - y1))

        elif gap <= INTERP_SPLINE_MAX_GAP:
            # Catmull-Rom spline using 4 control points
            p0_idx = prev_idx - 1 if prev_idx > 0 else prev_idx
            p3_idx = next_idx + 1 if next_idx < len(known_ts) - 1 else next_idx

            p0 = ball_hq[known_ts[p0_idx]]
            p1 = ball_hq[prev_t]
            p2 = ball_hq[next_t]
            p3 = ball_hq[known_ts[p3_idx]]

            alpha = (t - prev_t) / gap if gap > 0 else 0
            x = _catmull_rom(alpha, p0[0], p1[0], p2[0], p3[0])
            y = _catmull_rom(alpha, p0[1], p1[1], p2[1], p3[1])
            # Clamp to valid range
            result[t] = (max(0.0, min(1.0, x)), max(0.0, min(1.0, y)))

    return result


def _catmull_rom(t, p0, p1, p2, p3):
    """Catmull-Rom spline interpolation at parameter t in [0,1]."""
    return 0.5 * (
        2 * p1
        + (-p0 + p2) * t
        + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t * t
        + (-p0 + 3 * p1 - 3 * p2 + p3) * t * t * t
    )


# ── Possession timeline ──────────────────────────────────────────────────────

def _build_possession(timestamps, player_by_ts, ball, assignments):
    timeline = []
    current = None
    poss_since = None
    poss_start_pos = None
    challenger = None
    challenger_streak = 0

    # Direction tracking for dribble detection
    direction_history = {}  # tid -> list of (x, y) positions during possession

    for t in timestamps:
        players = player_by_ts.get(t, [])
        ball_pos = ball.get(t)
        has_ball = ball_pos is not None

        if not has_ball:
            poss_dur = (t - poss_since) if poss_since else 0
            travel = 0
            dir_changes = 0
            if current and poss_start_pos:
                for p in players:
                    if p["track_id"] == current:
                        travel = _dist(p["cx"], p["cy"], *poss_start_pos)
                        break
                dir_changes = _count_direction_changes(
                    direction_history.get(current, []))
            timeline.append({
                "t": t, "bx": 0.5, "by": 0.5,
                "possessor": current, "has_ball": False,
                "poss_duration": poss_dur, "player_travel": travel,
                "direction_changes": dir_changes,
            })
            continue

        bx, by = ball_pos
        nearest = None
        nearest_dist = float("inf")
        current_dist = float("inf")

        for p in players:
            tid = p["track_id"]
            if tid not in assignments:
                continue
            d = _dist(p["cx"], p["foot_y"], bx, by)
            effective_d = d
            if tid == current:
                effective_d = max(0, d - POSSESSION_STICKY_BONUS)
                current_dist = d
            if effective_d < nearest_dist:
                nearest_dist = effective_d
                nearest = tid

        if nearest is None or nearest_dist > POSSESSION_RADIUS:
            if current is not None and current_dist <= POSSESSION_RADIUS * 1.8:
                pass
            else:
                current = None
                poss_since = None
                poss_start_pos = None
                challenger = None
                challenger_streak = 0
                direction_history.clear()
        elif nearest != current:
            if nearest == challenger:
                challenger_streak += 1
            else:
                challenger = nearest
                challenger_streak = 1
            if challenger_streak >= POSSESSION_HYSTERESIS or current is None:
                current = nearest
                poss_since = t
                direction_history[current] = []
                for p in players:
                    if p["track_id"] == nearest:
                        poss_start_pos = (p["cx"], p["cy"])
                        direction_history[current].append(poss_start_pos)
                        break
                challenger = None
                challenger_streak = 0
        else:
            challenger = None
            challenger_streak = 0

        # Track position for direction changes
        if current:
            for p in players:
                if p["track_id"] == current:
                    if current not in direction_history:
                        direction_history[current] = []
                    direction_history[current].append((p["cx"], p["cy"]))
                    break

        poss_dur = (t - poss_since) if poss_since else 0
        travel = 0
        dir_changes = 0
        if current and poss_start_pos:
            for p in players:
                if p["track_id"] == current:
                    travel = _dist(p["cx"], p["cy"], *poss_start_pos)
                    break
            dir_changes = _count_direction_changes(
                direction_history.get(current, []))

        timeline.append({
            "t": t, "bx": bx, "by": by,
            "possessor": current, "has_ball": True,
            "poss_duration": poss_dur, "player_travel": travel,
            "direction_changes": dir_changes,
        })

    return timeline


def _count_direction_changes(positions):
    """Count significant direction changes in a position trace."""
    if len(positions) < 3:
        return 0

    # Sample every few positions to reduce noise
    step = max(1, len(positions) // 20)
    sampled = positions[::step]
    if len(sampled) < 3:
        return 0

    changes = 0
    for i in range(1, len(sampled) - 1):
        x0, y0 = sampled[i - 1]
        x1, y1 = sampled[i]
        x2, y2 = sampled[i + 1]

        dx1, dy1 = x1 - x0, y1 - y0
        dx2, dy2 = x2 - x1, y2 - y1

        len1 = math.hypot(dx1, dy1)
        len2 = math.hypot(dx2, dy2)
        if len1 < 0.001 or len2 < 0.001:
            continue

        # Angle between consecutive direction vectors
        cos_angle = (dx1 * dx2 + dy1 * dy2) / (len1 * len2)
        cos_angle = max(-1.0, min(1.0, cos_angle))
        angle_deg = math.degrees(math.acos(cos_angle))

        if angle_deg >= DRIBBLE_DIRECTION_ANGLE:
            changes += 1

    return changes


# ── Ball velocity ────────────────────────────────────────────────────────────

def _compute_velocity(ball_ts, ball):
    velocity = {}
    for i in range(1, len(ball_ts)):
        t_prev, t_curr = ball_ts[i - 1], ball_ts[i]
        dt = t_curr - t_prev
        if dt <= 0 or dt > 0.5:
            velocity[t_curr] = 0
            continue
        x1, y1 = ball[t_prev]
        x2, y2 = ball[t_curr]
        velocity[t_curr] = _dist(x1, y1, x2, y2) / dt
    if ball_ts:
        velocity[ball_ts[0]] = 0
    return velocity


def _rolling_avg_speed(ball_ts, velocity, window=2.0):
    """Rolling average ball speed over a time window."""
    avg = {}
    for i, t in enumerate(ball_ts):
        speeds = []
        for j in range(max(0, i - 30), i):
            t2 = ball_ts[j]
            if t - t2 > window:
                continue
            if t2 in velocity:
                speeds.append(velocity[t2])
        avg[t] = sum(speeds) / len(speeds) if speeds else 0.01
    return avg


# ── Goal detection signals ───────────────────────────────────────────────────

def _measure_disappearance_after(t, all_ts, ball_ts_set):
    """After a shot at time t, measure how long ball disappears."""
    idx = bisect.bisect_right(all_ts, t)

    gap_start = None
    for j in range(idx, min(idx + 60, len(all_ts))):
        if all_ts[j] not in ball_ts_set:
            gap_start = all_ts[j]
            break

    if gap_start is None:
        return 0

    gap_end = gap_start
    gap_idx = bisect.bisect_left(all_ts, gap_start)
    for j in range(gap_idx, len(all_ts)):
        if all_ts[j] in ball_ts_set:
            break
        gap_end = all_ts[j]

    return gap_end - gap_start


def _measure_team_convergence(shot_t, all_ts, player_by_ts, assigned_tids,
                               window=5.0, radius=0.15, min_players=3):
    """Check if teammates converge (celebrate) after a shot."""
    idx = bisect.bisect_right(all_ts, shot_t)
    end_t = shot_t + window

    for j in range(idx, len(all_ts)):
        t = all_ts[j]
        if t > end_t:
            break

        players = player_by_ts.get(t, [])
        team_players = [p for p in players if p["track_id"] in assigned_tids]
        if len(team_players) < min_players:
            continue

        # Check if any cluster of min_players exists within radius
        for anchor in team_players:
            nearby = sum(
                1 for p in team_players
                if _dist(p["cx"], p["cy"], anchor["cx"], anchor["cy"]) <= radius
            )
            if nearby >= min_players:
                return True

    return False


def _measure_play_stoppage(shot_t, all_ts, player_by_ts, assigned_tids,
                            window=3.0, speed_threshold=0.005):
    """Check if play stops (all players slow down) after a shot."""
    idx = bisect.bisect_right(all_ts, shot_t)
    end_t = shot_t + window

    # Collect average team speed in the window 1-3s after shot
    prev_positions = {}
    low_speed_frames = 0
    total_frames = 0

    for j in range(idx, len(all_ts)):
        t = all_ts[j]
        if t > end_t:
            break
        if t < shot_t + 1.0:
            # Skip first second (ball still in flight)
            players = player_by_ts.get(t, [])
            for p in players:
                if p["track_id"] in assigned_tids:
                    prev_positions[p["track_id"]] = (p["cx"], p["cy"])
            continue

        players = player_by_ts.get(t, [])
        team_speeds = []
        for p in players:
            tid = p["track_id"]
            if tid not in assigned_tids:
                continue
            if tid in prev_positions:
                d = _dist(p["cx"], p["cy"], *prev_positions[tid])
                team_speeds.append(d)
            prev_positions[tid] = (p["cx"], p["cy"])

        if team_speeds:
            avg_team_speed = sum(team_speeds) / len(team_speeds)
            total_frames += 1
            if avg_team_speed < speed_threshold:
                low_speed_frames += 1

    # Stoppage if most frames show low speed
    return total_frames > 0 and low_speed_frames / total_frames > 0.5


# ── Utilities ────────────────────────────────────────────────────────────────

def _find_player_pos(t, track_id, player_by_ts):
    players = player_by_ts.get(t, [])
    for p in players:
        if p["track_id"] == track_id:
            return (p["cx"], p["cy"])
    return None


def _dist(x1, y1, x2, y2):
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5


def _add_event(events, last_events, match_id, player_id, event_type,
               timestamp, cooldown, x=None, y=None) -> bool:
    key = (event_type, player_id)
    last_t = last_events.get(key, -999)
    if timestamp - last_t < cooldown:
        return False

    event = {
        "match_id": match_id,
        "player_id": player_id,
        "event_type": event_type,
        "timestamp_seconds": round(timestamp, 2),
    }
    if x is not None:
        event["x"] = round(x, 4)
    if y is not None:
        event["y"] = round(y, 4)

    events.append(event)
    last_events[key] = timestamp
    return True
