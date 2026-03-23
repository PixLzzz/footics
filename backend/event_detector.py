"""
Automatic event detection from player tracking + ball tracking data.

Single-team analysis: detects events for the tracked team only.
Detects: passes, interceptions, shots, goals, assists, key passes, dribbles.

Pipeline:
  1. Load & index tracking data
  2. Interpolate ball positions to fill gaps (ball detected in ~40% of frames)
  3. Build possession timeline with hysteresis (anti-flicker)
  4. Compute ball velocity between frames
  5. Derive events from possession changes + ball trajectory
  6. Second pass: confirm goals via ball disappearance
  7. Post-processing: remove shot→goal duplicates, detect assists & key passes
"""

import logging
from sqlalchemy.orm import Session
from models import (
    Match, Player, TrackingFrame, BallFrame,
    TrackAssignment, MatchEvent,
)

logger = logging.getLogger(__name__)

# ── Thresholds tuned for futsal / five-a-side ────────────────────────────────
# Futsal: smaller field, players closer together, faster ball movement.
# All distances in normalized 0-1 coordinates.

# Possession
POSSESSION_DIST = 0.08            # Tighter than 11v11 — players closer to ball in futsal
POSSESSION_HYSTERESIS = 3         # 3 consecutive frames before switching (anti-flicker)
POSSESSION_STICKY_BONUS = 0.020   # Strong sticky bonus — fewer false switches
POSSESSION_MIN_DURATION = 0.4     # Slightly longer to filter noise

# Passes
PASS_MIN_BALL_TRAVEL = 0.04       # Shorter passes in futsal are valid
EVENT_COOLDOWN = 1.5              # Faster pace → shorter cooldown

# Shots & Goals
GOAL_ZONE_X = 0.08               # Wider goal zone — futsal goals take up more of the frame
GOAL_Y_MIN = 0.15                # Wider vertical range for goal detection
GOAL_Y_MAX = 0.85
SHOT_ZONE_X = 0.30               # Larger shot zone — in futsal you shoot from further relative to field
SHOT_MIN_SPEED = 0.025           # Lower threshold — smaller field = lower absolute speed
GOAL_DISAPPEAR_MIN = 0.25        # Ball vanishes faster in futsal (net catch)

# Dribbles
DRIBBLE_MIN_PLAYER_TRAVEL = 0.05 # Less travel needed — field is smaller
DRIBBLE_MIN_DURATION = 1.2       # Shorter dribbles count

# Assists & key passes
ASSIST_WINDOW = 10.0             # Faster play → shorter assist window
KEY_PASS_WINDOW = 8.0

# Ball interpolation
INTERP_GAP_MAX_CONFIDENT = 2.5   # Slightly more generous for noisy ball detection
INTERP_GAP_MAX_LOW_CONF = 6.0    # Allow longer interpolation gaps


# ── Main entry point ─────────────────────────────────────────────────────────

def detect_events(match_id: int, db: Session, attacks_right: bool = True) -> list[dict]:
    """
    Analyze tracking + ball data to auto-detect match events for the tracked team.

    Returns list of detected events as dicts.
    """
    match = db.query(Match).filter_by(id=match_id).first()
    if not match:
        raise ValueError("Match not found")

    # ── Load data ─────────────────────────────────────────────────────────
    assignments = _load_assignments(match_id, db)
    if not assignments:
        raise ValueError("Aucun joueur assigné aux tracks. Assignez d'abord les joueurs.")

    player_by_ts, ball_by_ts = _load_tracking_data(match_id, db)

    all_player_ts = sorted(player_by_ts.keys())

    if not all_player_ts:
        logger.warning(f"Match {match_id}: no player tracking data")
        return []

    # ── Interpolate ball positions to fill gaps ───────────────────────────
    interpolated_ball = _interpolate_ball_positions(all_player_ts, ball_by_ts)

    # Use ALL player timestamps (not just the intersection with ball)
    timestamps = all_player_ts

    ball_ts_set_original = set(ball_by_ts.keys())
    n_original = len(ball_ts_set_original & set(all_player_ts))
    n_interpolated = sum(1 for t in all_player_ts if t in interpolated_ball and t not in ball_ts_set_original)
    n_none = sum(1 for t in all_player_ts if t not in interpolated_ball)
    logger.info(
        f"Match {match_id}: {len(all_player_ts)} player frames, "
        f"{n_original} with real ball, {n_interpolated} interpolated, "
        f"{n_none} without ball data"
    )

    # Team attacks right → goal is at x=1.0; attacks left → goal at x=0.0
    target_goal_x = 1.0 if attacks_right else 0.0

    # ── Phase 1: Possession timeline with hysteresis ──────────────────────
    possession_tl = _build_possession_timeline(
        timestamps, player_by_ts, interpolated_ball, assignments
    )

    # ── Phase 2: Ball velocity (only for timestamps with ball data) ──────
    ts_with_ball = sorted(t for t in timestamps if t in interpolated_ball)
    ball_velocity = _compute_ball_velocity(ts_with_ball, interpolated_ball)

    # ── Phase 3: Ball gap map (for goal confirmation, uses original ball) ─
    ball_gaps = _build_ball_gaps(all_player_ts, ball_ts_set_original)

    # ── Phase 4: Derive events from timeline ──────────────────────────────
    events = []
    last_events = {}  # (event_type, player_id) → last timestamp
    pass_chain = []   # [(t, player_id, ball_x, ball_y)] for assists/key_pass

    prev = None
    for entry in possession_tl:
        t = entry["t"]
        bx, by = entry["bx"], entry["by"]
        track_id = entry["possessor"]
        ball_known = entry.get("ball_known", True)

        if track_id is None or track_id not in assignments:
            prev = entry
            continue

        info = assignments[track_id]
        pid = info["player_id"]

        # Skip ball-dependent logic when ball position is unknown
        if not ball_known:
            prev = entry
            continue

        dist_to_goal = abs(bx - target_goal_x)

        # ── Possession change events ──────────────────────────────────
        if (prev and prev["possessor"] is not None
                and prev["possessor"] != track_id
                and prev["possessor"] in assignments
                and prev.get("ball_known", True)):
            old = assignments[prev["possessor"]]
            poss_dur = entry.get("poss_duration", 0)

            if poss_dur >= POSSESSION_MIN_DURATION:
                ball_travel = _dist(bx, by, prev["bx"], prev["by"])

                # Both players are from the same team → PASS
                if ball_travel >= PASS_MIN_BALL_TRAVEL:
                    added = _add_event(events, last_events, match_id,
                                       old["player_id"], "pass", t, EVENT_COOLDOWN,
                                       x=bx, y=by)
                    if added:
                        pass_chain.append((t, old["player_id"], bx, by))
            else:
                # Very short possession by someone else then back → possible INTERCEPTION
                # (ball was briefly contested, current player won it back)
                _add_event(events, last_events, match_id,
                           pid, "interception", t, EVENT_COOLDOWN,
                           x=bx, y=by)

        # ── Shot detection (ball moving fast toward goal) ─────────────
        if GOAL_ZONE_X <= dist_to_goal < SHOT_ZONE_X:
            speed = ball_velocity.get(t, 0)
            speed_threshold = SHOT_MIN_SPEED
            if dist_to_goal < SHOT_ZONE_X * 0.4:
                speed_threshold *= 0.6  # easier to detect close shots

            if speed >= speed_threshold:
                added = _add_event(events, last_events, match_id,
                                   pid, "shot", t, cooldown=3.0,
                                   x=bx, y=by)
                if added:
                    # Key pass: last pass that led to this shot
                    for pt, pp_id, px, py in reversed(pass_chain):
                        if pp_id != pid and t - pt < KEY_PASS_WINDOW:
                            _add_event(events, last_events, match_id,
                                       pp_id, "key_pass", pt, cooldown=5.0,
                                       x=px, y=py)
                            break

        # ── Shot heuristic: possession near goal + ball disappears ────
        if dist_to_goal < SHOT_ZONE_X and track_id in assignments:
            # Look ahead: does ball disappear soon?
            _check_shot_on_disappearance(
                t, pid, bx, by, dist_to_goal, all_player_ts, ball_ts_set_original,
                events, last_events, match_id, pass_chain
            )

        # ── Dribble detection ─────────────────────────────────────────
        poss_dur = entry.get("poss_duration", 0)
        player_travel = entry.get("player_travel", 0)
        if poss_dur >= DRIBBLE_MIN_DURATION and player_travel >= DRIBBLE_MIN_PLAYER_TRAVEL:
            _add_event(events, last_events, match_id,
                       pid, "dribble", t, cooldown=DRIBBLE_MIN_DURATION,
                       x=bx, y=by)

        prev = entry

    # ── Phase 5: Goal confirmation (ball disappearance after goal zone) ───
    goal_candidates = _detect_goal_candidates(
        all_player_ts, player_by_ts, ball_by_ts, interpolated_ball, ball_gaps,
        assignments, target_goal_x, possession_tl, timestamps
    )

    for gt, gpid, gx, gy in goal_candidates:
        added = _add_event(events, last_events, match_id,
                           gpid, "goal", gt, cooldown=10.0,
                           x=gx, y=gy)
        if added:
            # Assist: last pass before this goal
            for pt, pp_id, px, py in reversed(pass_chain):
                if pp_id != gpid and gt - pt < ASSIST_WINDOW:
                    _add_event(events, last_events, match_id,
                               pp_id, "assist", pt, cooldown=10.0,
                               x=px, y=py)
                    break

    # ── Phase 6: Post-processing ──────────────────────────────────────────
    goal_times = {e["timestamp_seconds"] for e in events if e["event_type"] == "goal"}
    goal_players = {e["player_id"] for e in events if e["event_type"] == "goal"}

    # Remove shots that became goals (within 5s, same player)
    events[:] = [
        e for e in events
        if not (
            e["event_type"] == "shot"
            and e["player_id"] in goal_players
            and any(abs(e["timestamp_seconds"] - gt) < 5 for gt in goal_times)
        )
    ]

    # ── Sort & persist ────────────────────────────────────────────────────
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


# ── Ball interpolation ────────────────────────────────────────────────────────

def _interpolate_ball_positions(all_player_ts, ball_by_ts):
    """
    Interpolate ball positions for player timestamps that lack ball data.

    Strategy:
    - Gap < 2.0s: linear interpolation (confident)
    - Gap 2-5s: linear interpolation (low confidence, still usable)
    - Gap > 5s: no interpolation (returns None for that timestamp)

    Returns augmented dict: {timestamp: (x, y)} containing both original and
    interpolated ball positions. Timestamps with gaps > 5s are omitted.
    """
    result = {}

    # Copy all original ball positions
    for t, pos in ball_by_ts.items():
        result[t] = pos

    if not ball_by_ts:
        return result

    # Build sorted list of known ball timestamps
    known_ts = sorted(ball_by_ts.keys())

    if len(known_ts) < 2:
        return result

    # For each player timestamp without ball data, try to interpolate
    ki = 0  # index into known_ts
    for t in all_player_ts:
        if t in result:
            continue  # already have ball data

        # Advance ki so known_ts[ki] is the last known ts <= t
        while ki < len(known_ts) - 1 and known_ts[ki + 1] <= t:
            ki += 1

        # Find surrounding known timestamps
        prev_t = None
        next_t = None

        if ki < len(known_ts) and known_ts[ki] <= t:
            prev_t = known_ts[ki]
        elif ki > 0:
            prev_t = known_ts[ki - 1]

        # Find next known timestamp after t
        for j in range(max(ki, 0), len(known_ts)):
            if known_ts[j] > t:
                next_t = known_ts[j]
                break

        if prev_t is None and next_t is None:
            continue

        # Only prev known
        if prev_t is not None and next_t is None:
            gap = t - prev_t
            if gap <= INTERP_GAP_MAX_LOW_CONF:
                result[t] = ball_by_ts[prev_t]  # hold last known position
            continue

        # Only next known
        if prev_t is None and next_t is not None:
            gap = next_t - t
            if gap <= INTERP_GAP_MAX_LOW_CONF:
                result[t] = ball_by_ts[next_t]  # use next known position
            continue

        # Both prev and next known — linear interpolation
        gap = next_t - prev_t
        if gap > INTERP_GAP_MAX_LOW_CONF:
            continue  # gap too large, skip

        # Linear interpolation factor
        alpha = (t - prev_t) / gap if gap > 0 else 0.0
        x1, y1 = ball_by_ts[prev_t]
        x2, y2 = ball_by_ts[next_t]
        ix = x1 + alpha * (x2 - x1)
        iy = y1 + alpha * (y2 - y1)
        result[t] = (ix, iy)

    return result


# ── Data loading ──────────────────────────────────────────────────────────────

def _load_assignments(match_id: int, db: Session) -> dict:
    """Load track_id → player info mapping. Excludes referee tracks."""
    assignments = {}
    for a in db.query(TrackAssignment).filter_by(match_id=match_id).all():
        if a.is_referee:
            continue  # skip referee — excluded from all event analysis
        if not a.player_id:
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
    """Load player tracking and ball data, indexed by timestamp."""
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
            "foot_y": tf.bbox_y + tf.bbox_h,
            "cy": tf.bbox_y + tf.bbox_h / 2,
        })

    ball_frames = (
        db.query(BallFrame)
        .filter_by(match_id=match_id)
        .order_by(BallFrame.timestamp_seconds)
        .all()
    )
    ball_by_ts = {bf.timestamp_seconds: (bf.x, bf.y) for bf in ball_frames}

    return player_by_ts, ball_by_ts


# ── Possession timeline with hysteresis ───────────────────────────────────────

def _build_possession_timeline(timestamps, player_by_ts, ball_by_ts, assignments):
    """
    Build frame-by-frame possession with hysteresis to prevent flickering.

    Handles None ball positions: maintains current possessor without switching,
    and appends timeline entry with last known ball position.

    Returns list of dicts with:
      t, bx, by, possessor (track_id or None), poss_duration, player_travel, ball_known
    """
    timeline = []
    current_possessor = None
    possession_since = None
    possession_start_pos = None
    challenger = None
    challenger_streak = 0
    last_known_bx = 0.5
    last_known_by = 0.5

    for t in timestamps:
        ball_pos = ball_by_ts.get(t)
        players = player_by_ts.get(t, [])

        if ball_pos is None:
            # No ball data: maintain current possessor, use last known ball position
            poss_dur = (t - possession_since) if possession_since else 0
            player_travel = 0
            if possession_start_pos and current_possessor:
                for p in players:
                    if p["track_id"] == current_possessor:
                        player_travel = _dist(p["cx"], p["cy"],
                                              possession_start_pos[0], possession_start_pos[1])
                        break

            timeline.append({
                "t": t, "bx": last_known_bx, "by": last_known_by,
                "possessor": current_possessor,
                "poss_duration": poss_dur,
                "player_travel": player_travel,
                "ball_known": False,
            })
            continue

        bx, by = ball_pos
        last_known_bx = bx
        last_known_by = by

        nearest_track = None
        nearest_dist = float("inf")
        current_poss_dist = float("inf")

        for p in players:
            tid = p["track_id"]
            if tid not in assignments:
                continue
            d = _dist(p["cx"], p["foot_y"], bx, by)

            effective_d = d
            if tid == current_possessor:
                effective_d = max(0, d - POSSESSION_STICKY_BONUS)
                current_poss_dist = d

            if effective_d < nearest_dist:
                nearest_dist = effective_d
                nearest_track = tid

        if nearest_track is None or nearest_dist > POSSESSION_DIST:
            if current_possessor is not None and current_poss_dist <= POSSESSION_DIST * 1.5:
                pass
            else:
                current_possessor = None
                possession_since = None
                possession_start_pos = None
                challenger = None
                challenger_streak = 0

            poss_dur = (t - possession_since) if possession_since else 0
            timeline.append({
                "t": t, "bx": bx, "by": by,
                "possessor": current_possessor,
                "poss_duration": poss_dur,
                "player_travel": 0,
                "ball_known": True,
            })
            continue

        if nearest_track != current_possessor:
            if nearest_track == challenger:
                challenger_streak += 1
            else:
                challenger = nearest_track
                challenger_streak = 1

            if challenger_streak >= POSSESSION_HYSTERESIS or current_possessor is None:
                current_possessor = nearest_track
                possession_since = t
                for p in players:
                    if p["track_id"] == nearest_track:
                        possession_start_pos = (p["cx"], p["cy"])
                        break
                challenger = None
                challenger_streak = 0
        else:
            challenger = None
            challenger_streak = 0

        poss_dur = (t - possession_since) if possession_since else 0
        player_travel = 0
        if possession_start_pos and current_possessor:
            for p in players:
                if p["track_id"] == current_possessor:
                    player_travel = _dist(p["cx"], p["cy"],
                                          possession_start_pos[0], possession_start_pos[1])
                    break

        timeline.append({
            "t": t, "bx": bx, "by": by,
            "possessor": current_possessor,
            "poss_duration": poss_dur,
            "player_travel": player_travel,
            "ball_known": True,
        })

    return timeline


# ── Ball velocity ─────────────────────────────────────────────────────────────

def _compute_ball_velocity(timestamps, ball_by_ts):
    """Compute ball speed between consecutive frames. Returns {timestamp: speed}."""
    velocity = {}
    for i in range(1, len(timestamps)):
        t_prev, t_curr = timestamps[i - 1], timestamps[i]
        dt = t_curr - t_prev
        if dt <= 0:
            velocity[t_curr] = 0
            continue
        pos_prev = ball_by_ts.get(t_prev)
        pos_curr = ball_by_ts.get(t_curr)
        if pos_prev is None or pos_curr is None:
            velocity[t_curr] = 0
            continue
        x1, y1 = pos_prev
        x2, y2 = pos_curr
        dist = _dist(x1, y1, x2, y2)
        velocity[t_curr] = dist / dt
    if timestamps:
        velocity[timestamps[0]] = 0
    return velocity


# ── Ball gap map ──────────────────────────────────────────────────────────────

def _build_ball_gaps(all_player_ts, ball_ts_set):
    """For each timestamp without ball detection, compute gap duration until ball reappears."""
    ball_gaps = {}
    for i, t in enumerate(all_player_ts):
        if t not in ball_ts_set:
            gap_end = t
            for t2 in all_player_ts[i:]:
                if t2 in ball_ts_set:
                    gap_end = t2
                    break
                gap_end = t2
            ball_gaps[t] = gap_end - t
    return ball_gaps


# ── Shot heuristic: ball disappearance near goal ─────────────────────────────

def _check_shot_on_disappearance(t, pid, bx, by, dist_to_goal, all_player_ts,
                                  ball_ts_set_original, events, last_events,
                                  match_id, pass_chain):
    """
    Heuristic: if player has possession near goal and ball disappears from
    original detections for >0.5s, register as shot candidate.
    """
    # Find where t is in the timeline
    try:
        idx = all_player_ts.index(t)
    except ValueError:
        return

    # Look ahead for ball disappearance
    disappear_start = None
    for j in range(idx + 1, min(idx + 30, len(all_player_ts))):
        t2 = all_player_ts[j]
        if t2 in ball_ts_set_original:
            if disappear_start is not None:
                break  # ball came back
            continue
        else:
            if disappear_start is None:
                disappear_start = t2

    if disappear_start is None:
        return

    # Find how long ball is gone
    disappear_end = disappear_start
    for j in range(all_player_ts.index(disappear_start), len(all_player_ts)):
        t2 = all_player_ts[j]
        if t2 in ball_ts_set_original:
            break
        disappear_end = t2

    gap_duration = disappear_end - disappear_start
    if gap_duration < 0.5:
        return

    added = _add_event(events, last_events, match_id,
                       pid, "shot", t, cooldown=3.0,
                       x=bx, y=by)
    if added:
        for pt, pp_id, px, py in reversed(pass_chain):
            if pp_id != pid and t - pt < KEY_PASS_WINDOW:
                _add_event(events, last_events, match_id,
                           pp_id, "key_pass", pt, cooldown=5.0,
                           x=px, y=py)
                break


# ── Goal detection (ball disappearance confirmation) ──────────────────────────

def _detect_goal_candidates(all_player_ts, player_by_ts, ball_by_ts, interpolated_ball,
                            ball_gaps, assignments, target_goal_x, possession_tl,
                            timestamps):
    """
    Detect goals by finding frames where:
    1. Ball enters goal zone (correct side for the team)
    2. Ball Y is within goal post range
    3. Ball then disappears for GOAL_DISAPPEAR_MIN+ seconds

    Also detects goals via possession near goal + ball disappearance,
    even without ball being seen in the exact goal zone.
    """
    poss_at_t = {}
    for entry in possession_tl:
        if entry["possessor"] and entry["possessor"] in assignments:
            info = assignments[entry["possessor"]]
            poss_at_t[entry["t"]] = info["player_id"]

    ball_ts_set = set(ball_by_ts.keys())
    candidates = []
    candidate = None  # (t, player_id, ball_x, ball_y)

    # Strategy 1: Ball seen in goal zone + disappearance (original logic with relaxed zone)
    for t in all_player_ts:
        if t in ball_ts_set:
            bx, by = ball_by_ts[t]

            in_goal_zone = abs(bx - target_goal_x) < GOAL_ZONE_X
            y_in_goal = GOAL_Y_MIN <= by <= GOAL_Y_MAX

            if in_goal_zone and y_in_goal:
                possessor_pid = _find_possessor_near_t(t, poss_at_t, timestamps, window=3.0)
                if possessor_pid:
                    candidate = (t, possessor_pid, bx, by)
        else:
            if candidate is not None:
                gap = ball_gaps.get(t, 0)
                if gap >= GOAL_DISAPPEAR_MIN:
                    candidates.append(candidate)
                    candidate = None

    # Strategy 2: Ball seen in goal zone via interpolated positions
    # Use a slightly wider zone for interpolated data
    INTERP_GOAL_ZONE_X = GOAL_ZONE_X  # already relaxed to 0.06
    for t in all_player_ts:
        if t in ball_ts_set:
            continue  # already handled above
        if t not in interpolated_ball:
            continue

        bx, by = interpolated_ball[t]
        in_goal_zone = abs(bx - target_goal_x) < INTERP_GOAL_ZONE_X
        y_in_goal = GOAL_Y_MIN <= by <= GOAL_Y_MAX

        if in_goal_zone and y_in_goal:
            possessor_pid = _find_possessor_near_t(t, poss_at_t, timestamps, window=3.0)
            if possessor_pid:
                # Check for ball disappearance after
                gap = ball_gaps.get(t, 0)
                if gap >= GOAL_DISAPPEAR_MIN:
                    cand = (t, possessor_pid, bx, by)
                    # Avoid duplicates (within 5s of existing candidate)
                    if not any(abs(c[0] - t) < 5.0 for c in candidates):
                        candidates.append(cand)

    # Strategy 3: Possession near goal + ball disappearance
    # If a player had possession within SHOT_ZONE_X of goal and ball disappears
    # for >1s, treat as goal candidate even without ball in exact goal zone
    _POSS_NEAR_GOAL_DISAPPEAR = 1.0  # ball must vanish 1s+ for this heuristic
    for entry in possession_tl:
        if not entry["possessor"] or entry["possessor"] not in assignments:
            continue
        if not entry.get("ball_known", True):
            continue

        bx = entry["bx"]
        dist_to_goal = abs(bx - target_goal_x)
        if dist_to_goal > SHOT_ZONE_X:
            continue

        t = entry["t"]
        pid = assignments[entry["possessor"]]["player_id"]

        # Check if ball disappears shortly after this frame
        gap_after = _find_gap_after(t, all_player_ts, ball_ts_set)
        if gap_after >= _POSS_NEAR_GOAL_DISAPPEAR:
            cand = (t, pid, bx, entry["by"])
            # Only add if not within 5s of existing candidate
            if not any(abs(c[0] - t) < 5.0 for c in candidates):
                candidates.append(cand)

    return candidates


def _find_gap_after(t, all_player_ts, ball_ts_set):
    """Find how long the ball disappears after timestamp t."""
    try:
        idx = all_player_ts.index(t)
    except ValueError:
        # Binary search fallback
        import bisect
        idx = bisect.bisect_left(all_player_ts, t)

    # Look for start of gap
    gap_start = None
    for j in range(idx + 1, len(all_player_ts)):
        t2 = all_player_ts[j]
        if t2 not in ball_ts_set:
            gap_start = t2
            break

    if gap_start is None:
        return 0

    # Find end of gap
    gap_end = gap_start
    start_idx = all_player_ts.index(gap_start)
    for j in range(start_idx, len(all_player_ts)):
        t2 = all_player_ts[j]
        if t2 in ball_ts_set:
            break
        gap_end = t2

    return gap_end - gap_start


def _find_possessor_near_t(t, poss_at_t, timestamps, window=3.0):
    """Find the possessor player_id at or just before timestamp t (within window seconds)."""
    if t in poss_at_t:
        return poss_at_t[t]
    for ts in reversed(timestamps):
        if ts > t:
            continue
        if t - ts > window:
            break
        if ts in poss_at_t:
            return poss_at_t[ts]
    return None


# ── Utilities ─────────────────────────────────────────────────────────────────

def _dist(x1, y1, x2, y2):
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5


def _add_event(events, last_events, match_id, player_id, event_type,
               timestamp, cooldown, x=None, y=None) -> bool:
    """Add event if cooldown has elapsed. Returns True if added."""
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
