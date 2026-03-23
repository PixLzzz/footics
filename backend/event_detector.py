"""
Automatic event detection from player tracking + ball tracking data.

Detects: passes, interceptions, shots, goals, assists, key passes, dribbles, saves.

Pipeline:
  1. Load & index tracking data
  2. Build possession timeline with hysteresis (anti-flicker)
  3. Compute ball velocity between frames
  4. Derive events from possession changes + ball trajectory
  5. Second pass: confirm goals via ball disappearance
  6. Post-processing: remove shot→goal duplicates, detect saves, assists, key passes
"""

import logging
from sqlalchemy.orm import Session
from models import (
    Match, Player, TrackingFrame, BallFrame,
    TrackAssignment, MatchEvent,
)

logger = logging.getLogger(__name__)

# ── Thresholds (all in normalized 0-1 coordinates) ───────────────────────────

# Possession
POSSESSION_DIST = 0.08            # Max ball-player distance to claim possession
POSSESSION_HYSTERESIS = 3         # Consecutive frames before switching possessor
POSSESSION_STICKY_BONUS = 0.015   # Current possessor gets this distance advantage
POSSESSION_MIN_DURATION = 0.3     # Min seconds of possession before generating events

# Passes
PASS_MIN_BALL_TRAVEL = 0.05       # Min ball travel (Euclidean) for a pass to count
EVENT_COOLDOWN = 2.0              # Min seconds between same event type for same player

# Shots & Goals
GOAL_ZONE_X = 0.03               # Ball within 3% from edge = goal zone
GOAL_Y_MIN = 0.20                # Goal vertical range (top bound, normalized)
GOAL_Y_MAX = 0.80                # Goal vertical range (bottom bound, normalized)
SHOT_ZONE_X = 0.25               # Ball within 25% from goal = shot zone
SHOT_MIN_SPEED = 0.04            # Min ball speed (norm units/frame interval) for a shot
GOAL_DISAPPEAR_MIN = 0.5         # Ball must vanish 0.5s+ to confirm goal

# Dribbles
DRIBBLE_MIN_PLAYER_TRAVEL = 0.06 # Min player movement during possession
DRIBBLE_MIN_DURATION = 1.5       # Min seconds holding ball while moving

# Assists & key passes
ASSIST_WINDOW = 15.0             # Max seconds between pass and goal for assist
KEY_PASS_WINDOW = 10.0           # Max seconds between pass and shot for key pass

# Saves
SAVE_WINDOW = 3.0                # Seconds after shot to check if it was saved


# ── Main entry point ─────────────────────────────────────────────────────────

def detect_events(match_id: int, db: Session, home_attacks_right: bool = True) -> list[dict]:
    """
    Analyze tracking + ball data to auto-detect match events.

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
    ball_ts_set = set(ball_by_ts.keys())
    timestamps = sorted(set(all_player_ts) & ball_ts_set)

    if not timestamps:
        logger.warning(f"Match {match_id}: no overlapping timestamps between players and ball")
        return []

    logger.info(f"Match {match_id}: analyzing {len(timestamps)} frames with ball+player data")

    home_goal_x = 1.0 if home_attacks_right else 0.0
    away_goal_x = 0.0 if home_attacks_right else 1.0

    def target_goal_x(team_id):
        return home_goal_x if team_id == match.team_home_id else away_goal_x

    # ── Phase 1: Possession timeline with hysteresis ──────────────────────
    possession_tl = _build_possession_timeline(
        timestamps, player_by_ts, ball_by_ts, assignments
    )

    # ── Phase 2: Ball velocity ────────────────────────────────────────────
    ball_velocity = _compute_ball_velocity(timestamps, ball_by_ts)

    # ── Phase 3: Ball gap map (for goal confirmation) ─────────────────────
    ball_gaps = _build_ball_gaps(all_player_ts, ball_ts_set)

    # ── Phase 4: Derive events from timeline ──────────────────────────────
    events = []
    last_events = {}  # (event_type, player_id) → last timestamp
    pass_chain = []   # [(t, player_id, team_id, ball_x, ball_y)] for assists/key_pass
    shots_pending = []  # [(t, player_id, team_id)] for save detection

    prev = None
    for entry in possession_tl:
        t = entry["t"]
        bx, by = entry["bx"], entry["by"]
        track_id = entry["possessor"]

        if track_id is None or track_id not in assignments:
            prev = entry
            continue

        info = assignments[track_id]
        pid = info["player_id"]
        tid = info["team_id"]
        tgx = target_goal_x(tid)
        dist_to_goal = abs(bx - tgx)

        # ── Possession change events ──────────────────────────────────
        if (prev and prev["possessor"] is not None
                and prev["possessor"] != track_id
                and prev["possessor"] in assignments):
            old = assignments[prev["possessor"]]
            poss_dur = entry.get("poss_duration", 0)

            if poss_dur >= POSSESSION_MIN_DURATION:
                ball_travel = _dist(bx, by, prev["bx"], prev["by"])

                if old["team_id"] == tid:
                    # Same team → PASS (only if ball actually moved)
                    if ball_travel >= PASS_MIN_BALL_TRAVEL:
                        added = _add_event(events, last_events, match_id,
                                           old["player_id"], "pass", t, EVENT_COOLDOWN,
                                           x=bx, y=by)
                        if added:
                            pass_chain.append((t, old["player_id"], old["team_id"], bx, by))
                else:
                    # Different team → INTERCEPTION
                    _add_event(events, last_events, match_id,
                               pid, "interception", t, EVENT_COOLDOWN,
                               x=bx, y=by)
                    pass_chain.clear()  # reset on turnover

        # ── Shot detection (ball moving fast toward goal) ─────────────
        if GOAL_ZONE_X <= dist_to_goal < SHOT_ZONE_X:
            speed = ball_velocity.get(t, 0)
            # Require minimum speed, or lower threshold if very close to goal
            speed_threshold = SHOT_MIN_SPEED
            if dist_to_goal < SHOT_ZONE_X * 0.4:
                speed_threshold *= 0.6  # easier to detect close shots

            if speed >= speed_threshold:
                added = _add_event(events, last_events, match_id,
                                   pid, "shot", t, cooldown=3.0,
                                   x=bx, y=by)
                if added:
                    shots_pending.append((t, pid, tid))
                    # Key pass: last pass from same team that led to this shot
                    for pt, pp_id, pt_id, px, py in reversed(pass_chain):
                        if pt_id == tid and pp_id != pid and t - pt < KEY_PASS_WINDOW:
                            _add_event(events, last_events, match_id,
                                       pp_id, "key_pass", pt, cooldown=5.0,
                                       x=px, y=py)
                            break

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
        all_player_ts, player_by_ts, ball_by_ts, ball_gaps,
        assignments, match, home_goal_x, away_goal_x,
        possession_tl, timestamps
    )

    for gt, gpid, gtid, gx, gy in goal_candidates:
        added = _add_event(events, last_events, match_id,
                           gpid, "goal", gt, cooldown=10.0,
                           x=gx, y=gy)
        if added:
            # Assist: last pass from same team before this goal
            for pt, pp_id, pt_id, px, py in reversed(pass_chain):
                if pt_id == gtid and pp_id != gpid and gt - pt < ASSIST_WINDOW:
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

    # Save detection: shots that didn't become goals
    for st, sp_id, stid in shots_pending:
        if any(abs(st - gt) < SAVE_WINDOW * 2 for gt in goal_times):
            continue  # this shot led to a goal, no save

        # Find nearest defending player to their own goal at shot time
        defending_tid = (match.team_away_id if stid == match.team_home_id
                         else match.team_home_id)
        defending_goal_x = (away_goal_x if stid == match.team_home_id
                            else home_goal_x)

        closest_t = min(timestamps, key=lambda x: abs(x - st)) if timestamps else None
        if closest_t and closest_t in player_by_ts:
            keeper_pid = _find_nearest_defender_to_goal(
                player_by_ts[closest_t], assignments, defending_tid, defending_goal_x
            )
            if keeper_pid:
                _add_event(events, last_events, match_id,
                           keeper_pid, "save", round(st + 0.5, 2), cooldown=3.0)

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

    Hysteresis: a new player must be closest to the ball for POSSESSION_HYSTERESIS
    consecutive frames before possession actually switches. The current possessor
    also gets a small distance bonus (POSSESSION_STICKY_BONUS).

    Returns list of dicts with:
      t, bx, by, possessor (track_id or None), poss_duration, player_travel
    """
    timeline = []
    current_possessor = None
    possession_since = None
    possession_start_pos = None  # (px, py) of player when they got possession
    challenger = None            # track_id of challenger
    challenger_streak = 0        # consecutive frames challenger is closest

    for t in timestamps:
        bx, by = ball_by_ts[t]
        players = player_by_ts.get(t, [])

        # Find nearest assigned player, with sticky bonus for current possessor
        nearest_track = None
        nearest_dist = float("inf")
        current_poss_dist = float("inf")

        for p in players:
            tid = p["track_id"]
            if tid not in assignments:
                continue
            d = _dist(p["cx"], p["foot_y"], bx, by)

            # Apply sticky bonus: current possessor is "closer" than they really are
            effective_d = d
            if tid == current_possessor:
                effective_d = max(0, d - POSSESSION_STICKY_BONUS)
                current_poss_dist = d

            if effective_d < nearest_dist:
                nearest_dist = effective_d
                nearest_track = tid

        # Nobody close enough
        if nearest_track is None or nearest_dist > POSSESSION_DIST:
            # Keep current possessor if they're still reasonably close
            if current_possessor is not None and current_poss_dist <= POSSESSION_DIST * 1.5:
                pass  # retain possession
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
            })
            continue

        # Hysteresis: don't switch immediately
        if nearest_track != current_possessor:
            if nearest_track == challenger:
                challenger_streak += 1
            else:
                challenger = nearest_track
                challenger_streak = 1

            if challenger_streak >= POSSESSION_HYSTERESIS or current_possessor is None:
                # Switch possession
                current_possessor = nearest_track
                possession_since = t
                # Find player position for travel tracking
                for p in players:
                    if p["track_id"] == nearest_track:
                        possession_start_pos = (p["cx"], p["cy"])
                        break
                challenger = None
                challenger_streak = 0
        else:
            # Same possessor, reset challenger
            challenger = None
            challenger_streak = 0

        # Compute possession duration and player travel
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
        x1, y1 = ball_by_ts[t_prev]
        x2, y2 = ball_by_ts[t_curr]
        dist = _dist(x1, y1, x2, y2)
        velocity[t_curr] = dist / dt  # normalized units per second
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


# ── Goal detection (ball disappearance confirmation) ──────────────────────────

def _detect_goal_candidates(all_player_ts, player_by_ts, ball_by_ts, ball_gaps,
                            assignments, match, home_goal_x, away_goal_x,
                            possession_tl, timestamps):
    """
    Detect goals by finding frames where:
    1. Ball enters goal zone (correct side for the possessing team)
    2. Ball Y is within goal post range (not on sideline)
    3. Ball then disappears for GOAL_DISAPPEAR_MIN+ seconds

    Uses the possession timeline from the main pass to know who had the ball.
    """
    # Build a quick lookup: timestamp → possessor info from the timeline
    poss_at_t = {}
    for entry in possession_tl:
        if entry["possessor"] and entry["possessor"] in assignments:
            info = assignments[entry["possessor"]]
            poss_at_t[entry["t"]] = (info["player_id"], info["team_id"])

    ball_ts_set = set(ball_by_ts.keys())
    candidates = []
    candidate = None  # (t, player_id, team_id, ball_x, ball_y)

    for t in all_player_ts:
        if t in ball_ts_set:
            bx, by = ball_by_ts[t]

            in_home_goal = abs(bx - home_goal_x) < GOAL_ZONE_X
            in_away_goal = abs(bx - away_goal_x) < GOAL_ZONE_X

            # Ball Y must be in plausible goal range (not on sideline)
            y_in_goal = GOAL_Y_MIN <= by <= GOAL_Y_MAX

            if (in_home_goal or in_away_goal) and y_in_goal:
                # Find last known possessor near this timestamp
                possessor_info = _find_possessor_near_t(t, poss_at_t, timestamps, window=3.0)
                if possessor_info:
                    pid, tid = possessor_info
                    # Verify possessor's team attacks this goal
                    if tid == match.team_home_id:
                        attacks_x = home_goal_x
                    else:
                        attacks_x = away_goal_x
                    correct_goal = ((in_home_goal and attacks_x == home_goal_x) or
                                    (in_away_goal and attacks_x == away_goal_x))
                    if correct_goal:
                        candidate = (t, pid, tid, bx, by)
        else:
            # Ball not detected — confirm candidate if gap is long enough
            if candidate is not None:
                gap = ball_gaps.get(t, 0)
                if gap >= GOAL_DISAPPEAR_MIN:
                    candidates.append(candidate)
                    candidate = None

    return candidates


def _find_possessor_near_t(t, poss_at_t, timestamps, window=3.0):
    """Find the possessor at or just before timestamp t (within window seconds)."""
    if t in poss_at_t:
        return poss_at_t[t]
    # Search backwards through nearby timestamps
    for ts in reversed(timestamps):
        if ts > t:
            continue
        if t - ts > window:
            break
        if ts in poss_at_t:
            return poss_at_t[ts]
    return None


# ── Save detection helper ─────────────────────────────────────────────────────

def _find_nearest_defender_to_goal(players, assignments, defending_team_id, goal_x):
    """Find the defending team's player closest to their own goal. Returns player_id or None."""
    best_pid = None
    best_dist = float("inf")
    for p in players:
        tid = p["track_id"]
        if tid not in assignments:
            continue
        info = assignments[tid]
        if info["team_id"] != defending_team_id:
            continue
        d = abs(p["cx"] - goal_x)
        if d < best_dist:
            best_dist = d
            best_pid = info["player_id"]
    # Only count as save if someone is actually near the goal (within 15% of pitch)
    if best_dist > 0.15:
        return None
    return best_pid


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
