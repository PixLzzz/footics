"""Team-aware auto-assignment with 1-to-1 matching.

Strategy:
  1. Extract jersey colour from manually-assigned player tracks → "team colour profile"
  2. Filter unassigned tracks: keep only those whose jersey colour matches the team
  3. Use Hungarian algorithm for optimal 1-to-1 player→track matching
  4. Handle track fragmentation: allow same player on multiple non-overlapping tracks
"""

import cv2
import numpy as np
import logging
from collections import defaultdict
from sqlalchemy.orm import Session
from sqlalchemy import func

from models import TrackingFrame, TrackAssignment, Match, Player
from video_processor import extract_frame

logger = logging.getLogger(__name__)

# Progress tracking: {match_id: {"current": int, "total": int, "percent": float, "phase": str}}
assignment_progress = {}

# --------------------------------------------------------------------------- #
# Feature extraction
# --------------------------------------------------------------------------- #

def _safe_crop(frame: np.ndarray, bbox_x, bbox_y, bbox_w, bbox_h, region="full"):
    """Crop a region from a frame using normalised bbox coordinates."""
    h_img, w_img = frame.shape[:2]
    px_x = int(bbox_x * w_img)
    px_y = int(bbox_y * h_img)
    px_w = int(bbox_w * w_img)
    px_h = int(bbox_h * h_img)

    if region == "torso":
        px_h = int(px_h * 0.55)
    elif region == "upper":
        px_h = int(px_h * 0.35)
    elif region == "lower":
        offset = int(px_h * 0.55)
        px_y += offset
        px_h -= offset

    x1 = max(0, px_x)
    y1 = max(0, px_y)
    x2 = min(w_img, px_x + px_w)
    y2 = min(h_img, px_y + px_h)

    crop = frame[y1:y2, x1:x2]
    if crop.size == 0 or crop.shape[0] < 4 or crop.shape[1] < 4:
        return None
    return crop


def _resize_crop(crop: np.ndarray, max_h=64) -> np.ndarray:
    if crop.shape[0] <= max_h:
        return crop
    scale = max_h / crop.shape[0]
    return cv2.resize(crop, (max(1, int(crop.shape[1] * scale)), max_h), interpolation=cv2.INTER_AREA)


def extract_color_histogram(crop: np.ndarray, bins=16) -> np.ndarray | None:
    """HSV 2D histogram (H x S), L1-normalised."""
    if crop is None:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [bins, bins], [0, 180, 0, 256])
    cv2.normalize(hist, hist, norm_type=cv2.NORM_L1)
    return hist.flatten()


def extract_spatial_color(frame: np.ndarray, bbox_x, bbox_y, bbox_w, bbox_h) -> np.ndarray | None:
    """Mean colour of upper third vs lower third in LAB space (6D vector)."""
    upper = _safe_crop(frame, bbox_x, bbox_y, bbox_w, bbox_h, "upper")
    lower = _safe_crop(frame, bbox_x, bbox_y, bbox_w, bbox_h, "lower")
    if upper is None or lower is None:
        return None
    lab_upper = cv2.cvtColor(upper, cv2.COLOR_BGR2LAB)
    lab_lower = cv2.cvtColor(lower, cv2.COLOR_BGR2LAB)
    mean_upper = np.mean(lab_upper.reshape(-1, 3), axis=0).astype(np.float32) / 255.0
    mean_lower = np.mean(lab_lower.reshape(-1, 3), axis=0).astype(np.float32) / 255.0
    return np.concatenate([mean_upper, mean_lower])


def extract_features(
    frame: np.ndarray,
    bbox_x: float, bbox_y: float, bbox_w: float, bbox_h: float,
) -> dict | None:
    """Extract feature vectors for a single detection."""
    torso = _safe_crop(frame, bbox_x, bbox_y, bbox_w, bbox_h, "torso")
    if torso is None:
        return None
    torso = _resize_crop(torso)

    features = {}

    color_hist = extract_color_histogram(torso)
    if color_hist is not None:
        features["color_hist"] = color_hist

    spatial = extract_spatial_color(frame, bbox_x, bbox_y, bbox_w, bbox_h)
    if spatial is not None:
        features["spatial_color"] = spatial

    return features if features else None


# --------------------------------------------------------------------------- #
# Feature comparison
# --------------------------------------------------------------------------- #

FEATURE_WEIGHTS = {
    "color_hist": 0.55,
    "spatial_color": 0.45,
}


def compare_features(feat1: dict, feat2: dict) -> float:
    """Compare two feature dicts — returns distance in [0, 1]. Lower = more similar."""
    total_weight = 0.0
    total_dist = 0.0

    for key, weight in FEATURE_WEIGHTS.items():
        if key not in feat1 or key not in feat2:
            continue

        v1, v2 = feat1[key], feat2[key]

        if key == "color_hist":
            dist = cv2.compareHist(
                v1.astype(np.float32), v2.astype(np.float32),
                cv2.HISTCMP_BHATTACHARYYA,
            )
        else:
            norm1 = np.linalg.norm(v1)
            norm2 = np.linalg.norm(v2)
            if norm1 < 1e-8 or norm2 < 1e-8:
                dist = 1.0
            else:
                cos_sim = np.dot(v1, v2) / (norm1 * norm2)
                dist = 1.0 - float(np.clip(cos_sim, -1, 1))

        total_dist += weight * dist
        total_weight += weight

    if total_weight < 0.1:
        return 1.0

    return total_dist / total_weight


def compare_color_only(feat1: dict, feat2: dict) -> float:
    """Compare only the color histogram — used for team filtering."""
    if "color_hist" not in feat1 or "color_hist" not in feat2:
        return 1.0
    return cv2.compareHist(
        feat1["color_hist"].astype(np.float32),
        feat2["color_hist"].astype(np.float32),
        cv2.HISTCMP_BHATTACHARYYA,
    )


# --------------------------------------------------------------------------- #
# Streaming feature extraction
# --------------------------------------------------------------------------- #

def _stream_extract_features(
    video_path: str,
    samples: list[tuple[float, int, float, float, float, float]],
    progress_cb=None,
) -> list[dict | None]:
    """Extract features in a single sequential video pass. One frame in RAM at a time."""
    features: list[dict | None] = [None] * len(samples)
    if not samples:
        return features

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return features

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        cap.release()
        return features

    # Group sample indices by frame number
    samples_by_frame: dict[int, list[int]] = defaultdict(list)
    for idx, (ts, _tid, _bx, _by, _bw, _bh) in enumerate(samples):
        fn = int(ts * fps)
        samples_by_frame[fn].append(idx)

    target_frames = sorted(samples_by_frame.keys())
    total = len(target_frames)
    current_frame = 0

    for target_idx, target_fn in enumerate(target_frames):
        gap = target_fn - current_frame
        if gap > 500:
            cap.set(cv2.CAP_PROP_POS_FRAMES, target_fn)
            current_frame = target_fn
        else:
            while current_frame < target_fn:
                if not cap.grab():
                    break
                current_frame += 1

        ret, frame = cap.read()
        current_frame += 1

        if not ret:
            continue

        for sample_idx in samples_by_frame[target_fn]:
            _ts, _tid, bx, by, bw, bh = samples[sample_idx]
            features[sample_idx] = extract_features(frame, bx, by, bw, bh)

        del frame

        if progress_cb and target_idx % 3 == 0:
            progress_cb(target_idx, total)

    cap.release()
    if progress_cb:
        progress_cb(total, total)
    return features


# --------------------------------------------------------------------------- #
# Main auto-assignment logic
# --------------------------------------------------------------------------- #

def auto_assign_tracks(
    match_id: int,
    db: Session,
    threshold: float = 0.38,
) -> dict:
    """Team-aware auto-assignment with 1-to-1 matching.

    Strategy:
      1. Build reference features from manually assigned player tracks
      2. Compute team colour profile → filter out opposing team tracks
      3. Hungarian algorithm for optimal 1-to-1 player↔track matching
      4. For remaining team-coloured tracks: assign to most similar player
         only if they don't temporally overlap with existing assignments
    """
    match = db.query(Match).filter(Match.id == match_id).first()
    if not match:
        return {"error": "Match not found", "assigned_count": 0, "details": []}

    video_path = match.video_path

    # ── Load existing assignments ───────────────────────────────────────
    existing = db.query(TrackAssignment).filter_by(match_id=match_id).all()
    assigned_track_ids = {a.track_id for a in existing}
    player_assignments = {a.track_id: a for a in existing if a.player_id and not a.is_referee}
    referee_assignments = {a.track_id: a for a in existing if a.is_referee}

    if not player_assignments:
        return {
            "assigned_count": 0,
            "details": [],
            "message": "Assignez manuellement au moins un joueur de votre équipe d'abord.",
        }

    # ── Gather track-level metadata ─────────────────────────────────────
    track_stats = (
        db.query(
            TrackingFrame.track_id,
            func.count(TrackingFrame.id).label("cnt"),
            func.min(TrackingFrame.timestamp_seconds).label("t_min"),
            func.max(TrackingFrame.timestamp_seconds).label("t_max"),
        )
        .filter(TrackingFrame.match_id == match_id)
        .group_by(TrackingFrame.track_id)
        .all()
    )
    track_frame_count = {t.track_id: t.cnt for t in track_stats}
    track_time_range = {t.track_id: (t.t_min, t.t_max) for t in track_stats}

    MIN_TRACK_FRAMES = 30
    unassigned_track_ids = [
        tid for tid, cnt in track_frame_count.items()
        if tid not in assigned_track_ids and cnt >= MIN_TRACK_FRAMES
    ]

    if not unassigned_track_ids:
        return {"assigned_count": 0, "details": [], "message": "Aucun track non-assigné avec assez de frames."}

    ref_track_ids = set(player_assignments.keys()) | set(referee_assignments.keys())
    total_tracks = len(ref_track_ids) + len(unassigned_track_ids)

    assignment_progress[match_id] = {
        "current": 0, "total": total_tracks, "percent": 0,
        "phase": "Collecte des échantillons...",
    }

    # ── Collect samples ─────────────────────────────────────────────────
    N_SAMPLES = 3
    all_samples: list[tuple[float, int, float, float, float, float]] = []

    def _pick_samples(track_id: int, n: int):
        cnt = track_frame_count.get(track_id, 0)
        if cnt == 0:
            return []
        step = max(1, cnt // (n + 1))
        picked = []
        for i in range(n):
            offset = step * (i + 1)
            if offset >= cnt:
                break
            row = (
                db.query(TrackingFrame)
                .filter_by(match_id=match_id, track_id=track_id)
                .order_by(TrackingFrame.timestamp_seconds)
                .offset(offset).limit(1).first()
            )
            if row:
                picked.append(row)
        if not picked:
            row = (
                db.query(TrackingFrame)
                .filter_by(match_id=match_id, track_id=track_id)
                .order_by(TrackingFrame.timestamp_seconds)
                .offset(cnt // 2).limit(1).first()
            )
            if row:
                picked.append(row)
        return picked

    ref_samples_map: dict[int, list] = {}
    una_samples_map: dict[int, list] = {}

    for tid in ref_track_ids:
        picked = _pick_samples(tid, N_SAMPLES)
        indices = []
        for f in picked:
            idx = len(all_samples)
            all_samples.append((f.timestamp_seconds, f.track_id, f.bbox_x, f.bbox_y, f.bbox_w, f.bbox_h))
            indices.append(idx)
        ref_samples_map[tid] = indices

    for tid in unassigned_track_ids:
        picked = _pick_samples(tid, N_SAMPLES)
        indices = []
        for f in picked:
            idx = len(all_samples)
            all_samples.append((f.timestamp_seconds, f.track_id, f.bbox_x, f.bbox_y, f.bbox_w, f.bbox_h))
            indices.append(idx)
        una_samples_map[tid] = indices

    # ── Stream through video ────────────────────────────────────────────
    n_unique_frames = len({int(s[0] * 30) for s in all_samples})
    logger.info(f"Match {match_id}: {len(all_samples)} samples across ~{n_unique_frames} frames")

    def on_progress(current, total):
        pct = round(current / total * 50, 1) if total > 0 else 0
        assignment_progress[match_id].update({
            "current": current, "total": total,
            "percent": pct,
            "phase": f"Analyse vidéo ({current}/{total} frames)...",
        })

    all_features = _stream_extract_features(video_path, all_samples, progress_cb=on_progress)

    # ── Build averaged feature vectors per track ────────────────────────
    assignment_progress[match_id].update({"percent": 55, "phase": "Construction des profils..."})

    def _average_features(indices):
        valid = [all_features[i] for i in indices if all_features[i] is not None]
        if not valid:
            return None
        averaged = {}
        for key in valid[0].keys():
            arrays = [f[key] for f in valid if key in f]
            if arrays:
                averaged[key] = np.mean(arrays, axis=0).astype(np.float32)
        return averaged

    ref_features: dict[int, dict] = {}
    for tid, indices in ref_samples_map.items():
        feat = _average_features(indices)
        if feat:
            ref_features[tid] = feat

    una_features: dict[int, dict] = {}
    for tid, indices in una_samples_map.items():
        feat = _average_features(indices)
        if feat:
            una_features[tid] = feat

    # ── PHASE 1: Team colour filter ─────────────────────────────────────
    assignment_progress[match_id].update({"percent": 60, "phase": "Filtrage par couleur d'équipe..."})

    # Compute average team colour from assigned player tracks (exclude referees)
    team_color_refs = []
    for tid, assignment in player_assignments.items():
        if tid in ref_features:
            team_color_refs.append(ref_features[tid])

    if not team_color_refs:
        return {"assigned_count": 0, "details": [], "message": "Pas assez de données de référence."}

    # Average team colour profile
    team_color_profile = {}
    for key in team_color_refs[0].keys():
        arrays = [f[key] for f in team_color_refs if key in f]
        if arrays:
            team_color_profile[key] = np.mean(arrays, axis=0).astype(np.float32)

    # Also compute referee colour profile if we have referee refs
    referee_color_profile = None
    if referee_assignments:
        ref_refs = [ref_features[tid] for tid in referee_assignments if tid in ref_features]
        if ref_refs:
            referee_color_profile = {}
            for key in ref_refs[0].keys():
                arrays = [f[key] for f in ref_refs if key in f]
                if arrays:
                    referee_color_profile[key] = np.mean(arrays, axis=0).astype(np.float32)

    # ── BRIGHTNESS-BASED TEAM FILTER (most reliable) ──────────────────
    # Measure torso brightness for each assigned player → team brightness
    # Then hard-reject any track whose brightness differs by > tolerance
    BRIGHTNESS_TOLERANCE = 45

    # Get team brightness from assigned player tracks
    team_brightness_samples = []
    cap_b = cv2.VideoCapture(video_path)
    fps_b = cap_b.get(cv2.CAP_PROP_FPS) if cap_b.isOpened() else 0

    if fps_b > 0:
        for tid in player_assignments:
            for idx in ref_samples_map.get(tid, []):
                ts, _tid, bx, by, bw, bh = all_samples[idx]
                frame_num = int(ts * fps_b)
                cap_b.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
                ret, frame = cap_b.read()
                if ret:
                    crop = _safe_crop(frame, bx, by, bw, bh, "torso")
                    if crop is not None:
                        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
                        team_brightness_samples.append(float(np.mean(hsv[:, :, 2])))
                    del frame

    cap_b.release()

    team_brightness = float(np.median(team_brightness_samples)) if team_brightness_samples else None
    logger.info(f"Team brightness: {team_brightness:.0f}" if team_brightness else "No brightness data")

    # Also use pre-computed team labels if available
    try:
        from team_classifier import get_team_labels
        team_labels = get_team_labels(match_id)
        if team_labels:
            from collections import Counter
            assigned_labels = [team_labels.get(tid) for tid in player_assignments if tid in team_labels]
            assigned_labels = [l for l in assigned_labels if l is not None]
            user_team_label = Counter(assigned_labels).most_common(1)[0][0] if assigned_labels else None
            team_label_set = {tid for tid, l in team_labels.items() if l == user_team_label} if user_team_label is not None else None
        else:
            team_label_set = None
    except Exception:
        team_label_set = None

    # Now filter tracks using BOTH brightness and team labels
    TEAM_COLOR_THRESHOLD = 0.40
    team_tracks = []
    other_tracks = []

    # Get brightness for unassigned tracks
    una_brightness = {}
    cap_b2 = cv2.VideoCapture(video_path)
    fps_b2 = cap_b2.get(cv2.CAP_PROP_FPS) if cap_b2.isOpened() else 0

    if fps_b2 > 0:
        for tid in unassigned_track_ids:
            for idx in una_samples_map.get(tid, []):
                ts, _tid, bx, by, bw, bh = all_samples[idx]
                frame_num = int(ts * fps_b2)
                cap_b2.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
                ret, frame = cap_b2.read()
                if ret:
                    crop = _safe_crop(frame, bx, by, bw, bh, "torso")
                    if crop is not None:
                        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
                        una_brightness[tid] = float(np.mean(hsv[:, :, 2]))
                    del frame
                break  # one sample is enough for brightness

    cap_b2.release()

    for tid, feat in una_features.items():
        # HARD FILTER 1: brightness check (most reliable)
        if team_brightness is not None and tid in una_brightness:
            if abs(una_brightness[tid] - team_brightness) > BRIGHTNESS_TOLERANCE:
                other_tracks.append(tid)
                continue

        # HARD FILTER 2: team label check (from pre-computed classification)
        if team_label_set is not None and tid in team_labels:
            if tid not in team_label_set:
                other_tracks.append(tid)
                continue

        # SOFT FILTER 3: color histogram comparison (fallback)
        team_dist = compare_color_only(feat, team_color_profile)

        if referee_color_profile:
            ref_dist = compare_color_only(feat, referee_color_profile)
            if ref_dist < team_dist and ref_dist < TEAM_COLOR_THRESHOLD:
                assignment = TrackAssignment(
                    match_id=match_id, track_id=tid,
                    player_id=None, is_referee=True,
                )
                db.add(assignment)
                logger.info("Auto-assigned track %d -> referee (color_dist=%.4f)", tid, ref_dist)
                continue

        if team_dist < TEAM_COLOR_THRESHOLD:
            team_tracks.append(tid)
        else:
            other_tracks.append(tid)

    logger.info(
        f"Team filter: {len(team_tracks)} team tracks, {len(other_tracks)} filtered out, "
        f"threshold={TEAM_COLOR_THRESHOLD}"
    )

    if not team_tracks:
        db.commit()
        return {
            "assigned_count": 0,
            "details": [],
            "message": f"Aucun track non-assigné ne correspond à la couleur de l'équipe ({len(other_tracks)} filtrés).",
        }

    # ── PHASE 2: Hungarian algorithm — 1-to-1 matching ──────────────────
    assignment_progress[match_id].update({"percent": 70, "phase": "Assignation optimale (Hongrois)..."})

    # Get players that still need a track
    # A player "needs a track" if they have no assignment, OR if all their existing
    # tracks ended (allowing re-assignment for fragmented tracks)
    assigned_player_ids = {a.player_id for a in player_assignments.values()}

    # Get ALL team players (not just those already assigned)
    team_id = match.team_id
    all_team_players = []
    if team_id:
        all_team_players = db.query(Player).filter_by(team_id=team_id).all()
    else:
        # Fallback: use players from existing assignments
        pids = {a.player_id for a in player_assignments.values()}
        all_team_players = db.query(Player).filter(Player.id.in_(pids)).all()

    # Build player→reference features map
    player_refs: dict[int, dict] = {}
    for tid, assignment in player_assignments.items():
        if tid in ref_features:
            player_refs[assignment.player_id] = ref_features[tid]

    # Players who don't have any track yet
    unassigned_players = [p for p in all_team_players if p.id not in assigned_player_ids]

    details = []
    assigned_count = 0

    if unassigned_players and team_tracks:
        # Build cost matrix [players x tracks]
        n_players = len(unassigned_players)
        n_tracks = len(team_tracks)
        cost_matrix = np.full((n_players, n_tracks), 1.0, dtype=np.float64)

        for i, player in enumerate(unassigned_players):
            if player.id not in player_refs:
                continue
            ref_feat = player_refs[player.id]
            for j, tid in enumerate(team_tracks):
                if tid in una_features:
                    cost_matrix[i, j] = compare_features(ref_feat, una_features[tid])

        # Hungarian algorithm
        try:
            from scipy.optimize import linear_sum_assignment
            row_indices, col_indices = linear_sum_assignment(cost_matrix)
        except ImportError:
            # Fallback: greedy matching
            logger.warning("scipy not available, using greedy matching")
            row_indices, col_indices = _greedy_assignment(cost_matrix)

        matched_tracks = set()
        for row_idx, col_idx in zip(row_indices, col_indices):
            dist = cost_matrix[row_idx, col_idx]
            if dist >= threshold:
                continue

            player = unassigned_players[row_idx]
            tid = team_tracks[col_idx]

            assignment = TrackAssignment(
                match_id=match_id, track_id=tid,
                player_id=player.id, is_referee=False,
            )
            db.add(assignment)
            assigned_count += 1
            matched_tracks.add(tid)

            details.append({
                "track_id": tid,
                "player_id": player.id,
                "player_name": player.name,
                "is_referee": False,
                "distance": round(dist, 4),
            })
            logger.info("Hungarian: track %d -> %s (dist=%.4f)", tid, player.name, dist)

        # Remove matched tracks from pool
        team_tracks = [tid for tid in team_tracks if tid not in matched_tracks]

    # ── PHASE 3: Extra fragments — same player, non-overlapping time ────
    assignment_progress[match_id].update({"percent": 85, "phase": "Assignation des fragments restants..."})

    if team_tracks:
        # For remaining team-coloured tracks, find the best matching ALREADY-assigned player
        # but only if the track doesn't overlap temporally
        all_current_assignments = db.query(TrackAssignment).filter_by(match_id=match_id).all()

        # Build per-player time ranges
        player_time_ranges: dict[int, list[tuple[float, float]]] = defaultdict(list)
        for a in all_current_assignments:
            if a.player_id and a.track_id in track_time_range:
                t_min, t_max = track_time_range[a.track_id]
                player_time_ranges[a.player_id].append((t_min, t_max))

        def overlaps(tid, player_id):
            if tid not in track_time_range:
                return True
            t_min, t_max = track_time_range[tid]
            for (r_min, r_max) in player_time_ranges.get(player_id, []):
                # Allow small overlap (2s tolerance for track transitions)
                if t_min < r_max - 2.0 and t_max > r_min + 2.0:
                    return True
            return False

        for tid in team_tracks:
            if tid not in una_features:
                continue

            feat = una_features[tid]
            best_dist = float("inf")
            best_pid = None

            for pid, ref_feat in player_refs.items():
                dist = compare_features(feat, ref_feat)
                if dist < best_dist and dist < threshold:
                    if not overlaps(tid, pid):
                        best_dist = dist
                        best_pid = pid

            if best_pid is not None:
                assignment = TrackAssignment(
                    match_id=match_id, track_id=tid,
                    player_id=best_pid, is_referee=False,
                )
                db.add(assignment)
                assigned_count += 1

                # Update time ranges
                if tid in track_time_range:
                    player_time_ranges[best_pid].append(track_time_range[tid])

                player = db.query(Player).filter_by(id=best_pid).first()
                details.append({
                    "track_id": tid,
                    "player_id": best_pid,
                    "player_name": player.name if player else "?",
                    "is_referee": False,
                    "distance": round(best_dist, 4),
                    "fragment": True,
                })
                logger.info("Fragment: track %d -> player %d (dist=%.4f, non-overlapping)", tid, best_pid, best_dist)

    db.commit()

    assignment_progress[match_id].update({
        "percent": 100, "phase": "Terminé",
        "current": assigned_count, "total": assigned_count,
    })

    return {
        "assigned_count": assigned_count,
        "filtered_out": len(other_tracks),
        "details": details,
    }


def _greedy_assignment(cost_matrix: np.ndarray):
    """Simple greedy matching fallback if scipy is not available."""
    n_rows, n_cols = cost_matrix.shape
    row_indices = []
    col_indices = []
    used_rows = set()
    used_cols = set()

    # Flatten and sort by cost
    costs = []
    for i in range(n_rows):
        for j in range(n_cols):
            costs.append((cost_matrix[i, j], i, j))
    costs.sort()

    for cost, i, j in costs:
        if i not in used_rows and j not in used_cols:
            row_indices.append(i)
            col_indices.append(j)
            used_rows.add(i)
            used_cols.add(j)

    return row_indices, col_indices
