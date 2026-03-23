"""Multi-feature appearance-based track auto-assignment.

Uses a combination of visual cues to re-identify the same person across
different YOLO track IDs:
  1. HSV colour histogram (torso) — jersey colour
  2. Body proportions — height/width ratio, torso/leg split
  3. Colour spatial distribution — top vs bottom colour difference
  4. Edge/texture density — captures patterns like stripes, logos
  5. Dominant colour clustering — robust against lighting changes
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
    """Crop a region from a frame using normalised bbox coordinates.

    region: "full", "torso" (top 55%), "upper" (top 35%), "lower" (bottom 45%)
    """
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


def extract_color_histogram(crop: np.ndarray, bins=16) -> np.ndarray | None:
    """HSV 2D histogram (H x S), L1-normalised."""
    if crop is None:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [bins, bins], [0, 180, 0, 256])
    cv2.normalize(hist, hist, norm_type=cv2.NORM_L1)
    return hist.flatten()


def extract_body_proportions(frame: np.ndarray, bbox_x, bbox_y, bbox_w, bbox_h) -> np.ndarray | None:
    """Aspect ratio + torso/full ratio as a 2D feature vector."""
    h_img, w_img = frame.shape[:2]
    px_w = bbox_w * w_img
    px_h = bbox_h * h_img
    if px_h < 10 or px_w < 5:
        return None
    aspect = px_w / px_h  # width/height ratio
    # Normalised bbox area relative to frame (correlates with distance from camera)
    rel_area = (px_w * px_h) / (w_img * h_img)
    return np.array([aspect, rel_area], dtype=np.float32)


def extract_spatial_color(frame: np.ndarray, bbox_x, bbox_y, bbox_w, bbox_h) -> np.ndarray | None:
    """Mean colour of upper third vs lower third in LAB space (6D vector).

    Captures jersey vs shorts colour difference — unique per player outfit.
    """
    upper = _safe_crop(frame, bbox_x, bbox_y, bbox_w, bbox_h, "upper")
    lower = _safe_crop(frame, bbox_x, bbox_y, bbox_w, bbox_h, "lower")
    if upper is None or lower is None:
        return None
    lab_upper = cv2.cvtColor(upper, cv2.COLOR_BGR2LAB)
    lab_lower = cv2.cvtColor(lower, cv2.COLOR_BGR2LAB)
    mean_upper = np.mean(lab_upper.reshape(-1, 3), axis=0).astype(np.float32) / 255.0
    mean_lower = np.mean(lab_lower.reshape(-1, 3), axis=0).astype(np.float32) / 255.0
    return np.concatenate([mean_upper, mean_lower])


def extract_texture_density(crop: np.ndarray) -> np.ndarray | None:
    """Edge density (Canny) as a scalar feature — captures stripes, logos, patterns."""
    if crop is None:
        return None
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    density = np.mean(edges > 0).astype(np.float32)
    return np.array([density], dtype=np.float32)


def _resize_crop(crop: np.ndarray, max_h=64) -> np.ndarray:
    """Resize crop to max_h pixels tall, preserving aspect ratio."""
    if crop.shape[0] <= max_h:
        return crop
    scale = max_h / crop.shape[0]
    return cv2.resize(crop, (max(1, int(crop.shape[1] * scale)), max_h), interpolation=cv2.INTER_AREA)


# --------------------------------------------------------------------------- #
# Combined feature vector
# --------------------------------------------------------------------------- #

def extract_features(
    frame: np.ndarray,
    bbox_x: float, bbox_y: float, bbox_w: float, bbox_h: float,
) -> dict | None:
    """Extract feature vectors for a single detection (fast version).

    Uses 3 cheap features: color histogram, spatial color, body proportions.
    All crops are resized to 64px max for speed.
    """
    torso = _safe_crop(frame, bbox_x, bbox_y, bbox_w, bbox_h, "torso")
    if torso is None:
        return None
    torso = _resize_crop(torso)

    features = {}

    color_hist = extract_color_histogram(torso)
    if color_hist is not None:
        features["color_hist"] = color_hist

    proportions = extract_body_proportions(frame, bbox_x, bbox_y, bbox_w, bbox_h)
    if proportions is not None:
        features["proportions"] = proportions

    spatial = extract_spatial_color(frame, bbox_x, bbox_y, bbox_w, bbox_h)
    if spatial is not None:
        features["spatial_color"] = spatial

    texture = extract_texture_density(torso)
    if texture is not None:
        features["texture"] = texture

    return features if features else None


# --------------------------------------------------------------------------- #
# Feature comparison
# --------------------------------------------------------------------------- #

# Weights for each feature type in the final similarity score
FEATURE_WEIGHTS = {
    "color_hist": 0.35,
    "spatial_color": 0.30,
    "proportions": 0.15,
    "texture": 0.20,
}


def compare_features(feat1: dict, feat2: dict) -> float:
    """Compare two feature dicts and return a distance (lower = more similar).

    Returns a weighted combination of per-feature distances, normalised to [0, 1].
    """
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
        elif key == "proportions":
            # Relative difference for proportions
            diff = np.abs(v1 - v2) / (np.abs(v1) + np.abs(v2) + 1e-6)
            dist = float(np.mean(diff))
        else:
            # Cosine distance for other vector features
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
        return 1.0  # not enough features to compare

    return total_dist / total_weight


# --------------------------------------------------------------------------- #
# Batch frame extraction
# --------------------------------------------------------------------------- #

def _stream_extract_features(
    video_path: str,
    samples: list[tuple[float, int, float, float, float, float]],
    progress_cb=None,
) -> list[dict | None]:
    """Extract features from video in a single streaming pass.

    Reads through the video sequentially. For each needed timestamp,
    decodes the frame, extracts features for all samples at that timestamp,
    then immediately discards the frame. Only keeps tiny feature vectors in RAM.

    Parameters
    ----------
    samples : list of (timestamp, track_id, bbox_x, bbox_y, bbox_w, bbox_h)
    progress_cb : callable(current, total) or None

    Returns
    -------
    list of feature dicts (same order as samples), None for failed extractions.
    """
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
    from collections import defaultdict
    samples_by_frame: dict[int, list[int]] = defaultdict(list)
    for idx, (ts, _tid, _bx, _by, _bw, _bh) in enumerate(samples):
        fn = int(ts * fps)
        samples_by_frame[fn] = samples_by_frame.get(fn, [])
        samples_by_frame[fn].append(idx)

    target_frames = sorted(samples_by_frame.keys())
    total = len(target_frames)
    current_frame = 0

    for target_idx, target_fn in enumerate(target_frames):
        # Navigate to target frame
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

        # Extract features for ALL samples at this frame, then discard frame
        for sample_idx in samples_by_frame[target_fn]:
            _ts, _tid, bx, by, bw, bh = samples[sample_idx]
            features[sample_idx] = extract_features(frame, bx, by, bw, bh)

        # Frame goes out of scope here — memory freed immediately
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
    """Auto-assign unassigned tracks using multi-feature appearance matching.

    1. Build reference feature vectors for every already-assigned player/referee
       (averaged from 5 samples spread across the track).
    2. For every unassigned track (>= 10 frames), sample 5 frames, extract features,
       and compare against all references.
    3. If the best match distance < threshold, create a TrackAssignment.

    Progress is exposed via assignment_progress[match_id].
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

    if not player_assignments and not referee_assignments:
        return {
            "assigned_count": 0,
            "details": [],
            "message": "Assignez manuellement au moins un joueur d'abord.",
        }

    # ── Gather track-level metadata ─────────────────────────────────────
    track_stats = (
        db.query(
            TrackingFrame.track_id,
            func.count(TrackingFrame.id).label("cnt"),
        )
        .filter(TrackingFrame.match_id == match_id)
        .group_by(TrackingFrame.track_id)
        .all()
    )
    track_frame_count = {t.track_id: t.cnt for t in track_stats}

    ref_track_ids = set(player_assignments.keys()) | set(referee_assignments.keys())
    # Only consider tracks with >= 50 frames (smaller = transient noise)
    MIN_TRACK_FRAMES = 50
    unassigned_track_ids = [
        tid for tid, cnt in track_frame_count.items()
        if tid not in assigned_track_ids and cnt >= MIN_TRACK_FRAMES
    ]

    if not unassigned_track_ids:
        return {"assigned_count": 0, "details": [], "message": "Aucun track non-assigné avec assez de frames."}

    total_tracks = len(ref_track_ids) + len(unassigned_track_ids)
    assignment_progress[match_id] = {
        "current": 0, "total": total_tracks, "percent": 0,
        "phase": "Extraction des caractéristiques de référence...",
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

    # ── Stream through video: extract frames + features in one pass ─────
    # Only ONE frame in memory at a time — no RAM accumulation.
    n_unique_frames = len({int(s[0] * 30) for s in all_samples})  # approx unique frames
    logger.info(f"Match {match_id}: {len(all_samples)} samples across ~{n_unique_frames} frames for {len(unassigned_track_ids)} unassigned tracks")

    def on_progress(current, total):
        pct = round(current / total * 70, 1) if total > 0 else 0  # 0-70%
        assignment_progress[match_id].update({
            "current": current, "total": total,
            "percent": pct,
            "phase": f"Analyse vidéo ({current}/{total} frames)...",
        })

    all_features = _stream_extract_features(video_path, all_samples, progress_cb=on_progress)

    # ── Build reference feature vectors (averaged per track) ────────────
    assignment_progress[match_id].update({
        "percent": 72, "phase": "Construction des profils de référence...",
    })
    ref_features: dict[int, dict] = {}
    for tid, indices in ref_samples_map.items():
        valid = [all_features[i] for i in indices if all_features[i] is not None]
        if valid:
            averaged = {}
            for key in valid[0].keys():
                arrays = [f[key] for f in valid if key in f]
                if arrays:
                    averaged[key] = np.mean(arrays, axis=0).astype(np.float32)
            ref_features[tid] = averaged

    # Map player_id -> reference features
    player_refs: dict[int, dict] = {}
    for tid, assignment in player_assignments.items():
        if tid in ref_features:
            player_refs[assignment.player_id] = ref_features[tid]

    referee_ref_feats: list[dict] = []
    for tid in referee_assignments:
        if tid in ref_features:
            referee_ref_feats.append(ref_features[tid])

    # Average referee features
    referee_ref: dict | None = None
    if referee_ref_feats:
        referee_ref = {}
        for key in referee_ref_feats[0].keys():
            arrays = [f[key] for f in referee_ref_feats if key in f]
            if arrays:
                referee_ref[key] = np.mean(arrays, axis=0).astype(np.float32)

    # ── Match unassigned tracks ─────────────────────────────────────────
    details = []
    assigned_count = 0
    n_unassigned = len(unassigned_track_ids)

    for i, tid in enumerate(unassigned_track_ids):
        indices = una_samples_map.get(tid, [])
        valid = [all_features[i_s] for i_s in indices if all_features[i_s] is not None]
        if not valid:
            continue

        # Average features for this track
        track_feats = {}
        for key in valid[0].keys():
            arrays = [f[key] for f in valid if key in f]
            if arrays:
                track_feats[key] = np.mean(arrays, axis=0).astype(np.float32)

        best_dist = float("inf")
        best_player_id: int | None = None
        best_is_referee = False

        # Compare against each player
        for pid, ref_feat in player_refs.items():
            dist = compare_features(track_feats, ref_feat)
            if dist < best_dist:
                best_dist = dist
                best_player_id = pid
                best_is_referee = False

        # Compare against referee reference
        if referee_ref is not None:
            dist = compare_features(track_feats, referee_ref)
            if dist < best_dist:
                best_dist = dist
                best_player_id = None
                best_is_referee = True

        if best_dist < threshold:
            assignment = TrackAssignment(
                match_id=match_id,
                track_id=tid,
                player_id=best_player_id,
                is_referee=best_is_referee,
            )
            db.add(assignment)
            assigned_count += 1

            label = "referee" if best_is_referee else f"player {best_player_id}"
            details.append({
                "track_id": tid,
                "player_id": best_player_id,
                "is_referee": best_is_referee,
                "distance": round(best_dist, 4),
            })
            logger.info("Auto-assigned track %d -> %s (dist=%.4f)", tid, label, best_dist)

        if i % 5 == 0:
            pct = 75 + round(i / n_unassigned * 25, 1)  # matching = 75-100%
            assignment_progress[match_id].update({
                "current": i, "total": n_unassigned,
                "percent": pct,
                "phase": f"Comparaison des tracks ({i}/{n_unassigned})...",
            })

    db.commit()

    return {
        "assigned_count": assigned_count,
        "details": details,
    }
