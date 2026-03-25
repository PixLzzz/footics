"""
Frame-by-frame identity correction with brightness team filter + ReID appearance.

Pipeline:
  1. Build player profiles: brightness (team filter) + ReID features (discrimination)
  2. HARD REJECT any detection whose brightness differs from team by > threshold
  3. Among team detections: match to players using cost matrix:
     - Position + velocity prediction (70% weight) — primary spatial signal
     - ReID appearance similarity (20% weight) — distinguishes same-team players
     - Color histogram distance (10% weight) — lightweight backup
  4. Hungarian algorithm for optimal 1-to-1 assignment
  5. Update track_ids in database

Design decisions:
  - Brightness filter is the FIRST gate — reliably separates white vs dark jerseys
  - ReID features (142D descriptor) replace color histogram as the main appearance
    signal for same-team discrimination. They combine color, spatial layout, and
    body proportions — much more discriminative than histogram alone.
  - Position+velocity is still dominant (70%) because appearance changes with
    pose/angle/occlusion, but position is always reliable for short time gaps.
  - ReID gallery from the tracking phase is loaded if available, avoiding
    re-extraction from video for the assigned player profiles.
"""

import cv2
import numpy as np
import logging
from collections import defaultdict
from sqlalchemy.orm import Session
from sqlalchemy import func

from models import TrackingFrame, TrackAssignment, Match

logger = logging.getLogger(__name__)

correction_progress = {}

SAMPLE_EVERY = 3
PROFILE_SAMPLES = 10
MAX_MATCH_COST = 0.60
MAX_POSITION_JUMP = 0.12
BRIGHTNESS_TOLERANCE = 45

# Cost weights — ReID now takes a significant share from color histogram
POSITION_WEIGHT = 0.70
REID_WEIGHT = 0.20
COLOR_WEIGHT = 0.10


def _extract_torso_crop(frame, bbox_x, bbox_y, bbox_w, bbox_h):
    h_img, w_img = frame.shape[:2]
    x1 = int(max(0, (bbox_x + bbox_w * 0.15) * w_img))
    x2 = int(min(w_img, (bbox_x + bbox_w * 0.85) * w_img))
    y1 = int(max(0, (bbox_y + bbox_h * 0.15) * h_img))
    y2 = int(min(h_img, (bbox_y + bbox_h * 0.60) * h_img))
    if x2 - x1 < 6 or y2 - y1 < 6:
        return None
    crop = frame[y1:y2, x1:x2]
    return crop if crop.size > 0 else None


def _get_brightness(crop):
    if crop is None:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    return float(np.mean(hsv[:, :, 2]))


def _get_color_hist(crop):
    if crop is None:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [12, 12], [0, 180, 0, 256])
    cv2.normalize(hist, hist, norm_type=cv2.NORM_L1)
    return hist.flatten()


def _hist_distance(h1, h2):
    if h1 is None or h2 is None:
        return 1.0
    return cv2.compareHist(
        h1.astype(np.float32), h2.astype(np.float32),
        cv2.HISTCMP_BHATTACHARYYA,
    )


def _pos_distance(x1, y1, x2, y2):
    d = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
    return min(1.0, d / 0.25)


def correct_identities(match_id: int, db: Session) -> dict:
    match = db.query(Match).filter(Match.id == match_id).first()
    if not match:
        return {"error": "Match not found"}

    existing = db.query(TrackAssignment).filter_by(match_id=match_id).all()
    player_assignments = {a.track_id: a for a in existing if a.player_id and not a.is_referee}

    if len(player_assignments) < 2:
        return {"error": "Assignez au moins 2 joueurs manuellement d'abord."}

    correction_progress[match_id] = {
        "percent": 0, "phase": "Chargement...", "done": False,
    }

    all_frames = (
        db.query(TrackingFrame)
        .filter_by(match_id=match_id)
        .order_by(TrackingFrame.timestamp_seconds, TrackingFrame.track_id)
        .all()
    )
    if not all_frames:
        return {"error": "Aucune donnée de tracking."}

    frames_by_ts = defaultdict(list)
    for f in all_frames:
        frames_by_ts[f.timestamp_seconds].append(f)

    all_timestamps = sorted(frames_by_ts.keys())
    total_ts = len(all_timestamps)

    correction_progress[match_id] = {
        "percent": 3, "phase": "Chargement ReID...", "done": False,
    }

    # ── Load ReID features from saved gallery ────────────────────────
    # These were extracted during tracking (video_processor.py) so we
    # don't need to re-extract from video for known tracks.
    reid_extractor = None
    saved_reid = {}
    try:
        from tracker.reid import AppearanceExtractor, cosine_similarity
        from video_processor import load_reid_gallery
        reid_extractor = AppearanceExtractor()
        saved_reid = load_reid_gallery(match_id)
        if saved_reid:
            logger.info(f"Match {match_id}: loaded ReID gallery ({len(saved_reid)} tracks)")
    except Exception as e:
        logger.warning(f"ReID not available: {e}")

    correction_progress[match_id] = {
        "percent": 5, "phase": "Profils joueurs...", "done": False,
    }

    cap = cv2.VideoCapture(match.video_path)
    if not cap.isOpened():
        return {"error": "Cannot open video."}
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        cap.release()
        return {"error": "Invalid FPS."}

    player_canonical_tid = {}
    player_brightness = {}
    player_hists = {}
    player_reid = {}  # pid → 142D ReID descriptor

    for orig_tid, assignment in player_assignments.items():
        pid = assignment.player_id
        player_canonical_tid[pid] = orig_tid

        # Try to load ReID descriptor from saved gallery
        if orig_tid in saved_reid:
            player_reid[pid] = saved_reid[orig_tid]

        track_frames = [f for f in all_frames if f.track_id == orig_tid]
        if not track_frames:
            continue

        n = min(PROFILE_SAMPLES, len(track_frames))
        step = max(1, len(track_frames) // (n + 1))
        sampled = [track_frames[step * (i + 1)] for i in range(n)
                    if step * (i + 1) < len(track_frames)]
        if not sampled:
            sampled = [track_frames[len(track_frames) // 2]]

        b_vals, h_vals, reid_vals = [], [], []
        for sf in sampled:
            frame_num = int(sf.timestamp_seconds * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()
            if not ret:
                continue
            crop = _extract_torso_crop(frame, sf.bbox_x, sf.bbox_y, sf.bbox_w, sf.bbox_h)
            b = _get_brightness(crop)
            h = _get_color_hist(crop)
            if b is not None:
                b_vals.append(b)
            if h is not None:
                h_vals.append(h)

            # Extract ReID features if not already loaded from gallery
            if pid not in player_reid and reid_extractor and crop is not None:
                feat = reid_extractor.extract_features(crop, sf.bbox_w, sf.bbox_h)
                if feat is not None:
                    reid_vals.append(feat)

        if b_vals:
            player_brightness[pid] = float(np.median(b_vals))
        if h_vals:
            player_hists[pid] = np.median(h_vals, axis=0).astype(np.float32)
        if reid_vals and pid not in player_reid:
            player_reid[pid] = np.median(np.stack(reid_vals), axis=0).astype(np.float32)
            norm = np.linalg.norm(player_reid[pid])
            if norm > 0:
                player_reid[pid] /= norm

    if not player_brightness:
        cap.release()
        return {"error": "Impossible d'extraire les profils."}

    team_brightness = float(np.median(list(player_brightness.values())))
    logger.info(
        f"Match {match_id}: team brightness={team_brightness:.0f}, "
        f"ReID profiles={len(player_reid)}/{len(player_canonical_tid)}"
    )

    # Team labels
    team_track_ids = None
    try:
        from team_classifier import get_team_labels
        team_labels = get_team_labels(match_id)
        if team_labels:
            from collections import Counter
            al = [team_labels.get(tid) for tid in player_assignments if tid in team_labels]
            al = [l for l in al if l is not None]
            if al:
                user_label = Counter(al).most_common(1)[0][0]
                team_track_ids = {tid for tid, l in team_labels.items() if l == user_label}
    except Exception:
        pass

    correction_progress[match_id] = {
        "percent": 10, "phase": "Correction frame par frame...", "done": False,
    }

    detection_corrections = {}
    player_ids = list(player_canonical_tid.keys())
    n_players = len(player_ids)
    player_state = {}

    sampled_set = set(range(0, total_ts, SAMPLE_EVERY))
    sampled_set.add(0)
    if total_ts > 0:
        sampled_set.add(total_ts - 1)

    last_read_frame = -1
    has_reid = bool(player_reid) and reid_extractor is not None

    for ts_idx, ts in enumerate(all_timestamps):
        detections = frames_by_ts[ts]
        is_sampled = ts_idx in sampled_set

        det_positions = [(det.bbox_x + det.bbox_w / 2, det.bbox_y + det.bbox_h / 2)
                         for det in detections]

        if is_sampled:
            frame_num = int(ts * fps)
            frame = None
            if frame_num > last_read_frame:
                skip = frame_num - last_read_frame - 1
                if 0 < skip < 30:
                    for _ in range(skip):
                        cap.grab()
                elif skip >= 30 or last_read_frame < 0:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
                ret, frame = cap.read()
                if ret:
                    last_read_frame = frame_num

            team_det_indices = []
            if frame is not None:
                for i, det in enumerate(detections):
                    if team_track_ids is not None and det.track_id not in team_track_ids:
                        continue
                    crop = _extract_torso_crop(frame, det.bbox_x, det.bbox_y, det.bbox_w, det.bbox_h)
                    b = _get_brightness(crop)
                    if b is not None and abs(b - team_brightness) > BRIGHTNESS_TOLERANCE:
                        continue
                    team_det_indices.append(i)
            else:
                for i, det in enumerate(detections):
                    if team_track_ids is not None and det.track_id not in team_track_ids:
                        continue
                    team_det_indices.append(i)

            if team_det_indices:
                n_det = len(team_det_indices)
                cost = np.full((n_players, n_det), 1.0, dtype=np.float64)

                # Extract appearance features for detections
                det_hists = {}
                det_reid_feats = {}
                if frame is not None:
                    for di, det_i in enumerate(team_det_indices):
                        det = detections[det_i]
                        crop = _extract_torso_crop(frame, det.bbox_x, det.bbox_y, det.bbox_w, det.bbox_h)
                        det_hists[di] = _get_color_hist(crop)
                        # Extract ReID features for this detection
                        if has_reid and crop is not None:
                            feat = reid_extractor.extract_features(
                                crop, det.bbox_w, det.bbox_h)
                            if feat is not None:
                                det_reid_feats[di] = feat

                for pi, pid in enumerate(player_ids):
                    state = player_state.get(pid)
                    if state:
                        pred_x = state["pos"][0] + state["vel"][0]
                        pred_y = state["pos"][1] + state["vel"][1]
                    else:
                        pred_x, pred_y = None, None

                    for di, det_i in enumerate(team_det_indices):
                        # ── Position cost (70%) ──────────────────────
                        if pred_x is not None:
                            pos_dist = _pos_distance(
                                det_positions[det_i][0], det_positions[det_i][1],
                                pred_x, pred_y,
                            )
                        elif state:
                            pos_dist = _pos_distance(
                                det_positions[det_i][0], det_positions[det_i][1],
                                state["pos"][0], state["pos"][1],
                            )
                        else:
                            pos_dist = 0.5

                        # ── ReID appearance cost (20%) ───────────────
                        reid_dist = 0.5
                        if has_reid and pid in player_reid and di in det_reid_feats:
                            sim = float(np.dot(player_reid[pid], det_reid_feats[di]))
                            reid_dist = 1.0 - max(0.0, sim)

                        # ── Color histogram cost (10%) ───────────────
                        color_dist = 0.5
                        if pid in player_hists and di in det_hists:
                            color_dist = _hist_distance(det_hists[di], player_hists[pid])

                        cost[pi, di] = (
                            POSITION_WEIGHT * pos_dist
                            + REID_WEIGHT * reid_dist
                            + COLOR_WEIGHT * color_dist
                        )

                try:
                    from scipy.optimize import linear_sum_assignment
                    row_idx, col_idx = linear_sum_assignment(cost)
                except ImportError:
                    row_idx, col_idx = _greedy_match(cost)

                for ri, ci in zip(row_idx, col_idx):
                    if cost[ri, ci] > MAX_MATCH_COST:
                        continue
                    pid = player_ids[ri]
                    det_i = team_det_indices[ci]
                    _update_state(player_state, pid, det_positions[det_i])
                    detection_corrections[detections[det_i].id] = player_canonical_tid[pid]

            if frame is not None:
                del frame

            if ts_idx % 100 == 0:
                pct = 10 + round(ts_idx / total_ts * 80, 1)
                correction_progress[match_id] = {
                    "percent": pct,
                    "phase": f"Correction ({ts_idx}/{total_ts})...",
                    "done": False,
                }

        else:
            # Non-sampled frames: position-only matching (fast)
            team_det_indices = list(range(len(detections)))
            if team_track_ids is not None:
                team_det_indices = [i for i, det in enumerate(detections)
                                     if det.track_id in team_track_ids]

            available = [pid for pid in player_ids if pid in player_state]
            if available and team_det_indices:
                n_avail = len(available)
                n_det = len(team_det_indices)
                cost = np.full((n_avail, n_det), 1.0, dtype=np.float64)

                for pi, pid in enumerate(available):
                    state = player_state[pid]
                    pred_x = state["pos"][0] + state["vel"][0]
                    pred_y = state["pos"][1] + state["vel"][1]
                    for di, det_i in enumerate(team_det_indices):
                        cost[pi, di] = _pos_distance(
                            det_positions[det_i][0], det_positions[det_i][1],
                            pred_x, pred_y,
                        )

                row_idx, col_idx = _greedy_match(cost)
                for ri, ci in zip(row_idx, col_idx):
                    if cost[ri, ci] > MAX_POSITION_JUMP / 0.25:
                        continue
                    pid = available[ri]
                    det_i = team_det_indices[ci]
                    _update_state(player_state, pid, det_positions[det_i])
                    detection_corrections[detections[det_i].id] = player_canonical_tid[pid]

    cap.release()

    correction_progress[match_id] = {
        "percent": 92, "phase": "Application...", "done": False,
    }

    corrected_count = 0
    items = list(detection_corrections.items())
    for i in range(0, len(items), 500):
        for fid, new_tid in items[i:i + 500]:
            db.query(TrackingFrame).filter(TrackingFrame.id == fid).update(
                {"track_id": new_tid}, synchronize_session=False
            )
            corrected_count += 1
        db.flush()
    db.commit()

    db.query(TrackAssignment).filter_by(match_id=match_id).delete()
    for pid, tid in player_canonical_tid.items():
        if db.query(TrackingFrame).filter_by(match_id=match_id, track_id=tid).first():
            db.add(TrackAssignment(
                match_id=match_id, track_id=tid,
                player_id=pid, is_referee=False,
            ))
    db.commit()

    correction_progress[match_id] = {
        "percent": 100, "phase": "Terminé",
        "done": True, "corrected": corrected_count,
    }

    logger.info(f"Match {match_id}: corrected {corrected_count}/{len(all_frames)}")

    return {
        "corrected_count": corrected_count,
        "player_count": n_players,
        "total_detections": len(all_frames),
    }


def _update_state(player_state, pid, new_pos):
    old = player_state.get(pid)
    if old:
        vx = new_pos[0] - old["pos"][0]
        vy = new_pos[1] - old["pos"][1]
        vx = 0.6 * vx + 0.4 * old["vel"][0]
        vy = 0.6 * vy + 0.4 * old["vel"][1]
    else:
        vx, vy = 0.0, 0.0
    player_state[pid] = {"pos": new_pos, "vel": (vx, vy)}


def _greedy_match(cost_matrix):
    n_rows, n_cols = cost_matrix.shape
    rows, cols = [], []
    used_r, used_c = set(), set()
    flat = sorted((cost_matrix[r, c], r, c)
                  for r in range(n_rows) for c in range(n_cols))
    for _, r, c in flat:
        if r not in used_r and c not in used_c:
            rows.append(r)
            cols.append(c)
            used_r.add(r)
            used_c.add(c)
    return rows, cols
