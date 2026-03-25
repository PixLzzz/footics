"""
Video analysis pipeline: YOLO detection + multi-object tracking + ReID.

Pipeline architecture:
    1. YOLO v8s detection — persons (class 0) + ball (class 32)
    2. Multi-object tracking — BoxMOT DeepOCSORT+CLIPReID or ultralytics BoT-SORT
    3. ReID feature extraction — 142D appearance descriptors per track (during tracking)
    4. Post-processing — track merging (appearance-aware), noise removal, confidence scoring
    5. Trajectory smoothing — Savitzky-Golay offline smoothing
    6. Team classification — brightness-based jersey separation

Design decisions:
    - YOLOv8s (small) balances speed and accuracy for real-time processing
    - BoxMOT DeepOCSORT is preferred (CLIPReID for re-identification), with
      graceful fallback to ultralytics BoT-SORT if not installed
    - ReID features are extracted DURING tracking (not as a separate pass)
      to reuse the already-decoded video frames
    - Post-processing uses ReID features for smarter track merging
    - Static camera assumed (gmc_method: none) — no motion compensation needed
"""

import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from sqlalchemy.orm import Session
from sqlalchemy import func
from models import Match, TrackingFrame, BallFrame, MatchStatus
import logging

# Import tracker modules
from tracker.reid import AppearanceExtractor, TrackGallery
from tracker.smoother import TrajectorySmoother
from tracker.postprocess import TrackPostProcessor

logger = logging.getLogger(__name__)

# Process every Nth frame
SAMPLE_RATE = 1

# Minimum confidence for detections
MIN_PERSON_CONF = 0.30
MIN_BALL_CONF = 0.20

# Max image size for inference
INFER_SIZE = 640

# Track merging parameters
MERGE_MAX_GAP_S = 4.0
MERGE_MAX_DISTANCE = 0.15

# Field filtering
FIELD_Y_MIN = 0.02
FIELD_Y_MAX = 0.98

# Progress tracking
analysis_progress = {}
analysis_cancel = set()


def get_video_info(video_path: str) -> dict:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = frame_count / fps if fps > 0 else 0
    cap.release()

    return {
        "fps": fps, "frame_count": frame_count,
        "width": width, "height": height,
        "duration_seconds": duration,
    }


def extract_frame(video_path: str, timestamp_seconds: float) -> np.ndarray | None:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_number = int(timestamp_seconds * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    cap.release()
    return frame if ret else None


def _init_tracker():
    """Initialize BoxMOT DeepOCSORT tracker with CLIPReID.

    Falls back to ultralytics built-in tracker if BoxMOT is not available.
    """
    try:
        from boxmot import DeepOCSort
        tracker = DeepOCSort(
            reid_weights=Path("clipreid_market1501.pt"),  # auto-downloads
            device="cpu",  # will be updated below
            half=False,
            det_thresh=MIN_PERSON_CONF,
            max_age=150,        # keep track alive for 150 frames (5s at 30fps)
            min_hits=3,         # need 3 hits before track is confirmed
            iou_threshold=0.3,  # IoU threshold for association
        )
        logger.info("BoxMOT DeepOCSORT+CLIPReID initialized")
        return tracker, "boxmot"
    except ImportError:
        logger.warning("BoxMOT not installed, falling back to ultralytics tracker")
        return None, "ultralytics"
    except Exception as e:
        logger.warning(f"BoxMOT init failed ({e}), falling back to ultralytics tracker")
        return None, "ultralytics"


def process_video(match_id: int, db: Session):
    """Run YOLO detection + BoxMOT DeepOCSORT tracking on the match video."""
    match = db.query(Match).filter(Match.id == match_id).first()
    if not match:
        logger.error(f"Match {match_id} not found")
        return

    match.status = MatchStatus.PROCESSING
    db.commit()

    try:
        video_path = match.video_path
        info = get_video_info(video_path)
        match.duration_seconds = info["duration_seconds"]
        match.fps = info["fps"]
        match.width = info["width"]
        match.height = info["height"]
        db.commit()

        # Load YOLO model (detection only)
        import torch
        device = "cpu"
        if torch.cuda.is_available():
            try:
                _test = torch.zeros(1).cuda()
                del _test
                device = 0
            except Exception:
                device = "cpu"
        model = YOLO("yolov8s.pt")
        logger.info(f"YOLO running on: {'GPU ' + torch.cuda.get_device_name(0) if device == 0 else 'CPU'}")

        # Initialize BoxMOT tracker
        tracker, tracker_type = _init_tracker()
        if tracker and device == 0:
            tracker.device = torch.device("cuda:0")

        # Fallback tracker config for ultralytics
        tracker_config = str(Path(__file__).parent / "futsal_botsort.yaml")

        # Clear existing tracking data
        db.query(TrackingFrame).filter(TrackingFrame.match_id == match_id).delete()
        db.query(BallFrame).filter(BallFrame.match_id == match_id).delete()
        db.commit()

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_number = 0
        track_batch = []
        ball_batch = []

        # ── ReID: Initialize appearance extractor and gallery ──────────
        # Features are extracted DURING tracking to reuse decoded frames.
        # This avoids a second pass over the video.
        reid_extractor = AppearanceExtractor()
        reid_gallery = TrackGallery()

        analysis_progress[match_id] = {"current": 0, "total": total_frames, "percent": 0}
        analysis_cancel.discard(match_id)

        while cap.isOpened():
            if match_id in analysis_cancel:
                logger.info(f"Match {match_id} analysis cancelled by user")
                analysis_cancel.discard(match_id)
                cap.release()
                match.status = MatchStatus.UPLOADED
                db.commit()
                analysis_progress.pop(match_id, None)
                return

            if frame_number % SAMPLE_RATE != 0:
                ret = cap.grab()
                if not ret:
                    break
                frame_number += 1
                continue

            ret, frame = cap.read()
            if not ret:
                break

            timestamp = frame_number / fps if fps > 0 else 0

            analysis_progress[match_id] = {
                "current": frame_number, "total": total_frames,
                "percent": round(frame_number / total_frames * 100, 1) if total_frames > 0 else 0,
            }

            if tracker_type == "boxmot":
                # ── BoxMOT path: YOLO detect → BoxMOT track ──────────────
                det_results = model.predict(
                    frame, classes=[0, 32], conf=MIN_BALL_CONF,
                    verbose=False, device=device, imgsz=INFER_SIZE,
                )

                if det_results and det_results[0].boxes is not None:
                    boxes = det_results[0].boxes

                    # Separate person and ball detections
                    person_dets = []
                    for i in range(len(boxes)):
                        cls = int(boxes.cls[i].item())
                        conf = float(boxes.conf[i].item())
                        x1, y1, x2, y2 = boxes.xyxy[i].tolist()

                        if cls == 32:  # Ball
                            cx = ((x1 + x2) / 2) / w
                            cy = ((y1 + y2) / 2) / h
                            ball_batch.append(BallFrame(
                                match_id=match_id, frame_number=frame_number,
                                timestamp_seconds=round(timestamp, 3),
                                x=round(cx, 4), y=round(cy, 4),
                                confidence=round(conf, 3),
                            ))
                        elif cls == 0 and conf >= MIN_PERSON_CONF:
                            person_dets.append([x1, y1, x2, y2, conf])

                    # Feed person detections to BoxMOT tracker
                    if person_dets:
                        dets_array = np.array(person_dets, dtype=np.float32)
                        # BoxMOT expects: N×5 array [x1, y1, x2, y2, conf]
                        tracks = tracker.update(dets_array, frame)
                        # tracks: N×8 array [x1, y1, x2, y2, id, conf, cls, idx]

                        if tracks is not None and len(tracks) > 0:
                            for t in tracks:
                                x1, y1, x2, y2 = t[0], t[1], t[2], t[3]
                                tid = int(t[4])
                                conf = float(t[5]) if len(t) > 5 else 0.5

                                nx = max(0, x1 / w)
                                ny = max(0, y1 / h)
                                nw = max(0, (x2 - x1) / w)
                                nh = max(0, (y2 - y1) / h)

                                center_y = ny + nh / 2
                                if center_y < FIELD_Y_MIN or center_y > FIELD_Y_MAX:
                                    continue
                                if nw > 0.25 or nh > 0.7:
                                    continue

                                # Extract ReID appearance features for this detection
                                feat = reid_extractor.extract_from_frame(
                                    frame, nx, ny, nw, nh)
                                reid_gallery.update(tid, feat)

                                track_batch.append(TrackingFrame(
                                    match_id=match_id, frame_number=frame_number,
                                    timestamp_seconds=round(timestamp, 3),
                                    track_id=tid,
                                    bbox_x=round(nx, 4), bbox_y=round(ny, 4),
                                    bbox_w=round(nw, 4), bbox_h=round(nh, 4),
                                    confidence=round(conf, 3),
                                ))

            else:
                # ── Ultralytics fallback path ─────────────────────────────
                track_results = model.track(
                    frame, classes=[0, 32], conf=MIN_BALL_CONF,
                    persist=True, verbose=False,
                    tracker=tracker_config, device=device, imgsz=INFER_SIZE,
                )

                if track_results and track_results[0].boxes is not None:
                    boxes = track_results[0].boxes
                    for i in range(len(boxes)):
                        cls = int(boxes.cls[i].item())
                        conf = float(boxes.conf[i].item())
                        x1, y1, x2, y2 = boxes.xyxy[i].tolist()

                        if cls == 0 and conf >= MIN_PERSON_CONF:
                            if boxes.id is None:
                                continue
                            tid = int(boxes.id[i].item())
                            nx = max(0, x1 / w)
                            ny = max(0, y1 / h)
                            nw = max(0, (x2 - x1) / w)
                            nh = max(0, (y2 - y1) / h)

                            center_y = ny + nh / 2
                            if center_y < FIELD_Y_MIN or center_y > FIELD_Y_MAX:
                                continue
                            if nw > 0.25 or nh > 0.7:
                                continue

                            # Extract ReID appearance features for this detection
                            feat = reid_extractor.extract_from_frame(
                                frame, nx, ny, nw, nh)
                            reid_gallery.update(tid, feat)

                            track_batch.append(TrackingFrame(
                                match_id=match_id, frame_number=frame_number,
                                timestamp_seconds=round(timestamp, 3),
                                track_id=tid,
                                bbox_x=round(nx, 4), bbox_y=round(ny, 4),
                                bbox_w=round(nw, 4), bbox_h=round(nh, 4),
                                confidence=round(conf, 3),
                            ))

                        elif cls == 32:
                            cx = ((x1 + x2) / 2) / w
                            cy = ((y1 + y2) / 2) / h
                            ball_batch.append(BallFrame(
                                match_id=match_id, frame_number=frame_number,
                                timestamp_seconds=round(timestamp, 3),
                                x=round(cx, 4), y=round(cy, 4),
                                confidence=round(conf, 3),
                            ))

            if len(track_batch) >= 500:
                db.bulk_save_objects(track_batch)
                db.commit()
                track_batch = []
            if len(ball_batch) >= 500:
                db.bulk_save_objects(ball_batch)
                db.commit()
                ball_batch = []

            frame_number += 1

        cap.release()

        if track_batch:
            db.bulk_save_objects(track_batch)
        if ball_batch:
            db.bulk_save_objects(ball_batch)
        db.commit()

        # ── Post-processing pipeline ────────────────────────────────────
        post = TrackPostProcessor(fps=fps)

        # Phase 1: Appearance-aware track merging
        analysis_progress[match_id] = {
            "current": total_frames, "total": total_frames,
            "percent": 88, "phase": "Fusion des tracks (apparence)...",
        }
        merge_count = _merge_fragmented_tracks(match_id, db, reid_gallery)
        if merge_count:
            logger.info(f"Match {match_id}: merged {merge_count} fragmented tracks")

        # Phase 2: Remove short noise tracks
        analysis_progress[match_id] = {
            "current": total_frames, "total": total_frames,
            "percent": 90, "phase": "Suppression du bruit...",
        }
        _remove_noise_tracks(match_id, db, min_frames=15)

        # Phase 3: Trajectory smoothing (reduces jitter in stored positions)
        analysis_progress[match_id] = {
            "current": total_frames, "total": total_frames,
            "percent": 92, "phase": "Lissage des trajectoires...",
        }
        _smooth_all_tracks(match_id, db)

        # Phase 4: Compute and log track confidence scores
        analysis_progress[match_id] = {
            "current": total_frames, "total": total_frames,
            "percent": 94, "phase": "Scores de confiance...",
        }
        _compute_track_confidences(match_id, db, post, reid_gallery)

        # Phase 5: Team classification
        analysis_progress[match_id] = {
            "current": total_frames, "total": total_frames,
            "percent": 96, "phase": "Classification des équipes...",
        }
        try:
            from team_classifier import classify_teams
            team_result = classify_teams(match_id, db)
            logger.info(f"Match {match_id}: team classification done — {team_result}")
        except Exception as e:
            logger.warning(f"Team classification failed: {e}")
            import traceback
            traceback.print_exc()

        # Save ReID gallery for later use (identity correction, auto-assign)
        _save_reid_gallery(match_id, reid_gallery)

        match.status = MatchStatus.ANALYZED
        db.commit()
        analysis_progress[match_id] = {"current": total_frames, "total": total_frames, "percent": 100}

        ball_count = db.query(BallFrame).filter_by(match_id=match_id).count()
        track_count = db.query(func.count(func.distinct(TrackingFrame.track_id))).filter(TrackingFrame.match_id == match_id).scalar()
        logger.info(f"Match {match_id}: analysis complete. {track_count} tracks, {ball_count} ball detections")

    except Exception as e:
        logger.error(f"Error processing match {match_id}: {e}")
        import traceback
        traceback.print_exc()
        match.status = MatchStatus.ERROR
        db.commit()
        analysis_progress.pop(match_id, None)
        raise


def _remove_noise_tracks(match_id: int, db: Session, min_frames: int = 15):
    """Remove tracks with fewer than min_frames detections (noise)."""
    short_tracks = (
        db.query(TrackingFrame.track_id)
        .filter(TrackingFrame.match_id == match_id)
        .group_by(TrackingFrame.track_id)
        .having(func.count(TrackingFrame.id) < min_frames)
        .all()
    )
    if short_tracks:
        short_ids = [t[0] for t in short_tracks]
        db.query(TrackingFrame).filter(
            TrackingFrame.match_id == match_id,
            TrackingFrame.track_id.in_(short_ids),
        ).delete(synchronize_session='fetch')
        db.commit()
        logger.info(f"Match {match_id}: removed {len(short_ids)} noise tracks (<{min_frames} frames)")


def _merge_fragmented_tracks(match_id: int, db: Session,
                              reid_gallery: TrackGallery = None) -> int:
    """Merge fragmented tracks that likely belong to the same person.

    Two-pass approach:
      Pass 1 (appearance-aware): If ReID gallery is available, merge tracks
              that have HIGH appearance similarity AND are close in space/time.
              This allows slightly larger spatial gaps because we're confident
              they're the same person based on appearance.
      Pass 2 (position-only): Merge remaining tracks using strict position rules.

    Rules (both passes):
      - Track A must END before track B STARTS (no temporal overlap)
      - Gap between A end and B start must be < MERGE_MAX_GAP_S
      - Last position of A must be close to first position of B
    """
    from tracker.reid import cosine_similarity

    track_info = (
        db.query(
            TrackingFrame.track_id,
            func.count(TrackingFrame.id).label("cnt"),
            func.min(TrackingFrame.timestamp_seconds).label("first_ts"),
            func.max(TrackingFrame.timestamp_seconds).label("last_ts"),
        )
        .filter(TrackingFrame.match_id == match_id)
        .group_by(TrackingFrame.track_id)
        .all()
    )

    if len(track_info) < 2:
        return 0

    tracks = {}
    for t in track_info:
        tracks[t.track_id] = {
            "cnt": t.cnt, "first_ts": t.first_ts, "last_ts": t.last_ts,
        }

    bbox_cache = {}

    def _get_bbox_center(track_id, timestamp):
        key = (track_id, timestamp)
        if key not in bbox_cache:
            row = (
                db.query(TrackingFrame)
                .filter_by(match_id=match_id, track_id=track_id, timestamp_seconds=timestamp)
                .first()
            )
            if row:
                bbox_cache[key] = (row.bbox_x + row.bbox_w / 2, row.bbox_y + row.bbox_h / 2)
            else:
                bbox_cache[key] = None
        return bbox_cache[key]

    merge_pairs = []
    sorted_tids = sorted(tracks.keys(), key=lambda tid: tracks[tid]["first_ts"])

    for i, tid_a in enumerate(sorted_tids):
        info_a = tracks[tid_a]
        last_pos_a = _get_bbox_center(tid_a, info_a["last_ts"])
        if last_pos_a is None:
            continue

        for j in range(i + 1, len(sorted_tids)):
            tid_b = sorted_tids[j]
            info_b = tracks[tid_b]

            gap = info_b["first_ts"] - info_a["last_ts"]
            if gap < 0:
                continue
            if gap > MERGE_MAX_GAP_S:
                break

            first_pos_b = _get_bbox_center(tid_b, info_b["first_ts"])
            if first_pos_b is None:
                continue

            dist = ((last_pos_a[0] - first_pos_b[0]) ** 2 + (last_pos_a[1] - first_pos_b[1]) ** 2) ** 0.5

            # Check appearance similarity if gallery available
            appearance_sim = 0.0
            if reid_gallery:
                desc_a = reid_gallery.get_descriptor(tid_a)
                desc_b = reid_gallery.get_descriptor(tid_b)
                if desc_a is not None and desc_b is not None:
                    appearance_sim = cosine_similarity(desc_a, desc_b)

            # Appearance-aware: relax distance threshold if high appearance match
            effective_max_dist = MERGE_MAX_DISTANCE
            if appearance_sim > 0.6:
                effective_max_dist = MERGE_MAX_DISTANCE * 1.5  # Allow 50% more distance

            if dist < effective_max_dist:
                # Combined score: lower = better merge candidate
                score = dist * (1.0 - 0.5 * appearance_sim)
                merge_pairs.append((score, dist, gap, appearance_sim, tid_a, tid_b))

    merge_pairs.sort()
    used = set()
    merge_count = 0

    for score, dist, gap, app_sim, tid_a, tid_b in merge_pairs:
        if tid_a in used or tid_b in used:
            continue

        info_a = tracks[tid_a]
        info_b = tracks[tid_b]
        keep = tid_a if info_a["cnt"] >= info_b["cnt"] else tid_b
        frag = tid_b if keep == tid_a else tid_a

        updated = (
            db.query(TrackingFrame)
            .filter_by(match_id=match_id, track_id=frag)
            .update({"track_id": keep})
        )
        if updated > 0:
            merge_count += 1
            used.add(tid_a)
            used.add(tid_b)
            tracks[keep] = {
                "cnt": info_a["cnt"] + info_b["cnt"],
                "first_ts": min(info_a["first_ts"], info_b["first_ts"]),
                "last_ts": max(info_a["last_ts"], info_b["last_ts"]),
            }
            logger.info(
                f"Merged track {frag} into {keep} "
                f"(dist={dist:.4f}, gap={gap:.2f}s, appearance={app_sim:.2f})"
            )

    if merge_count:
        db.commit()

    return merge_count


def _smooth_all_tracks(match_id: int, db: Session):
    """Apply trajectory smoothing to all tracks in the database.

    Uses Savitzky-Golay filter to reduce position jitter while preserving
    sharp direction changes (important for dribble detection).
    """
    smoother = TrajectorySmoother(window_size=7, poly_order=2)

    track_ids = (
        db.query(TrackingFrame.track_id)
        .filter(TrackingFrame.match_id == match_id)
        .distinct()
        .all()
    )

    smoothed_count = 0
    for (tid,) in track_ids:
        frames = (
            db.query(TrackingFrame)
            .filter_by(match_id=match_id, track_id=tid)
            .order_by(TrackingFrame.timestamp_seconds)
            .all()
        )

        if len(frames) < 7:  # Too short for smoothing window
            continue

        # Extract center positions
        positions = [
            (f.bbox_x + f.bbox_w / 2, f.bbox_y + f.bbox_h / 2)
            for f in frames
        ]

        smoothed = smoother.smooth_trajectory(positions)

        # Apply smoothed centers back (preserve bbox dimensions)
        for i, f in enumerate(frames):
            cx, cy = smoothed[i]
            new_x = cx - f.bbox_w / 2
            new_y = cy - f.bbox_h / 2
            # Only update if change is significant (avoid unnecessary writes)
            if abs(new_x - f.bbox_x) > 0.0005 or abs(new_y - f.bbox_y) > 0.0005:
                f.bbox_x = round(new_x, 4)
                f.bbox_y = round(new_y, 4)
                smoothed_count += 1

    db.commit()
    logger.info(f"Match {match_id}: smoothed {smoothed_count} positions across {len(track_ids)} tracks")


def _compute_track_confidences(match_id: int, db: Session,
                                post: TrackPostProcessor,
                                reid_gallery: TrackGallery):
    """Compute and log confidence scores for all tracks.

    Confidence scores are stored in a JSON file alongside team labels.
    They help the user prioritize which tracks to assign manually.
    """
    import json

    track_ids = (
        db.query(TrackingFrame.track_id)
        .filter(TrackingFrame.match_id == match_id)
        .distinct()
        .all()
    )

    confidences = {}
    for (tid,) in track_ids:
        frames = (
            db.query(TrackingFrame)
            .filter_by(match_id=match_id, track_id=tid)
            .all()
        )
        frame_dicts = [{
            "timestamp_seconds": f.timestamp_seconds,
            "bbox_x": f.bbox_x, "bbox_y": f.bbox_y,
            "bbox_w": f.bbox_w, "bbox_h": f.bbox_h,
            "confidence": f.confidence,
        } for f in frames]

        score = post.compute_track_confidence(
            frame_dicts, gallery=reid_gallery, track_id=tid)
        confidences[str(tid)] = score

    # Save to disk
    path = f"./data/track_confidence_{match_id}.json"
    with open(path, "w") as f:
        json.dump(confidences, f)

    high_conf = sum(1 for s in confidences.values() if s > 0.5)
    logger.info(
        f"Match {match_id}: {len(confidences)} tracks scored, "
        f"{high_conf} high-confidence (>0.5)"
    )


def _save_reid_gallery(match_id: int, gallery: TrackGallery):
    """Save ReID gallery descriptors to disk for later use.

    The identity corrector and auto-assigner can load these to improve
    their matching without re-extracting features from video.
    """
    import json

    descriptors = {}
    for tid in gallery.track_ids:
        desc = gallery.get_descriptor(tid)
        if desc is not None:
            descriptors[str(tid)] = desc.tolist()

    path = f"./data/reid_gallery_{match_id}.json"
    with open(path, "w") as f:
        json.dump(descriptors, f)

    logger.info(f"Match {match_id}: saved ReID gallery ({len(descriptors)} tracks)")


def load_reid_gallery(match_id: int) -> dict:
    """Load saved ReID gallery descriptors from disk.

    Returns dict of track_id (int) → feature vector (np.ndarray).
    """
    import json

    path = f"./data/reid_gallery_{match_id}.json"
    try:
        with open(path) as f:
            data = json.load(f)
        return {int(k): np.array(v, dtype=np.float32) for k, v in data.items()}
    except (FileNotFoundError, json.JSONDecodeError):
        return {}
