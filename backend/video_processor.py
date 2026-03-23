import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from sqlalchemy.orm import Session
from models import Match, TrackingFrame, BallFrame, MatchStatus
import logging

logger = logging.getLogger(__name__)

# Process every Nth frame — lower = better tracking but slower
SAMPLE_RATE = 2

# Minimum confidence for detections
MIN_PERSON_CONF = 0.25
MIN_BALL_CONF = 0.15

# Max image size for inference (shorter side)
INFER_SIZE = 640

# Custom tracker config path (relative to this file)
TRACKER_CONFIG = str(Path(__file__).parent / "futsal_botsort.yaml")

# Track merging: max gap (seconds) and max position jump to consider two tracks the same person
MERGE_MAX_GAP_S = 3.0
MERGE_MAX_DISTANCE = 0.12  # normalized coords

# Progress tracking: {match_id: {"current": int, "total": int, "percent": float}}
analysis_progress = {}

# Cancellation set: add match_id here to stop a running analysis
analysis_cancel = set()


def get_video_info(video_path: str) -> dict:
    """Extract video metadata."""
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
        "fps": fps,
        "frame_count": frame_count,
        "width": width,
        "height": height,
        "duration_seconds": duration,
    }


def extract_frame(video_path: str, timestamp_seconds: float) -> np.ndarray | None:
    """Extract a single frame at a given timestamp."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_number = int(timestamp_seconds * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    cap.release()

    if ret:
        return frame
    return None


def process_video(match_id: int, db: Session):
    """Run YOLO detection + built-in BoT-SORT tracking on the match video."""
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

        # Load YOLO model — detect usable device
        import torch
        device = "cpu"
        if torch.cuda.is_available():
            try:
                _test = torch.zeros(1).cuda()
                _test += 1
                del _test
                device = 0  # GPU index for ultralytics
            except Exception:
                device = "cpu"
        model = YOLO("yolov8s.pt")  # small model — much better than nano for person tracking
        logger.info(f"YOLO running on: {'GPU ' + torch.cuda.get_device_name(0) if device == 0 else 'CPU'}")

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

        analysis_progress[match_id] = {"current": 0, "total": total_frames, "percent": 0}
        analysis_cancel.discard(match_id)

        while cap.isOpened():
            # Check for cancellation
            if match_id in analysis_cancel:
                logger.info(f"Match {match_id} analysis cancelled by user")
                analysis_cancel.discard(match_id)
                cap.release()
                match.status = MatchStatus.UPLOADED
                db.commit()
                analysis_progress.pop(match_id, None)
                return

            # Skip frames we don't need — grab() is fast (no decode)
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

            # Update progress
            analysis_progress[match_id] = {
                "current": frame_number,
                "total": total_frames,
                "percent": round(frame_number / total_frames * 100, 1) if total_frames > 0 else 0,
            }

            # ── Single inference: person (0) + ball (32) with tracking ──
            track_results = model.track(
                frame,
                classes=[0, 32],
                conf=MIN_BALL_CONF,  # use lower conf to catch balls too
                persist=True,
                verbose=False,
                tracker=TRACKER_CONFIG,
                device=device,
                imgsz=INFER_SIZE,
            )

            if track_results and track_results[0].boxes is not None:
                boxes = track_results[0].boxes
                for i in range(len(boxes)):
                    cls = int(boxes.cls[i].item())
                    conf = float(boxes.conf[i].item())
                    x1, y1, x2, y2 = boxes.xyxy[i].tolist()

                    if cls == 0 and conf >= MIN_PERSON_CONF:
                        # Person — need track ID
                        if boxes.id is None:
                            continue
                        tid = int(boxes.id[i].item())
                        nx = max(0, x1 / w)
                        ny = max(0, y1 / h)
                        nw = max(0, (x2 - x1) / w)
                        nh = max(0, (y2 - y1) / h)

                        track_batch.append(TrackingFrame(
                            match_id=match_id,
                            frame_number=frame_number,
                            timestamp_seconds=round(timestamp, 3),
                            track_id=tid,
                            bbox_x=round(nx, 4),
                            bbox_y=round(ny, 4),
                            bbox_w=round(nw, 4),
                            bbox_h=round(nh, 4),
                            confidence=round(conf, 3),
                        ))

                    elif cls == 32:
                        # Ball — take position, no track ID needed
                        cx = ((x1 + x2) / 2) / w
                        cy = ((y1 + y2) / 2) / h
                        ball_batch.append(BallFrame(
                            match_id=match_id,
                            frame_number=frame_number,
                            timestamp_seconds=round(timestamp, 3),
                            x=round(cx, 4),
                            y=round(cy, 4),
                            confidence=round(conf, 3),
                        ))

            # Bulk insert every 500 records
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

        # Insert remaining
        if track_batch:
            db.bulk_save_objects(track_batch)
        if ball_batch:
            db.bulk_save_objects(ball_batch)
        db.commit()

        # ── Post-processing: merge fragmented tracks ──────────────────
        analysis_progress[match_id] = {
            "current": total_frames, "total": total_frames,
            "percent": 95, "phase": "Fusion des tracks fragmentés...",
        }
        merge_count = _merge_fragmented_tracks(match_id, db)
        if merge_count:
            logger.info(f"Match {match_id}: merged {merge_count} fragmented tracks")

        match.status = MatchStatus.ANALYZED
        db.commit()
        analysis_progress[match_id] = {"current": total_frames, "total": total_frames, "percent": 100}
        logger.info(f"Match {match_id} analysis complete. Ball detections: {db.query(BallFrame).filter_by(match_id=match_id).count()}")

    except Exception as e:
        logger.error(f"Error processing match {match_id}: {e}")
        match.status = MatchStatus.ERROR
        db.commit()
        analysis_progress.pop(match_id, None)
        raise


def _merge_fragmented_tracks(match_id: int, db: Session) -> int:
    """Merge fragmented tracks that likely belong to the same person.

    When BoT-SORT loses a track and re-detects the person, it assigns a new ID.
    This function finds track pairs where:
      - Track A ends within MERGE_MAX_GAP_S seconds before track B starts
      - The last bbox of A is close to the first bbox of B (< MERGE_MAX_DISTANCE)
      - Track B has fewer frames than A (B is the fragment)

    All frames of the smaller track are reassigned to the larger track's ID.
    """
    from sqlalchemy import func

    # Get track stats: first/last timestamp, frame count, first/last bbox center
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

    # Build lookup: track_id -> {cnt, first_ts, last_ts}
    tracks = {}
    for t in track_info:
        tracks[t.track_id] = {
            "cnt": t.cnt, "first_ts": t.first_ts, "last_ts": t.last_ts,
        }

    # For each track, cache the bbox center at first and last frame
    def _get_bbox_center(track_id, timestamp):
        row = (
            db.query(TrackingFrame)
            .filter_by(match_id=match_id, track_id=track_id, timestamp_seconds=timestamp)
            .first()
        )
        if row:
            return (row.bbox_x + row.bbox_w / 2, row.bbox_y + row.bbox_h / 2)
        return None

    # Sort tracks by first_ts for efficient pairing
    sorted_tids = sorted(tracks.keys(), key=lambda tid: tracks[tid]["first_ts"])

    # Find merge candidates: for each track that ends, find the best continuation
    merge_map = {}  # fragment_tid -> keep_tid
    merged = set()

    for i, tid_a in enumerate(sorted_tids):
        if tid_a in merged:
            continue
        info_a = tracks[tid_a]
        last_pos_a = _get_bbox_center(tid_a, info_a["last_ts"])
        if last_pos_a is None:
            continue

        best_match = None
        best_dist = MERGE_MAX_DISTANCE

        for j in range(i + 1, len(sorted_tids)):
            tid_b = sorted_tids[j]
            if tid_b in merged:
                continue
            info_b = tracks[tid_b]

            # Time gap check
            gap = info_b["first_ts"] - info_a["last_ts"]
            if gap < 0:
                continue  # overlapping tracks — not the same person
            if gap > MERGE_MAX_GAP_S:
                break  # sorted by first_ts, no more candidates

            first_pos_b = _get_bbox_center(tid_b, info_b["first_ts"])
            if first_pos_b is None:
                continue

            dist = ((last_pos_a[0] - first_pos_b[0]) ** 2 + (last_pos_a[1] - first_pos_b[1]) ** 2) ** 0.5
            if dist < best_dist:
                best_dist = dist
                best_match = tid_b

        if best_match is not None:
            # Merge smaller into larger
            keep = tid_a if info_a["cnt"] >= tracks[best_match]["cnt"] else best_match
            fragment = best_match if keep == tid_a else tid_a
            merge_map[fragment] = keep
            merged.add(fragment)

    # Apply merges
    merge_count = 0
    for fragment_tid, keep_tid in merge_map.items():
        updated = (
            db.query(TrackingFrame)
            .filter_by(match_id=match_id, track_id=fragment_tid)
            .update({"track_id": keep_tid})
        )
        merge_count += 1 if updated > 0 else 0

    if merge_count:
        db.commit()

    return merge_count
