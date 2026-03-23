import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from sqlalchemy.orm import Session
from models import Match, TrackingFrame, BallFrame, MatchStatus
import logging

logger = logging.getLogger(__name__)

# Sample every N frames to speed up processing
SAMPLE_RATE = 5

# Progress tracking: {match_id: {"current": int, "total": int, "percent": float}}
analysis_progress = {}


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
    """Run YOLO detection + DeepSORT tracking on the match video."""
    from deep_sort_realtime.deepsort_tracker import DeepSort

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

        # Load YOLO model — try GPU, fallback to CPU
        import torch
        model = YOLO("yolov8n.pt")
        device = "cpu"
        if torch.cuda.is_available():
            try:
                model.to("cuda")
                # Quick test to ensure CUDA kernels work
                _test = torch.zeros(1).cuda()
                del _test
                device = "cuda"
            except Exception:
                model.to("cpu")
        logger.info(f"YOLO running on: {device}")

        # Initialize DeepSORT tracker
        use_gpu = device == "cuda"
        tracker = DeepSort(
            max_age=30,
            n_init=3,
            nms_max_overlap=1.0,
            max_cosine_distance=0.3,
            nn_budget=100,
            embedder="mobilenet",
            half=use_gpu,
            embedder_gpu=use_gpu,
        )

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

        while cap.isOpened():
            if frame_number % SAMPLE_RATE != 0:
                # Skip frames efficiently without decoding
                ret = cap.grab()
                if not ret:
                    break
                frame_number += 1
                continue

            ret, frame = cap.read()
            if not ret:
                break

            if True:  # Process this frame
                timestamp = frame_number / fps if fps > 0 else 0

                analysis_progress[match_id] = {
                    "current": frame_number,
                    "total": total_frames,
                    "percent": round(frame_number / total_frames * 100, 1) if total_frames > 0 else 0,
                }

                # YOLO detection: person (0) + sports ball (32)
                results = model(frame, classes=[0, 32], conf=0.25, verbose=False)

                person_dets = []
                if results and results[0].boxes is not None:
                    boxes = results[0].boxes
                    for i in range(len(boxes)):
                        cls = int(boxes.cls[i].item())
                        x1, y1, x2, y2 = boxes.xyxy[i].tolist()
                        conf = float(boxes.conf[i].item())

                        if cls == 0:  # person
                            person_dets.append(
                                ([x1, y1, x2 - x1, y2 - y1], conf, "person")
                            )
                        elif cls == 32:  # sports ball
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

                # DeepSORT update (always call even with empty detections)
                tracks = tracker.update_tracks(person_dets, frame=frame)
                for track in tracks:
                    if not track.is_confirmed():
                        continue
                    tid = track.track_id
                    ltrb = track.to_ltrb()  # left, top, right, bottom
                    x1, y1, x2, y2 = ltrb
                    det_conf = track.get_det_conf()
                    if det_conf is None:
                        det_conf = 0.0

                    # Normalize coordinates
                    nx = max(0, x1 / w)
                    ny = max(0, y1 / h)
                    nw = (x2 - x1) / w
                    nh = (y2 - y1) / h

                    track_batch.append(TrackingFrame(
                        match_id=match_id,
                        frame_number=frame_number,
                        timestamp_seconds=round(timestamp, 3),
                        track_id=int(tid),
                        bbox_x=round(nx, 4),
                        bbox_y=round(ny, 4),
                        bbox_w=round(nw, 4),
                        bbox_h=round(nh, 4),
                        confidence=round(det_conf, 3),
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
