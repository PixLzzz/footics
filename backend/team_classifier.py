"""
Team classification based on jersey brightness.

Simple, fast, and extremely reliable for teams with visually different jerseys.
No external dependencies needed — just OpenCV.

Strategy:
  1. For each track, sample frames and crop the torso
  2. Compute average brightness (V channel in HSV)
  3. K-means with k=2 on brightness alone → two teams
  4. Store labels as JSON for quick access

This works perfectly when one team wears light jerseys and the other dark.
For very similar jerseys, it falls back to HSV hue clustering.
"""

import cv2
import numpy as np
import json
import logging
from collections import defaultdict
from pathlib import Path
from sqlalchemy.orm import Session
from sqlalchemy import func

from models import TrackingFrame, Match

logger = logging.getLogger(__name__)

SAMPLES_PER_TRACK = 8
MIN_TRACK_FRAMES = 30


def _extract_torso_crop(frame, bbox_x, bbox_y, bbox_w, bbox_h):
    """Crop the torso region (top 60%, center 70%) from a detection."""
    h_img, w_img = frame.shape[:2]

    # Torso: top 15%-60% vertically, center 70% horizontally
    h_box = bbox_h * h_img
    w_box = bbox_w * w_img

    x1 = int(max(0, (bbox_x + bbox_w * 0.15) * w_img))
    x2 = int(min(w_img, (bbox_x + bbox_w * 0.85) * w_img))
    y1 = int(max(0, (bbox_y + bbox_h * 0.15) * h_img))
    y2 = int(min(h_img, (bbox_y + bbox_h * 0.60) * h_img))

    if x2 - x1 < 6 or y2 - y1 < 6:
        return None

    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    return crop


def _get_brightness(crop):
    """Get average brightness of a crop (V channel in HSV, 0-255)."""
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    return float(np.mean(hsv[:, :, 2]))


def _get_hue_sat(crop):
    """Get average hue and saturation of a crop."""
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    return float(np.mean(hsv[:, :, 0])), float(np.mean(hsv[:, :, 1]))


def classify_teams(match_id: int, db: Session) -> dict:
    """Classify tracks into 2 teams based on jersey brightness.

    Returns dict with classification stats.
    """
    match = db.query(Match).filter(Match.id == match_id).first()
    if not match:
        return {"error": "Match not found"}

    # Get tracks with enough frames
    track_info = (
        db.query(
            TrackingFrame.track_id,
            func.count(TrackingFrame.id).label("cnt"),
        )
        .filter(TrackingFrame.match_id == match_id)
        .group_by(TrackingFrame.track_id)
        .having(func.count(TrackingFrame.id) >= MIN_TRACK_FRAMES)
        .order_by(func.count(TrackingFrame.id).desc())
        .all()
    )

    if len(track_info) < 3:
        return {"error": "Not enough tracks to classify"}

    cap = cv2.VideoCapture(match.video_path)
    if not cap.isOpened():
        return {"error": "Cannot open video"}

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        cap.release()
        return {"error": "Invalid FPS"}

    # ── Sample brightness + color for each track ──────────────────────
    track_features = {}  # track_id → {"brightness": float, "hue": float, "sat": float}

    for t in track_info:
        tid = t.track_id

        track_frames = (
            db.query(TrackingFrame)
            .filter_by(match_id=match_id, track_id=tid)
            .order_by(TrackingFrame.timestamp_seconds)
            .all()
        )

        if len(track_frames) < 3:
            continue

        step = max(1, len(track_frames) // (SAMPLES_PER_TRACK + 1))
        sampled = [track_frames[step * (i + 1)] for i in range(SAMPLES_PER_TRACK)
                    if step * (i + 1) < len(track_frames)]

        brightness_vals = []
        hue_vals = []
        sat_vals = []

        for sf in sampled:
            frame_num = int(sf.timestamp_seconds * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()
            if not ret:
                continue

            crop = _extract_torso_crop(frame, sf.bbox_x, sf.bbox_y, sf.bbox_w, sf.bbox_h)
            if crop is None:
                continue

            brightness_vals.append(_get_brightness(crop))
            h, s = _get_hue_sat(crop)
            hue_vals.append(h)
            sat_vals.append(s)

        if len(brightness_vals) >= 2:
            track_features[tid] = {
                "brightness": float(np.median(brightness_vals)),
                "hue": float(np.median(hue_vals)),
                "sat": float(np.median(sat_vals)),
            }

    cap.release()

    if len(track_features) < 3:
        return {"error": "Not enough valid tracks"}

    # ── Classify using brightness ─────────────────────────────────────
    tids = list(track_features.keys())
    brightness = np.array([track_features[tid]["brightness"] for tid in tids])

    # Check if brightness alone separates well (big gap between clusters)
    sorted_b = np.sort(brightness)
    gaps = np.diff(sorted_b)
    max_gap_idx = np.argmax(gaps)
    max_gap = gaps[max_gap_idx]
    brightness_range = sorted_b[-1] - sorted_b[0]

    if brightness_range > 30 and max_gap > brightness_range * 0.15:
        # Good separation by brightness — use simple threshold
        threshold = (sorted_b[max_gap_idx] + sorted_b[max_gap_idx + 1]) / 2
        labels = (brightness > threshold).astype(int)
        logger.info(f"Match {match_id}: team split by brightness threshold={threshold:.0f} "
                    f"(range={sorted_b[0]:.0f}-{sorted_b[-1]:.0f}, gap={max_gap:.0f})")
    else:
        # Brightness too similar — fall back to K-means on brightness + saturation
        from sklearn.cluster import KMeans
        features = np.column_stack([
            brightness / 255.0,
            np.array([track_features[tid]["sat"] for tid in tids]) / 255.0,
        ])
        kmeans = KMeans(n_clusters=2, n_init=10, random_state=42)
        labels = kmeans.fit_predict(features)
        logger.info(f"Match {match_id}: team split by K-means (brightness+saturation)")

    # Build result
    team_labels = {}
    for tid, label in zip(tids, labels):
        team_labels[tid] = int(label)

    # Count per team
    team0 = sum(1 for v in team_labels.values() if v == 0)
    team1 = sum(1 for v in team_labels.values() if v == 1)

    # Save to disk
    data_dir = Path("./data")
    data_dir.mkdir(parents=True, exist_ok=True)
    labels_path = data_dir / f"team_labels_{match_id}.json"
    with open(labels_path, "w") as f:
        json.dump({
            "match_id": match_id,
            "track_labels": {str(k): v for k, v in team_labels.items()},
            "track_features": {str(k): v for k, v in track_features.items()},
        }, f)

    logger.info(f"Match {match_id}: classified {len(team_labels)} tracks — "
                f"Team 0: {team0}, Team 1: {team1}")

    return {
        "total_classified": len(team_labels),
        "team_0": team0,
        "team_1": team1,
    }


def get_team_labels(match_id: int) -> dict:
    """Load team labels from disk. Returns {track_id: team_label} or empty dict."""
    labels_path = Path(f"./data/team_labels_{match_id}.json")
    if not labels_path.exists():
        return {}
    try:
        with open(labels_path) as f:
            data = json.load(f)
        return {int(k): v for k, v in data.get("track_labels", {}).items()}
    except Exception:
        return {}


def is_same_team(crop, assigned_brightness, threshold=40):
    """Quick check: does this crop's brightness match the assigned team?

    Args:
        crop: BGR image crop of the torso
        assigned_brightness: average brightness of the user's team (0-255)
        threshold: max brightness difference to accept

    Returns: True if same team
    """
    if crop is None:
        return False
    b = _get_brightness(crop)
    return abs(b - assigned_brightness) < threshold
