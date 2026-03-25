"""
Post-processing for tracking results: merging, denoising, confidence scoring.

WHY post-processing:
    Raw tracker output has several issues that post-processing fixes:
    1. Track fragmentation — one player creates 3-4 track IDs over a match
       due to brief occlusions or detection drops
    2. Noise tracks — false detections (shadows, logos, spectators) create
       short spurious tracks
    3. No confidence scores — all tracks are treated equally even though
       some are clearly more reliable than others

DESIGN DECISIONS:

    1. Appearance-aware merging:
       The original merge algorithm only used position (last/first point
       within distance threshold). We add appearance similarity from the
       ReID module — two tracks that look similar AND are close in space/time
       are much more likely to be the same player.

    2. Multi-factor confidence scoring:
       Track confidence = weighted combination of:
       - Duration (longer tracks are more reliable)
       - Detection confidence (mean YOLO confidence)
       - Consistency (low variance in position changes = smooth track)
       - Appearance stability (consistent appearance features over time)

    3. Hierarchical merging:
       First merge obvious cases (high appearance similarity, small gap).
       Then merge borderline cases. This prevents chain-merging errors
       where A merges with B, then AB merges with C incorrectly.
"""

import numpy as np
import logging
from typing import Optional
from collections import defaultdict

logger = logging.getLogger(__name__)

# Merge thresholds
MERGE_MAX_GAP_S = 4.0          # Max time gap between tracks to consider merging
MERGE_MAX_DISTANCE = 0.15      # Max spatial distance (normalized)
MERGE_MIN_APPEARANCE_SIM = 0.5  # Min ReID similarity for appearance-aware merge
MERGE_POSITION_WEIGHT = 0.4
MERGE_APPEARANCE_WEIGHT = 0.6

# Noise removal
MIN_TRACK_FRAMES = 15          # Tracks shorter than this are removed
MIN_TRACK_DURATION_S = 0.5     # Tracks shorter than 0.5s are removed

# Confidence scoring weights
CONF_DURATION_WEIGHT = 0.25
CONF_DETECTION_WEIGHT = 0.30
CONF_CONSISTENCY_WEIGHT = 0.25
CONF_APPEARANCE_WEIGHT = 0.20


class TrackPostProcessor:
    """Post-process raw tracking results for cleaner, more reliable output."""

    def __init__(self, fps: float = 30.0):
        self.fps = fps

    def remove_noise_tracks(self, tracks_by_id: dict,
                            min_frames: int = MIN_TRACK_FRAMES,
                            min_duration: float = MIN_TRACK_DURATION_S
                            ) -> dict:
        """Remove short/spurious tracks.

        A track is noise if:
        - It has fewer than min_frames detections, OR
        - Its total duration is less than min_duration seconds

        WHY both criteria: A track could have many frames but in a tiny
        time window (rapid false detections), or few frames spread over
        a long time (intermittent ghost).
        """
        cleaned = {}
        removed = 0

        for tid, frames in tracks_by_id.items():
            if len(frames) < min_frames:
                removed += 1
                continue

            timestamps = [f["timestamp_seconds"] for f in frames]
            duration = max(timestamps) - min(timestamps) if timestamps else 0

            if duration < min_duration:
                removed += 1
                continue

            cleaned[tid] = frames

        if removed > 0:
            logger.info(f"Removed {removed} noise tracks (< {min_frames} frames "
                        f"or < {min_duration}s)")

        return cleaned

    def merge_fragmented_tracks(self, tracks_by_id: dict,
                                gallery=None,
                                max_gap: float = MERGE_MAX_GAP_S,
                                max_dist: float = MERGE_MAX_DISTANCE
                                ) -> tuple[dict, dict]:
        """Merge fragmented tracks that likely belong to the same player.

        Two-pass approach:
          Pass 1 (strict): High appearance similarity + close in space/time
          Pass 2 (relaxed): Position-only for tracks without appearance features

        Args:
            tracks_by_id: Dict of track_id → list of frame dicts
            gallery: Optional TrackGallery for appearance-aware merging
            max_gap: Maximum time gap between track end and start
            max_dist: Maximum spatial distance between last/first positions

        Returns:
            (merged_tracks_by_id, merge_map) where merge_map maps
            old track IDs to their new (merged) track IDs
        """
        # Build track summaries
        summaries = {}
        for tid, frames in tracks_by_id.items():
            frames_sorted = sorted(frames, key=lambda f: f["timestamp_seconds"])
            first = frames_sorted[0]
            last = frames_sorted[-1]
            summaries[tid] = {
                "start": first["timestamp_seconds"],
                "end": last["timestamp_seconds"],
                "first_pos": (
                    first["bbox_x"] + first["bbox_w"] / 2,
                    first["bbox_y"] + first["bbox_h"] / 2,
                ),
                "last_pos": (
                    last["bbox_x"] + last["bbox_w"] / 2,
                    last["bbox_y"] + last["bbox_h"] / 2,
                ),
                "n_frames": len(frames),
            }

        merge_map = {}  # old_tid → new_tid
        active_ids = set(tracks_by_id.keys())

        # Pass 1: Appearance-aware merging (if gallery available)
        if gallery:
            self._merge_pass(
                summaries, active_ids, merge_map,
                gallery=gallery,
                max_gap=max_gap,
                max_dist=max_dist * 1.5,  # Slightly relaxed distance with appearance
                min_appearance_sim=MERGE_MIN_APPEARANCE_SIM,
            )

        # Pass 2: Position-only merging for remaining tracks
        self._merge_pass(
            summaries, active_ids, merge_map,
            gallery=None,
            max_gap=max_gap,
            max_dist=max_dist,
            min_appearance_sim=0,
        )

        # Apply merges
        merged_tracks = {}
        for tid in active_ids:
            final_tid = self._resolve_merge(tid, merge_map)
            if final_tid not in merged_tracks:
                merged_tracks[final_tid] = []
            merged_tracks[final_tid].extend(tracks_by_id.get(tid, []))

        # Also add frames from merged-away tracks
        for old_tid, new_tid in merge_map.items():
            final_tid = self._resolve_merge(new_tid, merge_map)
            if old_tid in tracks_by_id:
                if final_tid not in merged_tracks:
                    merged_tracks[final_tid] = []
                merged_tracks[final_tid].extend(tracks_by_id[old_tid])

        # Sort frames within each track
        for tid in merged_tracks:
            merged_tracks[tid].sort(key=lambda f: f["timestamp_seconds"])

        n_merged = len(merge_map)
        if n_merged > 0:
            logger.info(
                f"Merged {n_merged} track fragments → "
                f"{len(merged_tracks)} tracks (was {len(tracks_by_id)})"
            )

        return merged_tracks, merge_map

    def _merge_pass(self, summaries, active_ids, merge_map,
                    gallery, max_gap, max_dist, min_appearance_sim):
        """Single merge pass."""
        sorted_ids = sorted(active_ids,
                            key=lambda t: summaries[t]["n_frames"],
                            reverse=True)

        for i, tid_a in enumerate(sorted_ids):
            if tid_a not in active_ids:
                continue

            for tid_b in sorted_ids[i + 1:]:
                if tid_b not in active_ids:
                    continue

                sa = summaries[tid_a]
                sb = summaries[tid_b]

                # Track A must end before Track B starts (no temporal overlap)
                if sa["end"] < sb["start"]:
                    gap = sb["start"] - sa["end"]
                    last_pos = sa["last_pos"]
                    first_pos = sb["first_pos"]
                elif sb["end"] < sa["start"]:
                    gap = sa["start"] - sb["end"]
                    last_pos = sb["last_pos"]
                    first_pos = sa["first_pos"]
                else:
                    continue  # Overlapping tracks — different players

                if gap > max_gap:
                    continue

                dist = np.hypot(
                    last_pos[0] - first_pos[0],
                    last_pos[1] - first_pos[1],
                )
                if dist > max_dist:
                    continue

                # Appearance check
                if gallery and min_appearance_sim > 0:
                    desc_a = gallery.get_descriptor(tid_a)
                    desc_b = gallery.get_descriptor(tid_b)
                    if desc_a is not None and desc_b is not None:
                        from .reid import cosine_similarity
                        sim = cosine_similarity(desc_a, desc_b)
                        if sim < min_appearance_sim:
                            continue
                    elif min_appearance_sim > 0:
                        continue  # Skip if appearance required but not available

                # Merge: keep the longer track ID
                if sa["n_frames"] >= sb["n_frames"]:
                    keep, remove = tid_a, tid_b
                else:
                    keep, remove = tid_b, tid_a

                merge_map[remove] = keep
                active_ids.discard(remove)

                # Update summary for merged track
                summaries[keep] = {
                    "start": min(sa["start"], sb["start"]),
                    "end": max(sa["end"], sb["end"]),
                    "first_pos": sa["first_pos"] if sa["start"] <= sb["start"] else sb["first_pos"],
                    "last_pos": sa["last_pos"] if sa["end"] >= sb["end"] else sb["last_pos"],
                    "n_frames": sa["n_frames"] + sb["n_frames"],
                }

                logger.debug(
                    f"Merge track {remove} → {keep} "
                    f"(gap={gap:.1f}s, dist={dist:.3f})"
                )

    def _resolve_merge(self, tid, merge_map, depth=0):
        """Follow merge chain to find final track ID."""
        if depth > 50:
            return tid
        if tid in merge_map:
            return self._resolve_merge(merge_map[tid], merge_map, depth + 1)
        return tid

    def compute_track_confidence(self, track_frames: list[dict],
                                 gallery=None,
                                 track_id: int = -1) -> float:
        """Compute a confidence score for a track (0.0 to 1.0).

        Higher confidence = more likely to be a real, consistent player track.

        Components:
          1. Duration score: Longer tracks are more reliable
          2. Detection confidence: Higher YOLO confidence = more reliable
          3. Motion consistency: Smooth motion = real player (not noise)
          4. Appearance stability: Consistent appearance = same person

        WHY this combination:
            Duration alone would favor long ghost tracks.
            Detection confidence alone would miss partially occluded players.
            Motion consistency alone would penalize fast-moving players.
            Combined, they form a robust quality metric.
        """
        if not track_frames:
            return 0.0

        timestamps = [f["timestamp_seconds"] for f in track_frames]
        duration = max(timestamps) - min(timestamps) if len(timestamps) > 1 else 0

        # 1. Duration score (saturates at 60s)
        duration_score = min(1.0, duration / 60.0)

        # 2. Detection confidence (mean YOLO confidence)
        confidences = [f.get("confidence", 0.5) for f in track_frames]
        det_score = float(np.mean(confidences))

        # 3. Motion consistency (inverse of position variance)
        consistency_score = self._motion_consistency(track_frames)

        # 4. Appearance stability
        appearance_score = 0.5  # Default if no gallery
        if gallery and track_id >= 0:
            appearance_score = self._appearance_stability(gallery, track_id)

        confidence = (
            CONF_DURATION_WEIGHT * duration_score
            + CONF_DETECTION_WEIGHT * det_score
            + CONF_CONSISTENCY_WEIGHT * consistency_score
            + CONF_APPEARANCE_WEIGHT * appearance_score
        )

        return round(min(1.0, max(0.0, confidence)), 3)

    def _motion_consistency(self, track_frames: list[dict]) -> float:
        """Score how consistent (smooth) the track's motion is.

        Computed as 1 - normalized_jitter, where jitter is the variance
        of frame-to-frame velocity changes (acceleration).

        Smooth tracks → low jitter → high score.
        Noisy/jumping tracks → high jitter → low score.
        """
        if len(track_frames) < 3:
            return 0.5

        sorted_frames = sorted(track_frames, key=lambda f: f["timestamp_seconds"])

        velocities = []
        for i in range(1, len(sorted_frames)):
            dt = sorted_frames[i]["timestamp_seconds"] - sorted_frames[i - 1]["timestamp_seconds"]
            if dt <= 0 or dt > 1.0:
                continue
            dx = (sorted_frames[i]["bbox_x"] - sorted_frames[i - 1]["bbox_x"]) / dt
            dy = (sorted_frames[i]["bbox_y"] - sorted_frames[i - 1]["bbox_y"]) / dt
            velocities.append((dx, dy))

        if len(velocities) < 2:
            return 0.5

        # Compute acceleration (velocity changes)
        accels = []
        for i in range(1, len(velocities)):
            ax = velocities[i][0] - velocities[i - 1][0]
            ay = velocities[i][1] - velocities[i - 1][1]
            accels.append(np.hypot(ax, ay))

        if not accels:
            return 0.5

        # Low jitter = high consistency
        jitter = float(np.std(accels))
        # Normalize: jitter < 0.01 → perfect, jitter > 0.1 → bad
        return max(0.0, min(1.0, 1.0 - jitter / 0.1))

    def _appearance_stability(self, gallery, track_id: int) -> float:
        """Score how consistent the track's appearance is over time.

        High stability = same person throughout.
        Low stability = possible ID switch or noise.
        """
        if not hasattr(gallery, '_galleries'):
            return 0.5

        features = gallery._galleries.get(track_id, [])
        if len(features) < 2:
            return 0.5

        # Compare each feature to the gallery median
        desc = gallery.get_descriptor(track_id)
        if desc is None:
            return 0.5

        sims = []
        for feat in features:
            sim = float(np.dot(feat, desc))
            sims.append(sim)

        # High mean similarity = stable appearance
        return float(np.mean(sims))

    def compute_all_confidences(self, tracks_by_id: dict,
                                gallery=None) -> dict[int, float]:
        """Compute confidence scores for all tracks.

        Returns:
            Dict of track_id → confidence (0.0 to 1.0)
        """
        scores = {}
        for tid, frames in tracks_by_id.items():
            scores[tid] = self.compute_track_confidence(
                frames, gallery=gallery, track_id=tid)
        return scores
