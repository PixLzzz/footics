"""
Appearance-based Re-Identification (ReID) module.

WHY this approach:
    Pure position/velocity matching fails when players cross paths or occlude
    each other. We need VISUAL features to tell players apart.

    Heavy NN-based ReID (CLIPReID, OSNet) requires torch/transformers which
    are large optional dependencies. This module provides a lightweight but
    effective alternative using classical CV features that work with just
    OpenCV + NumPy.

DESIGN DECISIONS:

    1. Multi-feature descriptor (not just color histogram):
       - HSV color histogram (16×8 bins, H×S) → captures jersey color distribution
       - Spatial color layout (4 quadrants in LAB space) → captures WHERE colors are
       - Body proportions (aspect ratio) → distinguishes player builds
       Combined = 142D vector. Much more discriminative than histogram alone.

    2. Gallery-based matching:
       Each track maintains a gallery of features sampled over time.
       The descriptor is the median of gallery samples (robust to outliers).
       This handles appearance variation due to pose, lighting, etc.

    3. Cosine similarity for matching:
       Scale-invariant, works well for histogram-like features.
       Threshold-based matching with configurable confidence.

    4. Torso-only crops:
       Legs change appearance with stride. Head is too small.
       Torso (jersey) is the most stable appearance region.
"""

import cv2
import numpy as np
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Feature dimensions
HIST_H_BINS = 16    # Hue bins
HIST_S_BINS = 8     # Saturation bins
HIST_DIM = HIST_H_BINS * HIST_S_BINS  # 128D
SPATIAL_DIM = 12    # 4 quadrants × 3 LAB channels
PROPORTION_DIM = 2  # aspect ratio + relative area
TOTAL_DIM = HIST_DIM + SPATIAL_DIM + PROPORTION_DIM  # 142D

# Gallery settings
MAX_GALLERY_SIZE = 30       # Max samples per track
GALLERY_SAMPLE_INTERVAL = 5  # Sample every N frames


class AppearanceExtractor:
    """Extract multi-feature appearance descriptor from a person crop.

    The descriptor is a 142D vector combining:
      - HSV color histogram (128D): Overall color distribution of the torso.
        Using H and S channels only (not V) makes it lighting-invariant.
      - Spatial color layout (12D): Mean LAB color of 4 quadrants
        (upper-left, upper-right, lower-left, lower-right). This captures
        WHERE colors appear — e.g., a player with dark shorts and white
        jersey differs from one with all white.
      - Body proportions (2D): Aspect ratio and relative area. Different
        players have different builds even in the same jersey.
    """

    def extract_torso(self, frame: np.ndarray,
                      bbox_x: float, bbox_y: float,
                      bbox_w: float, bbox_h: float) -> Optional[np.ndarray]:
        """Extract torso crop from frame using normalized bbox coordinates.

        Crops the central 70% width × upper 45% height of the bbox.
        This targets the jersey area, avoiding arms (side edges)
        and legs (lower portion).
        """
        h_img, w_img = frame.shape[:2]
        # Crop torso region: center 70% width, 15%-60% height
        x1 = int(max(0, (bbox_x + bbox_w * 0.15) * w_img))
        x2 = int(min(w_img, (bbox_x + bbox_w * 0.85) * w_img))
        y1 = int(max(0, (bbox_y + bbox_h * 0.15) * h_img))
        y2 = int(min(h_img, (bbox_y + bbox_h * 0.60) * h_img))

        if x2 - x1 < 8 or y2 - y1 < 8:
            return None
        crop = frame[y1:y2, x1:x2]
        return crop if crop.size > 0 else None

    def extract_features(self, crop: np.ndarray,
                         bbox_w: float = 0, bbox_h: float = 0) -> Optional[np.ndarray]:
        """Extract 142D appearance descriptor from a torso crop.

        Args:
            crop: BGR torso crop image
            bbox_w: Normalized bbox width (for proportion features)
            bbox_h: Normalized bbox height (for proportion features)

        Returns:
            142D float32 feature vector, L2-normalized
        """
        if crop is None or crop.size == 0:
            return None

        features = []

        # ── 1. HSV Color Histogram (128D) ──────────────────────────────
        # WHY H+S only: Value (brightness) varies with lighting/shadows.
        # H+S captures the actual color regardless of illumination.
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist(
            [hsv], [0, 1], None,
            [HIST_H_BINS, HIST_S_BINS],
            [0, 180, 0, 256],
        )
        cv2.normalize(hist, hist, norm_type=cv2.NORM_L1)
        features.append(hist.flatten())

        # ── 2. Spatial Color Layout (12D) ───────────────────────────────
        # WHY LAB: Perceptually uniform — distances in LAB correlate with
        # human-perceived color differences better than RGB or HSV.
        # WHY 4 quadrants: Captures spatial arrangement (jersey vs shorts,
        # left vs right side of body).
        lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB).astype(np.float32)
        h, w = lab.shape[:2]
        mid_h, mid_w = h // 2, w // 2

        quadrants = [
            lab[:mid_h, :mid_w],    # upper-left
            lab[:mid_h, mid_w:],    # upper-right
            lab[mid_h:, :mid_w],    # lower-left
            lab[mid_h:, mid_w:],    # lower-right
        ]
        spatial = []
        for q in quadrants:
            if q.size > 0:
                spatial.extend(q.mean(axis=(0, 1)) / 255.0)
            else:
                spatial.extend([0.5, 0.5, 0.5])
        features.append(np.array(spatial, dtype=np.float32))

        # ── 3. Body Proportions (2D) ───────────────────────────────────
        # WHY: Players have different builds. Even in the same jersey,
        # a tall thin player differs from a short stocky one.
        # Aspect ratio of the crop + normalized area give a rough shape.
        aspect = (crop.shape[1] / crop.shape[0]) if crop.shape[0] > 0 else 1.0
        area = bbox_w * bbox_h if bbox_w > 0 and bbox_h > 0 else 0.01
        features.append(np.array([aspect, area * 10.0], dtype=np.float32))

        # Concatenate and L2-normalize
        feat = np.concatenate(features).astype(np.float32)
        norm = np.linalg.norm(feat)
        if norm > 0:
            feat /= norm
        return feat

    def extract_from_frame(self, frame: np.ndarray,
                           bbox_x: float, bbox_y: float,
                           bbox_w: float, bbox_h: float) -> Optional[np.ndarray]:
        """Convenience: extract features directly from frame + bbox."""
        crop = self.extract_torso(frame, bbox_x, bbox_y, bbox_w, bbox_h)
        if crop is None:
            return None
        return self.extract_features(crop, bbox_w, bbox_h)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two feature vectors.

    WHY cosine: It measures angle between vectors, ignoring magnitude.
    This makes it invariant to overall brightness changes that might
    scale the feature vector. Values range from -1 (opposite) to 1 (identical).
    For our L2-normalized features, this equals the dot product.
    """
    if a is None or b is None:
        return 0.0
    return float(np.dot(a, b))


def feature_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Distance between two feature vectors (0 = identical, 1 = very different).

    Converts cosine similarity to a distance metric for use in cost matrices.
    """
    return 1.0 - max(0.0, cosine_similarity(a, b))


class TrackGallery:
    """Manages appearance feature galleries for multiple tracks.

    DESIGN DECISION: Gallery-based approach
        Instead of keeping only the latest feature per track, we maintain
        a gallery of features sampled over time. The track descriptor is
        the MEDIAN of gallery samples.

        WHY median (not mean): Robust to outliers. If a player is partially
        occluded in one frame, that bad sample won't corrupt the descriptor.

        WHY gallery (not single feature): Player appearance varies with:
        - Pose changes (facing camera vs. turned away)
        - Partial occlusions (another player in front)
        - Lighting changes (moving across the field)
        The gallery captures this variation and the median finds the
        "most typical" appearance.
    """

    def __init__(self, max_size: int = MAX_GALLERY_SIZE):
        self.max_size = max_size
        self._galleries: dict[int, list[np.ndarray]] = {}
        self._descriptors: dict[int, Optional[np.ndarray]] = {}
        self._frame_counters: dict[int, int] = {}  # for sampling interval

    def update(self, track_id: int, feature: Optional[np.ndarray],
               force: bool = False) -> None:
        """Add a feature to a track's gallery.

        Samples every GALLERY_SAMPLE_INTERVAL frames to avoid redundancy.
        Set force=True to bypass the sampling interval.
        """
        if feature is None:
            return

        # Sampling: only store every N-th frame's features
        counter = self._frame_counters.get(track_id, 0) + 1
        self._frame_counters[track_id] = counter
        if not force and counter % GALLERY_SAMPLE_INTERVAL != 0:
            return

        if track_id not in self._galleries:
            self._galleries[track_id] = []

        gallery = self._galleries[track_id]
        gallery.append(feature)

        # Keep gallery bounded
        if len(gallery) > self.max_size:
            # Remove oldest samples (keep most recent)
            self._galleries[track_id] = gallery[-self.max_size:]

        # Invalidate cached descriptor
        self._descriptors[track_id] = None

    def get_descriptor(self, track_id: int) -> Optional[np.ndarray]:
        """Get the aggregated descriptor for a track.

        Returns the element-wise median of all gallery features.
        Cached until new features are added.
        """
        if track_id not in self._galleries or not self._galleries[track_id]:
            return None

        if self._descriptors.get(track_id) is not None:
            return self._descriptors[track_id]

        gallery = self._galleries[track_id]
        if len(gallery) == 1:
            desc = gallery[0].copy()
        else:
            stacked = np.stack(gallery)
            desc = np.median(stacked, axis=0).astype(np.float32)
            # Re-normalize after median
            norm = np.linalg.norm(desc)
            if norm > 0:
                desc /= norm

        self._descriptors[track_id] = desc
        return desc

    def match_to_gallery(self, query_feature: np.ndarray,
                         candidate_ids: Optional[list[int]] = None,
                         threshold: float = 0.55) -> list[tuple[int, float]]:
        """Find tracks whose appearance matches the query feature.

        Args:
            query_feature: Feature vector to match
            candidate_ids: Restrict search to these track IDs (None = all)
            threshold: Minimum cosine similarity to consider a match

        Returns:
            List of (track_id, similarity) sorted by similarity descending
        """
        if query_feature is None:
            return []

        ids_to_check = candidate_ids or list(self._galleries.keys())
        results = []

        for tid in ids_to_check:
            desc = self.get_descriptor(tid)
            if desc is None:
                continue
            sim = cosine_similarity(query_feature, desc)
            if sim >= threshold:
                results.append((tid, sim))

        results.sort(key=lambda x: x[1], reverse=True)
        return results

    def get_cross_similarity_matrix(self, track_ids: list[int]) -> np.ndarray:
        """Compute pairwise similarity matrix between tracks.

        Useful for track merging decisions — if two non-overlapping tracks
        have high similarity, they likely belong to the same player.
        """
        n = len(track_ids)
        sim_matrix = np.zeros((n, n), dtype=np.float32)

        descs = [self.get_descriptor(tid) for tid in track_ids]

        for i in range(n):
            for j in range(i + 1, n):
                if descs[i] is not None and descs[j] is not None:
                    s = cosine_similarity(descs[i], descs[j])
                    sim_matrix[i, j] = s
                    sim_matrix[j, i] = s
            sim_matrix[i, i] = 1.0

        return sim_matrix

    @property
    def track_ids(self) -> list[int]:
        return list(self._galleries.keys())

    def gallery_size(self, track_id: int) -> int:
        return len(self._galleries.get(track_id, []))

    def clear(self, track_id: Optional[int] = None) -> None:
        """Clear gallery for a specific track or all tracks."""
        if track_id is not None:
            self._galleries.pop(track_id, None)
            self._descriptors.pop(track_id, None)
            self._frame_counters.pop(track_id, None)
        else:
            self._galleries.clear()
            self._descriptors.clear()
            self._frame_counters.clear()
