"""
Multi-player tracking pipeline for five-a-side football video analysis.

Architecture overview:
    ┌─────────┐   ┌──────────┐   ┌──────┐   ┌──────────┐   ┌─────────────┐
    │ Detector │──▶│ Tracker  │──▶│ ReID │──▶│ Smoother │──▶│ FieldMapper │
    │ (YOLO)  │   │(ByteTrack│   │      │   │ (Kalman+ │   │(Homography) │
    │         │   │/DeepSort)│   │      │   │  SavGol) │   │             │
    └─────────┘   └──────────┘   └──────┘   └──────────┘   └─────────────┘

Design decisions:
    1. YOLO v8s for detection — best speed/accuracy tradeoff for real-time
    2. ByteTrack/DeepOCSORT for tracking — handles occlusions via Kalman prediction
    3. Multi-feature ReID — HSV histogram + spatial color + body proportions
       for robust re-identification without heavy NN dependencies
    4. Dual smoothing — Kalman for online, Savitzky-Golay for offline post-processing
    5. Homography field mapping — 4-point transform for bird's-eye view
"""

from .reid import AppearanceExtractor, TrackGallery
from .smoother import TrajectorySmoother, KalmanPointTracker
from .field_mapping import FieldMapper
from .postprocess import TrackPostProcessor

__all__ = [
    "AppearanceExtractor",
    "TrackGallery",
    "TrajectorySmoother",
    "KalmanPointTracker",
    "FieldMapper",
    "TrackPostProcessor",
]
