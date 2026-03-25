"""
Trajectory smoothing and Kalman filtering for player tracking.

WHY smoothing is necessary:
    Raw tracker output contains jitter from:
    1. Detection noise — YOLO bbox coordinates fluctuate ±2-5px per frame
    2. Tracker association errors — brief mis-associations cause position jumps
    3. Interpolation artifacts — linear interpolation between sampled frames
       creates angular paths instead of smooth curves

    Smoothing reduces these artifacts while preserving true player motion.

DESIGN DECISIONS:

    1. Dual approach: Online Kalman + Offline Savitzky-Golay
       - Kalman Filter runs during tracking for real-time prediction
         (needed for tracker association — "where will this player be next?")
       - Savitzky-Golay runs post-tracking on the full trajectory
         (produces cleaner paths for visualization and event detection)

    2. Kalman state = [x, y, vx, vy, ax, ay]
       Using a constant-acceleration model (not constant-velocity) because
       players accelerate/decelerate frequently in futsal.

    3. Savitzky-Golay with window=7, poly=2
       Small window preserves sharp direction changes (important for dribbles).
       Quadratic polynomial fits curves better than linear for player paths.

    4. Outlier detection before smoothing
       A single bad detection can corrupt the smoothed trajectory.
       We detect outliers (position jumps > 3σ) and interpolate over them
       before applying the smoother.
"""

import numpy as np
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class KalmanPointTracker:
    """2D Kalman filter for tracking a single point (player center).

    State vector: [x, y, vx, vy, ax, ay]
    Measurement: [x, y]

    WHY constant-acceleration model:
        Players in futsal frequently accelerate and decelerate.
        A constant-velocity model (state = [x, y, vx, vy]) introduces lag
        when players speed up or slow down. The acceleration terms allow
        the filter to anticipate speed changes.

    WHY custom implementation (not cv2.KalmanFilter):
        - More readable and debuggable
        - Easy to tune process/measurement noise
        - No OpenCV dependency for this specific feature
        - Can add custom logic (like velocity clamping)
    """

    def __init__(self, dt: float = 1.0 / 30.0,
                 process_noise: float = 0.01,
                 measurement_noise: float = 0.005):
        """
        Args:
            dt: Time step between frames (1/fps)
            process_noise: How much we trust the motion model.
                Higher = more responsive to changes but noisier.
            measurement_noise: How much we trust detections.
                Higher = smoother but slower to react.
        """
        self.dt = dt

        # State: [x, y, vx, vy, ax, ay]
        self.x = np.zeros(6, dtype=np.float64)

        # State transition matrix (constant acceleration)
        self.F = np.eye(6, dtype=np.float64)
        self.F[0, 2] = dt        # x += vx * dt
        self.F[1, 3] = dt        # y += vy * dt
        self.F[0, 4] = 0.5 * dt * dt  # x += 0.5 * ax * dt²
        self.F[1, 5] = 0.5 * dt * dt  # y += 0.5 * ay * dt²
        self.F[2, 4] = dt        # vx += ax * dt
        self.F[3, 5] = dt        # vy += ay * dt

        # Measurement matrix (we only observe x, y)
        self.H = np.zeros((2, 6), dtype=np.float64)
        self.H[0, 0] = 1.0
        self.H[1, 1] = 1.0

        # Process noise covariance
        self.Q = np.eye(6, dtype=np.float64) * process_noise
        # Position has less process noise than velocity/acceleration
        self.Q[0, 0] *= 0.1
        self.Q[1, 1] *= 0.1

        # Measurement noise covariance
        self.R = np.eye(2, dtype=np.float64) * measurement_noise

        # State covariance (high initial uncertainty)
        self.P = np.eye(6, dtype=np.float64) * 1.0

        self.initialized = False

    def initialize(self, x: float, y: float) -> None:
        """Initialize state with first measurement."""
        self.x = np.array([x, y, 0, 0, 0, 0], dtype=np.float64)
        self.P = np.eye(6, dtype=np.float64) * 0.1
        self.initialized = True

    def predict(self) -> tuple[float, float]:
        """Predict next state (call before update).

        Returns predicted (x, y) position.
        """
        if not self.initialized:
            return 0.0, 0.0

        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

        # Clamp velocity to reasonable range (players can't teleport)
        # Max speed ~10 m/s on a 40m court → ~0.25 normalized units/s
        max_v = 0.25 * self.dt
        self.x[2] = np.clip(self.x[2], -max_v, max_v)
        self.x[3] = np.clip(self.x[3], -max_v, max_v)

        return float(self.x[0]), float(self.x[1])

    def update(self, x: float, y: float) -> tuple[float, float]:
        """Update state with new measurement.

        Returns corrected (x, y) position.
        """
        if not self.initialized:
            self.initialize(x, y)
            return x, y

        z = np.array([x, y], dtype=np.float64)

        # Innovation (measurement residual)
        y_innov = z - self.H @ self.x

        # Innovation covariance
        S = self.H @ self.P @ self.H.T + self.R

        # Kalman gain
        K = self.P @ self.H.T @ np.linalg.inv(S)

        # State update
        self.x = self.x + K @ y_innov

        # Covariance update (Joseph form for numerical stability)
        I_KH = np.eye(6) - K @ self.H
        self.P = I_KH @ self.P @ I_KH.T + K @ self.R @ K.T

        return float(self.x[0]), float(self.x[1])

    @property
    def position(self) -> tuple[float, float]:
        return float(self.x[0]), float(self.x[1])

    @property
    def velocity(self) -> tuple[float, float]:
        return float(self.x[2]), float(self.x[3])

    @property
    def speed(self) -> float:
        return float(np.hypot(self.x[2], self.x[3]))


class TrajectorySmoother:
    """Post-processing smoother for complete trajectories.

    Applied AFTER tracking is complete (offline) to produce clean paths
    for visualization and event detection.

    DESIGN DECISIONS:

        1. Outlier removal first, then smoothing.
           A single bad detection can drag the smoothed curve off course.
           We detect outliers using the median absolute deviation (MAD)
           which is more robust than standard deviation.

        2. Savitzky-Golay filter (not moving average):
           - Preserves sharp features (direction changes) better than MA
           - Fits a polynomial to each window → follows curves naturally
           - No phase shift (unlike exponential moving average)

        3. Separate X and Y smoothing:
           Player motion in X and Y are independent. Smoothing them
           separately is correct and simpler.
    """

    def __init__(self, window_size: int = 7, poly_order: int = 2):
        """
        Args:
            window_size: Savitzky-Golay window (must be odd). Larger = smoother
                but loses fast direction changes. 7 is good for 30fps tracking.
            poly_order: Polynomial order. 2 (quadratic) fits player motion curves.
        """
        if window_size % 2 == 0:
            window_size += 1
        self.window_size = window_size
        self.poly_order = poly_order

    def smooth_trajectory(self, positions: list[tuple[float, float]]
                          ) -> list[tuple[float, float]]:
        """Smooth a full trajectory.

        Args:
            positions: List of (x, y) tuples in chronological order

        Returns:
            Smoothed positions (same length as input)
        """
        if len(positions) < self.window_size:
            return list(positions)  # Too short to smooth

        xs = np.array([p[0] for p in positions], dtype=np.float64)
        ys = np.array([p[1] for p in positions], dtype=np.float64)

        # Step 1: Detect and interpolate outliers
        xs = self._fix_outliers(xs)
        ys = self._fix_outliers(ys)

        # Step 2: Apply Savitzky-Golay filter
        xs_smooth = self._savgol(xs)
        ys_smooth = self._savgol(ys)

        return list(zip(xs_smooth.tolist(), ys_smooth.tolist()))

    def _fix_outliers(self, values: np.ndarray, threshold: float = 3.0
                      ) -> np.ndarray:
        """Detect and interpolate over outliers using MAD.

        WHY MAD (Median Absolute Deviation):
            Standard deviation is sensitive to outliers — the very thing
            we're trying to detect. MAD uses the median, making it robust.
            A value is an outlier if |value - median| > threshold * MAD.

        Outliers are replaced by linear interpolation from neighbors.
        """
        if len(values) < 5:
            return values

        # Compute frame-to-frame differences
        diffs = np.diff(values)
        median_diff = np.median(np.abs(diffs))
        mad = np.median(np.abs(diffs - np.median(diffs)))

        if mad < 1e-10:
            return values  # No variation → no outliers

        result = values.copy()
        for i in range(1, len(values)):
            if abs(diffs[i - 1]) > median_diff + threshold * mad:
                # Outlier detected — interpolate
                # Find next non-outlier
                j = i + 1
                while j < len(values) - 1:
                    if j < len(diffs) and abs(diffs[j - 1]) <= median_diff + threshold * mad:
                        break
                    j += 1

                if j < len(values):
                    # Linear interpolation between i-1 and j
                    for k in range(i, min(j, len(values))):
                        alpha = (k - (i - 1)) / (j - (i - 1))
                        result[k] = result[i - 1] + alpha * (values[j] - result[i - 1])

        return result

    def _savgol(self, values: np.ndarray) -> np.ndarray:
        """Apply Savitzky-Golay filter.

        Implemented without scipy dependency for portability.
        Uses least-squares polynomial fitting in a sliding window.
        """
        try:
            from scipy.signal import savgol_filter
            return savgol_filter(values, self.window_size, self.poly_order)
        except ImportError:
            pass

        # Fallback: simple weighted moving average
        # Less elegant than SavGol but still smooths effectively
        n = len(values)
        result = values.copy()
        half = self.window_size // 2

        for i in range(half, n - half):
            window = values[i - half:i + half + 1]
            # Triangle weights (center has most weight)
            weights = np.array([1 + half - abs(j - half)
                                for j in range(len(window))], dtype=np.float64)
            weights /= weights.sum()
            result[i] = np.dot(window, weights)

        return result

    def smooth_bboxes(self, bboxes: list[dict]) -> list[dict]:
        """Smooth bounding box trajectories for a single track.

        Args:
            bboxes: List of dicts with keys: bbox_x, bbox_y, bbox_w, bbox_h, timestamp_seconds

        Returns:
            Same list with smoothed bbox coordinates
        """
        if len(bboxes) < self.window_size:
            return bboxes

        # Extract center positions
        positions = [
            (b["bbox_x"] + b["bbox_w"] / 2,
             b["bbox_y"] + b["bbox_h"] / 2)
            for b in bboxes
        ]

        smoothed = self.smooth_trajectory(positions)

        # Apply smoothed centers back to bboxes
        result = []
        for i, b in enumerate(bboxes):
            new_b = dict(b)
            cx, cy = smoothed[i]
            new_b["bbox_x"] = cx - b["bbox_w"] / 2
            new_b["bbox_y"] = cy - b["bbox_h"] / 2
            result.append(new_b)

        return result
