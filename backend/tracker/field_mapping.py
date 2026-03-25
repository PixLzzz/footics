"""
Homography-based field mapping for bird's-eye view projection.

WHY field mapping:
    Camera perspective distorts spatial relationships:
    - Players far from camera appear smaller and closer together
    - Distances between players are not proportional in pixel space
    - Tactical analysis requires true field positions

    A homography transform maps the perspective view to a top-down view
    where distances are proportional to real-world distances.

DESIGN DECISIONS:

    1. 4-point homography (not camera calibration):
       Full camera calibration requires intrinsic parameters (focal length,
       distortion coefficients) which we don't have. A 4-point homography
       only needs 4 corresponding points (court corners in video ↔ court
       corners in the real world). Simpler and sufficient for flat-field sports.

    2. Standard futsal court dimensions (40m × 20m):
       The output coordinate system uses meters, so player positions,
       distances, and speeds are in real-world units. Makes event detection
       thresholds (e.g., "pass distance > 5m") more meaningful.

    3. User-provided corner points:
       Automatic court line detection is unreliable from behind-the-goal
       camera angles. Instead, the user marks 4 court corners once per match.
       The homography is stored with the match metadata.

    4. Bidirectional transform:
       pixel→field for analytics, field→pixel for visualization overlays.

COORDINATE SYSTEMS:
    - Pixel (normalized): (0, 0) = top-left, (1, 1) = bottom-right of video
    - Field (meters): (0, 0) = top-left corner of court, (40, 20) = bottom-right
"""

import cv2
import numpy as np
import json
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Standard futsal court dimensions (meters)
FUTSAL_LENGTH = 40.0
FUTSAL_WIDTH = 20.0


class FieldMapper:
    """Maps between pixel coordinates and bird's-eye field coordinates.

    Usage:
        # User clicks 4 court corners in the video (normalized coords)
        corners = [(0.1, 0.3), (0.9, 0.3), (0.05, 0.95), (0.95, 0.95)]
        mapper = FieldMapper(corners)

        # Transform player position to field coords
        field_x, field_y = mapper.pixel_to_field(0.5, 0.6)
        # → (20.0, 12.5) approximately

        # Compute real-world distance between two players
        dist = mapper.real_distance(player1_x, player1_y, player2_x, player2_y)
        # → 8.3 (meters)
    """

    def __init__(self, corner_points: list[tuple[float, float]],
                 field_length: float = FUTSAL_LENGTH,
                 field_width: float = FUTSAL_WIDTH):
        """Initialize with 4 court corner points in normalized pixel coords.

        Args:
            corner_points: 4 tuples of (x, y) in normalized [0, 1] coords.
                Order: [top-left, top-right, bottom-left, bottom-right]
                as seen in the video frame.
            field_length: Court length in meters (default: 40m for futsal)
            field_width: Court width in meters (default: 20m for futsal)
        """
        if len(corner_points) != 4:
            raise ValueError("Exactly 4 corner points required")

        self.field_length = field_length
        self.field_width = field_width
        self.corner_points = corner_points

        # Source points (pixel, normalized)
        src = np.array(corner_points, dtype=np.float32)

        # Destination points (field coordinates in meters)
        # Map to: top-left → (0, 0), top-right → (length, 0),
        #         bottom-left → (0, width), bottom-right → (length, width)
        dst = np.array([
            [0, 0],
            [field_length, 0],
            [0, field_width],
            [field_length, field_width],
        ], dtype=np.float32)

        # Compute homography matrices
        self.H, _ = cv2.findHomography(src, dst)
        self.H_inv, _ = cv2.findHomography(dst, src)

        if self.H is None or self.H_inv is None:
            raise ValueError("Failed to compute homography — check corner points")

        logger.info(
            f"FieldMapper initialized: {field_length}×{field_width}m, "
            f"corners={corner_points}"
        )

    def pixel_to_field(self, x: float, y: float) -> tuple[float, float]:
        """Transform normalized pixel coordinates to field coordinates (meters).

        Args:
            x, y: Normalized pixel coordinates [0, 1]

        Returns:
            (field_x, field_y) in meters from top-left corner of court
        """
        pt = np.array([[[x, y]]], dtype=np.float32)
        transformed = cv2.perspectiveTransform(pt, self.H)
        fx, fy = transformed[0, 0]
        return float(fx), float(fy)

    def field_to_pixel(self, fx: float, fy: float) -> tuple[float, float]:
        """Transform field coordinates (meters) back to normalized pixel coords.

        Args:
            fx, fy: Field coordinates in meters

        Returns:
            (x, y) normalized pixel coordinates [0, 1]
        """
        pt = np.array([[[fx, fy]]], dtype=np.float32)
        transformed = cv2.perspectiveTransform(pt, self.H_inv)
        x, y = transformed[0, 0]
        return float(x), float(y)

    def real_distance(self, x1: float, y1: float,
                      x2: float, y2: float) -> float:
        """Compute real-world distance (meters) between two pixel positions.

        This is much more accurate than Euclidean distance in pixel space
        because it accounts for perspective distortion.
        """
        fx1, fy1 = self.pixel_to_field(x1, y1)
        fx2, fy2 = self.pixel_to_field(x2, y2)
        return float(np.hypot(fx2 - fx1, fy2 - fy1))

    def real_speed(self, x1: float, y1: float,
                   x2: float, y2: float, dt: float) -> float:
        """Compute real-world speed (m/s) between two positions.

        Args:
            x1, y1: Start position (normalized pixels)
            x2, y2: End position (normalized pixels)
            dt: Time delta in seconds

        Returns:
            Speed in meters per second
        """
        if dt <= 0:
            return 0.0
        return self.real_distance(x1, y1, x2, y2) / dt

    def transform_positions(self, positions: list[dict]) -> list[dict]:
        """Transform a batch of positions from pixel to field coords.

        Args:
            positions: List of dicts with 'cx', 'cy' keys (normalized pixel)

        Returns:
            Same list with added 'field_x', 'field_y' keys (meters)
        """
        result = []
        for pos in positions:
            fx, fy = self.pixel_to_field(pos["cx"], pos["cy"])
            new_pos = dict(pos)
            new_pos["field_x"] = round(fx, 2)
            new_pos["field_y"] = round(fy, 2)
            result.append(new_pos)
        return result

    def to_json(self) -> str:
        """Serialize to JSON for storage in database."""
        return json.dumps({
            "corners": self.corner_points,
            "field_length": self.field_length,
            "field_width": self.field_width,
        })

    @classmethod
    def from_json(cls, data: str) -> "FieldMapper":
        """Deserialize from JSON."""
        d = json.loads(data) if isinstance(data, str) else data
        corners = [tuple(c) for c in d["corners"]]
        return cls(
            corner_points=corners,
            field_length=d.get("field_length", FUTSAL_LENGTH),
            field_width=d.get("field_width", FUTSAL_WIDTH),
        )

    def generate_minimap(self, player_positions: list[dict],
                         width: int = 400, height: int = 200,
                         ball_pos: Optional[tuple[float, float]] = None
                         ) -> np.ndarray:
        """Generate a bird's-eye view minimap image.

        Args:
            player_positions: List of dicts with cx, cy, track_id, and
                optionally player_name, team_label
            width, height: Output image dimensions in pixels
            ball_pos: Optional (x, y) normalized ball position

        Returns:
            BGR image of the minimap
        """
        # Green field background
        img = np.full((height, width, 3), (34, 139, 34), dtype=np.uint8)

        # Scale factors: field meters → image pixels
        sx = (width - 20) / self.field_length
        sy = (height - 20) / self.field_width
        ox, oy = 10, 10  # offset for border

        # Draw court lines
        self._draw_court_lines(img, sx, sy, ox, oy)

        # Draw players
        for p in player_positions:
            fx, fy = self.pixel_to_field(p["cx"], p["cy"])

            # Clamp to field bounds
            fx = max(0, min(self.field_length, fx))
            fy = max(0, min(self.field_width, fy))

            px = int(ox + fx * sx)
            py = int(oy + fy * sy)

            # Color by team
            team = p.get("team_label", -1)
            color = (255, 100, 100) if team == 0 else (100, 100, 255) if team == 1 else (200, 200, 200)

            cv2.circle(img, (px, py), 6, color, -1)
            cv2.circle(img, (px, py), 6, (255, 255, 255), 1)

            # Label
            label = p.get("player_name", str(p.get("track_id", "?")))
            if len(label) > 6:
                label = label[:6]
            cv2.putText(img, label, (px - 10, py - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3,
                        (255, 255, 255), 1, cv2.LINE_AA)

        # Draw ball
        if ball_pos:
            bfx, bfy = self.pixel_to_field(ball_pos[0], ball_pos[1])
            bfx = max(0, min(self.field_length, bfx))
            bfy = max(0, min(self.field_width, bfy))
            bpx = int(ox + bfx * sx)
            bpy = int(oy + bfy * sy)
            cv2.circle(img, (bpx, bpy), 4, (0, 255, 255), -1)

        return img

    def _draw_court_lines(self, img, sx, sy, ox, oy):
        """Draw standard futsal court markings."""
        white = (255, 255, 255)
        fl, fw = self.field_length, self.field_width

        def pt(x, y):
            return (int(ox + x * sx), int(oy + y * sy))

        # Outer boundary
        cv2.rectangle(img, pt(0, 0), pt(fl, fw), white, 1)

        # Center line
        cv2.line(img, pt(fl / 2, 0), pt(fl / 2, fw), white, 1)

        # Center circle (radius ~3m)
        center = pt(fl / 2, fw / 2)
        radius = int(3.0 * min(sx, sy))
        cv2.circle(img, center, radius, white, 1)

        # Goal areas (6m × 3m from goal line for futsal)
        for gx in [0, fl]:
            sign = 1 if gx == 0 else -1
            y1 = fw / 2 - 3
            y2 = fw / 2 + 3
            dx = 6 * sign
            cv2.rectangle(img, pt(gx, y1), pt(gx + dx, y2), white, 1)
