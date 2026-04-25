"""Per-point HSV color filter for RealSense colored point clouds.

Pure NumPy/OpenCV — no ROS imports. Used by `ball_detector.py` to gate
points by color before centroid computation, and exercised standalone by
the tests in `test/test_hsv_filter.py`.

The RealSense `/camera/camera/depth/color/points` topic publishes
`sensor_msgs/PointCloud2` with an `rgb` field that is a 32-bit float whose
underlying bytes are 0x00 RR GG BB. We unpack it without leaving NumPy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import cv2
import numpy as np


@dataclass
class HSVRange:
    """Inclusive HSV bounds. OpenCV convention: H in [0, 179], S/V in [0, 255]."""
    lower: Sequence[int]
    upper: Sequence[int]

    def as_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        return (np.array(self.lower, dtype=np.uint8),
                np.array(self.upper, dtype=np.uint8))


def unpack_rgb_float(rgb_packed: np.ndarray) -> np.ndarray:
    """Decode the PointCloud2 packed-float `rgb` field into an Nx3 uint8 array.

    Each scalar's raw 4 bytes are 0x00 RR GG BB (PCL convention). Output is
    BGR-ordered to match OpenCV — i.e. column 0 is blue, 1 is green, 2 is red.
    """
    rgb_packed = np.ascontiguousarray(rgb_packed, dtype=np.float32)
    raw = rgb_packed.view(np.uint32)
    b = (raw & 0xFF).astype(np.uint8)
    g = ((raw >> 8) & 0xFF).astype(np.uint8)
    r = ((raw >> 16) & 0xFF).astype(np.uint8)
    return np.stack([b, g, r], axis=1)


def hsv_mask_from_bgr(bgr: np.ndarray, ranges: Sequence[HSVRange]) -> np.ndarray:
    """Vectorized point-wise HSV mask. `bgr` is Nx3 uint8.

    Returns a boolean array of length N: True where the point's color falls
    within ANY of the supplied HSV ranges (OR'd, for red wraparound).
    """
    if bgr.size == 0:
        return np.zeros(0, dtype=bool)

    # cv2.cvtColor expects HxWx3 — reshape to a 1xN strip so the call works
    # without copying every channel.
    strip = bgr.reshape(1, -1, 3)
    hsv = cv2.cvtColor(strip, cv2.COLOR_BGR2HSV).reshape(-1, 3)

    combined = np.zeros(hsv.shape[0], dtype=bool)
    for r in ranges:
        lo, hi = r.as_arrays()
        in_range = np.all((hsv >= lo) & (hsv <= hi), axis=1)
        combined |= in_range
    return combined


def hsv_mask_from_packed_rgb(rgb_packed: np.ndarray,
                             ranges: Sequence[HSVRange]) -> np.ndarray:
    """Convenience: unpack PointCloud2 rgb floats then HSV-filter."""
    bgr = unpack_rgb_float(rgb_packed)
    return hsv_mask_from_bgr(bgr, ranges)
