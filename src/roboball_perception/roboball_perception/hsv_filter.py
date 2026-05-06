"""Tiny HSV helper for filtering colored point clouds.

RealSense `PointCloud2` uses a packed float `rgb` field (PCL style):
0x00 RR GG BB. We unpack that and run an HSV inRange.
"""

from dataclasses import dataclass
from typing import Sequence

import cv2
import numpy as np


@dataclass
class HSVRange:
    lower: Sequence[int]
    upper: Sequence[int]

    def as_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        return np.array(self.lower, dtype=np.uint8), np.array(self.upper, dtype=np.uint8)


def unpack_rgb_float(rgb_packed: np.ndarray) -> np.ndarray:
    """Turn packed float rgb into Nx3 uint8 BGR (OpenCV order)."""
    rgb_packed = np.ascontiguousarray(rgb_packed, dtype=np.float32)
    raw = rgb_packed.view(np.uint32)
    b = (raw & 0xFF).astype(np.uint8)
    g = ((raw >> 8) & 0xFF).astype(np.uint8)
    r = ((raw >> 16) & 0xFF).astype(np.uint8)
    return np.stack([b, g, r], axis=1)


def hsv_mask_from_bgr(bgr: np.ndarray, ranges: Sequence[HSVRange]) -> np.ndarray:
    if bgr.size == 0:
        return np.zeros(0, dtype=bool)

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
