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
        return (
            np.asarray(self.lower, dtype=np.uint8),
            np.asarray(self.upper, dtype=np.uint8),
        )


def unpack_rgb_float(rgb_packed: np.ndarray) -> np.ndarray:
    """Decode PointCloud2 packed-float rgb into Nx3 BGR uint8.

    Faster version: avoids np.stack allocation by preallocating output.
    """
    rgb_packed = np.ascontiguousarray(rgb_packed, dtype=np.float32)
    raw = rgb_packed.view(np.uint32)

    bgr = np.empty((raw.shape[0], 3), dtype=np.uint8)
    bgr[:, 0] = raw & 0xFF
    bgr[:, 1] = (raw >> 8) & 0xFF
    bgr[:, 2] = (raw >> 16) & 0xFF
    return bgr


def hsv_mask_from_bgr(bgr: np.ndarray, ranges: Sequence[HSVRange]) -> np.ndarray:
    """Vectorized HSV mask. Faster version using cv2.inRange."""
    n = bgr.shape[0]
    if n == 0:
        return np.zeros(0, dtype=bool)

    # OpenCV accepts Nx1x3 or 1xNx3. Nx1x3 is natural for point lists.
    hsv = cv2.cvtColor(bgr.reshape(-1, 1, 3), cv2.COLOR_BGR2HSV)

    if len(ranges) == 1:
        lo, hi = ranges[0].as_arrays()
        return cv2.inRange(hsv, lo, hi).reshape(-1).astype(bool)

    combined = np.zeros((n,), dtype=np.uint8)
    for r in ranges:
        lo, hi = r.as_arrays()
        combined |= cv2.inRange(hsv, lo, hi).reshape(-1)

    return combined.astype(bool)


def hsv_mask_from_packed_rgb(
    rgb_packed: np.ndarray,
    ranges: Sequence[HSVRange],
) -> np.ndarray:
    """Convenience: unpack PointCloud2 rgb floats then HSV-filter."""
    bgr = unpack_rgb_float(rgb_packed)
    return hsv_mask_from_bgr(bgr, ranges)