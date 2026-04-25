"""
Strike objective policies for high-level bounce behavior.

These policies sit between trajectory prediction and execution:
  - Input: raw predicted impact point from `/strike_target`.
  - Output: adjusted impact point passed to IK / strike execution.

Current implementations are conservative stubs so the system remains safe:
they primarily clamp or lightly reshape xy targets. Future work can add
contact-angle and paddle-velocity shaping to actively steer rebounds.
"""

from dataclasses import dataclass
from typing import Literal

import numpy as np

ObjectiveMode = Literal[
    'intercept',
    'center_spot',
    'xy_zone',
    'human_rally',
    'circle',
]


@dataclass(frozen=True)
class ObjectiveConfig:
    mode: ObjectiveMode = 'intercept'
    center_xy: tuple[float, float] = (0.45, 0.0)
    zone_min_xy: tuple[float, float] = (0.20, -0.30)
    zone_max_xy: tuple[float, float] = (0.75, 0.30)
    human_xy: tuple[float, float] = (0.90, 0.0)
    circle_center_xy: tuple[float, float] = (0.45, 0.0)
    circle_radius: float = 0.20
    circle_speed_rad_s: float = 0.7
    blend_gain: float = 0.35


class StrikeObjectivePolicy:
    """Select and apply a high-level objective to a predicted impact point."""

    def __init__(self, config: ObjectiveConfig):
        self.config = config
        self._toggle_state = False

    def apply(self, impact_xyz: np.ndarray, now_sec: float) -> np.ndarray:
        """
        Return a new xyz impact target for the configured objective.

        NOTE: This is intentionally XY-only in the scaffold; z is preserved.
        """
        mode = self.config.mode
        if mode == 'intercept':
            return impact_xyz.copy()
        if mode == 'center_spot':
            return self._center_spot(impact_xyz)
        if mode == 'xy_zone':
            return self._xy_zone(impact_xyz)
        if mode == 'human_rally':
            return self._human_rally(impact_xyz)
        if mode == 'circle':
            return self._circle(impact_xyz, now_sec)
        return impact_xyz.copy()

    def _center_spot(self, impact_xyz: np.ndarray) -> np.ndarray:
        # Stub: nudge the target toward one desired xy location.
        target = impact_xyz.copy()
        center = np.array(self.config.center_xy, dtype=np.float64)
        target[:2] = (1.0 - self.config.blend_gain) * target[:2] + self.config.blend_gain * center
        return target

    def _xy_zone(self, impact_xyz: np.ndarray) -> np.ndarray:
        # Stub: clamp impact to stay inside a permitted rectangle.
        target = impact_xyz.copy()
        zone_min = np.array(self.config.zone_min_xy, dtype=np.float64)
        zone_max = np.array(self.config.zone_max_xy, dtype=np.float64)
        target[0] = np.clip(target[0], zone_min[0], zone_max[0])
        target[1] = np.clip(target[1], zone_min[1], zone_max[1])
        return target

    def _human_rally(self, impact_xyz: np.ndarray) -> np.ndarray:
        """
        Stub: alternate bias between robot-centered and human-side targets.

        This does not yet model contact dynamics, so it cannot guarantee a true
        back-and-forth rally. It scaffolds the mode switch and target shaping.
        """
        target = impact_xyz.copy()
        robot_side = np.array(self.config.center_xy, dtype=np.float64)
        human_side = np.array(self.config.human_xy, dtype=np.float64)
        desired = human_side if self._toggle_state else robot_side
        self._toggle_state = not self._toggle_state
        target[:2] = (1.0 - self.config.blend_gain) * target[:2] + self.config.blend_gain * desired
        return target

    def _circle(self, impact_xyz: np.ndarray, now_sec: float) -> np.ndarray:
        # Stub: move desired xy point around a time-varying circle.
        target = impact_xyz.copy()
        theta = self.config.circle_speed_rad_s * now_sec
        cx, cy = self.config.circle_center_xy
        desired = np.array(
            [cx + self.config.circle_radius * np.cos(theta),
             cy + self.config.circle_radius * np.sin(theta)],
            dtype=np.float64,
        )
        target[:2] = (1.0 - self.config.blend_gain) * target[:2] + self.config.blend_gain * desired
        return target
