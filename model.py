"""Face geometry/state model with low-allocation integration."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class FaceModel:
    def __init__(self, json_path: Path, width: int, height: int, fit_ratio: float = 0.65) -> None:
        raw = json.loads(json_path.read_text(encoding="utf-8"))
        verts = np.asarray(raw["vertices"], dtype=np.float32)[:, :2]
        self.edges = np.asarray(raw["edges"], dtype=np.int32)

        src_min = verts.min(axis=0)
        src_max = verts.max(axis=0)
        span = src_max - src_min
        src_center = (src_min + src_max) * 0.5
        dst_center = np.array([width * 0.5, height * 0.5], dtype=np.float32)
        scale = min(width * fit_ratio / span[0], height * fit_ratio / span[1])

        self.base = np.empty_like(verts)
        self.base[:, 0] = dst_center[0] + (verts[:, 0] - src_center[0]) * scale
        self.base[:, 1] = dst_center[1] - (verts[:, 1] - src_center[1]) * scale

        self.count = self.base.shape[0]
        self.center = dst_center

        self.pos = self.base.copy()
        self.prev_pos = self.base.copy()
        self.vel = np.zeros_like(self.base)

        # Scratch buffers to avoid per-frame allocations.
        self._delta = np.zeros_like(self.base)
        self._speed = np.zeros(self.count, dtype=np.float32)

    def integrate(self, dt: float, target: np.ndarray, spring_stiffness: float, damping: float) -> None:
        self.prev_pos[:] = self.pos
        np.subtract(target, self.pos, out=self._delta)
        self.vel += self._delta * (spring_stiffness * dt)
        self.vel *= damping

        np.square(self.vel[:, 0], out=self._speed)
        self._speed += self.vel[:, 1] * self.vel[:, 1]
        np.sqrt(self._speed, out=self._speed)

        self.pos += self.vel * dt

    def clamp_speed(self, max_speed: float) -> None:
        sp = self._speed
        over = sp > max_speed
        if not np.any(over):
            return
        scale = max_speed / (sp[over] + 1e-6)
        self.vel[over, 0] *= scale
        self.vel[over, 1] *= scale
