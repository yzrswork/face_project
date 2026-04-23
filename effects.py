"""Simulation effects: boids and trails."""

from __future__ import annotations

import numpy as np


class BoidController:
    def __init__(self, count: int) -> None:
        self.count = count
        self._dx = np.zeros((count, count), dtype=np.float32)
        self._dy = np.zeros((count, count), dtype=np.float32)
        self._dist2 = np.zeros((count, count), dtype=np.float32)

    def apply(
        self,
        pos: np.ndarray,
        vel: np.ndarray,
        *,
        neighbor_r: float,
        sep_r: float,
        sep_force: float,
        align_force: float,
        coh_force: float,
        dt: float,
        audio_drive: float,
    ) -> None:
        # Dense vectorized neighbor computation (small-N friendly, deterministic).
        self._dx[:] = pos[:, 0][:, None] - pos[:, 0][None, :]
        self._dy[:] = pos[:, 1][:, None] - pos[:, 1][None, :]
        self._dist2[:] = self._dx * self._dx + self._dy * self._dy

        np.fill_diagonal(self._dist2, np.inf)

        neighbor_mask = self._dist2 < (neighbor_r * neighbor_r)
        sep_mask = self._dist2 < (sep_r * sep_r)

        # Separation force.
        inv_dist = 1.0 / (np.sqrt(self._dist2, dtype=np.float32) + 1e-4)
        sep_x = np.sum(self._dx * inv_dist * sep_mask, axis=1)
        sep_y = np.sum(self._dy * inv_dist * sep_mask, axis=1)

        # Alignment/cohesion averages.
        n_count = np.maximum(1, neighbor_mask.sum(axis=1))
        mean_vx = (neighbor_mask * vel[:, 0][None, :]).sum(axis=1) / n_count
        mean_vy = (neighbor_mask * vel[:, 1][None, :]).sum(axis=1) / n_count

        mean_px = (neighbor_mask * pos[:, 0][None, :]).sum(axis=1) / n_count
        mean_py = (neighbor_mask * pos[:, 1][None, :]).sum(axis=1) / n_count

        boost = 0.35 + audio_drive * 0.9
        vel[:, 0] += (sep_x * sep_force + (mean_vx - vel[:, 0]) * align_force + (mean_px - pos[:, 0]) * coh_force) * dt * boost
        vel[:, 1] += (sep_y * sep_force + (mean_vy - vel[:, 1]) * align_force + (mean_py - pos[:, 1]) * coh_force) * dt * boost


class TrailBuffer:
    def __init__(self, trail_len: int, point_count: int) -> None:
        self.length = trail_len
        self.data = np.zeros((trail_len, point_count, 2), dtype=np.float32)
        self.ptr = 0
        self.filled = 0

    def push(self, pos: np.ndarray) -> None:
        self.data[self.ptr, :, :] = pos
        self.ptr = (self.ptr + 1) % self.length
        self.filled = min(self.filled + 1, self.length)

    def iter_newest_first(self):
        for age in range(self.filled):
            idx = (self.ptr - 1 - age) % self.length
            yield age, self.data[idx]
