"""Rendering backend abstraction + pygame implementation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pygame

from effects import TrailBuffer


@dataclass
class QualitySettings:
    edge_stride: int = 1
    trail_stride: int = 1
    draw_trails: bool = True


class RendererBase:
    def draw(self, *args, **kwargs) -> None:  # pragma: no cover - interface only
        raise NotImplementedError


class PygameRenderer(RendererBase):
    def __init__(self, screen: pygame.Surface, edges: np.ndarray, width: int, height: int) -> None:
        self.screen = screen
        self.edges = edges
        self.width = width
        self.height = height

        self.layer_wire = pygame.Surface((width, height), pygame.SRCALPHA)
        self.layer_fx = pygame.Surface((width, height), pygame.SRCALPHA)
        self.layer_crt_static = pygame.Surface((width, height), pygame.SRCALPHA)
        self._build_static_crt()

        # Reused integer point buffer to avoid allocations.
        self._pts_i32 = np.zeros((edges.max() + 1, 2), dtype=np.int32)

    def _build_static_crt(self) -> None:
        self.layer_crt_static.fill((0, 0, 0, 0))
        for y in range(0, self.height, 2):
            pygame.draw.line(self.layer_crt_static, (0, 24, 8, 20), (0, y), (self.width, y), 1)

    def _draw_wire(self, pts: np.ndarray, color: tuple[int, int, int], stride: int) -> None:
        self.layer_wire.fill((0, 0, 0, 0))
        for i in range(0, len(self.edges), stride):
            a, b = self.edges[i]
            pygame.draw.line(self.layer_wire, color, pts[a], pts[b], 1)

    def _draw_trails(self, trails: TrailBuffer, stride: int) -> None:
        self.layer_fx.fill((0, 0, 0, 0))
        for age, p in trails.iter_newest_first():
            if age % stride:
                continue
            alpha = max(0, 90 - age * 10)
            if alpha <= 0:
                continue
            pi = p.astype(np.int32)
            for i in range(0, len(self.edges), 2):
                a, b = self.edges[i]
                pygame.draw.line(self.layer_fx, (30, 180, 90, alpha), pi[a], pi[b], 1)

    def draw(
        self,
        prev_pos: np.ndarray,
        pos: np.ndarray,
        alpha: float,
        trails: TrailBuffer,
        quality: QualitySettings,
        wire_color: tuple[int, int, int],
        bg_color: tuple[int, int, int],
    ) -> None:
        self.screen.fill(bg_color)
        interp = prev_pos + (pos - prev_pos) * alpha
        np.rint(interp, out=self._pts_i32)

        self._draw_wire(self._pts_i32, wire_color, quality.edge_stride)
        self.screen.blit(self.layer_wire, (0, 0))

        if quality.draw_trails:
            self._draw_trails(trails, quality.trail_stride)
            self.screen.blit(self.layer_fx, (0, 0))

        self.screen.blit(self.layer_crt_static, (0, 0))
