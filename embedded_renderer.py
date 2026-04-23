"""Placeholder embedded backend API for RP2040/RP2350 ports.

This module mirrors the renderer interface so animation/model code can be reused.
"""

from __future__ import annotations


class EmbeddedRenderer:
    def draw(self, *args, **kwargs) -> None:
        # Implement using MicroPython frame buffer or C scanline rasterizer.
        # Same call contract as renderer.PygameRenderer.draw.
        raise NotImplementedError
