"""High-level animation state machine independent of renderer backend."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from audio import AudioState

FORM_FACE = "face"
FORM_VORTEX = "vortex"
FORM_SINGULARITY = "singularity"


@dataclass
class AnimatorOutput:
    target: np.ndarray
    use_boids: bool
    trail_rate: int


class Animator:
    def __init__(self, base: np.ndarray, center: np.ndarray) -> None:
        self.base = base
        self.center = center
        self.target = base.copy()

        n = base.shape[0]
        t = np.linspace(0.0, 1.0, n, dtype=np.float32)
        ang = t * (7.0 * np.pi)
        rad = (1.0 - t) * 220.0 + 12.0
        self.vortex = np.empty_like(base)
        self.vortex[:, 0] = center[0] + np.cos(ang) * rad
        self.vortex[:, 1] = center[1] + np.sin(ang) * rad

        self.form = FORM_FACE
        self.silence_frames = 0
        self.silent = False

    def step(self, audio: AudioState, tick: int) -> AnimatorOutput:
        # silence detection (portable, no audio backend dependency)
        if audio.rms < 0.12:
            self.silence_frames += 1
            if self.silence_frames > 180:
                self.silent = True
                self.form = FORM_FACE
        else:
            self.silence_frames = 0
            self.silent = False

        # Simple deterministic form switching.
        if not self.silent and audio.kick and audio.low > 0.75:
            if self.form == FORM_FACE:
                self.form = FORM_VORTEX
            elif self.form == FORM_VORTEX:
                self.form = FORM_SINGULARITY
            else:
                self.form = FORM_FACE

        pulse = 1.0 + 0.08 * audio.low
        if self.form == FORM_FACE:
            self.target[:] = self.center + (self.base - self.center) * pulse
            use_boids = audio.low > 0.45
        elif self.form == FORM_VORTEX:
            wobble = 1.0 + 0.05 * np.sin(tick * 0.03)
            self.target[:] = self.center + (self.vortex - self.center) * wobble
            use_boids = True
        else:  # singularity
            self.target[:] = self.center
            use_boids = False

        trail_rate = 1 if audio.high > 0.35 else 2
        return AnimatorOutput(target=self.target, use_boids=use_boids, trail_rate=trail_rate)
