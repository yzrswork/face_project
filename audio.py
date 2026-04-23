"""Audio analyzer abstraction with deterministic stub fallback."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass


@dataclass
class AudioState:
    low: float = 0.0
    mid: float = 0.0
    high: float = 0.0
    rms: float = 0.0
    kick: bool = False


class AudioInputStub:
    """Low-cost deterministic pseudo audio for testing/embedded parity."""

    def __init__(self) -> None:
        self._t0 = time.perf_counter()
        self._prev_low = 0.0

    def update(self) -> AudioState:
        t = time.perf_counter() - self._t0
        low = 0.5 + 0.5 * math.sin(t * 2.2)
        mid = 0.5 + 0.5 * math.sin(t * 1.3 + 1.2)
        high = 0.5 + 0.5 * math.sin(t * 3.7 + 0.6)

        # deterministic pulse every ~0.46s.
        pulse = 1.0 if int(t * 2.2) != int((t - 1.0 / 120.0) * 2.2) else 0.0
        low = max(low, pulse * 0.95)

        kick = (low - self._prev_low) > 0.18 and low > 0.62
        self._prev_low = low
        return AudioState(low=low, mid=mid, high=high, rms=(low + mid + high) / 3.0, kick=kick)
