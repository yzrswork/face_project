"""Runtime configuration shared across modules."""

from dataclasses import dataclass


@dataclass(frozen=True)
class RuntimeConfig:
    width: int = 1280
    height: int = 720
    fps_target: int = 60
    sim_hz: int = 120
    max_frame_dt: float = 0.1
    max_sim_steps_per_frame: int = 4

    fit_ratio: float = 0.65
    wire_color: tuple[int, int, int] = (20, 235, 110)
    bg_color: tuple[int, int, int] = (0, 0, 0)

    boid_neighbor_r: float = 90.0
    boid_sep_r: float = 28.0
    boid_sep_force: float = 45.0
    boid_align_force: float = 8.0
    boid_coh_force: float = 7.5
    max_speed: float = 110.0

    spring_stiffness: float = 14.0
    spring_damping: float = 0.93

    trail_len: int = 10

    low_band_gain: float = 1.5
    high_band_gain: float = 0.8


DEFAULT_CONFIG = RuntimeConfig()
