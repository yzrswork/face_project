"""Main modular runtime entrypoint."""

from __future__ import annotations

import time
from pathlib import Path

import pygame

from animator import Animator
from audio import AudioInputStub
from config import DEFAULT_CONFIG
from effects import BoidController, TrailBuffer
from model import FaceModel
from renderer import PygameRenderer, QualitySettings


def select_quality(frame_dt: float, audio_high: float) -> QualitySettings:
    # Dynamic quality scaling: deterministic thresholds and cheap branching.
    if frame_dt > 0.020:
        return QualitySettings(edge_stride=2, trail_stride=2, draw_trails=False)
    if frame_dt > 0.017:
        return QualitySettings(edge_stride=2, trail_stride=2, draw_trails=True)
    if audio_high > 0.65:
        return QualitySettings(edge_stride=1, trail_stride=2, draw_trails=True)
    return QualitySettings(edge_stride=1, trail_stride=1, draw_trails=True)


def run() -> None:
    cfg = DEFAULT_CONFIG
    fixed_dt = 1.0 / cfg.sim_hz

    root = Path(__file__).resolve().parent
    json_path = root / "face_wireframe.json"

    pygame.init()
    screen = pygame.display.set_mode((cfg.width, cfg.height))
    pygame.display.set_caption("face_project modular app")
    clock = pygame.time.Clock()

    model = FaceModel(json_path, cfg.width, cfg.height, cfg.fit_ratio)
    animator = Animator(model.base, model.center)
    audio = AudioInputStub()
    boids = BoidController(model.count)
    trails = TrailBuffer(cfg.trail_len, model.count)
    renderer = PygameRenderer(screen, model.edges, cfg.width, cfg.height)

    running = True
    accumulator = 0.0
    last_t = time.perf_counter()
    tick = 0
    quality = QualitySettings()

    while running:
        now = time.perf_counter()
        frame_dt = min(now - last_t, cfg.max_frame_dt)
        last_t = now
        accumulator += frame_dt
        quality = select_quality(frame_dt, getattr(audio, "_prev_low", 0.0))

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False

        sim_steps = 0
        while accumulator >= fixed_dt and sim_steps < cfg.max_sim_steps_per_frame:
            astate = audio.update()
            anim_out = animator.step(astate, tick)
            if anim_out.use_boids:
                boids.apply(
                    model.pos,
                    model.vel,
                    neighbor_r=cfg.boid_neighbor_r,
                    sep_r=cfg.boid_sep_r,
                    sep_force=cfg.boid_sep_force,
                    align_force=cfg.boid_align_force,
                    coh_force=cfg.boid_coh_force,
                    dt=fixed_dt,
                    audio_drive=astate.low,
                )

            model.integrate(fixed_dt, anim_out.target, cfg.spring_stiffness, cfg.spring_damping)
            model.clamp_speed(cfg.max_speed)

            if tick % anim_out.trail_rate == 0:
                trails.push(model.pos)

            accumulator -= fixed_dt
            sim_steps += 1
            tick += 1

        alpha = accumulator / fixed_dt
        renderer.draw(model.prev_pos, model.pos, alpha, trails, quality, cfg.wire_color, cfg.bg_color)
        pygame.display.flip()
        clock.tick(cfg.fps_target)

    pygame.quit()


if __name__ == "__main__":
    run()
