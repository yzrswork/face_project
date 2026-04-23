# Portable Realtime Architecture (Desktop -> RP2040/RP2350)

## 1. Subsystem analysis of `matrix_face_v13_001.py`

Major subsystems currently mixed in one file:

- **Audio FFT analysis**
  - `read_fft()` and the per-frame FFT + smoothing pipeline in main loop.
- **Silence detection**
  - `update_silence()` with RMS median gating + silence frame counters.
- **Boids behavior**
  - `boid_update()` with separation/alignment/cohesion over all vertices.
- **Singularity/Vortex/Form transitions**
  - `update_forms()` handles form FSM, singularity contract/explode/reform, pulse.
- **Trails / Echoes**
  - `update_trail()`, `draw_trails()`, `draw_echoes()`.
- **Shockwaves**
  - `update_draw_shockwaves()` and related state lists.
- **Rendering layers**
  - multiple layers (`layer_wire`, `layer_fx`, `layer_crt`, `phosphor`) composited every frame.

The monolithic loop increases coupling and makes backend portability difficult.

## 2. Module mapping

Recommended structure (implemented scaffold):

- `config.py`
  - constants and tunables; deterministic timing/quality knobs.
- `audio.py`
  - `AudioState` + `AudioInputStub`; later swap with PyAudio backend.
- `model.py`
  - geometry loading from unchanged `face_wireframe.json`, position/velocity integration.
- `animator.py`
  - animation FSM: form switching, pulse targets, silence behavior.
- `effects.py`
  - boid force and trail ring buffer.
- `renderer.py`
  - renderer interface and pygame backend.
- `embedded_renderer.py`
  - future embedded backend contract stub.
- `app.py`
  - fixed timestep loop, quality scaling, module orchestration.

## 3. Performance policies applied

- Fixed simulation timestep with accumulator and max simulation steps/frame.
- Reused numpy buffers (`FaceModel` scratch arrays, boid pairwise arrays, trail ring buffer).
- Cached integer screen coordinates in renderer before edge draws.
- Static CRT layer prebuilt once.
- Quality scaler reduces edge/trail density and can disable heavy effects.

## 4. Pygame -> Embedded portability strategy

Animation logic is backend-independent:

- `Animator`, `FaceModel`, and `BoidController` do not depend on pygame.
- Renderer only consumes interpolated points + edges and quality settings.
- For RP2040/RP2350:
  - replace `PygameRenderer` with `EmbeddedRenderer` implementation,
  - replace `AudioInputStub` with ADC/I2S envelope input,
  - keep simulation/fsm modules mostly unchanged.

This isolates platform-specific code to I/O and rasterization layers.

## 5. Notes for MicroPython/C ports

- Use fixed-point math where FPU throughput is limited.
- Reduce vertex count / edge stride by quality profile.
- Keep ring-buffer trails short and optional.
- Run animation at a fixed low Hz (e.g., 60) and decouple scanout.
