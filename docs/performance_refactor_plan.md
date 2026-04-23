# Face Project Architecture and Performance Refactor Plan

## 1) Current architecture

The current runtime (`matrix_face_v13_001.py`) is a **single-file, stateful loop** with three major responsibilities mixed together:

1. **Model loading / geometry generation**
   - Loads `face_wireframe.json`, normalizes and scales points, generates alternate forms (`HELIX`, `SPHERE`, `VORTEX`).
2. **Simulation / animation state updates**
   - Reads audio FFT + RMS, updates form transitions, boids, singularity, trails, shockwaves, and silence logic.
3. **Rendering**
   - Draws multiple full-screen layers each frame (`bg`, `wire`, `fx`, `crt`, `abr`, `phosphor`) and composits everything.

This architecture works, but it strongly couples timing, simulation, and drawing, making optimization and testing difficult.

---

## 2) Inefficient parts (high impact first)

### A. O(N²) boid update in Python loop
`boid_update()` iterates each point and recomputes distances against all points using masks every frame. That is expensive and spikes CPU as point count grows.

**Improve:**
- Use a spatial grid / hash to limit neighbor checks.
- Or use a periodic boid update (e.g., every 2–3 frames) with interpolation.

### B. Per-vertex/per-edge draw calls in Python
The renderer performs many `pygame.draw.line` and `pygame.draw.circle` calls in nested loops across multiple effects (wire, trails, echoes, arcs, dots).

**Improve:**
- Cache integer screen coordinates once per frame.
- Use effect quality scaling (skip every other edge/dot under load).
- Draw static overlays less frequently (CRT scanline layer can be redrawn every 2–4 frames).

### C. Multiple full-screen clears and alpha compositing each frame
Several surfaces are fully cleared and blended each frame, even if content changes minimally.

**Improve:**
- Keep static layer cache (scanline, border) and only rebuild when size/theme changes.
- Reduce number of full-screen alpha layers.
- Render expensive post effects conditionally based on audio energy or frame budget.

### D. FFT pipeline does repeated work and allocations
`read_fft()` creates arrays every frame (`frombuffer`, type cast, optional padding, abs rfft output). This is normal but can be tightened.

**Improve:**
- Reuse pre-allocated numpy buffers where possible.
- Read audio in a background thread with a ring buffer; render thread just consumes latest analysis.

### E. Variable timestep simulation
Physics and transitions are tied to `clock.tick(FPS)` and frame cadence. If frame time spikes, motion and transitions become uneven.

**Improve:**
- Fixed simulation timestep (`dt = 1/120` or `1/90`) with accumulator.
- Render interpolation factor (`alpha`) for smooth visuals.

---

## 3) Better rendering loop (practical)

Use this loop structure:

```text
while running:
    frame_dt = timer.elapsed()
    accumulator += clamp(frame_dt, 0, max_frame)

    process_input()

    while accumulator >= fixed_dt:
        audio_analyzer.update()
        animator.step(fixed_dt, audio_state)
        model.integrate(fixed_dt, animator.output)
        accumulator -= fixed_dt

    alpha = accumulator / fixed_dt
    renderer.draw(model, alpha)
    present()
```

### Why this helps
- **Smoother animation:** simulation becomes deterministic, no large jumps.
- **Lower CPU:** you can cap simulation steps/frame and dynamically reduce effect complexity.
- **Cleaner architecture:** clear separation between input, simulation, and drawing.

---

## 4) JSON format improvements

Current JSON is essentially:

```json
{
  "vertices": [[x, y], ...],
  "edges": [[i, j], ...]
}
```

Recommended v2 format:

```json
{
  "version": 2,
  "units": "normalized",
  "topology": {
    "vertices": [[x, y], ...],
    "edges": [[i, j], ...],
    "regions": {
      "left_eye": [indices...],
      "right_eye": [indices...],
      "mouth": [indices...]
    }
  },
  "metadata": {
    "author": "...",
    "source": "..."
  },
  "constraints": {
    "pinned_vertices": [indices...],
    "edge_groups": {
      "jaw": [edge_indices...],
      "brow": [edge_indices...]
    }
  },
  "animation_hints": {
    "default_morph_order": ["FACE", "HELIX", "SPHERE", "VORTEX"],
    "pulse_region": "face_outline"
  }
}
```

### Why this helps
- Stable schema versioning for future tools.
- Regions/groups unlock targeted effects (blink, lip sync, per-region stiffness).
- Constraints allow controlled deformation without hardcoding index lists in Python.

---

## 5) Modularization proposal

Split into modules:

- `model.py`
  - Geometry and dynamic state (`positions`, `velocities`, target forms).
- `animation.py`
  - Form transitions, singularity state machine, silence logic, boid controller.
- `audio.py`
  - Stream capture + FFT analysis + smoothing.
- `renderer.py`
  - Layer composition, quality scaling, draw primitives.
- `app.py`
  - Main fixed-step loop and orchestration.

Core interfaces:

- `FaceModel.step(dt, forces, target_positions)`
- `Animator.step(dt, audio_state, model) -> AnimationOutput`
- `Renderer.draw(model, interp_alpha, debug_state)`

This design makes it easier to:
- benchmark each subsystem independently,
- disable effects for profiling,
- test animation logic without opening a Pygame window.

---

## 6) Improved example code

See `examples/modular_face_runtime.py` for a practical baseline that demonstrates:

- fixed timestep simulation,
- `FaceModel` / `Animator` / `Renderer` separation,
- cached integer points for edge rendering,
- simple quality scaler hook.

Use it as a migration starting point rather than a full visual replacement.
