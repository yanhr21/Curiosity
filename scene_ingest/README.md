# scene_ingest — dataset → Newton physics + tactile

Lift **pre-arranged scene datasets** (SAGE-10k first; then REST3D, RoboCasa, BEHAVIOR-1K, and
our own `genpipe` GLBs) into Newton physics + tactile simulation environments. Design:
`../claude_context/dataset_ingestion.md`.

```
dataset layout ──[adapter]──▶ SceneSpec (IR) ──[newton_build]──▶ newton.ModelBuilder + sensors
```

- `spec.py` — dataset-agnostic intermediate representation (`SceneSpec`, `ObjectSpec`, `RoomSpec`, …).
- `adapters/sage.py` — SAGE-10k `layout_*.json` → `SceneSpec` (pure stdlib; no `newton` needed).
- `newton_build.py` — `SceneSpec` → `newton.ModelBuilder` via the verified clock recipe
  (convex-hull collision + hydroelastic SDF + full visual, material from PBR + mass, static walls).

## Try it

Parse a SAGE scene (no `newton` env needed):
```bash
python -m scene_ingest.adapters.sage \
  $MY_DATA_HOME/robot_baby_data/_inspect/layout_84b703fb.json
```

Build a Newton model (needs the `newton` conda env):
```bash
conda activate newton
python -m scene_ingest.newton_build <scene>/layout_<id>.json --build
```

## Status

`spec.py` + `adapters/sage.py` are runnable stdlib. `newton_build.py` is a **reference
implementation not yet run in the live `newton` env** — the mesh up-axis, bbox-fit mode, SDF
resolution, robot pad setup, and sensor wiring are calibration knobs. For a real tactile run,
fork `example_panda_clock_metal.py` / `tactile_video.py` (the verified panda + hydroelastic-pad +
`SensorContact` stack) and drive it from a `SceneSpec`.

## Add a dataset

Write `adapters/<name>.py` with `load_<name>_scene(path, ...) -> SceneSpec`. Only the adapter is
dataset-specific; `newton_build.py` is shared.
