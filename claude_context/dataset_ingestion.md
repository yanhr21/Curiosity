# Dataset → Newton Ingestion Framework (physics + tactile)

Design for turning **pre-arranged scene datasets** (SAGE-10k first, then REST3D, RoboCasa,
BEHAVIOR-1K, and our own `genpipe` GLBs) into **Newton physics + tactile simulation
environments**. Reference implementation: `scene_ingest/` (repo root). Grounded in the verified
asset→Newton recipe from `example_panda_clock_metal.py` (see `context.md` §7, `tactile_pipeline.md` §7).

---

## 1. Why

`genpipe/` (PixelDiT→TRELLIS.2) generates **one object at a time**. The Robot-Baby *Explore* stage
needs whole **human-arranged, contact-rich scenes** with physics and touch. Datasets like **SAGE-10k**
already ship 10k arranged rooms / 565k objects with **mass, PBR, and placement** baked in — far cheaper
than generating scenes from scratch. We want a single path that lifts *any* such dataset into a Newton
`ModelBuilder` + tactile/vision sensors, with a **physical-enrichment** layer on top.

**Two asset sources, one sink:** `genpipe` (generative, per-object) and this ingestion path (datasets,
whole-scene) both feed the same Newton scene builder + tactile stack.

## 2. Architecture — adapter → SceneSpec → Newton

```
SAGE-10k  ─┐
REST3D    ─┤   per-dataset          dataset-agnostic          one Newton builder
RoboCasa  ─┼──▶  ADAPTER    ──────▶   SceneSpec (IR)   ──────▶  build_newton_scene()
BEHAVIOR  ─┤   (parse layout)        objects+room+robot        + tactile/vision sensors
genpipe   ─┘                         +sensors+randomization    + solvers (MuJoCo/VBD)
```

- **Adapter** (`scene_ingest/adapters/<name>.py`): parses a dataset's native layout into `SceneSpec`.
  The *only* dataset-specific code. SAGE reads `layout_*.json` + `objects/*.ply`.
- **SceneSpec** (`scene_ingest/spec.py`): a small, physics-first intermediate representation. Decouples
  "what the scene is" from "how Newton builds it".
- **Builder** (`scene_ingest/newton_build.py`): `SceneSpec → newton.ModelBuilder`, dataset-agnostic.
  Encodes the one verified recipe (convex-hull collision + hydroelastic SDF + full-mesh visual, material
  from PBR + mass, static room, articulated doors), plus robot + sensor attachment and enrichment.

## 3. SceneSpec (intermediate representation)

Physics-first, minimal, SI units, Z-up (Newton convention).

- **`ObjectSpec`** — one placeable body:
  - `mesh_path` (PLY/GLB/OBJ) · `bbox_dims (w,l,h) [m]` (target physical size) · `up_axis` (source frame)
  - `position (x,y,z) [m]` · `rotation` (quat or Euler-Z°) — pose in room frame, floor at z=0
  - `mass [kg]` · `material` (see below) · `is_static` (fixed vs. dynamic) · `is_deformable` (→ FEM)
  - `semantic_type` (e.g. "mug") · `support_parent` (floor | wall | `<object_id>`) — spawn/settle order
- **`MaterialSpec`** — `friction (mu)`, `restitution`, `density [kg/m³]` (or `mass`→derive), `rigid_kh`
  (`1e12` for hard, compliant pad handled at the robot), `young_E`/`poisson_nu` when deformable.
- **`RoomSpec`** — `walls[]` (start/end/height/thickness → static extruded boxes), `floor` (ground
  plane or box), `doors[]` (→ revolute articulation on a wall), `bounds`.
- **`RobotSpec`** — urdf/asset + base pose + init joint config + which links are tactile pads.
- **`SensorSpec`** — tactile (`SensorContact` + hydroelastic surface on pad links), vision
  (`SensorTiledCamera`), proprio (`SensorIMU`).
- **`RandomizationSpec`** — per-field distributions for the enrichment layer (§7).

## 4. SAGE-10k adapter (the mapping)

SAGE scene = `layout_<id>.json` + `objects/<source_id>.ply` (+ `_texture.png`) + `materials/`.
Field mapping `layout.rooms[i]` → `SceneSpec`:

| SAGE field | SceneSpec | Notes |
|---|---|---|
| `dimensions {width,length,height}` | `RoomSpec.bounds` | metres, floor z=0 |
| `walls[] {start_point,end_point,height,thickness}` | `RoomSpec.walls` | static extruded boxes |
| `doors[] {wall_id, position_on_wall, width, opens_inward}` | `RoomSpec.doors` | → revolute joint (optional) |
| `objects[].source_id` | `ObjectSpec.mesh_path` = `objects/<source_id>.ply` | one PLY per object |
| `objects[].position {x,y,z}` | `ObjectSpec.position` | metres, room frame |
| `objects[].rotation {x,y,z}` (Euler°) | `ObjectSpec.rotation` | yaw=z; → quat |
| `objects[].dimensions {w,l,h}` | `ObjectSpec.bbox_dims` | **fit mesh bbox to this** (scale) |
| `objects[].mass` | `ObjectSpec.mass` | kg (0.25–60 seen) |
| `objects[].pbr_parameters {metallic,roughness}` | `MaterialSpec` | metallic→hard/rigid_kh; roughness→mu heuristic |
| `objects[].type` | `semantic_type` | "table","mug",… |
| `objects[].place_id` | `support_parent` | `floor` / `<object_id>` / `wall` → settle order |

**Calibration steps (need the live `newton` env to tune):**
1. **Up-axis:** SAGE layout is Z-up (floor z=0); confirm the raw `.ply` up-axis and whether the mesh is
   pre-centred or pre-posed. The `kits/export_glb.py` path applies the layout transform, so the PLY is a
   **local-frame** mesh → we apply `scale · R(rotation) · T(position)` ourselves. (GLB export is Y-up;
   raw PLY handling must be verified — start by fitting mesh AABB to `bbox_dims`.)
2. **Scale:** rescale mesh so its AABB matches `bbox_dims` (SAGE dims are physical); do **not** trust raw
   mesh units.
3. **Seat on support:** drop each object so its bottom sits ~one contact-margin above its `support_parent`
   surface (avoids frame-0 launch — see `context.md` §6 FEM-launch gotcha).

## 5. Newton builder (dataset-agnostic) — the verified recipe

Per `ObjectSpec` (mirrors `load_clock_geometry` + the clock `Example.__init__`):

```python
scene = trimesh.load(mesh_path); mesh = one_geometry(scene)
v = fit_bbox(rotate_to_Zup(mesh.vertices), bbox_dims)         # scale to physical size
hull = trimesh.Trimesh(v, faces).convex_hull                   # collision proxy
density = mass / hull.volume                                   # derive density from mass + volume
cfg = ModelBuilder.ShapeConfig(kh=material.rigid_kh, mu=material.mu,
                               restitution=material.restitution, density=density,
                               is_hydroelastic=True, gap=0.01)
b = builder.add_body(xform=transform(position, quat(rotation)))
builder.add_shape_mesh(body=b, mesh=newton.Mesh(hull_v, hull_f), cfg=cfg)   # collision (+visual full mesh)
# static objects (walls, heavy furniture): add as a fixed body / shape on world
```
Then, once for the scene:
- **Room:** walls → static boxes (`add_shape_box` on the world body); floor → `add_ground_plane` or a
  thin static box with the floor material; doors → a hinged panel body + revolute joint (optional; skip
  for a static room first).
- **Convex-hull + hydroelastic:** `builder.approximate_meshes(method="convex_hull",
  shape_indices=..., keep_visual_shapes=True)` for non-contact objects; for objects the robot will touch,
  build a mesh SDF (`mesh.build_sdf(max_resolution=64, narrow_band_range=(-0.01,0.01), margin=gap)`) and set
  `ShapeFlags.HYDROELASTIC` (dense tactile pressure needs the hydroelastic surface).
- **Deformables** (`is_deformable`): `add_soft_mesh` (tetrahedralized) + `SolverVBD` alongside the rigid
  solver, one model/state (see `tactile_pipeline.md` §4/§6 for the coupling recipe).

**Materials (verified reasoning, `context.md` §5):** all *rigid* materials share `rigid_kh≈1e12`; they
differ only by `mu` / `restitution` / `density`. Map `pbr.metallic`→hard vs. soft-ish `mu`, `roughness`→
`mu` heuristic; only genuinely deformable items become FEM (differentiated by Young's `E`).

## 6. Robot + tactile + vision attachment

Reuse the `panda_hydro` / clock stack verbatim (fork, don't rebuild — `context.md` §6):
- **Robot:** `builder.add_urdf(download_asset("franka_emika_panda")/…)` (or G1 for whole-body); set
  finger/hand shapes to `HYDROELASTIC` with a **compliant pad** (`pad_kh≈1e10`) for a broad contact patch.
- **Tactile:** two `SensorContact`s (pads←object for per-pad grip/shear; object←pads for shear-on-object,
  world frame) + `HydroelasticSDF.get_contact_surface()` for the **dense per-face pressure map**
  (`pressure = kh·max(0,−signed_depth)`). Per-step order: `solver.step → update_contacts → sensor.update →
  get_contact_surface` (`context.md` §4). Pad-centric heatmap window (fixed pad frame).
- **Vision:** `SensorTiledCamera` (RGB/depth) for the third-person / wrist view the Robot-Baby policy uses.
- **Proprio:** `SensorIMU`.

## 7. Physical enrichment (domain randomization) — the Robot-Baby differentiator

The point of SDG here is *not* to reproduce SAGE's exact scene but to **randomize the physics the demo
never revealed**. Per object, sample distributions over: **mass, center-of-mass offset, friction `mu`,
restitution, hydroelastic compliance, articulation resistance, attachment strength**, plus scene-level
lighting/materials. This is the `RandomizationSpec`; it's what makes the same intent require a *different*
solution under different physics (ties to the main-branch `context.md` "outcome, not trajectory" pitch and
`stack.md` enrichment layer).

## 8. Validation (simulation-ready gate)

Mirror SAGE's own stability check (`kits/load_isaacsim_demo.py` → stable/unstable ratio) using Newton:
settle the scene N steps under gravity, flag objects whose pose drifts > ε (fell / penetrated / launched),
and reuse this repo's **penetration metric** (`d = dot(n, bx_b−bx_a) − (m0+m1)`; pen = `max(0,−d)`,
`context.md` §3). Emit a per-scene report; drop or re-seat failing objects. Gate a scene as "sim-ready"
before it enters an RL curriculum.

## 9. Scaling

- **Cost:** ingest is cheap vs. generation — load PLY + convex hull ≈ tens of ms/object; hydroelastic SDF
  ≈ 0.3 s/object (only for objects the robot touches — build lazily). No PixelDiT/TRELLIS cost.
- **Storage:** consume SAGE in place (`$MY_DATA_HOME/robot_baby_data`, 870 GB). Cache built SDFs beside the
  meshes (~0.5 MB/object) or rebuild at load (0.3 s).
- **Fleet:** one scene → one Newton world; batch scenes across GPUs (1 stream/GPU, as in `genpipe/README.md`).
  Shard by scene id.

## 10. Roadmap (this framework)

1. `scene_ingest/spec.py` + `adapters/sage.py` + `newton_build.py` (reference scaffold — **done, untested
   in the live env**).
2. Bring up the `newton` conda env; run one SAGE room headless → USD + a settle/stability report.
3. Add a Franka + tactile pads; pick one object off a SAGE table and record a tactile video (reuse
   `tactile_video.py` composite).
4. Wire `RandomizationSpec`; sweep mass/friction/compliance on the picked object.
5. Second adapter (REST3D or a `genpipe` GLB scene) to prove the IR is dataset-agnostic.
6. Feed sim-ready scenes to the Robot-Baby *Explore* stage (main branch `2026_7_14_mike`).

## 11. How this fits the two branches

- **This branch (`mike_2026_7_21_newton`)** = the Newton engine fork + tactile stack + `genpipe` +
  **this ingestion framework** → the *environment factory*.
- **Main branch (`2026_7_14_mike`, Robot Baby)** = demonstration-conditioned, outcome-driven RL that
  *consumes* these environments (Explore/Learn). SAGE-10k, REST3D, and `genpipe` are complementary scene
  sources feeding the same Newton physics+tactile sink.
