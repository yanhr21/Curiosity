# Phase 00 Newton Asset Generation References

Date: 2026-06-29

This note records the source-backed constraints for Phase 00 asset generation.
It is not simulation evidence and not a training result.

## Official Newton Points Used

1. Newton is GPU-accelerated, targets robotics simulation, supports OpenUSD,
   differentiability, and user extensibility. The project README also lists
   CUDA-capable NVIDIA GPUs as a requirement for GPU use.

   Source: https://github.com/newton-physics/newton

2. Newton 1.3.0 documentation exposes user-guide sections for visualization,
   worlds, mass/inertia, sensors, USD parsing, and collisions/contact. These
   are the relevant official surfaces for asset generation validation.

   Source: https://newton-physics.github.io/newton/stable/

3. Newton supports multiple independent worlds in one model. The Worlds guide
   recommends homogeneous multi-world workflows with replication and warns
   that visual separation should use viewer-level world offsets instead of
   moving physics worlds far from the origin.

   Source: https://newton-physics.github.io/newton/stable/concepts/worlds.html

4. Newton visualization supports common viewer methods, headless ViewerGL
   frame capture, ViewerFile offline recording, ViewerUSD time-sampled USD
   export, debug overlays such as contacts, and image logging including tiled
   camera outputs. Phase 00 validation must therefore produce real videos,
   contact sheets/frame browsers, and contact overlays/traces from compute
   runs instead of static design-only SVGs.

   Source: https://newton-physics.github.io/newton/stable/guide/visualization.html

5. Newton sensors include SensorContact and SensorTiledCamera. SensorContact
   requires contact force arrays and SensorTiledCamera is the current camera
   path for color/depth rendering. Phase 00 must validate contact/tactile
   proxies and camera outputs together.

   Source: https://newton-physics.github.io/newton/stable/concepts/sensors.html

6. Newton USD parsing documents Newton collision schema behavior, including
   SDF collision and hydroelastic contact attributes. For hydroelastic contact,
   Newton-specific attributes must be authored on Newton-ready assets; PhysX
   attributes alone are not enough.

   Source: https://newton-physics.github.io/newton/stable/concepts/usd_parsing.html

7. Newton mass/inertia guidance says dynamic bodies should have positive mass,
   density-based inference is preferred when possible, and detailed inertia
   validation should be enabled during development. Phase 00 must therefore
   report mass/inertia warnings and treat invalid mass/inertia as an asset
   blocker.

   Source: https://newton-physics.github.io/newton/stable/concepts/mass_inertia.html

8. Newton contact documentation exposes rigid contact count and contact data,
   and the Isaac Sim Newton backend warns that MuJoCo Warp has a contact
   buffer limit (`nconmax`) that can drop excess contacts. Phase 00 must record
   contact counts and fail if logs show contact-limit overflow.

   Sources:
   - https://newton-physics.github.io/newton/stable/concepts/collisions.html
   - https://docs.isaacsim.omniverse.nvidia.com/6.0.1/physics/newton_physics.html

9. The Newton URDF-to-USD converter describes OpenUSD conversion of visual
   geometry, materials, collision geometry, and joints, producing standalone
   OpenUSD artifacts suitable for Newton import/simulation. It recommends
   confirming USD visual geometry and physics properties in USD tooling.

   Source: https://github.com/newton-physics/urdf-usd-converter

10. Isaac Lab asset-import guidance notes that Omniverse uses USD for assets,
    importers convert URDF/MJCF/mesh assets into USD, and instanceable assets
    matter for large-scale simulation. Phase 00 should avoid ad-hoc asset
    hacks and keep generated/converted assets inspectable and reusable.

    Source: https://isaac-sim.github.io/IsaacLab/main/source/how-to/import_new_asset.html

11. Newton examples expose `--num-frames` as the direct control for total
    simulated frames, and the official README examples show this value being
    increased for longer runs. Phase 00 therefore must not rely on a short
    smoke-test horizon when the goal is asset generation and validation.

    Source: https://github.com/newton-physics/newton

12. The Isaac Sim Newton backend preserves compatibility with standard USD
    Physics schemas while using Newton as the active simulation backend, so
    generated assets and validation evidence should remain USD/physics-schema
    inspectable rather than ad-hoc one-off data.

    Source: https://docs.isaacsim.omniverse.nvidia.com/6.0.1/physics/newton_physics.html

## Project Consequence

Phase 00 cannot be considered generated or verified after only a few static
files. A valid Phase 00 asset-generation attempt must run in a Curiosity-owned
tmux-held H200 allocation and produce real Newton outputs:

- fresh official Newton sanity;
- per-cell camera output;
- contact/contact-proxy traces;
- frame browser;
- contact sheet;
- full rollout video or dense-frame equivalent;
- mass/inertia/collision/contact-limit checks;
- manual visual inspection;
- split/no-leakage audit.

The full Phase 00 H200 generation profile must use at least 1800 simulated
frames per catalog cell by default, with long-hold validation and dense rollout
video evidence. Shorter runs may only be recorded as explicitly requested
diagnostics and must not be promoted to full asset generation evidence.

Login-node work is limited to documentation, config editing, shell syntax
checks, JSON validation, and light job/session inspection.
