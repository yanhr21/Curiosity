# Phase 00 Tactile Semantic Reference Audit

Date: 2026-07-01

This audit records the official 2026 tactile references that can address the
current Gate 00D/00E blockers:

- validated gel/marker photometric semantics comparable to the user reference
  video;
- validated deformation-marker tracking on the pad surface;
- validated real contact-area semantics beyond point-contact density proxy;
- validated channel-level semantic equivalence beyond layout matching.

This audit is lightweight source/document inspection only. No simulation,
rendering, training, dependency installation, model loading, dataset conversion,
or Python-heavy validation was run on the login node.

## Local Official Checkouts

### UniVTAC

- Official repository: `https://github.com/univtac/UniVTAC`
- Local path: `external/UniVTAC`
- Local commit: `05bcd3edb92237107efa40105292a24f1a9fd761`
- Branch: `main`
- Role: official visuo-tactile manipulation benchmark and policy reference.
- Platform: Isaac Lab plus TacEx/UIPC tactile simulation.
- Sensors named by upstream: GelSight Mini, ViTai GF225, XenseWS. Current
  README notes collection/evaluation support is presently GelSight Mini first.
- Task families named by upstream: contact collection, lift bottle, lift can,
  insert HDMI, insert hole, insert tube, pull out key, put bottle in shelf,
  grasp and classify.
- Data schema value for this project:
  - synchronized head/wrist RGB;
  - left/right tactile `rgb`, `rgb_marker`, `depth`, `marker`, and tactile
    sensor pose;
  - robot joint/end-effector state and action-compatible deployment interface.
- Policy value for this project:
  - ACT baseline with tactile-full and vision-only configs;
  - ablation configs for modality comparison;
  - ViTAL-style CLIP-pretrained visuo-tactile encoder path.
- Use in Phase 00:
  - official reference for what a dense tactile dataset should contain;
  - official reference for future vision+tactile, tactile-only, vision-only,
    and ablation policy gates;
  - possible source of task/baseline comparison after separate official sanity.
- Not allowed conclusion:
  UniVTAC is not yet the Newton-native infant checkpoint and is not a current
  closed Gate 00D/00E proof. It needs compute-side official sanity before any
  compatibility or baseline claim.

### TaCauchy

- Official repository: `https://github.com/figsama/TaCauchy`
- Local path: `external/TaCauchy`
- Local commit: `c228cfe9050904cd5d71d64f6eb5104768d4cbda`
- Branch: `main`
- Role: official FEM/vision-based tactile simulation reference.
- Platform: Isaac Sim 5.0, Isaac Lab 2.2.1, UIPC/libuipc, TacEx.
- Sensors named by upstream: GelSight Mini, DIGIT, 9DTact.
- Core tactile semantics named by upstream:
  - direct Cauchy stress extraction;
  - normal pressure;
  - tangential traction;
  - adaptive mesh refinement;
  - force-field visualization;
  - tactile RGB image viewer.
- Use in Phase 00:
  - semantic reference for pressure, traction, stress, and contact-area fields;
  - target for replacing candidate point-contact area proxy with a validated
    surface/mesh-area interpretation;
  - target for validating whether our candidate `Fn`/`Ft`/marker maps preserve
    physically meaningful pressure and tangential traction structure.
- Not allowed conclusion:
  TaCauchy is not a grasp policy checkpoint. It is not Newton-native. It should
  validate tactile semantics and comparison channels after compute-side sanity,
  not replace the current Newton base by silent downgrade.

## Web-Only / Not Yet Locally Validated References

### Tacmap

- Public paper path found: `https://arxiv.org/abs/2602.21625`
- Local official code: not found during this audit.
- Role: geometry-consistent tactile mapping reference, especially penetration
  or deform-map semantics.
- Current status: comparison gap until official code/config is found.

### ControlTac

- Project path found: `https://dongyuluo.github.io/controltac/`
- Paper path found: `https://arxiv.org/abs/2505.20498`
- Local official code: not found during this audit.
- Role: tactile image generation/control reference conditioned by contact,
  force, or pose.
- Current status: comparison gap until official code/config is found.

## Secondary Available References

### TACTO

- Official repository path checked by remote query:
  `https://github.com/facebookresearch/tacto`
- Observed remote HEAD: `a21d0c4626d74a546d94859226c2fea348babb6a`
- Role: older optical tactile simulator reference.
- Current status: secondary fallback only; do not prefer it over UniVTAC,
  TaCauchy, Taccel, or Newton/Taccel mainline without a blocker.

### FreeTacMan

- Official repository path checked by remote query:
  `https://github.com/OpenDriveLab/FreeTacMan`
- Local path: `external/FreeTacMan`
- Local commit: `9285740a5d33385d3a9cf5ccdb185e3387b547bd`
- Role: real visuo-tactile data/pretraining reference.
- Relevant upstream paths:
  `pretrain/` for tactile encoder pretraining and `policy/act/` for ACT-style
  tactile-image policy training.
- Current status: secondary reference for real-data representation and future
  tactile pretraining/baseline design, not a simulator replacement and not a
  Newton infant checkpoint.

### DiffTactile

- Official repository path checked by remote query:
  `https://github.com/Genesis-Embodied-AI/DiffTactile`
- Observed remote HEAD: `c4bf43d44071758aea68a5c7ae125fc8257bb8e1`
- Role: differentiable tactile simulator/reference for soft tactile physics
  and tactile optimization tasks.
- Current status: secondary remote-only reference. It was not cloned because
  mandatory Gate 00F environment work for UniVTAC/TaCauchy remains the priority.

## Phase 00 Integration Decision

The active Phase 00 mainline remains Newton latest main plus direct MJWarp
contact-force export, because it already provides:

- official Newton runtime above 82 FPS;
- base grasp/lift evidence;
- candidate direct `Fn` and `Ft`;
- steel-spec material override evidence;
- synchronized scene/tactile/mechanics video;
- candidate gel/marker-style rendering;
- channel layout audit against the user reference video.

UniVTAC and TaCauchy now become mandatory semantic-validation references before
curiosity restarts:

1. Run official UniVTAC and TaCauchy sanity only inside a Curiosity-owned
   tmux-held Slurm allocation with prebuilt environments.
2. Extract their official tactile observation/channel schemas into an adapter
   spec without renaming candidate/proxy fields into official keys.
3. Build a Phase 00 semantic validation target:
   Newton candidate channels must be compared against official tactile-channel
   definitions for RGB marker, depth/deformation, normal pressure, tangential
   traction, contact area, and time-series force mechanics.
4. Gate 00D/00E stay open until this comparison either passes or records a
   faithful blocker.

## Immediate Next Actions

1. Prepare compute-side sanity launchers for UniVTAC and TaCauchy that only
   activate prebuilt local environments.
2. If no suitable prebuilt Isaac/TacEx/UIPC environment exists, record that as
   an environment blocker instead of installing on compute nodes.
3. Write the tactile schema bridge:
   - `univtac.tactile.left.rgb`
   - `univtac.tactile.left.rgb_marker`
   - `univtac.tactile.left.depth`
   - `univtac.tactile.left.marker`
   - `tacauchy.normal_pressure`
   - `tacauchy.tangential_traction`
   - `tacauchy.cauchy_stress`
   - `candidate.newton_mjw.Fn`
   - `candidate.newton_mjw.Ft`
   - `candidate.newton_mjw.marker_flow`
   - `candidate.newton_mjw.area_proxy`
4. Extend Gate 00D/00E review so passing current candidate visuals is not
   enough; it must also pass or explicitly block against the official semantic
   references above.
