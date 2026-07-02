# Latest 2026-07-01 Codebase Refresh

Date: 2026-07-01

Classification: web/source refresh only. This is not training, not official
sanity, not model loading, not simulation, and not Gate completion.

Machine-readable record:
`experiments/configs/phase00/ref_tactile/latest_20260701_web_codebase_refresh_v1.json`

## Sources Checked

The latest web/source refresh confirms the current serious reference set:

- Newton official GitHub: `https://github.com/newton-physics/newton`
- UniVTAC official GitHub: `https://github.com/univtac/UniVTAC`
- TaCauchy official GitHub: `https://github.com/figsama/TaCauchy`
- TacEx official GitHub: `https://github.com/DH-Ng/TacEx`
- Isaac Lab official TacSL docs:
  `https://isaac-sim.github.io/IsaacLab/main/source/overview/core-concepts/sensors/visuo_tactile_sensor.html`
- FTP-1 official GitHub: `https://github.com/michaelyuancb/ftp1-policy`
- AnyTouch2 official GitHub: `https://github.com/GeWu-Lab/AnyTouch2`

## New Local Sources

Three official sources were added in blobless/sparse mode:

- `external/ftp1-policy`, commit
  `dd7cda66c7e97a170e0435fc6c4428b350cbdcc0`
- `external/AnyTouch2`, commit
  `82c5677d9cf0176d97a1fe04745f63cd02dd6f54`
- `external/IsaacLab_official`, commit
  `b4c321024792976150ca55fddb26fa34480d974e`

No checkpoint was downloaded, no model was loaded, and no dependency install was
run.

## IsaacLab TacSL Finding

The official Isaac Lab main branch is now the correct TacSL source to track.
The sparse checkout includes:

- `docs/source/overview/core-concepts/sensors/visuo_tactile_sensor.rst`
- `scripts/demos/sensors/tacsl_sensor.py`
- `source/isaaclab_contrib/isaaclab_contrib/sensors/tacsl_sensor/`

The official docs describe TacSL-backed visuo-tactile sensing with tactile RGB
images, force field distributions, and intermediate tactile measurements. The
data container exposes:

- `tactile_depth_image`
- `tactile_rgb_image`
- `tactile_points_pos_w`
- `tactile_points_quat_w`
- `penetration_depth`
- `tactile_normal_force`
- `tactile_shear_force`

The config exposes `normal_contact_stiffness`, `friction_coefficient`,
`tangential_stiffness`, `contact_object_prim_path_expr`, `tactile_array_size`,
and `tactile_margin`. The demo entrypoint supports `--use_tactile_rgb`,
`--use_tactile_ff`, `--normal_contact_stiffness`, `--tangential_stiffness`,
`--friction_coefficient`, and `--save_viz`.

Gate effect: official IsaacLab TacSL should be added to Gate 00F as a serious
semantic-validation candidate. It does not clear Gate 00F until a
dependency-complete approved IsaacLab environment or prebuilt container exists
and official sanity runs inside tmux-held Slurm.

## FTP-1 Finding

FTP-1 is a 2026 generalist foundation tactile policy source. Its README links
pretrained and UniVTAC-finetuned checkpoints through HuggingFace/ModelScope,
describes OpenPI-based fine-tuning, and uses UniVTAC as an example domain. It
expects tactile inputs such as GelSightMini image tensors with sensor labels
and action chunks from an inference wrapper.

Gate effect: FTP-1 is a future serious policy/checkpoint baseline, not a
Newton-native infant checkpoint and not current Gate 00D/00E/00F completion
evidence.

## AnyTouch2 Finding

AnyTouch2 is an ICLR 2026 optical tactile representation source. Its README
links a HuggingFace model checkpoint and ToucHD/Sparsh evaluation datasets, but
checkpoint/data access may require completing HuggingFace forms, and the quick
start script is still marked coming soon.

Gate effect: AnyTouch2 is a future tactile representation/encoder comparison
reference. It is not a grasp controller and not current official tactile
simulation sanity.

## Decision

Keep Newton d58 as the current strongest complete runtime/tactile evidence
chain until 8c501 downstream tactile evidence exists. Latest Newton 8c501 is
not blocked by FPS because its 80.1/80.8 FPS H200 runs are acceptable for
continuing. Add official IsaacLab TacSL to the Gate 00F semantic-validation
shortlist. Add FTP-1 and AnyTouch2 to the future policy/representation
comparison shortlist. Curiosity training remains
disallowed until dense tactile/base/official semantic gates pass.
