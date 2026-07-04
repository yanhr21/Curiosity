# Isaac Arena G1 Loco-Manipulation Preflight

Date: 2026-07-02.

## User Requirement

Use Isaac for real, high-quality physics simulation. Do not downgrade to a toy
or browser-only visualization.

## Real Target Selected

Official Isaac Lab-Arena workflow:

```text
galileo_g1_locomanip_pick_and_place
```

This is the Arena G1 loco-manipulation brown-box-to-blue-bin task:

- Unitree G1 humanoid with whole-body controller;
- Galileo lab scene;
- rigid brown box;
- blue sorting bin;
- PhysX simulation;
- official closed-loop GR00T policy path;
- camera/viewport MP4 recording through Arena policy runner.

The official docs describe the task as requiring navigation, squatting,
turning, walking, picking, placing, and bimanual manipulation.

## Official Code Prepared

Local sources:

- `external/IsaacLab-Arena` at
  `8a74e794b621b0f8d3627d096a1bae9ce11e7b56`.
- `external/IsaacLab-Arena/submodules/IsaacLab` fetched at pinned commit
  `55df2c34390ba94b22d41879514c5485c5115462`.
- `external/IsaacLab-Arena/submodules/Isaac-GR00T` fetched at pinned commit
  `e29d8fc50b0e4745120ae3fb72447986fe638aa6`.

The pinned commits come from Arena's own submodule entries.

## Runtime Environments Prepared

Two local shared-filesystem environments were prepared on the login node for
later activation inside a Slurm compute allocation:

- Isaac/Arena environment:
  `/public/home/yanhongru/envs/isaac_arena_py312`, Python 3.12.6.
- GR00T server environment:
  `/public/home/yanhongru/envs/gr00t_n16_py310`, Python 3.10.20.

The split environment is intentional. Arena/Isaac Sim use the Isaac Sim 6.0.1
Python stack, while the official GR00T server path needs the GR00T/N1.6
inference dependency stack including PyTorch 2.7.1 and flash-attn. The Arena
evaluation script runs the official GR00T server from the GR00T environment and
the Isaac simulation from the Isaac/Arena environment.

Dependency check status:

- GR00T environment: `uv pip check` passes.
- Isaac/Arena environment: `uv pip check` still reports 9 metadata
  incompatibilities. The notable ones are `warp-lang` 1.13.0 vs IsaacLab's
  pinned 1.12.0, `daqp` 0.8.5 vs IsaacLab's pinned 0.7.2, missing
  `nvidia-srl-usd-to-urdf` for teleop/mimic extras, and Isaac Sim metadata
  pins for `llvmlite`, `packaging`, `pillow`, `typing-extensions`, and
  `coverage`. These are not claimed resolved until the compute-node simulation
  smoke test proves the official policy runner can launch.

## Official Assets

HuggingFace auth works for user `Railgun526`.

Dry-run result for generated simulation dataset:

```text
nvidia/Arena-G1-Loco-Manipulation-Task
arena_g1_loco_manipulation_dataset_generated.hdf5
revision arena_v0.2_lab_v3.0
size: 23.4G
```

This dataset is not required for pretrained closed-loop evaluation if the
official tuned checkpoint is available.

The official checkpoint path from Arena docs is:

```text
nvidia/GN1x-Tuned-Arena-G1-Loco-Manipulation
revision: gn1_6
local target:
/public/home/yanhongru/models/isaaclab_arena/locomanipulation_tutorial/checkpoint-20000
```

The official tuned checkpoint was downloaded as an inference-only subset:

```text
/public/home/yanhongru/models/isaaclab_arena/locomanipulation_tutorial/checkpoint-20000
total size: 6.2G
```

Included files:

```text
config.json
embodiment_id.json
latest
model-00001-of-00002.safetensors
model-00002-of-00002.safetensors
model.safetensors.index.json
processor_config.json
statistics.json
README.md
.gitattributes
experiment_cfg/*
```

Large training-state artifacts such as optimizer states were intentionally not
downloaded. They are not needed for the first official closed-loop inference
smoke test and would add tens of GB.

## Scripts Added

- `scripts/isaac/download_arena_g1_official_assets.sh`
- `scripts/isaac/run_arena_g1_locomanip_eval.sh`

The download script may run on the login node because it only fetches official
assets and does not load models or run simulation. The evaluation script refuses
to run on `mgmtserver*`.

The evaluation script is intentionally strict:

- it requires the official Arena source;
- it requires the pinned Isaac-GR00T server script;
- it requires the official checkpoint directory;
- it requires the prepared Isaac/Arena and GR00T Python environments;
- it launches the official GR00T server from the GR00T environment;
- it runs Arena `policy_runner.py` on
  `galileo_g1_locomanip_pick_and_place`;
- it records camera and viewport videos.

## 2026-07-03 Update: Compute Smoke Progress

Completed:

- Official Arena, IsaacLab, and Isaac-GR00T repositories are cloned.
- Official Arena G1 loco-manipulation inference checkpoint is present.
- GR00T server environment dependency metadata check passes.
- Evaluation script is wired to the official GR00T server and Arena policy
  runner.
- Login-node downloads completed for the remote USD assets that blocked compute
  nodes:
  - `/public/home/yanhongru/isaac_asset_mirror/Assets/Isaac/6.0/galileo_locomanip.usd`
  - `/public/home/yanhongru/isaac_asset_mirror/Assets/Isaac/6.0/brown_box.usd`
  - `/public/home/yanhongru/isaac_asset_mirror/Assets/Isaac/6.0/blue_sorting_bin.usd`
  - `/public/home/yanhongru/isaac_asset_mirror/Assets/Isaac/6.0/g1_29dof_with_hand_rev_1_0.usd`
- Arena source now uses local-first USD fallback for the official Galileo
  background, brown box, blue sorting bin, and G1 embodiment. This is a
  cluster asset-resolution patch only; it does not change the task, policy,
  controller, or physics model.
- Local Omniverse extension cache now includes a compatibility shim:
  `omni.physics.tensors.impl.api -> omni.physics.tensors.api`. This fixed the
  IsaacLab/Omniverse namespace mismatch that blocked ContactSensor creation.

Not yet completed:

- The official Arena task has not completed a rollout.
- No MP4 physics-simulation evidence exists yet.
- The latest smoke was interrupted by us after it stayed in Galileo scene
  PhysX mesh cooking for more than 15 minutes without new log progress.

This is not complete until `scripts/isaac/run_arena_g1_locomanip_eval.sh`
successfully runs inside a compute allocation and produces an MP4 plus logs.
Do not treat the current state as a task success.

## Compute Smoke Attempts

Curiosity-owned tmux sessions were used for Slurm-held compute allocations:

```text
curiosity_isaac_arena_g1_run_0702
curiosity_isaac_arena_g1_run_0703
```

The smoke tests are diagnostics only, not real training or success claims.

Important logs:

- `logs/isaac_arena_g1_locomanip/gr00t_server_20260703_011554.log`
- `logs/isaac_arena_g1_locomanip/arena_eval_20260703_011554.log`

Observed progression:

- GR00T checkpoint loads from the official tuned checkpoint.
- Isaac AppLauncher starts on compute node GPU.
- Arena policy runner dynamically imports the official
  `Gr00tRemoteClosedloopPolicy`.
- The generated Arena config shows local USD paths for Galileo, brown box, blue
  sorting bin, and G1.
- The prior `omni.physics.tensors.impl` import blocker is resolved.
- Environment creation reaches PhysX cooking for Galileo shelf meshes.

Current blocker:

```text
PhysX cooking repeatedly warns that Galileo shelf triangle/convex meshes are
too large or oblong, falls back to CPU collision, and then stops producing new
log output. The smoke was interrupted after prolonged no-progress behavior.
```

This may be first-run cooking latency, a bad heavy collision asset path, or a
need for a simplified official diagnostic scene. It is not yet evidence that
the official closed-loop loco-manipulation task works in this environment.

## 2026-07-03 Pivot: Direct Isaac Carry Scene First

The official Arena/Galileo/GR00T path is no longer treated as a blocker for the
project's main execution path. It remains useful as a reference implementation
and possible later baseline, but waiting on the full Galileo scene and official
policy stack is too slow for building the research task.

New immediate path:

- construct the carry scene directly in Isaac/PhysX;
- validate primitive floor, target marker, and a dynamic rigid carry box first;
- parameterize box mass, size, and initial pose;
- record box pose and optional robot base pose to CSV;
- add robot embodiment only after the pure box scene is confirmed running;
- then add contact/probing/carry objectives incrementally.

New scripts:

- `scripts/isaac/build_minimal_carry_scene.py`
- `scripts/isaac/run_minimal_carry_scene.sh`

The minimal scene has a `--skip-robot` / `SKIP_ROBOT=1` mode that intentionally
does not import the Arena G1 asset path. This mode is a scene-construction and
PhysX smoke test only. It is not a policy, not a carrying result, and not a
success claim.

Current pending smoke:

```text
tmux: curiosity_minimal_carry_box_0703
Slurm job: 162515
command:
SKIP_ROBOT=1 STEPS=120 \
OUTPUT_DIR=/public/home/yanhongru/Curiosity/experiments/outputs/minimal_carry_scene/smoke_20260703_skip_robot_120steps \
bash scripts/isaac/run_minimal_carry_scene.sh
```

## 2026-07-03 Direct Scene Smoke Results

The direct Isaac scene now runs far enough to create the world, floor, target
marker, and dynamic carry box. The official Arena/Galileo scene is no longer
blocking this path.

Implemented changes:

- `scripts/isaac/build_minimal_carry_scene.py`
  - box-only mode: `--skip-robot`;
  - no-render default for headless CSV smoke;
  - explicit headless Kit path through the launcher script;
  - `SimulationCfg(use_fabric=False)` for observable smoke;
  - `/physics/updateToUsd=True` and `/physics/updateVelocitiesToUsd=True`;
  - USD world-pose logging for the carry box.
- `scripts/isaac/run_minimal_carry_scene.sh`
  - `SKIP_ROBOT`, `RENDER`, `DEVICE`, box size, box mass, and output directory
    environment controls.

Confirmed CPU PhysX smoke:

```text
tmux: curiosity_minimal_carry_box_norender_0703
Slurm job: 162517
node: server64
command:
DEVICE=cpu SKIP_ROBOT=1 RENDER=0 STEPS=120 \
OUTPUT_DIR=/public/home/yanhongru/Curiosity/experiments/outputs/minimal_carry_scene/smoke_20260703_skip_robot_usd_update_120steps \
bash scripts/isaac/run_minimal_carry_scene.sh
```

Evidence:

```text
log:
/public/home/yanhongru/Curiosity/logs/minimal_carry_scene/minimal_carry_scene_20260703_120419.log

csv:
/public/home/yanhongru/Curiosity/experiments/outputs/minimal_carry_scene/smoke_20260703_skip_robot_usd_update_120steps/minimal_carry_scene_state.csv
```

Observed box z position:

```text
step 0:   z = 0.4463
step 40:  z = 0.1962
step 50:  z = 0.1750
step 119: z = 0.1750
```

This is the expected drop-and-settle behavior for a box with height 0.35 m on
the floor. This confirms the direct Isaac floor, rigid box, gravity, collision,
and USD-pose logging path on CPU PhysX.

CUDA diagnostic:

```text
DEVICE=cuda:0 SKIP_ROBOT=1 RENDER=0 STEPS=120 \
OUTPUT_DIR=/public/home/yanhongru/Curiosity/experiments/outputs/minimal_carry_scene/smoke_20260703_skip_robot_cuda_usd_update_120steps \
bash scripts/isaac/run_minimal_carry_scene.sh
```

The CUDA run completes but the USD-recorded box pose remains static at the
initial z position. Direct IsaacLab tensor reads also fail in this environment
with invalidated `omni.physx.tensors` simulation views. Therefore CUDA/tensor
state access is not yet validated and must not be used for training claims.

Current status:

- direct Isaac scene construction: working;
- CPU PhysX box dynamics and collision: working;
- USD pose logging with `updateToUsd`: working on CPU;
- CUDA/tensor state path: blocked;
- robot integration: not started in this direct scene path.
