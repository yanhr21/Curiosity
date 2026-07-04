# TODO 01: Simulation Platform Preflight

- [x] Open a Curiosity-owned tmux session for compute work:
  `curiosity_isaac_arena_g1_run_0702`.
- [x] Acquire a persistent Slurm allocation with `srun` or `salloc`.
- [x] Confirm the node is not a login or management node before running any
  Python, Isaac, rendering, or model-loading command.
- [x] Choose Arena source state and record why: use the official cloned
  IsaacLab-Arena source and its pinned IsaacLab / Isaac-GR00T submodules first,
  because this is the documented G1 loco-manipulation path.
- [x] Prepare local shared-filesystem Isaac/Arena environment:
  `/public/home/yanhongru/envs/isaac_arena_py312`.
- [x] Prepare local shared-filesystem GR00T server environment:
  `/public/home/yanhongru/envs/gr00t_n16_py310`.
- [x] Download official Arena G1 loco-manipulation inference checkpoint under
  `/public/home/yanhongru/models/isaaclab_arena/locomanipulation_tutorial/checkpoint-20000`.
- [x] Run official Arena G1 loco-manipulation closed-loop smoke test inside
  compute allocation.
- [x] Record GR00T server log, Arena eval log, exact command, node, and Slurm
  job id for the blocked smoke.
- [ ] Record MP4 output path from a completed official smoke; none exists yet.
- [x] Record official Arena blocker: Galileo scene PhysX mesh cooking stalls
  after local USD and ContactSensor issues are fixed.
- [x] Pivot to direct Isaac scene construction instead of waiting on official
  Arena/Galileo/GR00T assets.
- [x] Add a minimal Isaac carry-scene scaffold script that can run without the
  Arena G1 asset path: `scripts/isaac/build_minimal_carry_scene.py`.
- [x] Add a compute-node launcher for the minimal carry scene:
  `scripts/isaac/run_minimal_carry_scene.sh`.
- [x] Add `SKIP_ROBOT=1` / `--skip-robot` mode so floor, target marker, and
  dynamic carry box can be validated before robot integration.
- [x] Run `SKIP_ROBOT=1` minimal carry-scene smoke inside a Curiosity-owned
  Slurm allocation and record CSV/log output.
- [x] Verify CPU PhysX box dynamics in the direct Isaac scene:
  `experiments/outputs/minimal_carry_scene/smoke_20260703_skip_robot_usd_update_120steps/minimal_carry_scene_state.csv`.
- [ ] Resolve GPU/tensor pipeline issue: `DEVICE=cuda:0` completes the script
  but the USD-recorded box pose stays static, while direct tensor reads fail
  with invalidated `omni.physx.tensors` simulation view.
- [ ] Re-enable the G1 asset only after the pure box scene is verified.
- [ ] Re-run official Arena smoke after the Galileo cooking blocker is resolved.
- [ ] Produce MP4 evidence from the official task; none exists yet.
- [ ] Treat GR00T-VisualSim2Real, WBC-AGILE, and Arena policies as references
  or baselines, not as blockers for building the unknown-load carry scene.
- [x] Create a preflight report under `experiments/reports/`.
