# Local experiment index

`experiments/` is local-only and Git-ignored. It contains only the checkpoints, traces, videos and
small summaries needed to reproduce or inspect current conclusions. Failed launches, invalid
comparisons, superseded calibrations, duplicate renders, intermediate checkpoints and runtime logs
are under root `legacy/`.

## 1. `demo_following/`

- `teacher_only_carry45_gate_corrected_v1/`: 20-profile zero-residual prerequisite gate;
- `matched_reward_identity_same_teacher_v1/`: current 64-update causal comparison, both policy
  checkpoints, matched frozen traces/results and full videos;
- `predictor/`: frozen 11.9M causal future-mismatch predictor result and checkpoint;
- `smp_prior/`: generic official MimicKit TinyMDM prior used identically by both arms;
- `selected_demo_smp_v1/`: official CarryBox45/KickBox21 single-clip priors and the failed
  independent semantic-extension gate;
- `runtime_assets/`: compact frozen inputs required by current scripts.

The old 1216-update experiment changed both teacher and reward demo and used a defective task. It
has been moved to `legacy/repository_cleanup_20260823/experiments/demo_following/` and is not an
active result.

## 2. `online_patch_tactile_mass_adaptation/`

- `corrected_rerun_20260820/`: only the model1100 tactile-only checkpoint/config and the
  model1100/model1250 frozen diagnostic summaries;
- `leakage_sweep_v1/`: three-seed, five-mass fixed-action proprioception/tactile leakage traces;
- `friction_feasibility_after_ps/`: independent frozen-Refiner `4 friction x 2 mass` sweep;
- `visualizations/official_refiner_mu1p5_6x_friction_hold_single_env/`: the one retained heavy-box
  synchronized world/bilateral-27-patch video;
- `runtime_assets/`: preconverted G1 asset required by active collectors.

No valid corrected matched Z/P/PS result exists. Historical formal endpoints, evaluations and
their videos are archived and must not be used for a tactile-benefit claim.

## 3. `isaaclab_g1_anatomical27_object_demos/`

Four retained native IsaacLab/PhysX sensor examples:

- normal CarryBox;
- free palm-area lift;
- 2 kg palm grip;
- release/failure behavior.

Each directory keeps the raw online trace, summary, world video, synchronized bilateral tactile
video and render metadata so the visualization can be independently regenerated.

## 4. `sugar_reproduction/`

- `outputs/final/official_sugar/`: frozen Refiner checkpoint, baseline summaries/videos and one
  released CarryBox inference;
- `assets/official_tacsl/`: official R15 USD and GelSight calibration.

Routine logs have been archived. Exact commands and result expectations are in
[`DOCS/reproducibility.md`](../DOCS/reproducibility.md).
