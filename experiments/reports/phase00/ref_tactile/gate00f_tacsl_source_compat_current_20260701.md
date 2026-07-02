# Gate 00F TacSL Source Compatibility Current Check

- Date: `2026-07-01`
- Classification: `source_compat_validation_not_runtime_not_gate_completion`
- Validator:
  `src/newton_tactile_curiosity/gate00f_tacsl_source_compat_validate.py`
- Summary:
  `experiments/outputs/phase00/ref_tactile/tacsl_source_compat/p00_tacsl_src_compat_20260701/tacsl_source_compat_summary.json`
- Status: `pass_tacsl_source_compat`

## Findings

- Local IsaacLab source `VERSION`: `2.3.2`
- Candidate image ref: `nvcr.io/nvidia/isaac-lab:2.3.2`
- Required TacSL data fields are present:
  `tactile_depth_image`, `tactile_rgb_image`, `tactile_points_pos_w`,
  `tactile_points_quat_w`, `penetration_depth`, `tactile_normal_force`, and
  `tactile_shear_force`.
- Required demo flags are present:
  `--use_tactile_rgb`, `--use_tactile_ff`, `--normal_contact_stiffness`,
  `--tangential_stiffness`, `--friction_coefficient`,
  `--contact_object_type`, `--save_viz`, and `--enable_cameras`.
- Required imports are present for `isaaclab_contrib` TacSL classes and
  `GELSIGHT_R15_CFG`.

## Gate Effect

This is positive source-compatibility evidence for the IsaacLab TacSL
container candidate, but it is not runtime evidence. No container was pulled,
no image was built, no module was imported, no Isaac Sim process ran, and no
runtime was registered as dependency-complete. Gate 00F remains open.
