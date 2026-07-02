# Phase 00 Direct Force Path Audit

Date: 2026-07-01

Status: active blocker note. This is source audit evidence, not training.

## Finding

Newton v1.3.0 has an official `SensorContact` path that can report
`total_force`, `total_force_friction`, `force_matrix`, and
`force_matrix_friction`, but that path depends on a `Contacts.force` buffer
being populated by `solver.update_contacts(...)`.

For `SolverMuJoCo`, `update_contacts()` is documented in the implementation as
updating `Contacts` from MuJoCo contacts when running with
`use_mujoco_contacts`. It converts `mjw_data.contact` and `mjw_data.efc.force`
into Newton contact arrays.

The active base environment is the official Newton Panda hydro example:
`newton.examples.robot.example_robot_panda_hydro`. That example constructs
`SolverMuJoCo(..., use_mujoco_contacts=False, ...)` and gets contacts from the
Newton hydroelastic collision pipeline. Therefore the official sensor/update
path is not directly compatible with the current hydroelastic Newton-contacts
path.

## Evidence

- Official sensor example:
  `external/newton_v1.3/newton/examples/sensors/example_sensor_contact.py`
  creates `SensorContact`, creates `Contacts` with requested force attributes,
  calls `solver.update_contacts(self.contacts, self.state_0)`, then calls
  `sensor.update(...)`.
- Official implementation:
  `external/newton_v1.3/newton/_src/solvers/mujoco/solver_mujoco.py` defines
  `update_contacts()` as a conversion from MuJoCo contact data and
  `mj_data.efc.force`.
- Active Panda hydro example:
  `external/newton_v1.3/newton/examples/robot/example_robot_panda_hydro.py`
  uses `use_mujoco_contacts=False`.
- Diagnostic run:
  `p00_force_probe_20260701_032310` tried to request `Contacts.force`, recreate
  collision-pipeline contacts, and call `SolverMuJoCo.update_contacts()` after
  each official Panda hydro step. It failed with CUDA illegal memory access and
  produced no valid force arrays.

## Consequence

Direct solver `Ft` and direct `SensorContact.total_force_friction` are blocked
for the current official Panda hydro path. The current valid fields must remain
explicitly labeled as Newton hydro-derived proxies:

- `hydro_proxy.Fn`
- `hydro_proxy.stress`
- `hydro_proxy.Ft_capacity`
- `hydro_proxy.shear_vector_map`
- `hydro_proxy.deform_proxy_map`

Do not present these as direct tactile force measurements.

## Next Faithful Options

- Find an official Newton hydroelastic force export path, if one exists, that
  does not go through MuJoCo `use_mujoco_contacts`.
- Test whether a MuJoCo-contact variant of the base can still pass grasp/lift
  and export `SensorContact` friction, but treat it as a separate variant
  unless it preserves the hydroelastic tactile evidence.
- Continue exporting T-Rex-aligned F6 proxy fields from Newton hydro reducer
  buffers, with explicit provenance, until direct force is solved.

## MuJoCo-Contact Variant Result

Run `p00_mjc_sensor_v1_20260701_034541` confirms that the official
`SensorContact` path can produce direct force/friction on a related Panda
MuJoCo-contact variant:

- status: `pass_nonzero_friction`
- max total force norm: `29.69374656677246`
- max total friction norm: `8.532435417175293`
- max per-counterpart force norm: `18.80536460876465`
- max per-counterpart friction norm: `4.5348076820373535`
- lift success over `0.15` m: `true`
- max object lift: `0.21197126805782318` m
- update errors: `0`

This is useful as a direct-force comparison source, but it is not the active
hydro tactile base. It explicitly uses
`SolverMuJoCo(use_mujoco_contacts=True)` and no Newton hydro collision pipeline,
so it cannot close the reference-video hydro tactile gate by itself.
