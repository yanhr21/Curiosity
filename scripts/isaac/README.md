# Isaac Carry-Scene Execution

This directory is for the real Isaac/PhysX path. It must not be replaced by a
toy browser, hand-drawn, or kinematic imitation.

Current project priority is direct Isaac scene construction for unknown-load
carrying. Official Arena/Galileo/GR00T assets are useful references and possible
baselines, but they are not a prerequisite for building the core scene.

## Current Main Path

Verified pure Isaac core-World dynamic payload substrate:

```bash
STAMP=YYYYMMDD_core_world_fixed_payload \
STEPS=240 TARGET_SPEED=0.30 PAYLOAD_MASS=4.0 JOINT_MODE=center_weld DEVICE=cpu \
bash scripts/isaac/run_core_world_simapp_fixed_payload_carry.sh
```

This is the current verified non-tensor Isaac dynamics path. It uses pure
`SimulationApp`, Isaac Sim core `World`, CPU PhysX, a dynamic carrier body, and
a physical payload box attached by a USD fixed joint. Validated run
`20260704_core_world_simapp_fixed_payload_centerweld_diag2` completed 240/240
steps with carrier and payload both traveling 0.3596 m, relative payload error
0.0 m, fall events 0, and payload drop events 0. This is not legged walking,
unknown-object grasping, or learned carrying; it is the dynamic substrate to
extend while the official tensor-policy routes remain broken.

Adaptive dynamic fixed-payload carry task:

```bash
STAMP=YYYYMMDD_core_world_adaptive_payload \
DEVICE=cpu STEPS=360 TARGET_X=0.30 PAYLOAD_MASS=8.0 PAYLOAD_COM_X=0.04 \
ROBOT_HEIGHT=1.35 ROBOT_MASS=48.0 ARM_LENGTH=0.52 MAX_PAYLOAD=16.0 \
BASE_SPEED=0.34 \
bash scripts/isaac/run_core_world_simapp_adaptive_payload_carry.sh
```

This extends the verified dynamic substrate with strategy selection,
probe/settle/carry phases, visible walking-support foot markers, gait/support
proxy metrics, target-distance metrics, balance-margin proxy, effort proxy,
fall/drop metrics, CSV state, and summary JSON. Validated sweep
`20260704_core_world_adaptive_payload_walkproxy_diag1` ran two cases in Slurm
job `165292`: `low_front_carry` moved carrier/payload 0.3287 m with final
target distance 0.0287 m and `min_support_margin_x_proxy_m` 0.1185 m;
`chest_supported_slow` moved carrier/payload 0.2568 m with final target
distance 0.0568 m and `min_support_margin_x_proxy_m` 0.1109 m. Both had fall
events 0, drop events 0, and payload relative error 0.0 m. This remains
diagnostic-only: no legged walking, no free-object grasping, no learned
balance.

One-startup strategy sweep:

```bash
STAMP=YYYYMMDD_core_world_adaptive_payload_sweep \
DEVICE=cpu PRESET_SWEEP=strategy_smoke \
bash scripts/isaac/run_core_world_simapp_adaptive_payload_carry.sh
```

Validated sweep `20260704_core_world_adaptive_payload_strategy_sweep1` ran two
cases in one `SimulationApp` startup. Strategy counts were
`low_front_carry: 1` and `chest_supported_slow: 1`; both cases completed
360/360 steps with fall events 0, drop events 0, payload relative error 0.0 m,
and nonzero carrier/payload travel. This is the preferred way to test adaptive
strategy cases until repeated Kit startup stability is understood.

Quasi-static walker fixed-payload carry diagnostic:

```bash
STAMP=YYYYMMDD_core_world_quasistatic_walker \
DEVICE=cpu STEPS=420 TARGET_X=0.18 PAYLOAD_MASS=8.0 PAYLOAD_COM_X=0.04 \
ROBOT_MASS=48.0 ROBOT_HEIGHT=1.20 ARM_LENGTH=0.52 MAX_PAYLOAD=16.0 \
BASE_SPEED=0.30 GAIT_FREQUENCY=1.15 \
bash scripts/isaac/run_core_world_simapp_quasistatic_walker_carry.sh
```

This is the strongest current direct Isaac carrying-task diagnostic. It uses
pure `SimulationApp` core `World`, a dynamic walker body, a dynamic physical
payload fixed by USD joint, four visible/colliding support feet, visible leg
struts, one-foot-swing/three-foot-stance creep gait, support-state logging,
support-margin gating, target hold, and fall/drop metrics. Validated run
`20260704_core_world_quasistatic_walker_hold_diag3` completed 420/420 steps,
moved body/payload 0.1659 m, ended 0.0141 m from the target, had payload
relative error 0.0 m, minimum support margin 0.1325 m, fall events 0, and
payload drop events 0. This remains diagnostic-only: torso motion is still
commanded through rigid-body velocity control, not a verified articulated
walking policy, and the payload is fixed rather than grasped as an unknown
free object.

Staged free-box carry diagnostic:

```bash
STAMP=YYYYMMDD_core_world_staged_free_box \
DEVICE=cpu STEPS=560 TARGET_X=0.48 BOX_X=0.28 BOX_MASS=8.0 BOX_COM_X=0.04 \
ROBOT_MASS=48.0 ROBOT_HEIGHT=1.20 ARM_LENGTH=0.52 MAX_PAYLOAD=16.0 \
BASE_SPEED=0.30 GAIT_FREQUENCY=1.15 ATTACHMENT_MODE=dynamic-contact-proxy \
CARRIER_MODE=dynamic-velocity \
bash scripts/isaac/run_core_world_simapp_staged_free_box_carry.sh
```

This is the next direct Isaac task scaffold. The box starts as a free dynamic
rigid body, the carrier approaches and runs a short probing phase, then a
logged staged lift/hold event activates the selected attach proxy before
carrying to the target. The point is to establish the task structure and
metrics for a later contact-grasping and learned-control version: free-box
phase, probing attempts, attach step, target distance, support margin, fall
events, box drop events, and proxy grip gap. This is explicitly not contact
grasping, not articulated locomotion, and not a learned policy.

Current 2026-07-04 status: implemented and validated as a direct Isaac
diagnostic scaffold. Smoke attempts `20260704_core_world_staged_free_box_diag1` and
`20260704_core_world_staged_free_box_diag2` did not reach scene construction
because `SimulationApp` stalled during startup. Fresh allocation health check
`20260704_core_world_quasistatic_health_server10` completed 20/20 steps.
`20260704_core_world_staged_free_box_diag3_server10` then completed 520/520
steps with attach step 260, 77 probe attempts, body travel 0.3492 m, box
travel 0.0563 m, fall events 0, and box drop events 0. It is a negative
attach-quality result: PhysX warned that the runtime fixed joint had disjoint
body transforms, the box snapped backward on attach, final target distance was
0.1648 m, and `box_relative_error_m_after_attach` was 0.1868 m. The attach
logic was revised to use a two-step settle/attach sequence and a measured
body-box relative joint pose. Fresh-allocation run
`20260704_core_world_staged_free_box_diag4_settle_attach` completed 560/560
steps with attach prep step 260, attach step 261, final target distance
0.0111 m, body travel 0.3591 m, box travel 0.1515 m, minimum support margin
0.1299 m, fall events 0, and box drop events 0. It is still not final
success: PhysX continued to warn about disjoint fixed-joint body transforms and
`box_relative_error_m_after_attach` remained 0.0824 m.

The current strongest staged free-box mode is `ATTACHMENT_MODE=dynamic-contact-proxy`
with `CARRIER_MODE=dynamic-velocity`. It keeps left/right palm, chest, and
forearm-shelf proxy bodies dynamic; after staged attach, the box is not
directly velocity-servoed, so box motion comes through PhysX contact with
those proxies. Validated run
`20260704_core_world_staged_free_box_diag15_dynamic_contact_proxy_standby`
completed 420/420 steps, attached at step 91, selected `low_front_creep`,
moved the body 0.25052 m and the box 0.48505 m, ended 0.01490 m from the
target, had final relative error 0.06311 m, peak relative error 0.06348 m,
contact-proxy grip gap 0.06643 m, max grip gap 0.06818 m, minimum support
margin 0.13252 m, fall/drop events 0, and no disjoint warning. It is still a
diagnostic bridge: the lift is staged, proxy bodies are velocity commanded,
and the carrier is not an articulated walking robot.

The stricter balance/hold validation is
`20260704_core_world_staged_free_box_diag16_dynamic_contact_balance_hold`.
It completed 430/430 steps, attached at step 91, moved the body 0.25052 m and
the box 0.48435 m, ended 0.01474 m from the target, had final/peak relative
error 0.06428 m, max grip gap 0.06825 m, `target_hold_steps=24`,
`carry_phase_steps=338`, `min_stance_count=3.0`,
`min_support_margin_after_attach_m=0.13252`, max command speed 0.174 m/s,
fall/drop events 0, and no disjoint warning. This is the current strongest
staged free-box carry diagnostic, but it is not final robot-carrying success:
walking and balance are support-proxy verified, not generated by an
articulated robot controller.

Gravity/contact-support audit: `BODY_VERTICAL_MODE=preserve` was added because
the default velocity-commanded carrier writes `[speed, 0, 0]` and therefore
zeros vertical velocity. With `PHYSICAL_SUPPORT_MODE=deck`,
`20260704_core_world_staged_free_box_diag17_dynamic_contact_preserve_z_deck`
completed but failed: vertical-velocity preservation was available, but the
support deck was too tightly preloaded and launched the body upward
(`max_body_z_deviation_m=1.39366`), causing 18 box drop events and final target
distance 0.31088 m. A `SUPPORT_DECK_GAP` patch is available; rerun the
preserve-z/deck diagnostic in a fresh allocation with `SUPPORT_DECK_GAP=0.02`
and checker flags `--expect-support-deck-gap 0.02 --max-body-z-deviation 0.08`
before treating physical support as verified.

`PHYSICAL_SUPPORT_MODE=runway` is a fixed long support surface meant to avoid
moving-deck energy injection. Run
`20260704_core_world_staged_free_box_diag19_dynamic_contact_preserve_z_runway`
completed 430/430 without fall/drop events, but failed the strict physical
support gate: `target_hold_steps=0`, `max_body_z_deviation_m=1.92729`, final
relative error 1.69750 m, and max grip gap 0.21576 m. Treat this as negative
evidence for the current support-surface approach, not as balance success.
Next work should use geometry-consistent physical support with no launch or
return to a real articulated foot-contact controller.

The earlier `ATTACHMENT_MODE=contact-proxy-servo` mode remains useful as a
clean servo-proxy baseline. Validated run
`20260704_core_world_staged_free_box_diag13_contact_proxy_servo` completed
420/420 steps, attached at step 91, selected `low_front_creep`, moved the body
0.20545 m and the dynamic box 0.47544 m, ended 0.01456 m from the target, had
final relative error `3.67e-06`, peak relative error `3.91e-06`, contact-proxy
grip gap `3.67e-06` m, max grip gap `3.91e-06` m, minimum support margin
0.13252 m, fall/drop events 0, and no disjoint warning. It adds explicit
left/right palm and chest support proxy geometry while still directly
velocity-servoing the box.

Use the lightweight checker after each staged free-box run:

```bash
python3 scripts/isaac/check_staged_free_box_summary.py \
  experiments/outputs/core_world_simapp_staged_free_box_carry/STAMP/core_world_simapp_staged_free_box_carry_summary.json \
  --log logs/core_world_simapp_staged_free_box_carry/core_world_simapp_staged_free_box_carry_STAMP.log \
  --require-attach --require-contact-proxy --require-dynamic-contact-proxy \
  --forbid-disjoint-warning \
  --max-target-distance 0.05 --max-relative-error 0.08 \
  --max-peak-relative-error 0.12 --max-contact-proxy-gap 0.12 \
  --min-body-travel 0.18 --min-box-travel 0.40 \
  --min-support-margin 0.10 --min-support-margin-after-attach 0.10 \
  --min-stance-count 3 --min-target-hold-steps 5 \
  --expect-strategy low_front_creep \
  --expect-attachment-mode dynamic-contact-proxy \
  --expect-carrier-mode dynamic-velocity
```

For any future claim that the robot itself walks and balances while carrying,
add the no-root shortcut gate:

```bash
python3 scripts/isaac/check_staged_free_box_summary.py SUMMARY.json \
  --require-no-root-shortcut \
  --require-articulated-carrier
```

This gate is intentionally stricter than the current scaffold checks. It
rejects velocity-commanded root-body motion, body pose writes, and box pose
writes, so old passing scaffolds such as `diag54` are not final robot-walking
evidence.

Validated negative check: applying this gate to
`20260704_core_world_staged_free_box_diag4_settle_attach` fails because
`box_relative_error_m_after_attach` is 0.0824 m and the log contains the
disjoint fixed-joint warning.

Official Go2 callback locomotion smoke:

```bash
STAMP=YYYYMMDD_go2_callback_nopayload \
DEVICE=cuda STEPS=220 WARMUP_STEPS=20 COMMAND_X=1.0 COMMAND_Y=0.0 COMMAND_YAW=0.0 \
PAYLOAD_MODE=none \
bash scripts/isaac/run_official_go2_callback_locomotion_smoke.sh
```

This is the current direct route for recovering real robot locomotion. It
matches NVIDIA's installed Go2 tests more closely than the older manual-loop
smoke by using `timeline.play()` and a
`SimulationManager.register_callback(..., IsaacEvents.POST_PHYSICS_STEP)`
policy loop. Passing it would provide Go2 walking evidence only. If no-payload
walking passes, use `PAYLOAD_MODE=fixed_base PAYLOAD_MASS=2.0` as the next
fixed-payload balance-under-load diagnostic.

Current 2026-07-04 status is negative. `OFFICIAL_TEST_KIT_ARGS=1` now correctly
passes NVIDIA's policy-example physx-test Kit settings through
`SimulationApp(extra_args=...)`. Even with those settings,
`20260704_go2_callback_officialkit_extraargs_diag7` still hard-exited at
`SimulationManager.set_physics_sim_device(cuda:0)`. With
`SIMULATION_MANAGER_MODE=skip_device_dt`,
`20260704_go2_callback_officialkit_extraargs_skip_diag8` completed but had
`callback_forward_calls=0`, `travel_xy_m=0.0`, and the same invalid
`/World/Go2/Geometry/base` physics simulation view warning. Do not rerun this
route unchanged. The official `kit/dev/repo.sh test` runner also tried to fetch
packman Python on the compute node and was interrupted; prepare those
dependencies locally before using that runner.

Velocity-assisted articulated carrier diagnostic:

```bash
STAMP=YYYYMMDD_core_world_quad_assisted \
BASE_VELOCITY_ASSIST=1 BASE_ASSIST_MODE=velocity TARGET_SPEED=0.24 \
STEPS=180 PAYLOAD_MASS=4.0 DEVICE=cpu \
bash scripts/isaac/run_core_world_dynamic_quadruped_carry_scene.sh
```

This route keeps the custom articulated quadruped and fixed payload, applies
joint-position gait actions, and optionally applies a clearly labeled base
assist through the Isaac core articulation API. `BASE_ASSIST_MODE=velocity`
sets root linear velocity; `BASE_ASSIST_MODE=pose` explicitly pose-commands
the articulation root. Both are diagnostic scaffold modes, not unassisted
walking or final carrying success.

Current 2026-07-04 results:

- `20260704_core_world_dynamic_quadruped_fixed_payload_diag2_runtime_pose`
  fixed the earlier false 0-travel metric by reading runtime prim poses with
  `SingleArticulation.get_world_pose()` and `SingleRigidPrim.get_world_pose()`.
  It completed 260/260 with torso travel 0.2107 m, box travel 0.2063 m, and
  box drops 0, but fall gate triggered 19 times and max tilt reached
  3.269 rad. This is unstable, not carrying success.
- `20260704_core_world_dynamic_quadruped_fixed_payload_diag3_slow_stable`
  reduced speed and gait amplitudes, but still had 21 fall-gate events.
- `20260704_core_world_dynamic_quadruped_fixed_payload_diag5_pose_assist_retry`
  is the current cleanest articulated fixed-payload scaffold: completed
  260/260, `base_assist_mode=pose`, max joint motion 0.3726 rad, torso and box
  travel both about 0.2340 m, min torso z 0.6193 m, max tilt 0, fall/drop
  events 0, and control errors 0. Use it only as a task-construction scaffold
  before replacing fixed payload and root pose assist with free-box contact and
  a real balance/locomotion controller.
- `PAYLOAD_MODE=staged_free_box` extends the same articulated scaffold so the
  box begins as a free dynamic body, is probed, staged-lifted, attached, and
  carried. The current best run is
  `20260704_core_world_dynamic_quadruped_staged_free_box_diag2_pose_lock_target_hold`:
  completed 540/540, attached at step 90, used `STAGED_ATTACH_MODE=pose-lock`
  and `BASE_ASSIST_MODE=pose`, reached `target_hold_steps=5`, final target
  distance 0.04495 m, torso travel 0.4050 m, box travel 0.37165 m,
  fall/drop events 0, no disjoint warning, and final/peak relative errors near
  `3e-08`. This is the current best articulated staged-free-box task
  scaffold, but it is still not physical grasping or unassisted walking.
- The physical fixed-joint staged attach route is negative. Both
  `20260704_core_world_dynamic_quadruped_staged_free_box_diag3_fixed_joint`
  and
  `20260704_core_world_dynamic_quadruped_staged_free_box_diag4_fixed_joint_runtime_create`
  produced PhysX disjoint fixed-joint warnings, box drop events 2, no target
  hold, kilometer-scale box target errors, and relative errors above
  2800 m. Do not rerun this fixed-joint route unchanged; replace it with a
  better contact/grasp formulation.
- `STAGED_ATTACH_MODE=velocity-servo` is a non-explosive dynamic-body
  transition scaffold, but it does not yet replace pose-lock. In
  `20260704_core_world_dynamic_quadruped_staged_free_box_diag6_velocity_servo_target_stop`,
  the box remained dynamic after attach, root motion stopped at the target,
  and the run completed 540/540 with fall/drop events 0, disjoint warning
  false, and `target_hold_steps=5`. However the box lagged the carry pose:
  final target distance was 0.10941 m and final relative error was 0.10943 m,
  so the strict checker failed. This is a useful bridge toward contact/proxy
  grasping, not a valid physical carry result.
- `STAGED_ATTACH_MODE=contact-proxy` creates dynamic left/right palm, chest,
  and forearm-shelf proxy bodies and drives those proxies around the carry
  pose without directly pose-locking or velocity-servoing the box. Current
  result `20260704_core_world_dynamic_quadruped_staged_free_box_diag7_contact_proxy`
  is negative: completed 540/540, attached at step 90, contact proxy enabled,
  fall events 0, and no disjoint warning, but box drop events were 34, target
  hold 0, final target distance 0.28981 m, final/peak relative error
  0.60057 m, and max contact-proxy gap 1.37539 m. The current proxies do not
  establish a stable grip; next work needs pre-closed geometry, stronger shelf
  support, normal clamping, or a controlled grip-force/constraint hybrid.

Checker:

```bash
python3 scripts/isaac/check_dynamic_quadruped_carry_summary.py SUMMARY.json \
  --log LOG.txt --expect-payload-mode staged_free_box \
  --expect-staged-attach-mode pose-lock --expect-base-assist-mode pose \
  --require-attach --forbid-disjoint-warning --max-fall-events 0 \
  --max-box-drop-events 0 --min-target-hold-steps 5 \
  --max-target-distance 0.05 --max-relative-error 0.001 \
  --max-peak-relative-error 0.001 --min-torso-travel 0.35 \
  --min-box-travel 0.35 --max-tilt 0.01
```

Adaptive direct Isaac sweep:

```bash
STAMP=YYYYMMDD_adaptive_direct_sweep STEPS=180 DEVICE=cuda:0 RENDER=0 \
bash scripts/isaac/run_adaptive_probe_carry_sweep.sh
```

This is the current main direct-scene path. It runs the adaptive active-probing
scaffold across mass, size, COM offset, robot height, arm length, and payload
limits, then writes an aggregate JSON. It is useful for checking task
parameterization and morphology/load-dependent posture-selection plumbing. It
is still diagnostic-only: kinematic carrier, box pose following, no dynamic
robot balance, no contact grasping, no learned policy.

Validated sweep:

```text
experiments/outputs/adaptive_probe_carry_scene_sweeps/adaptive_probe_sweep_20260704_adaptive_direct_sweep1/adaptive_probe_sweep_summary.json
```

Result: 5/5 cases completed, drop cases 0, target threshold hits 5/5 at
0.08 m, minimum support-margin proxy 0.0769 m, strategy counts
`front_carry: 1`, `low_front_carry: 1`, `chest_supported_slow: 3`.

Experimental ANYmal articulation control diagnostic:

```bash
STAMP=YYYYMMDD_anymal_exp_art_smoke \
STEPS=180 DEVICE=cpu RENDER=0 \
bash scripts/isaac/run_anymal_experimental_articulation_smoke.sh
```

Current 2026-07-04 result is negative. The local ANYmal-C USD loads and exposes
12 DOFs, but the physics tensor entity remains invalid after warmup, so joint
state reads and position-target validation fail. Do not treat this as walking,
balancing, or carrying evidence, and do not wait on it before advancing the
direct Isaac task scaffold.

Standalone Isaac Sim core-World articulation diagnostic:

```bash
STAMP=YYYYMMDD_core_world_quad_smoke \
STEPS=240 PAYLOAD_MASS=4.0 TARGET_X=0.8 DEVICE=cpu RENDER=0 \
bash scripts/isaac/run_core_world_dynamic_quadruped_carry_scene.sh
```

This was added as a control-path repair attempt after the USD drive-target
route failed to produce travel and the IsaacLab-backed `SingleArticulation`
path hit `PhysxManager` compatibility errors. It avoids IsaacLab
`SimulationContext` and tensor APIs, builds the same kind of fixed-payload
articulated carrier directly in Isaac Sim core `World`, and tries
`SingleArticulation.apply_action()`. Current 2026-07-04 result is partial but
not a carry success: `20260704_core_world_quad_payload_shim_diag7` initialized
the articulation and exposed 8 DOFs, with joint motion responding to commands,
but torso and payload travel stayed 0.0 m; later steps produced non-finite
PhysX bounds and NaN joint states. Later runtime-pose diagnostics showed the
custom articulation can be moved with root assists, but velocity assist is
unstable and pose assist is only a scaffold. Do not report this route as
walking, balancing, or unknown-box carrying.

Dynamic USD/PhysX quadruped fixed-payload route:

```bash
STAMP=YYYYMMDD_usd_dynamic_quad_smoke \
STEPS=900 PAYLOAD_MASS=4.0 TARGET_X=1.0 DEVICE=cuda:0 RENDER=0 \
bash scripts/isaac/run_usd_dynamic_quadruped_carry_scene.sh
```

This is the current direct dynamic Isaac attempt. It avoids the failing
IsaacLab tensor APIs and builds the robot from USD rigid bodies, revolute
joints, fixed joints, and USD Physics drive target attributes. The first target
is a dynamically simulated quadruped walking while carrying a physical box
payload fixed to its torso. It is not unknown-object grasping, active
free-contact carry, or learned control. Do not report it as success until a
compute-node smoke proves walking, balance, carried-box travel, no drops, and
no falls.

Current 2026-07-04 status: implemented but not yet walking. GPU articulation
root smoke hit PhysX direct-GPU `setDriveTarget()` errors and travel stayed
0. Disabling articulation root on GPU and CPU removed that error but still
produced 0 travel. The `CONTROL_MODE=core_articulation` path failed before
rollout because deprecated `SingleArticulation` is incompatible with the
current IsaacLab `PhysxManager` context. A later CPU + articulation-root +
USD-drive smoke completed 300/300 steps but still had torso/box travel 0.0.
The next step is to change runtime articulation-control API, not to tune gait
amplitudes.

Adaptive active-probing carry scaffold:

```bash
STAMP=20260704_adaptive_probe_carry_scene_smoke2_clean \
STEPS=300 BOX_MASS=8.0 BOX_SIZE_X=0.58 BOX_SIZE_Y=0.38 BOX_SIZE_Z=0.36 \
BOX_COM_X=0.04 ROBOT_HEIGHT=1.45 ROBOT_MASS=52.0 ARM_LENGTH=0.58 \
MAX_PAYLOAD=16.0 TARGET_X=2.15 DEVICE=cuda:0 RENDER=0 \
bash scripts/isaac/run_adaptive_probe_carry_scene.sh
```

This is now the fastest Isaac path for the actual research question. It builds
approach, probing, posture adjustment, lift, and carry phases; estimates load
from micro-lift/nudge proxy signals; chooses a carry posture from morphology
and load; and logs belief, support margin proxy, effort proxy, drops, target
distance, and strategy. It is diagnostic-only: the carrier is kinematic and the
box pose is followed after the decision.

Validated adaptive smokes:

```text
experiments/outputs/adaptive_probe_carry_scene/20260704_adaptive_probe_carry_scene_smoke2_clean/adaptive_probe_carry_scene_summary.json
experiments/outputs/adaptive_probe_carry_scene/20260704_adaptive_probe_carry_scene_smoke3_chest/adaptive_probe_carry_scene_summary.json
```

Results: the 8 kg default case selected `low_front_carry` and completed
300/300 steps with drop 0; the short-arm 11 kg larger-box case selected
`chest_supported_slow` and completed 260/260 steps with drop 0. These results
validate scene/plumbing and morphology-dependent posture selection only, not
dynamic humanoid balance, grasping, true contact carrying, or video-conditioned
RL.

Direct carrying-task scene smoke:

```bash
DEVICE=cuda:0 RENDER=0 STEPS=240 BOX_MASS=6.0 TARGET_X=2.2 \
STAMP=20260704_direct_carry_task_scene_smoke4 \
bash scripts/isaac/run_direct_carry_task_scene.sh
```

This is the current fastest Isaac path for task construction. It creates a
kinematic humanoid proxy with approach, probe, lift, and carry phases, a massed
box, a target marker, and CSV/summary metrics for box travel, drop events, and
support-margin proxy. It is diagnostic-only: it is not learned balance,
grasping, contact-rich carrying, or autonomous posture selection.

Current validated smoke:

```text
experiments/outputs/direct_carry_task_scene/20260704_direct_carry_task_scene_smoke4/direct_carry_task_scene_summary.json
```

Result: 240/240 steps, final carry phase, final box-to-target distance
`0.0348 m`, box drop events `0`, minimum support-margin proxy `0.111 m`.

Minimal direct-scene smoke:

```bash
DEVICE=cpu SKIP_ROBOT=1 RENDER=0 STEPS=120 \
OUTPUT_DIR=/public/home/yanhongru/Curiosity/experiments/outputs/minimal_carry_scene/smoke_YYYYMMDD \
bash scripts/isaac/run_minimal_carry_scene.sh
```

This must run inside a Curiosity-owned `tmux` session with a persistent Slurm
compute allocation. It refuses to run on `mgmtserver*`.

The minimal scene creates:

- Isaac/PhysX primitive floor;
- visual target marker;
- dynamic rigid carry box with configurable mass and size;
- optional G1 articulation after the box-only scene is verified;
- CSV state logging at `minimal_carry_scene_state.csv`.

`SKIP_ROBOT=1` is intentionally a scaffold and smoke test. It validates Isaac
scene construction and box physics only. It is not a carrying policy or success
claim.

Current validated smoke:

```text
/public/home/yanhongru/Curiosity/experiments/outputs/minimal_carry_scene/smoke_20260703_skip_robot_usd_update_120steps/minimal_carry_scene_state.csv
```

In that run the box falls from about `z=0.446` to `z=0.175`, which is the
expected settle height for a `0.35 m` tall box on the floor.

Known limitation:

- `DEVICE=cpu SKIP_ROBOT=1` validates the direct scene and box rigid-body
  dynamics only.
- G1 WBC local assets have been downloaded under the local Isaac asset mirror
  and verified on a compute node with
  `scripts/isaac/check_g1_wbc_local_assets.py`.
- CPU G1 stand smokes reached scene setup and loaded both HomieV2 ONNX files
  but failed before stepping because PhysX tensor state access returned
  `Failed to get DOF positions from backend`, including after the
  Fabric/tensor logging patch.
- The direct scene now defaults to Fabric-enabled tensor state logging. Do not
  spend more time on CPU-only G1 articulation smoke in this cluster
  environment; the next required smoke is
  `DEVICE=cuda:0 SKIP_ROBOT=0 WBC_MODE=stand`.
- The launcher explicitly sets `LC_ALL` and `LANG` defaults to `C.UTF-8`.
  Without an explicit locale, the uv CPython used by the Isaac venv has shown
  intermittent startup failures while importing the standard `encodings`
  package.

## Official Arena Reference Path

Target:

```text
Isaac Lab-Arena Galileo G1 loco-manipulation task
+ Unitree G1 WBC embodiment
+ rigid brown box
+ PhysX simulation
+ official GR00T closed-loop policy checkpoint
+ recorded MP4 camera/viewport evidence
```

The official Arena task is:

```text
galileo_g1_locomanip_pick_and_place
```

The official tutorial describes it as G1 navigating through the Galileo lab
environment, picking up a brown box from a shelf, and placing it into a blue
bin. It uses PhysX at 200 Hz with 50 Hz control.

## Scripts

- `download_arena_g1_official_assets.sh`
  Downloads the official tuned checkpoint, and optionally the generated HDF5
  simulation dataset. This is a download-only script and may run on the login
  node when large downloads are acceptable.

- `run_arena_g1_locomanip_eval.sh`
  Runs the official GR00T server and Arena closed-loop policy evaluation inside
  the prepared Isaac/Arena and GR00T environments. This is the real
  physics-simulation command path and must run only inside a compute
  allocation.

- `build_minimal_carry_scene.py`
  Builds the direct Isaac carry scene and writes pose-state CSV evidence. Use
  `--skip-robot` to avoid G1/Arena imports while validating the basic box scene.
  Use `--wbc-mode stand|walk` for the official Arena HomieV2 WBC smoke.

- `build_direct_carry_task_scene.py`
  Current direct task-scene diagnostic. It avoids the blocked G1 articulation
  tensor path and builds a kinematic humanoid proxy carrying sequence in Isaac:
  approach, probe, lift, carry, target, CSV state log, and summary JSON. Its
  results must not be reported as humanoid locomotion, balance, grasping, or
  learned carrying success.

- `run_direct_carry_task_scene.sh`
  Compute-node launcher for the direct carrying-task scene. Set `STEPS`,
  `BOX_MASS`, `BOX_SIZE_X/Y/Z`, `TARGET_X`, and `RENDER`. It refuses to run on
  login nodes.

- `build_usd_dynamic_quadruped_carry_scene.py`
  Non-tensor dynamic Isaac route. It creates a four-legged robot directly from
  USD/PhysX rigid bodies and joints, attaches a physical box by fixed joint,
  drives the legs through USD Physics drive target attributes, and logs torso
  pose, box pose, travel, tilt, fall, and drop metrics.

- `run_usd_dynamic_quadruped_carry_scene.sh`
  Compute-node launcher for the USD dynamic quadruped route. Key variables:
  `STEPS`, `PAYLOAD_MASS`, `TARGET_X`, `GAIT_FREQUENCY`,
  `HIP_AMPLITUDE_DEG`, `KNEE_AMPLITUDE_DEG`, `DEVICE`, and `RENDER`.

- `run_anymal_experimental_articulation_smoke.py`
  Official-asset articulation control diagnostic. It loads the local ANYmal-C
  USD asset, wraps it with `isaacsim.core.experimental.prims.Articulation`,
  applies position targets to a few DOFs, and logs measured joint motion. This
  is a control-path smoke only.

- `run_anymal_experimental_articulation_smoke.sh`
  Compute-node launcher for the experimental ANYmal articulation smoke.

- `build_core_world_dynamic_quadruped_carry_scene.py`
  Standalone Isaac Sim core-World control-path diagnostic. It creates a custom
  USD articulated quadruped with a fixed payload, registers it through
  `SingleArticulation`, applies joint-position actions, and logs joint motion,
  torso travel, box travel, falls, and drops. This is not a learned carrying
  method.

- `run_core_world_dynamic_quadruped_carry_scene.sh`
  Compute-node launcher for the standalone core-World diagnostic. Key
  variables: `STEPS`, `PAYLOAD_MASS`, `TARGET_X`, `GAIT_FREQUENCY`,
  `HIP_AMPLITUDE_DEG`, `KNEE_AMPLITUDE_DEG`, `DEVICE`, and `RENDER`.

- `build_core_world_simapp_quasistatic_walker_carry.py`
  Current non-tensor Isaac bridge scene for walking-support carrying. It wraps
  the verified dynamic fixed-payload core-World route in a quasi-static
  four-foot creep gait with support-margin and target-hold metrics. It is not
  final robot locomotion evidence.

- `run_core_world_simapp_quasistatic_walker_carry.sh`
  Compute-node launcher for the quasi-static walker carry diagnostic. Key
  variables: `STEPS`, `TARGET_X`, `PAYLOAD_MASS`, `PAYLOAD_COM_X`,
  `ROBOT_MASS`, `ROBOT_HEIGHT`, `ARM_LENGTH`, `MAX_PAYLOAD`, `BASE_SPEED`,
  `GAIT_FREQUENCY`, and `DEVICE`.

- `build_adaptive_probe_carry_scene.py`
  Current adaptive task-scene scaffold. It avoids external models and directly
  represents active probing, load belief, morphology-aware posture selection,
  and carrying metrics in Isaac. It must not be reported as dynamic robot
  locomotion, learned control, or true contact-box carrying.

- `run_adaptive_probe_carry_scene.sh`
  Compute-node launcher for the adaptive scaffold. Key variables:
  `BOX_MASS`, `BOX_SIZE_X/Y/Z`, `BOX_COM_X/Y/Z`, `ROBOT_HEIGHT`,
  `ROBOT_MASS`, `ARM_LENGTH`, `MAX_PAYLOAD`, `TARGET_X`, `DEVICE`, and
  `RENDER`.

- `run_adaptive_probe_carry_sweep.sh`
  Compute-node launcher for a 5-case adaptive scaffold sweep over box and robot
  parameters. It writes per-case outputs and an aggregate summary under
  `experiments/outputs/adaptive_probe_carry_scene_sweeps/`.

- `aggregate_adaptive_probe_sweep.py`
  Lightweight summary aggregator for adaptive sweep case outputs. It records
  completion, drop cases, target-threshold hits, support-margin proxy, and
  strategy counts.

- `build_velocity_controlled_dynamic_carry_scene.py`
  Dynamic rigid-body carry control-path diagnostic. It creates a dynamic torso,
  dynamic fixed-joint payload, and visual walking legs. `CONTROL_MODE=velocity_attr`
  has been tested on CPU and GPU and is negative: travel remains 0.0.
  `CONTROL_MODE=physx_force` has also been tested on CPU, GPU, and direct-step
  CPU paths and is negative: travel remains 0.0. GPU force mode also hits
  PhysX direct-GPU `addForce()`/`addTorque()` restrictions.

- `run_velocity_controlled_dynamic_carry_scene.sh`
  Compute-node launcher for the dynamic rigid-body diagnostic. Key variables:
  `CONTROL_MODE`, `STEPS`, `PAYLOAD_MASS`, `TARGET_X`, `TARGET_SPEED`,
  `TARGET_HEIGHT`, `DEVICE`, and `RENDER`.

- `build_physx_force_cube_smoke.py`
  Bare non-tensor `CuboidCfg.func` force/fall isolation. CPU smokes with
  direct PhysX stepping and normal `sim.step()` both showed 0 travel and no
  gravity drop, so this route is not evidence of active rigid-body dynamics.

- `build_physx_force_rigidobject_cube_smoke.py`
  `RigidObjectCfg` force/fall isolation. USD-only reads stayed fixed; tensor
  root-state reads failed with `Failed to get rigid body transforms from
  backend`, matching the broader RigidObject tensor blocker.

- `build_core_world_dynamic_cube_smoke.py`
  Isaac Sim core `DynamicCuboid` isolation. The default-ground run stalled on
  Nucleus asset-root lookup; the local-ground run progressed to object
  creation but stalled before `world.reset()`. Treat this as a blocked core API
  diagnostic, not a validated dynamic route.

- `run_anymal_payload_carry.py`
  Official ANYmal-C RSL-RL locomotion payload diagnostic. Current manager-based
  PhysX smokes fail before rollout with `Failed to get DOF velocities from
  backend`, matching the broader IsaacLab articulation tensor issue. Treat all
  outputs as diagnostics, not carrying success.

- `build_contact_carry_scene.py`
  Low-level contact diagnostic with kinematic palms and a dynamic box. Smoke1
  was negative: the dynamic box did not move when palms were moved by USD xform
  edits.

- `build_contact_carry_rigid_scene.py`
  Follow-up low-level contact diagnostic that writes palm poses and velocities
  through the RigidObject simulation API. GPU and CPU smokes both failed with
  `Failed to set rigid body transforms in backend`; do not rerun unchanged.

- `run_minimal_carry_scene.sh`
  Compute-node launcher for the direct Isaac carry scene. Set `SKIP_ROBOT=1`
  for box-only smoke, `WBC_MODE=stand` for G1 standing, and
  `ATTACH_BOX=fixed_torso` only for a labeled payload-balance diagnostic.
  `SKIP_EXPLICIT_STATE_RESET=1` is a diagnostic-only switch for isolating
  Articulation/RigidObject state-write failures; it must not be used as
  carrying success evidence. Box pose can be adjusted with
  `BOX_POS_X/Y/Z`; fixed payload joint placement can be adjusted with
  `ATTACH_LOCAL_POS0_X/Y/Z`.

- `run_g1_wbc_smoke_sequence.sh`
  Compute-node-only sequence runner. It runs stand first, then walk, then a
  fixed-torso payload balance diagnostic. It stops at the first failure. Set
  `RUN_PAYLOAD=0` to run only stand and walk while debugging locomotion. By
  default it runs `check_carry_smoke_summary.py` after each completed smoke;
  set `CHECK_SUMMARY=0` only when debugging summary generation itself.

- `check_g1_wbc_local_assets.py`
  Compute-node-only check that loads the local G1 URDF/mesh assets and official
  HomieV2 ONNX policies without launching Isaac Sim.

- `check_carry_smoke_summary.py`
  Lightweight post-run checker for `minimal_carry_scene_summary.json`. It only
  checks diagnostic gates such as completed steps, fall events, box-drop events,
  and minimum robot travel distance; it is not a full success verifier.

- `build_proxy_carry_scene.py`
  Diagnostic-only Isaac scaffold that avoids the currently failing G1
  articulation tensor path. It creates a kinematic carrier, pose-follow payload
  box, target marker, CSV state log, and summary JSON. It is useful for keeping
  the carry-scene output skeleton moving, but it is not humanoid walking,
  balancing, grasping, or carrying evidence.

- `build_core_world_simapp_staged_free_box_carry.py`
  Pure Isaac SimulationApp/Core World staged free-box diagnostic. The box starts
  as a free `DynamicCuboid`, then the scene runs probe, staged lift, attach,
  carry-to-target, and target-hold phases while logging CSV and summary JSON.
  `ATTACHMENT_MODE=fixed-joint` is the physical-constraint experiment and is
  currently negative: `diag6` and `diag8` still produced PhysX disjoint-joint
  warnings, snapping, high post-attach relative error, or drop events.
  `ATTACHMENT_MODE=kinematic-pose-lock` is an explicitly labeled task scaffold,
  not a physical grasp. The earlier validated diagnostic
  `20260704_core_world_staged_free_box_diag9_kinematic_pose_lock` completed
  360/360 steps, attached at step 91, reached target hold with final
  box-target distance 0.01455 m, had fall/drop events 0, and passed
  `check_staged_free_box_summary.py` with no disjoint warning. The improved
  `20260704_core_world_staged_free_box_diag10_dynamic_velocity_pose_lock`
  uses `CARRIER_MODE=dynamic-velocity`: the carrier body is a dynamic rigid
  body commanded by velocity, while the attached box is pose-locked as a
  scaffold. It completed 360/360, reached final box-target distance 0.01455 m,
  had body travel 0.20545 m, box travel 0.47545 m, post-attach relative error
  `2.78e-08`, fall/drop events 0, no disjoint warning, and
  `min_support_margin_m=0.13252`. Use `diag10` as the current cleanest direct
  Isaac staged free-box scaffold while replacing the placeholder with real
  contact/constraint grasping and verified articulated locomotion.
  The heavier/short-arm diagnostic
  `20260704_core_world_staged_free_box_diag11_chest_dynamic_pose_lock`
  selected `chest_supported_creep` and also passed: completed 560/560, final
  box-target distance 0.01473 m, body travel 0.22527 m, box travel 0.52527 m,
  post-attach relative error `2.78e-08`, fall/drop events 0, no disjoint
  warning, and `min_support_margin_m=0.12430`. This is evidence of
  morphology/load-dependent posture switching inside the scaffold, not a final
  physical grasping or articulated-locomotion claim.
  `ATTACHMENT_MODE=velocity-servo-grasp` was the first validated non-contact
  attach proxy. In
  `20260704_core_world_staged_free_box_diag12_velocity_servo_grasp`, the box
  remained a dynamic rigid body after staged attach and was controlled by a
  velocity servo instead of direct pose-lock. The diagnostic selected
  `low_front_creep`, completed 420/420, reached final target distance
  0.01456 m, had body travel 0.20545 m, box travel 0.47544 m, final relative
  error `3.67e-06`, peak relative error `3.91e-06`, fall/drop events 0, no
  disjoint warning, and `min_support_margin_m=0.13252`.
  `ATTACHMENT_MODE=contact-proxy-servo` is the validated direct Isaac
  staged free-box servo-proxy baseline. It adds explicit left/right palm and
  chest support proxy geometry plus grip-gap metrics while still using
  dynamic-box velocity servo. In
  `20260704_core_world_staged_free_box_diag13_contact_proxy_servo`, the
  diagnostic selected `low_front_creep`, completed 420/420, attached at
  step 91, reached final target distance 0.01456 m, had body travel
  0.20545 m, box travel 0.47544 m, final relative error `3.67e-06`, peak
  relative error `3.91e-06`, contact-proxy grip gap `3.67e-06` m, max grip
  gap `3.91e-06` m, fall/drop events 0, no disjoint warning, and
  `min_support_margin_m=0.13252`. This remains a diagnostic bridge, not
  physical contact grasping or verified articulated locomotion.
  `ATTACHMENT_MODE=dynamic-contact-proxy` is the current strongest staged
  free-box diagnostic. It keeps the palm/chest/shelf proxies as dynamic rigid
  bodies and does not directly velocity-servo the box after attach. Negative
  control `20260704_core_world_staged_free_box_diag14_dynamic_contact_proxy`
  failed because proxies were active before attach, causing 27 box drop events,
  final target distance 0.4703 m, and peak relative error 0.4172 m. After
  pre-attach standby gating,
  `20260704_core_world_staged_free_box_diag15_dynamic_contact_proxy_standby`
  passed: completed 420/420, final target distance 0.01490 m, body travel
  0.25052 m, box travel 0.48505 m, final relative error 0.06311 m, peak
  relative error 0.06348 m, max contact-proxy grip gap 0.06818 m, fall/drop
  events 0, no disjoint warning, and `min_support_margin_m=0.13252`. This is
  not final robot carrying success because lift is staged, proxies are
  velocity commanded, and locomotion is not an articulated gait.

- `check_staged_free_box_summary.py`
  Lightweight JSON/log checker for staged free-box diagnostics. It can require
  an attach step, cap final target distance, cap post-attach relative error,
  reject fall/drop events, and optionally reject PhysX disjoint fixed-joint
  warnings. It can also require minimum body travel, box travel, and support
  margin. It only reads text/JSON and is safe on the login node.
  For multi-posture diagnostics, use `--expect-strategy`,
  `--expect-attachment-mode`, and `--expect-carrier-mode` so a run cannot pass
  by reaching the target with the wrong carrying posture or scaffold mode.

- `build_core_world_dynamic_quadruped_carry_scene.py` and
  `run_core_world_dynamic_quadruped_carry_scene.sh`
  Core World articulated-carrier migration path. It creates a custom USD
  quadruped with revolute hip/knee joints controlled through
  `SingleArticulation.apply_action()`, a staged free dynamic box, and optional
  palm/chest/shelf/front-stop contact proxies. This is the direct Isaac path
  for replacing the velocity-commanded body carrier with an articulated
  foot-contact carrier. The current passing diagnostic is
  `20260705_core_world_dynamic_quad_diag39b_proxy_preplaced`: 760/760 steps,
  attach step 90, `target_hold_latched=True`, `target_hold_steps=26`, torso
  travel 0.33040 m, box travel 0.40108 m, final target distance 0.07327 m,
  final/peak relative error 0.10293/0.10440 m, final/peak contact-proxy gap
  0.08576/0.08787 m, `max_joint_motion_rad=0.87277`, fall/drop events 0, no
  disjoint warning, and control errors 0. This is still an
  articulated-scaffold diagnostic, not final robot carrying: base pose assist
  and staged proxy pre-placement are still active.
  Follow-up root-assist reduction diagnostics are negative as final locomotion
  evidence. `diag40` root velocity assist fell; `diag41` upright velocity
  reduced roll/pitch but still fell by height; `diag42`/`diag43` avoided
  falls and drops with no root pose writes but could not reach the target;
  `diag44b` post-step velocity writes did not improve over `diag43`; direct
  Python `diag45e`/`diag46b` showed `base_x_command_scale=-1.0` is not a
  solution. Use these runs to justify moving to a foot/support-driven
  controller instead of continuing to tune root velocity writes.
  The first foot/support-drive diagnostic is also negative. `SUPPORT_DRIVE=1`
  adds labeled support pads and the checker can now require zero root pose,
  root velocity, and root angular-velocity writes. `diag47b`
  (`20260705_core_world_dynamic_quad_diag47b_support_drive_dynamic_pads_no_root`)
  completed 760/760 with zero root writes, staged free box, and contact
  proxies, but failed with 70 fall events, 53 box drop events, no target hold,
  final target distance 4.11404 m, max tilt 3.19234 rad, and late non-finite
  PhysX state. `diag48`
  (`20260705_core_world_dynamic_quad_diag48_stand_fixed_payload_no_root`)
  reduced the problem to fixed payload, zero target speed, zero gait
  amplitude, no support drive, and zero root writes; it still failed with 20
  fall events and max tilt 2.81150 rad. The next required diagnostic is
  no-root stand/balance before any more long-distance staged-carry tuning.
  Follow-up no-root stand diagnostics `diag49`-`diag53` added explicit neutral
  hip/knee targets, stance/foot morphology parameters, contact friction, and
  hip/knee PD parameters. All still failed. `diag49` proved neutral targets
  alone do not fix standing; `diag50` showed wider feet delay but do not
  prevent falling under 4 kg fixed payload; `diag51` showed a 0.5 kg payload
  is easier but still unstable; `diag52` showed very high friction/PD reduces
  fall count but creates non-finite PhysX state; `diag53` showed moderate
  friction/PD regresses. Treat the current custom two-DOF vertical-leg carrier
  as an insufficient stand/balance base until it is redesigned or replaced
  with a controller-backed robot.
  Known launcher caveat: recent `BASE_X_COMMAND_SCALE` env runs did not
  propagate the value into Python summary, while direct Python invocation with
  `--base-x-command-scale -1.0` did. Use direct Python commands for that knob
  until the launcher path is repaired.

- `check_dynamic_quadruped_carry_summary.py`
  Lightweight JSON/log checker for the custom dynamic quadruped diagnostics.
  It can require staged attach, contact proxies, no disjoint warnings, fall and
  drop caps, target-hold steps, target distance, relative error, contact-proxy
  gap, torso/box travel, tilt, measured joint motion, and control-error caps.
  It also supports strict root-write caps:
  `--max-root-pose-writes`, `--max-root-velocity-writes`, and
  `--max-root-angular-velocity-writes`.
  It also reports the body-aware hold fields:
  `target_body_x_m`, `target_body_margin_m`, `min_hold_torso_travel_m`, and
  `target_hold_body_ready`.

## Cluster Rule

Do not run simulation, rendering, model loading, or evaluation on
`mgmtserver02` or any login node. Run `run_arena_g1_locomanip_eval.sh` inside a
Curiosity-owned `tmux` session with a persistent `srun` or `salloc` compute
allocation.

## Required Prepared Environment

The scripts assume the official Arena and GR00T environments are already
installed. They do not install dependencies, build Docker images, create venvs,
or solve packages.

Pinned official code prepared under:

- `external/IsaacLab-Arena`
- `external/IsaacLab-Arena/submodules/IsaacLab` at `55df2c3`
- `external/IsaacLab-Arena/submodules/Isaac-GR00T` at `e29d8fc`

Prepared environments:

- Isaac/Arena: `/public/home/yanhongru/envs/isaac_arena_py312`
- GR00T server: `/public/home/yanhongru/envs/gr00t_n16_py310`

Prepared official inference checkpoint:

```text
/public/home/yanhongru/models/isaaclab_arena/locomanipulation_tutorial/checkpoint-20000
```

Prepared local USD mirror for compute nodes that cannot open the public S3
asset URLs directly:

```text
/public/home/yanhongru/isaac_asset_mirror/Assets/Isaac/6.0/galileo_locomanip.usd
/public/home/yanhongru/isaac_asset_mirror/Assets/Isaac/6.0/brown_box.usd
/public/home/yanhongru/isaac_asset_mirror/Assets/Isaac/6.0/blue_sorting_bin.usd
/public/home/yanhongru/isaac_asset_mirror/Assets/Isaac/6.0/g1_29dof_with_hand_rev_1_0.usd
/public/home/yanhongru/isaac_asset_mirror/Assets/Isaac/6.0/Isaac/IsaacLab/Arena/wbc_policy/
```

Current diagnostic status:

- Official GR00T checkpoint loads.
- Arena policy runner reaches scene creation with local USD paths.
- `omni.physics.tensors.impl.api` import mismatch is patched locally in the
  Omniverse extension cache as a compatibility shim.
- No MP4 success evidence exists yet.
- Latest blocker: official Galileo scene PhysX mesh cooking stalls on shelf
  collision meshes after repeated CPU fallback warnings.
- Direct G1 CPU and GPU smokes currently fail before stepping with
  `Failed to get DOF positions from backend`, including after
  `InteractiveScene`, `SKIP_EXPLICIT_STATE_RESET=1`, and
  `DISABLE_USD_PHYSICS_UPDATES=1` diagnostics. This is a G1/IsaacLab tensor
  backend blocker, not a WBC success or failure result.
- Current pivot: use the adaptive active-probing Isaac scaffold as the main
  task-construction path while the G1 articulation tensor path is repaired or
  replaced with a better official Arena entry path.
- Current staged free-box status: fixed-joint attach is a negative result;
  kinematic pose-lock passes only as a labeled task scaffold. The current
  direct Isaac path is dynamic-contact-proxy with palm/chest/shelf proxies plus
  a front-stop proxy; the box is not directly velocity-servoed after attach.
  Latest `diag29`-`diag35` work added nonpenetrating carry geometry, lifted
  carry pose, tunable contact-proxy gain/speed, and a nonpenetrating approach
  trigger based on `box_x - carry_x`. Current strongest scaffold:
  `20260704_core_world_staged_free_box_diag34_nonpenetrating_target_hold_1350`
  passed the checker for `low_front_creep` with 1350/1350 steps, attach step
  340, `target_hold_steps=97`, body travel 0.49526 m, box travel 0.34040 m,
  final target distance 0.05999 m, final/peak relative error 0.05155 m,
  final/peak contact-proxy gap 0.04926/0.04931 m, fall/drop events 0, and no
  disjoint warning. Strategy-diversity scaffold
  `20260704_core_world_staged_free_box_diag35_chest_supported_nonpenetrating`
  passed for heavy `chest_supported_creep` with 1600/1600 steps, attach step
  339, `target_hold_steps=295`, final target distance 0.05992 m, final/peak
  relative error 0.04152/0.04167 m, final/peak contact-proxy gap
  0.04456/0.04667 m, fall/drop events 0, and no disjoint warning. These are
  still not final physical robot carrying: the carrier is still a
  velocity-commanded dynamic body with support proxies, not an articulated
  walking controller.
- Do not pursue `BODY_VERTICAL_MODE=height-lock` unchanged. `diag28` kept
  body-z deviation at 0 but broke approach and never attached, so it is too
  artificial as an active path. It may only be referenced as a negative
  scaffold diagnostic.
- Preserve-z staged free-box checks are negative as balance evidence. `diag20`
  corrected support-surface height and reached the target without fall/drop,
  but still failed strict gates with `target_hold_steps=0`,
  `max_body_z_deviation_m=0.47210`, final relative error 0.28226 m, and max
  contact-proxy grip gap 0.19360 m. Do not keep tuning the staged support
  surface as if it were robot gait; move the task interface toward articulated
  carrier scaffolds and then real foot/contact controllers.
