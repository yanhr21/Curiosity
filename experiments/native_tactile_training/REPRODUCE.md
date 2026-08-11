# Reproduce the no-vision tactile-policy result

## Prerequisites and fixed inputs

Use the same Python 3.11, Isaac Sim 5.1, official CarryBox motion, released
Refiner checkpoint, and retained GPU allocation described by the representation
reproduction guide. Set `CURIOSITY_ISAAC_PYTHON` if needed. Training and
evaluation use motion 45, seed 13011, two environments during training, and
one deterministic environment during frozen evaluation. Output directories
must not exist before launch.

Run every command below from the repository root. The two local-only official
inputs must exist at:

```text
SUGAR/data/CarryBox/data_045/
experiments/sugar_reproduction/outputs/final/official_sugar/baseline/ckpts/refiner_model10000.pt
```

Do not launch a second experiment concurrently. Use
`launch_retained_child.sh` for a long command so that it gets its own recorded
process group and the retained Slurm shell remains available after completion.
The launcher writes the child PID/PGID, log, and exit status; completion of a
child never releases the allocation.

## Scope

This package asks whether native whole-hand tactile changes, and eventually
helps, a SUGAR CarryBox policy without RGB or measured current object state.
It is separate from the completed sensor visualization in
`../native_tactile_representation/whole_hand_carrybox_v3/`:

- the representation package validates the physical sensor and its complete
  CarryBox video;
- this package trains and freezes a policy that can consume the same tactile
  tensor.

The actor input is the official-width `890-D` SUGAR reference-only state plus
the raw tactile history. The reference-only state contains robot state and the
commanded motion plan. It excludes RGB, measured current object pose and
velocity, object mass, friction, contact labels, and contact velocity. Tactile
is serialized as `[hand,history,patch,channel,row,column]` with shape
`[2,4,27,3,20,25]` (`324000` values). The existing serious SUGAR spatial
encoder maps each hand to `128-D`; both embeddings enter the unchanged
official actor before its first hidden layer.

## Frozen endpoint used by the current videos

```text
matched_tactile_64u_seed13011_20260811/model_63.pt
```

It was trained serially against the matched exact-zero arm
`matched_zero_64u_seed13011_20260811/model_63.pt`. Both began from the same
released Refiner weights, seed, physics, optimizer, teacher, and frame-zero
continuous CarryBox route. Only the tactile adapter can change the accepted
actor mapping.

## Reproduce training

Run inside the retained allocation. The two arms must be serial:

```bash
bash scripts/sugar/native_tactile/run_native_tactile_bcppo_training.sh \
  tactile experiments/native_tactile_training/reproduced/tactile 64 2 13011

bash scripts/sugar/native_tactile/run_native_tactile_bcppo_training.sh \
  zero experiments/native_tactile_training/reproduced/zero 64 2 13011
```

Compare the initial and final model states with
`compare_native_tactile_training_endpoints.py`. Frozen numerical evaluation
uses `run_native_tactile_bcppo_evaluation.sh`. Camera-free results and
camera-enabled presentation rollouts are separate cohorts because enabling the
renderer changes GPU simulation numerics and can change the termination step.

## Reproduce the three synchronized policy videos

The single entry point below runs one frozen checkpoint under live tactile,
exact-zero actor tactile, and a fixed permutation of the 27 anatomical patches
within each hand. Rollouts are serial. Each output video shows the actual
CarryBox world state on top and all 54 physical patch fields below. Its header
states what entered the actor; in the zeroed condition the displayed physical
sensor remains visible even though the actor receives exact zeros.

```bash
bash scripts/sugar/native_tactile/run_frozen_tactile_policy_visualizations.sh \
  experiments/native_tactile_training/matched_tactile_64u_seed13011_20260811/model_63.pt \
  experiments/native_tactile_training/reproduced_policy_visualizations
```

Every renderer checks that the completed H.264 decodes from first frame to
last. `summary.json` then verifies the same checkpoint, seed, physical
condition, initial state, reference trajectory, and permutation across all
three rollouts.

## Reproduce the frozen tactile-authority curve

This no-learning diagnostic keeps the checkpoint and physical rollout fixed
while multiplying only the appended tactile-feature columns of `actor.0` by
`0`, `0.25`, `0.5`, `0.75`, or `1.0` after loading:

```bash
bash scripts/sugar/native_tactile/run_native_tactile_authority_curve.sh \
  experiments/native_tactile_training/matched_tactile_64u_seed13011_20260811/model_63.pt \
  experiments/native_tactile_training/reproduced_authority_curve
```

The summary requires exact initial-state and pre-contact-action equality and
requires scale zero to make live and zeroed tactile actions bitwise equal on
every visited state.

## Reproduce the bounded 16-update stability result

Run the two arms serially:

```bash
bash scripts/sugar/native_tactile/run_native_tactile_bcppo_training.sh \
  bounded_tactile experiments/native_tactile_training/reproduced/bounded_tactile 16 2 13011

bash scripts/sugar/native_tactile/run_native_tactile_bcppo_training.sh \
  bounded_zero experiments/native_tactile_training/reproduced/bounded_zero 16 2 13011
```

Evaluate the two `model_15.pt` checkpoints with
`run_native_tactile_bcppo_evaluation.sh`, summarize the pair with
`summarize_native_tactile_frozen_pair.py`, and run
`summarize_native_tactile_contact_teacher_alignment.py` on the tactile NPZ.
The retained result is negative for stable correction: lift increases, but
reward, duration, and same-state contact teacher alignment worsen.

The next action-residual route uses `residual_tactile` and `residual_zero` in
the same training and evaluation entry points. It keeps the hidden `0.15` cap,
restores official all-sample distillation, and limits every normalized tactile
action residual to `0.1`. Its 16-update gate is mixed: common-horizon lift and
tracking improve, while reward and termination worsen. Use
`summarize_native_tactile_common_horizon.py` after frozen evaluation; do not
compare cumulative rewards across unequal rollout lengths. The next admitted
run was the fresh serial 64-update `residual_tactile`/`residual_zero` pair. It
is complete and negative at this seed. Its live result remains separate from
the static audit in `action_residual_fusion_static_audit_20260811/report.json`.

For a complete fresh 64-update action-residual pair, run exactly one arm at a
time:

```bash
bash scripts/sugar/native_tactile/run_native_tactile_bcppo_training.sh \
  residual_tactile experiments/native_tactile_training/reproduced/residual_tactile 64 2 13011

bash scripts/sugar/native_tactile/run_native_tactile_bcppo_training.sh \
  residual_zero experiments/native_tactile_training/reproduced/residual_zero 64 2 13011
```

Then compare the endpoints and run frozen rollouts, still serially:

```bash
python scripts/sugar/native_tactile/compare_native_tactile_training_endpoints.py \
  --tactile experiments/native_tactile_training/reproduced/residual_tactile \
  --zero experiments/native_tactile_training/reproduced/residual_zero \
  --output experiments/native_tactile_training/reproduced/endpoint_comparison.json

bash scripts/sugar/native_tactile/run_native_tactile_bcppo_evaluation.sh \
  residual_tactile \
  experiments/native_tactile_training/reproduced/residual_tactile/model_63.pt \
  experiments/native_tactile_training/reproduced/residual_tactile_eval.json

bash scripts/sugar/native_tactile/run_native_tactile_bcppo_evaluation.sh \
  residual_zero \
  experiments/native_tactile_training/reproduced/residual_zero/model_63.pt \
  experiments/native_tactile_training/reproduced/residual_zero_eval.json
```

The evaluator writes a same-stem NPZ beside each JSON. Produce the matched,
common-horizon, and supported-contact summaries with:

```bash
python scripts/sugar/native_tactile/summarize_native_tactile_frozen_pair.py \
  --tactile experiments/native_tactile_training/reproduced/residual_tactile_eval.json \
  --zero experiments/native_tactile_training/reproduced/residual_zero_eval.json \
  --output experiments/native_tactile_training/reproduced/frozen_pair.json

python scripts/sugar/native_tactile/summarize_native_tactile_common_horizon.py \
  --tactile experiments/native_tactile_training/reproduced/residual_tactile_eval.npz \
  --zero experiments/native_tactile_training/reproduced/residual_zero_eval.npz \
  --output experiments/native_tactile_training/reproduced/common_horizon.json

python scripts/sugar/native_tactile/summarize_native_tactile_contact_teacher_alignment.py \
  --trace experiments/native_tactile_training/reproduced/residual_tactile_eval.npz \
  --output experiments/native_tactile_training/reproduced/contact_teacher_alignment.json
```

The endpoint comparison must show identical `model_prelearn.pt` tensors before
the result is interpreted. The pair summary must pass its matched-arm, seed,
physics, reference, and disabled-event checks. Report the common-horizon task
metrics and contact-supported teacher alignment separately; neither one seed
nor a changed action is sufficient evidence that tactile generally helps.

To reproduce the final human-review pair, evaluate both 64-update checkpoints
again with `--record_bundle ABSOLUTE_BUNDLE_PATH --enable_cameras`, render each
bundle with `render_native_tactile_policy_rollout.py`, passing both bundle roots
as repeated `--scale-bundle-root` arguments, and compose them with:

```bash
python scripts/sugar/native_tactile/compose_native_tactile_policy_pair.py \
  --left tactile_trained_world_and_bilateral_tactile.mp4 \
  --right zero_trained_world_and_bilateral_tactile.mp4 \
  --output tactile_trained_vs_zero_trained_side_by_side.mp4 \
  --fps 50
```

The retained pair must report 348 fully decoded frames at `2560 x 720`. The
record-bundle path must be absolute because the evaluator launches from the
`SUGAR/` directory.

## Reproduce the held-out teacher-residual gate

The failed 64-update PPO route is not extended. The next experiment first asks
whether the same serious spatial adapter contains useful information at all.
One serial entry point collects five continuous frame-zero physical conditions,
trains only the eight declared adapter tensors for 400 contact minibatches,
selects without reading either test condition, and writes the selected model,
per-row predictions, and result:

```bash
bash scripts/sugar/native_tactile/run_native_tactile_bcppo_training.sh \
  residual_zero \
  experiments/native_tactile_training/reproduced/warmstart_export \
  1 2 13011

bash scripts/sugar/native_tactile/run_native_tactile_teacher_residual_gate.sh \
  experiments/native_tactile_training/reproduced/teacher_residual_gate \
  experiments/native_tactile_training/reproduced/warmstart_export/model_prelearn.pt
```

The first command is only a reproducible export route for the exact
pre-learning official warm start; the gate consumes `model_prelearn.pt`, not
the one-update endpoint. If that exact pre-learning checkpoint already exists,
pass it directly as the optional second argument and skip the export command.
For an unattended retained-allocation launch, wrap the gate as follows:

```bash
bash scripts/sugar/native_tactile/launch_retained_child.sh \
  --record experiments/native_tactile_training/runtime/reproduce_gate.process \
  --status experiments/native_tactile_training/runtime/reproduce_gate.status \
  --log experiments/native_tactile_training/runtime/reproduce_gate.log \
  --tag reproduce_teacher_residual_gate \
  -- bash scripts/sugar/native_tactile/run_native_tactile_teacher_residual_gate.sh \
    experiments/native_tactile_training/reproduced/teacher_residual_gate \
    experiments/native_tactile_training/reproduced/warmstart_export/model_prelearn.pt
```

The collection policy is always the exact-zero official actor. Live tactile is
recorded pre-action and therefore cannot alter which state enters any split.
Training conditions are `0.5/0.75 kg`, selection is `0.625 kg` with friction
`0.40/0.30`, and untouched tests are `1.0 kg` at default friction and `0.5 kg`
with friction `0.25/0.20`. These are training-task masses; the separate
canonical successful visualization intentionally uses `0.3023376 kg`.

The retained result is
`heldout_contact_residual_gate_v1_20260811/training/report.json`. Live tactile
reduces the combined held-out teacher-action MAE from `0.08335` to `0.06147`
(`26.26%`), improves both individual tests, and loses the advantage when the
27 anatomical patches are permuted. The entry point writes both
`training/report.json` and `independent_audit.json`. To rerun only the
independent reconstruction against an existing result, use:

```bash
python scripts/sugar/native_tactile/audit_native_tactile_teacher_residual_gate.py \
  --result-root \
    experiments/native_tactile_training/heldout_contact_residual_gate_v1_20260811 \
  --output \
    experiments/native_tactile_training/reproduced/teacher_residual_audit.json
```

That audit checks the saved predictions and the initial/selected model tensors;
it passes every check without a hash workflow. The result establishes
held-out predictability, not behavior improvement. Contact in these heavier
closed-loop policy states is much broader than in the `0.3023376 kg` canonical
successful trace, so it must not be presented as the sparse canonical grasp or
as calibrated force.

## Reproduce the frozen behavior gate

Before PPO, compare the one selected checkpoint under live tactile and the
exact-zero/no-read observation on both untouched conditions:

```bash
bash scripts/sugar/native_tactile/run_frozen_teacher_residual_policy_gate.sh \
  experiments/native_tactile_training/heldout_contact_residual_gate_v1_20260811/training/model_best.pt \
  experiments/native_tactile_training/reproduced/teacher_residual_policy_gate
```

The entry point runs all four no-learning rollouts serially and applies the
predeclared reward, tracking, and termination rule. The retained result is
negative. On held-out `1.0 kg`, live tactile improves mean position error by
`0.000713 m` but loses `2.19394` reward and terminates eight steps earlier. On
held-out low friction, it gains `0.35444` reward with equal duration but has
`0.000745 m` worse mean position error and `0.03916 m` less lift. Therefore
this supervised initialization does not proceed to PPO.

The corresponding human-review evidence is also one serial command. It reruns
the same two untouched conditions with cameras, displays the actual CarryBox
world above all 54 physical patch maps, labels whether live tactile or exact
zero/no-read enters the actor, and writes one side-by-side H.264 per condition:

```bash
bash scripts/sugar/native_tactile/run_teacher_residual_policy_visualizations.sh \
  experiments/native_tactile_training/heldout_contact_residual_gate_v1_20260811/training/model_best.pt \
  experiments/native_tactile_training/reproduced/teacher_residual_policy_videos
```

Expected final files are
`heldout_heavy_1p0kg_live_vs_zero.mp4` and
`heldout_low_friction_0p5kg_live_vs_zero.mp4`, plus the four full-resolution
individual videos and their JSON/NPZ source records. Every H.264 is fully
decoded before the script reports completion. Camera-enabled rollouts are
presentation evidence; the camera-free policy gate supplies the matched
numerical decision because rendering can alter simulator timing.

## Output and interpretation checklist

A valid reproduction has all of the following:

- the exact `324000-D` `[2,4,27,3,20,25]` tactile history and no RGB in the
  actor;
- five distinct physical collection conditions with no held-out test used for
  optimization or selection;
- only the eight declared tactile-adapter tensors changed and the official
  actor base columns unchanged;
- saved selection/test predictions, `training/report.json`, and an independent
  audit that reconstructs them;
- four frozen camera-free behavior rollouts and the predeclared per-condition
  gate report;
- two synchronized live-versus-zero H.264 comparison videos for human review.

The held-out prediction gate answers whether tactile contains information
about the privileged teacher action. The frozen behavior gate separately asks
whether that information improves closed-loop CarryBox behavior. A positive
prediction result must never be reported as policy or task improvement.

## Claim boundary

The frozen dependence experiment can show that the actor reads tactile and
that anatomical patch order matters. A single seed with mixed tracking and
termination outcomes cannot establish that tactile generally improves the
task. CarryBox is the only validated task. KickBox needs its own SDF object,
sensors on the foot/leg or another actual contact body, task binding, and
matched evaluation before it can be called supported.
