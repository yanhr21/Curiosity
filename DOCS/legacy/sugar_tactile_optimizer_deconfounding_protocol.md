# SUGAR tactile optimizer-deconfounding protocol

## Status and activation boundary

This protocol is predeclared on 2026-07-15 while the fresh reference-only
seed-42 branches are still below iteration 700 of 7098 and before any final
checkpoint or closed-loop performance result exists. It does not alter or
restart those running jobs, their source manifests, checkpoints, frozen final
suite, or statistical report.

The active seed-42 triplet must finish and be evaluated exactly as launched, so
its as-implemented outcome is retained rather than hidden. However, the audit
below discovered a cross-role optimizer confound before the final result. The
current triplet therefore cannot by itself support a clean tactile-attribution
claim, even if its already frozen analyzer writes `advantage_proven=true`.
Seeds 43/44 must not replicate that optimizer contract. The next formal branch
must first pass this deconfounding protocol and remains separate from the
current exact-state, reference-only, and latent-contact-dynamics reports.

All claims remain `high-fidelity simulated tactile`. This protocol adds no
physical GelSight calibration or sim-to-real evidence.

## Pre-result evidence

The three serialized `agent.yaml` files are byte-identical (SHA256
`0b39e1fd779cefaaa9169141de95adaf2f8e82ef743d411a7a4c746c0c58e187`) and
declare five learning epochs, four minibatches, initial learning rate `1e-3`,
adaptive schedule, desired KL `0.01`, and gradient clipping `1.0`.

That apparent configuration match does not produce matched optimization. The
installed official RSL-RL PPO creates one Adam over all policy parameters. On
every minibatch, actor KL changes one shared learning rate and writes it to
every optimizer parameter group. The reference-only warm start is configured
after optimizer construction: the accepted actor and action noise are frozen,
the tactile encoder weights and actor tactile-input columns are enabled, and
the full critic remains enabled. Consequently:

- the zero-role actor is exactly fixed, so its near-zero KL drives the shared
  rate to the `1e-2` adaptive ceiling;
- full and pressure-only change their tactile actor paths, so their KL drives
  the same shared rate down or up;
- the critic therefore receives a role-dependent learning rate even though it
  has the same architecture, input semantics, initialization, and serialized
  settings in all roles.

A read-only TensorBoard-prefix audit freezes common steps `0..636`. Its local
report is
`experiments/sugar_reproduction/outputs/CarryBox_20260715_tactile_reference_only_optimizer_dynamics_audit/seed42_prefix_step636.json`
(SHA256
`53ad1320b59990c44da7057e0f933d233b4db3cace054d73e29942163e054636`).
The audit script is
`scripts/sugar/audit_sugar_reference_only_optimizer_dynamics.py` (SHA256
`be464fbade95e7cc5445eb2fd8d7c733d31affca8afe1c16a6a89cb3d205c04c`).
It binds the installed PPO source (SHA256
`deafc8c947eba4df3e91b393869426cdab8d7b71e05974c3734125d2331d7d1c`)
and reports:

| role | LR median | LR rolling-100 mean | maximum value loss |
| --- | ---: | ---: | ---: |
| full | `3.375e-5` | `1.25206e-4` | `41.8297` at step 0 |
| zero | `1.0e-2` | `1.0e-2` | `43226.5703` at step 418 |
| pressure-only | `5.0625e-5` | `1.19053e-4` | `9.96935` at step 0 |

All 637 zero-role learning-rate samples are at `1e-2`. Its other largest value
losses include `3010.5774`, `1507.0386`, `1276.9685`, and `571.1518`; neither
live-tactile role shows comparable spikes in the same prefix. The audit marks
both `optimizer_dynamics_matched` and
`clean_tactile_performance_attribution` false. These are optimizer-contract
facts, not task-performance evidence and not proof that the optimizer alone
causes the early reward difference.

The fresh uninterrupted checkpoints later reached common `model_1000.pt` and
passed the strict actor-freeze audit. That local report is
`experiments/sugar_reproduction/outputs/CarryBox_20260716_tactile_reference_only_policy_weight_audit/fresh_restart_diagnostics/reference_only_seed42_model1000_strict.json`
(SHA256
`1ba27229dd086a99e7dd1ade71b863d13331f6072dedfa574a95550367499c5e`).
It confirms that zero's entire actor remains bitwise exact and full/pressure
change only permitted tactile-path tensors. It also quantifies the optimizer
split in critic state: zero critic relative L2 drift is `11.9670` with maximum
absolute change `23.0055`, versus full/pressure relative drift `1.1525/1.3177`
and maximum changes `1.38895/1.39194`. This remains checkpoint-contract
diagnosis, not mid-run performance evidence.

## Interpretation boundary for the running triplet

The current run still answers a useful, narrower question: how the launched
reference-only tactile method performs against its launched zero and pressure
controls under the original shared adaptive optimizer. Its checkpoint
identity, taxel exposure, action dependence, interventions, and final task
success must still be reported truthfully.

It does not cleanly isolate the tactile modality because role-dependent actor
KL also changes critic optimization. A positive frozen gate is a candidate
as-implemented result, not a project tactile-advantage claim; a negative gate
does not prove the official TacSL signal is unhelpful. Reward, episode length,
failure reduction, or the zero critic's value-loss spikes cannot replace the
predeclared success endpoint.

## Fixed deconfounded optimizer contract

The first clean follow-up keeps the current reference-only environment and
changes only optimizer plumbing. It must use official RSL-RL PPO semantics plus
minimal auditable adapter code; it may not replace PPO, the actor, critic, or
TacSL encoder with a simplified local model.

Use one checkpoint-compatible Adam with two named, disjoint trainable parameter
groups constructed after the tactile finetune gate:

- `critic`: all and only the official-width privileged critic parameters,
  fixed learning rate `1e-3`;
- `tactile_actor`: the seven non-bias spatial encoder weights plus
  `actor.0.weight`, whose existing gradient mask freezes all 890 accepted-SUGAR
  base columns and exposes only the 256 tactile columns, fixed learning rate
  `1e-4`.

Keep the accepted actor tensors, encoder biases, and action noise frozen. Keep
five epochs, four minibatches, PPO clipping, loss coefficients, advantage
calculation, reward, termination, rollout length, gradient-norm limit, and
7098-update/512-environment budget unchanged. Set the adaptive schedule off for
this branch so actor KL cannot rewrite the critic group. The critic rate is the
official configured initial rate; the tactile-actor rate is fixed before the
current final result and is close to the observed live-role mean while bounding
the aggressive first-update transient. There is no learning-rate search in the
formal branch.

The implementation must log both real parameter-group rates separately, plus
approximate actor KL, value loss, surrogate loss, and per-group gradient norms.
The existing single `Loss/learning_rate` scalar is insufficient. Checkpoints
must serialize both named groups and their Adam states without losing standard
RSL-RL resume/evaluation compatibility.

## Isolated implementation status

The optimizer adapter, config, guarded task registration, and training entry
are prepared as isolated new files. They are not referenced by the running
triplet's source manifest or task IDs. The adapter subclasses official RSL-RL
`PPO`, calls `super().update()` exactly once, retains the official rollout,
GAE, losses, minibatches, global clipping, and checkpoint field, and only
rebuilds Adam into the two declared groups. Its post-update probe uses
deterministic `act_inference`, so metric logging does not consume policy-action
RNG state.

An official-`model_10000.pt` structure diagnostic passes all checks. The local
report is
`experiments/sugar_reproduction/outputs/CarryBox_20260715_tactile_reference_only_optimizer_deconfounding_preflight/optimizer_structure_audit.json`
(SHA256
`5f0b070af9da053128d815909f754de76aed8add2271070cfa67a0c8a1ff3512`).
It proves that the two groups are disjoint and exhaustive over trainable
parameters, the declared rates survive an inherited official PPO update, the
eight required LR/loss/KL/gradient metrics are finite, only the permitted 16
tensors change in a direct gradient diagnostic, all frozen tensors and 890
actor base columns remain bitwise exact, and zero tactile still produces the
accepted actor exactly. This synthetic official-update check is optimizer
plumbing evidence only, not simulator training or task performance.

The guarded three-task Gym entry/source audit also passes at
`experiments/sugar_reproduction/outputs/CarryBox_20260715_tactile_reference_only_optimizer_deconfounding_preflight/registration_audit.json`
(SHA256
`f405f098d6ecb33d56c05648fed35d486e65c77995348fbf9af7e92edfcbda5d`).
It verifies fail-closed enablement and the three full/zero/pressure entry points.
Runtime Hydra serialization remains explicitly pending because importing the
IsaacLab config registry requires an Isaac App runtime. That check and a
`1 env x 1 update` official-code smoke diagnostic must run later in a separate
allocation; neither is allowed to compete with the active formal training.
The fail-closed post-smoke auditor is prepared at
`scripts/sugar/audit_sugar_reference_only_optimizer_clean_runtime_smoke.py`.
It requires the serialized custom algorithm/fixed rates, two named checkpoint
optimizer groups, complete finite TensorBoard LR/KL/gradient metrics, exact
base-actor freezing, permitted tactile-path learning, matching runtime/source
manifests, and no simulator error signatures.

An isolated training wrapper is also prepared at
`scripts/sugar/run_official_sugar_carrybox_reference_only_optimizer_clean_refiner.sh`.
It fixes the three new task IDs, official checkpoint/assets, v3 mounts, formal
`512 x 7098` budget, fixed rates, prerequisite reports, runtime metadata, and a
21-file source manifest. Diagnostic mode is limited to 1-8 environments and
1-2 updates with an explicit diagnostic/smoke label. Formal mode fails closed
until the fixed full-role `1 env x 1 update` runtime report exists, passes, and
every source recorded by that smoke still matches current bytes. The prepared
wrapper therefore cannot start the next experiment prematurely or reuse a
stale integration check.

## Admission gates before formal training

Compute-node diagnostics must establish all of the following without using
task success to select an optimizer:

1. full/zero/pressure pre-update model states are bitwise identical and bind
   the same official `model_10000.pt`;
2. parameter-group name sets are pairwise identical, disjoint, exhaustive over
   trainable parameters, and exclude every frozen parameter;
3. every recorded critic rate is exactly `1e-3` and every tactile-actor rate is
   exactly `1e-4` in all roles;
4. zero taxels preserve the entire accepted actor exactly after updates; base
   actor columns and all other frozen actor tensors remain bitwise exact;
5. full/pressure updates touch only the seven encoder weights and tactile input
   columns, and all losses/gradients remain finite;
6. live official TacSL fields retain spatial pressure and signed two-axis shear
   exposure, and model-1000 evaluation shows nonzero same-policy action
   dependence without using success as a tuning gate;
7. all source, optimizer, environment, checkpoint, and runtime identities are
   serialized and fail closed on mismatch.

Reduced runs remain diagnostics using official SUGAR/TacSL code. They are not
training progress or advantage evidence.

## Formal comparison and branch order

The clean optimizer branch repeats matched full, zero, and pressure-only
training from the bitwise-identical official warm start under the current
reference-only environment distribution. It retains the frozen six-condition,
seven-intervention evaluation and the two success-primary paired-bootstrap
gates: full-live versus matched zero, and full-policy live versus its zero
intervention. No current-run episodes or reports may be pooled into it.

Only a passing clean seed42 may activate clean seeds 43/44. A research-level
advantage still requires all three independent matched training seeds to pass
and positive seed-bootstrap lower bounds for both success comparisons.

If the clean current-distribution branch remains negative despite valid taxel
exposure and action dependence, activate the separately predeclared
`DOCS/sugar_tactile_latent_contact_dynamics_protocol.md` using this same
deconfounded optimizer contract. This ordering separates optimizer stability
from the hidden-dynamics hypothesis and prevents result-selected joint tuning.
