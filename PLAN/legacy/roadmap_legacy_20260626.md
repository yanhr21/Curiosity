# Roadmap

## Phase 0: Repository and baseline audit

Goal: establish official dependencies and identify blockers before writing project code.

Tasks:

1. Clone official Newton repository into `external/`.
2. Clone official T-Rex repository into `external/`.
3. Check whether official T-Rex checkpoints are downloadable and what tensor schema they expect.
4. Check whether Taccel or another tactile simulation layer is needed for dense tactile maps.
5. Write `docs/dependency_audit.md` with exact commit hashes, checkpoint URLs, expected sensor schema, and blockers.

Exit criteria:

- official repo URLs and commits recorded;
- official checkpoint availability recorded;
- clear decision on whether Newton contact outputs are enough or Taccel-like tactile simulation is required.

## Phase 1: Newton scene specification

Goal: define a minimal but contact-rich manipulation world.

Tasks:

1. Define a tabletop scene with cups and distractor objects.
2. Specify object parameters: shape, mass, friction, restitution, handle geometry, fragility, and hazard tags.
3. Choose robot embodiment closest to T-Rex compatibility.
4. Define observation schema for vision, proprioception, contact, tactile force, tactile deformation, object state, and action.
5. Write `docs/environment_spec.md`.

Exit criteria:

- object set specified;
- robot and sensor choices specified;
- observation/action schema specified.

## Phase 2: T-Rex adapter design

Goal: map Newton tactile outputs into official T-Rex input format without inventing a replacement model.

Tasks:

1. Inspect official T-Rex tactile tensor schema.
2. Define Newton-to-T-Rex mapping for per-finger wrench, force history, deformation/contact map, and hand state.
3. Identify unsupported fields or embodiment mismatch.
4. Write `docs/trex_adapter_spec.md`.
5. Only after the spec is accepted, implement adapter code under `src/newton_tactile_curiosity/adapters/`.

Exit criteria:

- adapter input/output schema documented;
- incompatibilities documented;
- no hand-written substitute VQ-VAE or T-Rex-like encoder introduced.

## Phase 2.5: Compatibility conversion fast lane

Goal: keep moving despite public raw-vs-converted dataset mismatch, while
preserving provenance labels and official-code constraints.

Decision:

The public `joint58 -> compat62_f6` route is a compatibility route, not a
faithful in-lab EEF or Newton closed-loop route. Dataset mismatch is not a stop
gate for compatibility diagnostics, but every result must carry that label.

Tasks:

1. Use the clean split2/loss20 `checkpoint-0-3200` as the active T-Rex
   compatibility baseline.
2. Keep official Qwen, official T-Rex code, official released checkpoints, and
   official embedded tactile VQ-VAE paths.
3. Permit glue that adapts startup order, cache use, visual masking, logging,
   and schema validation; forbid fake tensors or placeholder models.
4. Treat split2/loss30 as unpromoted until a diagnostic both reaches metrics
   stably and meets utilization requirements.
5. Build a schema-first Newton/Taccel sample exporter that writes real fields
   only and fails on missing T-Rex-required fields.
6. Record every run as compatibility, diagnostic, sanity, or real training;
   never blur these categories.

Exit criteria:

- compatibility-route limitation is recorded in docs and reports;
- at least one real Newton/Taccel candidate sample is validated against the
  T-Rex schema without padding/fake tactile/action tensors;
- any promoted training run has fresh official sanity, one-hour GPU use, and
  utilization evidence.

## Phase 2.6: Sharpa+f_tac source-gap conversion lane

Goal: convert the current `sharpa_ftac_sync_v0` source mismatch into executable
candidate-source work without promoting incomplete data to official T-Rex
schema.

Decision:

The current same-episode package is browser-visible provenance, not a failed
dead end. It remains useful as the baseline for contact-quality improvement,
while schema promotion stays blocked until real synchronized source fields
exist.

Tasks:

1. Preserve the current same-episode candidate package and its
   `blocked_zero_contact_source_limited` contact-quality audit.
2. Implement a bounded f_tac contact-positive sweep that varies only real
   simulator pose/control/contact parameters through official APIs.
3. Run the sweep inside the tmux-held allocation after fresh official Taccel
   sanity and local-venv activation.
4. If contact becomes nonzero, rebuild the same-episode candidate stream and
   rerun package validation, visual inspection, contact-quality audit, and
   strict inventory.
5. Resolve bimanual 62D state/action semantics from official Sharpa/Dexmate
   sources or write an exact-equivalence proposal. Do not pad f_tac 15-DOF or
   Allegro 50D state into 62D.
6. Resolve camera stream semantics by separating real official head/wrist
   streams from diagnostic camera-equivalent views.
7. Run official T-Rex collate/model sanity only after strict inventory passes
   with real source fields.

Exit criteria:

- at least one synchronized candidate stream has nonzero tactile contact or
  deformation evidence;
- strict inventory reports exactly which official fields are still absent;
- no generated exact T-Rex schema keys appear before source closure;
- the next failure, if any, is a concrete source/API issue rather than an
  open-ended dataset mismatch.

## Phase 2.7: Non-gated marker-frontier source conversion lane

Goal: convert validated real marker/deformation evidence into reusable
curiosity source assets while exact T-Rex schema promotion remains blocked.

Decision:

Dataset mismatch is not a reason to stop. It only blocks exact-schema naming.
Current real marker evidence should be preserved as `taccel.marker.*` and used
for scheduling, reward diagnostics, visual replay, and provenance analysis.

Tasks:

1. Maintain the unified source manifest
   `experiments/outputs/curiosity_marker_frontier_source_manifest_v1_20260626.json`.
2. Use the manifest as the source ledger for marker-frontier curiosity
   scheduling, keeping contact magnitude and temporal novelty separate.
3. Add one genuinely new real source dimension before repeating the same
   Tac-Man marker-only pair: `newton.contact.*`, official Taccel calibrated
   force/deform, bimanual state/action, or accepted camera streams.
4. Keep strict T-Rex schema promotion blocked until real bimanual state/action,
   `[10,6]` F6, ten tactile deform streams, and accepted camera streams exist.
5. Record every new source package with visual evidence, sanity check,
   validation JSON, and an explicit namespace.

Exit criteria:

- manifest validation passes with eight marker branch packages and no generated
  T-Rex fields;
- next conversion task targets a new source dimension or a new frontier;
- no exact T-Rex key is created from marker-only evidence.

## Phase 2.8: T-Rex-independent Newton curiosity source lane

Goal: keep the main Newton tactile curiosity work moving even when exact T-Rex
episode schema is unavailable.

Decision:

T-Rex exact input format is a promotion gate, not the project gate. If a source
cannot faithfully satisfy T-Rex fields, use it under explicit provenance names
and continue building Newton/Taccel curiosity evidence. Do not wait on T-Rex
format closure before collecting real object, contact, camera, proprioceptive,
or tactile-marker signals.

Tasks:

1. Treat the inspected Newton cube and pen rollouts as valid `candidate.newton.*`
   curiosity sources.
2. Treat Newton proxy cameras as `newton.camera.*` visual/provenance sources,
   not as T-Rex `observation.images.*`.
3. Treat Taccel marker/deformation outputs as `taccel.marker.*` or
   `taccel.ftac.*` tactile sources, not as calibrated F6.
4. Build multi-object frontier scheduling from real source metrics: object lift,
   object motion, contact count, EEF travel, tactile marker displacement, and
   temporal novelty.
5. Permit generated/scripted simulation episodes as candidate data when they
   are clearly labeled and pass official simulator sanity plus visual checks.
6. Run T-Rex schema inventory as fail-loud reporting only; do not let expected
   inventory failure stop Newton/Taccel source collection.
7. Train and validate a clearly labeled Newton-only curiosity bootstrap over
   validated `candidate.newton.*` transitions when exact T-Rex promotion is
   blocked. The first allowed implementation is
   `newton_icm_curiosity_bootstrap_v0`: offline inverse/forward dynamics over
   inspected Newton cube/pen sources, one GPU-hour minimum, GPU-utilization
   logging, final checkpoint, reward CSV, loss curve, and validator output.
   This is not T-Rex, not VQ-VAE progress, not policy success, and not
   tactile-dominant control.

Exit criteria:

- at least two Newton objects are represented in a validated curiosity frontier;
- each source package has sanity, visual evidence, validation JSON, and a clear
  namespace;
- the Newton-only curiosity bootstrap has either passed final validation or
  recorded a concrete failure with logs/checkpoint state preserved;
- blocked reports name concrete source gaps instead of generic schema mismatch;
- no alternative source is promoted as faithful T-Rex data without strict
  inventory pass.

## Phase 3: Demonstration bootstrap

Goal: provide a minimal action prior so curiosity does not start from pure random motion.

Tasks:

1. Define demonstration types:
   - successful simple cup lift;
   - stable tactile hold;
   - slip recovery;
   - failed over-force grasp;
   - failed under-force grasp;
   - exploratory surface touch/slide.
2. Decide whether demonstrations come from scripted controller, teleoperation, or official datasets.
3. Store synchronized RGB-D, robot action, proprioception, tactile force/deformation, object state, and success/failure label.
4. Write `docs/demonstration_spec.md`.

Exit criteria:

- demonstration schema documented;
- at least one source path chosen;
- failure demonstrations included, not only success demonstrations.

## Phase 4: Curiosity reward design

Goal: define intrinsic rewards that encourage persistent useful exploration.

Tasks:

1. Implement controllable latent prediction target design on paper first.
2. Define learning-progress reward for tactile/object/body latents.
3. Define impact-driven reward for meaningful object/contact changes.
4. Define competence-progress scheduler over subgoals.
5. Define archive/frontier mechanism for returning to promising states after failure.
6. Write `docs/curiosity_reward_spec.md`.

Exit criteria:

- reward terms documented with formulas;
- no raw-video prediction reward as the primary objective;
- no unbounded prediction-error objective;
- failure and safety penalties included.

## Phase 5: Training curriculum

Goal: stage learning from contact discovery to tactile-only manipulation.

Tasks:

1. Contact discovery: touch, press, slide.
2. Stable contact: form and maintain contact patch.
3. Easy lift: simple cup/cylinder.
4. Difficult lift: handles, slippery objects, thin objects, heavier objects.
5. Tactile-only hold and regrasp after initial visual localization.
6. Generalization to unseen shapes, masses, and friction.

Exit criteria:

- curriculum schedule documented;
- tactile-only probability schedule documented;
- success/failure metrics documented.

## Phase 6: Evaluation and ablations

Goal: prove whether tactile curiosity helps.

Required comparisons:

1. Vision-only curiosity.
2. Tactile-only curiosity.
3. Vision+tactile curiosity.
4. No curiosity, sparse reward only.
5. Demonstration-only policy.
6. Demonstration + tactile curiosity.
7. Official T-Rex tactile encoder vs raw force baseline, clearly labeled as baseline.

Metrics:

- stable lift success;
- drop rate;
- excessive-force rate;
- slip recovery rate;
- tactile-only success;
- unseen object generalization;
- sample efficiency.

Exit criteria:

- evaluation table template exists;
- each ablation has a clear hypothesis.
