# Corrected Causal Posture Grid V2: Negative Behavior Result

Date: 2026-07-27

## Decision

The corrected V2 matched training contract is causally valid, but its frozen
behavior is rejected. This is not a completed strategy-discovery result.

The complete grid contains:

- no-demo and frozen internal-reward arms;
- deterministic-mean and fixed-Gaussian action modes;
- policy updates 1, 131, 261, and 521;
- 27 exact mass/friction/COM profiles per cell; and
- 432 total rollouts.

Every rollout terminates in `unsafe_fall`. There are zero successes, zero
recoveries, zero alternative-strategy candidates, and zero render candidates.
The pair audit passes its structural/causal checks only and explicitly sets
`stable_cross_seed_success_proven=false` and
`completion_authorized=false`.

## Bound evidence

- Frozen grid config SHA256:
  `5d17d3dcc85c5b1f9be58208791216e4e938122807d537c43a02f2e7861dd48a`
- Pair audit SHA256:
  `a6d0d73e3ff75a1b49cb4f51522a84a1bec40890de0ed60edf823347c4025c17`
- Final merge-audit record SHA256:
  `36f9919f72b689a3e74c464f9cb6a324507cae47d7d00f503ae7bd2e27170326`

All four merged producers and independent audits pass. They independently
reconstruct the exact causal previous action, real TacSL frames 100--103,
direct spatial pressure and signed two-axis shear, slip/failure runtime,
teacher/residual action mapping, all 27 physics tuples, terminal pre-reset
state, reward ledgers, and the absence of mass/friction/COM/success/reward
oracles from actor and original-ICM inputs.

The first server02 attempt, job `202300`, lost its Vulkan device during scene
construction and produced no result or trace. It remains quarantined as
infrastructure-only evidence. Jobs `202376`--`202381` reran the unchanged
frozen config while excluding server01/server02/server52 and produced the
accepted complete negative grid.

## Diagnosis

The failure is not caused only by a late bad checkpoint:

- update-1 deterministic residual mean L2 is only about `0.0036`, yet all 54
  no-demo/demo deterministic rollouts are already unsafe;
- in the nominal mass/friction/center-COM profile, tactile-confirmed failure
  closes at step 3 for every deterministic checkpoint, bilateral contact lasts
  only 5--7 frames, and no later direct-TacSL recontact occurs;
- the frozen official motion-45 source itself has bilateral direct-palm TacSL
  only on source indices 102--107 and is not a successful terminal rollout;
  therefore the current contact seed is a transient failure seed, not a
  positive stable nominal foundation; and
- policy residual means then drift strongly. Mean deterministic residual L2
  grows from `0.0036 -> 0.0486 -> 1.684 -> 5.644` in no-demo and
  `0.0035 -> 1.003 -> 3.236 -> 6.054` with internal reward.

The internal reward is therefore integrated and causally active, but it does
not create a usable strategy. Its admitted nominal predictor is masked to zero
after tactile-confirmed failure and has no positive alternative-strategy
training sequence. It cannot by itself teach bottom support, side bracing, or
a lower heavy-load posture after failure.

## Next allowed boundary

Do not add more updates to this contract and do not weaken success/safety
criteria.

The next runtime action is a no-learning, exact-source-layout synchronized
diagnostic comparing:

1. the near-zero update-1 no-demo nominal rollout; and
2. the update-521 internal-reward nominal rollout.

It must replay the admitted grid checkpoints read-only, preserve the corrected
action-102 plus TacSL-100--103 bootstrap, stop on the first unsafe terminal
state, and retain synchronized world RGB, direct pressure/signed shear,
slip/failure, posture/contact, and official GelSight RGB/depth. These failed
videos are diagnosis only and never completion evidence.

Only after that diagnosis may a new training contract be frozen. It must first
establish a safe positive nominal foundation or reachable lift/stability
curriculum before asking ICM to discover post-failure alternatives. Original
ICM must remain an independently learned forward/inverse novelty model; a
curriculum, safety constraint, slip term, or task-success reward may not be
relabeled as curiosity.
