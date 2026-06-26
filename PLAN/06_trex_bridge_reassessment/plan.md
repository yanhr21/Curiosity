# Phase 06: T-Rex Bridge Reassessment

## Goal

Reassess whether a T-Rex bridge is worth building after Newton-native
adaptation has demonstrated useful behavior.

## Reassessment Conditions

Only revisit strict T-Rex promotion if all are plausible:

- bimanual 62D state/action/action_abs source;
- accepted synchronized head/right-wrist/left-wrist camera source;
- calibrated nonzero `[10,6]` F6;
- ten dense tactile deformation streams;
- strict inventory pass without padding or renaming.

## Possible Outcomes

- use T-Rex as a frozen/reference policy;
- post-train T-Rex on faithful new task data;
- keep T-Rex as reference only and publish Newton-native adaptation results.

## Completion Criteria

- Explicit go/no-go decision.
- If no-go, blocker is concrete.
- If go, strict data contract and sanity gates are documented before any run.

## Reference Checkpoint Sanity

2026-06-27: ran a reference-only sanity check for the currently staged official
T-Rex checkpoint assets inside the existing Curiosity allocation.

- Launcher: `experiments/configs/launch_trex_checkpoint_sanity_tmux.sh`.
- Compute runner: `experiments/configs/run_trex_checkpoint_sanity_in_alloc.sh`.
- Integrity checker: `experiments/configs/trex_checkpoint_integrity_sanity.py`.
- Model-load checker:
  `experiments/configs/trex_midtrain_model_load_sanity.py`.
- Log: `logs/trex/trex_checkpoint_current_sanity_20260627_2055.log`.
- Integrity output:
  `experiments/outputs/trex_checkpoint_current_sanity_20260627_2055_integrity.json`.
- Model-load output:
  `experiments/outputs/trex_checkpoint_current_sanity_20260627_2055_midtrain_model_load.json`.
- Report:
  `experiments/reports/2026-06-27_phase06_trex_checkpoint_current_sanity.md`.

Result:

- checkpoint integrity sanity: pass;
- official midtrain model-load sanity: pass;
- midtrain embedded VQ-VAE, deform encoder/proj, tactile code embedder, and
  TacF6 stats: present;
- pretrain stage-1 state dict and Qwen safetensors: present;
- no training;
- no placeholder model;
- no generated T-Rex fields.

Interpretation: this is useful reference-checkpoint evidence only. It does not
make the Newton Panda lift-hold source T-Rex-compatible and does not change the
short-term Newton-native scripted-prior route. Strict bridge promotion remains
blocked until real synchronized bimanual state/action/action_abs, accepted
cameras, calibrated nonzero `[10,6]` F6, and ten dense tactile deformation
streams exist without padding or renaming.
