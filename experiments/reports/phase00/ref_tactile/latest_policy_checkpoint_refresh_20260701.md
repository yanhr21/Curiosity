# Latest Policy/Checkpoint Refresh

- created_at: `2026-07-01`
- classification: `source_checkpoint_availability_refresh_not_training_not_gate_completion`
- curiosity_training_allowed: `false`

## T-Rex

Official source now has a concrete released path, not just a paper reference:

- project page: `https://tactile-reactive-dexterous.github.io/`
- official repo: `https://github.com/ZhuoyangLiu2005/T-Rex`
- latest main snapshot: `external/T-Rex_43ff` at
  `43ff632259d76f08373c085c53111825060d029b`
- full-pipeline snapshot: `external/T-Rex_full_b23` at
  `b23eafe564a1457cd4eacb889aaf6fbf29a29034`

The main branch ships post-training and inference code. The full-pipeline
branch contains pretrain/midtrain scripts. The official README lists two
released Hugging Face checkpoints:

- `miniFranka/T-Rex_pretrain_mecka22k_epoch1`
- `miniFranka/T-Rex_midtrain_mecka23k_ucb100_vqvae_epoch6`

The midtrain checkpoint is the important future candidate: it is described as
tactile-reactive cascaded flow with an embedded temporal tactile VQ-VAE, and
the repo says to start there for task post-training.

This does not clear the current gate. T-Rex is bimanual Dexmate/Sharpa Wave,
with 10 fingertip tactile sensors, LeRobot v3.0 data, and eef-62 action/state
contracts. Current Newton Panda evidence is single-arm and still has candidate
or proxy tactile semantics. Gate 00F must pass before any T-Rex bridge can be a
serious training claim.

## FTP-1

FTP-1 remains a future serious general tactile policy baseline/reference:

- project page: `https://ftp1-policy.github.io/`
- official repo: `https://github.com/michaelyuancb/ftp1-policy`
- local source: `external/ftp1-policy`
- checkpoint: `MJJJJ1064/ftp1_v0426_50kstep`

The checkpoint card describes a 4B-parameter 50k-step pretrained checkpoint
with image, state, language, and optional tactile inputs, producing action
chunks. It is not Newton-native base grasp evidence and does not clear Gate 00F.

## AnyTouch2 and Sparsh

AnyTouch2 is a future tactile representation reference:

- official repo: `https://github.com/GeWu-Lab/AnyTouch2`
- local source: `external/AnyTouch2`
- checkpoint path in official README: `xxuan01/AnyTouch2-Model`
- force dataset: `BAAI/ToucHD-Force`

The ToucHD-Force dataset card reports 722,436 touch-force pairs and the larger
ToucHD collection reports 2,426,174 contact samples. This is useful for
force-aware tactile representation design, not for claiming a Newton controller.

Sparsh is another serious tactile representation/force-field reference because
its official repo releases pretrained touch backbones and force-field decoder
checkpoints. It remains reference-only in this refresh.

## Decision

The next policy/model path after Gate 00F should be:

- keep Newton d58 as the current strongest base/runtime candidate evidence;
- use T-Rex midtrain as the strongest tactile-reactive architecture reference
  only after a faithful Newton-to-T-Rex data contract exists;
- use FTP-1 as a serious general tactile policy baseline/reference;
- use AnyTouch2/Sparsh as tactile encoder or force-field representation
  baselines.

No model was loaded or trained in this refresh. No checkpoint was downloaded.
Curiosity training remains disallowed.
