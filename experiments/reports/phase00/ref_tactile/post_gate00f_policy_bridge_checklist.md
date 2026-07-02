# Post-Gate 00F Policy Bridge Checklist

- created_at: `2026-07-01`
- classification: `future_execution_checklist_not_training_not_gate_completion`

This checklist defines what happens after the tactile/base gates are closed. It
does not authorize checkpoint loading, dataset conversion, training, or
curiosity claims while Gate 00D/00E/00F remain open.

## Preconditions

Before any policy checkpoint work:

- Gate 00D must pass or have user-accepted faithful blocker evidence.
- Gate 00E must pass with base grasp/lift/hold and dense tactile/mechanics
  evidence.
- Gate 00F must pass official UniVTAC/TaCauchy/IsaacLab TacSL semantic sanity
  or have user-accepted faithful blockers.
- Any model loading, conversion, training, evaluation, or rendering must run
  inside a Curiosity-owned tmux-held Slurm allocation.
- Dependency installation or package builds must not run on compute nodes.

## T-Rex Bridge

Use `external/T-Rex_43ff` and `external/T-Rex_full_b23` as the official source
snapshots. The preferred future checkpoint is
`miniFranka/T-Rex_midtrain_mecka23k_ucb100_vqvae_epoch6`, because it embeds the
tactile VQ-VAE and starts from tactile-reactive midtraining.

Do not promote a Newton dataset to T-Rex compatibility until it has:

- head RGB compatible with the slow image stream;
- left/right wrist or equivalent fast-view RGB streams, or a documented
  faithful adaptation;
- state/action compatibility with eef-62 or a validated adapter;
- high-frequency tactile F6 history shaped as hands/fingers/6-axis wrench;
- deformation or tactile image sequence aligned with force history;
- slow/fast timing metadata;
- train/validation/held-out split without leakage;
- normalization stats compatible with checkpoint expectations.

Required ablations: vision+tactile, tactile-only masked vision, vision-only
disable-tactile, and noisy/mismatched tactile.

The minimum valid first claim is only that the official checkpoint loads and
runs on a schema-valid converted held-out batch. Policy success requires
held-out rollout metrics that beat the strongest baseline without safety
regression.

## FTP-1 Baseline

FTP-1 is a future serious general tactile policy baseline. Before using
`MJJJJ1064/ftp1_v0426_50kstep`, prepare a zarr conversion spec for Newton dense
visuo-tactile episodes, checkpoint asset compatibility evidence, declared
metrics, and held-out comparisons against Newton base and T-Rex bridge when
available.

## Representation Baselines

AnyTouch2 and Sparsh are tactile encoder or force-field representation
baselines. Before use, they need checkpoint access or an official blocker,
input-channel mapping from Newton tactile maps, force/deformation prediction
metrics, and ablations proving value beyond vision-only and scalar contact
proxies.

## Forbidden Shortcuts

- Do not hand-roll a toy VQ-VAE or transformer and call it T-Rex.
- Do not rename Newton proxy contact counts into T-Rex F6.
- Do not claim curiosity success from checkpoint loading, lower training loss,
  or a single rendered video.
- Do not start real curiosity training before the tactile/base gates are
  closed.
