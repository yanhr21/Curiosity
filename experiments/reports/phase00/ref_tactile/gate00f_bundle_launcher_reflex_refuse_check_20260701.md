# Gate 00F Bundle Launcher Reflex Refuse Check

- created_at: `2026-07-01`
- classification: `launcher_safety_check_not_training_not_sanity`
- status: `pass_reflex_resource_refused`

Command:

```bash
JOB_ID=160860 RUN_TAG=p00_gate00f_bundle_reflex_refuse_test WINDOW_NAME=p00_gate00f_bundle_refuse bash experiments/configs/phase00/ref_tactile/launch_gate00f_reference_bundle_tmux.sh
```

Observed result:

- exit code: `5`
- message: `ERROR: Slurm job 160860 is not Curiosity-owned by workdir: /public/home/yanhongru/ICLR2027/Reflex`

This confirms the new Gate 00F reference bundle launcher refuses the currently
running Reflex-owned allocation instead of accidentally reusing it. This is a
launcher safety check only, not training and not official reference sanity.
