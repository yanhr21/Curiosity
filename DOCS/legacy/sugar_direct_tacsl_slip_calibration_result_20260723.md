# Direct TacSL slip calibration result — 2026-07-23

## Verdict

The direct pressure/shear estimator and its controlled official-TacSL probe
protocol are implemented and auditable, but **Stage E is not admitted**. The
frozen detector passed calibration and two development validation directions,
then missed four predeclared bounds on a genuinely fresh contact state and
fresh `3x-2y` tangential direction. No threshold or ranking rule was changed
after that evaluation was opened.

This is a useful negative result. It shows that the current global
pressure/shear-utilization thresholds are conservative and precise, but do not
yet generalize evenly between the two R15 palms. No SMP+ICM policy training was
started.

All sensor claims in this record are **high-fidelity simulated tactile**. They
are not physical GelSight calibration, hardware tactile, or sim-to-real.

## Deployed-input and oracle boundary

The deployed estimator in
`SUGAR/source/sugar_rl/sugar_rl/tasks/locomanip/direct_tactile_slip.py` reads
only the per-hand direct tactile history and reset mask. Its history contains
taxel-resolved normal pressure and signed two-axis shear. It computes
per-taxel shear magnitude, friction utilization, saturation fraction,
pressure/load change, footprint overlap, centroid motion, contact topology,
hysteretic incipient/gross state, and event start/end.

The exact SDF contact normal and contact-conditioned hand-box relative
tangential velocity are exported only as calibration/evaluation labels. The
runtime audit passes all of the following:

- deployed input is only tactile history plus reset mask;
- simulator oracle is absent from estimator input;
- task reward, success, lift, hidden mass, and hidden friction are absent;
- threshold search uses only the declared calibration split; and
- validation and evaluation are opened only after parameters are frozen.

Release and regrasp are represented as contact-topology transitions. They are
not forced into a false zero-slip class and are not counted as uncontrolled
slip events. This distinction is required for later alternative-grasp
discovery.

## Controlled probe v6

The final probe artifact is:

`experiments/sugar_smp_exploration/audits/direct_tacsl_controlled_slip_probes_v6/`

It contains 64 sixteen-step sequences over both hands and four splits, covering
no contact, stick, positive/negative incipient slip, positive/negative gross
slip, release, and regrasp. Incipient and gross commanded tangential speeds
are `0.001 m/s` and `0.05 m/s`. The gross trajectory is triangular, so it
retains contact rather than drifting off the elastomer.

All 14 probe gates pass. In particular, all 32 sliding sequences contain
14–16 direct-contact steps, the exact tangential oracle matches the commands,
stick remains near zero, SDF-normal release/regrasp motion stays below half the
gross speed, release loses contact, regrasp restores it, and oracle/contact
normal values are zero outside direct contact.

The calibration/development base frames are unchanged from the locked method.
The final evaluation frames are left `150` and right `124`. They exclude both
previously opened evaluation pairs, left `135/153` and right `82/92`. The final
direction is `tangent_three_x_minus_two_y`, which was not used for threshold
selection.

Artifact hashes:

- `probes.npz`: `4c60676c953b4eff075f5447f04b5104f686f0e69110d01df714924f247805e6`
- `proof.json`: `c1f8050fc5e041077998eb3484efbb8fb8cde9af30e07ec5576f25ec7c1325e1`
- collector source: `d4174fdd5d0ade5948c70f9fbff260855b4f6cdb4a2ad3bf61a18c79f2a1fe2f`

## Frozen calibration v9

The final result is:

`experiments/sugar_smp_exploration/audits/direct_tacsl_slip_calibration_v9/calibration.json`

The v9 parameter object is byte-for-byte equal after canonical JSON sorting to
the v8 object chosen before this fresh evaluation. The important on-thresholds
remain:

- incipient utilization: `0.09790964797139168`;
- gross utilization: `0.6982263475656509`; and
- maximum accepted transition centroid speed: `60.22145462036133 taxels/s`.

The calibration and two development validations all pass. Their metrics are:

| Split | Incipient recall | Gross recall | Incipient precision | Gross precision | Negative FPR | Worst latency |
|---|---:|---:|---:|---:|---:|---:|
| calibration | 0.9310 | 0.9286 | 1.0000 | 1.0000 | 0.0000 | 1 |
| validation | 0.9667 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 1 |
| diagonal validation | 0.9500 | 0.9667 | 1.0000 | 1.0000 | 0.0000 | 1 |

The once-opened fresh evaluation does not pass:

| Metric | Frozen gate | Fresh result | Status |
|---|---:|---:|---|
| incipient recall | >= 0.90 | 0.8966 | fail |
| gross recall | >= 0.90 | 0.8571 | fail |
| incipient precision | >= 0.90 | 1.0000 | pass |
| gross precision | >= 0.90 | 1.0000 | pass |
| negative false-positive rate | <= 0.05 | 0.0000 | pass |
| incipient classified as gross | <= 0.10 | 0.0000 | pass |
| worst event latency | <= 3 steps | 4 steps | fail |
| left-hand sliding recall | >= 0.85 | 0.9667 | pass |
| right-hand sliding recall | >= 0.85 | 0.8214 | fail |
| no-contact accuracy | 1.00 | 1.0000 | pass |
| unevaluable sliding sequences | 0 | 0 | pass |
| transition sequences with unmasked slip | 0 | 0 | pass |

The failure is therefore missed/delayed right-palm detection, not false slip,
oracle leakage, loss of contact in the evaluation trajectory, or confusion of
release/regrasp with sliding. The `calibration.json` SHA256 is
`c9857cc43d78243e6d5b3ce32464cb17845b7b8bab52a6de3714e28fb5f9bf2f`.

## Rendered evidence

The three figures were generated on the retained compute allocation and were
visually inspected. Each shows the spatial pressure footprint and centroid,
signed-shear magnitude, utilization/spatial-motion histories and thresholds,
and exact oracle label against the frozen detector:

- `validation_slip_detector_summary.png`, SHA256
  `f28c074841fa827a4100ebc4ef00265ca6911f9e556779adc84b9e710c27a0be`;
- `validation_diagonal_slip_detector_summary.png`, SHA256
  `bab9a11aab4f4f297fd9788fabdf7cddafc73917d525de4c4e89319a29b624da`;
- `heldout_slip_detector_summary.png`, SHA256
  `8a94954ab4a486a44a780547b3d75b3fa39a781fbd241e2de244bbefba42ff38`.

The fresh figure makes the generalization gap visible: the detector remains
clean on no-contact, stick, release and regrasp, while some new-direction
incipient/gross frames cross the frozen utilization boundary late or not at
all.

## Scientific boundary and next gate

The current code is sufficient as an auditable estimator prototype and as a
negative baseline. It is not sufficient to claim robust tactile-only slip
detection, and it must not be connected to policy reward as an admitted Stage
E component.

The next valid attempt must be a new predeclared protocol, not threshold tuning
on this evaluation. It should expand calibration across palm/load/contact
footprint, mass, friction, tangential direction, and declared TacSL noise or
latency conditions; predeclare whether normalization or calibration is shared
or per-sensor; and reserve a seventh unseen contact state/direction as the new
single-use evaluation. The v9 evaluation remains permanently development data
after this opening.

This result does not change the meaning of curiosity. Original ICM remains the
separately learned forward/inverse discovery subsystem whose signal is
prediction error on unfamiliar controllable transitions. Slip state, task
success, and reduced error are external measurements/objectives and are never
used to define `r_icm`.

Superseded probe v1–v5, calibration v8, and transition-disambiguation v1–v3
artifacts were moved outside the repository to
`/public/home/yanhongru/Curiosity_archive/workspace_siblings/Curiosity_archive_20260723_smp_formal_intermediates/slip_detector_pre_fresh_holdout/`.
At the time of this v9 record, only the v6 probe and v9 frozen evaluation
remained in the active direct-slip experiment directories.

## Subsequent locked follow-up

The predeclared follow-up was later executed through v10/v11/v12/v13. A
detector-blind 100-motion scan selected motion 45; v13 passed all expanded
motion-6 development groups, then failed its single-use frozen motion-45 gate.
Nominal gross precision was `0.8276`, incipient-as-gross was `0.1736`, and
worst latency was 16; one-step latency and combined stress each retained false
slip across four regrasp sequences. Motion 45 is now opened and cannot be
retuned or reused as held-out evidence. The current result and visual evidence
are recorded in
`DOCS/sugar_direct_tacsl_persistent_contact_v13_fresh_result_20260723.md`.
