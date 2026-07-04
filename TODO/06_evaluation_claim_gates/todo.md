# TODO 06: Evaluation And Claim Gates

- [ ] Define the strongest baseline before running final comparisons.
- [ ] Freeze held-out object and morphology splits.
- [ ] Freeze safety thresholds for falls, drops, excessive torque, object
  acceleration, and collision/contact force.
- [ ] Run every counted training attempt for at least one hour in a
  Curiosity-owned tmux-held Slurm allocation.
- [ ] Record GPU utilization evidence for counted training attempts.
- [ ] Record exact command, config, commit, checkpoint or failure, and log.
- [ ] Save MP4 rollout evidence for final evaluations.
- [ ] Run required ablations.
- [ ] Write final result classification: positive, negative, invalid, or
  blocked.
- [ ] Do not start a stronger claim if any gate is missing.
