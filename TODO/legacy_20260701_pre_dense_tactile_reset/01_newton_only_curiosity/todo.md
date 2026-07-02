# Phase 01 Newton-Only Dense Tactile Curiosity TODO

## Active Rules

- This is active after the 2026-07-01 user decision to proceed with Newton
  training instead of continuing to block on official reference runtimes.
- Current status: blocked by user pause on 2026-07-01. Do not start
  allocations, training, evaluation, data conversion, or further implementation
  until the user gives the next instruction.
- Gate 00F is low priority for now: final semantic validation/comparison gap
  only, not an active high-priority experiment and not a blocker for
  Newton-only Phase 01 once work resumes.
- Results must be labeled `Newton-only` until official UniVTAC/TaCauchy/
  IsaacLab TacSL validation passes.
- Do not claim Gate 00F completion or final reference-video tactile
  validation from Phase 01 results.
- Do not run simulation, rendering, dataset conversion, model loading,
  training, evaluation, or NumPy/PyTorch-heavy work on the login node.
- Use Curiosity-owned tmux-held Slurm allocations for all compute.
- The five real-training attempt stop gate applies.

## Training Setup

- [ ] Create Phase 01 training contract/config.
      - Declare data source, split, baseline, objective, metrics, safety
        metrics, video evidence, and ablations before training.
- [ ] Build or verify Newton dense tactile rollout dataset/manifest inside a
      compute allocation.
      - Required fields: scene RGB or visual state, `Fn`, `Ft`, shear,
        contact normal, contact-area proxy, penetration/compression proxy,
        object pose/lift, action/control, and time index.
      - If only the current 8c501 evidence is available, record it as a seed
        dataset and generate more rollouts in compute before real training.
- [ ] Run baseline evaluation.
      - Baselines: no-curiosity base controller, scripted feedback/probing,
        and random/grid probing where applicable.
- [ ] Train dense visuo-tactile prediction / learning-progress objective.
      - Must not be a toy placeholder represented as T-Rex/VQ-VAE/world-model.
      - Any simplified diagnostic must be labeled diagnostic only.
- [ ] Run closed-loop curiosity adaptation.
      - Must produce held-out metrics and full videos.
- [ ] Run tactile-mask ablations.
      - Required: vision+tactile, tactile-only masked vision, vision-only,
        noisy/mismatched tactile.
- [ ] Update Phase 01 real-training attempt ledger after each real one-hour
      attempt.
      - Classify each attempt as positive, negative, invalid, or blocked with
        evidence paths.
