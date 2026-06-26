# Phase 02: Baselines

## Goal

Establish serious baselines before evaluating curiosity.

## Baselines

1. No-adaptation scripted grasp-and-lift.
2. Scripted feedback adaptation.
3. Behavior cloning, diffusion-policy-style, ACT-style, or another documented
   manipulation baseline if demonstrations are available.
4. Newton-native contact-aware diagnostic baseline.

## Rules

- Do not introduce toy T-Rex, toy VQ-VAE, toy Transformer, or toy world model.
- Any small diagnostic model must be explicitly labeled as Newton-native
  diagnostic.
- Every baseline must report identical metrics.

## Completion Criteria

- Baseline commands exist.
- Metrics table format exists.
- At least one visual success and one failure case are saved.
