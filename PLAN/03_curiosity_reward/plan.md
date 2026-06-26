# Phase 03: Curiosity Reward

## Goal

Define and implement intrinsic objectives around physical prediction, not raw
pixel prediction.

## Reward Components

- object-motion prediction error;
- contact prediction error;
- unexpected object acceleration or slip proxy;
- lift-response mismatch under expected mass;
- bounded impact/useful-change reward;
- safety penalty for excessive force or unstable motion;
- no-op penalty;
- learning progress over time.

## Completion Criteria

- Reward spec exists.
- Components are logged separately.
- Reward can be evaluated on baseline rollouts before it is used for policy
  adaptation.
