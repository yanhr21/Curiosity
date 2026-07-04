# Plan 06: Evaluation And Claim Gates

## Purpose

Prevent overclaiming. A result is not a success unless it beats strong
baselines on held-out unknown-load carrying without safety regression.

## Required Held-Out Axes

- object mass;
- center of mass;
- shape and size;
- friction;
- contact affordance or handle availability;
- carry distance;
- carry duration;
- robot morphology;
- reference-video embodiment;
- perturbations during carry.

## Required Metrics

- carry distance;
- carry duration;
- drop rate;
- slip;
- contact loss;
- fall rate;
- recovery after perturbation;
- object acceleration;
- energy or torque cost;
- peak joint torque;
- balance margin;
- probing attempts;
- posture diversity;
- task success over strongest baseline.

## Required Baselines

- environment reward only;
- no video;
- active probing only;
- video only, no probing;
- scripted probing;
- fixed posture;
- retargeting baseline;
- behavior-cloning or video-conditioned supervised baseline;
- oracle load and center of mass;
- wrong-video and mismatched-video conditions.

## Success Claim

A success claim requires all of the following:

- beats the strongest declared baseline on held-out tasks;
- no safety regression;
- video causally improves over no-video;
- active probing causally improves over no-probing;
- morphology-dependent posture selection is visible and measurable;
- complete logs, commands, configs, commits, checkpoints or failure records,
  and MP4 rollout evidence.

## Negative Result

A negative result is useful if it is recorded honestly. Examples:

- video reward collapses to object displacement only;
- probing adds cost but not success;
- policy uses one posture for all morphologies;
- learned reward is fooled by wrong videos;
- stronger baseline wins.

## Exit Criteria

- A final report classifies the result as positive, negative, invalid, or
  blocked.
- No claim exceeds the evidence.

