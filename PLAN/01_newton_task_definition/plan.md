# Phase 01: Newton Task Definition

## Goal

Define the first Newton-native adaptation benchmark:

```text
lift-and-hold under object-property uncertainty
```

## Task Design

Objects:

- cup-like object with fill-level proxy;
- box and cylinder variants;
- optional pouch/compliant object after the rigid benchmark passes.

Object properties:

- mass: empty, half, full;
- friction: low, medium, high;
- initial pose perturbation;
- optional compliance and fragility/safety tags.

Observations:

- robot joint state;
- end-effector pose;
- object pose and velocity;
- action target;
- rigid contact count or contact proxy;
- proxy camera frames;
- optional Taccel marker evidence if clearly namespaced.

Actions:

- gripper closure target;
- lift velocity or EEF target;
- stabilization/regrasp parameter;
- scripted controller parameter for first baseline.

## Completion Criteria

- Task spec exists.
- Object parameter grid exists.
- Metrics are defined.
- First visual gate command and expected output paths are defined.
