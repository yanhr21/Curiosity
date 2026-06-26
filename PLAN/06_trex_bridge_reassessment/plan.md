# Phase 06: T-Rex Bridge Reassessment

## Goal

Reassess whether a T-Rex bridge is worth building after Newton-native
adaptation has demonstrated useful behavior.

## Reassessment Conditions

Only revisit strict T-Rex promotion if all are plausible:

- bimanual 62D state/action/action_abs source;
- accepted synchronized head/right-wrist/left-wrist camera source;
- calibrated nonzero `[10,6]` F6;
- ten dense tactile deformation streams;
- strict inventory pass without padding or renaming.

## Possible Outcomes

- use T-Rex as a frozen/reference policy;
- post-train T-Rex on faithful new task data;
- keep T-Rex as reference only and publish Newton-native adaptation results.

## Completion Criteria

- Explicit go/no-go decision.
- If no-go, blocker is concrete.
- If go, strict data contract and sanity gates are documented before any run.
