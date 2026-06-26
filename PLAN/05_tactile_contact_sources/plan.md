# Phase 05: Tactile And Contact Sources

## Goal

Add touch-like evidence only where it is real, useful, and clearly named.

## Allowed Namespaces

```text
newton.contact.*
newton.camera.*
newton.object.*
taccel.marker.*
taccel.ftac.*
candidate.*
```

## Forbidden Promotions

Do not create these fields unless their real contracts are satisfied:

```text
observation.tactile_f6
observation.tactile_deform.*
observation.images.*
observation.state
action
action_abs
```

## Completion Criteria

- Contact/tactile source improves or clarifies adaptation behavior.
- Every tactile visual has direct image paths.
- No partial source is renamed into T-Rex schema.
