# Phase 00 Simulator Path Selection

Date: 2026-07-01

Machine-readable record:
`experiments/configs/phase00/dense_tactile_infant/simulator_path_selection_20260701_v1.json`

## Decision

Use Newton `external/newton_8c501` at commit
`8c501b47847569fecdda97a9f7f01205c6f7964f` as the active Phase 00 dense
tactile infant simulator path.

This selects the simulator/base-evidence path only. It is not curiosity
training, not a final tactile semantic validation pass, and not a T-Rex schema
claim.

## Why This Path

The previous 8c501 compute-side chain produced positive candidate dense tactile
evidence:

- base grasp/lift final test;
- candidate direct `Fn` and `Ft`;
- steel candidate material settings;
- normal and contact-area proxy overlay;
- candidate marker/deformation rendering;
- reference comparison assets;
- channel-layout audit.

The strongest previous status record is:
`experiments/reports/phase00/ref_tactile/newton_8c501_cont_chain_status.md`.

## Boundary

The previous evidence remains candidate evidence. It must preserve provenance
as `candidate.newton_mjw.*` and must not rename proxies into official tactile
fields.

Known limitations:

- `area_proxy != real contact area`;
- marker render is not validated photometric GelSight output;
- candidate `Fn/Ft` is not validated official tactile force field;
- Gate 00F remains low-priority final semantic validation/comparison-gap work;
- curiosity training remains downstream of dense tactile/base evidence.

## Active Output Policy

New runs must use active paths:

```text
experiments/outputs/phase00/dense_tactile_infant/
experiments/visuals/phase00/dense_tactile_infant/
experiments/reports/phase00/dense_tactile_infant/
logs/newton/phase00/dense_tactile_infant/
```
