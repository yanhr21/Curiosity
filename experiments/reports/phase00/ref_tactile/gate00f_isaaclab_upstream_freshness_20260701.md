# Gate 00F IsaacLab Upstream Freshness

- Date: `2026-07-01`
- Classification: `source_freshness_not_runtime_not_gate_completion`

## Local Source

- Path: `external/IsaacLab_official`
- Remote: `https://github.com/isaac-sim/IsaacLab.git`
- Local HEAD: `b4c321024792976150ca55fddb26fa34480d974e`
- Local VERSION: `2.3.2`
- Local HEAD date: `2026-06-12T03:12:35-07:00`
- Local HEAD subject: `Fix beta2 docs deploy selection (#6164)`

## Upstream Probe

`git ls-remote` observed:

- `HEAD`: `b4c321024792976150ca55fddb26fa34480d974e`
- `refs/heads/main`: `b4c321024792976150ca55fddb26fa34480d974e`
- `refs/tags/v2.3.2`: `37ddf626871758333d6ed89cf64ad702aef127d0`
- `refs/tags/v3.0.0-beta`: `a4a7602f29e755e2673fe0022ea35566df6dd7d5`
- `refs/tags/v3.0.0-beta2`: `28a37cecdd433c22d9eabd6a5954add9f13a8951`

## Decision

The local IsaacLab official source currently matches upstream `main`/`HEAD`.
The `v3.0.0-beta*` tags exist and should remain visible as release/tag
context, but they do not make the local main checkout stale in this probe.

This freshness check does not register an IsaacLab TacSL runtime, pull a
container, import modules, run TacSL, or clear Gate 00F.
