# Runtime Gate Correction

Date: 2026-07-01

Classification: active gate correction. This is not training, not simulation,
not rendering, not Gate completion, and not curiosity success.

Machine-readable record:
`experiments/configs/phase00/ref_tactile/runtime_gate_correction_20260701_v1.json`

## Correction

The old `82 FPS` number is a historical diagnostic reference, not a hard
blocking threshold.

Runtime around `80 FPS` is acceptable for continuing dense tactile export,
reference-video comparison, channel audit, and Gate review. The Newton
`8c501...` benchmark runs measured `80.1 FPS` and `80.8 FPS`, so they must not
be treated as negative runtime-target evidence.

## Current Hard Gates

- No login-node simulation, rendering, training, model loading, dataset
  conversion, or heavy Python/Numpy/PyTorch work.
- Use only Curiosity-owned tmux-held Slurm allocations for compute work.
- Do not touch Reflex/OpenPI/Cosmos or other non-Curiosity resources.
- Do not commit unless the user explicitly asks.
- Do not use toy/degraded models as substitutes for official serious methods.
- Gate 00D: dense tactile/mechanics environment must include synchronized
  visual scene, Fn/Ft, shear, contact normal, contact area or proxy clearly
  labeled, penetration/compression where available, and material/mechanics
  statistics. Scalar contact count is not enough.
- Gate 00E: base controller/model must complete grasp/lift/hold and export
  tactile/mechanics evidence. FPS around 80 is acceptable; tactile evidence
  completeness is the real gate.
- Gate 00F: official semantic validation or faithful blocker evidence is still
  required for UniVTAC, TaCauchy, and IsaacLab TacSL.
- Gate 00G: curiosity training remains disallowed until Gate 00D/00E/00F are
  satisfied or faithful blockers are accepted.

## Next Action

Proceed with Newton `8c501...` dense tactile export, reference comparison,
channel audit, and Gate review when a Curiosity tmux-held Slurm allocation is
available. Do not spend time trying to optimize from 80 FPS to 82 FPS.
