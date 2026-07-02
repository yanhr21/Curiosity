# Latest Source Freshness V4

- Date: `2026-07-01`
- Classification:
  `source_freshness_audit_not_training_not_runtime_not_gate_completion`
- Machine-readable record:
  `experiments/configs/phase00/ref_tactile/latest_source_freshness_20260701_v4.json`

## Method

Used lightweight `git ls-remote` plus local `git rev-parse` only. No checkout
was modified, no dependency was installed, and no simulation, rendering, model
loading, training, or evaluation was run.

## Result

No tracked remote changed relative to the current active source records:

- Newton remote main: `8c501b47847569fecdda97a9f7f01205c6f7964f`
- Taccel remote main: `cb23bc251b531ba6908a3788c2f91423cd543149`
- T-Rex main: `43ff632259d76f08373c085c53111825060d029b`
- T-Rex full-pipeline: `b23eafe564a1457cd4eacb889aaf6fbf29a29034`
- IsaacLab official main: `b4c321024792976150ca55fddb26fa34480d974e`
- TacEx main: `adceed41afb7cb48f9ec1f66a662fb8e5a06627f`
- TaCauchy main: `c228cfe9050904cd5d71d64f6eb5104768d4cbda`
- UniVTAC main: `05bcd3edb92237107efa40105292a24f1a9fd761`
- FTP-1 main: `dd7cda66c7e97a170e0435fc6c4428b350cbdcc0`
- AnyTouch2 main: `82c5677d9cf0176d97a1fe04745f63cd02dd6f54`
- HydroShear HEAD: `a53a51cb74f0608ca53839415d7f1964a99f1db0`

## Gate Effect

This confirms the current source set is fresh for the tracked official refs.
It does not clear Gate 00F. The runtime blocker is unchanged: no
dependency-complete UniVTAC, TaCauchy, or IsaacLab TacSL runtime/container is
registered. Curiosity training remains disallowed.

Newton-specific correction: `external/newton_8c501` matches latest remote main,
and its H200 runs measured `80.1 FPS` and `80.8 FPS`, which is acceptable
around the current 80 FPS continuation threshold. `external/newton_d58`
remains the current strongest complete runtime/tactile evidence chain because
it already has candidate dense tactile export evidence, but 8c501 is not
blocked by FPS.
