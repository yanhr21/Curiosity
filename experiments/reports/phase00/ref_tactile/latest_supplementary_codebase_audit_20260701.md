# Latest Supplementary Codebase Audit

Date: 2026-07-01

Classification: supplementary source audit only. This is not runtime evidence,
not official sanity, not simulation, not model loading, not training, not
checkpoint evidence, and not Gate 00D/00E/00F completion.

Machine-readable record:
`experiments/configs/phase00/ref_tactile/latest_supplementary_codebase_audit_20260701_v1.json`

## Sources

### TactSim-IsaacLab

- Remote: `https://github.com/yuanqing-ai/TactSim-IsaacLab.git`
- Local path: `external/TactSim-IsaacLab`
- Local commit: `4f92257177cd0ee18928de720b880505ec7f7638`
- Role: secondary photometric GelSight/DIGIT-style tactile simulation
  reference.

Static finding: the repository contains IsaacLab-oriented tactile simulation
assets and scripts, including `demo_ts.py`, `ts_cfg.py`, `img_utils.py`,
`assets/digit_asmb.usd`, `assets/box.usd`, and image-processing examples. Its
README describes a vision tactile simulator tested with IsaacSim 4.0, CUDA
12.3, and a DIGIT/GelSight-style sensor setup.

Usefulness for Phase 00: this is relevant to the current photometric/marker
gap because it shows a concrete IsaacLab tactile-camera asset path,
background/no-contact subtraction, light/material handling, and postprocessed
tactile image generation. It can inform future Gate 00F photometric checks
after a dependency-complete IsaacLab runtime exists.

Limitations: no IsaacLab runtime was registered, no official demo was run, no
container was pulled or built, no module was imported, and no tactile output
was generated. It is a secondary reference and does not replace UniVTAC,
TaCauchy, or official IsaacLab TacSL sanity.

### newton-actuators

- Remote: `https://github.com/newton-physics/newton-actuators.git`
- Local path: `external/newton-actuators`
- Local commit: `134dacb0912f4b8ce0465ecebf564479f2e62315`
- Role: Newton actuator concept/background source only.

Static finding: the README states that starting with Newton 1.3 actuators are
created and used exclusively from built-in `newton.actuators`, and that this
standalone package is no longer maintained.

Usefulness for Phase 00: it confirms the active base-controller path should
stay inside current Newton rather than depending on this deprecated standalone
package.

Limitations: this is not a grasp checkpoint, not a tactile simulator, not a
runtime dependency, and not a base-model completion path.

### UniT

- Remote: `https://github.com/ZhengtongXu/UniT`
- Observed remote HEAD: `52a286520b09708934b25c77aa826360d72c79db`
- Local path: none after cleanup.
- Role: future tactile representation/VQGAN candidate only.

Static finding: only remote HEAD evidence is currently retained. A local clone
attempt did not produce a usable checkout and the partial directory was
removed to keep `external/` clean.

Usefulness for Phase 00: UniT may be relevant later for tactile representation
comparison after Gate 00F and the dense tactile data contract are valid.

Limitations: it is not local source evidence yet, not a checkpoint, not loaded,
and not usable for any current Gate or training claim.

## Gate Effect

- Gate 00D remains open because dense tactile semantics are still not
  validated.
- Gate 00E remains open because the base model/controller lacks accepted
  tactile semantic validation.
- Gate 00F remains open because official UniVTAC, TaCauchy, and IsaacLab TacSL
  runtime sanity still have not passed.
- Gate 00G curiosity training remains disallowed.

## Decision

Keep TactSim-IsaacLab as a secondary photometric tactile simulation reference,
keep newton-actuators as deprecated Newton background only, and keep UniT as a
remote-only future representation candidate until a clean local checkout and
checkpoint/source audit exist. None of these sources changes the current
decision: d58 remains the strongest Newton runtime/tactile candidate, and the
active blocker remains dependency-complete official tactile semantic runtime
validation.
