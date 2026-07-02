# TaCauchy Asset Blocker Audit

Status: Gate 00F remains blocked. This is an audit only; no assets were copied
and no official repo was mutated.

Official TaCauchy path:
- script: `external/TaCauchy/scripts/setup_assets.sh`
- source: `https://github.com/DH-Ng/TacEx.git`
- method: sparse checkout + `git lfs pull` + `rsync`
- blocker: current PATH has no `git-lfs`

Observed TaCauchy target:
- root: `external/TaCauchy/source/tacex_assets/tacex_assets/data`
- size: `1.8M`
- GelSight Mini files: only `Sensors/GelSight_Mini/docs/README.md`
- tactile test shape USD count: `0`

Observed UniVTAC bundled TacEx source:
- root: `external/UniVTAC/third_party/TacEx/source/tacex_assets/tacex_assets/data`
- size: `410M`
- contains `Sensors/GelSight_Mini/Sensor.usd`, `Case.usd`, Gelpad USDs, textures
- tactile test shape USD count: `21`

Candidate reuse:
`rsync -a --ignore-existing external/UniVTAC/third_party/TacEx/source/tacex_assets/tacex_assets/data/ external/TaCauchy/source/tacex_assets/tacex_assets/data/`

This command was not executed. It would copy roughly `410M` into
`external/TaCauchy`, which is a material mutation of an official external
repository. Keep it as a candidate path until the user explicitly approves it
or a cleaner official `git-lfs` asset setup becomes available.

Gate effect:
`reference_asset_availability` and `tacauchy_official_reference_sanity` remain
failed in `p00_gate_d58_marker_v1_20260701_071843`.
