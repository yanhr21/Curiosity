# Reference Asset Availability

Date: 2026-07-01

This is a lightweight file-presence audit only. It did not run asset setup,
Git LFS download, package import, simulation, rendering, data collection,
training, model loading, or Slurm allocation.

Machine-readable audit:
`experiments/configs/phase00/ref_tactile/envprep/reference_asset_availability_v1.json`

## Result

UniVTAC bundled TacEx has useful tactile assets present. After explicit user
approval on 2026-07-01, those bundled TacEx assets were copied into TaCauchy
with `rsync --ignore-existing`. The TaCauchy file-presence blocker is now
cleared, but this is still not TaCauchy official sanity and not Gate 00F
completion.

## TaCauchy

Asset root:
`external/TaCauchy/source/tacex_assets/tacex_assets/data`

Approved local reuse evidence:

- copy log:
  `logs/newton/phase00/ref_tactile/envprep/assets/tacauchy/approved_univtac_asset_reuse_copy_20260701.log`
- verify status:
  `experiments/outputs/phase00/ref_tactile/envprep/assets/tacauchy/verify_status.json`
- target size after copy: `412M`
- tactile test shape USD count after copy: `21`
- `Sensors/GelSight_Mini/Sensor.usd`: present

Observed assets after approved reuse:

- `Sensors/GelSight_Mini`: includes `Sensor.usd`, `Case.usd`, Gelpad USDs,
  textures, docs, and calibration files.
- `Sensors/digit`: 2 files, mainly docs and `params.json`; missing full USD
  and calibration assets.
- `Sensors/9dtact`: 4 files, including `Sensor.usd.before_output_fix` and
  `Sensor.usd.before_material_fix`; no final `Sensor.usd` was observed.
- `Robots/Franka`: one `digit_uipc_gelpad.usd.broken` file observed.
- `Props/tactile_test_shapes`: `21` USD files.

TaCauchy `scripts/setup_assets.sh` still requires `git-lfs`, network access to
TacEx, sparse checkout, `git lfs pull`, and `rsync`. That official LFS path was
not used because `git-lfs` is unavailable. The accepted substitute is local
reuse from the official local UniVTAC bundled TacEx tree.

## UniVTAC Bundled TacEx

Asset root:
`external/UniVTAC/third_party/TacEx/source/tacex_assets/tacex_assets/data`

Observed positive assets:

- GelSight Mini includes `Case.usd`, `Sensor.usd`, Gelpad USDs, textures,
  `params.json`, `dataPack.npz`, `polycalib.npz`, and `shadowTable.npz`.
- GF225 includes `Case.usd`, calibration NPZ/JSON files, and optical
  simulation utility files.
- Franka/GF225 includes Panda arm hand USDs and UIPC/GelSight pad USDs.
- `Props/tactile_test_shapes` includes multiple USD tactile test shapes.

These assets do not close Gate 00F by themselves because the official UniVTAC
environment and compute-side sanity are still missing.

## Interpretation

Gate 00F now has remaining blockers:

- target UniVTAC/TaCauchy environments do not exist;
- UniVTAC/TaCauchy official reference sanity has not passed.

Curiosity training remains disallowed.
