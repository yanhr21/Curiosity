# Bounded full-Utonia overfit qualification

`train_overfit.py` is independent of the archived `model.py`/`train.py`. It wraps
the original complete `ObjectPredictor`; a pre-hook reads its existing
chronological head features for ten extra linear task outputs. The original
backbone, sensor affine, state head, and forward path remain intact and trainable.
The extra outputs are eight physical force channels, contact presence, and
learned mass availability. This is known-mesh state fitting, not arbitrary-shape
reconstruction or a replacement backbone/temporal model.

The fixed TRAIN cases are groups 0/2/3/5: 5000–5003, 5008–5015, 5020–5023.
Each retains frames 31/1181/1281/1581/2381: 80 fit items. Each item has 32 real
observations at 0.02 s spacing; no padding or label-dependent history selection.
Every shuffled 20-batch epoch visits all 80 items once in paired four-setting
batches. Exactly 2000 optimizer updates are permitted.

Before model construction, all 16 collected cases must pass the original fixed
12 controller checks and the collection's `qualification_passed` flag. Every
case must have at least one available mass label at frame 1581 or 2381.
Availability requires the entire 32-frame window to be airborne > 0.01 m,
bilaterally loaded > 0.01 N, actual COM acceleration <= 0.5 m/s², and angular
speed <= 0.2 rad/s. These are label conditions, not model inputs or predictions.
Actual `object_com_w` is mandatory; an AABB center is never substituted. Old
recordings can only undergo explicitly unqualified `--legacy-input-check`.
No criterion depends on predicted error or agreement between force and GT mass.
Every failed/unavailable/no-contact item remains in state losses and metrics.

Mass precision loss directly uses relative mass error and excludes unavailable
targets before computing errors. Availability itself is learned with BCE; its
deployment threshold is fixed at 0.5. Force targets come only from the existing
20 feature columns: two encoded-side normal loads, hand-received shear vector,
and the observation-normal approximation to the object's normal force vector.
All eight channels use newtons. Pure-shear and combined support scalars remain
separately available in the data adapter. Fitted/CAD normals are not solver
interface normals, so the combined force is explicitly approximate.

Geometry loss uses 1024 fixed original mesh vertices, exact detached SciPy
KDTree neighbor indices, and torch Euclidean distances in both directions.
This is the declared known-mesh geometric task loss, **not official PyTorch3D
Chamfer**. Evaluation uses all 15626 original vertices. Raw rotation, center,
size, mass, force RMSE, and availability are retained separately. The relative
mass target and metric agree; decreasing log-MSE is not used as acceptance.

Fit gates apply to every item: center <= 1 cm, full-mesh symmetric nearest-vertex
distance <= 1 cm, every dimension error <= 5%, available mass error <= 5%,
eight-channel force RMSE <= 0.5 N, and correct availability on all 80 items.
Interpolation evaluates each original `31+25k` clock excluding the five fit
clocks: 90 per case / 1440 total. Per-case gates are center and mesh mean <= 1 cm
and p95 <= 2 cm, available mass mean <= 5% and p95 <= 10%, force RMSE <= 0.5 N,
availability balanced accuracy >= 0.95, and false-confident fraction <= 0.05.
All-state metrics include no-contact priors; interpolation is not generalization.

Both the initial model and fixed endpoint run independent batch1 evaluation on
the exact same 80 fit and 1440 interpolation inputs. Saved initial/final clock,
target, and mask equality is required. These arrays support real matched-time
before/after rendering; this training script does not render.

Use the existing interpreter with the explicit official repository path:

```bash
unset PYTHONHOME
export PYTHONPATH="$PWD/experiments/object_predictor_v1/deps:$PWD/experiments/object_predictor_v1/vendor/Utonia:$PWD"
/public/home/yanhongru/envs/sugar_py311_isaacsim510/bin/python -P \
  -m scripts.sugar.object_predictor.train_overfit \
  --data experiments/object_predictor_v1/overfit_repair_v1/controller_v2 \
  --output experiments/object_predictor_v1/overfit_repair_v1/full_utonia
```

GPU execution must match the sole RUNNING allocation/step/host recorded in
`overfit_repair_v1/ACTIVE_RESOURCE.json`, with fd9 holding the original exclusive
`held_pipeline.lock`. Pending, wrong-resource, or missing-lock runs are rejected.
`--check-data` performs CPU-only input
and label qualification without constructing a model; use a separate new output
directory because protocol/results are never overwritten.

Outputs include frozen protocol, original physical qualification, initial and
four fixed fit evaluations, actual 2000-step sample/loss records, full checkpoint
and Adam state, named parameter deltas/Adam clocks, each active backbone module's
actual change, and initial/final interpolation arrays. Only the original unused
54-element mask token is exempted. Numerical acceptance failure returns exit 2;
execution success alone is not acceptance. Rendering and the user's approval
are required before any subsequent stage. No budget extension is automatic.

One `ARTIFACTS.json` records actual resolved source paths/hashes at initialization.
Completion checks those sources stayed unchanged and binds `model.pt`, the four
initial/final fit/interpolation NPZ files, and `PROTOCOL.json` by SHA256. A renderer
must call `train_overfit.verify_artifacts(root)` before using these predictions;
an incomplete manifest or any changed binding is rejected.

CPU tests cover force signs/coordinate covariance, true COM requirements,
unavailable-label gradient exclusion, geometric symmetry and gradient behavior,
exact sampling, physical-gate failure, individual fit/interpolation gates, and
inactive-module detection. They do not establish GPU execution or learning.
