"""Fixed eight-input overfit of the complete released Active3D touch-only GCN.

This is NATIVE ABC qualification only: four fixed TRAIN objects, two published
five-grasp sequences each, and exactly 1000 full-batch Adam updates. The original
model, graph, 30000-point/three-repeat Chamfer and published labels are used
directly. No object ID, GT mesh or point label enters the model. Nothing here
implements a pressure-to-chart encoder, SUGAR sensing, policy, or generalization
experiment. A separate real-observation SUGAR gate remains required.

Use --check for NumPy-only input qualification (no torch/CUDA import). Actual
training requires --output NEW_DIRECTORY inside the retained compute step.
All eight cases, including a free-space-only input, remain in every report.
The fixed zero-touch condition is evaluation only: identical empty inputs cannot
be required to predict four distinct shapes. Saved NPZ files contain full true
and predicted meshes for subsequent actual rendering; this script does not
claim that those renders have been produced or visually inspected.

--geometry-repair explicitly adds the separately declared official PyTorch3D
regularizers and uniform vertex/face supervision. It defaults OFF, never changes
the original model, and requires a new output directory and released weights.

--resume-from explicitly continues the completed geometry step1000 with its
complete model+Adam for exactly 3000 additional updates to absolute step4000.
It requires an actual original Engine equivalence PASS; default remains 1000.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import random
import time
import traceback
from types import SimpleNamespace

from .qualify_active3d_training_backend import (
    EXPERIMENT, FACES, GLOBAL_VERTICES, LOSS_SCALE, PARAMETERS, REPEATS,
    SAMPLES, VERTICES, official_imports, require, write_json,
)


OBJECT_IDS = ("18704", "11898", "15737", "13266")
STEPS = 1000
REPORT_EVERY = 100
LEARNING_RATE = 3e-4
GLOBAL_CD_LIMIT = .45
FSCORE_DISTANCE = .01
FSCORE_MINIMUM = .95
OFFICIAL_NONREGRESSION_ATOL = 1e-6


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fixed_grasps(object_index, repeat):
    """Mirror the published validation selector; GPU setup checks its output."""
    seed = 5 * object_index + repeat
    indices = list(range(50))
    random.Random(seed).shuffle(indices)
    return seed, indices[:5]


def load_cases(dataset):
    """Read fixed native arrays only. Model inputs are strictly chart xyz/mask."""
    import numpy as np

    dataset = Path(dataset).resolve()
    selection_path = dataset / "selected48/SELECTION_MANIFEST.json"
    selection = json.loads(selection_path.read_text())
    require(tuple(selection["selected"]["recon_train"][:4]) == OBJECT_IDS,
            "Fixed first four TRAIN IDs changed; do not replace failed cases")
    native = dataset / "selected48/object_data"
    rows = []
    for object_index, object_id in enumerate(OBJECT_IDS):
        paths = dict(
            touch=native / "touch_charts" / object_id / "touch_charts.npy",
            target=native / "point_cloud_info" / f"{object_id}.npy",
            vertices=native / "object_info" / f"{object_id}_verts.npy",
            faces=native / "object_info" / f"{object_id}_faces.npy",
        )
        data = {key: np.load(path, allow_pickle=False) for key, path in paths.items()}
        all_touch = data["touch"].reshape(50, 4, 25, 4)
        points, vertices, faces = data["target"], data["vertices"], data["faces"]
        require(points.shape == (SAMPLES, 3) and points.dtype == np.float32
                and np.isfinite(points).all(), f"Invalid original point label: {object_id}")
        require(vertices.ndim == 2 and vertices.shape[1] == 3 and len(vertices) >= 3
                and np.isfinite(vertices).all() and (np.ptp(vertices, axis=0) > 0).all(),
                f"Invalid full original mesh vertices: {object_id}")
        require(faces.ndim == 2 and faces.shape[1] == 3 and len(faces) > 0
                and np.issubdtype(faces.dtype, np.integer)
                and faces.min() >= 0 and faces.max() < len(vertices),
                f"Invalid full original mesh indices: {object_id}")
        require(np.isfinite(all_touch).all() and np.isin(all_touch[..., 3], [0, 1, 2]).all(),
                f"Invalid published touch chart: {object_id}")
        # Official canonical normalization has longest extent 1/3.1. No fitted
        # transform, GT alignment, convex hull, mesh repair or point resampling.
        require(abs(float(np.ptp(vertices, axis=0).max()) - 1 / 3.1) < 2e-6,
                f"Unexpected canonical normalization: {object_id}")
        for repeat in range(2):
            seed, grasps = fixed_grasps(object_index, repeat)
            touch = all_touch[grasps, 1].astype(np.float32)
            rows.append(dict(
                object_id=object_id, split="recon_train", repeat=repeat,
                grasp_seed=seed, grasps=grasps,
                name=f"{object_id}_repeat{repeat}", touch=touch, target=points,
                truth_vertices=vertices, truth_faces=faces,
                provenance={key: dict(path=str(path), sha256=sha256(path))
                            for key, path in paths.items()},
                mask_vertex_counts={str(mask): int((touch[..., 3] == mask).sum())
                                    for mask in (0, 1, 2)},
                full_mesh_vertices=len(vertices), full_mesh_faces=len(faces),
            ))
    require(len(rows) == 8, "All eight predeclared inputs are required")
    for i, row in enumerate(rows):
        for other in rows[i+1:]:
            require(not (row["object_id"] != other["object_id"]
                         and np.array_equal(row["touch"], other["touch"])),
                    "Conflicting different-object identical inputs; preserve IDs and investigate")
    return rows, dict(selection_path=str(selection_path), selection_sha256=sha256(selection_path))


def public_row(row):
    return {key: value for key, value in row.items()
            if key not in ("touch", "target", "truth_vertices", "truth_faces")}


def case_gate(current, initial):
    """Gate every observed-input case; empty controls never determine success."""
    valid = all(math.isfinite(float(row[key])) and float(row[key]) >= 0
                for row in (current, initial)
                for key in ("global_cd_x9000", "global_fscore", "official_cd_x9000"))
    valid = valid and current["global_fscore"] <= 1 and initial["global_fscore"] <= 1
    return dict(
        finite_valid_metrics=valid,
        global_chamfer=current["global_cd_x9000"] <= GLOBAL_CD_LIMIT,
        global_fscore=current["global_fscore"] >= FSCORE_MINIMUM,
        official_loss_nonregression=(current["official_cd_x9000"]
            <= initial["official_cd_x9000"] + OFFICIAL_NONREGRESSION_ATOL),
    )


def all_observed_cases_pass(records):
    """A missing/duplicate case is a protocol failure, never a smaller mean."""
    observed = [row for row in records if row["condition"] == "observed"]
    expected = {f"{obj}_repeat{repeat}" for obj in OBJECT_IDS for repeat in range(2)}
    require(len(observed) == 8 and {row["name"] for row in observed} == expected,
            "Every one of the eight fixed observed inputs must occur exactly once")
    return all(row["passed"] for row in observed)


def training_objective(original_loss, vertices, geometry=None):
    """Disabled path returns the same tensor: original loss and gradient exact."""
    if geometry is None:
        return original_loss, {}
    return geometry.augment(original_loss, vertices)


def train(args, output, rows, selection):
    import numpy as np
    import torch
    import pytorch3d
    from pytorch3d.ops import knn_points

    from .retained_execution import require_active_resource
    resource = require_active_resource()
    require(torch.cuda.is_available(), "Allocated CUDA required")
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.cuda.set_device(0)
    utils, model_module = official_imports(args.vendor)
    loaders = importlib.import_module("pterotactyl.utility.data_loaders")
    require(Path(loaders.__file__).resolve() == args.vendor.resolve()
            / "pterotactyl/utility/data_loaders.py", "Original data loader required")
    config = json.loads((args.checkpoint / "config.json").read_text())
    require(config["finger"] and config["use_touch"] and not config["use_img"]
            and config["num_grasps"] == 5 and config["number_points"] == SAMPLES
            and config["loss_coeff"] == LOSS_SCALE and config["lr"] == LEARNING_RATE,
            "Released complete t_p configuration required")
    official_args = SimpleNamespace(**config)
    official_args.eval, official_args.val_grasps = True, 5
    native = args.dataset.resolve() / "selected48/object_data"
    loaders.TOUCH_LOCATION = str(native / "touch_charts") + "/"
    loader = object.__new__(loaders.mesh_loader_vision)
    loader.args = official_args
    for row in rows:
        loader.object_names = [(row["object_id"], row["grasp_seed"])]
        obj, grasps = loader.get_validation_instance(0)
        require(obj == row["object_id"] and grasps == row["grasps"],
                "Fixed action sequence disagrees with the official loader")
        require(np.array_equal(loader.get_touch_info(obj, grasps).numpy(), row["touch"]),
                "Fixed charts disagree with original finger selection and loader")
    official_args.eval = False
    geometry_enabled = bool(getattr(args, "geometry_repair", False))
    if geometry_enabled:
        from . import active3d_geometry_loss
        require(args.checkpoint.resolve() == (EXPERIMENT / "checkpoints/active3d_official/t_p").resolve(),
                "Geometry repair must restart from the original released t_p, not an overfit endpoint")
    resume = None
    start_step, end_step = 0, STEPS
    if getattr(args, "resume_from", None) is not None:
        from . import active3d_continuation as continuation
        resume = continuation.load_resume(args, [public_row(row) for row in rows], selection)
        start_step, end_step = continuation.START, continuation.END
    protocol = dict(
        scope="NATIVE ABC full-model overfit only; separate SUGAR actual-observation gate required",
        stop_boundary="No automatic generalization, SUGAR run, policy training or parameter sweep",
        objects=list(OBJECT_IDS), cases=[public_row(row) for row in rows], **selection,
        model="Unmodified official released t_p Deformation, all three deformation passes",
        parameters=PARAMETERS, graph_vertices=VERTICES, graph_faces=FACES,
        fixed_updates=end_step, batch_size=8, report_every=REPORT_EVERY,
        optimizer=dict(name="Adam", learning_rate=LEARNING_RATE, weight_decay=0),
        training_loss="Unmodified official utils.chamfer_distance over ALL global+touch faces",
        samples=SAMPLES, repeats=REPEATS, loss_scale=LOSS_SCALE,
        labels="All original published 30000-point canonical labels, no resampling or alignment",
        canonical="Original AABB-center / longest extent / 3.1; no predicted/GT pose fitting",
        input="Published five-grasp charts, official finger index 1, xyz+mask only; no object IDs",
        no_contact_case="15737/repeat0 has zero mask2 vertices and remains in every denominator",
        empty_control="All-zero charts, evaluation only; not supervised to reproduce distinct objects",
        global_metrics="2304 global faces only; original Chamfer plus independently sampled bidirectional F-score",
        fscore_sampling="Original batch_sample, 30000 points, one fixed-seed sample; squared PyTorch3D nearest distances compared with 0.01^2",
        gate=dict(every_observed_input=True, maximum_global_cd_x9000=GLOBAL_CD_LIMIT,
                  minimum_global_fscore=FSCORE_MINIMUM, fscore_distance_canonical=FSCORE_DISTANCE,
                  official_loss_must_not_exceed_initial=True,
                  official_loss_absolute_tolerance=OFFICIAL_NONREGRESSION_ATOL,
                  checkpoint_reload_required=True, actual_rendering_required_separately=True),
        seed=args.seed, metric_seed=args.seed+100000, dataset=str(args.dataset.resolve()),
        checkpoint=str(args.checkpoint.resolve()),
        checkpoint_sha256=sha256(args.checkpoint / "model"),
        checkpoint_config_sha256=sha256(args.checkpoint / "config.json"),
        official_model_sha256=sha256(model_module.__file__),
        official_utils_sha256=sha256(utils.__file__),
        torch=torch.__version__, torch_cuda=torch.version.cuda, pytorch3d=pytorch3d.__version__,
        renderer_executed=False, physics_controls=0, retained_resource=resource,
    )
    if geometry_enabled:
        protocol["geometry_repair"] = active3d_geometry_loss.configuration()
        protocol["geometry_repair_source_sha256"] = sha256(active3d_geometry_loss.__file__)
    if resume is not None:
        for key in ('official_model_sha256', 'official_utils_sha256', 'checkpoint_sha256',
                    'checkpoint_config_sha256', 'gate', 'metric_seed', 'optimizer',
                    'samples', 'repeats', 'loss_scale', 'geometry_repair'):
            require(protocol[key] == resume['protocol'][key], 'Continuation contract changed: '+key)
        protocol.update(continuation=resume['receipt'], new_optimizer_updates=end_step-start_step,
                        initial_render_source=str(resume['source']/'step_0000'),
                        continuation_helper_sha256=sha256(continuation.__file__))
    write_json(output / "PROTOCOL.json", protocol)
    write_json(output / "config.json", config)
    utils.set_seeds(args.seed)
    graph, initial_mesh = utils.load_mesh_vision(
        official_args, str(args.vendor / "pterotactyl/objects/vision_charts.obj"))
    require(initial_mesh.shape == (GLOBAL_VERTICES, 3)
            and graph["adj"].shape == (VERTICES, VERTICES)
            and graph["faces"].shape == (FACES, 3), "Full original graph required")
    global_faces = graph["faces"][(graph["faces"] < GLOBAL_VERTICES).all(dim=1)]
    require(global_faces.shape == (2304, 3), "Full global topology required")
    model = model_module.Deformation(graph, initial_mesh, official_args).cuda()
    released_weights = torch.load(args.checkpoint / "model", map_location="cpu", weights_only=True)
    model.load_state_dict(released_weights, strict=True)
    require(sum(p.numel() for p in model.parameters()) == PARAMETERS,
            "Complete official parameter count required")
    require(all(torch.equal(value.cpu(), released_weights[name])
                for name, value in model.state_dict().items()), "Released weight readback mismatch")
    model.requires_grad_(True)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=0)
    starting_weights = released_weights
    if resume is not None:
        starting_weights = continuation.restore(model, optimizer, resume)
    touch = torch.from_numpy(np.stack([row["touch"] for row in rows])).cuda()
    target = torch.from_numpy(np.stack([row["target"] for row in rows])).cuda()
    geometry = None
    if geometry_enabled:
        geometry = active3d_geometry_loss.GeometryRepairLoss(target,
            [row["truth_vertices"] for row in rows], [row["truth_faces"] for row in rows], global_faces)
    image = touch.new_zeros((8, 1))  # Original touch-only image argument is unused.
    require(touch.shape == (8, 5, 25, 4) and target.shape == (8, SAMPLES, 3),
            "Fixed native batch dimensions changed")
    np.savez_compressed(output / "GRAPH.npz", initial_mesh=initial_mesh.cpu().numpy(),
                        global_faces=global_faces.cpu().numpy(),
                        **{name: value.cpu().numpy() for name, value in graph.items()})
    with torch.no_grad():
        observed_charts = model_module.prepare_mesh(
            dict(img=image, touch_charts=touch), initial_mesh, official_args)
        empty_charts = model_module.prepare_mesh(
            dict(img=image, touch_charts=torch.zeros_like(touch)), initial_mesh, official_args)
    counts = dict(model_forwards=0, backwards=0, optimizer_updates=0)

    def forward(charts):
        vertices, masks = model(image, charts)
        counts["model_forwards"] += 1
        require(vertices.shape == (8, VERTICES, 3) and bool(torch.isfinite(vertices).all()),
                "Invalid full mesh output")
        require(torch.equal(vertices[:, GLOBAL_VERTICES:], charts["touch_charts"]),
                "Original fixed chart vertices changed")
        return vertices, masks

    def evaluate(step, initial=None):
        model.eval()
        directory = output / f"step_{step:04d}"
        directory.mkdir()
        records, saved = [], {}
        with torch.no_grad():
            for condition, charts in (("observed", observed_charts), ("empty", empty_charts)):
                vertices, masks = forward(charts)
                # Reset only metric RNG; each next training step sets its own
                # seed, so report frequency cannot change training samples.
                utils.set_seeds(args.seed+100000)
                official = LOSS_SCALE * utils.chamfer_distance(
                    vertices, graph["faces"], target, num=SAMPLES, repeat=REPEATS)
                utils.set_seeds(args.seed+100001)
                global_cd = LOSS_SCALE * utils.chamfer_distance(
                    vertices, global_faces, target, num=SAMPLES, repeat=REPEATS)
                utils.set_seeds(args.seed+100002)
                points = utils.batch_sample(vertices, global_faces, num=SAMPLES)
                pred_to_truth = knn_points(points, target, K=1).dists[..., 0]
                truth_to_pred = knn_points(target, points, K=1).dists[..., 0]
                precision = (pred_to_truth <= FSCORE_DISTANCE**2).float().mean(1)
                recall = (truth_to_pred <= FSCORE_DISTANCE**2).float().mean(1)
                fscore = 2*precision*recall/(precision+recall).clamp_min(1e-12)
                require(all(bool(torch.isfinite(value).all()) for value in
                            (official, global_cd, precision, recall, fscore)),
                        "Nonfinite evaluation metric")
                saved[condition] = (vertices.cpu().numpy(), masks.cpu().numpy())
                geometry_metrics = geometry.evaluate(vertices[:, :GLOBAL_VERTICES]) if geometry is not None else None
                for i, row in enumerate(rows):
                    metrics = dict(official_cd_x9000=float(official[i]),
                                   global_cd_x9000=float(global_cd[i]),
                                   global_precision=float(precision[i]),
                                   global_recall=float(recall[i]), global_fscore=float(fscore[i]))
                    record = dict(name=row["name"], object_id=row["object_id"],
                                  repeat=row["repeat"], condition=condition, metrics=metrics,
                                  mask_vertex_counts=row["mask_vertex_counts"] if condition == "observed"
                                  else {"0": 125, "1": 0, "2": 0})
                    if condition == "observed" and initial is not None:
                        record["checks"] = case_gate(metrics, initial[i]["metrics"])
                        record["passed"] = all(record["checks"].values())
                    if geometry_metrics is not None:
                        record["geometry_metrics"] = geometry_metrics[i]
                        record["geometry_checks"] = active3d_geometry_loss.geometry_case_gate(geometry_metrics[i])
                        record["geometry_passed"] = all(record["geometry_checks"].values())
                    records.append(record)
        for i, row in enumerate(rows):
            np.savez_compressed(directory / f'{row["name"]}.npz',
                input_touch_charts=row["touch"], target_points_canonical=row["target"],
                truth_vertices_canonical=row["truth_vertices"], truth_faces=row["truth_faces"],
                faces=graph["faces"].cpu().numpy(), global_faces=global_faces.cpu().numpy(),
                observed_vertices_canonical=saved["observed"][0][i],
                observed_masks=saved["observed"][1][i],
                empty_vertices_canonical=saved["empty"][0][i], empty_masks=saved["empty"][1][i])
        snapshot = dict(step=step, input_cases=8, records=records,
                        native_numeric_gate_passed=(all_observed_cases_pass(records)
                                                   if initial is not None else None),
                        **counts)
        if geometry is not None:
            snapshot["geometry_gate_passed"] = all(r["geometry_passed"] for r in records if r["condition"] == "observed")
            snapshot["native_repair_numeric_gate_passed"] = bool(snapshot["native_numeric_gate_passed"]
                                                                 and snapshot["geometry_gate_passed"])
        write_json(directory / "METRICS.json", snapshot)
        torch.save(model.state_dict(), directory / "model")
        torch.save(optimizer.state_dict(), directory / "optim")
        print("ACTIVE3D_OVERFIT_EVAL", json.dumps(snapshot, allow_nan=False), flush=True)
        return snapshot, saved["observed"][0]

    if resume is None:
        initial, _ = evaluate(0)
    else:
        initial = continuation.copy_initial_and_check_inputs(resume, output, rows, graph, initial_mesh)
    initial_rows = [r for r in initial["records"] if r["condition"] == "observed"]
    gradient_groups = ("positional_encoder", "mask_encoder", "mesh_deform_1", "mesh_deform_2")
    cumulative_group_nonzero = {name: False for name in gradient_groups}
    started = time.monotonic()
    snapshots = [dict(step=0, metrics="step_0000/METRICS.json", passed=None)]
    resume_replay = None
    if resume is not None:
        snapshots[0]['copied_from_source_not_new_forward'] = True
        resumed, _ = evaluate(start_step, initial_rows)
        resume_replay = continuation.check_replay(resume, output, rows)
        write_json(output/'RESUME_REPLAY.json', resume_replay)
        snapshots.append(dict(step=start_step, metrics=f'step_{start_step:04d}/METRICS.json',
                              passed=resumed['native_numeric_gate_passed'], new_optimizer_updates=0))
    for step in range(start_step+1, end_step+1):
        utils.set_seeds(args.seed+step)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        vertices, _ = forward(observed_charts)
        loss = LOSS_SCALE * utils.chamfer_distance(
            vertices, graph["faces"], target, num=SAMPLES, repeat=REPEATS).mean()
        require(bool(torch.isfinite(loss)) and loss.requires_grad, "Invalid original training loss")
        total_loss, geometry_record = training_objective(loss, vertices[:, :GLOBAL_VERTICES], geometry)
        total_loss.backward()
        counts["backwards"] += 1
        for name, parameter in model.named_parameters():
            require(parameter.grad is not None and bool(torch.isfinite(parameter.grad).all()),
                    f"Missing/nonfinite full-model gradient: {name}")
            group = name.split(".")[0]
            if group in cumulative_group_nonzero and bool(torch.count_nonzero(parameter.grad)):
                cumulative_group_nonzero[group] = True
        optimizer.step()
        counts["optimizer_updates"] += 1
        record = dict(step=step, official_training_loss=float(loss.detach()),
                      elapsed_seconds=time.monotonic()-started, **counts)
        if geometry is not None:
            record.update(total_training_loss=float(total_loss.detach()), geometry_repair=geometry_record)
        with (output / "train.jsonl").open("a") as stream:
            stream.write(json.dumps(record, allow_nan=False)+"\n")
        if step % 10 == 0:
            print("ACTIVE3D_OVERFIT_STEP", json.dumps(record), flush=True)
        if step % REPORT_EVERY == 0:
            final, final_vertices = evaluate(step, initial_rows)
            snapshots.append(dict(step=step, metrics=f"step_{step:04d}/METRICS.json",
                                  passed=final["native_numeric_gate_passed"]))
    # Reload the actual final saved checkpoint into a newly constructed complete
    # model. Do not compare against an in-memory copied model and call it reload.
    endpoint = output / f"step_{end_step:04d}/model"
    reloaded = model_module.Deformation(graph, initial_mesh, official_args).cuda()
    reloaded.load_state_dict(torch.load(endpoint, map_location="cpu", weights_only=True), strict=True)
    reloaded.eval().requires_grad_(False)
    state_exact = all(torch.equal(value, model.state_dict()[name])
                      for name, value in reloaded.state_dict().items())
    with torch.no_grad():
        replay = reloaded(image, observed_charts)[0]
        counts["model_forwards"] += 1
    reload_max = float(np.max(np.abs(replay.cpu().numpy()-final_vertices)))
    require(state_exact and np.isfinite(reload_max) and reload_max <= 1e-7,
            "Final checkpoint reload disagrees with saved full-mesh predictions")
    changed_groups = {group: any(not torch.equal(value.cpu(), starting_weights[name])
                                for name, value in model.state_dict().items()
                                if name.split(".")[0] == group)
                      for group in gradient_groups}
    require(all(cumulative_group_nonzero.values()) and all(changed_groups.values()),
            "Every official model group must receive gradients and update")
    require(counts["optimizer_updates"] == end_step-start_step and counts["backwards"] == end_step-start_step,
            "Fixed budget changed")
    result = dict(
        complete=True, scope="native_only", native_numeric_gate_passed=final["native_numeric_gate_passed"],
        sugar_gate_passed=False, sugar_gate_status="NOT_RUN_SEPARATE_REAL_OBSERVATION_GATE_REQUIRED",
        render_gate_status="PENDING_ACTUAL_MESH_RENDER_AND_INSPECTION", overall_goal_complete=False,
        fixed_input_cases=8, fixed_objects=list(OBJECT_IDS), snapshots=snapshots, **counts,
        checkpoint=str(endpoint), checkpoint_sha256=sha256(endpoint),
        checkpoint_reload=dict(full_state_exact=state_exact, output_max_abs=reload_max, passed=True),
        every_model_group_received_nonzero_gradient=cumulative_group_nonzero,
        every_model_group_changed=changed_groups, final_cases=final["records"],
        all_native_inputs_preserved=True, failed_cases=[r["name"] for r in final["records"]
            if r["condition"] == "observed" and not r["passed"]],
        source_files_unchanged=(sha256(model_module.__file__) == protocol["official_model_sha256"]
                               and sha256(utils.__file__) == protocol["official_utils_sha256"]),
        limitations=["Overfit only; no held-out generalization or policy result.",
                     "Published geometric charts, not learned SUGAR pressure/shear interpretation.",
                     "One fixed input has no mask2 contact vertices; all eight cases remain.",
                     "Original training loss includes immutable touch faces; it need not reach zero.",
                     "Saved full meshes are render inputs, not completed/inspected render images.",
                     "A separate SUGAR actual-observation gate is mandatory before claiming tactile transfer."],
    )
    require(result["source_files_unchanged"], "Official source changed during training")
    if resume is not None:
        require(sha256(continuation.__file__) == protocol['continuation_helper_sha256'],
                'Continuation helper changed during training')
        result.update(continuation=resume['receipt'], total_optimizer_updates=end_step,
                      resume_replay=resume_replay, updates_before_this_run=start_step)
    if geometry is not None:
        require(sha256(active3d_geometry_loss.__file__) == protocol["geometry_repair_source_sha256"],
                "Declared geometry loss changed during training")
        result.update(geometry_repair=protocol["geometry_repair"],
                      geometry_gate_passed=final["geometry_gate_passed"],
                      native_repair_numeric_gate_passed=final["native_repair_numeric_gate_passed"],
                      geometry_failed_cases=[r["name"] for r in final["records"]
                                             if r["condition"] == "observed" and not r["geometry_passed"]])
        result["formal_repair_gate"] = (active3d_geometry_loss.formal_repair_gate(result) if resume is None
                                        else continuation.formal_continuation_gate(result))
    write_json(output / "RESULT.json", result)
    lines = ["# 完整官方 Active3D：固定原生输入 overfit", "",
             f"四个 TRAIN ABC 物体、八组固定五抓取输入；完整 released t_p，累计 {end_step} 次全模型 Adam 更新。",
             "这仅是原生形状资格；SUGAR 实际观测 gate 尚未运行，未证明触觉数字到形状、搬运、材质或泛化。", "",
             f'原生数值 gate：{result["native_numeric_gate_passed"]}。实际完整网格渲染／人工检查仍待执行。', "",
             "| 输入 | 初始→最终官方 loss | 初始→最终全局 CD×9000 | 初始→最终 F@0.01 | 通过 |",
             "|---|---:|---:|---:|---|"]
    for before, after in zip(initial_rows, final["records"][:8], strict=True):
        a, b = before["metrics"], after["metrics"]
        lines.append(f'| {after["name"]} | {a["official_cd_x9000"]:.5f}→{b["official_cd_x9000"]:.5f} | '
                     f'{a["global_cd_x9000"]:.5f}→{b["global_cd_x9000"]:.5f} | '
                     f'{a["global_fscore"]:.3%}→{b["global_fscore"]:.3%} | {after["passed"]} |')
    lines += ["", "所有八例和零触摸对照均完整保存，不按结果换物体或动作。零触摸不作为不同形状精确回归任务。",
              "全局指标仅评价形变后的 1824 顶点／2304 面；训练始终使用官方全部 2464 面损失。",
              f"step_0000 与 step_{end_step:04d} 内每例 NPZ 包含完整真值、原生 charts、实际预测网格及零触摸输出，可直接用于后续真实渲染。",
              "固定预算结束；不自动追加训练、泛化、物理采集或调参。"]
    if geometry is not None:
        lines += ["", f'显式几何修复：旧数值 gate={result["native_numeric_gate_passed"]}；新增几何 gate={result["geometry_gate_passed"]}。',
                  f"新增 loss 只使用 GT 监督，不进入模型输入；完整原模型、固定八例，累计 {end_step} 更新。",
                  "正式修复 gate 还必须实际渲染并逐例检查；有限面内探针不是 Hausdorff/无自交证明。",
                  "接缝仅诊断，不硬焊，不对渲染做平滑或后处理。"]
    if resume is not None:
        lines += ['', f'本次仅新增 {end_step-start_step} 更新；完整恢复原 step{start_step} model+Adam，原损失/学习率/阈值不变。',
                  'step_0000 是原 released 输入/输出原样复制；step_1000 实际重放核对，均不计入本次新增更新。',
                  '每步随机种子为原 seed+绝对step；原Engine八输入等价PASS绑定于continuation记录。']
    (output / "REPORT.md").write_text("\n".join(lines)+"\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=EXPERIMENT / "datasets/active3d_official")
    parser.add_argument("--vendor", type=Path, default=EXPERIMENT / "vendor/Active-3D-Vision-and-Touch")
    parser.add_argument("--checkpoint", type=Path, default=EXPERIMENT / "checkpoints/active3d_official/t_p")
    parser.add_argument("--output", type=Path, help="Required for training; must not exist")
    parser.add_argument("--seed", type=int, default=20260916)
    parser.add_argument("--check", action="store_true", help="NumPy-only saved-input check; no model or CUDA import")
    parser.add_argument("--geometry-repair", action="store_true",
                        help="Explicit geometry-loss repair; default preserves original objective exactly")
    parser.add_argument('--resume-from', type=Path,
                        help='Complete geometry step1000 source; exactly +3000 to total4000, no other budget')
    parser.add_argument('--engine-verification', type=Path,
                        help='Original Engine equivalence RESULT.json; required before actual continuation')
    args = parser.parse_args()
    rows, selection = load_cases(args.dataset)
    if args.check:
        if args.resume_from is not None:
            from .active3d_continuation import load_resume
            resume = load_resume(args, [public_row(row) for row in rows], selection,
                                 require_engine=args.engine_verification is not None)
            print(json.dumps(dict(cpu_continuation_check=True, continuation=resume['receipt'],
                                  actual_engine_receipt_checked=args.engine_verification is not None,
                                  model_forwards=0, optimizer_updates=0), indent=2))
            return
        print(json.dumps(dict(cpu_input_check=True, cases=[public_row(row) for row in rows],
                              **selection, model_forwards=0, physics_controls=0), indent=2))
        return
    if args.output is None:
        parser.error("--output NEW_DIRECTORY is required for training")
    args.output.mkdir(parents=True, exist_ok=False)
    try:
        result = train(args, args.output, rows, selection)
    except Exception as error:
        write_json(args.output / "FAILURE.json", dict(complete=False, expected_input_cases=8,
                   fixed_objects=list(OBJECT_IDS), error=repr(error), traceback=traceback.format_exc(),
                   scope="native_only", sugar_gate_status="NOT_RUN", overall_goal_complete=False,
                   note="Existing snapshots/logs preserve completed work; no case substitution or automatic retry"))
        raise
    numeric = result["native_repair_numeric_gate_passed"] if args.geometry_repair else result["native_numeric_gate_passed"]
    raise SystemExit(0 if numeric else 2)


if __name__ == "__main__":
    main()
