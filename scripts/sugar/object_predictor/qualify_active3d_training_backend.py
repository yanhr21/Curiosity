"""Qualify the full official Active3D t_p model and official Chamfer backward.

Eight fixed TRAIN clocks, no optimizer and no parameter updates. Targets are
area samples of the saved true full mesh in the existing predicted coordinate
frame. This checks the training BACKEND, not the original voxel/ODM data pipeline
or learning/generalization. No model/loss/graph implementation is replaced.
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path
import sys
import time
import traceback
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT = ROOT / "experiments/object_predictor_v1"
EXPECTED = [(ep, frame) for ep in range(5000, 5004) for frame in (1206, 2206)]
SAMPLES = 30000
REPEATS = 3
LOSS_SCALE = 9000.0
VERTICES = 1949
GLOBAL_VERTICES = 1824
FACES = 2464
PARAMETERS = 1033599


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def resolve_source(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def load_train_cases(input_dir):
    """Read only the fixed TRAIN labels; never use truth to alter input charts."""
    import numpy as np
    from scipy.spatial.transform import Rotation

    input_dir = Path(input_dir).resolve()
    report = json.loads((input_dir / "RESULT.json").read_text())
    require(report["complete"], "Saved chart preparation is incomplete")
    require([(r["episode"], r["frame"]) for r in report["cases"]] == EXPECTED,
            "Require the original fixed eight TRAIN chart clocks in order")
    collection = json.loads((EXPERIMENT / "response_surface_dataset_v1/COLLECTION_RESULT.json").read_text())
    records = {r["episode"]: r for r in collection["records"]}
    mesh_path = EXPERIMENT / "rendered_carry_v1/object_mesh.npz"
    with np.load(mesh_path, allow_pickle=False) as z:
        reference = z["vertices"].astype(np.float64)
        faces = z["faces"].astype(np.int64)
    require(reference.ndim == 2 and reference.shape[1] == 3 and np.isfinite(reference).all(),
            "Invalid full reference mesh vertices")
    require(faces.ndim == 2 and faces.shape[1] == 3 and faces.min() >= 0 and faces.max() < len(reference),
            "Invalid full reference mesh triangles")
    dimensions = np.ptp(reference, axis=0)
    require(bool((dimensions > 0).all()), "Degenerate reference mesh dimensions")
    reference -= (reference.max(0) + reference.min(0)) / 2
    cases = []
    for entry in report["cases"]:
        episode, frame = entry["episode"], entry["frame"]
        record = records[episode]
        require(record["split"] == "train", f"Refusing non-TRAIN label: {episode}")
        path = input_dir / entry["input_file"]
        with np.load(path, allow_pickle=False) as z:
            touch = z["t_p_charts"].astype(np.float32).reshape(1, 125, 4)
            center = z["center_w"].astype(np.float64)
            orientation = z["orientation_w"].astype(np.float64)
            scale = float(z["canonical_scale_m"])
        require(np.isfinite(touch).all() and np.isin(touch[..., 3], [0, 1, 2]).all(),
                f"Invalid saved t_p charts: {path}")
        require(center.shape == (3,) and np.isfinite(center).all() and np.isfinite(scale) and scale > 0,
                "Invalid frozen predicted canonical frame")
        require(orientation.shape == (3, 3) and np.isfinite(orientation).all()
                and np.allclose(orientation.T @ orientation, np.eye(3), atol=2e-6, rtol=0)
                and abs(np.linalg.det(orientation) - 1) < 2e-6,
                "Saved predicted orientation is not a proper rotation")
        trace_path = resolve_source(record["source"]) / f"episode_{episode}.npz"
        with np.load(trace_path, allow_pickle=False) as z:
            pose = z["object_pose_w"][frame].astype(np.float64)
            true_rotation = Rotation.from_quat(pose[3:]).as_matrix()
            true_center = pose[:3] + z["object_local_center_m"][frame] @ true_rotation.T
            true_size = z["object_dimensions_m"][frame].astype(np.float64)
        require(np.isfinite(true_size).all() and (true_size > 0).all(), "Invalid TRAIN object dimensions")
        truth_world = (reference * (true_size / dimensions)) @ true_rotation.T + true_center
        # Frozen prediction frame defines coordinates for BOTH input and target.
        # Only the target contains GT. No GT alignment/correction of the input.
        truth_canonical = ((truth_world - center) @ orientation) / scale
        reconstructed = truth_canonical * scale @ orientation.T + center
        roundtrip = float(np.abs(reconstructed - truth_world).max())
        require(np.isfinite(truth_canonical).all() and roundtrip < 1e-6,
                "TRAIN target canonical/world roundtrip failed")
        cases.append(dict(episode=episode, frame=frame, input_file=str(path),
                          label_trace=str(trace_path), fullmesh_file=str(mesh_path),
                          touch=touch, center=center, orientation=orientation, scale=scale,
                          truth_world=truth_world, truth_canonical=truth_canonical,
                          truth_faces=faces, roundtrip_max_m=roundtrip))
    return cases


def tensor_stats(tensor):
    """Zeros are reported, not hidden; not every parameter element is active."""
    import torch

    if tensor is None:
        return dict(present=False, finite=False, nonzero_elements=0)
    tensor = tensor.detach()
    finite = bool(torch.isfinite(tensor).all())
    return dict(present=True, shape=list(tensor.shape), elements=tensor.numel(),
                finite=finite, nonzero_elements=int(torch.count_nonzero(tensor)),
                l2=float(torch.linalg.vector_norm(tensor.double())) if finite else None,
                absmax=float(tensor.abs().max()) if finite and tensor.numel() else None)


def official_imports(vendor):
    """Import original utils including all its genuine dependencies; no shim."""
    vendor = vendor.resolve()
    sys.path.insert(0, str(vendor))
    os.environ.setdefault("MPLBACKEND", "Agg")
    utils = importlib.import_module("pterotactyl.utility.utils")
    model = importlib.import_module("pterotactyl.reconstruction.vision.model")
    for module, relative in ((utils, "pterotactyl/utility/utils.py"),
                             (model, "pterotactyl/reconstruction/vision/model.py")):
        require(Path(module.__file__).resolve() == vendor / relative,
                f"Wrong official module resolved: {module.__file__}")
    require(utils.chamfer_distance.__module__ == utils.__name__, "Original Chamfer import required")
    require(utils.cuda_cd.__module__.startswith("pytorch3d."), "Original PyTorch3D Chamfer required")
    return utils, model


def qualify(args, output):
    import numpy as np
    import torch
    import pytorch3d

    cases = load_train_cases(args.inputs)
    require(os.environ.get("SLURM_STEP_ID"), "Run only in the retained explicit compute step")
    device = torch.device(args.device)
    require(device.type == "cuda" and torch.cuda.is_available(), "An allocated CUDA device is required")
    if device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.cuda.set_device(device)
    utils, model_module = official_imports(Path(args.vendor))
    checkpoint = Path(args.checkpoint)
    config = json.loads((checkpoint / "config.json").read_text())
    require(config["finger"] and config["use_touch"] and not config["use_img"]
            and config["num_grasps"] == 5, "Require the complete released t_p model")
    require(config["number_points"] == SAMPLES and config["loss_coeff"] == LOSS_SCALE,
            "Official checkpoint training loss contract changed")
    official_args = SimpleNamespace(**config)
    protocol = dict(
        scope="Training backend qualification only; no learning or generalization result.",
        cases=EXPECTED, split="train", batch_size=1, model_parameter_updates=0,
        optimizer_constructed=False, physics_steps=0, model_mode="train",
        expected_model_forwards=8, expected_backwards=8,
        model="Complete official released t_p Deformation, all three deformation passes",
        graph_vertices=VERTICES, graph_faces=FACES,
        loss="Direct original pterotactyl.utility.utils.chamfer_distance",
        prediction_area_samples_per_repeat=SAMPLES, repeats=REPEATS,
        loss_scale=LOSS_SCALE, target_points=SAMPLES,
        target="Saved TRAIN GT fullmesh, sampled by original utils.batch_sample under no_grad",
        target_pipeline_difference="Area sampling of saved true mesh, NOT the original extract_points 128^3 voxel/ODM/fill/surface/realign label pipeline.",
        coordinates="Both tensors use the saved frozen Utonia predicted canonical frame. GT modifies only the target, never charts or input frame.",
        fixed_touch_vertices="All 125 touch vertices and their 160 faces remain in Chamfer; original model deforms only the 1824 global vertices.",
        gradient_scope="Report every parameter tensor/element count, each named group, mask embedding rows, final vertices, and all three full GCN updates. Some elements are structurally unused.",
        torch=torch.__version__, torch_cuda=torch.version.cuda, pytorch3d=pytorch3d.__version__,
        utils_source=str(Path(utils.__file__).resolve()), model_source=str(Path(model_module.__file__).resolve()),
        checkpoint=str(checkpoint.resolve()),
    )
    write_json(output / "PROTOCOL.json", protocol)
    # Use even the original OBJ loader and graph construction, now that the real
    # PyTorch3D environment exists. Do not use inference-only AST import glue.
    utils.set_seeds(args.seed)
    graph, initial_mesh = utils.load_mesh_vision(official_args, str(Path(args.vendor) / "pterotactyl/objects/vision_charts.obj"))
    require(initial_mesh.shape == (GLOBAL_VERTICES, 3), "Global template was reduced")
    require(graph["adj"].shape == (VERTICES, VERTICES) and graph["faces"].shape == (FACES, 3),
            "Full global+touch graph is required")
    require(bool((graph["faces"][:2304] < GLOBAL_VERTICES).all())
            and bool((graph["faces"][2304:] >= GLOBAL_VERTICES).all()), "Unexpected global/touch face order")
    for key in ("adj", "origional"):
        require(bool(torch.isfinite(graph[key]).all()) and torch.allclose(
            graph[key].sum(1), torch.ones(len(graph[key]), device=device), atol=2e-6, rtol=0),
            f"Invalid original normalized adjacency: {key}")
    model = model_module.Deformation(graph, initial_mesh, official_args).cuda()
    weights = torch.load(checkpoint / "model", map_location="cpu", weights_only=True)
    model.load_state_dict(weights, strict=True)
    require(sum(p.numel() for p in model.parameters()) == PARAMETERS, "Incomplete official t_p model")
    require(all(torch.equal(value.cpu(), weights[name]) for name, value in model.state_dict().items()),
            "Strict full checkpoint readback differs")
    model.train().requires_grad_(True)
    require(all(value.device == device for value in graph.values())
            and initial_mesh.device == device and all(p.device == device for p in model.parameters()),
            "Model and complete graph must share the selected CUDA device")
    np.savez_compressed(output / "GRAPH.npz", initial_mesh=initial_mesh.cpu().numpy(),
                        **{name: tensor.cpu().numpy() for name, tensor in graph.items()})
    stages = []

    def save_stage(_module, _inputs, update):
        update.retain_grad()
        stages.append(update)

    hooks = [model.mesh_deform_1.register_forward_hook(save_stage),
             model.mesh_deform_2.register_forward_hook(save_stage)]
    rows = []
    for case_index, case in enumerate(cases):
        stages.clear()
        model.zero_grad(set_to_none=True)
        utils.set_seeds(args.seed + case_index)
        info = torch.as_tensor(case["touch"], dtype=torch.float32, device=device)
        image_batch = info.new_zeros((1, 1))  # Official touch-only batch convention.
        truth_vertices = torch.as_tensor(case["truth_canonical"], dtype=torch.float32, device=device).unsqueeze(0)
        truth_faces = torch.as_tensor(case["truth_faces"], dtype=torch.long, device=device)
        with torch.no_grad():
            target = utils.batch_sample(truth_vertices, truth_faces, num=SAMPLES)
            charts = model_module.prepare_mesh(dict(img=image_batch, touch_charts=info), initial_mesh, official_args)
        require(target.shape == (1, SAMPLES, 3) and not target.requires_grad
                and bool(torch.isfinite(target).all()), "Invalid TRAIN target sample")
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        vertices, mask = model(image_batch, charts)
        vertices.retain_grad()
        require(vertices.shape == (1, VERTICES, 3) and bool(torch.isfinite(vertices).all()), "Invalid full model output")
        require(torch.equal(vertices[:, GLOBAL_VERTICES:], info[..., :3]), "Fixed touch charts changed")
        require(len(stages) == 3 and all(s.shape == vertices.shape for s in stages), "Missing original deformation pass")
        # This is the actual, unmodified official loss. It area-samples the
        # global AND touch faces 3 times and calls pytorch3d.loss each time.
        cd = utils.chamfer_distance(vertices, graph["faces"], target, num=SAMPLES, repeat=REPEATS)
        loss = LOSS_SCALE * cd.mean()
        require(bool(torch.isfinite(loss)) and loss.requires_grad, "Chamfer loss is not finite/differentiable")
        loss.backward()
        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started
        gradients = {name: p.grad.detach().cpu().clone() if p.grad is not None else None
                     for name, p in model.named_parameters()}
        parameter_stats = {name: tensor_stats(gradient) for name, gradient in gradients.items()}
        groups = {}
        for group in ("positional_encoder", "mask_encoder", "mesh_deform_1", "mesh_deform_2"):
            selected = [value for name, value in parameter_stats.items() if name.split(".")[0] == group]
            groups[group] = dict(tensors=len(selected), all_present=all(v["present"] for v in selected),
                                 all_finite=all(v["finite"] for v in selected),
                                 nonzero_elements=sum(v["nonzero_elements"] for v in selected))
        stage_stats = [dict(pass_index=i + 1, module="mesh_deform_1" if i == 0 else "mesh_deform_2",
                            global_update_gradient=tensor_stats(s.grad[:, :GLOBAL_VERTICES] if s.grad is not None else None),
                            touch_update_gradient=tensor_stats(s.grad[:, GLOBAL_VERTICES:] if s.grad is not None else None))
                       for i, s in enumerate(stages)]
        checks = dict(
            every_parameter_tensor_has_finite_gradient=all(v["present"] and v["finite"] for v in parameter_stats.values()),
            every_model_group_has_nonzero_gradient=all(v["nonzero_elements"] > 0 for v in groups.values()),
            all_three_global_updates_receive_finite_nonzero_gradient=all(
                s["global_update_gradient"]["finite"] and s["global_update_gradient"]["nonzero_elements"] > 0 for s in stage_stats),
            all_three_touch_updates_receive_exact_zero_gradient=all(
                s["touch_update_gradient"]["finite"] and s["touch_update_gradient"]["nonzero_elements"] == 0 for s in stage_stats),
            final_vertex_gradient_finite=tensor_stats(vertices.grad)["finite"],
            target_not_differentiated=target.grad is None and not target.requires_grad,
            fixed_touch_output_exact=True,
        )
        stem = f"episode_{case['episode']}_frame_{case['frame']}"
        torch.save(gradients, output / f"{stem}_parameter_gradients.pt")
        arrays = dict(vertices_canonical=vertices.detach().cpu().numpy()[0],
                      vertices_world_m=vertices.detach().cpu().numpy()[0] * case["scale"] @ case["orientation"].T + case["center"],
                      faces=graph["faces"].cpu().numpy(), masks=mask.detach().cpu().numpy()[0],
                      target_points_canonical=target.cpu().numpy()[0],
                      truth_vertices_canonical=case["truth_canonical"], truth_vertices_world_m=case["truth_world"],
                      truth_faces=case["truth_faces"], input_touch_charts=case["touch"][0],
                      final_vertex_gradient=vertices.grad.cpu().numpy()[0] if vertices.grad is not None else np.empty((0, 3)),
                      center_w=case["center"], orientation_w=case["orientation"], canonical_scale_m=case["scale"])
        for i, stage in enumerate(stages):
            arrays[f"deformation_{i+1}_update"] = stage.detach().cpu().numpy()[0]
            arrays[f"deformation_{i+1}_gradient"] = stage.grad.cpu().numpy()[0] if stage.grad is not None else np.empty((0, 3))
        np.savez_compressed(output / f"{stem}_graph_output.npz", **arrays)
        mask_gradient = gradients["mask_encoder.model.0.weight"]
        row = dict(episode=case["episode"], frame=case["frame"], split="train", seed=args.seed + case_index,
                   input_file=case["input_file"], label_trace=case["label_trace"], fullmesh_file=case["fullmesh_file"],
                   target_roundtrip_max_m=case["roundtrip_max_m"], full_gt_vertices=len(case["truth_world"]),
                   full_gt_faces=len(case["truth_faces"]), official_chamfer=float(cd.detach().mean()),
                   scaled_loss=float(loss.detach()), forward_loss_backward_seconds=elapsed,
                   parameter_gradients=f"{stem}_parameter_gradients.pt", graph_output=f"{stem}_graph_output.npz",
                   parameter_stats=parameter_stats, groups=groups, deformation_passes=stage_stats,
                   mask_embedding_row_gradients=[tensor_stats(mask_gradient[i] if mask_gradient is not None else None) for i in range(4)],
                   final_global_vertex_gradient=tensor_stats(vertices.grad[:, :GLOBAL_VERTICES] if vertices.grad is not None else None),
                   final_touch_vertex_gradient=tensor_stats(vertices.grad[:, GLOBAL_VERTICES:] if vertices.grad is not None else None),
                   checks=checks, passed=all(checks.values()))
        rows.append(row)
        write_json(output / f"{stem}_gradient_report.json", row)
        print("ACTIVE3D_BACKEND_CLOCK", json.dumps({k: row[k] for k in ("episode", "frame", "scaled_loss", "passed")}), flush=True)
        require(row["passed"], f"Gradient qualification failed at {stem}; saved all available outputs and gradients")
        del vertices, mask, loss, cd, gradients, target, charts
    for hook in hooks:
        hook.remove()
    unchanged = all(torch.equal(value.cpu(), weights[name]) for name, value in model.state_dict().items())
    require(unchanged, "Model state changed despite zero optimizer updates")
    result = dict(complete=True, passed=True, protocol=protocol, parameters=PARAMETERS,
                  parameter_tensors=len(list(model.parameters())), complete_model_state_unchanged=unchanged,
                  model_parameter_updates=0, actual_forwards=len(rows), actual_backwards=len(rows), cases=rows,
                  limitations=["Backend forward/backward qualification is not successful learning or generalization.",
                               "Only eight TRAIN clocks of the existing known box; no new shapes, mass/material or policy learning.",
                               "Targets use direct fullmesh area sampling; the original voxel/ODM point-label pipeline is not reproduced.",
                               "Saved charts retain the existing small-contact/interpolation and frozen Utonia frame limitations.",
                               "A nonzero final touch-vertex loss gradient is not a trainable touch deformation: official touch positions stay fixed; GCN update gradients there are zero."])
    write_json(output / "RESULT.json", result)
    print("ACTIVE3D_TRAINING_BACKEND_PASS", json.dumps(dict(cases=len(rows), parameters=PARAMETERS, updates=0)), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", default=str(EXPERIMENT / "research_review_20260916/active3d_chart_inputs_v1"))
    parser.add_argument("--vendor", default=str(EXPERIMENT / "vendor/Active-3D-Vision-and-Touch"))
    parser.add_argument("--checkpoint", default=str(EXPERIMENT / "checkpoints/active3d_official/t_p"))
    parser.add_argument("--output", help="New directory, required for GPU execution; must not already exist")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20260916)
    parser.add_argument("--check", action="store_true", help="Read fixed TRAIN inputs and target coordinates only; no torch, model or CUDA import")
    args = parser.parse_args()
    if args.check:
        cases = load_train_cases(args.inputs)
        print(json.dumps(dict(cpu_input_check=True, cases=len(cases), fixed_clocks=EXPECTED,
                              max_target_roundtrip_m=max(c["roundtrip_max_m"] for c in cases),
                              model_forwards=0, backwards=0, model_parameter_updates=0), indent=2))
        return
    require(args.output, "--output is required for GPU qualification")
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    try:
        qualify(args, output)
    except Exception as exc:
        write_json(output / "ERROR.json", dict(complete=False, passed=False,
                   error_type=type(exc).__name__, error=str(exc),
                   missing_module=exc.name if isinstance(exc, ModuleNotFoundError) else None,
                   traceback=traceback.format_exc(), substitute_used=False, model_parameter_updates=0))
        raise


if __name__ == "__main__":
    main()
