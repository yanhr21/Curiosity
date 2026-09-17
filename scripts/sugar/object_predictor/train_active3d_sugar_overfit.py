"""Full official Active3D overfit on qualified, saved SUGAR touch sequences.

Four fixed TRAIN ABC meshes x two groups x five independent real fixture probes.
Inputs are observed geometry charts in the PUBLIC fixed fixture frame, not
native optical charts or GT geometry. All forty acquisition attempts must pass
the predeclared physical/chart qualification before any model/CUDA import.
Missing, failed, reordered or replaced attempts are rejected, never padded away.

The complete released t_p is initialized independently of native overfit. Model,
graph, original loss, optimizer, 1000-step budget and geometric gates match the
native qualification. No pressure-to-shape encoder, unknown-pose estimator,
material estimator, policy or held-out generalization is claimed. Five probes
are accumulated independent touches, not one continuous carrying rollout.

--check reads only saved observations/qualification and independent GT labels;
it neither creates a model nor initializes CUDA. Training additionally requires
--output NEW_DIRECTORY in the retained compute step. Rendering remains separate.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time
import traceback
from types import SimpleNamespace

from .qualify_active3d_training_backend import (
    EXPERIMENT, FACES, GLOBAL_VERTICES, LOSS_SCALE, PARAMETERS, REPEATS,
    SAMPLES, VERTICES, official_imports, require, write_json,
)
from .train_active3d_overfit import (
    OBJECT_IDS, STEPS, REPORT_EVERY, LEARNING_RATE, GLOBAL_CD_LIMIT,
    FSCORE_DISTANCE, FSCORE_MINIMUM, OFFICIAL_NONREGRESSION_ATOL,
    all_observed_cases_pass, case_gate, public_row, sha256,
)


def validate_qualification(result, protocol):
    """Reject incomplete physical denominators without selecting input charts."""
    from .report_fixture_sequences import DIRECTION_GROUPS, validate_sequence_protocol
    from .report_fixture_touch import QUALIFICATION_CRITERIA, qualification_checks

    attempts = validate_sequence_protocol(protocol)
    require(protocol["objects"] == list(OBJECT_IDS), "Fixed four TRAIN objects required")
    require("fixture_drive" in protocol, "Actual fixed fixture controller protocol required")
    require(result.get("complete") is True and result.get("qualification_passed") is True,
            "All forty actual acquisitions must qualify before SUGAR training")
    for key, expected in dict(expected_attempts=40, complete_recordings=40,
                              available_snapshot_charts=40, sequence_count=8,
                              charts_per_sequence=5, snapshot_frame=499).items():
        require(result.get(key) == expected, f"Incomplete/changed SUGAR denominator: {key}")
    require(result.get("input_file") == "sequence_inputs.npz"
            and result.get("input_key") == "charts"
            and result.get("input_shape") == [8, 5, 25, 4], "Unexpected actual sequence schema")
    criteria = result["qualification_criteria"]
    require(criteria.get("expected_attempts") == 40, "Forty-attempt qualification criteria required")
    for key, value in QUALIFICATION_CRITERIA.items():
        if key not in ("expected_attempts", "scope"):
            require(criteria.get(key) == value, f"Acquisition threshold changed: {key}")
    cases = result["cases"]
    require(len(cases) == 40, "Preserve all forty actual probe records")
    for row, expected in zip(cases, attempts, strict=True):
        require(all(row.get(key) == value for key, value in expected.items()),
                "Probe identity/order changed; do not replace failed directions")
        identity = f'object_{row["object_id"]}_direction_{row["direction_index"]:02d}'
        require(row.get("directory") == identity
                and row.get("chart_file") == f"sequence_charts/{identity}.npz",
                "Unexpected probe/chart source path")
        checks = qualification_checks(row)
        checks["record_integrity"] = not row.get("errors", [])
        require(row.get("qualification_checks") == checks and all(checks.values())
                and row.get("qualification_passed") is True,
                f"Actual probe failed declared qualification: {identity}")
        actual = row.get("attempt_record", {})
        require(actual.get("complete") is True
                and actual.get("recorded_controls") == 600
                and actual.get("actual_controls") == 600
                and actual.get("actual_physics_substeps") == 4800
                and actual.get("partial_control_substeps") == 0
                and all(actual.get(key) == value for key, value in expected.items()),
                f"Incomplete actual physical attempt: {identity}")
    sequences = result["sequences"]
    require(len(sequences) == 8, "All eight actual sequences required")
    for i, row in enumerate(sequences):
        object_id, group = OBJECT_IDS[i // 2], i % 2
        require(row.get("sequence_index") == i and row.get("object_id") == object_id
                and row.get("direction_group") == group
                and row.get("direction_indices") == list(DIRECTION_GROUPS[group])
                and row.get("attempt_indices") == list(range(i*5, i*5+5))
                and row.get("chart_files") == [r["chart_file"] for r in cases[i*5:i*5+5]]
                and row.get("observed_chart_count") == 5
                and row.get("qualification_passed") is True,
                f"SUGAR sequence identity/qualification changed: index {i}")
    return cases, sequences


def verify_observation_chart(path, trace_path, expected_chart):
    """Replay only sensor-local support and actual hand pose; no GT mesh read."""
    import numpy as np
    from scipy.spatial.transform import Rotation
    from .shape_fixture_assets import FIXTURE_CENTER_M, METERS_PER_CANONICAL_UNIT

    with np.load(path, allow_pickle=False) as z:
        chart = z["chart"]
        require(chart.shape == (1, 25, 4) and chart.dtype == np.float32
                and np.array_equal(chart[0], expected_chart)
                and np.isfinite(chart).all() and (chart[..., 3] == 2).all(),
                "Qualified observed chart must match its exact saved sequence slot")
        pos, indices, bary = z["source_pos_hand_m"], z["source_frame_indices"], z["barycentric"]
        active, pads, selected_pad = z["source_active"], z["source_pad"], int(z["selected_pad"])
        area, pressure = z["source_area_m2"], z["source_pressure_pa"]
        pose, hand, world = z["hand_pose_w"], z["chart_hand_m"], z["chart_world_m"]
        edges = z["support_edge_m"]
    require(pos.ndim == 2 and pos.shape[1] == 3 and np.isfinite(pos).all()
            and indices.shape == (25, 3) and np.issubdtype(indices.dtype, np.integer)
            and indices.min() >= 0 and indices.max() < len(pos), "Invalid observed support indices")
    require(all(v.shape == (len(pos),) for v in (active, pads, area, pressure))
            and np.isfinite(area).all() and np.isfinite(pressure).all()
            and active[indices].all() and (pads[indices] == selected_pad).all()
            and (area[indices] > 0).all() and (pressure[indices] > 0).all(),
            "Interpolation support must be active observed points on the selected pad")
    require(bary.shape == (25, 3) and np.isfinite(bary).all()
            and (bary >= -1e-10).all() and np.allclose(bary.sum(1), 1., atol=1e-8, rtol=0),
            "Invalid observation interpolation weights")
    replay = np.einsum("ni,nij->nj", bary, pos[indices])
    require(hand.shape == (25, 3) and np.allclose(replay, hand, atol=1e-12, rtol=0),
            "Saved local chart does not replay from real observed support")
    require(pose.shape == (1, 7) and np.isfinite(pose).all()
            and np.isclose(np.linalg.norm(pose[0, 3:]), 1., atol=1e-6, rtol=0),
            "Invalid measured hand pose")
    with np.load(trace_path, allow_pickle=False) as trace:
        measured = trace["hand_pose_w"]
        times = trace["time_s"]
        require(measured.shape == (600, 7) and times.shape == (600,)
                and np.array_equal(measured[499], pose[0])
                and np.isclose(times[499], 10., atol=1e-9, rtol=0),
                "Chart frame must be actual measured frame499 at 10 s")
    replay_world = replay @ Rotation.from_quat(pose[0, 3:]).as_matrix().T + pose[0, :3]
    canonical = (replay_world - np.asarray(FIXTURE_CENTER_M)) / METERS_PER_CANONICAL_UNIT
    require(world.shape == (25, 3) and np.allclose(replay_world, world, atol=1e-12, rtol=0)
            and np.allclose(canonical, chart[0, :, :3], atol=1e-7, rtol=0),
            "Measured hand/public fixture canonical transform disagrees")
    require(edges.shape == (25,) and np.isfinite(edges).all()
            and (edges >= 0).all() and (edges <= .0017+1e-12).all(),
            "Observation support exceeds fixed 1.7 mm chart edge limit")
    return dict(chart_file=str(path), chart_sha256=sha256(path), trace_file=str(trace_path),
                trace_sha256=sha256(trace_path), selected_pad=selected_pad,
                support_points=len(pos), canonical_roundtrip_max=float(abs(canonical-chart[0, :, :3]).max()))


def load_sugar_cases(root, prepared):
    """Qualification precedes labels and any model import; no native fallback."""
    import numpy as np
    from .shape_fixture_assets import FIXTURE_CENTER_M, METERS_PER_CANONICAL_UNIT

    root, prepared = Path(root).resolve(), Path(prepared).resolve()
    result = json.loads((root / "RESULT.json").read_text())
    protocol = json.loads((root / "PROTOCOL.json").read_text())
    cases, sequences = validate_qualification(result, protocol)
    chart_protocol = json.loads((root / "sequence_charts/PROTOCOL.json").read_text())
    require(chart_protocol.get("fixture_center_m") == list(FIXTURE_CENTER_M)
            and chart_protocol.get("meters_per_canonical_unit") == METERS_PER_CANONICAL_UNIT
            and chart_protocol.get("snapshot_frame") == 499
            and chart_protocol.get("max_interpolation_edge_m") == .0017,
            "Public observation coordinate/support contract changed")
    with np.load(root / "sequence_inputs.npz", allow_pickle=False) as z:
        require(z.files == ["charts"], "Only numeric observed charts may occur in sequence NPZ")
        inputs = z["charts"]
    require(inputs.shape == (8, 5, 25, 4) and inputs.dtype == np.float32
            and np.isfinite(inputs).all() and (inputs[..., 3] == 2).all(),
            "Every qualified sequence must retain all five actual charts unchanged")
    observation_sources = []
    for index, row in enumerate(cases):
        observation_sources.append(verify_observation_chart(root / row["chart_file"],
            root / row["directory"] / "trace.npz", inputs[index // 5, index % 5]))
    # Input construction is complete before any GT object label/mesh is opened.
    rows = []
    label_cache = {}
    for i, sequence in enumerate(sequences):
        obj = sequence["object_id"]
        if obj not in label_cache:
            paths = dict(target=prepared / "object_data/point_cloud_info" / f"{obj}.npy",
                         vertices=prepared / "object_data/object_info" / f"{obj}_verts.npy",
                         faces=prepared / "object_data/object_info" / f"{obj}_faces.npy")
            arrays = {key: np.load(path, allow_pickle=False) for key, path in paths.items()}
            target, vertices, faces = arrays["target"], arrays["vertices"], arrays["faces"]
            require(target.shape == (SAMPLES, 3) and target.dtype == np.float32
                    and np.isfinite(target).all(), f"Missing/invalid original 30000-point label: {obj}")
            require(vertices.ndim == 2 and vertices.shape[1] == 3 and np.isfinite(vertices).all()
                    and abs(float(np.ptp(vertices, axis=0).max())-1/3.1) < 2e-6,
                    f"Invalid complete canonical label mesh: {obj}")
            require(faces.ndim == 2 and faces.shape[1] == 3 and len(faces) > 0
                    and np.issubdtype(faces.dtype, np.integer)
                    and faces.min() >= 0 and faces.max() < len(vertices), "Invalid complete mesh indices")
            label_cache[obj] = (target, vertices, faces,
                {key: dict(path=str(path), sha256=sha256(path)) for key, path in paths.items()})
        target, vertices, faces, provenance = label_cache[obj]
        rows.append(dict(object_id=obj, split="recon_train", repeat=sequence["direction_group"],
            name=f'{obj}_repeat{sequence["direction_group"]}', direction_indices=sequence["direction_indices"],
            touch=inputs[i].copy(), target=target, truth_vertices=vertices, truth_faces=faces,
            provenance=dict(labels=provenance, observations=observation_sources[i*5:i*5+5]),
            mask_vertex_counts={"0": 0, "1": 0, "2": 125},
            full_mesh_vertices=len(vertices), full_mesh_faces=len(faces)))
    for i, row in enumerate(rows):
        for other in rows[i+1:]:
            require(not (row["object_id"] != other["object_id"]
                         and np.array_equal(row["touch"], other["touch"])),
                    "Different objects have identical observed inputs; do not silently fit contradictory labels")
    binding = dict(collection_root=str(root), prepared_label_root=str(prepared),
        actual_qualification_sha256=sha256(root / "RESULT.json"),
        actual_collection_protocol_sha256=sha256(root / "PROTOCOL.json"),
        sequence_inputs_sha256=sha256(root / "sequence_inputs.npz"),
        actual_qualified_attempts=40, fixture_center_m=list(FIXTURE_CENTER_M),
        meters_per_canonical_unit=METERS_PER_CANONICAL_UNIT)
    return rows, binding


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
    config = json.loads((args.checkpoint / "config.json").read_text())
    require(config["finger"] and config["use_touch"] and not config["use_img"]
            and config["num_grasps"] == 5 and config["number_points"] == SAMPLES
            and config["loss_coeff"] == LOSS_SCALE and config["lr"] == LEARNING_RATE,
            "Released complete t_p configuration required")
    official_args = SimpleNamespace(**config)
    official_args.eval = False
    protocol = dict(
        scope="SUGAR observed-geometry overfit only; public fixed fixture, not pressure-only shape understanding",
        stop_boundary="No automatic generalization, new physics, policy training or parameter sweep",
        objects=list(OBJECT_IDS), cases=[public_row(row) for row in rows], **selection,
        model="Unmodified official released t_p Deformation, all three deformation passes",
        parameters=PARAMETERS, graph_vertices=VERTICES, graph_faces=FACES,
        fixed_updates=STEPS, batch_size=8, report_every=REPORT_EVERY,
        optimizer=dict(name="Adam", learning_rate=LEARNING_RATE, weight_decay=0),
        training_loss="Unmodified official utils.chamfer_distance over ALL global+touch faces",
        samples=SAMPLES, repeats=REPEATS, loss_scale=LOSS_SCALE,
        labels="Original published canonical 30000-point labels in independently prepared assets",
        canonical="PUBLIC fixture: (world-[0,0,.75])/.93; actual measured hand pose; no GT pose or fitted alignment",
        input="Forty qualified observed charts, eight fixed five-touch histories; xyz+mask only, no object IDs",
        no_fallback="No native charts, native trained endpoint, dropped probe, replacement direction or mask repair",
        empty_control="All-zero charts, evaluation only; not supervised to reproduce distinct objects",
        global_metrics="2304 global faces only; original Chamfer plus independently sampled bidirectional F-score",
        fscore_sampling="Original batch_sample, 30000 points, one fixed-seed sample; squared PyTorch3D nearest distances compared with 0.01^2",
        gate=dict(every_observed_input=True, maximum_global_cd_x9000=GLOBAL_CD_LIMIT,
                  minimum_global_fscore=FSCORE_MINIMUM, fscore_distance_canonical=FSCORE_DISTANCE,
                  official_loss_must_not_exceed_initial=True,
                  official_loss_absolute_tolerance=OFFICIAL_NONREGRESSION_ATOL,
                  checkpoint_reload_required=True, actual_rendering_required_separately=True),
        seed=args.seed, metric_seed=args.seed+100000, input_root=str(args.inputs.resolve()), initialization="Independent original released checkpoint; NOT native overfit endpoint",
        checkpoint=str(args.checkpoint.resolve()),
        checkpoint_sha256=sha256(args.checkpoint / "model"),
        checkpoint_config_sha256=sha256(args.checkpoint / "config.json"),
        official_model_sha256=sha256(model_module.__file__),
        official_utils_sha256=sha256(utils.__file__),
        torch=torch.__version__, torch_cuda=torch.version.cuda, pytorch3d=pytorch3d.__version__,
        renderer_executed=False, physics_controls=0, retained_resource=resource,
    )
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
    touch = torch.from_numpy(np.stack([row["touch"] for row in rows])).cuda()
    target = torch.from_numpy(np.stack([row["target"] for row in rows])).cuda()
    image = touch.new_zeros((8, 1))  # Original touch-only image argument is unused.
    require(touch.shape == (8, 5, 25, 4) and target.shape == (8, SAMPLES, 3),
            "Fixed qualified SUGAR batch dimensions changed")
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
                        sugar_numeric_gate_passed=(all_observed_cases_pass(records)
                                                   if initial is not None else None),
                        **counts)
        write_json(directory / "METRICS.json", snapshot)
        torch.save(model.state_dict(), directory / "model")
        torch.save(optimizer.state_dict(), directory / "optim")
        print("ACTIVE3D_SUGAR_OVERFIT_EVAL", json.dumps(snapshot, allow_nan=False), flush=True)
        return snapshot, saved["observed"][0]

    initial, _ = evaluate(0)
    initial_rows = [r for r in initial["records"] if r["condition"] == "observed"]
    gradient_groups = ("positional_encoder", "mask_encoder", "mesh_deform_1", "mesh_deform_2")
    cumulative_group_nonzero = {name: False for name in gradient_groups}
    started = time.monotonic()
    snapshots = [dict(step=0, metrics="step_0000/METRICS.json", passed=None)]
    for step in range(1, STEPS+1):
        utils.set_seeds(args.seed+step)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        vertices, _ = forward(observed_charts)
        loss = LOSS_SCALE * utils.chamfer_distance(
            vertices, graph["faces"], target, num=SAMPLES, repeat=REPEATS).mean()
        require(bool(torch.isfinite(loss)) and loss.requires_grad, "Invalid original training loss")
        loss.backward()
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
        with (output / "train.jsonl").open("a") as stream:
            stream.write(json.dumps(record, allow_nan=False)+"\n")
        if step % 10 == 0:
            print("ACTIVE3D_SUGAR_OVERFIT_STEP", json.dumps(record), flush=True)
        if step % REPORT_EVERY == 0:
            final, final_vertices = evaluate(step, initial_rows)
            snapshots.append(dict(step=step, metrics=f"step_{step:04d}/METRICS.json",
                                  passed=final["sugar_numeric_gate_passed"]))
    # Reload the actual final saved checkpoint into a newly constructed complete
    # model. Do not compare against an in-memory copied model and call it reload.
    endpoint = output / f"step_{STEPS:04d}/model"
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
    changed_groups = {group: any(not torch.equal(value.cpu(), released_weights[name])
                                for name, value in model.state_dict().items()
                                if name.split(".")[0] == group)
                      for group in gradient_groups}
    require(all(cumulative_group_nonzero.values()) and all(changed_groups.values()),
            "Every official model group must receive gradients and update")
    require(counts["optimizer_updates"] == STEPS and counts["backwards"] == STEPS,
            "Fixed budget changed")
    result = dict(
        complete=True, scope="sugar_observed_geometry_only", sugar_numeric_gate_passed=final["sugar_numeric_gate_passed"],
        actual_acquisition_qualification_passed=True, native_gate_status="SEPARATE_NOT_INFERRED_FROM_THIS_RUN",
        render_gate_status="PENDING_ACTUAL_MESH_RENDER_AND_INSPECTION", overall_goal_complete=False,
        fixed_input_cases=8, fixed_objects=list(OBJECT_IDS), snapshots=snapshots, **counts,
        checkpoint=str(endpoint), checkpoint_sha256=sha256(endpoint),
        checkpoint_reload=dict(full_state_exact=state_exact, output_max_abs=reload_max, passed=True),
        every_model_group_received_nonzero_gradient=cumulative_group_nonzero,
        every_model_group_changed=changed_groups, final_cases=final["records"],
        all_actual_inputs_preserved=True, fixed_actual_probes=40, failed_cases=[r["name"] for r in final["records"]
            if r["condition"] == "observed" and not r["passed"]],
        source_files_unchanged=(sha256(model_module.__file__) == protocol["official_model_sha256"]
                               and sha256(utils.__file__) == protocol["official_utils_sha256"]),
        limitations=["Overfit only; no held-out generalization or policy result.",
                     "Observed idealized contact geometry; no learned pressure-to-chart interpretation.",
                     "Forty qualified real probes in eight fixed groups; no filtering or native fallback.",
                     "Original training loss includes immutable touch faces; it need not reach zero.",
                     "Saved full meshes are render inputs, not completed/inspected render images.",
                     "Known public frame/scale; no unknown pose, size, mass, material, policy or generalization result."],
    )
    require(result["source_files_unchanged"], "Official source changed during training")
    write_json(output / "RESULT.json", result)
    lines = ["# 完整官方 Active3D：实际 SUGAR 接触几何 overfit", "",
             "四个 TRAIN ABC 物体、四十次资格通过的实际探测、八组固定五触摸输入；完整 released t_p 独立初始化、1000 次更新。",
             "这仅是固定夹具下观测几何到形状的拟合，不是压力数字理解、未知位姿、搬运、材质或泛化。", "",
             f'SUGAR 数值 gate：{result["sugar_numeric_gate_passed"]}。实际完整网格渲染／人工检查仍待执行。', "",
             "| 输入 | 初始→最终官方 loss | 初始→最终全局 CD×9000 | 初始→最终 F@0.01 | 通过 |",
             "|---|---:|---:|---:|---|"]
    for before, after in zip(initial_rows, final["records"][:8], strict=True):
        a, b = before["metrics"], after["metrics"]
        lines.append(f'| {after["name"]} | {a["official_cd_x9000"]:.5f}→{b["official_cd_x9000"]:.5f} | '
                     f'{a["global_cd_x9000"]:.5f}→{b["global_cd_x9000"]:.5f} | '
                     f'{a["global_fscore"]:.3%}→{b["global_fscore"]:.3%} | {after["passed"]} |')
    lines += ["", "全部四十次资格通过的实际探测和八组输入完整保留，零触摸仅作对照；不按结果换物体或动作。",
              "全局指标仅评价形变后的 1824 顶点／2304 面；训练始终使用官方全部 2464 面损失。",
              "step_0000 与 step_1000 内每例 NPZ 包含完整真值、实际观测 charts、实际预测网格及零触摸输出，可直接用于后续真实渲染。",
              "固定预算结束；不自动追加训练、泛化、物理采集或调参。"]
    (output / "REPORT.md").write_text("\n".join(lines)+"\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True, help="Complete qualified actual fixture sequence root")
    parser.add_argument("--prepared", type=Path, default=EXPERIMENT / "datasets/active3d_official/prepared48")
    parser.add_argument("--vendor", type=Path, default=EXPERIMENT / "vendor/Active-3D-Vision-and-Touch")
    parser.add_argument("--checkpoint", type=Path, default=EXPERIMENT / "checkpoints/active3d_official/t_p")
    parser.add_argument("--output", type=Path, help="Required for training; must not exist")
    parser.add_argument("--seed", type=int, default=20260916)
    parser.add_argument("--check", action="store_true", help="Saved-only actual qualification/input/label check; no model or CUDA import")
    args = parser.parse_args()
    require(args.checkpoint.resolve() == (EXPERIMENT / "checkpoints/active3d_official/t_p").resolve(),
            "SUGAR must independently initialize from the original released checkpoint")
    rows, selection = load_sugar_cases(args.inputs, args.prepared)
    if args.check:
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
                   scope="sugar_actual_observations", sugar_gate_status="FAILED", overall_goal_complete=False,
                   note="Existing snapshots/logs preserve work; no native fallback, case substitution or automatic retry"))
        raise
    raise SystemExit(0 if result["sugar_numeric_gate_passed"] else 2)


if __name__ == "__main__":
    main()
