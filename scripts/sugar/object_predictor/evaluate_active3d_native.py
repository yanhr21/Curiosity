"""Fixed 48-object native-data reference for the complete released Active3D t_p.

Adapter only: original loader helpers, model, graph and Chamfer. Five fixed
grasp sequences per object, zero/five attempts paired, no model updates.
This is a subset diagnostic, not a reproduction of the full paper benchmark.
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

from .qualify_active3d_training_backend import (
    EXPERIMENT, official_imports, require, write_json, PARAMETERS,
    VERTICES, GLOBAL_VERTICES, FACES, SAMPLES, REPEATS, LOSS_SCALE,
)


def fixed_cases(dataset):
    manifest = json.loads((dataset / "selected48/SELECTION_MANIFEST.json").read_text())
    rows = []
    for split, count in (("recon_train", 32), ("valid", 8), ("test", 8)):
        ids = manifest["selected"][split]
        require(len(ids) == count and len(set(ids)) == count, "Fixed split changed")
        for index, object_id in enumerate(ids):
            for repeat in range(5):
                rows.append(dict(split=split, object_id=object_id, repeat=repeat,
                                 grasp_seed=5 * index + repeat))
    require(len({r["object_id"] for r in rows}) == 48, "Object split overlap")
    return rows


def evaluate(args):
    import numpy as np
    import torch

    require(os.environ.get("SLURM_JOB_ID") == "297947" and os.environ.get("SLURM_STEP_ID") == "0",
            "Use the retained explicit compute step")
    require(torch.cuda.is_available(), "Allocated CUDA required")
    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    utils, model_module = official_imports(args.vendor)
    loaders = importlib.import_module("pterotactyl.utility.data_loaders")
    require(Path(loaders.__file__).resolve() == args.vendor.resolve() / "pterotactyl/utility/data_loaders.py",
            "Official loader required")
    config = json.loads((args.checkpoint / "config.json").read_text())
    require(config["finger"] and config["use_touch"] and not config["use_img"]
            and config["num_grasps"] == 5 and config["number_points"] == SAMPLES
            and config["loss_coeff"] == LOSS_SCALE, "Released t_p configuration required")
    official_args = SimpleNamespace(**config)
    official_args.eval = True
    # Redirect only data roots; all reading, selection and padding code is official.
    native = args.dataset / "selected48/object_data"
    loaders.TOUCH_LOCATION = str(native / "touch_charts") + "/"
    loaders.POINT_CLOUD_LOCATION = str(native / "point_cloud_info") + "/"
    loader = object.__new__(loaders.mesh_loader_vision)
    loader.args = official_args
    rows = fixed_cases(args.dataset)
    for row in rows:
        loader.object_names = [(row["object_id"], row["grasp_seed"])]
        official_args.val_grasps = 5
        obj, grasps = loader.get_validation_instance(0)
        require(obj == row["object_id"] and len(grasps) == len(set(grasps)) == 5, "Invalid official action selection")
        row["grasps"] = grasps
    protocol = dict(
        scope="Fixed 48 native ABC objects; frozen released-model diagnostic, no Sugar inputs or policy claim",
        objects=48, paired_sequences=240, model_forwards=480, model_updates=0,
        selection="Original fixed manifest 32 TRAIN/8 VAL/8 TEST; five sequences each",
        grasp_seeds="5 * split-local manifest index + repeat 0..4; replaces original glob-order seed enumeration",
        conditions=[0, 5], five_means="Five grasp attempts, not necessarily five valid contacts",
        input="Original published chart coordinates and masks; original loader finger index 1 and five slots",
        target="All original published 30000 canonical points; no alignment or rescaling",
        official_loss="All 2464 global+touch faces; original squared-L2 Chamfer 30000 samples x3 x9000",
        auxiliary_loss="Same original Chamfer restricted to 2304 global faces; explicitly auxiliary",
        render_selection="repeat0 for every one of the 48 objects, regardless of scores",
        metric_seed=args.seed, checkpoint=str(args.checkpoint), dataset=str(args.dataset),
        cases=rows,
    )
    write_json(output / "PROTOCOL.json", protocol)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    utils.set_seeds(args.seed)
    graph, initial_mesh = utils.load_mesh_vision(official_args, str(args.vendor / "pterotactyl/objects/vision_charts.obj"))
    require(initial_mesh.shape == (GLOBAL_VERTICES, 3) and graph["adj"].shape == (VERTICES, VERTICES)
            and graph["faces"].shape == (FACES, 3), "Complete original graph required")
    global_faces = graph["faces"][(graph["faces"] < GLOBAL_VERTICES).all(dim=1)]
    require(len(global_faces) == 2304, "Unexpected original global topology")
    model = model_module.Deformation(graph, initial_mesh, official_args).cuda()
    weights = torch.load(args.checkpoint / "model", map_location="cpu", weights_only=True)
    model.load_state_dict(weights, strict=True)
    require(sum(p.numel() for p in model.parameters()) == PARAMETERS, "Incomplete official model")
    model.eval().requires_grad_(False)
    results = []
    for case_index, row in enumerate(rows):
        loader.object_names = [(row["object_id"], row["grasp_seed"])]
        utils.set_seeds(args.seed + case_index)
        target = loader.get_points(row["object_id"]).unsqueeze(0).cuda()
        require(target.shape == (1, SAMPLES, 3) and bool(torch.isfinite(target).all()), "Invalid published point label")
        saved = dict(faces=graph["faces"].cpu().numpy(), global_faces=global_faces.cpu().numpy())
        metrics = {}
        for attempts in (0, 5):
            official_args.val_grasps = attempts
            obj, grasps = loader.get_validation_instance(0)
            require(obj == row["object_id"] and grasps == row["grasps"][:attempts], "Action pairing changed")
            touch = loader.get_touch_info(obj, grasps).unsqueeze(0).cuda()
            require(touch.shape == (1, 5, 25, 4) and bool(torch.isfinite(touch).all())
                    and bool(torch.isin(touch[..., 3], touch.new_tensor([0, 1, 2])).all()), "Invalid native charts")
            image = touch.new_zeros((1, 1))  # Touch-only model ignores image contents.
            with torch.no_grad():
                charts = model_module.prepare_mesh(dict(img=image, touch_charts=touch), initial_mesh, official_args)
                vertices, masks = model(image, charts)
                require(vertices.shape == (1, VERTICES, 3) and bool(torch.isfinite(vertices).all()), "Invalid full output")
                require(torch.equal(vertices[:, GLOBAL_VERTICES:], touch.reshape(1, 125, 4)[..., :3]), "Fixed chart vertices changed")
                scores = {}
                for metric_index, (name, faces) in enumerate((("official_all_faces", graph["faces"]), ("auxiliary_global_only", global_faces))):
                    # Match random area samples between arms without outcome selection.
                    utils.set_seeds(args.seed + case_index * 2 + metric_index)
                    cd = utils.chamfer_distance(vertices, faces, target, num=SAMPLES, repeat=REPEATS)
                    require(bool(torch.isfinite(cd).all()), "Nonfinite official Chamfer")
                    scores[name] = float(cd.mean()) * LOSS_SCALE
            key = f"attempts{attempts}"
            metrics[key] = dict(**scores, mask_vertex_counts={str(m): int((touch[..., 3] == m).sum()) for m in (0, 1, 2)})
            saved[key + "_vertices"] = vertices[0].cpu().numpy()
            saved[key + "_touch"] = touch[0].cpu().numpy()
            saved[key + "_mask"] = masks[0].cpu().numpy()
        name = f'{row["split"]}_{row["object_id"]}_repeat{row["repeat"]}.npz'
        np.savez_compressed(output / name, **saved)
        record = dict(**row, metrics=metrics, output_file=name)
        results.append(record)
        with (output / "cases.jsonl").open("a") as stream:
            stream.write(json.dumps(record, allow_nan=False) + "\n")
        print("NATIVE_REFERENCE_CASE", case_index + 1, json.dumps(record), flush=True)
    unchanged = all(torch.equal(value.cpu(), weights[name]) for name, value in model.state_dict().items())
    require(unchanged, "Frozen checkpoint changed")
    summary = {}
    for split in ("recon_train", "valid", "test"):
        selected = [r for r in results if r["split"] == split]
        per_object = []
        for obj in dict.fromkeys(r["object_id"] for r in selected):
            cases = [r for r in selected if r["object_id"] == obj]
            require(len(cases) == 5, "Missing sequence; denominator cannot shrink")
            means = {arm: {metric: float(np.mean([r["metrics"][arm][metric] for r in cases]))
                           for metric in ("official_all_faces", "auxiliary_global_only")}
                     for arm in ("attempts0", "attempts5")}
            per_object.append(dict(object_id=obj, means=means))
        summary[split] = dict(objects=len(per_object), per_object=per_object,
            equal_object_mean={arm: {metric: float(np.mean([r["means"][arm][metric] for r in per_object]))
                for metric in ("official_all_faces", "auxiliary_global_only")} for arm in ("attempts0", "attempts5")},
            objects_improved={metric: sum(r["means"]["attempts5"][metric] < r["means"]["attempts0"][metric] for r in per_object)
                              for metric in ("official_all_faces", "auxiliary_global_only")})
    write_json(output / "RESULT.json", dict(complete=True, actual_model_forwards=480,
               model_updates=0, frozen_state_unchanged=unchanged, summary=summary, cases=results))
    lines = ["# 官方原生输入：固定48物体重建对照", "",
             "完整冻结 t_p，零次／五次抓取尝试配对，每物体五组动作。原生已知canonical坐标，非SUGAR触觉或搬运估计；不是全量论文评测复现。", "",
             "主指标是官方全部global+touch面Chamfer×9000；辅助只取global面。两者均为30000点、三次面积采样、双向平方L2。按对象内五次均值，再对对象等权。", "",
             "| 划分 | 物体 | 官方全表面 0→5 | 全局表面辅助 0→5 | 改善物体 全部/全局 |", "|---|---:|---|---|---|"]
    for split, value in summary.items():
        m = value["equal_object_mean"]
        pair = lambda k: f'{m["attempts0"][k]:.4f}→{m["attempts5"][k]:.4f}'
        lines.append(f'| {split} | {value["objects"]} | {pair("official_all_faces")} | {pair("auxiliary_global_only")} | {value["objects_improved"]} |')
    lines += ["", "TRAIN表现不能作为泛化证据；五次抓取不保证有效接触。全部mask和动作列表保留，未按接触成功更换。官方完整loss可能受固定触摸chart本身贡献影响，应同时检查辅助指标与真实形状渲染。", "", "实际48张网格渲染尚待独立render阶段。"]
    (output / "REPORT.md").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=EXPERIMENT / "datasets/active3d_official")
    parser.add_argument("--vendor", type=Path, default=EXPERIMENT / "vendor/Active-3D-Vision-and-Touch")
    parser.add_argument("--checkpoint", type=Path, default=EXPERIMENT / "checkpoints/active3d_official/t_p")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260916)
    args = parser.parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
