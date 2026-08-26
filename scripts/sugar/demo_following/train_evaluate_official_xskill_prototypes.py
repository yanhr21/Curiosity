#!/usr/bin/env python3
"""Train and gate the released XSkill prototype sequence on clean SUGAR RGB.

Without ``--sphere-corpus`` this reproduces the completed same-embodiment
compatibility audit.  With ``--sphere-corpus`` it follows the released XSkill
simulation contract: the two official dataset streams are the same source
motions rendered as G1 and as the task-independent sphere embodiment.  In both
modes the released model, architecture, augmentations, SwAV/Sinkhorn objective
and time-contrastive training step remain unchanged.  Local code supplies only
the SUGAR data streams, Lightning-free runtime glue, frozen metrics and a
machine-readable representation admission decision.  This is never policy
training.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import types
from collections import namedtuple
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import kendalltau
import torch
from torch import nn
from torch.utils.data import DataLoader


TASKS = ("CarryBox", "KickBox")
REFERENCE_IDS = {"CarryBox": 45, "KickBox": 21}
SEED = 45
FRAMES = 64
SLIDE = 8
FINAL_EPOCH = 79
BATCH_SIZE = 28
XSKILL_COMMIT = "b748071daeb031d6b42a8dcb88c38c52297e20af"
SPHERE_BODY_NAMES = (
    "left_rubber_hand",
    "right_rubber_hand",
    "left_ankle_roll_link",
    "right_ankle_roll_link",
)


def install_lightning_compatibility_stub() -> None:
    """Let the released model class run without changing its neural code.

    XSkill uses Lightning only for orchestration.  Its released Model remains
    the source of the exact forward, Sinkhorn and training-step losses used
    below; this stub supplies the nn.Module base expected by that source.
    """

    module = types.ModuleType("pytorch_lightning")
    module.LightningModule = nn.Module
    sys.modules["pytorch_lightning"] = module


def import_official_xskill(official_repo: Path):
    sys.path.insert(0, str(official_repo))
    install_lightning_compatibility_stub()
    from xskill.dataset.dataset import ConcatDataset, EpisodeTrajDataset
    from xskill.dataset.frame_samplers import UniformDownSampleSampler
    from xskill.model import core as official_core
    from xskill.model.core import Model
    from xskill.model.encoder import CNN, VisualMotionEncoder, VisualMotionPrior
    from xskill.model.transformer import PositionalEncoding, TorchTransformerEncoder
    from xskill.utility.transform import get_transform_pipeline

    return SimpleNamespace(
        ConcatDataset=ConcatDataset,
        EpisodeTrajDataset=EpisodeTrajDataset,
        UniformDownSampleSampler=UniformDownSampleSampler,
        official_core=official_core,
        Model=Model,
        CNN=CNN,
        VisualMotionEncoder=VisualMotionEncoder,
        VisualMotionPrior=VisualMotionPrior,
        PositionalEncoding=PositionalEncoding,
        TorchTransformerEncoder=TorchTransformerEncoder,
        get_transform_pipeline=get_transform_pipeline,
    )


def verify_corpus(corpus: Path) -> dict[str, dict[str, list[int]]]:
    expected = {"train": (80, 80), "valid": (10, 10), "test": (10, 9)}
    inventory: dict[str, dict[str, list[int]]] = {}
    for split, counts in expected.items():
        inventory[split] = {}
        for task, count in zip(TASKS, counts, strict=True):
            task_dir = corpus / split / task
            directories = sorted(
                (path for path in task_dir.iterdir() if path.is_dir()),
                key=lambda path: int(path.name),
            )
            if len(directories) != count:
                raise RuntimeError(f"{task_dir}: expected {count} motions, found {len(directories)}")
            for directory in directories:
                frames = sorted(directory.glob("*.png"), key=lambda path: int(path.stem))
                if len(frames) != FRAMES or [int(path.stem) for path in frames] != list(range(FRAMES)):
                    raise RuntimeError(f"{directory}: expected exact RGB frames 0..63")
            inventory[split][task] = [int(path.name) for path in directories]
    for task in TASKS:
        split_sets = [set(inventory[split][task]) for split in ("train", "valid", "test")]
        if any(split_sets[i] & split_sets[j] for i in range(3) for j in range(i + 1, 3)):
            raise RuntimeError(f"{task}: source motion IDs overlap across splits")
        if REFERENCE_IDS[task] not in split_sets[0]:
            raise RuntimeError(f"{task}: reference {REFERENCE_IDS[task]} is not train-only")
    return inventory


def verify_sphere_render_manifests(corpus: Path) -> list[dict[str, object]]:
    manifests = []
    for task, final_id in (("CarryBox", 99), ("KickBox", 98)):
        path = corpus / f"RENDER_RESULT_SPHERE_{task}_000_{final_id:03d}.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        contract = payload.get("sphere_agent_contract") or {}
        if not payload.get("passed") or payload.get("embodiment") != "sphere":
            raise RuntimeError(f"invalid sphere render result: {path}")
        if tuple(contract.get("body_names", ())) != SPHERE_BODY_NAMES:
            raise RuntimeError(f"task-dependent or incomplete sphere body set: {path}")
        if float(contract.get("radius_m", -1.0)) != 0.05:
            raise RuntimeError(f"sphere radius does not match released XSkill: {path}")
        if contract.get("original_g1_visibility") is not False:
            raise RuntimeError(f"original G1 remains visible: {path}")
        manifests.append(payload)
    return manifests


def verify_cross_corpus_visual_difference(
    g1_corpus: Path,
    sphere_corpus: Path,
    inventory: dict[str, dict[str, list[int]]],
) -> dict[str, float | int]:
    differences = []
    for split in ("train", "valid", "test"):
        for task in TASKS:
            for motion_id in inventory[split][task]:
                g1 = cv2.imread(
                    str(g1_corpus / split / task / str(motion_id) / "32.png"),
                    cv2.IMREAD_COLOR,
                )
                sphere = cv2.imread(
                    str(sphere_corpus / split / task / str(motion_id) / "32.png"),
                    cv2.IMREAD_COLOR,
                )
                if g1 is None or sphere is None or g1.shape != sphere.shape:
                    raise RuntimeError(f"invalid paired center frames for {split}/{task}/{motion_id}")
                differences.append(float(np.mean(np.abs(g1.astype(np.float32) - sphere.astype(np.float32)))))
    if not differences or min(differences) <= 0.0:
        raise RuntimeError("sphere corpus contains a pixel-identical embodiment pair")
    return {
        "paired_center_frames": len(differences),
        "minimum_mean_absolute_pixel_difference": float(min(differences)),
        "mean_mean_absolute_pixel_difference": float(np.mean(differences)),
        "maximum_mean_absolute_pixel_difference": float(max(differences)),
    }


def write_stream_masks(output: Path, inventory: dict[str, dict[str, list[int]]]) -> tuple[Path, Path]:
    carry = inventory["train"]["CarryBox"]
    kick = inventory["train"]["KickBox"]
    if carry != kick or len(carry) != 80:
        raise RuntimeError("XSkill two-stream adapter requires matching 80-motion train IDs")
    masks = ([index % 2 == 0 for index in range(80)], [index % 2 == 1 for index in range(80)])
    paths = (output / "source_stream_a_mask.json", output / "source_stream_b_mask.json")
    for path, mask in zip(paths, masks, strict=True):
        path.write_text(json.dumps(mask) + "\n", encoding="utf-8")
    return paths


def make_model(official) -> nn.Module:
    temporal = official.TorchTransformerEncoder(
        query_dim=256,
        heads=4,
        dim_feedforward=512,
        n_layer=8,
        rep_dim=256,
        use_encoder=False,
        input_dim=None,
        pos_encoder=official.PositionalEncoding(size=256, max_len=10, frequency=10),
    )
    encoder = official.VisualMotionEncoder(
        vision_encoder=official.CNN(out_size=256),
        nmb_prototypes=128,
        state_size=256,
        out_size=256,
        vision_only=True,
        normalize=True,
        start_end=True,
        goal_condition=False,
        temporal_transformer_encoder=temporal,
    )
    prior = official.VisualMotionPrior(
        vision_encoder=official.CNN(out_size=128),
        out_size=128,
        vision_only=True,
        nmb_prototypes=128,
        normalize=False,
    )
    return official.Model(
        encoder_q=encoder,
        epsilon=0.03,
        sinkhorn_iterations=3,
        dim=128,
        T=0.1,
        lr=1e-4,
        stack_frames=1,
        slide=SLIDE,
        skill_prior=prior,
        skill_prior_encoder=None,
        freeze_prototypes_epoch=0,
        n_negative_samples=16,
        clutser_T=0.1,
        reverse_augment=False,
        time_augment=True,
        swav_loss_coef=0.5,
        steps_per_epoch=None,
        use_lr_scheduler=False,
        use_temperature_scheduler=False,
        cluster_loss_coef=1,
        positive_window=4,
        negative_window=12,
        pretrain_pipeline=official.get_transform_pipeline(
            ["random_crop_112_112", "color_jitter", "grayscale", "gaussian_blur", "normalize"]
        ),
    )


def make_training_loader(
    official,
    corpus: Path,
    masks: tuple[Path, Path] | None,
    sphere_corpus: Path | None = None,
) -> DataLoader:
    sampler_args = dict(downsample_ratio=1, offset=0, num_frames=FRAMES)
    datasets = []
    stream_specs = (
        [(corpus, masks[0]), (corpus, masks[1])]
        if sphere_corpus is None and masks is not None
        else [(corpus, None), (sphere_corpus, None)]
    )
    for stream_corpus, mask in stream_specs:
        if stream_corpus is None:
            raise RuntimeError("cross-embodiment XSkill requires both corpora")
        dataset = official.EpisodeTrajDataset(
            frame_sampler=official.UniformDownSampleSampler(**sampler_args),
            _allowed_dirs=[str(stream_corpus / "train" / task) for task in TASKS],
            slide=SLIDE,
            seed=SEED,
            sort_numerical=True,
            vid_mask=str(mask) if mask is not None else None,
            max_get_threads=8,
            resize_shape=[124, 124],
        )
        expected_length = 80 if sphere_corpus is None else 160
        if len(dataset) != expected_length:
            raise RuntimeError(
                f"source stream must contain {expected_length} motions, found {len(dataset)}"
            )
        datasets.append(dataset)
    pairing_audit = None
    if sphere_corpus is not None:
        paired_keys = []
        for index in range(len(datasets[0])):
            stream_keys = []
            for dataset in datasets:
                class_index, video_index = dataset._indexfile[index]
                directory = Path(dataset._get_video_path(class_index, video_index))
                stream_keys.append((directory.parent.name, int(directory.name)))
            if stream_keys[0] != stream_keys[1]:
                raise RuntimeError(
                    f"official XSkill stream index {index} is not source-paired: {stream_keys}"
                )
            paired_keys.append(stream_keys[0])
        if len(set(paired_keys)) != 160:
            raise RuntimeError("official XSkill paired training stream is not one-to-one")
        pairing_audit = {
            "paired_indices": len(paired_keys),
            "unique_source_motion_pairs": len(set(paired_keys)),
            "elementwise_task_and_motion_id_match": True,
        }
    generator = torch.Generator().manual_seed(SEED)
    loader = DataLoader(
        official.ConcatDataset(*datasets),
        batch_size=BATCH_SIZE,
        num_workers=0,
        shuffle=True,
        pin_memory=True,
        persistent_workers=False,
        drop_last=True,
        generator=generator,
    )
    loader.xskill_generator = generator
    loader.xskill_pairing_audit = pairing_audit
    return loader


def move_batch(batch, device: torch.device):
    moved = []
    for stream in batch:
        moved.append(type(stream)(stream.im_q.to(device, non_blocking=True), stream.index, stream.info))
    return tuple(moved)


def scalar_log(payload: dict[str, object]) -> dict[str, float]:
    result = {}
    for key, value in payload.items():
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().item()
        if isinstance(value, (int, float, np.floating)):
            result[key] = float(value)
    return result


def save_resume(
    path: Path,
    model: nn.Module,
    optimizers: tuple[torch.optim.Optimizer, torch.optim.Optimizer],
    loader: DataLoader,
    epoch: int,
) -> None:
    payload = {
        "epoch": epoch,
        "state_dict": model.state_dict(),
        "optimizers": [optimizer.state_dict() for optimizer in optimizers],
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all(),
        "numpy_rng": np.random.get_state(),
        "python_rng": random.getstate(),
        "loader_rng": loader.xskill_generator.get_state(),
    }
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def restore_resume(
    path: Path,
    model: nn.Module,
    optimizers: tuple[torch.optim.Optimizer, torch.optim.Optimizer],
    loader: DataLoader,
) -> int:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["state_dict"], strict=True)
    for optimizer, state in zip(optimizers, payload["optimizers"], strict=True):
        optimizer.load_state_dict(state)
    torch.set_rng_state(payload["torch_rng"])
    torch.cuda.set_rng_state_all(payload["cuda_rng"])
    np.random.set_state(payload["numpy_rng"])
    random.setstate(payload["python_rng"])
    loader.xskill_generator.set_state(payload["loader_rng"])
    return int(payload["epoch"]) + 1


def train_official_model(
    official,
    corpus: Path,
    output: Path,
    device: torch.device,
    masks: tuple[Path, Path] | None,
    sphere_corpus: Path | None = None,
) -> tuple[Path, Path, dict[str, object] | None]:
    initial_path = output / "model_pretrain_init.pt"
    final_path = output / "model_epoch79.pt"
    resume_path = output / "model_training_latest.pt"
    log_path = output / "TRAINING_LOG.jsonl"

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    model = make_model(official)
    if not initial_path.exists():
        torch.save({"state_dict": model.state_dict(), "seed": SEED}, initial_path)
    model.to(device).train()
    loader = make_training_loader(official, corpus, masks, sphere_corpus)
    optimizer_pair = model.configure_optimizers()
    if not isinstance(optimizer_pair, tuple) or len(optimizer_pair) != 2:
        raise RuntimeError("released XSkill must expose encoder and skill-prior optimizers")
    model.optimizers = lambda: optimizer_pair
    model.manual_backward = lambda loss: loss.backward()
    model.trainer = SimpleNamespace(current_epoch=0, max_epochs=FINAL_EPOCH + 2)

    captured: list[dict[str, float]] = []
    official.official_core.wandb.log = lambda payload: captured.append(scalar_log(payload))
    start_epoch = restore_resume(resume_path, model, optimizer_pair, loader) if resume_path.exists() else 0
    if final_path.exists():
        return initial_path, final_path, loader.xskill_pairing_audit

    with log_path.open("a", encoding="utf-8") as log_file:
        for epoch in range(start_epoch, FINAL_EPOCH + 1):
            model.trainer.current_epoch = epoch
            captured.clear()
            for batch_index, batch in enumerate(loader):
                model.training_step(move_batch(batch, device), batch_index)
            if not captured:
                raise RuntimeError("released XSkill training step emitted no loss record")
            keys = ("encoder_loss", "repre_loss", "cluster_loss", "skill_prior_loss")
            summary = {
                key: float(np.mean([row[key] for row in captured if key in row])) for key in keys
            }
            if not all(np.isfinite(value) for value in summary.values()):
                raise RuntimeError(f"non-finite XSkill training loss at epoch {epoch}: {summary}")
            row = {"epoch": epoch, "steps": len(loader), **summary}
            log_file.write(json.dumps(row) + "\n")
            log_file.flush()
            print(f"XSKILL_EPOCH {epoch:02d} {json.dumps(summary, sort_keys=True)}", flush=True)
            if epoch % 5 == 4 or epoch == FINAL_EPOCH:
                save_resume(resume_path, model, optimizer_pair, loader, epoch)

    torch.save(
        {
            "state_dict": model.state_dict(),
            "epoch": FINAL_EPOCH,
            "official_commit": XSKILL_COMMIT,
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        },
        final_path,
    )
    return initial_path, final_path, loader.xskill_pairing_audit


def load_frames(directory: Path, reverse: bool = False) -> torch.Tensor:
    paths = sorted(directory.glob("*.png"), key=lambda path: int(path.stem))
    if reverse:
        paths = paths[::-1]
    frames = []
    for path in paths:
        frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if frame is None:
            raise RuntimeError(f"failed to decode {path}")
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(cv2.resize(frame, (124, 124), interpolation=cv2.INTER_AREA))
    array = np.stack(frames).astype(np.float32) / 255.0
    return torch.from_numpy(array).permute(0, 3, 1, 2).contiguous()


def embed_motion(model: nn.Module, pipeline: nn.Module, directory: Path, device: torch.device, reverse=False):
    frames = pipeline(load_frames(directory, reverse=reverse).to(device))
    windows = torch.stack([frames[index : index + SLIDE + 1] for index in range(FRAMES - SLIDE)])
    raw_parts, prototype_parts = [], []
    with torch.no_grad():
        for chunk in windows.split(64):
            state = model.encoder_q.get_state_representation(chunk, None)
            raw = model.encoder_q.get_traj_representation(state)
            normalized = nn.functional.normalize(raw, dim=1, p=2)
            logits = model.encoder_q.prototypes(normalized)
            raw_parts.append(raw.cpu())
            prototype_parts.append(torch.softmax(logits / model.T, dim=1).cpu())
    raw = torch.cat(raw_parts).numpy().astype(np.float64)
    prototypes = torch.cat(prototype_parts).numpy().astype(np.float64)
    if not np.isfinite(raw).all() or not np.isfinite(prototypes).all():
        raise RuntimeError(f"non-finite XSkill embedding for {directory}")
    return {"raw": raw, "prototype": prototypes}


def embed_corpus(official, model: nn.Module, corpus: Path, device: torch.device):
    pipeline = official.get_transform_pipeline(["center_crop_112_112", "normalize"]).to(device)
    result = {split: {task: {} for task in TASKS} for split in ("train", "valid", "test")}
    model.eval().to(device)
    for split in result:
        for task in TASKS:
            directories = sorted((corpus / split / task).iterdir(), key=lambda path: int(path.name))
            for directory in directories:
                result[split][task][int(directory.name)] = embed_motion(
                    model, pipeline, directory, device
                )
    reversed_references = {
        task: embed_motion(
            model,
            pipeline,
            corpus / "train" / task / str(REFERENCE_IDS[task]),
            device,
            reverse=True,
        )
        for task in TASKS
    }
    return result, reversed_references


def temporal_metrics(task_embeddings: dict[int, dict[str, np.ndarray]], field: str):
    errors, taus = [], []
    items = sorted(task_embeddings.items())
    for query_id, query_fields in items:
        for candidate_id, candidate_fields in items:
            if query_id == candidate_id:
                continue
            query, candidate = query_fields[field], candidate_fields[field]
            nearest = np.argmin(cdist(query, candidate, "sqeuclidean"), axis=1)
            expected = np.linspace(0, len(candidate) - 1, len(query))
            errors.append(float(np.mean(np.abs(nearest - expected)) / (len(candidate) - 1)))
            tau = kendalltau(np.arange(len(nearest)), nearest).correlation
            if np.isfinite(tau):
                taus.append(float(tau))
    if not errors or not taus:
        raise RuntimeError("not enough held-out motions for temporal retrieval")
    return {
        "normalized_temporal_mae": float(np.mean(errors)),
        "kendalls_tau": float(np.mean(taus)),
        "ordered_video_pairs": len(errors),
    }


def dtw_cost(first: np.ndarray, second: np.ndarray) -> float:
    distances = cdist(first, second, "sqeuclidean")
    rows, columns = distances.shape
    table = np.full((rows + 1, columns + 1), np.inf, dtype=np.float64)
    table[0, 0] = 0.0
    for row in range(1, rows + 1):
        for column in range(1, columns + 1):
            table[row, column] = distances[row - 1, column - 1] + min(
                table[row - 1, column],
                table[row, column - 1],
                table[row - 1, column - 1],
            )
    return float(table[rows, columns] / (rows + columns))


def reference_metrics(embeddings, reversed_references, split: str, field: str):
    rows = []
    references = {
        task: embeddings["train"][task][REFERENCE_IDS[task]][field] for task in TASKS
    }
    for task in TASKS:
        unrelated = TASKS[1 - TASKS.index(task)]
        for motion_id, fields in sorted(embeddings[split][task].items()):
            query = fields[field]
            correct_cost = dtw_cost(query, references[task])
            unrelated_cost = dtw_cost(query, references[unrelated])
            reversed_cost = dtw_cost(query, reversed_references[task][field])
            rows.append(
                {
                    "task": task,
                    "motion_id": motion_id,
                    "correct_cost": correct_cost,
                    "unrelated_cost": unrelated_cost,
                    "reversed_cost": reversed_cost,
                    "correct_task_wins": bool(correct_cost < unrelated_cost),
                    "ordered_demo_wins": bool(correct_cost < reversed_cost),
                }
            )
    return {
        "task_reference_accuracy": float(np.mean([row["correct_task_wins"] for row in rows])),
        "ordered_reference_accuracy": float(np.mean([row["ordered_demo_wins"] for row in rows])),
        "queries": len(rows),
        "rows": rows,
    }


def cross_temporal_metrics(query_embeddings, candidate_embeddings, split: str, field: str):
    """Measure time retrieval between the same source motion in two embodiments."""

    result = {}
    for task in TASKS:
        query_task = query_embeddings[split][task]
        candidate_task = candidate_embeddings[split][task]
        if set(query_task) != set(candidate_task):
            raise RuntimeError(f"{split}/{task}: embodiment source IDs do not match")
        errors, taus = [], []
        for motion_id in sorted(query_task):
            query = query_task[motion_id][field]
            candidate = candidate_task[motion_id][field]
            nearest = np.argmin(cdist(query, candidate, "sqeuclidean"), axis=1)
            expected = np.linspace(0, len(candidate) - 1, len(query))
            errors.append(float(np.mean(np.abs(nearest - expected)) / (len(candidate) - 1)))
            tau = kendalltau(np.arange(len(nearest)), nearest).correlation
            # A collapsed retrieval maps every query frame to one candidate
            # frame.  scipy reports an undefined tau in that case; it is a
            # temporal failure and must score zero rather than disappear from
            # the aggregate.
            taus.append(float(tau) if np.isfinite(tau) else 0.0)
        if not errors:
            raise RuntimeError(f"{split}/{task}: no cross-embodiment temporal metric")
        result[task] = {
            "normalized_temporal_mae": float(np.mean(errors)),
            "kendalls_tau": float(np.mean(taus)),
            "source_motion_pairs": len(errors),
        }
    return result


def cross_reference_metrics(
    query_embeddings,
    reference_embeddings,
    reversed_reference_embeddings,
    split: str,
    field: str,
):
    """Retrieve ordered task references exclusively from the other embodiment."""

    rows = []
    references = {
        task: reference_embeddings["train"][task][REFERENCE_IDS[task]][field]
        for task in TASKS
    }
    for task in TASKS:
        unrelated = TASKS[1 - TASKS.index(task)]
        for motion_id, fields in sorted(query_embeddings[split][task].items()):
            query = fields[field]
            correct_cost = dtw_cost(query, references[task])
            unrelated_cost = dtw_cost(query, references[unrelated])
            reversed_cost = dtw_cost(
                query,
                reversed_reference_embeddings[task][field],
            )
            rows.append(
                {
                    "task": task,
                    "motion_id": motion_id,
                    "correct_cost": correct_cost,
                    "unrelated_cost": unrelated_cost,
                    "reversed_cost": reversed_cost,
                    "correct_task_wins": bool(correct_cost < unrelated_cost),
                    "ordered_demo_wins": bool(correct_cost < reversed_cost),
                }
            )
    return {
        "task_reference_accuracy": float(np.mean([row["correct_task_wins"] for row in rows])),
        "ordered_reference_accuracy": float(np.mean([row["ordered_demo_wins"] for row in rows])),
        "queries": len(rows),
        "rows": rows,
    }


def prototype_usage(embeddings, split: str):
    probabilities = np.concatenate(
        [
            fields["prototype"]
            for task in TASKS
            for fields in embeddings[split][task].values()
        ],
        axis=0,
    )
    assignments = np.argmax(probabilities, axis=1)
    counts = np.bincount(assignments, minlength=128)
    distribution = counts / counts.sum()
    entropy = -np.sum(distribution[distribution > 0] * np.log(distribution[distribution > 0]))
    return {
        "used_prototypes": int(np.count_nonzero(counts)),
        "assignment_entropy": float(entropy),
        "maximum_possible_entropy": float(np.log(128)),
        "counts": counts.tolist(),
    }


def evaluate_from_embeddings(embeddings, reversed_references):
    return {
        split: {
            field: {
                "temporal": {
                    task: temporal_metrics(embeddings[split][task], field) for task in TASKS
                },
                "references": reference_metrics(embeddings, reversed_references, split, field),
            }
            for field in ("raw", "prototype")
        }
        for split in ("valid", "test")
    } | {"prototype_usage": prototype_usage(embeddings, "test")}


def evaluate(official, model: nn.Module, corpus: Path, device: torch.device):
    embeddings, reversed_references = embed_corpus(official, model, corpus, device)
    return evaluate_from_embeddings(embeddings, reversed_references)


def evaluate_cross_embodiment(
    official,
    model: nn.Module,
    g1_corpus: Path,
    sphere_corpus: Path,
    device: torch.device,
):
    g1_embeddings, g1_reversed = embed_corpus(official, model, g1_corpus, device)
    sphere_embeddings, sphere_reversed = embed_corpus(
        official,
        model,
        sphere_corpus,
        device,
    )
    cross = {}
    for direction, query, reference, reversed_reference in (
        ("g1_to_sphere", g1_embeddings, sphere_embeddings, sphere_reversed),
        ("sphere_to_g1", sphere_embeddings, g1_embeddings, g1_reversed),
    ):
        cross[direction] = {
            split: {
                field: {
                    "temporal": cross_temporal_metrics(query, reference, split, field),
                    "references": cross_reference_metrics(
                        query,
                        reference,
                        reversed_reference,
                        split,
                        field,
                    ),
                }
                for field in ("raw", "prototype")
            }
            for split in ("valid", "test")
        }
    return {
        "g1_within": evaluate_from_embeddings(g1_embeddings, g1_reversed),
        "sphere_within": evaluate_from_embeddings(sphere_embeddings, sphere_reversed),
        "cross": cross,
    }


def aggregate_temporal(result, split: str, field: str, metric: str) -> float:
    return float(np.mean([result[split][field]["temporal"][task][metric] for task in TASKS]))


def aggregate_cross_temporal(result, direction: str, split: str, field: str, metric: str) -> float:
    return float(
        np.mean(
            [
                result["cross"][direction][split][field]["temporal"][task][metric]
                for task in TASKS
            ]
        )
    )


def load_model_checkpoint(official, path: Path, device: torch.device):
    model = make_model(official)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    incompatible = model.load_state_dict(payload["state_dict"], strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"non-strict XSkill checkpoint load: {incompatible}")
    return model.to(device)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--sphere-corpus", type=Path)
    parser.add_argument("--official-repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    corpus = args.corpus.expanduser().resolve()
    sphere_corpus = args.sphere_corpus.expanduser().resolve() if args.sphere_corpus else None
    official_repo = args.official_repo.expanduser().resolve()
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / "REPRESENTATION_RESULT.json"
    if result_path.exists():
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        print(f"XSKILL_REPRESENTATION_RESULT passed={payload['passed']} output={result_path}", flush=True)
        return
    if not torch.cuda.is_available():
        raise RuntimeError("official XSkill audit requires a CUDA compute step")
    device = torch.device(args.device)
    inventory = verify_corpus(corpus)
    sphere_inventory = None
    sphere_manifests = None
    visual_difference = None
    if sphere_corpus is None:
        masks = write_stream_masks(output, inventory)
    else:
        sphere_inventory = verify_corpus(sphere_corpus)
        if sphere_inventory != inventory:
            raise RuntimeError("G1 and sphere source-ID inventories are not elementwise identical")
        sphere_manifests = verify_sphere_render_manifests(sphere_corpus)
        visual_difference = verify_cross_corpus_visual_difference(
            corpus,
            sphere_corpus,
            inventory,
        )
        masks = None
        paired_manifest = {
            "protocol": "official_xskill_paired_g1_sphere_train_stream_v1",
            "pairs": [
                {"task": task, "motion_id": motion_id}
                for task in TASKS
                for motion_id in inventory["train"][task]
            ],
            "pair_count": 160,
            "g1_corpus": str(corpus),
            "sphere_corpus": str(sphere_corpus),
            "visual_difference": visual_difference,
        }
        (output / "PAIRED_TRAIN_STREAM.json").write_text(
            json.dumps(paired_manifest, indent=2) + "\n",
            encoding="utf-8",
        )
    official = import_official_xskill(official_repo)
    initial_path, trained_path, official_stream_pairing = train_official_model(
        official,
        corpus,
        output,
        device,
        masks,
        sphere_corpus,
    )

    initial_model = load_model_checkpoint(official, initial_path, device)
    initial = (
        evaluate(official, initial_model, corpus, device)
        if sphere_corpus is None
        else evaluate_cross_embodiment(
            official,
            initial_model,
            corpus,
            sphere_corpus,
            device,
        )
    )
    del initial_model
    torch.cuda.empty_cache()
    trained_model = load_model_checkpoint(official, trained_path, device)
    trained = (
        evaluate(official, trained_model, corpus, device)
        if sphere_corpus is None
        else evaluate_cross_embodiment(
            official,
            trained_model,
            corpus,
            sphere_corpus,
            device,
        )
    )

    if sphere_corpus is not None:
        criteria = {}
        summary = {}
        for direction in ("g1_to_sphere", "sphere_to_g1"):
            trained_mae = aggregate_cross_temporal(
                trained, direction, "test", "prototype", "normalized_temporal_mae"
            )
            initial_mae = aggregate_cross_temporal(
                initial, direction, "test", "prototype", "normalized_temporal_mae"
            )
            trained_tau = aggregate_cross_temporal(
                trained, direction, "test", "prototype", "kendalls_tau"
            )
            initial_tau = aggregate_cross_temporal(
                initial, direction, "test", "prototype", "kendalls_tau"
            )
            criteria[f"{direction}_test_prototype_temporal_mae_improves_5pct"] = bool(
                trained_mae <= 0.95 * initial_mae
            )
            criteria[f"{direction}_test_prototype_tau_improves_0p05"] = bool(
                trained_tau >= initial_tau + 0.05
            )
            summary[f"{direction}_trained_test_prototype_temporal_mae"] = trained_mae
            summary[f"{direction}_initial_test_prototype_temporal_mae"] = initial_mae
            summary[f"{direction}_trained_test_prototype_kendalls_tau"] = trained_tau
            summary[f"{direction}_initial_test_prototype_kendalls_tau"] = initial_tau
            for split in ("valid", "test"):
                references = trained["cross"][direction][split]["prototype"]["references"]
                task_accuracy = references["task_reference_accuracy"]
                order_accuracy = references["ordered_reference_accuracy"]
                criteria[f"{direction}_{split}_prototype_task_accuracy_at_least_0p75"] = bool(
                    task_accuracy >= 0.75
                )
                criteria[f"{direction}_{split}_prototype_order_accuracy_at_least_0p75"] = bool(
                    order_accuracy >= 0.75
                )
                summary[f"{direction}_{split}_prototype_task_accuracy"] = task_accuracy
                summary[f"{direction}_{split}_prototype_order_accuracy"] = order_accuracy
        result = {
            "protocol": "official_xskill_cross_embodiment_sugar_prototype_gate_v1",
            "scope": (
                "released XSkill skill/prototype discovery on source-ID-disjoint clean SUGAR RGB; "
                "paired G1 and task-independent four-end-effector sphere renderings of the same "
                "motions; representation admission only, not SAT or policy following"
            ),
            "official_components": {
                "commit": XSKILL_COMMIT,
                "architecture": "released 3-layer CNN, 8-layer 4-head temporal Transformer, 128 prototypes",
                "training": "released SwAV/Sinkhorn plus time-contrastive training_step through epoch79",
                "window_frames": SLIDE + 1,
                "sampled_frames": FRAMES,
                "batch_size": BATCH_SIZE,
                "seed": SEED,
            },
            "adaptation_boundary": (
                "released Franka sphere intervention is mapped to a fixed four-end-effector G1 "
                "sphere embodiment because Carry and Kick use different effectors; the visible set "
                "is identical for both tasks and the original G1 is hidden"
            ),
            "g1_corpus": str(corpus),
            "sphere_corpus": str(sphere_corpus),
            "corpus_inventory": inventory,
            "sphere_render_manifests": [
                {
                    "task": payload["task"],
                    "motion_ids": payload["motion_ids"],
                    "sphere_agent_contract": payload["sphere_agent_contract"],
                }
                for payload in sphere_manifests
            ],
            "paired_visual_difference": visual_difference,
            "official_stream_pairing": official_stream_pairing,
            "checkpoints": {"initial": str(initial_path), "trained_epoch79": str(trained_path)},
            "initial": initial,
            "trained": trained,
            "summary": summary,
            "criteria": criteria,
            "passed": bool(all(criteria.values())),
        }
        result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(summary, indent=2), flush=True)
        print(
            f"XSKILL_REPRESENTATION_RESULT passed={result['passed']} output={result_path}",
            flush=True,
        )
        return

    trained_mae = aggregate_temporal(trained, "test", "raw", "normalized_temporal_mae")
    initial_mae = aggregate_temporal(initial, "test", "raw", "normalized_temporal_mae")
    trained_tau = aggregate_temporal(trained, "test", "raw", "kendalls_tau")
    initial_tau = aggregate_temporal(initial, "test", "raw", "kendalls_tau")
    criteria = {
        "raw_test_temporal_mae_relative_improvement_at_least_5pct": bool(
            trained_mae <= 0.95 * initial_mae
        ),
        "raw_test_kendalls_tau_improvement_at_least_0p05": bool(
            trained_tau >= initial_tau + 0.05
        ),
        "raw_valid_task_reference_accuracy_at_least_0p75": bool(
            trained["valid"]["raw"]["references"]["task_reference_accuracy"] >= 0.75
        ),
        "raw_test_task_reference_accuracy_at_least_0p75": bool(
            trained["test"]["raw"]["references"]["task_reference_accuracy"] >= 0.75
        ),
        "raw_valid_ordered_reference_accuracy_at_least_0p75": bool(
            trained["valid"]["raw"]["references"]["ordered_reference_accuracy"] >= 0.75
        ),
        "raw_test_ordered_reference_accuracy_at_least_0p75": bool(
            trained["test"]["raw"]["references"]["ordered_reference_accuracy"] >= 0.75
        ),
    }
    result = {
        "protocol": "official_xskill_same_embodiment_sugar_prototype_gate_v1",
        "scope": (
            "released XSkill temporal skill/prototype discovery on source-ID-disjoint SUGAR G1 RGB; "
            "same embodiment representation admission only, not cross-embodiment replication or policy following"
        ),
        "official_components": {
            "commit": XSKILL_COMMIT,
            "architecture": "released 3-layer CNN, 8-layer 4-head temporal Transformer, 128 prototypes",
            "training": "released SwAV/Sinkhorn plus time-contrastive training_step through epoch79",
            "window_frames": SLIDE + 1,
            "sampled_frames": FRAMES,
            "batch_size": BATCH_SIZE,
            "seed": SEED,
        },
        "adaptation_boundary": (
            "SUGAR currently provides one IsaacLab G1 visual embodiment, so the two unlabeled source "
            "streams are disjoint halves of the robot-only train split; no human/robot alignment claim is made"
        ),
        "corpus_inventory": inventory,
        "checkpoints": {"initial": str(initial_path), "trained_epoch79": str(trained_path)},
        "initial": initial,
        "trained": trained,
        "summary": {
            "trained_test_raw_temporal_mae": trained_mae,
            "initial_test_raw_temporal_mae": initial_mae,
            "trained_test_raw_kendalls_tau": trained_tau,
            "initial_test_raw_kendalls_tau": initial_tau,
            "trained_valid_raw_task_accuracy": trained["valid"]["raw"]["references"]["task_reference_accuracy"],
            "trained_test_raw_task_accuracy": trained["test"]["raw"]["references"]["task_reference_accuracy"],
            "trained_valid_raw_order_accuracy": trained["valid"]["raw"]["references"]["ordered_reference_accuracy"],
            "trained_test_raw_order_accuracy": trained["test"]["raw"]["references"]["ordered_reference_accuracy"],
            "trained_test_used_prototypes": trained["prototype_usage"]["used_prototypes"],
        },
        "criteria": criteria,
        "passed": bool(all(criteria.values())),
    }
    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["summary"], indent=2), flush=True)
    print(f"XSKILL_REPRESENTATION_RESULT passed={result['passed']} output={result_path}", flush=True)


if __name__ == "__main__":
    main()
