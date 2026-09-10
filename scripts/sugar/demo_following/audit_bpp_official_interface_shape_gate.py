#!/usr/bin/env python3
"""Audit the frozen BPP 29-D/121-D interface against official constructors.

No adapter or replacement module is defined here.  The script strict-loads the
released model, instantiates the same released classes with only the five
contracted shape-meta changes, and counts shape changes by unique Parameter
identity (while retaining every state-dict alias in the evidence).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
from typing import Any


EXPECTED_CONTRACT_SHA256 = (
    "b7b1780672fcf3462e707fcbf769fc6479834c471ba2b34815db05cdea36644a"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-repo", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_output(repo: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), *args], text=True
    ).strip()


def main() -> int:
    args = parse_args()
    source = args.source_repo.resolve()
    checkpoint = args.checkpoint.resolve()
    contract_path = args.contract.resolve()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any] = {
        "protocol": "sugar_bpp_official_interface_shape_gate_v1",
        "passed": False,
        "host": socket.gethostname(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "source_repo": str(source),
        "checkpoint": str(checkpoint),
        "contract": str(contract_path),
        "checks": {},
        "claim_boundary": (
            "Passing proves only that the released constructors produce exactly "
            "the eight unique shape-new trainable Parameter identities frozen by "
            "the interface contract, with zero shape-new buffers. It does not "
            "load an adapted model, run a forward, or admit training."
        ),
    }

    def check(name: str, condition: bool) -> None:
        result["checks"][name] = bool(condition)
        if not condition:
            raise RuntimeError(f"failed check: {name}")

    try:
        check("running_inside_slurm", bool(result["slurm_job_id"]))
        check("source_exists", (source / ".git").exists())
        check("checkpoint_exists", checkpoint.is_file())
        check("contract_exists", contract_path.is_file())
        contract = json.loads(contract_path.read_text(encoding="utf-8"))
        contract_sha256 = sha256_file(contract_path)
        result["contract_sha256"] = contract_sha256
        check(
            "contract_sha256_matches_frozen",
            contract_sha256 == EXPECTED_CONTRACT_SHA256,
        )
        source_commit = git_output(source, "rev-parse", "HEAD")
        result["source_commit"] = source_commit
        check(
            "source_commit_matches_contract",
            source_commit == contract["official_artifact"]["source_commit"],
        )
        check(
            "source_checkout_clean",
            not git_output(
                source, "status", "--porcelain", "--untracked-files=no"
            ),
        )
        checkpoint_sha256 = sha256_file(checkpoint)
        result["checkpoint_sha256"] = checkpoint_sha256
        check(
            "checkpoint_sha256_matches_contract",
            checkpoint_sha256
            == contract["official_artifact"]["checkpoint_sha256"],
        )

        sys.path.insert(0, str(source))
        os.chdir(source)
        import dill  # pylint: disable=import-outside-toplevel
        import hydra  # pylint: disable=import-outside-toplevel
        from omegaconf import OmegaConf  # pylint: disable=import-outside-toplevel
        import torch  # pylint: disable=import-outside-toplevel

        check("cuda_available", torch.cuda.is_available())
        result["cuda_device"] = torch.cuda.get_device_name(0)
        check("h200_compute_node", "H200" in result["cuda_device"])

        with checkpoint.open("rb") as stream:
            payload = torch.load(
                stream,
                map_location="cpu",
                pickle_module=dill,
                weights_only=False,
            )
        state_dict = payload["state_dicts"]["model"]
        pool_cfg_path = (
            "model.obs_encoder.obs_encoder.use_pool_modality_pos_embed"
        )
        paper_config_path = (
            source
            / "behavior_prompting"
            / "train_network"
            / "config"
            / "libero_policy_dunetp.yaml"
        )
        paper_config = OmegaConf.load(paper_config_path)
        paper_pool_value = OmegaConf.select(paper_config, pool_cfg_path)
        check("paper_config_disables_pool_pos_embed", paper_pool_value is False)

        official_cfg = OmegaConf.create(
            OmegaConf.to_container(payload["cfg"], resolve=False)
        )
        OmegaConf.update(
            official_cfg,
            pool_cfg_path,
            paper_pool_value,
            merge=False,
            force_add=True,
        )
        official_model = hydra.utils.instantiate(official_cfg.model)
        strict_result = official_model.load_state_dict(
            state_dict.copy(), strict=True
        )
        check(
            "unmodified_official_model_strict_loaded",
            not strict_result.missing_keys and not strict_result.unexpected_keys,
        )

        sugar_cfg = OmegaConf.create(
            OmegaConf.to_container(official_cfg, resolve=False)
        )
        # The released checkpoint serializes already-resolved copies of
        # ``task.shape_meta`` at every official constructor boundary.  Updating
        # only the source task node therefore does not alter the model that
        # Hydra instantiates.  Apply the same five contracted semantic values
        # to every official model-owned copy; no field outside shape_meta is
        # changed.
        shape_meta_roots = (
            "shape_meta",
            "task.shape_meta",
            "model.shape_meta",
            "model.obs_encoder.shape_meta",
            "model.obs_encoder.obs_encoder.shape_meta",
            "model.obs_encoder.obs_encoder.obs_encoder.shape_meta",
        )
        shape_updates: dict[str, Any] = {}
        relative_updates = {
            "action.shape": [29],
            "prompt_chunk_n_actions": 50,
            "obs.ee_pos.shape": [68],
            "obs.ee_ori.shape": [29],
            "obs.gripper_states.shape": [24],
        }
        for root in shape_meta_roots:
            check(
                f"serialized_shape_meta_copy_exists::{root}",
                OmegaConf.select(sugar_cfg, root) is not None,
            )
            for suffix, value in relative_updates.items():
                path = f"{root}.{suffix}"
                OmegaConf.update(
                    sugar_cfg, path, value, merge=False, force_add=False
                )
                shape_updates[path] = value

        effective_shape_metas = {
            root: OmegaConf.to_container(
                OmegaConf.select(sugar_cfg, root), resolve=True
            )
            for root in shape_meta_roots
        }
        effective_shape_meta = effective_shape_metas["model.shape_meta"]
        result["shape_updates"] = shape_updates
        result["serialized_shape_meta_roots"] = list(shape_meta_roots)
        result["effective_model_shape_meta"] = {
            "action_shape": effective_shape_meta["action"]["shape"],
            "prompt_chunk_n_actions": effective_shape_meta[
                "prompt_chunk_n_actions"
            ],
            "lowdim_shapes": {
                key: effective_shape_meta["obs"][key]["shape"]
                for key in ("ee_pos", "ee_ori", "gripper_states")
            },
        }
        check(
            "effective_action_shape_is_29",
            effective_shape_meta["action"]["shape"] == [29],
        )
        check(
            "effective_prompt_chunk_is_50",
            effective_shape_meta["prompt_chunk_n_actions"] == 50,
        )
        for root, candidate in effective_shape_metas.items():
            check(
                f"effective_shape_meta_copy_matches_contract::{root}",
                candidate["action"]["shape"] == [29]
                and candidate["prompt_chunk_n_actions"] == 50
                and candidate["obs"]["ee_pos"]["shape"] == [68]
                and candidate["obs"]["ee_ori"]["shape"] == [29]
                and candidate["obs"]["gripper_states"]["shape"] == [24],
            )

        torch.manual_seed(271500)
        torch.cuda.manual_seed_all(271500)
        sugar_model = hydra.utils.instantiate(sugar_cfg.model)

        official_named = dict(
            official_model.named_parameters(remove_duplicate=False)
        )
        sugar_named = dict(
            sugar_model.named_parameters(remove_duplicate=False)
        )
        official_trainable = {
            name: parameter
            for name, parameter in official_named.items()
            if parameter.requires_grad
        }
        sugar_trainable = {
            name: parameter
            for name, parameter in sugar_named.items()
            if parameter.requires_grad
        }
        check(
            "trainable_parameter_alias_key_sets_equal",
            set(official_trainable) == set(sugar_trainable),
        )

        groups: dict[int, dict[str, Any]] = {}
        for name, sugar_parameter in sugar_trainable.items():
            official_parameter = official_trainable[name]
            if tuple(official_parameter.shape) == tuple(sugar_parameter.shape):
                continue
            group = groups.setdefault(
                id(sugar_parameter),
                {
                    "aliases": [],
                    "official_shape": list(official_parameter.shape),
                    "sugar_shape": list(sugar_parameter.shape),
                },
            )
            check(
                f"shape_consistent_within_alias_group::{name}",
                group["official_shape"] == list(official_parameter.shape)
                and group["sugar_shape"] == list(sugar_parameter.shape),
            )
            group["aliases"].append(name)

        mismatch_groups = sorted(
            groups.values(), key=lambda row: row["aliases"][0]
        )
        for row in mismatch_groups:
            row["aliases"].sort()
            first_name = row["aliases"][0]
            module_path = first_name.rsplit(".", 1)[0]
            module = dict(
                sugar_model.named_modules(remove_duplicate=False)
            )[module_path]
            row["module_class"] = (
                f"{type(module).__module__}.{type(module).__name__}"
            )
        result["unique_trainable_shape_mismatch_count"] = len(mismatch_groups)
        result["trainable_shape_mismatch_alias_count"] = sum(
            len(row["aliases"]) for row in mismatch_groups
        )
        result["trainable_shape_mismatch_groups"] = mismatch_groups

        official_buffers = dict(
            official_model.named_buffers(remove_duplicate=False)
        )
        sugar_buffers = dict(sugar_model.named_buffers(remove_duplicate=False))
        buffer_mismatches = []
        for name in sorted(set(official_buffers) | set(sugar_buffers)):
            official_buffer = official_buffers.get(name)
            sugar_buffer = sugar_buffers.get(name)
            if official_buffer is None or sugar_buffer is None or (
                tuple(official_buffer.shape) != tuple(sugar_buffer.shape)
            ):
                buffer_mismatches.append(
                    {
                        "name": name,
                        "official_shape": (
                            list(official_buffer.shape)
                            if official_buffer is not None
                            else None
                        ),
                        "sugar_shape": (
                            list(sugar_buffer.shape)
                            if sugar_buffer is not None
                            else None
                        ),
                    }
                )
        result["buffer_shape_mismatches"] = buffer_mismatches
        check("shape_new_buffer_count_zero", not buffer_mismatches)
        check(
            "shape_new_unique_trainable_parameter_count_is_contract_eight",
            len(mismatch_groups)
            == contract["expected_shape_new_parameter_tensor_count"],
        )
        result["passed"] = True
    except Exception as error:  # fail closed with complete evidence
        result["error"] = {
            "type": type(error).__name__,
            "message": str(error),
        }
    finally:
        output.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
