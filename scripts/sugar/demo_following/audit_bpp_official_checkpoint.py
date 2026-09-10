#!/usr/bin/env python3
"""Fail-closed audit of the released BPP Goal Chain checkpoint.

This script imports and instantiates the official Behavior Prompting Policy
classes from a pinned source checkout.  It does not define or substitute a
model.  The audit verifies artifact identity, checkpoint/config topology, an
exact strict state-dict load, and the two independent official visual towers.
"""

from __future__ import annotations

import argparse
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-repo", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--interface-contract", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_output(repo: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), *args], text=True
    ).strip()


def canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def require(checks: dict[str, bool], key: str, condition: bool) -> None:
    checks[key] = bool(condition)
    if not condition:
        raise RuntimeError(f"failed check: {key}")


def main() -> int:
    args = parse_args()
    source_repo = args.source_repo.resolve()
    checkpoint = args.checkpoint.resolve()
    contract_path = args.interface_contract.resolve()
    output_dir = args.output_dir.resolve()

    result_path = output_dir / "BPP_OFFICIAL_CHECKPOINT_AUDIT.json"
    manifest_path = output_dir / "BPP_OFFICIAL_STATE_DICT_KEY_SHAPE_MANIFEST.json"
    result: dict[str, Any] = {
        "protocol": "sugar_bpp_official_checkpoint_strict_audit_v2",
        "passed": False,
        "checks": {},
        "source_repo": str(source_repo),
        "checkpoint": str(checkpoint),
        "interface_contract": str(contract_path),
        "slurm": {
            "job_id": os.environ.get("SLURM_JOB_ID"),
            "job_name": os.environ.get("SLURM_JOB_NAME"),
            "node": os.environ.get("SLURMD_NODENAME") or os.uname().nodename,
            "job_gpus": os.environ.get("SLURM_JOB_GPUS"),
            "step_gpus": os.environ.get("SLURM_STEP_GPUS"),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
        "claim_boundary": (
            "Passing proves artifact identity, released Goal Chain topology, "
            "and an exact strict load of the official EMA policy state only. "
            "It does not prove a SUGAR adapter, prompt dependence, training, "
            "or physical success."
        ),
    }

    try:
        checks = result["checks"]
        require(checks, "source_repo_exists", source_repo.is_dir())
        require(checks, "checkpoint_exists", checkpoint.is_file())
        require(checks, "interface_contract_exists", contract_path.is_file())

        contract = json.loads(contract_path.read_text(encoding="utf-8"))
        contract_sha256 = sha256_file(contract_path)
        checkpoint_sha256 = sha256_file(checkpoint)
        source_commit = git_output(source_repo, "rev-parse", "HEAD")
        source_dirty = bool(
            git_output(
                source_repo, "status", "--porcelain", "--untracked-files=no"
            )
        )
        artifact = contract["official_artifact"]

        result.update(
            {
                "contract_sha256": contract_sha256,
                "checkpoint_sha256": checkpoint_sha256,
                "checkpoint_bytes": checkpoint.stat().st_size,
                "source_commit": source_commit,
                "source_dirty": source_dirty,
            }
        )
        require(
            checks,
            "source_commit_matches_contract",
            source_commit == artifact["source_commit"],
        )
        require(checks, "source_checkout_clean", not source_dirty)
        require(
            checks,
            "checkpoint_sha256_matches_contract",
            checkpoint_sha256 == artifact["checkpoint_sha256"],
        )

        sys.path.insert(0, str(source_repo))
        import dill  # pylint: disable=import-outside-toplevel
        import hydra  # pylint: disable=import-outside-toplevel
        from omegaconf import OmegaConf  # pylint: disable=import-outside-toplevel
        import torch  # pylint: disable=import-outside-toplevel

        from behavior_prompting.train_network.model.prompt.prompt_obs_encoder import (  # pylint: disable=import-outside-toplevel
            PairPromptObsEncoder,
            PairPromptTransformerTokenizer,
        )
        from behavior_prompting.train_network.model.vision.transformer_obs_encoder import (  # pylint: disable=import-outside-toplevel
            TransformerObsEncoder,
        )
        from behavior_prompting.train_network.policy.diffusion_unet_policy import (  # pylint: disable=import-outside-toplevel
            DiffusionUnetPolicy,
        )

        expected_versions = {
            "torch": "2.8.0",
            "torchvision": "0.23.0",
            "accelerate": "1.10.1",
            "huggingface-hub": "0.35.3",
            "transformers": "4.57.1",
            "diffusers": "0.35.1",
            "hydra-core": "1.2.0",
            "timm": "1.0.20",
        }
        runtime_versions = {
            package: metadata.version(package) for package in expected_versions
        }
        result["runtime_versions"] = runtime_versions
        result["torch_cuda_build"] = torch.version.cuda
        require(checks, "running_inside_slurm", bool(os.environ.get("SLURM_JOB_ID")))
        require(checks, "cuda_available", torch.cuda.is_available())
        require(
            checks,
            "official_cuda_12_8_build",
            bool(torch.version.cuda) and torch.version.cuda.startswith("12.8"),
        )
        cuda_device_names = [
            torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())
        ]
        result["cuda_device_names"] = cuda_device_names
        require(
            checks,
            "h200_compute_node",
            bool(cuda_device_names) and all("H200" in name for name in cuda_device_names),
        )
        for package, expected_version in expected_versions.items():
            require(
                checks,
                f"official_version::{package}",
                runtime_versions[package].split("+")[0] == expected_version,
            )

        with checkpoint.open("rb") as stream:
            payload = torch.load(
                stream, map_location="cpu", pickle_module=dill, weights_only=False
            )
        require(checks, "payload_is_mapping", isinstance(payload, dict))
        require(checks, "payload_has_cfg", "cfg" in payload)
        require(checks, "payload_has_state_dicts", "state_dicts" in payload)
        require(
            checks,
            "payload_has_model_state_dict",
            isinstance(payload["state_dicts"], dict)
            and "model" in payload["state_dicts"],
        )

        cfg = OmegaConf.create(
            OmegaConf.to_container(payload["cfg"], resolve=False)
        )
        state_dict = payload["state_dicts"]["model"]
        result["payload_state_dict_keys"] = sorted(payload["state_dicts"])
        result["payload_pickle_keys"] = sorted(payload.get("pickles", {}))
        require(checks, "model_state_dict_is_mapping", isinstance(state_dict, dict))
        require(checks, "model_state_dict_nonempty", len(state_dict) > 0)

        pool_cfg_path = (
            "model.obs_encoder.obs_encoder.use_pool_modality_pos_embed"
        )
        checkpoint_pool_value = OmegaConf.select(cfg, pool_cfg_path)
        official_config_path = (
            source_repo
            / "behavior_prompting"
            / "train_network"
            / "config"
            / "libero_policy_dunetp.yaml"
        )
        require(checks, "official_paper_config_exists", official_config_path.is_file())
        official_config = OmegaConf.load(official_config_path)
        official_pool_value = OmegaConf.select(official_config, pool_cfg_path)
        require(
            checks,
            "official_paper_config_disables_pool_modality_pos_embed",
            official_pool_value is False,
        )
        OmegaConf.update(
            cfg,
            pool_cfg_path,
            official_pool_value,
            merge=False,
            force_add=True,
        )
        result["official_config_override"] = {
            "path": str(official_config_path),
            "sha256": sha256_file(official_config_path),
            "checkpoint_serialized_value": checkpoint_pool_value,
            "official_value": official_pool_value,
            "effective_value": OmegaConf.select(cfg, pool_cfg_path),
            "scope": [pool_cfg_path],
        }

        def select(path: str) -> Any:
            return OmegaConf.select(cfg, path)

        expected = contract["topology_invariants"]
        official = contract["official_interface"]
        config_evidence = {
            "model_target": select("model._target_"),
            "num_epochs": select("training.num_epochs"),
            "batch_size": select("dataloader.batch_size"),
            "action_dimension": select("task.shape_meta.action.shape.0"),
            "action_horizon": select("task.shape_meta.action.horizon"),
            "prompt_action_chunk_steps": select("task.shape_meta.prompt_chunk_n_actions"),
            "prompt_sequence_length": select("task.shape_meta.prompt_sequence_length"),
            "rgb_model_name": select("model.obs_encoder.obs_encoder.obs_encoder.model_name"),
            "share_rgb_model": select("model.obs_encoder.obs_encoder.obs_encoder.share_rgb_model"),
            "use_pool_modality_pos_embed": select(pool_cfg_path),
            "diffusion_train_timesteps": select("model.noise_scheduler.num_train_timesteps"),
            "diffusion_inference_steps": select("model.num_inference_steps"),
            "unet_down_dims": list(select("model.down_dims")),
        }
        result["config_evidence"] = config_evidence
        require(checks, "goal_chain_num_epochs_is_7", config_evidence["num_epochs"] == 7)
        require(checks, "released_batch_size_is_100", config_evidence["batch_size"] == 100)
        require(
            checks,
            "official_action_dimension_matches_contract",
            config_evidence["action_dimension"] == official["action_dimension"],
        )
        require(
            checks,
            "official_prompt_action_chunk_matches_contract",
            config_evidence["prompt_action_chunk_steps"]
            == official["prompt_action_chunk_steps"],
        )
        require(
            checks,
            "action_horizon_matches_contract",
            config_evidence["action_horizon"] == expected["action_horizon"],
        )
        require(
            checks,
            "rgb_backbone_matches_contract",
            config_evidence["rgb_model_name"] == expected["rgb_backbone"],
        )
        require(
            checks,
            "rgb_weight_sharing_disabled",
            config_evidence["share_rgb_model"] is False,
        )
        require(
            checks,
            "pool_modality_pos_embed_disabled_for_released_paper_model",
            config_evidence["use_pool_modality_pos_embed"] is False,
        )
        require(
            checks,
            "diffusion_schedule_matches_contract",
            config_evidence["diffusion_train_timesteps"]
            == expected["diffusion_train_timesteps"]
            and config_evidence["diffusion_inference_steps"]
            == expected["diffusion_inference_steps"],
        )
        require(
            checks,
            "unet_widths_match_contract",
            config_evidence["unet_down_dims"] == expected["unet_down_dims"],
        )

        key_shape_manifest = []
        nonfinite_keys = []
        non_tensor_state_keys = []
        for key, value in sorted(state_dict.items()):
            if not torch.is_tensor(value):
                non_tensor_state_keys.append(
                    {
                        "key": key,
                        "python_type": (
                            f"{type(value).__module__}.{type(value).__name__}"
                        ),
                    }
                )
                continue
            if not bool(torch.isfinite(value).all()):
                nonfinite_keys.append(key)
            key_shape_manifest.append(
                {
                    "key": key,
                    "shape": list(value.shape),
                    "dtype": str(value.dtype),
                    "numel": value.numel(),
                }
            )
        result["nonfinite_state_dict_keys"] = nonfinite_keys
        result["non_tensor_state_dict_keys"] = non_tensor_state_keys
        require(checks, "all_state_dict_tensors_finite", not nonfinite_keys)
        require(
            checks,
            "only_official_training_split_extra_state_is_non_tensor",
            [row["key"] for row in non_tensor_state_keys]
            in ([], ["_extra_training_split_info"]),
        )
        manifest_sha256 = canonical_json_sha256(key_shape_manifest)
        manifest = {
            "protocol": "sugar_bpp_official_state_dict_key_shape_manifest_v1",
            "checkpoint_sha256": checkpoint_sha256,
            "tensor_count": len(key_shape_manifest),
            "total_numel": sum(row["numel"] for row in key_shape_manifest),
            "manifest_sha256": manifest_sha256,
            "tensors": key_shape_manifest,
        }
        atomic_json(manifest_path, manifest)
        result["state_dict"] = {
            "tensor_count": manifest["tensor_count"],
            "total_numel": manifest["total_numel"],
            "key_shape_manifest_sha256": manifest_sha256,
            "manifest_path": str(manifest_path),
        }

        model = hydra.utils.instantiate(cfg.model)
        released_pool_pos_embed = (
            model.obs_encoder.obs_encoder.pool_modality_pos_embed
        )
        result["released_model_pool_modality_pos_embed"] = {
            "checkpoint_cfg_field_present": (
                config_evidence["use_pool_modality_pos_embed"] is not None
            ),
            "checkpoint_cfg_value": config_evidence[
                "use_pool_modality_pos_embed"
            ],
            "instantiated_parameter_present": (
                released_pool_pos_embed is not None
            ),
            "instantiated_parameter_shape": (
                list(released_pool_pos_embed.shape)
                if released_pool_pos_embed is not None
                else None
            ),
            "instantiated_parameter_requires_grad": (
                bool(released_pool_pos_embed.requires_grad)
                if released_pool_pos_embed is not None
                else None
            ),
        }
        require(checks, "official_policy_class", isinstance(model, DiffusionUnetPolicy))
        # BasePolicy.load_state_dict removes the official extra-state entry in
        # place, so use shallow copies and keep the checkpoint payload immutable.
        # First collect an exact key diagnostic.  A non-strict diagnostic is not
        # accepted as inheritance: any missing/unexpected key fails closed, and
        # a real strict=True call is still mandatory when the diagnostic is clean.
        incompatibility = model.load_state_dict(state_dict.copy(), strict=False)
        missing_keys = list(incompatibility.missing_keys)
        unexpected_keys = list(incompatibility.unexpected_keys)
        result["strict_load_diagnostic"] = {
            "missing_keys": missing_keys,
            "unexpected_keys": unexpected_keys,
        }
        checks["strict_load_missing_keys_zero"] = not missing_keys
        checks["strict_load_unexpected_keys_zero"] = not unexpected_keys
        if missing_keys or unexpected_keys:
            raise RuntimeError(
                "official checkpoint cannot strict-load into its released "
                f"configuration: missing={missing_keys}, "
                f"unexpected={unexpected_keys}"
            )
        strict_incompatibility = model.load_state_dict(
            state_dict.copy(), strict=True
        )
        require(
            checks,
            "strict_load_call_completed",
            not strict_incompatibility.missing_keys
            and not strict_incompatibility.unexpected_keys,
        )

        pair_encoders = [m for m in model.modules() if isinstance(m, PairPromptObsEncoder)]
        tokenizers = [
            m for m in model.modules() if isinstance(m, PairPromptTransformerTokenizer)
        ]
        visual_encoders = [
            m for m in model.modules() if isinstance(m, TransformerObsEncoder)
        ]
        require(checks, "one_pair_prompt_encoder", len(pair_encoders) == 1)
        require(checks, "one_pair_prompt_tokenizer", len(tokenizers) == 1)
        require(checks, "one_shared_prompt_current_visual_encoder", len(visual_encoders) == 1)

        visual_encoder = visual_encoders[0]
        rgb_keys = expected["rgb_modalities"]
        require(
            checks,
            "both_rgb_modalities_present",
            all(key in visual_encoder.key_model_map for key in rgb_keys),
        )
        tower_a = visual_encoder.key_model_map[rgb_keys[0]]
        tower_b = visual_encoder.key_model_map[rgb_keys[1]]
        require(checks, "rgb_tower_modules_are_distinct", tower_a is not tower_b)
        tower_a_parameters = list(tower_a.parameters())
        tower_b_parameters = list(tower_b.parameters())
        require(
            checks,
            "rgb_tower_parameter_counts_match",
            len(tower_a_parameters) == len(tower_b_parameters) > 0,
        )
        require(
            checks,
            "rgb_tower_parameter_storage_is_disjoint",
            all(a.data_ptr() != b.data_ptr() for a, b in zip(tower_a_parameters, tower_b_parameters)),
        )

        named_parameters = dict(model.named_parameters())
        named_buffers = dict(model.named_buffers())
        result["model"] = {
            "class": f"{type(model).__module__}.{type(model).__name__}",
            "parameter_tensor_count": len(named_parameters),
            "parameter_count": sum(value.numel() for value in named_parameters.values()),
            "trainable_parameter_count": sum(
                value.numel() for value in named_parameters.values() if value.requires_grad
            ),
            "buffer_tensor_count": len(named_buffers),
            "pair_prompt_encoder_count": len(pair_encoders),
            "pair_prompt_tokenizer_count": len(tokenizers),
            "transformer_obs_encoder_count": len(visual_encoders),
            "rgb_tower_parameter_count_each": [
                sum(value.numel() for value in tower_a_parameters),
                sum(value.numel() for value in tower_b_parameters),
            ],
        }

        checks["all_checks_passed"] = all(checks.values())
        result["passed"] = checks["all_checks_passed"]
    except Exception as exc:  # fail closed while preserving evidence
        result["error"] = {"type": type(exc).__name__, "message": str(exc)}
        result["passed"] = False
    finally:
        atomic_json(result_path, result)

    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
