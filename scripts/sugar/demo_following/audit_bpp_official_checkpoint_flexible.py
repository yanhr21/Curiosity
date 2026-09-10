#!/usr/bin/env python3
"""Fail-closed strict-load audit for an official BPP checkpoint.

This utility never adapts checkpoint tensors.  It instantiates the exact policy
described by the checkpoint's serialized Hydra configuration plus, when
explicitly requested, the released paper configuration override that is not
serialized in the checkpoint.  A non-strict call is used only to obtain a
complete incompatibility diagnostic; passing still requires a subsequent real
``strict=True`` call.
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-repo", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--artifact-label", required=True)
    parser.add_argument("--expected-source-commit")
    parser.add_argument("--expected-checkpoint-sha256")
    parser.add_argument(
        "--apply-official-libero-policy-dunetp",
        action="store_true",
        help=(
            "Apply only the released libero_policy_dunetp.yaml setting "
            "use_pool_modality_pos_embed=false before instantiation."
        ),
    )
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
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    result: dict[str, Any] = {
        "protocol": "sugar_bpp_official_checkpoint_flexible_strict_audit_v1",
        "artifact_label": args.artifact_label,
        "source_repo": str(source),
        "checkpoint": str(checkpoint),
        "host": socket.gethostname(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "checks": {},
        "passed": False,
        "claim_boundary": (
            "Passing proves only that the complete official checkpoint strict-loads "
            "into the official class instantiated from its serialized config and "
            "the explicitly recorded released paper-config override. "
            "It does not admit a SUGAR interface, optimizer, or training run."
        ),
    }

    def check(name: str, value: bool) -> None:
        result["checks"][name] = bool(value)
        if not value:
            raise RuntimeError(f"failed check: {name}")

    try:
        check("running_inside_slurm", bool(result["slurm_job_id"]))
        check("source_repo_exists", (source / ".git").exists())
        check("checkpoint_exists", checkpoint.is_file())
        check("checkpoint_nonempty", checkpoint.stat().st_size > 0)

        source_commit = git_output(source, "rev-parse", "HEAD")
        result["source_commit"] = source_commit
        result["source_dirty"] = bool(
            git_output(source, "status", "--porcelain", "--untracked-files=no")
        )
        check("source_checkout_clean", not result["source_dirty"])
        if args.expected_source_commit:
            check(
                "source_commit_matches_expected",
                source_commit == args.expected_source_commit,
            )

        checkpoint_sha256 = sha256_file(checkpoint)
        result["checkpoint_bytes"] = checkpoint.stat().st_size
        result["checkpoint_sha256"] = checkpoint_sha256
        if args.expected_checkpoint_sha256:
            check(
                "checkpoint_sha256_matches_expected",
                checkpoint_sha256 == args.expected_checkpoint_sha256,
            )

        sys.path.insert(0, str(source))
        os.chdir(source)
        import hydra  # pylint: disable=import-outside-toplevel
        from omegaconf import OmegaConf  # pylint: disable=import-outside-toplevel
        import torch  # pylint: disable=import-outside-toplevel

        check("cuda_available", torch.cuda.is_available())
        result["cuda_device"] = torch.cuda.get_device_name(0)
        check("h200_compute_node", "H200" in result["cuda_device"])
        result["torch_version"] = torch.__version__
        result["torch_cuda_build"] = torch.version.cuda

        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        check("payload_is_mapping", isinstance(payload, dict))
        check("payload_has_cfg", "cfg" in payload)
        check("payload_has_model_state_dict", "model" in payload.get("state_dicts", {}))
        cfg = OmegaConf.create(
            OmegaConf.to_container(payload["cfg"], resolve=False)
        )
        state_dict = payload["state_dicts"]["model"]
        check("model_state_dict_is_mapping", isinstance(state_dict, dict))

        pool_cfg_path = (
            "model.obs_encoder.obs_encoder.use_pool_modality_pos_embed"
        )
        checkpoint_pool_value = OmegaConf.select(cfg, pool_cfg_path)
        result["official_config_override"] = {
            "requested": bool(args.apply_official_libero_policy_dunetp),
            "checkpoint_serialized_value": checkpoint_pool_value,
            "applied": False,
        }
        if args.apply_official_libero_policy_dunetp:
            official_config_path = (
                source
                / "behavior_prompting"
                / "train_network"
                / "config"
                / "libero_policy_dunetp.yaml"
            )
            check("official_paper_config_exists", official_config_path.is_file())
            official_config_sha256 = sha256_file(official_config_path)
            official_config = OmegaConf.load(official_config_path)
            official_pool_value = OmegaConf.select(
                official_config, pool_cfg_path
            )
            check(
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
            result["official_config_override"].update(
                {
                    "path": str(official_config_path),
                    "sha256": official_config_sha256,
                    "official_value": official_pool_value,
                    "effective_value": OmegaConf.select(cfg, pool_cfg_path),
                    "applied": True,
                    "scope": [pool_cfg_path],
                }
            )

        def select(path: str) -> Any:
            return OmegaConf.select(cfg, path)

        result["checkpoint_config"] = {
            "model_target": select("model._target_"),
            "num_epochs": select("training.num_epochs"),
            "batch_size": select("dataloader.batch_size"),
            "action_dimension": select("task.shape_meta.action.shape.0"),
            "action_horizon": select("task.shape_meta.action.horizon"),
            "prompt_action_chunk_steps": select(
                "task.shape_meta.prompt_chunk_n_actions"
            ),
            "pool_modality_cfg_value": select(pool_cfg_path),
        }

        tensor_entries = {
            key: value for key, value in state_dict.items() if torch.is_tensor(value)
        }
        result["state_dict"] = {
            "entry_count": len(state_dict),
            "tensor_count": len(tensor_entries),
            "total_numel": sum(value.numel() for value in tensor_entries.values()),
            "all_tensors_finite": all(
                bool(torch.isfinite(value).all())
                for value in tensor_entries.values()
                if value.is_floating_point() or value.is_complex()
            ),
            "has_pool_modality_pos_embed": any(
                key.endswith("pool_modality_pos_embed") for key in state_dict
            ),
        }
        check(
            "all_state_dict_tensors_finite",
            result["state_dict"]["all_tensors_finite"],
        )

        model = hydra.utils.instantiate(cfg.model)
        model_pool = getattr(
            getattr(model.obs_encoder, "obs_encoder", None),
            "pool_modality_pos_embed",
            None,
        )
        result["instantiated_pool_modality_pos_embed"] = {
            "present": model_pool is not None,
            "shape": list(model_pool.shape) if model_pool is not None else None,
            "requires_grad": (
                bool(model_pool.requires_grad) if model_pool is not None else None
            ),
        }
        result["model_unique_parameter_count"] = sum(
            parameter.numel() for parameter in model.parameters()
        )

        diagnostic = model.load_state_dict(state_dict.copy(), strict=False)
        missing = sorted(diagnostic.missing_keys)
        unexpected = sorted(diagnostic.unexpected_keys)
        result["strict_load_diagnostic"] = {
            "missing_keys": missing,
            "unexpected_keys": unexpected,
        }
        result["checks"]["strict_load_missing_keys_zero"] = not missing
        result["checks"]["strict_load_unexpected_keys_zero"] = not unexpected
        if missing or unexpected:
            raise RuntimeError(
                f"official checkpoint is not strict-loadable: "
                f"missing={missing}, unexpected={unexpected}"
            )

        strict_result = model.load_state_dict(state_dict.copy(), strict=True)
        check(
            "strict_true_call_completed",
            not strict_result.missing_keys and not strict_result.unexpected_keys,
        )
        result["passed"] = True
    except Exception as error:  # fail closed while preserving structured evidence
        result["error"] = {
            "type": type(error).__name__,
            "message": str(error),
        }
    finally:
        output.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
