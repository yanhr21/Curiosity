# Copyright (c) 2026, Curiosity Project.
# SPDX-License-Identifier: BSD-3-Clause

"""Frozen official MimicKit TinyMDM/ESM scorer for live SUGAR windows.

This module is integration glue only.  It imports the pinned upstream
``TinyMDMModel`` and ``DiffNormalizer`` and preserves ``SMPAgent``'s ensemble
SDS equations, rollout-delayed normalizer update, and exponential reward.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import sys
from typing import Any

import torch
import yaml


WORKSPACE_ROOT = Path(__file__).resolve().parents[5]
MIMICKIT_ROOT = WORKSPACE_ROOT / "MimicKit"
MIMICKIT_PYTHON = MIMICKIT_ROOT / "mimickit"
PINNED_MIMICKIT_COMMIT = "2ed1e6c093bb0829f55d33cb4f7a1731cfe6cb69"


def _git_head(repo: Path) -> str:
    head = (repo / ".git/HEAD").read_text(encoding="utf-8").strip()
    if not head.startswith("ref: "):
        return head
    ref = head.removeprefix("ref: ")
    loose = repo / ".git" / ref
    if loose.is_file():
        return loose.read_text(encoding="utf-8").strip()
    for line in (repo / ".git/packed-refs").read_text(encoding="utf-8").splitlines():
        if line and not line.startswith("#") and line.endswith(f" {ref}"):
            return line.split()[0]
    raise RuntimeError(f"cannot resolve MimicKit HEAD {head}")


@dataclass(frozen=True)
class OfficialSMPScorerCfg:
    diffusion_steps: tuple[int, ...] = (22, 15, 8)
    sds_loss_scale: float = 6.0
    smp_reward_scale: float = 1.0
    eval_batch_size: int = 4096
    sds_normalizer_samples: int | None = None

    def __post_init__(self) -> None:
        if self.diffusion_steps != (22, 15, 8):
            raise ValueError("initial SUGAR scorer preserves official SMP ESM steps [22,15,8]")
        if self.sds_loss_scale != 6.0:
            raise ValueError("initial SUGAR scorer preserves official sds_loss_scale=6")
        if self.smp_reward_scale <= 0.0 or self.eval_batch_size < 1:
            raise ValueError("invalid SMP reward scale or evaluation batch size")
        if self.sds_normalizer_samples is not None and self.sds_normalizer_samples < 1:
            raise ValueError("sds_normalizer_samples must be positive or None")


class OfficialSugarSMPScorer:
    """Load an admitted prior and reproduce official SMP reward computation."""

    def __init__(
        self,
        prior_dir: str | Path,
        device: torch.device | str,
        cfg: OfficialSMPScorerCfg = OfficialSMPScorerCfg(),
    ) -> None:
        self.prior_dir = Path(prior_dir).expanduser().resolve()
        self.device = torch.device(device)
        self.cfg = cfg
        if self.device.type != "cuda":
            raise ValueError("formal SMP scoring must run on a compute GPU")
        if _git_head(MIMICKIT_ROOT) != PINNED_MIMICKIT_COMMIT:
            raise RuntimeError("pinned MimicKit checkout drift")
        self.upstream_commit = PINNED_MIMICKIT_COMMIT

        result_path = self.prior_dir / "result.json"
        model_path = self.prior_dir / "model.pt"
        config_path = self.prior_dir / "diffusion_config.yaml"
        for path in (result_path, model_path, config_path):
            if not path.is_file():
                raise FileNotFoundError(path)
        self.prior_result = json.loads(result_path.read_text(encoding="utf-8"))
        if (
            self.prior_result.get("protocol")
            != "sugar_g1_box_official_tinymdm_training_v1"
            or self.prior_result.get("passed") is not True
            or self.prior_result.get("formal_run") is not True
            or self.prior_result.get("admitted_as_prior") is not True
            or self.prior_result.get("completed_iterations") != 200000
        ):
            raise RuntimeError("refusing an incomplete or diagnostic TinyMDM prior")
        self.prior_identity = {
            "protocol": self.prior_result["protocol"],
            "completed_iterations": int(
                self.prior_result["completed_iterations"]
            ),
            "parameter_count_trainable": int(
                self.prior_result["parameter_count_trainable"]
            ),
            "mimickit_commit": self.prior_result["mimickit_commit"],
        }
        if self.prior_identity["mimickit_commit"] != self.upstream_commit:
            raise RuntimeError("TinyMDM prior and MimicKit checkout differ")

        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if (
            config.get("input_channel") != 216
            or config.get("num_disc_obs_steps") != 10
            or config.get("control_freq") != 50
            or config.get("model_name") != "tiny_mdm"
        ):
            raise ValueError("formal SUGAR prior runtime contract drift")
        # Training metadata records the original absolute experiment path.
        # Runtime bundles are relocatable: TinyMDM's small environment schema
        # lives beside its frozen checkpoint and must be resolved there.
        env_config_path = self.prior_dir / "env_config.yaml"
        if not env_config_path.is_file():
            raise FileNotFoundError(env_config_path)
        config["env_config"] = str(env_config_path)
        config["input_dim"] = 10 * 216

        sys.path.insert(0, str(MIMICKIT_PYTHON))
        from learning.diff_normalizer import DiffNormalizer  # noqa: PLC0415
        from learning.tinymdm.tinymdm_model import TinyMDMModel  # noqa: PLC0415

        self.prior = TinyMDMModel(config, self.device).to(self.device)
        incompatible = self.prior.load_state_dict(
            torch.load(model_path, map_location=self.device, weights_only=True),
            strict=True,
        )
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise RuntimeError(f"formal TinyMDM state incompatibility: {incompatible}")
        self.prior.eval()
        for parameter in self.prior.parameters():
            parameter.requires_grad_(False)

        self.sds_normalizer = DiffNormalizer(
            [len(cfg.diffusion_steps)], device=self.device, dtype=torch.float32
        )
        self.transitions_scored = 0
        self.normalizer_samples_recorded = 0
        self.normalizer_updates = 0
        self._pending_normalizer_samples = 0

    def _normalizer_learning_active(self) -> bool:
        maximum = self.cfg.sds_normalizer_samples
        return maximum is None or self.normalizer_samples_recorded < maximum

    @torch.no_grad()
    def score(
        self, feature_windows: torch.Tensor, record_normalizer: bool = True
    ) -> dict[str, torch.Tensor]:
        if feature_windows.ndim != 3 or feature_windows.shape[1:] != (10, 216):
            raise ValueError(
                f"SMP windows must be (batch,10,216), got {tuple(feature_windows.shape)}"
            )
        if feature_windows.device != self.device or not torch.isfinite(feature_windows).all():
            raise ValueError("SMP windows must be finite and resident on the scorer device")
        batch = feature_windows.shape[0]
        raw_chunks = []
        reward_chunks = []
        normalized_chunks = []
        for begin in range(0, batch, self.cfg.eval_batch_size):
            windows = feature_windows[begin : begin + self.cfg.eval_batch_size]
            normalized_features = self.prior.normalize(windows).reshape(windows.shape[0], -1)
            raw_sds = self.prior.ESM_SDS_loss(
                norm_x_obs=normalized_features,
                t_lst=list(self.cfg.diffusion_steps),
            )
            if record_normalizer and self._normalizer_learning_active():
                self.sds_normalizer.record(raw_sds)
                self._pending_normalizer_samples += int(raw_sds.shape[0])
                self.normalizer_samples_recorded += int(raw_sds.shape[0])
            normalized_sds = self.sds_normalizer.normalize(raw_sds)
            mean_normalized = normalized_sds.mean(dim=-1)
            reward = torch.exp(-mean_normalized * self.cfg.sds_loss_scale)
            reward = reward * self.cfg.smp_reward_scale
            raw_chunks.append(raw_sds)
            normalized_chunks.append(normalized_sds)
            reward_chunks.append(reward)
        self.transitions_scored += batch
        raw = torch.cat(raw_chunks, dim=0)
        normalized = torch.cat(normalized_chunks, dim=0)
        reward = torch.cat(reward_chunks, dim=0)
        return {
            "smp_reward": reward,
            "raw_sds_by_step": raw,
            "normalized_sds_by_step": normalized,
            "raw_sds_mean": raw.mean(dim=-1),
        }

    @torch.no_grad()
    def commit_rollout_normalizer(self) -> None:
        """Match ``SMPAgent._train_iter``: update only after rollout scoring."""

        if self._pending_normalizer_samples > 0:
            self.sds_normalizer.update()
            self.normalizer_updates += 1
            self._pending_normalizer_samples = 0

    def state_dict(self) -> dict[str, Any]:
        if self._pending_normalizer_samples != 0:
            raise RuntimeError("checkpoint SMP scorer only at a rollout boundary")
        return {
            "protocol": "sugar_official_smp_scorer_v1",
            "config": asdict(self.cfg),
            "prior_identity": dict(self.prior_identity),
            "upstream_commit": self.upstream_commit,
            "sds_normalizer_state_dict": self.sds_normalizer.state_dict(),
            "transitions_scored": self.transitions_scored,
            "normalizer_samples_recorded": self.normalizer_samples_recorded,
            "normalizer_updates": self.normalizer_updates,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if state.get("protocol") != "sugar_official_smp_scorer_v1":
            raise ValueError("unexpected SMP scorer checkpoint protocol")
        expected_cfg = asdict(self.cfg)
        # JSON checkpoints may materialize tuples as lists.
        actual_cfg = dict(state.get("config", {}))
        actual_cfg["diffusion_steps"] = tuple(actual_cfg.get("diffusion_steps", ()))
        if actual_cfg != expected_cfg:
            raise ValueError("SMP scorer checkpoint config drift")
        if state.get("prior_identity") != self.prior_identity:
            raise ValueError("SMP scorer checkpoint prior drift")
        if state.get("upstream_commit") != self.upstream_commit:
            raise ValueError("SMP scorer checkpoint upstream source drift")
        self.sds_normalizer.load_state_dict(state["sds_normalizer_state_dict"], strict=True)
        self.transitions_scored = int(state["transitions_scored"])
        self.normalizer_samples_recorded = int(state["normalizer_samples_recorded"])
        self.normalizer_updates = int(state["normalizer_updates"])
        self._pending_normalizer_samples = 0
