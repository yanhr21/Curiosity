"""BCPPO instrumentation for the native whole-hand tactile experiment."""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch

from sugar_rl.utils.rsl_rl_bcppo import BCPPO


class NativeTactileTrainingBCPPO(BCPPO):
    """Official SUGAR BCPPO with an optional one-update tactile signal report."""

    def __init__(
        self,
        policy,
        contact_balanced_distillation: bool = False,
        **kwargs,
    ):
        self.contact_balanced_distillation = bool(contact_balanced_distillation)
        self._contact_distill_supported_samples = 0
        self._contact_distill_total_samples = 0
        self._contact_distill_supported_batches = 0
        super().__init__(policy, **kwargs)

    def _reset_contact_distill_stats(self) -> None:
        self._contact_distill_supported_samples = 0
        self._contact_distill_total_samples = 0
        self._contact_distill_supported_batches = 0

    def _reduce_distill_loss(self, per_sample_loss, obs_batch):
        if not self.contact_balanced_distillation:
            return super()._reduce_distill_loss(per_sample_loss, obs_batch)
        tactile = obs_batch[self.policy.tactile_obs_group]
        if tactile.ndim != 2 or tactile.shape[0] != per_sample_loss.shape[0]:
            raise RuntimeError(
                "contact-balanced distillation received mismatched tactile batch "
                f"{tuple(tactile.shape)} and loss {tuple(per_sample_loss.shape)}"
            )
        supported = tactile.ne(0).any(dim=-1)
        supported_count = int(torch.count_nonzero(supported).item())
        self._contact_distill_supported_samples += supported_count
        self._contact_distill_total_samples += int(supported.numel())
        if supported_count:
            self._contact_distill_supported_batches += 1
            return per_sample_loss[supported].mean()
        return per_sample_loss.mean()

    def _add_contact_distill_stats(self, losses: dict[str, float]) -> dict[str, float]:
        if not self.contact_balanced_distillation:
            return losses
        total = self._contact_distill_total_samples
        return {
            **losses,
            "contact_distill_supported_fraction": (
                float(self._contact_distill_supported_samples / total)
                if total
                else 0.0
            ),
            "contact_distill_supported_batches": float(
                self._contact_distill_supported_batches
            ),
        }

    @staticmethod
    def _append_trace(path: str, record: dict[str, object]) -> None:
        output_path = Path(path).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, sort_keys=True) + "\n")

    def update(self):
        self._reset_contact_distill_stats()
        output = os.environ.get("SUGAR_NATIVE_TACTILE_TRAINING_SIGNAL")
        contact_output = os.environ.get("SUGAR_NATIVE_TACTILE_CONTACT_SIGNAL")
        trace_output = os.environ.get("SUGAR_NATIVE_TACTILE_TRAINING_TRACE")
        if not output and not contact_output and not trace_output:
            return self._add_contact_distill_stats(super().update())

        tactile_group = self.policy.tactile_obs_group
        tactile = self.storage.observations[tactile_group]
        if tactile.ndim != 3 or tactile.shape[-1] != 324000:
            raise RuntimeError(
                "Native tactile rollout shape mismatch: expected "
                f"[steps,envs,324000], got {tuple(tactile.shape)}"
            )
        update_index = int(self.update_step)
        compact_stats = {
            "update": update_index,
            "nonzero_fraction": float(
                torch.count_nonzero(tactile).item() / tactile.numel()
            ),
            "fraction_abs_gt_0p01": float(
                torch.count_nonzero(tactile.abs() > 0.01).item()
                / tactile.numel()
            ),
            "frames_with_any_signal": int(
                torch.count_nonzero(tactile.ne(0).any(dim=-1)).item()
            ),
            "frame_count": int(tactile.shape[0] * tactile.shape[1]),
            "abs_max": float(tactile.abs().max().item()),
        }

        report_targets: list[tuple[str, str]] = []
        if output and self.update_step == 0:
            report_targets.append((output, "first PPO update"))
        if (
            contact_output
            and compact_stats["frames_with_any_signal"] > 0
            and not Path(contact_output).expanduser().resolve().exists()
        ):
            report_targets.append((contact_output, "first update with live tactile contact"))
        full_report = bool(report_targets)
        if not full_report:
            losses = self._add_contact_distill_stats(super().update())
            if trace_output:
                self._append_trace(
                    trace_output,
                    {
                        **compact_stats,
                        "losses": {
                            name: float(value) for name, value in losses.items()
                        },
                    },
                )
            return losses

        maps = tactile.reshape(*tactile.shape[:2], 2, 4, 27, 3, 20, 25)
        channel_names = ("normal", "signed_shear_x", "signed_shear_y")
        rollout_stats = {
            "shape": list(tactile.shape),
            "nonzero_values": int(torch.count_nonzero(tactile).item()),
            "total_values": int(tactile.numel()),
            "nonzero_fraction": float(
                torch.count_nonzero(tactile).item() / tactile.numel()
            ),
            "frames_with_any_signal": int(
                torch.count_nonzero(tactile.ne(0).any(dim=-1)).item()
            ),
            "frame_count": int(tactile.shape[0] * tactile.shape[1]),
            "abs_max": float(tactile.abs().max().item()),
            "channels": {},
            "absolute_thresholds": {},
        }
        for threshold in (1.0e-6, 1.0e-4, 1.0e-3, 1.0e-2):
            count = int(torch.count_nonzero(tactile.abs() > threshold).item())
            rollout_stats["absolute_thresholds"][str(threshold)] = {
                "count": count,
                "fraction": float(count / tactile.numel()),
            }
        for channel_index, channel_name in enumerate(channel_names):
            values = maps[..., channel_index, :, :]
            rollout_stats["channels"][channel_name] = {
                "nonzero_values": int(torch.count_nonzero(values).item()),
                "minimum": float(values.min().item()),
                "maximum": float(values.max().item()),
                "abs_max": float(values.abs().max().item()),
            }

        with torch.no_grad():
            encoded = self.policy.actor_tactile_encoder(
                tactile.reshape(-1, tactile.shape[-1])
            )
        encoded_stats = {
            "shape": list(encoded.shape),
            "nonzero_values": int(torch.count_nonzero(encoded).item()),
            "abs_max": float(encoded.abs().max().item()),
            "mean_abs": float(encoded.abs().mean().item()),
        }
        first_actor_layer = self.policy.actor[0]
        actor_base_width = int(self.policy.num_actor_base_obs)
        if not isinstance(first_actor_layer, torch.nn.Linear):
            raise RuntimeError("actor.0 is not the expected Linear layer")
        with torch.no_grad():
            raw_correction = torch.nn.functional.linear(
                encoded,
                first_actor_layer.weight[:, actor_base_width:],
                None,
            )
            cap = self.policy.tactile_preactivation_cap
            bounded_correction = (
                raw_correction
                if cap is None
                else cap * torch.tanh(raw_correction / cap)
            )
        correction_stats = {
            "mode": (
                "unbounded"
                if cap is None
                else "bounded_first_layer_preactivation"
            ),
            "cap": cap,
            "raw_abs_max": float(raw_correction.abs().max().item()),
            "raw_l2_max_per_sample": float(
                raw_correction.norm(dim=-1).max().item()
            ),
            "applied_abs_max": float(bounded_correction.abs().max().item()),
            "applied_l2_max_per_sample": float(
                bounded_correction.norm(dim=-1).max().item()
            ),
        }

        encoder_parameters = {
            name: parameter
            for name, parameter in self.policy.named_parameters()
            if name.startswith("actor_tactile_encoder.") and parameter.requires_grad
        }
        before = {
            name: parameter.detach().clone()
            for name, parameter in encoder_parameters.items()
        }
        gradient_stats = {
            name: {"calls": 0, "maximum_l2": 0.0, "maximum_abs": 0.0}
            for name in encoder_parameters
        }
        hooks = []
        for name, parameter in encoder_parameters.items():
            def record_gradient(gradient, *, parameter_name=name):
                stats = gradient_stats[parameter_name]
                stats["calls"] += 1
                stats["maximum_l2"] = max(
                    stats["maximum_l2"], float(gradient.norm().item())
                )
                stats["maximum_abs"] = max(
                    stats["maximum_abs"], float(gradient.abs().max().item())
                )
                return gradient

            hooks.append(parameter.register_hook(record_gradient))

        try:
            losses = self._add_contact_distill_stats(super().update())
        finally:
            for hook in hooks:
                hook.remove()

        parameter_stats = {}
        for name, parameter in encoder_parameters.items():
            delta = parameter.detach() - before[name]
            parameter_stats[name] = {
                "delta_l2": float(delta.norm().item()),
                "delta_abs_max": float(delta.abs().max().item()),
            }

        report = {
            "update": update_index,
            "triggers": [trigger for _, trigger in report_targets],
            "semantics": (
                "live rollout observation and actor tactile-encoder optimization; "
                "this proves signal entry, not policy benefit"
            ),
            "tactile_observation_group": tactile_group,
            "observation_shapes": {
                name: list(value.shape)
                for name, value in self.storage.observations.items()
            },
            "actor_contract": {
                "base_observation_width": int(self.policy.num_actor_base_obs),
                "encoded_tactile_width": int(
                    self.policy.actor_tactile_encoder.output_dim
                ),
                "raw_tactile_width": int(tactile.shape[-1]),
                "policy_groups": list(self.policy.obs_groups["policy"]),
                "critic_groups": list(self.policy.obs_groups["critic"]),
                "teacher_groups": list(self.policy.obs_groups["teacher"]),
            },
            "rollout": rollout_stats,
            "encoder_features_before_update": encoded_stats,
            "first_layer_tactile_correction_before_update": correction_stats,
            "encoder_gradients": gradient_stats,
            "encoder_parameter_change": parameter_stats,
            "losses": {name: float(value) for name, value in losses.items()},
        }
        for target, trigger in report_targets:
            output_path = Path(target).expanduser().resolve()
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(
                json.dumps(report, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            print(
                "[INFO] Native tactile training signal report "
                f"({trigger}): {output_path}"
            )
        if trace_output:
            self._append_trace(
                trace_output,
                {
                    **compact_stats,
                    "losses": {
                        name: float(value) for name, value in losses.items()
                    },
                },
            )
        return losses
