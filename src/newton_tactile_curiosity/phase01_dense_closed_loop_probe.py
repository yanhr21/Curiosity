#!/usr/bin/env python3
"""Newton-native dense tactile closed-loop curiosity probe/trainer.

This is a Phase01 Newton-native controller-training entry point. It is not
T-Rex, not an official tactile semantic claim, and not a success claim by
itself. The important contract is that dense tactile/mechanics observations
change later actions inside the rollout, and the training score includes an
online prediction-learning signal rather than offline sample reweighting.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import warp as wp

from newton_tactile_curiosity.phase00_mjw_direct_tactile_export import (
    accumulate_gaussian_per_point,
    center_of_pressure_proxy_from_map,
    frame_matrix,
    geom_pair_to_shapes,
    pad_side_for_pair,
    world_vector_to_body,
)
from newton_tactile_curiosity.phase00_mjw_force_audit import force_table_for_world, to_numpy
from newton_tactile_curiosity.phase00_sync_hydro_diagnostic import (
    SurfaceNullViewer,
    classify_shape,
    world_to_body,
)


FEATURE_NAMES = [
    "object_z",
    "object_dz",
    "total_fn",
    "total_ft",
    "left_fn",
    "right_fn",
    "fn_balance",
    "left_cop_y",
    "right_cop_y",
    "contact_area_cells",
    "shear_map_norm",
    "slip_proxy",
]

ACTION_NAMES = [
    "grip_close_delta",
    "lift_z_delta",
    "lateral_y_delta",
    "probe_y_delta",
]

PARAM_NAMES = [
    "fn_target",
    "grip_from_fn_deficit",
    "grip_from_slip",
    "lift_from_contact",
    "lift_from_slip_penalty",
    "balance_gain",
    "probe_amplitude",
    "probe_frequency",
]


FEATURE_ABLATION_MODES = [
    "none",
    "vision_only_proxy",
    "tactile_only_proxy",
    "noisy_tactile",
    "shuffled_lr_tactile",
]


@dataclass
class EpisodeResult:
    episode: int
    generation: int
    candidate: int
    scene: str
    override_mu: float | None
    override_kh: float | None
    feature_ablation: str
    params: np.ndarray
    score: float
    extrinsic_score: float
    intrinsic_score: float
    safety_cost: float
    max_lift: float
    final_lift: float
    hold_frames: int
    tail_hold_frames: int
    drop_after_lift: float
    max_total_fn: float
    max_total_ft: float
    action_change_l1: float
    closed_loop_action_changed: bool


class OnlineLinearPredictor:
    def __init__(self, in_dim: int, out_dim: int, lr: float):
        self.weights = np.zeros((in_dim + 1, out_dim), dtype=np.float32)
        self.lr = float(lr)

    def predict(self, x: np.ndarray) -> np.ndarray:
        xb = np.concatenate([x.astype(np.float32), np.ones(1, dtype=np.float32)])
        return xb @ self.weights

    def update(self, x: np.ndarray, target: np.ndarray) -> tuple[float, float]:
        xb = np.concatenate([x.astype(np.float32), np.ones(1, dtype=np.float32)])
        pred_before = np.nan_to_num(xb @ self.weights, nan=0.0, posinf=1.0e3, neginf=-1.0e3)
        target_safe = np.nan_to_num(target, nan=0.0, posinf=1.0e3, neginf=-1.0e3)
        diff_before = np.clip(pred_before - target_safe, -1.0e3, 1.0e3)
        err_before = float(np.mean(diff_before**2))
        grad = np.outer(xb, diff_before).astype(np.float32)
        grad = np.nan_to_num(grad, nan=0.0, posinf=1.0e3, neginf=-1.0e3)
        grad = np.clip(grad, -1.0e3, 1.0e3)
        self.weights -= self.lr * grad
        self.weights = np.nan_to_num(self.weights, nan=0.0, posinf=1.0e3, neginf=-1.0e3)
        self.weights = np.clip(self.weights, -1.0e3, 1.0e3)
        pred_after = np.nan_to_num(xb @ self.weights, nan=0.0, posinf=1.0e3, neginf=-1.0e3)
        diff_after = np.clip(pred_after - target_safe, -1.0e3, 1.0e3)
        err_after = float(np.mean(diff_after**2))
        return err_before, err_after


class DenseFeatureExtractor:
    def __init__(self, example: Any, map_size: int):
        self.map_size = int(map_size)
        labels = list(example.model.body_label)
        shape_body = example.model.shape_body.numpy()
        self.shape_classes = [classify_shape(i, shape_body, example.model) for i in range(example.model.shape_count)]
        self.left_body = next((i for i, label in enumerate(labels) if "leftfinger" in label.lower()), None)
        self.right_body = next((i for i, label in enumerate(labels) if "rightfinger" in label.lower()), None)
        self.object_body = int(example.object_body_local)
        self.last_object_z: float | None = None
        self.extent = (0.08, 0.08)
        self.center = np.zeros(2, dtype=np.float32)

    def read(self, example: Any) -> tuple[np.ndarray, dict[str, Any]]:
        body_q = example.state_0.body_q.numpy().astype(np.float32)
        object_z = float(body_q[self.object_body, 2])
        object_dz = 0.0 if self.last_object_z is None else object_z - self.last_object_z
        self.last_object_z = object_z

        shape = (self.map_size, self.map_size)
        left_fn_map = np.zeros(shape, dtype=np.float32)
        right_fn_map = np.zeros(shape, dtype=np.float32)
        left_ft_map = np.zeros(shape, dtype=np.float32)
        right_ft_map = np.zeros(shape, dtype=np.float32)
        left_area_map = np.zeros(shape, dtype=np.float32)
        right_area_map = np.zeros(shape, dtype=np.float32)
        left_shear_y = np.zeros(shape, dtype=np.float32)
        left_shear_z = np.zeros(shape, dtype=np.float32)
        right_shear_y = np.zeros(shape, dtype=np.float32)
        right_shear_z = np.zeros(shape, dtype=np.float32)

        left_fn = right_fn = left_ft = right_ft = 0.0
        sample_count = 0
        solver = example.solver
        mjw_data = solver.mjw_data
        contact = mjw_data.contact
        nacon = int(to_numpy(mjw_data.nacon).reshape(-1)[0])
        nacon = max(0, min(nacon, int(mjw_data.naconmax)))
        if nacon > 0:
            geom = to_numpy(contact.geom)[:nacon]
            pos = to_numpy(contact.pos)[:nacon].astype(np.float32)
            frames_mj = to_numpy(contact.frame)[:nacon]
            efc_address = to_numpy(contact.efc_address)[:nacon]
            worldid = to_numpy(contact.worldid)[:nacon].reshape(-1)
            efc_force = to_numpy(mjw_data.efc.force)
            geom_to_shape = to_numpy(solver.mjc_geom_to_newton_shape)

            for cidx in range(nacon):
                world = int(worldid[cidx]) if cidx < worldid.size else 0
                shape0, shape1 = geom_pair_to_shapes(geom[cidx], world, geom_to_shape)
                side_pair = pad_side_for_pair(self.shape_classes, shape0, shape1)
                if side_pair is None:
                    continue
                side, pad_is_shape0 = side_pair
                force_row = force_table_for_world(efc_force, world)
                addresses = [int(a) for a in np.asarray(efc_address[cidx]).reshape(-1) if 0 <= int(a) < force_row.size]
                if not addresses:
                    continue
                values = np.asarray([float(force_row[a]) for a in addresses], dtype=np.float32)
                fn = abs(float(values[0]))
                frame_mat = frame_matrix(frames_mj[cidx])
                tangent_world = np.zeros(3, dtype=np.float32)
                for value, basis in zip(values[1:], frame_mat[1 : 1 + max(0, len(values) - 1)], strict=False):
                    tangent_world += float(value) * basis.astype(np.float32)
                tangent_world *= -1.0 if pad_is_shape0 else 1.0
                ft = float(np.linalg.norm(tangent_world))

                body_idx = self.left_body if side == "left" else self.right_body
                if body_idx is None:
                    continue
                local_point = world_to_body(pos[cidx : cidx + 1], body_q[body_idx])[0]
                local_tangent = world_vector_to_body(tangent_world[None, :], body_q[body_idx])[0]
                local = local_point[None, :].astype(np.float32)
                fn_arr = np.asarray([fn], dtype=np.float32)
                ft_arr = np.asarray([ft], dtype=np.float32)
                tangent_yz = local_tangent[None, 1:3].astype(np.float32)
                zero_yz = np.zeros_like(tangent_yz)
                if side == "left":
                    accumulate_gaussian_per_point(left_fn_map, left_shear_y, left_shear_z, local, fn_arr, zero_yz, self.extent, self.center)
                    accumulate_gaussian_per_point(left_ft_map, left_shear_y, left_shear_z, local, ft_arr, tangent_yz, self.extent, self.center)
                    accumulate_gaussian_per_point(left_area_map, left_shear_y, left_shear_z, local, np.ones_like(fn_arr), zero_yz, self.extent, self.center)
                    left_fn += fn
                    left_ft += ft
                else:
                    accumulate_gaussian_per_point(right_fn_map, right_shear_y, right_shear_z, local, fn_arr, zero_yz, self.extent, self.center)
                    accumulate_gaussian_per_point(right_ft_map, right_shear_y, right_shear_z, local, ft_arr, tangent_yz, self.extent, self.center)
                    accumulate_gaussian_per_point(right_area_map, right_shear_y, right_shear_z, local, np.ones_like(fn_arr), zero_yz, self.extent, self.center)
                    right_fn += fn
                    right_ft += ft
                sample_count += 1

        total_fn = left_fn + right_fn
        total_ft = left_ft + right_ft
        denom = max(total_fn, 1.0e-6)
        fn_balance = (left_fn - right_fn) / denom
        left_cop = center_of_pressure_proxy_from_map(left_fn_map[None, :, :], self.extent, self.center)[0]
        right_cop = center_of_pressure_proxy_from_map(right_fn_map[None, :, :], self.extent, self.center)[0]
        left_cop_y = 0.0 if not np.isfinite(left_cop[0]) else float(left_cop[0])
        right_cop_y = 0.0 if not np.isfinite(right_cop[0]) else float(right_cop[0])
        area_cells = float((left_area_map > 0.0).sum() + (right_area_map > 0.0).sum())
        shear_norm = float(
            np.sqrt(left_shear_y * left_shear_y + left_shear_z * left_shear_z).sum()
            + np.sqrt(right_shear_y * right_shear_y + right_shear_z * right_shear_z).sum()
        )
        slip_proxy = float(total_ft / max(total_fn, 1.0e-6))

        features = np.asarray(
            [
                object_z,
                object_dz,
                min(total_fn / 80.0, 10.0),
                min(total_ft / 40.0, 10.0),
                min(left_fn / 40.0, 10.0),
                min(right_fn / 40.0, 10.0),
                fn_balance,
                left_cop_y,
                right_cop_y,
                area_cells / float(2 * self.map_size * self.map_size),
                min(shear_norm / 80.0, 10.0),
                min(slip_proxy, 10.0),
            ],
            dtype=np.float32,
        )
        meta = {
            "object_z": object_z,
            "object_dz": object_dz,
            "total_fn": total_fn,
            "total_ft": total_ft,
            "left_fn": left_fn,
            "right_fn": right_fn,
            "fn_balance": fn_balance,
            "contact_area_cells": area_cells,
            "shear_map_norm": shear_norm,
            "slip_proxy": slip_proxy,
            "sample_count": sample_count,
        }
        return features, meta


class ClosedLoopPolicy:
    def __init__(self, params: np.ndarray):
        self.params = params.astype(np.float32)
        self.action = np.zeros(len(ACTION_NAMES), dtype=np.float32)
        self.prev_action = np.zeros(len(ACTION_NAMES), dtype=np.float32)
        self.action_change_l1 = 0.0

    def update(self, frame: int, features: np.ndarray) -> np.ndarray:
        fn_target, grip_from_fn, grip_from_slip, lift_from_contact, lift_slip_penalty, balance_gain, probe_amp, probe_freq = [
            float(x) for x in self.params
        ]
        total_fn = float(features[2])
        slip_proxy = float(features[11])
        balance = float(features[6])
        contact_area = float(features[9])
        phase = 2.0 * math.pi * max(0.01, abs(probe_freq)) * frame / 60.0
        grip = np.clip(grip_from_fn * max(fn_target - total_fn, 0.0) + grip_from_slip * max(slip_proxy - 0.35, 0.0), -0.22, 0.28)
        lift = np.clip(lift_from_contact * contact_area - lift_slip_penalty * max(slip_proxy - 0.5, 0.0), -0.035, 0.045)
        lateral = np.clip(-balance_gain * balance, -0.025, 0.025)
        probe = np.clip(probe_amp * math.sin(phase), -0.018, 0.018)
        self.prev_action = self.action.copy()
        self.action = np.asarray([grip, lift, lateral, probe], dtype=np.float32)
        self.action_change_l1 += float(np.abs(self.action - self.prev_action).sum())
        return self.action


def apply_feature_ablation(args: argparse.Namespace, features: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    mode = str(getattr(args, "feature_ablation", "none"))
    transformed = features.copy()
    if mode == "none":
        return transformed
    if mode == "vision_only_proxy":
        transformed[2:] = 0.0
        return transformed
    if mode == "tactile_only_proxy":
        transformed[0:2] = 0.0
        return transformed
    if mode == "noisy_tactile":
        std = float(getattr(args, "feature_noise_std", 0.15))
        transformed[2:] = transformed[2:] + rng.normal(0.0, std, size=transformed[2:].shape).astype(np.float32)
        transformed[2:] = np.nan_to_num(transformed[2:], nan=0.0, posinf=10.0, neginf=-10.0)
        transformed[2:6] = np.clip(transformed[2:6], 0.0, 10.0)
        transformed[9:12] = np.clip(transformed[9:12], 0.0, 10.0)
        transformed[6:9] = np.clip(transformed[6:9], -10.0, 10.0)
        return transformed
    if mode == "shuffled_lr_tactile":
        transformed[4], transformed[5] = transformed[5], transformed[4]
        transformed[6] = -transformed[6]
        transformed[7], transformed[8] = transformed[8], transformed[7]
        return transformed
    raise ValueError(f"unsupported feature_ablation: {mode}")


def run_episode(args: argparse.Namespace, params: np.ndarray, predictor: OnlineLinearPredictor, episode: int, generation: int, candidate: int) -> EpisodeResult:
    from newton.examples.robot.example_robot_panda_hydro import Example
    import newton

    wp.set_device(args.device)
    viewer = SurfaceNullViewer(num_frames=args.num_frames)
    example = Example(viewer, SimpleNamespace(scene=args.scene, test=True, world_count=1))
    if args.override_mu is not None:
        example.model.shape_material_mu.fill_(float(args.override_mu))
    if args.override_kh is not None:
        example.model.shape_material_kh.fill_(float(args.override_kh))
    if args.override_mu is not None or args.override_kh is not None:
        example.solver.notify_model_changed(newton.ModelFlags.SHAPE_PROPERTIES)
        wp.synchronize()

    base_set_joint_targets = example.set_joint_targets
    policy = ClosedLoopPolicy(params)
    latest_action = np.zeros(len(ACTION_NAMES), dtype=np.float32)
    feature_rng = np.random.default_rng(int(getattr(args, "seed", 0)) + episode * 1009 + generation * 9176 + candidate * 37)

    def shifted_vec3(vec: Any, dx: float, dy: float, dz: float) -> Any:
        return wp.vec3(float(vec[0]) + dx, float(vec[1]) + dy, float(vec[2]) + dz)

    def closed_loop_set_joint_targets() -> None:
        cartesian_action = latest_action[1:4].copy()
        apply_cartesian = bool(np.abs(cartesian_action).sum() > 0.0)
        original_positions: list[Any] = []
        if apply_cartesian:
            dz = float(cartesian_action[0])
            dy = float(cartesian_action[1] + cartesian_action[2])
            original_positions = [waypoint[0] for waypoint in example.waypoints]
            for waypoint, original in zip(example.waypoints, original_positions, strict=False):
                waypoint[0] = shifted_vec3(original, 0.0, dy, dz)
        try:
            base_set_joint_targets()
        finally:
            if apply_cartesian:
                for waypoint, original in zip(example.waypoints, original_positions, strict=False):
                    waypoint[0] = original
        if float(np.abs(latest_action).sum()) <= 0.0:
            return
        targets = example.control.joint_target_q.numpy().reshape((example.world_count, -1)).astype(np.float32)
        grip_progress = np.clip(1.0 - targets[:, 7] / 0.06 + float(latest_action[0]), 0.0, 1.0)
        targets[:, 7] = 0.06 * (1.0 - grip_progress)
        targets[:, 8] = 0.06 * (1.0 - grip_progress)
        wp.copy(example.control.joint_target_q, wp.array(targets.reshape(-1), dtype=wp.float32))

    example.set_joint_targets = closed_loop_set_joint_targets
    extractor = DenseFeatureExtractor(example, args.map_size)

    features_prev: np.ndarray | None = None
    state_action_prev: np.ndarray | None = None
    initial_z: float | None = None
    max_z = -1.0e9
    max_total_fn = 0.0
    max_total_ft = 0.0
    intrinsic_score = 0.0
    safety_cost = 0.0
    hold_frames = 0
    z_series: list[float] = []

    for frame in range(args.num_frames):
        example.step()
        wp.synchronize()
        raw_features, meta = extractor.read(example)
        features = apply_feature_ablation(args, raw_features, feature_rng)
        if initial_z is None:
            initial_z = float(raw_features[0])
        max_z = max(max_z, float(raw_features[0]))
        z_series.append(float(raw_features[0]))
        max_total_fn = max(max_total_fn, float(meta["total_fn"]))
        max_total_ft = max(max_total_ft, float(meta["total_ft"]))
        if float(raw_features[0]) - initial_z > args.hold_lift_threshold:
            hold_frames += 1
        safety_cost += max(float(raw_features[11]) - args.max_safe_slip_proxy, 0.0)
        safety_cost += max(float(meta["total_fn"]) - args.max_safe_fn, 0.0) / max(args.max_safe_fn, 1.0)
        if state_action_prev is not None and features_prev is not None:
            err_before, err_after = predictor.update(state_action_prev, features)
            intrinsic_score += max(err_before - err_after, 0.0)
        action = policy.update(frame, features)
        latest_action = action
        features_prev = features
        state_action_prev = np.concatenate([features_prev, latest_action]).astype(np.float32)

    viewer.close()
    initial = z_series[0] if z_series else 0.0
    final_lift = (z_series[-1] - initial) if z_series else 0.0
    max_lift = max_z - initial
    stable_tail_frames = max(0, int(getattr(args, "stable_tail_frames", 60)))
    tail_window = z_series[-stable_tail_frames:] if stable_tail_frames > 0 else []
    tail_hold_frames = sum(1 for z in tail_window if float(z) - initial > args.hold_lift_threshold)
    drop_after_lift = max(0.0, max_z - (z_series[-1] if z_series else max_z))
    extrinsic_score = (
        args.score_lift_weight * max_lift
        + args.score_final_lift_weight * final_lift
        + args.score_hold_weight * hold_frames
        + args.score_tail_hold_weight * tail_hold_frames
        - args.score_drop_weight * drop_after_lift
    )
    score = extrinsic_score + args.intrinsic_weight * intrinsic_score - args.safety_weight * safety_cost
    if not np.isfinite(score):
        score = -1.0e9
    result = EpisodeResult(
        episode=episode,
        generation=generation,
        candidate=candidate,
        scene=str(args.scene),
        override_mu=None if args.override_mu is None else float(args.override_mu),
        override_kh=None if args.override_kh is None else float(args.override_kh),
        feature_ablation=str(getattr(args, "feature_ablation", "none")),
        params=params.copy(),
        score=float(score),
        extrinsic_score=float(extrinsic_score),
        intrinsic_score=float(intrinsic_score),
        safety_cost=float(safety_cost),
        max_lift=float(max_lift),
        final_lift=float(final_lift),
        hold_frames=int(hold_frames),
        tail_hold_frames=int(tail_hold_frames),
        drop_after_lift=float(drop_after_lift),
        max_total_fn=float(max_total_fn),
        max_total_ft=float(max_total_ft),
        action_change_l1=float(policy.action_change_l1),
        closed_loop_action_changed=bool(policy.action_change_l1 > 1.0e-6),
    )
    del extractor
    del example
    del viewer
    gc.collect()
    wp.synchronize()
    return result


def result_to_row(result: EpisodeResult) -> dict[str, Any]:
    row: dict[str, Any] = {
        "episode": result.episode,
        "generation": result.generation,
        "candidate": result.candidate,
        "scene": result.scene,
        "override_mu": result.override_mu,
        "override_kh": result.override_kh,
        "feature_ablation": result.feature_ablation,
        "score": result.score,
        "extrinsic_score": result.extrinsic_score,
        "intrinsic_score": result.intrinsic_score,
        "safety_cost": result.safety_cost,
        "max_lift": result.max_lift,
        "final_lift": result.final_lift,
        "hold_frames": result.hold_frames,
        "tail_hold_frames": result.tail_hold_frames,
        "drop_after_lift": result.drop_after_lift,
        "max_total_fn": result.max_total_fn,
        "max_total_ft": result.max_total_ft,
        "action_change_l1": result.action_change_l1,
        "closed_loop_action_changed": result.closed_loop_action_changed,
    }
    for name, value in zip(PARAM_NAMES, result.params, strict=False):
        row[f"param.{name}"] = float(value)
    return row


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/public/home/yanhongru/Curiosity"))
    parser.add_argument("--run-tag", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--scene", choices=["cube", "pen"], default="cube")
    parser.add_argument("--num-frames", type=int, default=180)
    parser.add_argument("--map-size", type=int, default=16)
    parser.add_argument("--generations", type=int, default=2)
    parser.add_argument("--population-size", type=int, default=4)
    parser.add_argument("--elite-count", type=int, default=2)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--predictor-lr", type=float, default=0.025)
    parser.add_argument("--intrinsic-weight", type=float, default=1.0)
    parser.add_argument("--safety-weight", type=float, default=1.0)
    parser.add_argument("--score-lift-weight", type=float, default=4.0)
    parser.add_argument("--score-final-lift-weight", type=float, default=0.0)
    parser.add_argument("--score-hold-weight", type=float, default=0.01)
    parser.add_argument("--score-tail-hold-weight", type=float, default=0.0)
    parser.add_argument("--score-drop-weight", type=float, default=2.0)
    parser.add_argument("--hold-lift-threshold", type=float, default=0.08)
    parser.add_argument("--stable-tail-frames", type=int, default=60)
    parser.add_argument("--max-safe-slip-proxy", type=float, default=0.85)
    parser.add_argument("--max-safe-fn", type=float, default=180.0)
    parser.add_argument("--feature-ablation", choices=FEATURE_ABLATION_MODES, default="none")
    parser.add_argument("--feature-noise-std", type=float, default=0.15)
    parser.add_argument("--override-mu", type=float, default=0.3)
    parser.add_argument("--override-kh", type=float, default=1.0e12)
    parser.add_argument("--train-mu-values", nargs="*", type=float, default=None)
    parser.add_argument("--min-duration-s", type=float, default=0.0)
    parser.add_argument("--target-duration-s", type=float, default=0.0)
    parser.add_argument("--sigma-min-frac", type=float, default=0.15)
    parser.add_argument("--sigma-decay", type=float, default=0.95)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)

    started = time.perf_counter()
    wp.set_device(args.device)
    try:
        wp.set_mempool_release_threshold(args.device, 0)
    except Exception:
        pass
    rng = np.random.default_rng(args.seed)
    mean = np.asarray([0.55, 0.18, 0.16, 0.035, 0.025, 0.018, 0.006, 1.0], dtype=np.float32)
    sigma = np.asarray([0.12, 0.06, 0.05, 0.015, 0.012, 0.008, 0.003, 0.35], dtype=np.float32)
    initial_sigma = sigma.copy()
    predictor = OnlineLinearPredictor(len(FEATURE_NAMES) + len(ACTION_NAMES), len(FEATURE_NAMES), args.predictor_lr)
    results: list[EpisodeResult] = []
    episode = 0
    train_mu_values = [float(mu) for mu in (args.train_mu_values or [])]
    if not train_mu_values:
        train_mu_values = [float(args.override_mu)] if args.override_mu is not None else [None]
    best_aggregate_score = -1.0e30
    best_aggregate_params = mean.copy()

    generation = 0
    while generation < args.generations or (
        args.target_duration_s > 0.0 and (time.perf_counter() - started) < args.target_duration_s
    ):
        gen_candidates: list[tuple[float, np.ndarray, EpisodeResult]] = []
        for candidate in range(args.population_size):
            params = mean + sigma * rng.standard_normal(len(PARAM_NAMES)).astype(np.float32)
            params[0] = float(np.clip(params[0], 0.05, 2.5))
            params[6] = float(np.clip(params[6], 0.0, 0.018))
            params[7] = float(np.clip(params[7], 0.1, 3.0))
            candidate_results: list[EpisodeResult] = []
            for train_mu in train_mu_values:
                run_args = SimpleNamespace(**vars(args))
                run_args.override_mu = train_mu
                result = run_episode(run_args, params, predictor, episode, generation, candidate)
                results.append(result)
                candidate_results.append(result)
                episode += 1
            finite_scores = [float(item.score) if np.isfinite(item.score) else -1.0e9 for item in candidate_results]
            aggregate_score = float(sum(finite_scores) / max(1, len(finite_scores)))
            best_candidate_result = max(candidate_results, key=lambda item: item.score)
            gen_candidates.append((aggregate_score, params, best_candidate_result))
            if aggregate_score > best_aggregate_score:
                best_aggregate_score = aggregate_score
                best_aggregate_params = params.copy()
        elites = sorted(gen_candidates, key=lambda item: item[0], reverse=True)[: max(1, args.elite_count)]
        elite_params = np.stack([item[1] for item in elites], axis=0)
        mean = elite_params.mean(axis=0).astype(np.float32)
        if elite_params.shape[0] > 1:
            sigma_candidate = elite_params.std(axis=0).astype(np.float32)
        else:
            sigma_candidate = (sigma * float(args.sigma_decay)).astype(np.float32)
        sigma_floor = initial_sigma * float(args.sigma_min_frac)
        sigma = np.maximum(sigma_candidate, sigma_floor)
        print(
            json.dumps(
                {
                    "event": "generation_complete",
                    "generation": int(generation),
                    "episode_count": int(episode),
                    "best_generation_aggregate_score": float(elites[0][0]),
                    "best_generation_lift": float(elites[0][2].max_lift),
                    "best_generation_final_lift": float(elites[0][2].final_lift),
                    "best_generation_hold_frames": int(elites[0][2].hold_frames),
                    "best_generation_tail_hold_frames": int(elites[0][2].tail_hold_frames),
                    "elapsed_s": float(time.perf_counter() - started),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        generation += 1

    elapsed_s = float(time.perf_counter() - started)
    best = max(results, key=lambda item: item.score)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.report_dir.mkdir(parents=True, exist_ok=True)
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    rows = [result_to_row(item) for item in results]
    csv_path = args.output_dir / "episode_metrics.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    checkpoint_path = args.checkpoint_dir / "dense_closed_loop_probe_checkpoint.npz"
    np.savez_compressed(
        checkpoint_path,
        policy_mean=mean,
        policy_sigma=sigma,
        best_params=best_aggregate_params,
        best_single_episode_params=best.params,
        best_aggregate_params=best_aggregate_params,
        predictor_weights=predictor.weights,
        param_names=np.asarray(PARAM_NAMES),
        feature_names=np.asarray(FEATURE_NAMES),
        action_names=np.asarray(ACTION_NAMES),
    )

    real_training_attempt = (not args.smoke) and elapsed_s >= 3600.0 and args.min_duration_s >= 3600.0
    status = "pass_smoke_closed_loop_dense_probe" if args.smoke else "complete_dense_closed_loop_training_attempt"
    if not any(item.closed_loop_action_changed for item in results):
        status = "fail_no_closed_loop_action_change"
    if args.min_duration_s > 0.0 and elapsed_s < args.min_duration_s:
        status = "fail_min_duration_not_met"

    summary = {
        "classification": "phase01_dense_closed_loop_curiosity_probe_v1",
        "run_tag": args.run_tag,
        "status": status,
        "smoke": bool(args.smoke),
        "real_training_attempt": bool(real_training_attempt),
        "not_curiosity_success": True,
        "not_trex": True,
        "not_official_tactile_semantic_validation": True,
        "closed_loop_action_changed_any": bool(any(item.closed_loop_action_changed for item in results)),
        "intrinsic_reward_affects_policy_selection": True,
        "sample_reweighting_only": False,
        "dense_feature_names": FEATURE_NAMES,
        "action_names": ACTION_NAMES,
        "param_names": PARAM_NAMES,
        "feature_ablation": args.feature_ablation,
        "feature_noise_std": float(args.feature_noise_std),
        "train_mu_values": [None if mu is None else float(mu) for mu in train_mu_values],
        "episode_count": len(results),
        "configured_generations": int(args.generations),
        "completed_generations": int(generation),
        "population_size": int(args.population_size),
        "best_score": float(best.score),
        "best_aggregate_score": float(best_aggregate_score),
        "best_max_lift": float(best.max_lift),
        "best_final_lift": float(best.final_lift),
        "best_hold_frames": int(best.hold_frames),
        "best_tail_hold_frames": int(best.tail_hold_frames),
        "best_intrinsic_score": float(best.intrinsic_score),
        "best_safety_cost": float(best.safety_cost),
        "best_action_change_l1": float(best.action_change_l1),
        "score_lift_weight": float(args.score_lift_weight),
        "score_final_lift_weight": float(args.score_final_lift_weight),
        "score_hold_weight": float(args.score_hold_weight),
        "score_tail_hold_weight": float(args.score_tail_hold_weight),
        "score_drop_weight": float(args.score_drop_weight),
        "stable_tail_frames": int(args.stable_tail_frames),
        "sigma_min_frac": float(args.sigma_min_frac),
        "sigma_decay": float(args.sigma_decay),
        "elapsed_s": elapsed_s,
        "metrics_csv": str(csv_path),
        "checkpoint": str(checkpoint_path),
        "next_required_step": "run non-smoke one-hour attempt and evaluate against strongest baseline" if args.smoke else "held-out strongest-baseline evaluation with safety metrics and videos",
    }
    summary_path = args.output_dir / "dense_closed_loop_probe_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    report_path = args.report_dir / "dense_closed_loop_probe.md"
    report_path.write_text(
        "# Phase01 Dense Closed-Loop Probe\n\n"
        f"- run_tag: `{args.run_tag}`\n"
        f"- status: `{status}`\n"
        f"- smoke: `{summary['smoke']}`\n"
        f"- real_training_attempt: `{summary['real_training_attempt']}`\n"
        f"- closed_loop_action_changed_any: `{summary['closed_loop_action_changed_any']}`\n"
        f"- best_score: `{summary['best_score']}`\n"
        f"- best_max_lift: `{summary['best_max_lift']}`\n"
        f"- best_hold_frames: `{summary['best_hold_frames']}`\n"
        f"- best_intrinsic_score: `{summary['best_intrinsic_score']}`\n"
        f"- checkpoint: `{checkpoint_path}`\n\n"
        "This is Newton-native dense closed-loop controller probing/training. It is not T-Rex, not final tactile semantic validation, and not curiosity success.\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if status.startswith("pass_") or status.startswith("complete_") else 1


if __name__ == "__main__":
    raise SystemExit(main())
