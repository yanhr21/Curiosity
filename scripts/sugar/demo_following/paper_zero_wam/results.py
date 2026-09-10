"""Maintain an experiment-local result page without recreating archived plans."""

from __future__ import annotations

import fcntl
import json
import os
from pathlib import Path
from typing import Any

from .artifacts import emit_json_best_effort, write_text_atomic
from .config import PROJECT_ROOT, PaperZeroWAMConfig


RESULT_PATHS = {
    "data": "latents/LATENT_RESULT.json",
    "schedule": "schedule/SCHEDULE_RESULT.json",
    "trace_contract": "schedule/TRAINING_TRACE_CONTRACT_AUDIT.json",
    "heldout_contract": "schedule/HELDOUT_DECISION_CONTRACT_AUDIT.json",
    "overfit": "overfit/OVERFIT_RESULT.json",
    "formal": "formal/FORMAL_TRAINING_RESULT.json",
    "heldout": "formal/heldout_prompt_gate/HELDOUT_PROMPT_RESULT.json",
    "physical_admission": "formal/PHYSICAL_ADMISSION_RESULT.json",
    "render": "formal/heldout_openloop_videos/RENDER_RESULT.json",
    "smallbox": "formal/smallbox_physical/SMALLBOX_RESULT.json",
    "motion_admission": "formal/MOTION_DISJOINT_ADMISSION_RESULT.json",
    "motion_disjoint": "formal/motion_disjoint_physical/MOTION_DISJOINT_RESULT.json",
}
FORMAL_PROGRESS_PATH = "formal/FORMAL_PROGRESS.json"


def _read(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"terminal result is not a JSON object: {path}")
    return value


def _state(value: dict[str, Any] | None) -> str:
    if value is None:
        return "pending"
    if value.get("admitted") is True:
        return "admitted"
    if value.get("admitted") is False:
        return "closed (not admitted)"
    if value.get("passed") is True:
        return "passed"
    if value.get("passed") is False:
        return "closed negative"
    if value.get("execution_completed") is True:
        return "completed"
    return "materialized"


def _compact(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _artifact_link(path_value: Any, target: Path) -> str:
    """Return a repository-relative Markdown link for a real rollout artifact."""

    path = Path(str(path_value))
    resolved = path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()
    try:
        resolved.relative_to(PROJECT_ROOT)
        page_relative = Path(os.path.relpath(resolved, start=target.parent))
        return f"[{path.name}]({page_relative.as_posix()})"
    except ValueError:
        # Custom test outputs can live outside the repository.  Keep their
        # exact path visible without pretending the project page can serve it.
        return f"`{path}`"


def _refresh_results_document_unlocked(
    experiment_root: Path | None = None, output: Path | None = None
) -> Path:
    config = PaperZeroWAMConfig()
    config.validate()
    root = (experiment_root or config.resolved(config.output_root)).resolve()
    target = (
        output
        or root / "results.md"
    ).resolve()
    values = {name: _read(root / relative) for name, relative in RESULT_PATHS.items()}
    formal_progress = _read(root / FORMAL_PROGRESS_PATH)
    runtime_config = _read(root / "overfit/CONFIG.json") or {}
    runtime_execution = runtime_config.get("execution")
    rows = [
        "| stage | state | terminal evidence |",
        "|---|---:|---|",
    ]
    for name, relative in RESULT_PATHS.items():
        link = Path(os.path.relpath(root / relative, start=target.parent)).as_posix()
        rows.append(f"| {name} | {_state(values[name])} | [{Path(relative).name}]({link}) |")

    details: list[str] = []
    data = values["data"]
    if data is not None:
        details.extend(
            [
                "## Data materialization",
                "",
                f"- trajectories: `{data.get('trajectory_count')}`; train action rows: `{data.get('action_training_rows')}`",
                f"- split counts: `{_compact(data.get('counts'))}`",
                f"- source causal clock (64 prompt RGB / 141 robot RGB): `{data.get('manifest_clock_exact')}`",
                f"- unique prompt / robot frame paths / overlap: `{data.get('prompt_frame_path_count')}` / `{data.get('robot_frame_path_count')}` / `{data.get('prompt_robot_frame_path_overlap_count')}`",
                f"- exact PhysX action trace/environment geometry: `{data.get('action_trace_geometry_exact')}`",
                f"- PhysX trace files / selected environments: `{data.get('action_trace_file_count')}` / `{data.get('action_trace_environment_pair_count')}`; shapes: `{_compact(data.get('action_trace_shape_environment_counts'))}`",
                f"- prompt/robot latent shapes: `{data.get('prompt_shape')}` / `{data.get('robot_shape')}`",
                "",
            ]
        )
    schedule = values["schedule"]
    if schedule is not None:
        totals = schedule.get("totals") or {}
        overfit_schedule = schedule.get("overfit") or {}
        overfit_per_update = overfit_schedule.get("per_update") or {}
        overfit_exposure_summary = {
            key: overfit_per_update.get(key)
            for key in (
                "latent_transition_exposures",
                "rgb_interval_exposures",
                "action_exposures",
                "two_pass_main_transformer_tokens",
                "supervised_transformer_token_exposures",
                "video_loss_elements",
                "action_loss_elements",
                "ifp_loss_elements",
            )
        }
        rank_loss_evidence = overfit_per_update.get("loss_rank_evidence") or []
        unit_rank_scales = sum(
            row.get("scales")
            == {"video": 1.0, "action": 1.0, "ifp": [1.0] * 4}
            for row in rank_loss_evidence
        )
        details.extend(
            [
                "## Formal data and optimizer budget",
                "",
                f"- complete epochs / optimizer updates / packed samples: `{totals.get('complete_epochs')}` / `{totals.get('optimizer_steps')}` / `{totals.get('packed_samples')}`",
                f"- latent transitions / RGB intervals / action targets: `{totals.get('latent_transition_exposures')}` / `{totals.get('rgb_interval_exposures')}` / `{totals.get('action_exposures')}`",
                f"- immutable logical schedule batch: `{_compact(schedule.get('batch'))}`",
                f"- actual execution (when configured): `{_compact(runtime_execution)}`",
                f"- prompt enabled/dropped samples: `{totals.get('prompt_enabled_samples')}` / `{totals.get('prompt_dropped_samples')}`",
                f"- frozen overfit source/phase cases: `{len(overfit_schedule.get('cases', []))}`; repeated updates: `{overfit_schedule.get('optimizer_steps')}`",
                f"- overfit per-update geometry: `{_compact(overfit_exposure_summary)}`",
                f"- overfit loss reduction / zero-target IFP heads / unit-scaled ranks: `{overfit_per_update.get('loss_reduction')}` / `{_compact(overfit_per_update.get('ifp_zero_target_heads'))}` / `{unit_rank_scales}/{len(rank_loss_evidence)}`",
                "",
            ]
        )
    trace_contract = values["trace_contract"]
    if trace_contract is not None:
        clean_checks = trace_contract.get("clean_checks") or {}
        overfit_checks = trace_contract.get("clean_overfit_execution_checks") or {}
        mutation_rejections = trace_contract.get("mutation_rejections") or {}
        details.extend(
            [
                "## Training trace contract",
                "",
                f"- result: `{_state(trace_contract)}`; schedule / optimizer rows: `{trace_contract.get('schedule_rows')}` / `{trace_contract.get('optimizer_trace_rows')}`",
                f"- formal / overfit execution checks passed: `{sum(value is True for value in clean_checks.values())}/{len(clean_checks)}` / `{sum(value is True for value in overfit_checks.values())}/{len(overfit_checks)}`",
                f"- injected contract violations rejected: `{sum(value is True for value in mutation_rejections.values())}/{len(mutation_rejections)}`",
                f"- model constructed / optimizer updates / hash checks: `{trace_contract.get('model_constructed')}` / `{trace_contract.get('optimizer_updates')}` / `{trace_contract.get('hash_checks')}`",
                "",
            ]
        )
    heldout_contract = values["heldout_contract"]
    if heldout_contract is not None:
        mutation_rejections = heldout_contract.get("mutation_rejections") or {}
        details.extend(
            [
                "## Held-out reducer contract",
                "",
                f"- result: `{_state(heldout_contract)}`; groups/scores: `{heldout_contract.get('matched_noise_groups')}` / `{heldout_contract.get('score_instances')}`",
                f"- clean decision/checks: `{heldout_contract.get('clean_decision_passed')}` / `{_compact(heldout_contract.get('clean_checks'))}`",
                f"- injected violations rejected: `{sum(value is True for value in mutation_rejections.values())}/{len(mutation_rejections)}`; details: `{_compact(mutation_rejections)}`",
                f"- model constructed / optimizer updates / hash checks: `{heldout_contract.get('model_constructed')}` / `{heldout_contract.get('optimizer_updates')}` / `{heldout_contract.get('hash_checks')}`",
                "",
            ]
        )
    overfit = values["overfit"]
    if overfit is not None:
        prompt_gate = overfit.get("prompt_gate") or {}
        prompt_gate_summary = {
            key: prompt_gate.get(key)
            for key in (
                "passed",
                "interventions",
                "condition_count",
                "loss_forward_count_per_rank",
                "one_step_connectivity_forward_count_per_rank",
                "model_forward_count_per_rank",
                "loss_forward_count",
                "one_step_connectivity_forward_count",
                "execution",
                "optimizer_updates_added",
                "prompt_loss_passed",
                "teacher_action_isolated",
                "generated_future_to_action_path_active",
            )
            if key in prompt_gate
        }
        details.extend(
            [
                "## Full-width overfit",
                "",
                f"- scientific result: `{_state(overfit)}`; execution completed: `{overfit.get('execution_completed')}`",
                f"- optimizer updates: `{overfit.get('optimizer_steps')}`; parameters: `{overfit.get('architecture_parameter_count')}`",
                f"- checkpoint policy: `{overfit.get('checkpoint_policy')}`",
                f"- execution checks: `{_compact((overfit.get('execution_decision') or {}).get('checks'))}`",
                f"- last/first loss ratios: `{_compact(overfit.get('last_to_first_loss_ratios'))}`",
                "- single-GPU ratio basis: fixed-noise final/initial matched prompt evaluation, not stochastic training-row losses",
                f"- prompt gate: `{_compact(prompt_gate_summary)}`",
                "",
            ]
        )
    formal = values["formal"]
    if formal_progress is not None:
        progress_link = Path(
            os.path.relpath(root / FORMAL_PROGRESS_PATH, start=target.parent)
        ).as_posix()
        details.extend(
            [
                "## Formal training progress",
                "",
                f"- complete epochs / planned: `{formal_progress.get('complete_epochs')}` / `{formal_progress.get('epochs_planned')}`",
                f"- optimizer updates / planned: `{formal_progress.get('optimizer_steps_completed')}` / `{formal_progress.get('optimizer_steps_planned')}`",
                f"- cumulative exposures: `{_compact(formal_progress.get('cumulative_exposures'))}`",
                f"- latest losses: `{_compact(formal_progress.get('latest_losses'))}`",
                f"- evidence: [FORMAL_PROGRESS.json]({progress_link})",
                "",
            ]
        )
    if formal is not None:
        decision = formal.get("formal_decision") or {}
        details.extend(
            [
                "## Formal training",
                "",
                f"- scientific result: `{_state(formal)}`; execution completed: `{formal.get('execution_completed')}`",
                f"- optimizer updates: `{formal.get('optimizer_steps')}`; batch: `{_compact(formal.get('batch'))}`",
                f"- checkpoint policy: `{formal.get('checkpoint_policy')}`",
                f"- execution checks: `{_compact((formal.get('execution_decision') or {}).get('checks'))}`",
                f"- last/first loss ratios: `{_compact(formal.get('last_to_first_loss_ratios'))}`",
                f"- decision checks: `{_compact(decision.get('checks'))}`",
                "",
            ]
        )
    heldout = values["heldout"]
    if heldout is not None:
        decision = heldout.get("decision") or {}
        details.extend(
            [
                "## Held-out prompt dependence",
                "",
                f"- result: `{_state(heldout)}`; execution completed: `{heldout.get('execution_completed')}`; groups/scores: `{heldout.get('group_count')}` / `{heldout.get('score_instance_count')}`",
                f"- decision checks: `{_compact(decision.get('checks'))}`",
                "",
            ]
        )
    physical_admission = values["physical_admission"]
    if physical_admission is not None:
        details.extend(
            [
                "## Physical admission",
                "",
                f"- state: `{_state(physical_admission)}`; reason: `{physical_admission.get('reason')}`",
                "",
            ]
        )
    render = values["render"]
    if render is not None:
        videos = [
            str(row["video"])
            for row in render.get("cases", [])
            if isinstance(row, dict) and row.get("video")
        ]
        details.extend(
            [
                "## Predictive rollout visualization",
                "",
                f"- rendered cases: `{len(videos)}`; inference: `{_compact(render.get('inference'))}`",
                f"- execution checks: `{_compact(render.get('execution_checks'))}`",
                f"- single-anchor directional diagnostic: `{render.get('all_single_anchor_directional_checks_passed')}`",
                *[f"- {_artifact_link(path, target)}" for path in videos],
                "",
            ]
        )
    smallbox = values["smallbox"]
    if smallbox is not None:
        smallbox_videos = [str(path) for path in smallbox.get("videos", []) if path]
        smallbox_frame_counts = [
            int(value) for value in smallbox.get("visualization_frame_counts", [])
        ]
        details.extend(
            [
                "## SMALLBOX closed loop",
                "",
                f"- result: `{_state(smallbox)}`; rollouts/videos: `{smallbox.get('rollout_count')}` / `{len(smallbox_videos)}`",
                f"- adapted / released-endpoint rollouts: `{smallbox.get('adapted_rollout_count')}` / `{smallbox.get('released_endpoint_rollout_count')}`",
                f"- outcomes: `{_compact(smallbox.get('outcomes'))}`",
                f"- semantic prompt specs: `{_compact(smallbox.get('prompt_specs'))}`",
                f"- requested/executed action rows / maximum error: `{smallbox.get('requested_executed_action_rows')}` / `{smallbox.get('maximum_requested_executed_action_error')}`",
                f"- encoded rollout videos / unique frame counts: `{len(smallbox_frame_counts)}` / `{sorted(set(smallbox_frame_counts))}`",
                *[f"- {_artifact_link(path, target)}" for path in smallbox_videos],
                "",
            ]
        )
    motion_admission = values["motion_admission"]
    if motion_admission is not None:
        details.extend(
            [
                "## Motion-disjoint admission",
                "",
                f"- state: `{_state(motion_admission)}`; reason: `{motion_admission.get('reason')}`",
                "",
            ]
        )
    motion = values["motion_disjoint"]
    if motion is not None:
        motion_videos = [str(path) for path in motion.get("videos", []) if path]
        motion_frame_counts = [
            int(value) for value in motion.get("visualization_frame_counts", [])
        ]
        details.extend(
            [
                "## Motion-disjoint closed loop",
                "",
                f"- result: `{_state(motion)}`; rollouts/videos: `{motion.get('rollout_count')}` / `{motion.get('visualization_count')}`",
                f"- adapted / released-endpoint rollouts: `{motion.get('adapted_rollout_count')}` / `{motion.get('released_endpoint_rollout_count')}`",
                f"- task summary: `{_compact(motion.get('task_summary'))}`",
                f"- source-bound semantic prompt spec groups: `{len(motion.get('semantic_prompt_specs', []))}`",
                f"- requested/executed action rows / maximum error: `{motion.get('requested_executed_action_rows')}` / `{motion.get('maximum_requested_executed_action_error')}`",
                f"- encoded rollout videos / unique frame counts: `{len(motion_frame_counts)}` / `{sorted(set(motion_frame_counts))}`",
                *[f"- {_artifact_link(path, target)}" for path in motion_videos],
                "",
            ]
        )

    text = "\n".join(
        [
            "# Plan 17 paper Zero-WAM results",
            "",
            "> Auto-refreshed from terminal experiment artifacts. Scientific pass/fail is preserved; missing stages remain pending. This path performs no file-digest checks.",
            "",
            "## Frozen execution protocol",
            "",
            f"- architecture parameters: `{config.expected_parameter_count}`",
            f"- formal batch / steps / epochs: `{config.world_size}` / `{config.optimizer_steps}` / `{config.epochs}`",
            f"- inference: chunk `{config.inference_chunk_size}`, video/action steps `{config.video_inference_steps}/{config.action_inference_steps}`, CFG `{config.video_guidance_scale}/{config.action_guidance_scale}`, SNR shift `{config.video_snr_shift}/{config.action_snr_shift}`",
            "- frozen physical evidence: SMALLBOX `40 adapted + 40 endpoint = 80`; motion-disjoint `760 adapted + 380 endpoint = 1,140`",
            "- physical prompt identity is explicit at split/task/source/latent-key/orientation level; rollout videos contain exactly `131` submitted frames",
            "- paired endpoint starts restore both internal action-history slots in addition to physics/root/joint/RGB state",
            "- physical-only distributed timeout: `12 h` for serial rank-zero endpoint baselines inside each FSDP batch",
            "- train/held-out/render distributed timeout: `2 h` for large-shard I/O and epoch checkpoint skew",
            "- released Tracker start: all five `history_length=5` terms reinitialized from the restored live state",
            "- final reducers reopen and directly compare all `15` adapted/endpoint restore-payload arrays",
            "- held-out evidence requires the unique ordered `0..389` group partition and ten frozen anchors per motion",
            "- formal/held-out/render/physical latent consumers share one digest-free identity/shape/finiteness loader",
            "- training ranks reuse the completed 199-selector admission and validate consumed selectors from an eight-shard CPU cache",
            "- per-rank runtime milestones locate full-model/FSDP/AdamW/first-update/checkpoint failures but are never admission evidence",
            "- missing-result guardians reconstruct overfit/formal/held-out decisions from complete trace/score logs before accepting a terminal",
            "- missing output alone cannot trigger recovery: the exact job/array element must have a scheduler-confirmed interruption; numerical failures forbid replay",
            "- held-out CPU caches preserve 1,950 forwards; matched-noise seeds reset after every condition load path",
            "- predictive cases bind test target 9/anchor 8 to explicit prompt task/source/orientation metadata",
            "- each predictive H.264 is capped to and evidenced by exactly eight frames `000..007`",
            "- terminal/progress JSON is authoritative; Markdown refresh failures warn but never change stage exit status",
            "- console progress/terminal JSON is best-effort and cannot let a broken Slurm log invalidate committed evidence",
            "",
            *rows,
            "",
            *details,
        ]
    ).rstrip() + "\n"
    write_text_atomic(target, text)
    return target


def refresh_results_document(
    experiment_root: Path | None = None, output: Path | None = None
) -> Path:
    """Serialize refreshes so concurrent terminal stages cannot lose a result."""

    config = PaperZeroWAMConfig()
    root = (experiment_root or config.resolved(config.output_root)).resolve()
    target = (
        output
        or root / "results.md"
    ).resolve()
    lock = root / ".results.lock"
    lock.parent.mkdir(parents=True, exist_ok=True)
    with lock.open("w", encoding="utf-8") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        return _refresh_results_document_unlocked(root, target)


def refresh_results_document_best_effort(
    experiment_root: Path | None = None, output: Path | None = None
) -> Path | None:
    """Refresh observational Markdown without changing a scientific stage exit status."""

    try:
        return refresh_results_document(experiment_root, output)
    except Exception as error:
        emit_json_best_effort(
            {
                "warning": "results_document_refresh_failed",
                "error_type": type(error).__name__,
                "error": str(error),
                "scientific_stage_status_unchanged": True,
            },
            error=True,
        )
        return None
