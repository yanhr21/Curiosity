from __future__ import annotations

from pathlib import Path
import ast
import re

import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import kendalltau


ROOT = Path(__file__).resolve().parents[1]
RENDERER = ROOT / "scripts/sugar/demo_following/render_xirl_reference_corpus.py"
CANARY = ROOT / "scripts/sugar/demo_following/run_xskill_sphere_canaries_then_hold.sh"
FULL = ROOT / "scripts/sugar/demo_following/run_xskill_sphere_full_corpus_then_hold.sh"
CROSS_RUNNER = (
    ROOT
    / "scripts/sugar/demo_following/run_official_xskill_cross_embodiment_audit_then_hold.sh"
)
ADAPTER = ROOT / "scripts/sugar/demo_following/train_evaluate_official_xskill_prototypes.py"
OFFICIAL_CHAIN = (
    ROOT
    / "experiments/runtime_assets/official_xskill_b748071/xskill/env/kitchen/"
    "relay_policy_learning/third_party/franka/assets/chain2.xml"
)
OFFICIAL_ASSETS = OFFICIAL_CHAIN.with_name("assets_sphere.xml")


def test_released_xskill_sphere_intervention_is_end_effector_only() -> None:
    chain = OFFICIAL_CHAIN.read_text(encoding="utf-8")
    assets = OFFICIAL_ASSETS.read_text(encoding="utf-8")
    visible_spheres = re.findall(
        r'<geom\s+size="0\.05"\s+type="sphere"\s+rgba="\.95 \.99 \.92 1"\s*/>',
        chain,
    )
    assert len(visible_spheres) == 2
    assert 'class="panda_viz" mesh="link1_viz" rgba="0 0 0 0"' in chain
    assert '<default class="panda_viz">' in assets
    assert 'rgba="0 0 0 0" mass="0"' in assets


def test_g1_compatibility_renderer_uses_fixed_task_independent_end_effectors() -> None:
    text = RENDERER.read_text(encoding="utf-8")
    assert 'parser.add_argument("--embodiment", choices=("g1", "sphere"), default="g1")' in text
    assert "XSKILL_SPHERE_RADIUS_M = 0.05" in text
    for name in (
        "left_rubber_hand",
        "right_rubber_hand",
        "left_ankle_roll_link",
        "right_ankle_roll_link",
    ):
        assert f'"{name}"' in text
    assert "sim_utils.find_matching_prims(scene[\"robot\"].cfg.prim_path)" in text
    assert "prim_utils.set_prim_visibility(robot_prim, False)" in text
    assert "sphere_agent.visualize(translations=sphere_positions)" in text
    assert "task_independent_visible_body_set" in text


def test_sphere_runners_preserve_clean_64_frame_source_contract() -> None:
    canary = CANARY.read_text(encoding="utf-8")
    full = FULL.read_text(encoding="utf-8")
    for runner in (canary, full):
        assert "--embodiment sphere" in runner
        assert "--frames-per-motion 64" in runner
        assert "--enable_cameras --headless --device cuda:0" in runner
        assert "VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json" in runner
    assert "render_reference CarryBox 45" in canary
    assert "render_reference KickBox 21" in canary
    assert 'jq -e \'.passed == true and .embodiment == "sphere"' in canary
    assert "render_task CarryBox 99" in full
    assert "render_task KickBox 98" in full
    assert 'all(.frame_counts[]; . == 64)' in full


def test_cross_embodiment_adapter_uses_official_two_stream_training_step() -> None:
    text = ADAPTER.read_text(encoding="utf-8")
    assert 'parser.add_argument("--sphere-corpus", type=Path)' in text
    assert "model.training_step(move_batch(batch, device), batch_index)" in text
    assert "[(corpus, None), (sphere_corpus, None)]" in text
    assert "expected_length = 80 if sphere_corpus is None else 160" in text
    assert "official XSkill stream index {index} is not source-paired" in text
    assert '"elementwise_task_and_motion_id_match": True' in text
    assert '"official_stream_pairing": official_stream_pairing' in text
    assert "G1 and sphere source-ID inventories are not elementwise identical" in text
    assert "cross_temporal_metrics" in text
    assert "cross_reference_metrics" in text
    assert "g1_to_sphere" in text
    assert "sphere_to_g1" in text
    assert "test_prototype_temporal_mae_improves_5pct" in text
    assert "test_prototype_tau_improves_0p05" in text
    assert '"passed": bool(all(criteria.values()))' in text


def test_cross_runner_never_bypasses_representation_admission() -> None:
    runner = CROSS_RUNNER.read_text(encoding="utf-8")
    assert "b748071daeb031d6b42a8dcb88c38c52297e20af" in runner
    assert "--sphere-corpus" in runner
    assert "XSKILL_CROSS_EMBODIMENT_AUTOMATIC_DECISION" in runner
    assert "GPU_HOLD_AFTER_XSKILL_CROSS_EMBODIMENT_AUDIT_READY" in runner
    assert "approval" not in runner.lower()


def test_cross_temporal_metric_is_exact_for_identical_paired_progress() -> None:
    module = ast.parse(ADAPTER.read_text(encoding="utf-8"))
    function = next(
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name == "cross_temporal_metrics"
    )
    namespace = {
        "np": np,
        "cdist": cdist,
        "kendalltau": kendalltau,
        "TASKS": ("CarryBox", "KickBox"),
    }
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(ADAPTER), "exec"), namespace)
    sequence = np.stack((np.arange(8), np.zeros(8)), axis=1).astype(np.float64)
    embeddings = {
        "test": {
            task: {9: {"prototype": sequence.copy()}}
            for task in namespace["TASKS"]
        }
    }
    result = namespace["cross_temporal_metrics"](
        embeddings,
        embeddings,
        "test",
        "prototype",
    )
    for task in namespace["TASKS"]:
        assert result[task]["normalized_temporal_mae"] == 0.0
        assert np.isclose(result[task]["kendalls_tau"], 1.0)


def test_cross_temporal_metric_counts_collapsed_retrieval_as_zero_tau() -> None:
    module = ast.parse(ADAPTER.read_text(encoding="utf-8"))
    function = next(
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name == "cross_temporal_metrics"
    )
    namespace = {
        "np": np,
        "cdist": cdist,
        "kendalltau": kendalltau,
        "TASKS": ("CarryBox", "KickBox"),
    }
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(ADAPTER), "exec"), namespace)
    query = np.stack((np.arange(8), np.zeros(8)), axis=1).astype(np.float64)
    collapsed = np.zeros((8, 2), dtype=np.float64)
    query_embeddings = {
        "test": {task: {9: {"prototype": query.copy()}} for task in namespace["TASKS"]}
    }
    candidate_embeddings = {
        "test": {task: {9: {"prototype": collapsed.copy()}} for task in namespace["TASKS"]}
    }
    result = namespace["cross_temporal_metrics"](
        query_embeddings,
        candidate_embeddings,
        "test",
        "prototype",
    )
    for task in namespace["TASKS"]:
        assert result[task]["kendalls_tau"] == 0.0
