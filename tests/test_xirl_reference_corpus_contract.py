from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import kendalltau


ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "scripts/sugar/demo_following/render_xirl_reference_corpus.py"
CONFIG_PATH = ROOT / "scripts/sugar/demo_following/xirl_configs/sugar_carry_kick_tcc.py"
EVAL_PATH = ROOT / "scripts/sugar/demo_following/evaluate_official_xirl_tcc_temporal_retrieval.py"
RUNNER_PATH = ROOT / "scripts/sugar/demo_following/run_official_xirl_tcc_pretrain_then_hold.sh"
FULL_RUNNER_PATH = ROOT / "scripts/sugar/demo_following/run_xirl_full_pipeline_then_hold.sh"
PREPARE_PATH = ROOT / "scripts/sugar/demo_following/prepare_official_xirl_runtime.sh"
DEVICE_PATCH_PATH = ROOT / "scripts/sugar/demo_following/xirl_compat/official_xirl_pytorch_device.patch"
OFFICIAL_XIRL = ROOT / "experiments/runtime_assets/official_google_research_xirl/xirl"
COMPAT_DEPS = ROOT / "experiments/runtime_assets/official_xirl_py311_compat_deps"


def _function(name: str):
    module = ast.parse(PATH.read_text(encoding="utf-8"))
    return next(node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == name)


def test_motion_split_is_source_id_disjoint() -> None:
    function = _function("split_for_motion")
    namespace: dict[str, object] = {}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(PATH), "exec"), namespace)
    split = namespace["split_for_motion"]
    assert split(18) == "valid"
    assert split(19) == "test"
    assert split(20) == "train"
    assert split(98) == "valid"
    assert {
        name: sum(split(motion_id) == name for motion_id in range(100))
        for name in ("train", "valid", "test")
    } == {"train": 80, "valid": 10, "test": 10}
    assert {
        name: sum(split(motion_id) == name for motion_id in range(99))
        for name in ("train", "valid", "test")
    } == {"train": 80, "valid": 10, "test": 9}


def test_renderer_declares_clean_exact_64_frame_contract() -> None:
    text = PATH.read_text(encoding="utf-8")
    assert "OUTPUT_FRAME_COUNT = 64" in text
    assert "RTX_RENDER_SIZE = 640" in text
    assert "interpolation=cv2.INTER_AREA" in text
    assert "cfg.sim.render_interval = cfg.decimation" in text
    assert 'parser.add_argument("--camera-width"' in text
    assert 'parser.add_argument("--camera-height"' in text
    assert 'parser.add_argument("--width"' not in text
    assert 'parser.add_argument("--height"' not in text
    assert "InteractiveScene(cfg.scene)" in text
    assert "SimulationContext(cfg.sim)" in text
    assert "gym.make(" not in text
    assert '"clean_frame_contract"' in text
    assert "no text, plot, border, metric or policy output" in text
    assert 'prim_path="{ENV_REGEX_NS}/XirlWorldCamera"' in text


def test_official_runner_has_exact_split_and_modern_device_contract() -> None:
    runner = RUNNER_PATH.read_text(encoding="utf-8")
    patch = DEVICE_PATCH_PATH.read_text(encoding="utf-8")
    assert '"train KickBox 80" "valid KickBox 10" "test KickBox 9"' in runner
    assert 'XIRL_TMPDIR="/public/home/yanhongru/.xirl_tmp_${SLURM_JOB_ID}"' in runner
    assert 'export TMPDIR="$XIRL_TMPDIR"' in runner
    assert "torch.eye(K, device=y.device)[y]" in patch
    assert "patch --batch --forward" in runner
    assert 'resume_args+=(--resume)' in runner
    assert "if ((latest_step < 4001)); then" in runner
    assert 'has("passed")' in runner
    full_runner = FULL_RUNNER_PATH.read_text(encoding="utf-8")
    assert "corpus_complete" in full_runner
    assert "partial immutable XIRL corpus exists" in full_runner
    prepare = PREPARE_PATH.read_text(encoding="utf-8")
    assert "807d4a2f41202059bac2446259d135a89ed3630a" in prepare
    assert "dd5824445b5c3ec9f5b0973c89ffd489500b9eae" in prepare
    assert "31fa989e2ecd6fcdbcdc6f9b70057ab28f6184f2" in prepare


def test_sugar_config_preserves_official_tcc_and_uses_motion_classes(
    monkeypatch,
) -> None:
    monkeypatch.syspath_prepend(str(OFFICIAL_XIRL))
    monkeypatch.syspath_prepend(str(COMPAT_DEPS))
    monkeypatch.setenv("SUGAR_XIRL_DATA_ROOT", "/tmp/xirl_data")
    monkeypatch.setenv("SUGAR_XIRL_RUN_ROOT", "/tmp/xirl_runs")
    spec = importlib.util.spec_from_file_location("sugar_xirl_tcc_config", CONFIG_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config = module.get_config()
    assert config.algorithm == "tcc"
    assert config.model.model_type == "resnet18_linear"
    assert config.model.embedding_size == 32
    assert config.frame_sampler.num_frames_per_sequence == 40
    assert config.loss.tcc.loss_type == "regression_mse"
    assert config.loss.tcc.similarity_type == "l2"
    assert config.optim.train_max_iters == 4000
    assert config.data.pretraining_video_sampler == "same_class"
    assert tuple(config.data.pretrain_action_class) == ("CarryBox", "KickBox")


def test_temporal_metric_is_exact_for_aligned_synthetic_sequences() -> None:
    module = ast.parse(EVAL_PATH.read_text(encoding="utf-8"))
    function = next(
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name == "temporal_metrics"
    )
    namespace = {"np": np, "cdist": cdist, "kendalltau": kendalltau}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(EVAL_PATH), "exec"), namespace)
    sequence = np.stack((np.arange(8), np.zeros(8)), axis=1).astype(np.float32)
    metrics = namespace["temporal_metrics"]({1: sequence, 2: sequence.copy()})
    assert metrics["normalized_temporal_mae"] == 0.0
    assert np.isclose(metrics["kendalls_tau"], 1.0)
