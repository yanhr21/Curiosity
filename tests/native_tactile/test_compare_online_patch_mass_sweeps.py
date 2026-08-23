import json
from pathlib import Path
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "SUGAR/scripts/sugar_rl/compare_online_patch_mass_sweeps.py"
SUMMARIZER = (
    ROOT / "SUGAR/scripts/sugar_rl/summarize_online_patch_mass_sweep.py"
)
PAIRING = ((151014, 152014), (151015, 152015), (151016, 152016))
FACTORS = (1.0, 1.5, 3.0, 6.0, 10.0)
PYTHON = shutil.which("python3") or sys.executable


def episode(branch: str) -> dict[str, object]:
    improved = branch in {"P", "PS"}
    return {
        "eligible_post_jump_window": True,
        "strict_sugar_eligible_post_jump_window": improved,
        "hold_success": improved,
        "strict_sugar_hold_success": improved,
        "drop": not improved,
        "safe_lower": False,
        "robot_fall": not improved,
        "reference_robot_deviation": not improved,
        "maximum_height_loss_m": 0.02 if improved else 0.20,
        "bilateral_patch_contact_fraction": 0.9 if improved else 0.5,
        "gross_slip_patch_fraction": 0.1 if improved else 0.4,
    }


def write_branch(root: Path, branch: str) -> None:
    for train_seed, evaluation_seed in PAIRING:
        for factor in FACTORS:
            run = root / f"train_{train_seed}_eval_{evaluation_seed}_{factor}"
            run.mkdir(parents=True)
            summary = {
                "branch": branch,
                "evaluation_view": "strict_sugar_reference",
                "training_seed": train_seed,
                "seed": evaluation_seed,
                "mass_factor": factor,
                "episodes": [episode(branch) for _ in range(20)],
            }
            (run / "summary.json").write_text(json.dumps(summary), encoding="utf-8")


def test_completed_paired_sweeps_compare_all_300_profiles(tmp_path: Path) -> None:
    roots = {branch: tmp_path / branch.lower() for branch in ("Z", "P", "PS")}
    for branch, root in roots.items():
        write_branch(root, branch)
    output = tmp_path / "comparison.json"
    subprocess.run(
        [
            PYTHON,
            str(SCRIPT),
            "--z-root",
            str(roots["Z"]),
            "--p-root",
            str(roots["P"]),
            "--ps-root",
            str(roots["PS"]),
            "--output",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["profile_count_per_branch"] == 300
    p_minus_z = result["comparisons"]["P-Z"]["3.0"]
    assert p_minus_z["hold_success"]["paired_profiles"] == 60
    assert p_minus_z["hold_success"]["mean_difference_first_minus_second"] == 1.0
    assert p_minus_z["drop"]["mean_difference_first_minus_second"] == -1.0
    assert (
        p_minus_z["strict_sugar_event_eligible"][
            "mean_difference_first_minus_second"
        ]
        == 1.0
    )
    assert (
        p_minus_z["reference_robot_deviation"][
            "mean_difference_first_minus_second"
        ]
        == -1.0
    )
    ps_minus_p = result["comparisons"]["PS-P"]["3.0"]
    assert ps_minus_p["hold_success"]["mean_difference_first_minus_second"] == 0.0
    assert result["inference"]["independent_unit"] == "training seed"
    assert result["inference"]["multiple_comparison_correction"] == (
        "Holm familywise correction"
    )
    assert p_minus_z["hold_success"]["exact_seed_sign_flip_pvalue"] == 0.25
    assert p_minus_z["hold_success"]["holm_familywise_pvalue"] == 1.0


def test_comparison_rejects_total_300_with_wrong_seed_factor_profile_counts(
    tmp_path: Path,
) -> None:
    roots = {branch: tmp_path / branch.lower() for branch in ("Z", "P", "PS")}
    for branch, root in roots.items():
        write_branch(root, branch)
    bad = roots["Z"] / "train_151014_eval_152014_1.0" / "summary.json"
    payload = json.loads(bad.read_text(encoding="utf-8"))
    payload["episodes"].pop()
    bad.write_text(json.dumps(payload), encoding="utf-8")
    extra = roots["Z"] / "train_151014_eval_152014_1.5" / "summary.json"
    payload = json.loads(extra.read_text(encoding="utf-8"))
    payload["episodes"].append(episode("Z"))
    extra.write_text(json.dumps(payload), encoding="utf-8")
    result = subprocess.run(
        [
            PYTHON,
            str(SCRIPT),
            "--z-root",
            str(roots["Z"]),
            "--p-root",
            str(roots["P"]),
            "--ps-root",
            str(roots["PS"]),
            "--output",
            str(tmp_path / "comparison.json"),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "exact 3x5x20 matched design" in result.stderr


def test_summary_accepts_only_the_exact_15_run_300_profile_design(
    tmp_path: Path,
) -> None:
    root = tmp_path / "z"
    write_branch(root, "Z")
    output = tmp_path / "summary.json"
    subprocess.run(
        [
            PYTHON,
            str(SUMMARIZER),
            "--input-root",
            str(root),
            "--output",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["source_runs"] == 15
    assert result["profiles"] == 300
    assert result["factors"]["3.0"]["strict_sugar_eligible_profiles"] == 0

    missing = next(root.glob("train_151014_eval_152014_1.0/summary.json"))
    missing.unlink()
    rejected = subprocess.run(
        [
            PYTHON,
            str(SUMMARIZER),
            "--input-root",
            str(root),
            "--output",
            str(tmp_path / "incomplete.json"),
        ],
        capture_output=True,
        text=True,
    )
    assert rejected.returncode != 0
    assert "exact 3x5 matched run set" in rejected.stderr
