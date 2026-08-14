import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "SUGAR/scripts/sugar_rl/compare_online_patch_mass_sweeps.py"
PAIRING = ((151014, 152014), (151015, 152015), (151016, 152016))
FACTORS = (1.0, 1.5, 3.0, 6.0, 10.0)


def episode(branch: str) -> dict[str, object]:
    improved = branch in {"P", "PS"}
    return {
        "eligible_post_jump_window": True,
        "hold_success": improved,
        "drop": not improved,
        "safe_lower": False,
        "robot_fall": not improved,
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
            sys.executable,
            str(SCRIPT),
            "--z-root",
            str(roots["Z"]),
            "--p-root",
            str(roots["P"]),
            "--ps-root",
            str(roots["PS"]),
            "--output",
            str(output),
            "--bootstrap-samples",
            "100",
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
    ps_minus_p = result["comparisons"]["PS-P"]["3.0"]
    assert ps_minus_p["hold_success"]["mean_difference_first_minus_second"] == 0.0
