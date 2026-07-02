"""Write official checkpoint blocker templates for Phase07 mainstream methods.

The templates are not blockers by themselves. They are structured files that
must be filled with a concrete access/incompatibility reason if an official
checkpoint cannot be obtained. This avoids silently replacing official methods
with toy substitutes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


METHODS = {
    "openpi_pi0": {
        "official_checkpoint_examples": [
            "gs://openpi-assets/checkpoints/pi0_base",
            "gs://openpi-assets/checkpoints/pi05_base",
            "gs://openpi-assets/checkpoints/pi05_droid",
        ],
        "official_repo": "external/openpi",
    },
    "gr00t": {
        "official_checkpoint_examples": [
            "https://huggingface.co/collections/nvidia/gr00t-n17",
        ],
        "official_repo": "external/Isaac-GR00T",
    },
    "diffusion_policy": {
        "official_checkpoint_examples": [
            "https://diffusion-policy.cs.columbia.edu/data/experiments/",
        ],
        "official_repo": "external/diffusion_policy",
    },
    "rtx": {
        "official_checkpoint_examples": [
            "gs://gdm-robotics-open-x-embodiment/open_x_embodiment_and_rt_x_oss/rt_1_x_jax",
        ],
        "official_repo": "external/open_x_embodiment",
    },
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/public/home/yanhongru/Curiosity"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/reports/phase07_official_checkpoint_blockers_v1_20260627"),
    )
    args = parser.parse_args()
    root = args.root.resolve()
    output_dir = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    for method, spec in METHODS.items():
        payload = {
            "classification": "phase07_official_checkpoint_blocker_template_v1",
            "status": "template_unfilled_not_a_blocker",
            "method": method,
            "official_repo": spec["official_repo"],
            "official_checkpoint_examples": spec["official_checkpoint_examples"],
            "not_checkpoint": True,
            "not_success_claim": True,
            "not_valid_until_filled": True,
            "required_if_checkpoint_unavailable": {
                "attempted_command": "",
                "attempted_date": "",
                "environment": "",
                "observed_error_or_access_limitation": "",
                "why_this_blocks_faithful_official_comparison": "",
                "why_no_toy_substitute_is_allowed": "Official checkpoint/code incompatibility must be recorded instead of replacing the method with a toy model.",
                "next_user_or_infrastructure_action_needed": "",
            },
        }
        path = output_dir / f"{method}_checkpoint_blocker_template.json"
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        results[method] = str(path.relative_to(root))
    manifest = {
        "classification": "phase07_official_checkpoint_blocker_templates_manifest_v1",
        "status": "pass_templates_written_not_blockers",
        "templates": results,
        "not_checkpoint": True,
        "not_success_claim": True,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
