"""Export the current project schema scaffold as JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from newton_tactile_curiosity.data_schemas.trex_contract import DEFAULT_TREX_CONTRACT
from newton_tactile_curiosity.envs.tabletop_scene_spec import default_tabletop_scene_spec


def build_schema() -> dict:
    return {
        "status": "schema scaffold only; no simulation, training, or model replacement",
        "trex_contract": DEFAULT_TREX_CONTRACT.as_json(),
        "newton_tabletop_scene": default_tabletop_scene_spec().as_json(),
        "blockers": [
            "standalone sharpa_wave_deform_encoder.pth remains unresolved for official T-Rex training/post-training scripts",
            "Taccel official peg sanity and mesh visual gate pass, but Newton/Taccel-to-T-Rex deform-map tensor export remains unvalidated",
            "Newton Allegro candidate is not a faithful bimanual 62D T-Rex embodiment",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    text = json.dumps(build_schema(), indent=2)
    if args.output is None:
        print(text)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
        print(args.output)


if __name__ == "__main__":
    main()
