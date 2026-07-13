#!/usr/bin/env bash
set -euo pipefail

if [[ "$(hostname)" == mgmtserver* ]]; then
  echo "Refusing to run SUGAR environment preflight on login/management node: $(hostname)" >&2
  exit 2
fi

ROOT_DIR="${ROOT_DIR:-/public/home/yanhongru/Curiosity}"
SUGAR_DIR="${SUGAR_DIR:-${ROOT_DIR}/SUGAR}"
ISAACLAB_DIR="${ISAACLAB_DIR:-${ROOT_DIR}/IsaacLab}"
PYTHON_BIN="${PYTHON_BIN:-python}"

if [[ ! -f "${SUGAR_DIR}/CURIOSITY_UPSTREAM_COMMIT" ]]; then
  echo "Missing vendored official SUGAR source at ${SUGAR_DIR}" >&2
  exit 3
fi

if [[ ! -f "${ISAACLAB_DIR}/VERSION" ]]; then
  echo "Missing vendored official IsaacLab source at ${ISAACLAB_DIR}" >&2
  exit 3
fi

isaaclab_tag="v$(tr -d '[:space:]' < "${ISAACLAB_DIR}/VERSION")-curiosity-glue"
if [[ "${isaaclab_tag}" != v2.3.0* ]]; then
  echo "[SUGAR-PREFLIGHT] Unexpected IsaacLab repository version: ${isaaclab_tag}" >&2
  exit 5
fi
echo "[SUGAR-PREFLIGHT] isaaclab_repo=${isaaclab_tag}"

required_paths=(
  "${SUGAR_DIR}/data/CarryBox"
  "${SUGAR_DIR}/descriptions/robots/g1/g1_29dof_rev_1_0_with_rubber_hand.urdf"
  "${SUGAR_DIR}/descriptions/objects/small_box/obj_aligned.usd"
  "${SUGAR_DIR}/demo_ckpts/CarryBox/tracker.pt"
  "${SUGAR_DIR}/demo_ckpts/CarryBox/generator.ckpt"
)

for path in "${required_paths[@]}"; do
  if [[ ! -e "${path}" ]]; then
    echo "[SUGAR-PREFLIGHT] missing required official asset: ${path}" >&2
    exit 4
  fi
done

"${PYTHON_BIN}" - <<'PY'
import importlib.metadata as md
import sys

errors = []

print(f"[SUGAR-PREFLIGHT] python={sys.executable}")
print(f"[SUGAR-PREFLIGHT] python_version={sys.version.split()[0]}")

if sys.version_info[:2] != (3, 11):
    errors.append(
        f"Python must be 3.11 for official SUGAR; found {sys.version.split()[0]}"
    )


def version_for(*names):
    for name in names:
        try:
            return name, md.version(name)
        except md.PackageNotFoundError:
            pass
    return names[0], None


checks = [
    ("isaacsim", ("isaacsim",), lambda v: v.startswith("5.1.0")),
    ("isaaclab", ("isaaclab",), lambda v: v == "0.47.2" or v.startswith("2.3.")),
    ("sugar_rl", ("sugar-rl", "sugar_rl"), lambda v: True),
    ("sugar_il", ("sugar-il", "sugar_il"), lambda v: True),
    ("rsl_rl", ("rsl-rl-lib", "rsl_rl", "rsl-rl"), lambda v: True),
    ("numpy", ("numpy",), lambda v: v == "1.26.0"),
    ("zarr", ("zarr",), lambda v: v == "2.12.0"),
    ("numcodecs", ("numcodecs",), lambda v: v == "0.12.1"),
    ("hydra_core", ("hydra-core",), lambda v: True),
    ("omegaconf", ("omegaconf",), lambda v: True),
    ("diffusers", ("diffusers",), lambda v: v == "0.32.1"),
    ("accelerate", ("accelerate",), lambda v: v == "1.2.1"),
    ("timm", ("timm",), lambda v: v == "1.0.12"),
    ("datasets", ("datasets",), lambda v: v == "2.6.1"),
    ("numba", ("numba",), lambda v: True),
    ("pydantic", ("pydantic",), lambda v: v == "2.11.4"),
]

for label, candidates, predicate in checks:
    package_name, version = version_for(*candidates)
    if version is None:
        errors.append(f"Missing package for {label}; tried {', '.join(candidates)}")
        print(f"[SUGAR-PREFLIGHT] {label}=MISSING")
        continue
    print(f"[SUGAR-PREFLIGHT] {label}={package_name}=={version}")
    if not predicate(version):
        errors.append(f"Unexpected {label} version: {package_name}=={version}")

if errors:
    print("[SUGAR-PREFLIGHT] FAILED")
    for error in errors:
        print(f"[SUGAR-PREFLIGHT] {error}")
    sys.exit(10)

print("[SUGAR-PREFLIGHT] PASS")
PY
