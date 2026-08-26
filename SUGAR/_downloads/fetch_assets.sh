#!/bin/bash
# Phase 0 assets for Plan 16, from the public SUGAR release (README "download data").
# Everything lands under SUGAR/ where data/, descriptions/ and demo_ckpts/ are already
# gitignored. SHA-256 of each archive is recorded next to it -- Plan 15 shipped an
# unpinned teacher (expected_sha256=None); this is the thing that stops a repeat.
set -eu
C=/lustre/fs12/portfolios/nvr/projects/nvr_nxp_visionconferencing/users/shengzew/robot_baby/Curiosity
cd "$C/SUGAR/_downloads"
declare -A IDS=(
  [descriptions.zip]=1wXNAjNMrfV0e-d2pQ6m9dm4xrG5lSoyD
  [data.zip]=1AIJWqS5rFGl5u2Qq6jCCTHKdh51SX2Sc
  [demo_ckpts.zip]=1Uc2SPPVvTboEgw4Scyuz3TmzNKDg-dx-
)
for f in descriptions.zip data.zip demo_ckpts.zip; do
  if [ -s "$f" ]; then echo "== $f already present"; else
    echo "== fetching $f (${IDS[$f]})"
    python3 -m gdown "${IDS[$f]}" -O "$f" || { echo "FETCH_FAILED $f"; continue; }
  fi
  sha256sum "$f" | tee "$f.sha256"
  echo "== unpacking $f into $C/SUGAR"
  unzip -q -o "$f" -d "$C/SUGAR" && echo "UNPACKED $f"
done
echo "===== FETCH_DONE ====="
