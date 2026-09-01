#!/bin/bash
# Run one g1_carrybox_policy configuration per GPU on the 8-GPU dev node.
#
# Every previous comparison in this project was taken with several benchmarks sharing one
# GPU, which is why the numbers disagreed with each other. One config per device is the
# fix: the scene is a single world, so it barely occupies an A100 and eight fit side by
# side with no contention.
#
#   bash sugar_newton/par_policy.sh sugar_newton/sweeps/ab_contact.txt
#
# Each line of the config file is a label, then a tab, then the argument string.
set -u
REPO=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
OUT=$REPO/sugar_newton/_par
mkdir -p "$OUT"
# shellcheck source=/dev/null
. "$REPO/env/activate.sh" || exit 2
cd "$REPO" || exit 2

i=0
declare -a LABELS
while IFS=$'\t' read -r label cfg; do
    case "$label" in ''|\#*) continue;; esac
    echo "GPU $i <- $label :: $cfg"
    LABELS[$i]=$label
    # stdbuf: the log is tailed while running, and python block-buffers into a pipe.
    CUDA_VISIBLE_DEVICES=$i stdbuf -oL -eL python -m \
        sugar_newton.validation.g1_carrybox_policy $cfg \
        > "$OUT/${i}_${label}.log" 2>&1 &
    i=$((i + 1))
done < "$1"
echo "=== $i configurations launched, waiting ==="
wait
echo "=== all done ==="
for j in $(seq 0 $((i - 1))); do
    echo "----- ${LABELS[$j]} -----"
    grep -E "frames in|box lift|box peak|joint tracking|actuator saturation|peak \|tau\||DIVERGED|Error|Traceback" \
        "$OUT/${j}_${LABELS[$j]}.log" || tail -4 "$OUT/${j}_${LABELS[$j]}.log"
done
