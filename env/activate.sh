#!/bin/bash
# Put the repo's environment on the shell. SOURCE this, do not execute it.
#
#   source env/activate.sh
#
# Everything is derived from this file's own location, so the checkout can live anywhere and
# there are no absolute paths to update -- that is the point. In particular `newton` is
# imported from the in-repo submodule third_party/newton, so a fresh
# `git clone --recurse-submodules` is runnable with no sibling checkout anywhere on disk.
#
# Safe to source more than once.

# ${BASH_SOURCE[0]} rather than $0: under `source`, $0 is the parent shell's name.
_RB_ENV_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export RB_REPO=$(dirname -- "$_RB_ENV_DIR")
export RB_NEWTON=$RB_REPO/third_party/newton
export RB_ENV_NAME=${RB_ENV_NAME:-robotbaby}

if [ ! -d "$RB_NEWTON/newton" ]; then
    echo "env/activate.sh: $RB_NEWTON/newton is missing." >&2
    echo "  the newton submodule is not checked out; run:" >&2
    echo "    git -C $RB_REPO submodule update --init --recursive third_party/newton" >&2
    return 1 2>/dev/null || exit 1
fi

# ---- conda ----------------------------------------------------------------------------
# `conda activate` is a SHELL FUNCTION, not the binary on PATH. In a non-interactive shell
# (which every one of these runners is) the function does not exist even when `conda` does,
# and activate fails with "Run 'conda init' before 'conda activate'". So the test has to be
# for the function, and the fix is to source conda.sh. RB_CONDA_BASE overrides the install.
if [ "$(type -t conda 2>/dev/null)" != "function" ]; then
    _rb_base=${RB_CONDA_BASE:-}
    # If the binary is on PATH it knows where its own base is; ask it rather than guess.
    [ -z "$_rb_base" ] && command -v conda >/dev/null 2>&1 && _rb_base=$(conda info --base 2>/dev/null)
    for _c in "$_rb_base" "$HOME/miniconda3" "$HOME/anaconda3" \
              /lustre/fsw/portfolios/nvr/users/"$USER"/miniconda3; do
        if [ -n "$_c" ] && [ -f "$_c/etc/profile.d/conda.sh" ]; then
            . "$_c/etc/profile.d/conda.sh"
            break
        fi
    done
    unset _c _rb_base
fi

if [ "$(type -t conda 2>/dev/null)" = "function" ]; then
    if conda env list | awk '{print $1}' | grep -qx "$RB_ENV_NAME"; then
        conda activate "$RB_ENV_NAME"
    else
        echo "env/activate.sh: conda env '$RB_ENV_NAME' not found -- run bash env/setup_env.sh" >&2
        return 1 2>/dev/null || exit 1
    fi
else
    echo "env/activate.sh: no conda found; set RB_CONDA_BASE to your install" >&2
    return 1 2>/dev/null || exit 1
fi

# ---- python path ----------------------------------------------------------------------
# sugar_newton from the repo root, newton from the submodule. Prepended so an in-repo edit
# always wins over anything pip happens to have installed under the same name.
export PYTHONPATH="$RB_REPO:$RB_NEWTON${PYTHONPATH:+:$PYTHONPATH}"

# ---- run-time knobs ------------------------------------------------------------------
# XET is a HuggingFace transfer backend that hangs behind this cluster's proxy.
export HF_HUB_DISABLE_XET=1
# Every figure here is written to a file, and there is no display on a compute node.
export MPLBACKEND=${MPLBACKEND:-Agg}
