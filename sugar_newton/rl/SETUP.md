# Getting rsl_rl, tensordict and wandb into the Newton venv

`train_bcppo.py` needs SUGAR's stack inside the Newton container. Two constraints make
this awkward, and both are environmental rather than anything about the code:

- The compute nodes have no package mirror.
- `Curiosity_newton/.venv/bin/python` is not executable from the login node, so
  `pip install` cannot simply be run against it.

The way through is to install *from the login node* into the venv's `site-packages` with
`--target`, cross-targeting cp312. Two traps:

1. **`--no-deps` is not enough on its own.** It leaves `orjson`, `pydantic_core`,
   `cloudpickle` and friends missing, and the failure surfaces as
   `ModuleNotFoundError: No module named 'orjson'` from deep inside `tensordict`.
2. **The full closure includes torch, torchvision, numpy and the whole `nvidia_*` CUDA
   stack.** Installing those replaces the venv's CUDA-matched `torch 2.11.0+cu128` with a
   stock PyPI build. They must be excluded.

So: resolve the closure with `pip download`, then install every wheel except the torch
stack, one at a time so a single bad tag does not abort the batch.

    SP=$NEWTON/.venv/lib/python3.12/site-packages
    W=/tmp/wheels
    PF="--python-version 312 --only-binary=:all: --implementation cp --abi cp312"

    pip download --dest $W $PF --platform manylinux2014_x86_64 \
        "rsl-rl-lib==3.0.1" "tensordict>=0.7.0" wandb

    EXCL='^(torch|torchvision|numpy|triton|nvidia_|sympy|networkx|filelock|fsspec|jinja2|markupsafe|mpmath|pillow|ml_dtypes)'
    for w in $W/*.whl; do
      n=$(basename "$w" | sed 's/-[0-9].*//')
      echo "$n" | grep -qiE "$EXCL" && continue
      pip install --target "$SP" --no-deps --upgrade "$w" || \
      for plat in manylinux_2_28_x86_64 manylinux_2_17_x86_64 manylinux2014_x86_64 any; do
        pip install --target "$SP" --no-deps --upgrade $PF --platform $plat "$w" && break
      done
    done

The retry loop matters: compiled wheels (`orjson`, `pydantic_core`, `pyyaml`, `onnx`,
`tensordict`, `charset_normalizer`) carry manylinux tags that pip refuses unless the same
`--platform`/`--abi` flags are given at *install* time, not just at download time.

Check afterwards that `ls $SP/torch-*.dist-info` still reads `torch-2.11.0+cu128`.
