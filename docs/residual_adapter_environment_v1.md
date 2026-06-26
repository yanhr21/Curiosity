# Residual Adapter Environment V1

## Purpose

This document records the local shared-filesystem environment used by the
Phase 04 Newton-native residual adapter trainer smoke. The environment is not
created or modified on compute nodes.

## Location

```text
envs/residual_adapter/.venv
```

## Creation

The first attempt used the uv-managed Python at
`/public/home/yanhongru/.local/bin/python3.10`, but that interpreter failed to
start pip due an `encodings` import error. The environment was recreated with
the system Python:

```bash
rm -rf envs/residual_adapter/.venv
/public/home/yanhongru/.local/bin/uv venv envs/residual_adapter/.venv --python /usr/bin/python3.10 --seed
```

Packages were installed locally with the Tsinghua mirror:

```bash
envs/residual_adapter/.venv/bin/python -m pip install 'torch==2.6.0' -i https://pypi.tuna.tsinghua.edu.cn/simple
envs/residual_adapter/.venv/bin/python -m pip install 'numpy<3' -i https://pypi.tuna.tsinghua.edu.cn/simple
```

Observed packages used by the smoke:

- `torch==2.6.0+cu124`;
- `numpy==2.2.6`;
- CUDA dependencies from the PyTorch 2.6.0 wheel set.

## Runtime Split

The residual-adapter trainer runner uses two environments:

- `NEWTON_VENV=envs/newton/.venv` for fresh official Newton sanity;
- `TRAINER_VENV=envs/residual_adapter/.venv` for PyTorch trainer execution.

This keeps Newton simulator dependencies separate from trainer dependencies and
avoids dependency installation on compute nodes.
