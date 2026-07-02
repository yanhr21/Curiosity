# RUNBOOK — set up & run the SDG pipeline (for another agent)

End-to-end guide to set up and run the **synthetic-data generation (SDG)** pipeline
in this repo: **PixelDiT (text→image) → TRELLIS.2 (image→3D GLB) → Newton (rigid-body
sim)**. Covers a workstation and a Slurm cluster, every gotcha we hit, how to run
generation, and the **current cluster state** (what's done / what's left).

See also: `genpipe/README.md` (architecture, per-item cost, fleet scaling),
`claude_context/experimental_conclusions.md` §8 (profiling), and the top-level
`context.md`.

---

## 0. What this is

- **Repo:** a fork of `newton`, branch **`shengzew/fps-collision-benchmarks`**.
  - Remotes: **`mikewang`** = `ssh://git@gitlab-master.nvidia.com:12051/shengzew/newton_mikewang.git`
    (push here); `origin` = `github.com/newton-physics/newton` (upstream, don't push).
- **Bridge code:** `genpipe/` — `run_pipeline.sh` (end-to-end), `trellis_image_to_glb.py`
  (image→GLB, with all the fixes baked in), `cluster_setup.sh` (env bootstrap), this file.
- **Submodules:** `third_party/PixelDiT` (NVlabs), `third_party/TRELLIS.2` (microsoft,
  has a nested `o-voxel/third_party/eigen` from gitlab.com).
- **Newton demo:** `example_panda_clock_metal.py` (Franka grips the generated clock, metal
  material, places in cup) + `tactile_clock_metal.py` (tactile measurement video).
- Outputs are gitignored (`pipeline_out/`, `tactile_clock_frames/`, `*.mp4`, GLBs).

## 1. Get the repo

```bash
git clone -b shengzew/fps-collision-benchmarks --recurse-submodules -j4 \
  ssh://git@gitlab-master.nvidia.com:12051/shengzew/newton_mikewang.git newton
# (or clone from a mirror and: git submodule update --init --recursive)
```

## 2. Conda envs

Three **separate** envs — their dependency stacks conflict with each other and with
`newton`. They are bridged by files on disk.

| env | purpose | how |
|---|---|---|
| `pixeldit` | text→image | `conda create -n pixeldit python=3.10 -y`; `pip install -r third_party/PixelDiT/requirements.txt` (torch 2.5 cu124) |
| `trellis2` | image→3D | see `genpipe/cluster_setup.sh` (torch 2.6 cu124 + conda cuda-toolkit 12.4 + xformers + CUDA extensions) |
| `newton` | rigid-body sim | not yet built on the cluster; locally it's an editable `newton` + `uv sync --extra examples` (warp, mujoco-warp, usd, torch) |

**One command does pixeldit + trellis2:**
```bash
bash genpipe/cluster_setup.sh "8.0;9.0"   # arg = TORCH_CUDA_ARCH_LIST (A100=8.0, H100=9.0)
```
It's idempotent-ish and prints `CLUSTER_SETUP_DONE` + an import smoke test on success.

## 3. Gotchas (all handled in the scripts — here so you understand them)

1. **`HF_HUB_DISABLE_XET=1`** — the hf-xet transfer protocol stalls at 0 B on our
   networks; forces plain HTTPS.
2. **flash-attn won't install** (its sdist build can't see torch under pip isolation) →
   use **xformers** (`ATTN_BACKEND=xformers`), a drop-in for TRELLIS's sparse+dense attn.
3. **No usable system `nvcc`** → install a conda **`cuda-toolkit 12.4`** into `trellis2`
   and set `CUDA_HOME=$CONDA_PREFIX`, `TORCH_CUDA_ARCH_LIST=<arch>` before compiling.
4. **DINOv3 image encoder is GATED** (`facebook/dinov3-vitl16-pretrain-lvd1689m`) →
   need an **`HF_TOKEN` with access** at *generation* time (not for env setup). The user's
   HF account (`McMvMc`) has access — supply the token via `export HF_TOKEN=...` (never commit it).
5. **RMBG-2.0 rembg is GATED** (`briaai/RMBG-2.0`) → `trellis_image_to_glb.py` auto-redirects
   to the ungated `ZhengPeng7/BiRefNet` (identical interface).
6. **BiRefNet** needs `einops` and fp32 weights (handled).
7. **transformers ≥5** nests DINOv3 blocks under `.model`; the script patches the feature walk.
8. **`trellis2` is a source package** (not pip-installed) — the runner adds its repo root to `sys.path`.

## 4. Run generation

```bash
# Full pipeline (needs HF_TOKEN with DINOv3 access):
export HF_TOKEN=hf_...      # DINOv3-authorized token
bash genpipe/run_pipeline.sh [PROMPT_FILE] [OUTDIR] [GPU]

# Just image→3D on an existing image:
conda activate trellis2
export CUDA_HOME=$CONDA_PREFIX PATH=$CONDA_PREFIX/bin:$PATH ATTN_BACKEND=xformers HF_TOKEN=hf_...
python genpipe/trellis_image_to_glb.py --image IMG.jpg --out obj.glb [--preview p.png]

# Batch many objects in ONE process to amortize the model load (see /tmp/trellis_batch.py
# pattern in the profiling notes): load pipeline once, loop pipe.run(img) per image.

# Newton clock demo (needs the `newton` env + a generated GLB at pipeline_out/object.glb):
python example_panda_clock_metal.py --viewer gl        # grip + place-in-cup
python tactile_clock_metal.py --frames 570             # tactile measurement video
```

**Cost (RTX 6000 Ada, 1 GPU, model resident):** ~8 s/image, ~62 s/object, ~5 s sim.
Both generative stages are **compute-bound** — no batch/concurrency speedup on one GPU, so
scale **out across GPUs** (1 stream/GPU). 10k objects ≈ 197 GPU-hr (~3 h on 8×8-GPU nodes),
~71 GB GLBs; hydroelastic-SDF conversion is ~0.3 s/object (+~5 GB). Details in `genpipe/README.md`.

## 5. Cluster specifics (oci-ord, Slurm)

- **Login node:** `oci-ord-cs`, conda base on Lustre, `git` + `gcc 11`, **no system `nvcc`**.
  Repo lives at `/lustre/fsw/portfolios/nvr/users/shengzew/sdg/newton`
  (real path `/lustre/fs12/.../nvr_nxp_visionconferencing/users/shengzew/sdg/newton`).
- **Slurm:** account **`nvr_nxp_visionconferencing`**, **8 GPUs/node**, partitions
  `backfill_singlenode` / `batch_singlenode` (gpu:8) and `cpu`. GRES is untyped (`gpu:8`) —
  confirm the GPU model with a job: `srun -A nvr_nxp_visionconferencing -p backfill_singlenode
  --gres=gpu:1 -t 2 nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader`.
- **⚠️ The login node kills heavy processes.** Our `cluster_setup.sh` got **SIGKILLed
  mid-`torch`-install** for `trellis2` when run on the login node. **Run the build inside a
  Slurm job**, e.g.:
  ```bash
  srun -A nvr_nxp_visionconferencing -p backfill_singlenode --gres=gpu:1 -t 120 \
    bash /lustre/fsw/portfolios/nvr/users/shengzew/sdg/newton/genpipe/cluster_setup.sh "8.0;9.0"
  ```
  or `sbatch` a wrapper. **Check whether compute nodes have internet** (`srun ... git ls-remote
  https://github.com/NVlabs/PixelDiT.git`); if they don't, the pip/clone steps must run on the
  login node in small pieces (or use a data-mover / offline wheels) — TBD.
- All GPU work (generation, verification) goes through `srun`/`sbatch`.

## 6. Driving the `ord_pod` screen session (if you're a Claude agent)

The user keeps a `screen` session **`ord_pod`** attached to the SSH into `oci-ord-cs`.
Send commands with `/tmp/scx.sh` (sends `cmd ; echo <sentinel> rc=$?`, then polls
`screen -S ord_pod -X hardcopy`). Hard-won rules:

- **Shell vars come back empty** through this channel (`$MY_HOME`, `$SDG`, even `X=1;echo $X`)
  — **use absolute paths**, no vars.
- **Terminal is ~64 cols** by default and wraps/mangles long lines → `screen -S ord_pod -X
  width 400 100` first (it can revert if the user reattaches — re-issue as needed).
- **Avoid escaped quotes** (`\"`) and **regex brackets** (`[^ ]`) — they get mangled; use
  `python -c "code with no inner quotes"`, `grep -E` with alternation only, `tr`.
- **Background long ops**: `nohup <cmd> > log 2>&1 & disown; echo STARTED` — the `disown`
  prevents async job-exit messages from corrupting your next command. Poll the log with
  `tail`/`grep` (filter conda progress bars: they contain `##########` and `\r`).
- To wait for a long job: `while kill -0 <pid> 2>/dev/null; do sleep 30; done; echo DONE`
  (run it via a **background** local `scx.sh` call so you're notified when it prints DONE).

## 7. CURRENT STATE (handoff checkpoint)

- ✅ Repo cloned on the cluster (`sdg/newton`, branch @ c21dfb3e + setup script c87b3e01,
  both submodules + eigen).
- ✅ **`pixeldit` env** created and `requirements.txt` installed (the build reached the
  TRELLIS.2 stage, so pixeldit finished; do a final `conda run -n pixeldit python -c "import
  torch,transformers"` to confirm).
- ⚠️ **`trellis2` env INCOMPLETE** — env created and torch install started, then the process
  was **killed by the login node** partway through the torch install. Remaining: finish
  torch, install cuda-toolkit 12.4, basic deps, xformers, einops, and the 5 CUDA extensions.
  **Fix: re-run `genpipe/cluster_setup.sh "8.0;9.0"` inside a Slurm job** (§5); it will skip
  the finished pixeldit env and redo trellis2.
- ⬜ **Verify** via a GPU `srun` job: `conda run -n trellis2 python -c "import torch;
  print(torch.cuda.is_available())"` + import nvdiffrast/o_voxel/flex_gemm/cumesh/xformers.
- ⬜ **`newton` sim env** on the cluster: not built yet.
- ⬜ **HF token** for DINOv3 must be exported before running generation.
- ⬜ First end-to-end generation run + (optionally) a multi-GPU batch for the 10k-object set.
