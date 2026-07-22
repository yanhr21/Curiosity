# G1-in-SAGE headless render — reproducible setup

Renders `example_g1_in_sage.py` (G1 loco policy in a dynamic-physics SAGE room) to mp4 on an
**oci-ord interactive GPU node**, inside the stock CUDA container. Software GL + Xvfb, because the
compute container has no EGL/hardware GL device.

## Why a per-container setup step is needed

The container's root filesystem is **ephemeral** — every fresh `srun` starts from the pristine
`.sqsh` image. Anything installed into `/usr` (system GL libs, the **Xvfb** X server) is gone on the
next container. What *does* persist (all on Lustre): the repo `.venv`, `~/miniconda3`, and this
repo. conda/uv envs therefore keep the *Python* deps across containers, but **cannot** carry the
system libraries — and conda-forge doesn't ship the Xvfb server. So a small setup step runs each
fresh container. To keep it fast and network-independent (archive.ubuntu.com is flaky here), the
`.deb` packages are cached on Lustre (`_toolcache/apt-debs/`) and installed offline via `dpkg`.

## One-time-per-container

```bash
# 1. grab an interactive GPU node in a screen (hold it; drive via screen -X stuff)
srun -A nvr_nxp_visionconferencing --partition interactive --time 4:00:00 --gpus 1 \
  --container-mounts=$HOME:/home,/lustre:/lustre \
  --container-image /lustre/fsw/portfolios/nvr/users/shengzew/cuda_docker_ver3.sqsh \
  --job-name nw_render --pty bash

# 2. inside the container: install GL libs + Xvfb from the Lustre deb-cache (offline, ~30s)
bash renders/setup_container.sh          # prints SETUP_CONTAINER_DONE

# 3. (first time only, if .venv is absent) build the Newton env
uv sync --extra examples --extra torch-cu12
```

`setup_container.sh` is idempotent. If the deb-cache is ever empty it falls back to `apt`
(with retries) and re-seeds the cache. The cache currently holds the full GL/Xvfb closure incl.
`xkb-data` (without which Xvfb dies: *"XKB: Failed to compile keymap"*). If you ever need to refresh
a deb, download it on the **login node** (better network) into `_toolcache/apt-debs/`.

## Render

```bash
bash renders/check2.sh        # 8-frame smoke test (verify framing/textures)
bash renders/render_two.sh 150   # rough (mu 1.4) + slippery (mu 0.05), frames only
```

`ffmpeg` is **not** in the container — the scripts write PNG frames to
`<out>.mp4.frames/f%05d.png`; assemble the mp4 on the **login node**:

```bash
ffmpeg -y -framerate 50 -i g1_rough.mp4.frames/f%05d.png -c:v libx264 -pix_fmt yuv420p -crf 18 g1_rough.mp4
```

## Files

- `setup_container.sh` — GL/Xvfb install from Lustre deb-cache (persistent), + uv.
- `render_env.sh` — Xvfb :99 + software-GL env prelude (`source` before running the example).
- `check2.sh` — short smoke test. `render_two.sh` — the two-version render.
- `_toolcache/apt-debs/` — cached `.deb`s (persist on Lustre, survive every container).
