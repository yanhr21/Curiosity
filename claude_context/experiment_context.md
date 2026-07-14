# Experiment Context — paths, tools, infra

*Quick reference for the Robot Baby project. Bring-up stage — most of this is target/placeholder and will fill
in as we build. Keep paths absolute; convert relative dates to absolute.*

## Locations

- **Project root:** `/lustre/fs12/portfolios/nvr/projects/nvr_nxp_visionconferencing/users/shengzew/robot_baby/`
- **Mirror (fsw):** `/lustre/fsw/portfolios/nvr/users/shengzew/robot_baby/`
- **Context site:** `claude_context/index.html` — serve with `python3 claude_context/serve.py`, open
  `http://localhost:8090/`.
- **This repo currently contains only the context scaffold** (no code yet).

## The context site

- `context.md` (repo root) — canonical top-level context.
- `claude_context/index.html` — rendered brief: Map · Motivation · Framework · Roadmap · **Related Work** · Docs.
- `claude_context/serve.py` — static server on `:8090` + `/branch` git auto-detect. Serves from
  `claude_context/`; all Docs-tab markdown is fetched **same-dir**.
- Docs tab renders (via marked.js): `TODOs.md`, `overview.md`, `architecture_sdg_engine.md`,
  `architecture_dcrl.md`, `related_works.md`, `experiment_context.md`.
- **Related Work tab** state (stars/pins/demotions) is stored in `localStorage` key `robotbaby.related.v1`
  (per-browser; "Reset" clears it). To add a paper, edit the `WORKS` array in `index.html`.
- Must be served over **HTTP** (not `file://`) or `fetch()` for the markdown/related data is blocked.

## Tooling / stack (target)

| Concern | Choice / candidate | Notes |
|---|---|---|
| Physics | **Newton** (Warp + OpenUSD) | GPU, MuJoCo-Warp; open-source, Apache-2.0 |
| Sim framework | **Isaac Sim / Isaac Lab** | multi-modal RL/IL, rendering |
| Assets | **REST3D** | image → physically-stable, sim-ready scenes |
| Tactile | **TacSL** (Isaac) | GPU visuotactile sim — evaluate |
| Demo encoder | R3M / TCN / VIP / RoboCLIP | TBD — see `related_works.md` |
| Reward-from-demo | VIP / RoboCLIP / TCN dist / Diffusion Reward | TBD |
| RL backbone | DAPG / RLPD / DemoStart-style curriculum | TBD |

## People

- **Shengze Wang** (shengzew@nvidia.com) — Part 1 SDG engine; co-lead Part 2.
- **Hongru** — co-lead Part 2 (demonstration-conditioned RL).

## Conventions

- Keep `TODOs.md` the single source of truth for status.
- Keep architecture docs *honest*: mark target vs. built; note when reality diverges.
- Prefer absolute dates (e.g. "2026-07-13") over relative in all context files.
