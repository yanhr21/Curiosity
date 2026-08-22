# Phase 0 assets — provenance

Fetched 2026-08-21 from the **public SUGAR release**, not from the runtime host.
Plan 16's TODO recorded these as living only at `/public/home/yanhongru/Curiosity`;
that path is on a different cluster and is not mounted here, but the SUGAR authors
publish the same assets from the repo README's "download data" section.

Source: <https://github.com/tianshuwu/SUGAR> (code, 913 KB, no assets) → README
"5. download data" → three Google Drive archives.

Reproduce with `bash _downloads/fetch_assets.sh` (idempotent; skips what is present).

## Archives

| archive | gdrive id | sha256 |
|---|---|---|
| `descriptions.zip` | `1wXNAjNMrfV0e-d2pQ6m9dm4xrG5lSoyD` | `9cd9bbaa70b272edd4e76e30087eb952b654c084b2552df57b13be1e2c23afeb` |
| `data.zip` | `1AIJWqS5rFGl5u2Qq6jCCTHKdh51SX2Sc` | `1ef55bba9400eb748ec1ffe86b6b434d1b922b14ea15efd06c1e47753f0e042f` |
| `demo_ckpts.zip` | `1Uc2SPPVvTboEgw4Scyuz3TmzNKDg-dx-` | `6a81ccc3de01b8eb7b13f716f88a8557efca6d8f6627a53a8ab130c4200004d1` |

## Individual assets

| sha256 | asset |
|---|---|
| `6cffb5942750bdd934ddd77e4c60218928e56015ae95bdb6ae8ce329cacc8530` | `descriptions/robots/g1/g1_29dof_rev_1_0_with_rubber_hand.urdf` |
| `ea406c98f622588e8537c14240c3101c4c3c75a333e74572d63b81f3fc391aa2` | `demo_ckpts/CarryBox/tracker.pt` |
| `7f8c9f6a141e21b3561c8186ed9c339028269423191e88f36bd431ac9a9470d6` | `demo_ckpts/CarryBox/generator.ckpt` |

## Against the Phase 0 checklist

| # | wanted | got |
|---|---|---|
| 1 | `descriptions/robots/g1/meshes` — geometry the 54 patches come from | **yes**, 165 files / 137 MB, with the rubber-hand URDF |
| 2 | `data/CarryBox` — reference motion (`--motion_folder`) | **yes**, 100 clips / 101 MB (all six tasks: 565 MB) |
| 3 | official CarryBox asset | **yes**, `descriptions/objects/big_box` and `small_box` |
| 4 | official Tracker checkpoint — actor warm start | **yes**, `demo_ckpts/CarryBox/tracker.pt` (+ `generator.ckpt`, not on the list) |
| 5 | teacher `refiner_model10000.pt` | **yes** — not from the public release; recovered from our own workspace backup on the Hub (see below) |
| 6 | `gelsight_r15_finger.usd` | **no** — it is a TacSL asset, not a SUGAR one. Plan 16 §C says do not port the TacSL sensing half, so this is likely not needed; confirm before chasing it. |

## The refiner: recovered, not published

**Found 2026-08-21.** It was never in SUGAR's public release, but it was in our own
archived workspace on the Hub:

```
hf download Railgun526/curiosity-workspace-large-files-20260708 --type dataset \
  --include "experiments/sugar_reproduction/outputs/final/official_sugar/baseline/ckpts/refiner_model10000.pt" \
  --local-dir <Curiosity>
```

which lands it at exactly the path the scripts look for
(`train_online_patch_mass_bcppo.py:16`). 14 957 503 bytes,
sha256 `a398a7293fcea0ef948234e5de47b990fa586d2efd4e54ad7e481151c16124c3`.

Verified as a real teacher, not a stub: `iter = 10000`, and the actor is
`890 -> 512 -> 256 -> 128 -> 29` with a matching critic head -- 29 outputs being the G1
29-DoF action space Plan 16 §C keeps. So the Phase 3 teacher gate can be *run* rather
than budgeted for.

## What the gap used to be

`demo_ckpts/` ships **tracker + generator only**. The refiner is trained as step 1 of
SUGAR's own `train.sh`:

```
--task Sugar-G129dof-CarryBox-Refiner --num_envs 4096 --max_iterations 30001
cp .../logs/refiner/model_30000.pt .../ckpts/refiner.pt
```

so `refiner_model10000.pt` was an artifact of *our* run, not a published file. The
refiner **training config is present** (`train_refiner/carry_box_refiner_env_cfg.py`,
plus the `..._anatomical_whole_hand_tacsl_audit_` variant). Options for the Phase 3
teacher gate: copy the checkpoint off the runtime host, or retrain — which Plan 16
already budgets for ("If it does not: retrain a teacher in Newton from the reference
motion").
