# Assets: what the branch does not carry, and how to get it

The branch carries all the code and a reproducible environment, but **not the assets** --
about 700 MB of robot descriptions, motion clips and checkpoints. A fresh clone therefore
fails at the first `URDF` lookup until you run the two steps below.

    bash SUGAR/_downloads/fetch_assets.sh                  # ~700 MB, checksummed
    source env/activate.sh
    python -m sugar_newton.validation.make_policy_assets   # derives the two .npz

## What is missing, and why

| path | size | needed for | hidden by |
| --- | --- | --- | --- |
| `SUGAR/descriptions/` | 137 MB | G1 URDF + collision meshes, box USDs | `SUGAR/.gitignore:77` |
| `SUGAR/data/CarryBox/` | 565 MB | the motion clips (`data_000` and friends) | `SUGAR/.gitignore:78` |
| `SUGAR/demo_ckpts/` | -- | the pretrained tracker checkpoint | `.gitignore` |
| `sugar_newton/validation/tracker_actor.npz` | 1.7 MB | the replayed policy's weights | `.gitignore` `*.npz` |
| `sugar_newton/validation/hand_hulls.npz` | 220 KB | cached hand convex hulls | `.gitignore` `*.npz` |

The first three are third-party bulk and do not belong in git. The last two are *derived*:
`make_policy_assets.py` regenerates them from `demo_ckpts`, so they are reproducible rather
than lost.

## Provenance

`SUGAR/_downloads/fetch_assets.sh` pulls three archives from the public SUGAR release by
pinned Google Drive id, records the SHA-256 of each next to it, and unpacks into `SUGAR/`.
The recorded hashes are tracked (`*.zip.sha256`), so a re-fetch that returns different bytes
is visible rather than silent -- an earlier plan shipped an unpinned teacher checkpoint, and
this is what stops a repeat.

The SUGAR tree itself is vendored at upstream commit `SUGAR/CURIOSITY_UPSTREAM_COMMIT`.

## Checking a clone

    python -c "
    from sugar_newton.validation.g1_carrybox_policy import URDF, CLIPS, BOX_USD, ACTOR_NPZ
    for p in (URDF, CLIPS, BOX_USD['small'], ACTOR_NPZ):
        print('ok ' if p.exists() else 'MISSING', p)"
