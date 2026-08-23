#!/usr/bin/env bash
set -euo pipefail

ROOT=/public/home/yanhongru/Curiosity
DEST="$ROOT/MimicKit"
REPOSITORY=https://github.com/xbpeng/MimicKit.git
COMMIT=2ed1e6c093bb0829f55d33cb4f7a1731cfe6cb69

if [[ ! -d "$DEST/.git" ]]; then
    git clone "$REPOSITORY" "$DEST"
fi

git -C "$DEST" fetch origin "$COMMIT"
git -C "$DEST" checkout --detach "$COMMIT"

observed=$(git -C "$DEST" rev-parse HEAD)
if [[ "$observed" != "$COMMIT" ]]; then
    echo "MimicKit checkout mismatch: expected $COMMIT, got $observed" >&2
    exit 1
fi

echo "official MimicKit ready at $DEST ($observed)"
