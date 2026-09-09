#!/bin/bash
# Images at 3 per product, then embeddings over those same 3. Sequential: both
# are heavy, and embed reads what images wrote.
cd "$HOME/Documents/projects/interior_catalog" || exit 1
say() { echo; echo "=== $1 — $(date '+%H:%M:%S') ==="; }

say "images — 3 per product"
python3 -m catalog images --per-product 3 --workers 3 --delay 0.4 --max-data-size 8GB

say "waiting for the embedding stack to finish installing"
while pgrep -f "pip install .*open_clip" >/dev/null 2>&1; do sleep 30; done

say "embed — text + 3 images per product"
python3 -m catalog embed --images-per-product 3

say "coverage";     python3 -m catalog coverage
say "completeness"; python3 -m catalog quality
say "disk";         python3 -m catalog du --max-data-size 8GB
say "DONE"
