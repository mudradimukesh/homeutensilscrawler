#!/bin/bash
# Runs after the reparse. Gentler than the first attempt on purpose: IKEA's
# image CDN was answering 6-worker/0.15s bursts with connection resets and 30s
# timeouts, and a timeout costs far more wall clock than the delay it saves.
# Two shots per product rather than three — that is what `embed` uses anyway.
cd "$HOME/Documents/projects/interior_catalog" || exit 1
say() { echo; echo "=== $1 — $(date '+%H:%M:%S') ==="; }

say "waiting for the reparse to finish"
while pgrep -f "scripts/finish.sh|data/logs/finish.sh" >/dev/null 2>&1; do sleep 20; done

say "images — 3 workers, 0.4s, 2 per product"
python3 -m catalog images --per-product 2 --workers 3 --delay 0.4 --max-data-size 8GB

say "coverage"; python3 -m catalog coverage
say "completeness"; python3 -m catalog quality
say "disk"; python3 -m catalog du --max-data-size 8GB
say "FINISHED"
