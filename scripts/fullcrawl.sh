#!/bin/bash
# Full catalogue crawl, capped at 8 GB. Sources run separately so HomeRun (a
# small operator, 610 pages) gets a gentler rate than IKEA's CDN.
cd "$HOME/Documents/projects/interior_catalog" || exit 1
say() { echo; echo "=== $1 — $(date '+%H:%M:%S') ==="; }

say "PHASE 1/5  HomeRun (~610 products, 1 req/s)"
python3 -m catalog scrape homerun --workers 3 --delay 1.0 --max-data-size 8GB

say "PHASE 2/5  IKEA India (~12,575 products, 2 req/s)"
python3 -m catalog scrape ikea_in --workers 5 --delay 0.5 --max-data-size 8GB

say "PHASE 3/5  load into SQLite"
python3 -m catalog load

say "PHASE 4/5  product images (3 per product)"
python3 -m catalog images --per-product 3 --workers 6 --delay 0.15 --max-data-size 8GB

say "PHASE 5/5  report"
python3 -m catalog du
python3 -m catalog coverage
say "DONE"
