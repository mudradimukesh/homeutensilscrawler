#!/bin/bash
# Waits for the full crawl to finish, then re-derives the IKEA records from the
# page cache with the AggregateOffer / JSON-LD-dimension fixes. Costs no
# requests: --reparse rebuilds the JSONL from cached HTML.
cd "$HOME/Documents/projects/interior_catalog" || exit 1
say() { echo; echo "=== $1 — $(date '+%H:%M:%S') ==="; }

say "waiting for the crawl to finish"
while pgrep -f "data/logs/fullcrawl.sh" >/dev/null 2>&1; do sleep 60; done
say "crawl finished — re-deriving IKEA records from the page cache"

# Only IKEA needs this: the parser fixes were in its offer and dimension
# handling. Family and completeness are recomputed for both sources at load.
python3 -m catalog scrape ikea_in --reparse --workers 6 --max-data-size 8GB
python3 -m catalog load

say "coverage"
python3 -m catalog coverage
say "completeness"
python3 -m catalog quality
say "disk"
python3 -m catalog du --max-data-size 8GB
say "FINISHED"
