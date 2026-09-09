#!/bin/bash
# Deletes image files nothing points at — but only once the image pass has
# finished. Mid-run, a file whose row is not yet committed is indistinguishable
# from an abandoned one, so pruning early throws away work in progress.
cd "$HOME/Documents/projects/interior_catalog" || exit 1
say() { echo; echo "=== $1 — $(date '+%H:%M:%S') ==="; }

say "waiting for the image pass to finish"
while pgrep -f "catalog images|images_retry.sh" >/dev/null 2>&1; do sleep 30; done

say "before"
python3 -m catalog prune-images
say "deleting stranded files"
python3 -m catalog prune-images --delete
say "disk"
python3 -m catalog du --max-data-size 8GB
say "DONE"
