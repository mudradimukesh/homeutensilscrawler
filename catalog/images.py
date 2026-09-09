"""Download product images and record them against the catalog.

The design model consumes these, so they are stored once on disk, addressed by
content hash. Two products sharing an identical shot (common on HomeRun, where
brand assets repeat across pack sizes) share one file and one embedding.
"""
from __future__ import annotations

import hashlib
import logging
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from .budget import DiskBudget, human
from .http import Fetcher

log = logging.getLogger(__name__)

_EXT = {".jpg", ".jpeg", ".png", ".webp", ".avif", ".gif"}


def _extension(url: str, blob: bytes) -> str:
    ext = Path(urlparse(url).path).suffix.lower()
    if ext in _EXT:
        return ".jpg" if ext == ".jpeg" else ext
    if blob[:4] == b"\x89PNG":
        return ".png"
    if blob[:3] == b"\xff\xd8\xff":
        return ".jpg"
    if blob[8:12] == b"WEBP":
        return ".webp"
    return ".jpg"


def prune_unreferenced(
    conn: sqlite3.Connection, image_dir: str | Path, dry_run: bool = True
) -> dict[str, Any]:
    """Delete image files no product row points at.

    Files are content-addressed, so a file is only genuinely stranded once the
    catalogue has stopped wanting it — a download interrupted before its rows
    were committed looks identical to one that has been abandoned. Run this
    after an image pass has completed, never during one, and it is dry by
    default because the only signal that a file is wanted lives in the database.
    """
    image_dir = Path(image_dir)
    referenced = {
        Path(r["local_path"]).resolve()
        for r in conn.execute(
            "SELECT local_path FROM product_images WHERE local_path IS NOT NULL")
        if r["local_path"]
    }
    stranded, freed = [], 0
    for path in image_dir.rglob("*"):
        if not path.is_file() or path.resolve() in referenced:
            continue
        stranded.append(path)
        freed += path.stat().st_size

    if not dry_run:
        for path in stranded:
            path.unlink(missing_ok=True)
        for d in sorted(image_dir.rglob("*"), reverse=True):
            if d.is_dir() and not any(d.iterdir()):
                d.rmdir()

    return {
        "referenced": len(referenced),
        "stranded": len(stranded),
        "bytes_freed": freed,
        "deleted": not dry_run,
    }


def download_missing(
    conn: sqlite3.Connection,
    fetcher: Fetcher,
    image_dir: str | Path,
    limit: int | None = None,
    per_product: int | None = None,
    workers: int = 4,
    budget: DiskBudget | None = None,
    commit_every: int = 200,
) -> dict[str, int]:
    """Fetch every image row that has no local file yet.

    `per_product` caps how many shots per product are pulled — the first few are
    the main and context images, which is usually all a visual index needs.
    """
    image_dir = Path(image_dir)
    image_dir.mkdir(parents=True, exist_ok=True)

    sql = "SELECT product_key, position, url FROM product_images WHERE local_path IS NULL"
    if per_product is not None:
        sql += f" AND position < {int(per_product)}"
    sql += " ORDER BY product_key, position"
    if limit:
        sql += f" LIMIT {int(limit)}"
    rows = conn.execute(sql).fetchall()
    log.info("images to download: %d", len(rows))

    counts = {"downloaded": 0, "deduped": 0, "failed": 0, "stopped": False}
    stop = threading.Event()

    def work(row):
        if stop.is_set():
            return row, None, "stopped"
        blob = fetcher.get_bytes(row["url"])
        if not blob:
            return row, None, None
        digest = hashlib.sha256(blob).hexdigest()
        path = image_dir / digest[:2] / f"{digest}{_extension(row['url'], blob)}"
        existed = path.exists()
        if not existed:
            # A duplicate costs nothing new on disk, so only a genuinely new file
            # is charged to the budget — and it is charged before it is written.
            if budget is not None and budget.would_exceed(len(blob)):
                stop.set()
                return row, None, "stopped"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(blob)
            if budget is not None:
                budget.add(len(blob))
        return row, path, (digest, existed)

    # Committed in batches. A single transaction around 37k downloads means a
    # run killed at hour nine records nothing: the files are on disk, the
    # database has never heard of them, and the next run re-fetches every byte.
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for row, path, meta in pool.map(work, rows):
            if meta == "stopped":
                counts["stopped"] = True
                continue
            if path is None:
                counts["failed"] += 1
                continue
            digest, existed = meta
            counts["deduped" if existed else "downloaded"] += 1
            conn.execute(
                "UPDATE product_images SET local_path=?, sha256=? "
                "WHERE product_key=? AND position=?",
                (str(path), digest, row["product_key"], row["position"]),
            )
            done += 1
            if done % commit_every == 0:
                conn.commit()
                log.info("images %d/%d  downloaded=%d deduped=%d failed=%d",
                         done, len(rows), counts["downloaded"], counts["deduped"],
                         counts["failed"])
    conn.commit()
    if counts["stopped"] and budget is not None:
        log.warning("image download stopped at %s of %s; %d images still pending",
                    human(budget.used), human(budget.limit),
                    len(rows) - counts["downloaded"] - counts["deduped"] - counts["failed"])
    return counts
