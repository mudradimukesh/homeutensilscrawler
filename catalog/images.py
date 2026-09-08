"""Download product images and record them against the catalog.

The design model consumes these, so they are stored once on disk, addressed by
content hash. Two products sharing an identical shot (common on HomeRun, where
brand assets repeat across pack sizes) share one file and one embedding.
"""
from __future__ import annotations

import hashlib
import logging
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urlparse

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


def download_missing(
    conn: sqlite3.Connection,
    fetcher: Fetcher,
    image_dir: str | Path,
    limit: int | None = None,
    per_product: int | None = None,
    workers: int = 4,
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

    counts = {"downloaded": 0, "deduped": 0, "failed": 0}

    def work(row):
        blob = fetcher.get_bytes(row["url"])
        if not blob:
            return row, None, None
        digest = hashlib.sha256(blob).hexdigest()
        path = image_dir / digest[:2] / f"{digest}{_extension(row['url'], blob)}"
        existed = path.exists()
        if not existed:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(blob)
        return row, path, (digest, existed)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for row, path, meta in pool.map(work, rows):
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
    conn.commit()
    return counts
