"""Scrape orchestration: discover, fetch, parse, append to JSONL.

JSONL is the canonical output rather than a direct database write, so a crawl can
be re-run into a fresh schema, diffed between days, or replayed after a parser fix
without touching either site again.
"""
from __future__ import annotations

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from .http import Fetcher
from .sources import SOURCES

log = logging.getLogger(__name__)


def already_scraped(path: Path) -> set[str]:
    """URLs present in an existing JSONL, so an interrupted crawl can resume."""
    seen: set[str] = set()
    if not path.exists():
        return seen
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            try:
                seen.add(json.loads(line)["url"])
            except (json.JSONDecodeError, KeyError):
                continue
    return seen


def scrape(
    source_name: str,
    out_path: str | Path,
    *,
    cache_dir: str | Path,
    limit: int | None = None,
    workers: int = 4,
    delay: float = 1.0,
    force: bool = False,
    resume: bool = True,
    reparse: bool = False,
    obey_robots: bool = True,
    cache_max_age: float | None = None,
) -> dict[str, int]:
    if source_name not in SOURCES:
        raise SystemExit(f"unknown source {source_name!r}; have {', '.join(SOURCES)}")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fetcher = Fetcher(cache_dir=cache_dir, delay=delay, obey_robots=obey_robots,
                      cache_max_age=cache_max_age)
    source = SOURCES[source_name](fetcher)

    if reparse and out_path.exists():
        # Parser changes are frequent; re-deriving from the page cache costs no
        # requests, so the old JSONL is rebuilt rather than appended to.
        out_path.unlink()
        log.info("reparse: rebuilding %s from the page cache", out_path.name)

    seen = already_scraped(out_path) if (resume and not force and not reparse) else set()
    if seen:
        log.info("resuming: %d products already in %s", len(seen), out_path.name)

    urls = [u for u in source.discover(limit=limit) if u not in seen]
    log.info("%s: %d product URLs to fetch", source_name, len(urls))

    counts = {"ok": 0, "failed": 0, "skipped": len(seen)}
    started = time.time()

    # Append as results land: a crawl killed at 80% keeps its 80%.
    with open(out_path, "a", encoding="utf-8") as fh, ThreadPoolExecutor(workers) as pool:
        futures = {pool.submit(source.scrape_one, u, force): u for u in urls}
        for n, fut in enumerate(as_completed(futures), 1):
            url = futures[fut]
            try:
                product = fut.result()
            except Exception:
                log.exception("worker died on %s", url)
                product = None
            if product is None:
                counts["failed"] += 1
            else:
                fh.write(product.to_json() + "\n")
                counts["ok"] += 1
            if n % 50 == 0 or n == len(urls):
                rate = n / max(time.time() - started, 1e-6)
                fh.flush()
                log.info("%s %d/%d (%.1f/s) ok=%d failed=%d",
                         source_name, n, len(urls), rate, counts["ok"], counts["failed"])
    return counts
