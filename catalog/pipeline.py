"""Scrape orchestration: discover, fetch, parse, append to JSONL.

JSONL is the canonical output rather than a direct database write, so a crawl can
be re-run into a fresh schema, diffed between days, or replayed after a parser fix
without touching either site again.
"""
from __future__ import annotations

import json
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from .budget import DEFAULT_LIMIT, BudgetExceeded, DiskBudget, human, parse_size
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
    budget: DiskBudget | None = None,
    max_data_size: int | str = DEFAULT_LIMIT,
) -> dict[str, int]:
    if source_name not in SOURCES:
        raise SystemExit(f"unknown source {source_name!r}; have {', '.join(SOURCES)}")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if budget is None:
        budget = DiskBudget(out_path.parent, parse_size(max_data_size))
    if budget.enabled and budget.would_exceed():
        raise SystemExit(
            f"data directory is already at {human(budget.used)} of the "
            f"{human(budget.limit)} limit — nothing crawled. Free space "
            f"(data/cache/ and data/thumbs/ are regenerable) or raise --max-data-size."
        )

    fetcher = Fetcher(cache_dir=cache_dir, delay=delay, obey_robots=obey_robots,
                      cache_max_age=cache_max_age, budget=budget)
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

    counts = {"ok": 0, "failed": 0, "skipped": len(seen), "stopped": False}
    started = time.time()
    stop = threading.Event()

    def fetch_one(url: str):
        if stop.is_set():
            return None                     # queued work drains without fetching
        return source.scrape_one(url, force)

    # Append as results land: a crawl killed at 80% keeps its 80%.
    with open(out_path, "a", encoding="utf-8") as fh, ThreadPoolExecutor(workers) as pool:
        futures = {pool.submit(fetch_one, u): u for u in urls}
        for n, fut in enumerate(as_completed(futures), 1):
            url = futures[fut]
            try:
                product = fut.result()
            except BudgetExceeded as exc:
                log.warning("%s", exc)
                counts["stopped"] = True
                stop.set()
                for pending in futures:
                    pending.cancel()
                break
            except Exception:
                log.exception("worker died on %s", url)
                product = None

            if product is None:
                if not stop.is_set():
                    counts["failed"] += 1
            else:
                line = product.to_json() + "\n"
                encoded = len(line.encode("utf-8"))
                # The JSONL is itself one of the larger things written, so it is
                # charged to the budget rather than growing outside it.
                if budget.would_exceed(encoded):
                    log.warning("disk budget reached while writing %s", out_path.name)
                    counts["stopped"] = True
                    stop.set()
                    for pending in futures:
                        pending.cancel()
                    break
                fh.write(line)
                budget.add(encoded)
                counts["ok"] += 1

            if n % 50 == 0 or n == len(urls):
                rate = n / max(time.time() - started, 1e-6)
                fh.flush()
                log.info("%s %d/%d (%.1f/s) ok=%d failed=%d",
                         source_name, n, len(urls), rate, counts["ok"], counts["failed"])

    if counts["stopped"]:
        log.warning("%s stopped early at %s of %s — %d products kept",
                    source_name, human(budget.used), human(budget.limit), counts["ok"])
    return counts
