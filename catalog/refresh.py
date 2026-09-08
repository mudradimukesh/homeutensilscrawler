"""One scheduled refresh cycle: re-crawl, load, fetch new images, embed, report.

Different from a first crawl in two ways that matter. It deliberately re-fetches
pages instead of resuming past them — the whole point is to notice that a price
moved — and it rewrites each source's JSONL rather than appending, so the file
does not accumulate a copy of the catalogue per run.

Only one refresh may run at a time. A crawl overrunning its own schedule would
otherwise have two processes writing the same JSONL and the same SQLite file.
"""
from __future__ import annotations

import json
import logging
import os
import signal
import sqlite3
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

from . import store
from .http import Fetcher
from .images import download_missing
from .pipeline import scrape
from .sources import SOURCES

log = logging.getLogger(__name__)


class Lock:
    """A PID lock file. A lock left by a dead process is reclaimed, not obeyed."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.acquired = False

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            if self._holder_alive():
                raise SystemExit(
                    f"a refresh is already running (pid {self.path.read_text().strip()}); "
                    f"delete {self.path} if that is wrong"
                )
            log.warning("clearing stale lock from a dead process")
            self.path.unlink(missing_ok=True)
            fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, str(os.getpid()).encode())
        os.close(fd)
        self.acquired = True
        return self

    def __exit__(self, *exc):
        if self.acquired:
            self.path.unlink(missing_ok=True)

    def _holder_alive(self) -> bool:
        try:
            pid = int(self.path.read_text().strip())
        except (ValueError, OSError):
            return False
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True                      # someone else's process, assume alive
        return True


def parse_duration(text: str) -> float:
    """'45m', '12h', '3d', '0' -> seconds."""
    text = str(text).strip().lower()
    if not text:
        raise ValueError("empty duration")
    units = {"s": 1, "m": 60, "h": 3600, "d": 86400, "w": 604800}
    if text[-1] in units:
        return float(text[:-1]) * units[text[-1]]
    return float(text)


def _price_changes_since(conn: sqlite3.Connection, since: str) -> list[dict]:
    """Products whose price differs from their previous observation."""
    rows = conn.execute(
        """
        WITH recent AS (
            SELECT h.product_key, h.observed_at, h.price,
                   LAG(h.price) OVER (PARTITION BY h.product_key ORDER BY h.observed_at) AS prev
              FROM price_history h
        )
        SELECT r.product_key, r.prev, r.price, p.name, p.currency, p.url
          FROM recent r JOIN products p ON p.key = r.product_key
         WHERE r.observed_at >= ? AND r.prev IS NOT NULL AND r.prev <> r.price
         ORDER BY ABS(COALESCE(r.price,0) - r.prev) DESC
        """,
        (since,),
    ).fetchall()
    return [dict(r) for r in rows]


def refresh(
    db_path: str | Path,
    data_dir: str | Path,
    *,
    sources: list[str] | None = None,
    stale_after: float = 0.0,
    delay: float = 1.0,
    workers: int = 4,
    limit: int | None = None,
    skip_images: bool = False,
    skip_embed: bool = False,
    images_per_product: int = 3,
) -> dict:
    """Run the full cycle once and return a report."""
    data_dir = Path(data_dir)
    names = sources or sorted(SOURCES)
    started = datetime.now(timezone.utc).isoformat()
    began = time.time()

    with Lock(data_dir / "refresh.lock"):
        conn = store.connect(db_path)
        run_id = conn.execute(
            "INSERT INTO crawl_runs (source, started_at) VALUES (?,?)",
            (",".join(names), started),
        ).lastrowid
        conn.commit()

        status, error = "failed", None
        report = {
            "run_id": run_id, "started_at": started, "sources": {},
            "products_new": 0, "products_changed": 0, "pages_ok": 0, "pages_failed": 0,
            "images_downloaded": 0, "embeddings_added": 0, "price_changes": [],
        }
        try:
            for name in names:
                out = data_dir / f"{name}.jsonl"
                counts = scrape(
                    name, out, cache_dir=data_dir / "cache", limit=limit,
                    workers=workers, delay=delay,
                    # rewrite the file, and treat a page older than stale_after as
                    # needing a re-fetch
                    reparse=True, cache_max_age=stale_after,
                )
                loaded = store.load_jsonl(conn, out)
                report["sources"][name] = {**counts, **loaded}
                report["pages_ok"] += counts["ok"]
                report["pages_failed"] += counts["failed"]
                report["products_new"] += loaded["new"]
                report["products_changed"] += loaded["changed"]

            if not skip_images:
                fetcher = Fetcher(cache_dir=data_dir / "cache", delay=max(delay / 3, 0.2))
                got = download_missing(
                    conn, fetcher, data_dir / "images",
                    per_product=images_per_product, workers=workers,
                )
                report["images"] = got
                report["images_downloaded"] = got["downloaded"]

            if not skip_embed:
                report["embeddings"] = _embed_quietly(conn)
                report["embeddings_added"] = sum(
                    v for k, v in report["embeddings"].items() if k in ("text", "image")
                )

            report["price_changes"] = _price_changes_since(conn, started)
            status = "ok"
        except Exception as exc:
            log.exception("refresh failed")
            status, error = "failed", f"{exc.__class__.__name__}: {exc}"
            report["error"] = error
        finally:
            conn.execute(
                "UPDATE crawl_runs SET finished_at=?, status=?, pages_ok=?, pages_failed=?, "
                "products_new=?, products_changed=?, price_changes=?, images_downloaded=?, "
                "embeddings_added=?, error=? WHERE id=?",
                (datetime.now(timezone.utc).isoformat(), status,
                 report["pages_ok"], report["pages_failed"], report["products_new"],
                 report["products_changed"], len(report["price_changes"]),
                 report["images_downloaded"], report["embeddings_added"],
                 error, run_id),
            )
            conn.commit()

        report["status"] = status
        report["seconds"] = round(time.time() - began, 1)
        return report


def _embed_quietly(conn) -> dict:
    """Embed if CLIP is installed; a missing optional dependency is not a failure."""
    from .embed import Embedder, embed_catalog

    try:
        return embed_catalog(conn, Embedder(), images_per_product=2)
    except RuntimeError as exc:
        log.info("skipping embeddings: %s", exc)
        return {"skipped": str(exc)}


def format_report(r: dict) -> str:
    lines = [
        f"refresh #{r['run_id']} {r['status']} in {r['seconds']}s",
        f"  pages   ok={r['pages_ok']} failed={r['pages_failed']}",
        f"  products new={r['products_new']} changed={r['products_changed']}",
        f"  images  +{r['images_downloaded']}   embeddings +{r['embeddings_added']}",
    ]
    changes = r.get("price_changes") or []
    if changes:
        lines.append(f"  price changes ({len(changes)}):")
        for c in changes[:15]:
            old = "—" if c["prev"] is None else f"{c['prev']:,.0f}"
            new = "—" if c["price"] is None else f"{c['price']:,.0f}"
            lines.append(f"    {c['name'][:58]:<58} {old} -> {new}")
        if len(changes) > 15:
            lines.append(f"    … and {len(changes) - 15} more")
    if r.get("error"):
        lines.append(f"  error: {r['error']}")
    return "\n".join(lines)


def watch(interval: float, **kwargs) -> None:
    """Run refresh forever on an interval — for a container or a `screen` session.

    On macOS prefer `catalog schedule install`, which uses launchd and therefore
    survives reboots and logouts.
    """
    stop = False

    def handle(signum, frame):
        nonlocal stop
        stop = True
        print("\n  finishing the current cycle, then stopping…")

    signal.signal(signal.SIGINT, handle)
    signal.signal(signal.SIGTERM, handle)

    while not stop:
        print(format_report(refresh(**kwargs)), flush=True)
        if stop:
            break
        nxt = datetime.now() + timedelta(seconds=interval)
        print(f"  next refresh at {nxt:%Y-%m-%d %H:%M}\n", flush=True)
        slept = 0.0
        while slept < interval and not stop:
            time.sleep(min(5.0, interval - slept))
            slept += 5.0
