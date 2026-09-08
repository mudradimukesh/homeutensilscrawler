"""Turn a flat list of discovered URLs into a category-balanced crawl order.

Sitemap order is the enemy here. IKEA's is roughly reverse-alphabetical, so any
prefix of it is a single product family: our first 30-product sample came back as
16 VOXTORP kitchen drawer fronts. A crawl that stops early — at the disk budget,
at the end of a schedule window, on a dropped connection — then leaves a
catalogue that cannot furnish a room, rather than one that is merely thin.

So the plan is built before anything is fetched:

    discovered URLs -> category queues -> family interleave -> weighted rounds

Each round takes `weight` products from every category, so the shape of the
catalogue is roughly right at *every* prefix of the crawl, not only when it
finishes. Within a category, families are rotated so one range cannot fill the
queue on its own.
"""
from __future__ import annotations

import logging
from collections import OrderedDict, defaultdict
from typing import Iterable, Sequence

from .taxonomy import DEFAULT_WEIGHTS, classify_slug, coverage_targets, family_of

log = logging.getLogger(__name__)

UNCLASSIFIED = "unclassified"


def _interleave_families(urls: Sequence[str]) -> list[str]:
    """Rotate through product families so one range cannot monopolise a queue."""
    families: "OrderedDict[str, list[str]]" = OrderedDict()
    for url in urls:
        families.setdefault(family_of(url), []).append(url)

    out: list[str] = []
    queues = list(families.values())
    while queues:
        still: list[list[str]] = []
        for q in queues:
            out.append(q.pop(0))
            if q:
                still.append(q)
        queues = still
    return out


def build_queues(urls: Iterable[str]) -> dict[str, list[str]]:
    """Category queues, each already family-interleaved."""
    raw: dict[str, list[str]] = defaultdict(list)
    seen: set[str] = set()
    for url in urls:
        if url in seen:            # exact duplicates in a sitemap are common
            continue
        seen.add(url)
        raw[classify_slug(url) or UNCLASSIFIED].append(url)
    return {cat: _interleave_families(items) for cat, items in raw.items()}


def plan(
    urls: Iterable[str],
    weights: dict[str, int] | None = None,
    limit: int | None = None,
    unclassified_weight: int = 1,
) -> list[str]:
    """Weighted round-robin over category queues. Deterministic for a given input."""
    weights = dict(weights or DEFAULT_WEIGHTS)
    weights.setdefault(UNCLASSIFIED, unclassified_weight)

    queues = build_queues(urls)
    # Heaviest first so a truncated round still favours the categories that matter.
    order = sorted(queues, key=lambda c: (-weights.get(c, 1), c))
    cursors = {c: 0 for c in order}

    ordered: list[str] = []
    while True:
        emitted = 0
        for cat in order:
            take = max(weights.get(cat, 1), 0)
            if take == 0:
                continue
            q, i = queues[cat], cursors[cat]
            chunk = q[i:i + take]
            if not chunk:
                continue
            ordered.extend(chunk)
            cursors[cat] = i + len(chunk)
            emitted += len(chunk)
            if limit and len(ordered) >= limit:
                return ordered[:limit]
        if emitted == 0:
            break
    return ordered


def plan_summary(urls: Sequence[str], weights: dict[str, int] | None = None,
                 limit: int | None = None) -> dict:
    """What the plan will crawl, and in what proportion — before any request."""
    weights = dict(weights or DEFAULT_WEIGHTS)
    queues = build_queues(urls)
    ordered = plan(urls, weights, limit)
    taken: dict[str, int] = defaultdict(int)
    for url in ordered:
        taken[classify_slug(url) or UNCLASSIFIED] += 1
    return {
        "discovered": len(set(urls)),
        "planned": len(ordered),
        "categories": {
            cat: {
                "available": len(queues.get(cat, [])),
                "planned": taken.get(cat, 0),
                "weight": weights.get(cat, 1),
            }
            for cat in sorted(queues, key=lambda c: (-weights.get(c, 1), c))
        },
    }


def format_summary(summary: dict, top: int | None = None) -> str:
    lines = [f"crawl plan: {summary['planned']:,} of {summary['discovered']:,} discovered URLs",
             f"  {'category':<18}{'weight':>7}{'planned':>9}{'available':>11}"]
    rows = list(summary["categories"].items())
    for cat, d in rows[:top] if top else rows:
        lines.append(f"  {cat:<18}{d['weight']:>7}{d['planned']:>9,}{d['available']:>11,}")
    return "\n".join(lines)


def coverage(conn, weights: dict[str, int] | None = None, target_total: int | None = None) -> dict:
    """Per-category counts of what is actually in the database.

    The crawl plan is a promise; this is the audit. `design_category` here is the
    value derived from the parsed page, not the slug guess used for queueing, so
    a large gap between the two means the slug classifier needs work.
    """
    weights = dict(weights or DEFAULT_WEIGHTS)
    rows = conn.execute(
        "SELECT COALESCE(design_category, '(unclassified)') AS cat, COUNT(*) AS n "
        "FROM products GROUP BY 1"
    ).fetchall()
    have = {r["cat"]: r["n"] for r in rows}
    total = sum(have.values())
    targets = coverage_targets(weights, target_total or total)
    return {
        "total": total,
        "categories": {
            cat: {"have": have.get(cat, 0), "target": targets.get(cat, 0),
                  "weight": weights.get(cat, 1)}
            for cat in sorted(set(have) | set(targets),
                              key=lambda c: (-weights.get(c, 1), c))
        },
        "empty_categories": sorted(c for c, w in weights.items() if w >= 3 and not have.get(c)),
    }


def format_coverage(cov: dict) -> str:
    lines = [f"catalogue coverage: {cov['total']:,} products",
             f"  {'category':<18}{'weight':>7}{'have':>8}{'target':>9}"]
    for cat, d in cov["categories"].items():
        flag = "" if d["have"] >= d["target"] else "  <-- thin"
        lines.append(f"  {cat:<18}{d['weight']:>7}{d['have']:>8,}{d['target']:>9,}{flag}")
    if cov["empty_categories"]:
        lines.append("  EMPTY, and needed for room design: "
                     + ", ".join(cov["empty_categories"]))
    return "\n".join(lines)
