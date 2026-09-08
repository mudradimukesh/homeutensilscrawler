"""Retrieval for the design AI, and the pricing pass that follows it.

Two entry points:

* `find_products` — hybrid retrieval. A SQL predicate narrows the catalogue
  (category, budget, stock, source), then keyword and vector rankings are fused.
  The filter runs first and exactly, because "under Rs.40,000 and in stock" is a
  constraint, not a preference — an approximate answer to it produces a quote the
  customer cannot actually buy.
* `price_design` — takes the object list the design model emits and turns it into
  a costed bill of materials.
"""
from __future__ import annotations

import json
import logging
import re
import sqlite3
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .embed import Embedder, load_matrix
from .quality import geometry_for
from .taxonomy import expand_category

log = logging.getLogger(__name__)

_WORD_RE = re.compile(r"[A-Za-z0-9]+")
RRF_K = 60          # standard reciprocal-rank-fusion damping

# Design prompts are written as phrases ("a white pendant lamp for the ceiling"),
# and in an OR query the filler words match everything with a long description.
_STOPWORDS = frozenset(
    "a an the and or of for with in on to at by from is are be this that it its"
    " some any my our your one two new nice good small large big".split()
)

# bm25 column weights, in the order products_fts declares them:
# key, name, brand, category_path, description, materials, colors, tags.
# The name is what a design label is actually trying to match; HomeRun ships
# 5 000-character SEO descriptions that otherwise swamp every query.
_BM25_WEIGHTS = (0.0, 10.0, 4.0, 3.0, 1.0, 2.0, 2.0, 1.5)


@dataclass
class Match:
    key: str
    name: str
    url: str
    price: float | None
    currency: str
    source: str
    brand: str | None = None
    design_category: str | None = None
    category: str | None = None
    availability: str = "unknown"
    price_unit: str | None = None
    product_hint: str | None = None      # source product_type, for confidence checks
    design_eligible: bool = False
    placement_geometry: str | None = None
    quality_gaps: list[str] = field(default_factory=list)
    image: str | None = None
    score: float = 0.0
    signals: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _fts_query(text: str) -> str | None:
    """FTS5 has its own operator syntax; user text must not leak into it."""
    words = [
        w for w in _WORD_RE.findall(text.lower())
        if len(w) > 1 and w not in _STOPWORDS
    ]
    return " OR ".join(f'"{w}"' for w in dict.fromkeys(words).keys()) or None


def _requires_placement(design_category: str | None) -> bool:
    """Whether products in this category are placed in a room as objects.

    Paint and cement are bought for a room but never placed in one, so demanding
    placement data of them would drop every material line from a renovation quote.
    """
    fine = expand_category(design_category) or [design_category]
    return any(geometry_for(c) != "none" for c in fine if c)


def _filter_sql(
    design_category: str | None,
    min_price: float | None,
    max_price: float | None,
    source: str | None,
    in_stock_only: bool,
    exclude: Sequence[str] | None,
    eligible_only: bool | str = False,
) -> tuple[str, list[Any]]:
    clauses, params = ["1=1"], []
    if eligible_only == "auto":
        eligible_only = _requires_placement(design_category)
    if eligible_only:
        clauses.append("design_eligible = 1")
    if design_category:
        # "lighting" must reach lamps and ceiling fittings alike, so a coarse
        # name expands to the fine categories it covers.
        fine = expand_category(design_category)
        clauses.append(f"design_category IN ({','.join('?' * len(fine))})")
        params += fine
    if min_price is not None:
        clauses.append("price >= ?")
        params.append(min_price)
    if max_price is not None:
        clauses.append("price IS NOT NULL AND price <= ?")
        params.append(max_price)
    if source:
        clauses.append("source = ?")
        params.append(source)
    if in_stock_only:
        clauses.append("availability IN ('in_stock','store_only')")
    if exclude:
        clauses.append(f"key NOT IN ({','.join('?' * len(exclude))})")
        params += list(exclude)
    return " AND ".join(clauses), params


def _rank_map(ordered_keys: Sequence[str]) -> dict[str, float]:
    return {k: 1.0 / (RRF_K + i + 1) for i, k in enumerate(ordered_keys)}


def find_products(
    conn: sqlite3.Connection,
    query_text: str | None = None,
    query_image: str | Path | bytes | None = None,
    *,
    design_category: str | None = None,
    min_price: float | None = None,
    max_price: float | None = None,
    source: str | None = None,
    in_stock_only: bool = False,
    exclude: Sequence[str] | None = None,
    k: int = 10,
    embedder: Embedder | None = None,
    candidate_pool: int = 400,
    eligible_only: bool | str = False,
) -> list[Match]:
    """Hybrid search over the catalogue. Works with keywords alone if nothing is embedded.

    `eligible_only=True` restricts to products that can actually be placed in a
    room; "auto" applies that test only to categories that are placed at all.
    """
    where, params = _filter_sql(design_category, min_price, max_price, source,
                                in_stock_only, exclude, eligible_only)
    allowed = {r["key"] for r in conn.execute(f"SELECT key FROM products WHERE {where}", params)}
    if not allowed:
        return []

    rankings: list[tuple[str, dict[str, float]]] = []

    # -- keyword ----------------------------------------------------------
    if query_text:
        fq = _fts_query(query_text)
        if fq:
            weights = ",".join(str(w) for w in _BM25_WEIGHTS)
            rows = conn.execute(
                f"SELECT key FROM products_fts WHERE products_fts MATCH ? "
                f"ORDER BY bm25(products_fts, {weights}) LIMIT ?",
                (fq, candidate_pool),
            ).fetchall()
            keys = [r["key"] for r in rows if r["key"] in allowed]
            if keys:
                rankings.append(("keyword", _rank_map(keys)))

    # -- vectors ----------------------------------------------------------
    if embedder is not None and (query_text or query_image is not None):
        model = embedder.key
        try:
            if query_text:
                qv = embedder.embed_texts([query_text])[0]
                for kind in ("text", "image"):
                    keys, mat = load_matrix(conn, kind, model)
                    if mat.size:
                        rankings.append((f"clip_{kind}", _vector_rank(keys, mat, qv, allowed, candidate_pool)))
            if query_image is not None:
                blob = query_image if isinstance(query_image, bytes) else Path(query_image)
                iv = embedder.embed_images([blob])[0]
                keys, mat = load_matrix(conn, "image", model)
                if mat.size:
                    rankings.append(("clip_image_query", _vector_rank(keys, mat, iv, allowed, candidate_pool)))
        except RuntimeError as exc:
            log.warning("vector search unavailable (%s); keyword only", exc)

    if not rankings:
        # Nothing to rank on — fall back to the cheapest in-scope products so a
        # category-only query ("any 600x600 floor tile") still returns something.
        rows = conn.execute(
            f"SELECT key FROM products WHERE {where} ORDER BY price IS NULL, price LIMIT ?",
            params + [k],
        ).fetchall()
        rankings.append(("fallback", _rank_map([r["key"] for r in rows])))

    fused: dict[str, float] = {}
    signals: dict[str, dict[str, float]] = {}
    for name, rmap in rankings:
        for key, score in rmap.items():
            fused[key] = fused.get(key, 0.0) + score
            signals.setdefault(key, {})[name] = round(score, 6)

    top = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)[:k]
    return _hydrate(conn, top, signals)


def _vector_rank(
    keys: Sequence[str], mat: np.ndarray, qv: np.ndarray, allowed: set[str], pool: int
) -> dict[str, float]:
    """Cosine similarity, best row per product, restricted to the filtered set."""
    sims = mat @ qv
    best: dict[str, float] = {}
    for key, s in zip(keys, sims):
        if key in allowed and s > best.get(key, -2.0):
            best[key] = float(s)
    ordered = sorted(best, key=best.get, reverse=True)[:pool]
    return _rank_map(ordered)


def _hydrate(conn, scored: Sequence[tuple[str, float]], signals: dict) -> list[Match]:
    if not scored:
        return []
    keys = [k for k, _ in scored]
    rows = {
        r["key"]: r
        for r in conn.execute(
            f"SELECT * FROM products WHERE key IN ({','.join('?' * len(keys))})", keys
        )
    }
    images = {}
    for r in conn.execute(
        f"SELECT product_key, url FROM product_images "
        f"WHERE product_key IN ({','.join('?' * len(keys))}) AND position = 0", keys
    ):
        images[r["product_key"]] = r["url"]

    out = []
    for key, score in scored:
        r = rows.get(key)
        if r is None:
            continue
        out.append(Match(
            key=key, name=r["name"], url=r["url"], price=r["price"],
            currency=r["currency"] or "INR", source=r["source"], brand=r["brand"],
            design_category=r["design_category"], category=r["category"],
            availability=r["availability"], price_unit=r["price_unit"],
            product_hint=r["product_type"],
            design_eligible=bool(r["design_eligible"]),
            placement_geometry=r["placement_geometry"],
            quality_gaps=json.loads(r["quality_gaps"] or "[]"),
            image=images.get(key), score=round(score, 6), signals=signals.get(key, {}),
        ))
    return out


# --------------------------------------------------------------------------
# pricing a finished design
# --------------------------------------------------------------------------
def label_confidence(label: str, m: Match) -> float:
    """How much of the design's own label is evidenced in the matched product.

    Rank fusion always returns something, so without a floor a request for a
    "brass chandelier" is quoted as whatever light fitting happened to rank
    first. This is deliberately shallow — a share of the label's content words
    found in the product's name, brand or category — because it is a veto on
    nonsense, not a second ranking signal.
    """
    tokens = [w for w in _WORD_RE.findall(label.lower()) if len(w) > 1 and w not in _STOPWORDS]
    if not tokens:
        return 0.0
    haystack = " ".join(
        x for x in (m.name, m.brand, m.category, m.design_category, m.product_hint) if x
    ).lower()
    return sum(1 for t in tokens if t in haystack) / len(tokens)


@dataclass
class LineItem:
    label: str
    quantity: float
    matched: Match | None
    unit_price: float | None
    line_total: float | None
    confidence: float = 0.0
    alternates: list[Match] = field(default_factory=list)
    note: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["matched"] = self.matched.to_dict() if self.matched else None
        d["alternates"] = [a.to_dict() for a in self.alternates]
        return d


def price_design(
    conn: sqlite3.Connection,
    items: Sequence[dict[str, Any]],
    *,
    embedder: Embedder | None = None,
    in_stock_only: bool = True,
    alternates: int = 3,
    min_confidence: float = 0.34,
    eligible_only: bool | str = "auto",
) -> dict[str, Any]:
    """Cost the object list a design produced.

    Each item is `{"label": ..., "quantity": 1, ...}` and may carry `design_category`,
    `max_price`, `source`, `crop` (a path to the cropped region of the render),
    plus any free-text hints (`color`, `material`, `style`) that sharpen the query.
    Distinct products are used for distinct objects, so a room does not come back
    priced as six copies of one chair.
    """
    lines: list[LineItem] = []
    used: list[str] = []
    total = 0.0
    unmatched = 0
    low_confidence = 0

    for raw in items:
        label = str(raw.get("label") or raw.get("name") or "").strip()
        if not label:
            continue
        qty = float(raw.get("quantity", raw.get("qty", 1)) or 1)
        hints = " ".join(
            str(raw[h]) for h in ("color", "colour", "material", "style", "notes", "description")
            if raw.get(h)
        )
        query = f"{label} {hints}".strip()

        matches = find_products(
            conn,
            query_text=query,
            query_image=raw.get("crop"),
            design_category=raw.get("design_category"),
            max_price=raw.get("max_price"),
            source=raw.get("source"),
            in_stock_only=in_stock_only,
            exclude=used,
            k=alternates + 4,
            embedder=embedder,
            eligible_only=eligible_only,
        )
        if not matches:
            # Say which wall was hit: nothing in the catalogue, or nothing in it
            # complete enough to place.
            note = "no catalogue match"
            if eligible_only:
                relaxed = find_products(
                    conn, query_text=query, design_category=raw.get("design_category"),
                    max_price=raw.get("max_price"), source=raw.get("source"),
                    in_stock_only=in_stock_only, exclude=used, k=1,
                    embedder=embedder, eligible_only=False,
                )
                if relaxed:
                    note = ("matches exist but none are complete enough to place — "
                            + ", ".join(relaxed[0].quality_gaps[:3]))
            lines.append(LineItem(label, qty, None, None, None, note=note))
            unmatched += 1
            continue

        # Take the best-ranked candidate that clears the floor, not merely the
        # best-ranked one: "cement for plastering" should reach the cement rather
        # than stop at the gypsum plaster that happened to rank above it.
        scored = [(label_confidence(label, m), m) for m in matches]
        for conf, m in scored:
            if "clip_image_query" in m.signals:
                conf = max(conf, 0.5)      # a visual match needs no lexical echo
            if conf >= min_confidence:
                best, best_conf = m, conf
                break
        else:
            low_confidence += 1
            lines.append(LineItem(
                label=label, quantity=qty, matched=None, unit_price=None, line_total=None,
                confidence=round(scored[0][0], 2), alternates=matches[:alternates],
                note="no match cleared the confidence floor; alternates listed for review",
            ))
            continue

        used.append(best.key)
        unit = best.price
        line_total = round(unit * qty, 2) if unit is not None else None
        if line_total is not None:
            total += line_total
        lines.append(LineItem(
            label=label, quantity=qty, matched=best, unit_price=unit,
            line_total=line_total, confidence=round(best_conf, 2),
            alternates=[m for m in matches if m.key != best.key][:alternates],
            note=None if unit is not None else "matched, but the source lists no price",
        ))

    return {
        "currency": "INR",
        "subtotal": round(total, 2),
        "items_priced": sum(1 for line in lines if line.line_total is not None),
        "items_unmatched": unmatched,
        "items_low_confidence": low_confidence,
        "note": "subtotal covers priced lines only; review any line without a match",
        "lines": [line.to_dict() for line in lines],
    }


def dump(obj: Any) -> str:
    return json.dumps(obj, indent=2, ensure_ascii=False)
