"""The tool boundary: the only surface a language model is given.

Three functions, and nothing else. The model never sees SQL, never sees a
filesystem path, never decides what a SKU costs and never invents inventory.
The one rule that makes the rest hold: a product may only appear in a priced
design if this service handed out its key first. Keys are recorded as they are
issued and validated on the way back in, so a hallucinated SKU is rejected
rather than quoted.

Everything returned is assembled field by field. Passing database rows straight
through is how local paths, raw payloads and internal columns leak into a
prompt.
"""
from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path
from typing import Any, Iterable

from . import store
from .search import find_products, get_family, price_design as _price_design

log = logging.getLogger(__name__)

MAX_RESULTS = 24
DESCRIPTION_CHARS = 400


def _dims(raw: str | dict | None) -> dict[str, int]:
    d = raw if isinstance(raw, dict) else json.loads(raw or "{}")
    return {k.replace("_mm", ""): int(v) for k, v in d.items() if v}


def _loads(v, default):
    if isinstance(v, (list, dict)):
        return v
    try:
        return json.loads(v or "null") or default
    except (TypeError, json.JSONDecodeError):
        return default


class CatalogService:
    """Backs the tools. One instance per conversation, because it holds the ledger."""

    def __init__(self, db_path: str | Path):
        self.conn: sqlite3.Connection = store.connect(db_path)
        self.issued: set[str] = set()

    # -- shaping -----------------------------------------------------------
    def _card(self, m) -> dict[str, Any]:
        """The compact form a model sees in a result list."""
        self.issued.add(m.key)
        return {
            "product_key": m.key,
            "name": m.name,
            "family": m.family_label,
            "variant": m.variant_label,
            "brand": m.brand,
            "category": m.design_category,
            "price": m.price,
            "currency": m.currency,
            "price_unit": m.price_unit,
            "price_state": m.price_state,
            "availability": m.availability,
            "placeable": m.design_eligible,
            "other_variants": max(m.family_size - 1, 0),
            "image_url": m.image,
            "source_url": m.url,
        }

    # -- tools -------------------------------------------------------------
    def search_catalog(
        self,
        query: str | None = None,
        category: str | None = None,
        min_price: float | None = None,
        max_price: float | None = None,
        max_width_mm: float | None = None,
        max_depth_mm: float | None = None,
        max_height_mm: float | None = None,
        placeable_only: bool = True,
        one_per_family: bool = True,
        limit: int = 8,
    ) -> dict[str, Any]:
        """Find products. Returns product keys the design may then use."""
        limit = max(1, min(int(limit or 8), MAX_RESULTS))
        matches = find_products(
            self.conn, query_text=query, design_category=category,
            min_price=min_price, max_price=max_price, in_stock_only=False,
            k=limit * 3 if any((max_width_mm, max_depth_mm, max_height_mm)) else limit,
            eligible_only=bool(placeable_only), collapse_families=bool(one_per_family),
        )

        # Dimensional limits are a hard filter, not a ranking hint: a wardrobe
        # that does not fit the wall is not a worse answer, it is a wrong one.
        if any((max_width_mm, max_depth_mm, max_height_mm)):
            keys = [m.key for m in matches]
            sizes = {
                r["key"]: (r["width_mm"], r["depth_mm"], r["length_mm"], r["height_mm"])
                for r in self.conn.execute(
                    f"SELECT key,width_mm,depth_mm,length_mm,height_mm FROM products "
                    f"WHERE key IN ({','.join('?' * len(keys))})", keys)
            } if keys else {}
            kept = []
            for m in matches:
                w, d, ln, h = sizes.get(m.key, (None, None, None, None))
                depth = d or ln
                if max_width_mm and w and w > max_width_mm:
                    continue
                if max_depth_mm and depth and depth > max_depth_mm:
                    continue
                if max_height_mm and h and h > max_height_mm:
                    continue
                kept.append(m)
            matches = kept

        return {
            "count": len(matches[:limit]),
            "results": [self._card(m) for m in matches[:limit]],
        }

    def get_product(self, product_key: str, include_variants: bool = True) -> dict[str, Any]:
        """Everything known about one product, including its other sizes and colours."""
        row = self.conn.execute(
            "SELECT * FROM products WHERE key = ?", (product_key,)).fetchone()
        if row is None:
            return {"error": "not_found", "product_key": product_key}
        self.issued.add(product_key)

        images = [
            {"image_url": r["url"], "role": r["role"], "asset": (r["sha256"] or "")[:16] or None,
             "width": r["width"], "height": r["height"]}
            for r in self.conn.execute(
                "SELECT url, role, sha256, width, height FROM product_images "
                "WHERE product_key = ? ORDER BY position LIMIT 8", (product_key,))
        ]
        out: dict[str, Any] = {
            "product_key": row["key"],
            "name": row["name"],
            "family": row["family_label"],
            "variant": row["variant_label"],
            "brand": row["brand"],
            "retailer": row["source"],
            "category": row["design_category"],
            "category_path": _loads(row["category_path"], []),
            "description": (row["description"] or "")[:DESCRIPTION_CHARS] or None,
            "price": row["price"],
            "currency": row["currency"],
            "price_unit": row["price_unit"],
            "price_state": ("unknown" if row["price"] is None
                            else "fresh" if row["price_current"] else "stale"),
            "member_price": _loads(row["attributes"], {}).get("member_price"),
            "availability": row["availability"],
            "dimensions_mm": _dims(row["dimensions"]),
            "stated_size": _loads(row["dimension_text"], []),
            "weight_kg": row["weight_kg"],
            "materials": _loads(row["materials"], []),
            "colors": _loads(row["colors"], []),
            "placement_geometry": row["placement_geometry"],
            "placeable": bool(row["design_eligible"]),
            "missing_for_placement": _loads(row["quality_gaps"], []),
            "images": images,
            "source_url": row["url"],
        }
        if include_variants and row["family_key"]:
            fam = get_family(self.conn, row["family_key"])
            out["family_variants"] = [
                {"product_key": v["key"], "variant": v.get("variant_label"),
                 "price": v["price"], "dimensions_mm": {
                     k.replace("_mm", ""): int(x) for k, x in (v.get("dimensions") or {}).items() if x}}
                for v in fam["variants"][:12]
            ]
            for v in out["family_variants"]:
                self.issued.add(v["product_key"])
        return out

    def price_design(self, objects: list[dict[str, Any]]) -> dict[str, Any]:
        """Cost a design. Every product_key must have come from this service."""
        if not isinstance(objects, list) or not objects:
            return {"error": "objects must be a non-empty list"}

        rejected, resolved = [], []
        for obj in objects:
            if not isinstance(obj, dict):
                continue
            key = obj.get("product_key")
            if key and key not in self.issued:
                # The model named a SKU the catalogue never handed it. That is a
                # fabricated product, and quoting it is the failure this
                # boundary exists to prevent.
                rejected.append({"product_key": key, "label": obj.get("label"),
                                 "reason": "product_key was not returned by this catalog session"})
                continue
            resolved.append(obj)

        priced = self._price_by_key(resolved) if resolved else {
            "currency": "INR", "subtotal": 0.0, "lines": [],
            "items_priced": 0, "items_unmatched": 0, "items_low_confidence": 0,
        }
        priced["rejected_items"] = rejected
        if rejected:
            priced["note"] = ("some items were rejected: a design may only use product keys "
                              "returned by search_catalog or get_product")
        return priced

    def _price_by_key(self, objects: list[dict[str, Any]]) -> dict[str, Any]:
        """Keyed items are looked up exactly; unkeyed ones fall back to search."""
        keyed = [o for o in objects if o.get("product_key")]
        unkeyed = [o for o in objects if not o.get("product_key")]

        lines: list[dict[str, Any]] = []
        subtotal = 0.0
        states = {"fresh": 0.0, "stale": 0.0, "unknown": 0.0}
        for obj in keyed:
            row = self.conn.execute(
                "SELECT key,name,price,currency,price_unit,price_current,availability,"
                "design_eligible FROM products WHERE key = ?", (obj["product_key"],)).fetchone()
            if row is None:
                lines.append({"label": obj.get("label"), "product_key": obj["product_key"],
                              "matched": None, "note": "no such product"})
                continue
            qty = float(obj.get("quantity", 1) or 1)
            unit = row["price"]
            total = round(unit * qty, 2) if unit is not None else None
            state = ("unknown" if unit is None else "fresh" if row["price_current"] else "stale")
            if total is not None:
                subtotal += total
                states[state] += total
            lines.append({
                "label": obj.get("label") or row["name"],
                "product_key": row["key"], "name": row["name"], "quantity": qty,
                "unit_price": unit, "line_total": total, "currency": row["currency"],
                "price_unit": row["price_unit"], "price_state": state,
                "availability": row["availability"], "placeable": bool(row["design_eligible"]),
            })

        fallback = _price_design(self.conn, unkeyed, eligible_only="auto") if unkeyed else None
        if fallback:
            for line in fallback["lines"]:
                m = line.get("matched")
                if m:
                    self.issued.add(m["product_key"] if "product_key" in m else m["key"])
                lines.append(line)
            subtotal += fallback["subtotal"]
            for k, v in fallback.get("subtotal_by_price_state", {}).items():
                states[k] = states.get(k, 0.0) + v

        return {
            "currency": "INR",
            "subtotal": round(subtotal, 2),
            "subtotal_by_price_state": {k: round(v, 2) for k, v in states.items()},
            "items_priced": sum(1 for line in lines if line.get("line_total") is not None),
            "items_unmatched": sum(1 for line in lines if line.get("line_total") is None),
            "items_low_confidence": (fallback or {}).get("items_low_confidence", 0),
            "lines": lines,
        }


# --------------------------------------------------------------------------
# Tool definitions handed to the model
# --------------------------------------------------------------------------
TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "name": "search_catalog",
        "description": (
            "Search the Indian furniture and building-materials catalogue. Returns real, "
            "purchasable products with their product_key. Use this before proposing any "
            "product: a design may only contain products returned here. Results are one "
            "per model line by default, so ask for the family's other sizes with get_product."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string",
                          "description": "Free text, e.g. 'dark grey 3-seat fabric sofa'."},
                "category": {"type": "string", "description":
                             "One of: beds, sofas, wardrobes, nightstands, dressers, tables, "
                             "desks, chairs, shelving, storage, rugs, curtains, bedding, "
                             "lamps, ceiling_lighting, mirrors, decor, bathroom, kitchen, "
                             "paint, flooring, tiles, electrical, plumbing, structural."},
                "min_price": {"type": "number", "description": "INR."},
                "max_price": {"type": "number", "description": "INR."},
                "max_width_mm": {"type": "number",
                                 "description": "Hard limit — nothing wider is returned."},
                "max_depth_mm": {"type": "number", "description": "Hard limit."},
                "max_height_mm": {"type": "number", "description": "Hard limit."},
                "placeable_only": {"type": "boolean", "description":
                                   "Default true: only products with dimensions complete "
                                   "enough to place in a room. Set false for materials such "
                                   "as paint or cement, which are never placed as objects."},
                "one_per_family": {"type": "boolean", "description":
                                   "Default true: collapse variants of the same model line."},
                "limit": {"type": "integer", "description": "1-24, default 8."},
            },
            "required": [],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "get_product",
        "description": (
            "Full detail for one product_key: dimensions, materials, colours, images, "
            "price state, and every other size and colour in the same model line. Use this "
            "to pick a specific variant, e.g. the 180 cm version of a bed."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "product_key": {"type": "string",
                                "description": "A product_key returned by search_catalog."},
                "include_variants": {"type": "boolean", "description": "Default true."},
            },
            "required": ["product_key"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "price_design",
        "description": (
            "Cost a finished design. Every product_key must have been returned earlier by "
            "search_catalog or get_product; invented keys are rejected. Prices come from the "
            "catalogue — never state a price this tool did not return."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "objects": {
                    "type": "array",
                    "description": "The objects placed in the room.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "product_key": {"type": "string"},
                            "label": {"type": "string",
                                      "description": "What this is in the room, e.g. 'bedside table'."},
                            "quantity": {"type": "number", "description": "Default 1."},
                        },
                        "required": ["product_key", "label"],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["objects"],
            "additionalProperties": False,
        },
    },
]


def dispatch(service: CatalogService, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Route one tool call. Unknown names are an error, never a silent no-op."""
    if name == "search_catalog":
        return service.search_catalog(**arguments)
    if name == "get_product":
        return service.get_product(**arguments)
    if name == "price_design":
        return service.price_design(**arguments)
    return {"error": f"unknown tool {name!r}"}
