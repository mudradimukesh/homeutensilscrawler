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
import hashlib
import logging
import math
import uuid
from datetime import datetime, timezone
import sqlite3
from pathlib import Path
from typing import Any, Iterable

from . import store
from .search import find_products, get_family
from .custom_builds import normalize_builds, material_filter
from .specs import variant_specs, PACK_NOUNS
from .embed import Embedder
from .quality import PRICE_TTL_DAYS, _is_current

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

    def __init__(self, db_path: str | Path, *, session_id: str | None = None,
                 embedder=None, use_vectors: bool = True, allow_store_only: bool = False,
                 image_root: str | Path | None = None):
        self.conn: sqlite3.Connection = store.connect(db_path)
        # Session IDs are supplied by the authenticated application, never by a model tool.
        self.session_id = session_id or uuid.uuid4().hex
        self.embedder = (embedder if embedder is not None else Embedder()) if use_vectors else None
        self.allow_store_only = allow_store_only
        self.image_root = Path(image_root or Path(db_path).parent / "images").resolve()
        with self.conn:
            self.conn.execute("INSERT OR IGNORE INTO catalog_sessions VALUES (?,?)",
                              (self.session_id, datetime.now(timezone.utc).isoformat()))

    @property
    def issued(self) -> set[str]:
        return {r[0] for r in self.conn.execute(
            "SELECT product_key FROM catalog_issued WHERE session_id=?", (self.session_id,))}

    def _issue(self, key: str) -> None:
        with self.conn:
            self.conn.execute("INSERT OR IGNORE INTO catalog_issued VALUES (?,?)",
                              (self.session_id, key))

    def register_query_image(self, image_bytes: bytes) -> str:
        """Application-only upload/crop registration. Tools receive opaque IDs, never paths."""
        if not isinstance(image_bytes, bytes) or not image_bytes or len(image_bytes) > 20 * 1024 * 1024:
            raise ValueError("image must contain 1 byte to 20 MiB")
        image_id = uuid.uuid4().hex
        with self.conn:
            self.conn.execute("INSERT INTO catalog_query_images VALUES (?,?,?)",
                              (self.session_id, image_id, image_bytes))
        return image_id

    # -- shaping -----------------------------------------------------------
    def _card(self, m) -> dict[str, Any]:
        """The compact form a model sees in a result list."""
        self._issue(m.key)
        dims = self.conn.execute("SELECT dimensions,specs,purchase_unit,pack_quantity,pack_uom,consumption_uom FROM products WHERE key=?", (m.key,)).fetchone()
        return {
            "dimensions_mm": _dims(dims[0]),
            "specifications": _loads(dims["specs"], {}),
            "purchase_unit": dims["purchase_unit"],
            "pack_quantity": dims["pack_quantity"], "pack_uom": dims["pack_uom"],
            "consumption_uom": dims["consumption_uom"],
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
        reference_image_id: str | None = None,
        render_ready_only: bool = False,
        material_kind: str | None = None,
    ) -> dict[str, Any]:
        """Find products. Returns product keys the design may then use."""
        limit = max(1, min(int(limit or 8), MAX_RESULTS))
        image_bytes = None
        if reference_image_id is not None:
            row = self.conn.execute(
                "SELECT image_bytes FROM catalog_query_images WHERE session_id=? AND image_id=?",
                (self.session_id, reference_image_id)).fetchone()
            if row is None:
                return {"error": "unknown_reference_image", "count": 0, "results": []}
            image_bytes = row[0]
        embedder = self.embedder
        if embedder is not None:
            if not self.conn.execute(
                "SELECT 1 FROM embeddings WHERE model=? AND model_version=? LIMIT 1",
                embedder.key).fetchone():
                embedder = None
        try:
            matches = find_products(
                self.conn, query_text=query, query_image=image_bytes, design_category=category,
                min_price=min_price, max_price=max_price, in_stock_only=True,
                allow_store_only=self.allow_store_only, k=limit, embedder=embedder,
                eligible_only=bool(placeable_only), collapse_families=bool(one_per_family),
                max_width_mm=max_width_mm, max_depth_mm=max_depth_mm, max_height_mm=max_height_mm,
                require_image_match=image_bytes is not None, render_ready_only=render_ready_only, fresh_only=True, material_kind=material_kind,
            )
        except (RuntimeError, ValueError, OSError) as exc:
            log.warning("catalog search failed: %s", exc)
            return {"error": "visual_search_unavailable" if image_bytes is not None else "invalid_search",
                    "count": 0, "results": []}
        used_vectors = any(any(k.startswith("clip_") for k in m.signals) for m in matches)
        return {
            "count": len(matches),
            "retrieval_mode": "hybrid" if used_vectors else "keyword",
            "results": [self._card(m) for m in matches],
        }

    def get_product(self, product_key: str, include_variants: bool = True) -> dict[str, Any]:
        """Everything known about one product, including its other sizes and colours."""
        row = self.conn.execute(
            "SELECT * FROM products WHERE key = ?", (product_key,)).fetchone()
        if row is None:
            return {"error": "not_found", "product_key": product_key}
        self._issue(product_key)

        images = [
            {"image_url": r["url"], "role": r["role"], "asset": r["sha256"] or None,
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
                            else "fresh" if row["price_current"] and _is_current(row["last_seen"], PRICE_TTL_DAYS) else "stale"),
            "member_price": _loads(row["attributes"], {}).get("member_price"),
            "availability": row["availability"],
            "observed_at": row["last_seen"],
            "dimensions_mm": _dims(row["dimensions"]),
            "stated_size": _loads(row["dimension_text"], []),
            "weight_kg": row["weight_kg"],
            # What a build needs and a finished product never does: the grade,
            # thickness and sheet size of a material, the load a fitting takes,
            # the sizes it comes in — and the difference between what a unit
            # costs, what it contains, and how it gets consumed.
            "specifications": _loads(row["specs"], {}),
            "purchase_unit": row["purchase_unit"],
            "pack_quantity": row["pack_quantity"],
            "pack_uom": row["pack_uom"],
            "consumption_uom": row["consumption_uom"],
            "materials": _loads(row["materials"], []),
            "colors": _loads(row["colors"], []),
            "placement_geometry": row["placement_geometry"],
            "placeable": bool(row["design_eligible"]),
            "missing_for_placement": _loads(row["quality_gaps"], []),
            "images": images,
            "source_url": row["url"],
        }
        purchase_variants = self.conn.execute(
            "SELECT variant_id,title,sku,price,available,options FROM product_variants WHERE product_key=? ORDER BY variant_id", (product_key,)).fetchall()
        out["purchase_variants"] = [{"variant_id": v["variant_id"], "title": v["title"], "sku": v["sku"],
                                     "price": v["price"], "available": bool(v["available"]),
                                     "options": _loads(v["options"], {}), "price_unit": row["price_unit"],
                                     "purchase_unit": row["purchase_unit"],
                                     "specifications": variant_specs(_loads(row["specs"], {}), v["title"], _loads(v["options"], {}), row["name"])} for v in purchase_variants]
        for variant in purchase_variants:
            self._issue("variant:" + json.dumps([product_key, variant["variant_id"]]))
        if include_variants and row["family_key"]:
            fam = get_family(self.conn, row["family_key"])
            out["family_variants"] = [
                {"product_key": v["key"], "variant": v.get("variant_label"),
                 "price": v["price"], "dimensions_mm": {
                     k.replace("_mm", ""): int(x) for k, x in (v.get("dimensions") or {}).items() if x}}
                for v in fam["variants"][:12]
            ]
            for v in out["family_variants"]:
                self._issue(v["product_key"])
        return out

    def price_design(self, objects: list[dict[str, Any]]) -> dict[str, Any]:
        """Cost a design. Every product_key must have come from this service."""
        if not isinstance(objects, list) or not objects:
            return {"error": "objects must be a non-empty list"}

        rejected, resolved = [], []
        for obj in objects:
            if not isinstance(obj, dict):
                rejected.append({"reason": "object must be a mapping"})
                continue
            key = obj.get("product_key")
            if not isinstance(key, str) or not key or key not in self.issued:
                # The model named a SKU the catalogue never handed it. That is a
                # fabricated product, and quoting it is the failure this
                # boundary exists to prevent.
                rejected.append({"product_key": key, "label": obj.get("label"),
                                 "reason": "product_key was not returned by this catalog session"})
                continue
            qty = obj.get("quantity", 1)
            if isinstance(qty, bool) or not isinstance(qty, (int, float)) or not math.isfinite(qty) or qty <= 0:
                rejected.append({"product_key": key, "reason": "quantity must be finite and positive"})
                continue
            row = self.conn.execute("SELECT availability,price,price_current,currency,last_seen FROM products WHERE key=?", (key,)).fetchone()
            allowed = ("in_stock", "store_only") if self.allow_store_only else ("in_stock",)
            if row is None or row["availability"] not in allowed:
                rejected.append({"product_key": key, "reason": "product is no longer available"})
                continue
            if row["price"] is None or not math.isfinite(row["price"]) or row["price"] < 0 or not row["price_current"] or not _is_current(row["last_seen"], PRICE_TTL_DAYS) or row["currency"] != "INR":
                rejected.append({"product_key": key, "reason": "current INR price required"})
                continue
            resolved.append(obj)

        priced = self._price_by_key(resolved) if resolved else {
            "currency": "INR", "subtotal": 0.0, "lines": [],
            "items_priced": 0, "items_unmatched": 0, "items_low_confidence": 0,
        }
        priced["rejected_items"] = rejected
        priced["complete"] = not rejected and priced["items_unmatched"] == 0
        if rejected:
            priced["note"] = ("some items were rejected: a design may only use product keys "
                              "returned by search_catalog or get_product")
        return priced

    def save_manifest(self, objects: list[dict[str, Any]], *, budget: float | None = None,
                      design_id: str | None = None, custom_builds: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        """Application-only immutable selection, saved before spending on rendering."""
        builds = normalize_builds([] if custom_builds is None else custom_builds)
        if not isinstance(objects, list) or not 1 <= len(objects) + len(builds) <= 12:
            raise ValueError("select between 1 and 12 furniture slots")
        if budget is not None and (isinstance(budget, bool) or not isinstance(budget, (int, float))
                                   or not math.isfinite(budget) or budget <= 0):
            raise ValueError("budget must be finite and positive")
        design_id = design_id or uuid.uuid4().hex
        request = {"objects": objects, "budget": budget}
        if builds:
            request["custom_builds"] = custom_builds
        existing = self.get_manifest(design_id)
        if existing:
            if existing["request"] != request:
                raise ValueError("design ID already belongs to another selection")
            return existing
        # The transaction pins product details, stock and prices to one database snapshot.
        with self.conn:
            self.conn.execute("BEGIN IMMEDIATE")
            quote = self.price_design(objects) if objects else self._price_by_key([])
            quote.setdefault("complete", True)
            if not quote.get("complete"):
                raise ValueError("invalid selection: " + json.dumps(quote.get("rejected_items", [])))
            builds, material_quote = self._quote_builds(builds)
            quote = self._combined_quote(quote, material_quote, bool(builds))
            if budget is not None and quote["subtotal"] > budget:
                raise ValueError("selected products exceed the room budget")
            products, assets = [], {}
            for index, obj in enumerate(objects):
                key = obj["product_key"]
                row = self.conn.execute("SELECT * FROM products WHERE key=?", (key,)).fetchone()
                variants = self.conn.execute("SELECT COUNT(*) FROM product_variants WHERE product_key=?", (key,)).fetchone()[0]
                if variants > 1:
                    raise ValueError("product has unresolved purchase variants: " + key)
                if not row["design_eligible"]:
                    raise ValueError("product lacks placement data: " + key)
                for axis in ("width", "depth", "height"):
                    limit = obj.get("max_" + axis + "_mm")
                    if limit is not None:
                        value = row[axis + "_mm"] or (row["length_mm"] if axis == "depth" else None)
                        if isinstance(limit, bool) or not isinstance(limit, (int, float)) or not math.isfinite(limit) or limit <= 0 or not value or value > limit:
                            raise ValueError("product does not satisfy " + axis + " constraint: " + key)
                candidates = self.conn.execute(
                    "SELECT local_path,sha256 FROM product_images WHERE product_key=? "
                    "AND local_path IS NOT NULL AND sha256 IS NOT NULL ORDER BY position", (key,)).fetchall()
                asset = None
                for candidate in candidates:
                    path = Path(candidate["local_path"]).resolve()
                    if not path.is_relative_to(self.image_root):
                        continue
                    try:
                        if path.stat().st_size > 20 * 1024 * 1024:
                            continue
                        blob = path.read_bytes()
                    except OSError:
                        continue
                    if hashlib.sha256(blob).hexdigest() != candidate["sha256"]:
                        continue
                    asset = candidate["sha256"]
                    assets[asset] = blob
                    break
                if asset is None:
                    raise ValueError("download product images before rendering: " + key)
                products.append({
                    "slot_id": str(index), "product_key": key, "name": row["name"],
                    "variant": row["variant_label"], "quantity": obj.get("quantity", 1),
                    "placement": obj.get("placement", obj.get("label", "")),
                    "dimensions_mm": _dims(row["dimensions"]),
                    "materials": _loads(row["materials"], []), "colors": _loads(row["colors"], []),
                    "source_url": row["url"], "asset": asset,
                    "unit_price": row["price"], "currency": row["currency"],
                    "availability": row["availability"], "observed_at": row["last_seen"],
                })
            snapshot = {"design_id": design_id, "request": request, "products": products,
                        "quote": quote, "budget": budget, "verification": None, "custom_builds": builds}
            self.conn.execute("INSERT INTO design_manifests VALUES (?,?,?,?,NULL)",
                              (design_id, self.session_id, datetime.now(timezone.utc).isoformat(),
                               json.dumps(snapshot, allow_nan=False)))
            self.conn.executemany("INSERT INTO design_assets VALUES (?,?,?)",
                                  [(design_id, asset, blob) for asset, blob in assets.items()])
        return snapshot

    def _quote_builds(self, builds, *, strict=True):
        subtotal, pending = 0.0, []
        for build in builds:
            for item in build["inputs"]:
                key = item["product_key"]
                item["product"] = None
                item["line_total"] = None
                if key is None:
                    pending.append({"slot_id": build["slot_id"], "query": item["query"], "reason": "material not sourced"})
                    continue
                check = self.price_design([{"product_key": key, "quantity": 1}])
                predicate, values = material_filter(item["role"])
                row = self.conn.execute("SELECT * FROM products WHERE key=? AND " + predicate, [key] + values).fetchone()
                variant_count = self.conn.execute("SELECT COUNT(*) FROM product_variants WHERE product_key=?", (key,)).fetchone()[0]
                variant_id = item.get("variant_id")
                variant = self.conn.execute("SELECT * FROM product_variants WHERE product_key=? AND variant_id=?", (key, variant_id)).fetchone() if variant_id else None
                variant_ok = (variant_count == 0 and not variant_id) or (
                    variant is not None and variant["available"] == 1
                    and variant["price"] is not None and math.isfinite(variant["price"]) and variant["price"] >= 0
                    and "variant:" + json.dumps([key, variant_id]) in self.issued)
                if not check.get("complete") or row is None or not variant_ok:
                    if strict:
                        raise ValueError("material must be an issued, available, unambiguous catalog input: " + key)
                    pending.append({"slot_id": build["slot_id"], "product_key": key, "reason": "material unavailable, ambiguous or no current price"})
                    continue
                unit_price = variant["price"] if variant is not None else row["price"]
                purchase_unit = row["purchase_unit"] or row["price_unit"]
                item["product"] = {"product_key": key, "variant_id": variant_id, "variant": variant["title"] if variant is not None else row["variant_label"],
                                   "name": row["name"], "unit_price": unit_price,
                                   "price_unit": row["price_unit"], "purchase_unit": purchase_unit,
                                   "pack_quantity": row["pack_quantity"], "pack_uom": row["pack_uom"], "consumption_uom": row["consumption_uom"],
                                   "specifications": variant_specs(_loads(row["specs"], {}), variant["title"], _loads(variant["options"], {}), row["name"]) if variant is not None else _loads(row["specs"], {}),
                                   "currency": row["currency"], "source_url": row["url"],
                                   "availability": row["availability"], "observed_at": row["last_seen"]}
                if item["quantity"] is None or not purchase_unit or item["purchase_unit"] != purchase_unit:
                    pending.append({"slot_id": build["slot_id"], "product_key": key, "reason": "quantity in catalog selling units needs confirmation"})
                    continue
                total_quantity = item["quantity"] * build["quantity"]
                item["total_purchase_quantity"] = math.ceil(total_quantity) if purchase_unit in PACK_NOUNS else total_quantity
                item["line_total"] = round(unit_price * item["total_purchase_quantity"], 2)
                subtotal += item["line_total"]
        return builds, {"subtotal": round(subtotal, 2), "pending": pending}

    @staticmethod
    def _combined_quote(ready, materials, has_custom):
        return {**ready, "subtotal": round(ready["subtotal"] + materials["subtotal"], 2),
                "ready_made_subtotal": ready["subtotal"], "materials_estimate_subtotal": materials["subtotal"],
                "pending_materials": materials["pending"],
                "complete": ready.get("complete", False) and not materials["pending"],
                "cost_complete": ready.get("complete", False) and not has_custom,
                "labour_included": False, "carpenter_review_required": has_custom}

    def get_manifest(self, design_id: str) -> dict[str, Any] | None:
        row = self.conn.execute("SELECT snapshot,verification FROM design_manifests "
                                "WHERE design_id=? AND session_id=?", (design_id, self.session_id)).fetchone()
        if row is None:
            return None
        result = json.loads(row[0])
        result["verification"] = json.loads(row[1]) if row[1] else None
        return result

    def finalize_manifest(self, design_id: str, verification: dict[str, Any],
                          render_bytes: bytes) -> dict[str, Any]:
        """Record an advisory visual review tied to exact output bytes and recheck stock.

        Only the application calls this after reviewing the image; it is not an LLM tool.
        A visual pass is never described as proof of exact SKU identity.
        """
        manifest = self.get_manifest(design_id)
        if manifest is None:
            raise ValueError("unknown design")
        if not isinstance(render_bytes, bytes) or not render_bytes:
            raise ValueError("render bytes required")
        expected = {p["slot_id"] for p in manifest["products"] + manifest.get("custom_builds", [])}
        items = verification.get("items", [])
        if not isinstance(items, list) or len(items) != len(expected) or any(not isinstance(x, dict) for x in items):
            raise ValueError("review must cover every selected slot once")
        if {x.get("slot_id") for x in items} != expected or any(x.get("status") not in ("match", "mismatch", "uncertain", "missing") for x in items):
            raise ValueError("invalid visual review")
        if not isinstance(verification.get("extra_objects"), list) or not isinstance(verification.get("room_preserved"), bool):
            raise ValueError("review must report extra objects and room preservation")
        with self.conn:
            self.conn.execute("BEGIN IMMEDIATE")
            objects = manifest["request"]["objects"]
            quote = self.price_design(objects) if objects else self._price_by_key([])
            quote.setdefault("complete", True)
            _, material_quote = self._quote_builds(normalize_builds(manifest["request"].get("custom_builds", [])), strict=False)
            quote = self._combined_quote(quote, material_quote, bool(manifest.get("custom_builds")))
            accepted = (all(x["status"] == "match" for x in items)
                        and not verification["extra_objects"] and verification["room_preserved"]
                        and quote.get("complete", False)
                        and (manifest["budget"] is None or quote["subtotal"] <= manifest["budget"]))
            review = {**verification, "status": "passed_visual_review" if accepted else "needs_review",
                      "render_sha256": hashlib.sha256(render_bytes).hexdigest(),
                      "current_quote": quote, "checked_at": datetime.now(timezone.utc).isoformat()}
            self.conn.execute("UPDATE design_manifests SET verification=? WHERE design_id=? AND session_id=?",
                              (json.dumps(review, allow_nan=False), design_id, self.session_id))
            self.conn.execute("INSERT OR REPLACE INTO design_assets VALUES (?,?,?)",
                              (design_id, review["render_sha256"], render_bytes))
        return self.get_manifest(design_id)

    def asset_bytes(self, design_id: str, asset: str) -> bytes:
        row = self.conn.execute("SELECT a.bytes FROM design_assets a JOIN design_manifests m "
                                "ON m.design_id=a.design_id WHERE a.design_id=? AND a.asset=? AND m.session_id=?",
                                (design_id, asset, self.session_id)).fetchone()
        if row is None:
            raise ValueError("unknown asset")
        return row[0]

    def _price_by_key(self, objects: list[dict[str, Any]]) -> dict[str, Any]:
        """Only validated exact keys reach pricing; semantic substitution is forbidden."""
        keyed = objects

        lines: list[dict[str, Any]] = []
        subtotal = 0.0
        states = {"fresh": 0.0, "stale": 0.0, "unknown": 0.0}
        for obj in keyed:
            row = self.conn.execute(
                "SELECT key,name,price,currency,price_unit,price_current,availability,last_seen,"
                "design_eligible FROM products WHERE key = ?", (obj["product_key"],)).fetchone()
            if row is None:
                lines.append({"label": obj.get("label"), "product_key": obj["product_key"],
                              "matched": None, "note": "no such product"})
                continue
            qty = float(obj.get("quantity", 1) or 1)
            unit = row["price"]
            total = round(unit * qty, 2) if unit is not None else None
            state = ("unknown" if unit is None else "fresh" if row["price_current"] and _is_current(row["last_seen"], PRICE_TTL_DAYS) else "stale")
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

        return {
            "currency": "INR",
            "subtotal": round(subtotal, 2),
            "subtotal_by_price_state": {k: round(v, 2) for k, v in states.items()},
            "items_priced": sum(1 for line in lines if line.get("line_total") is not None),
            "items_unmatched": sum(1 for line in lines if line.get("line_total") is None),
            "items_low_confidence": 0,
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
                "material_kind": {"type": "string", "enum": ["wood", "hardware", "finish"], "description": "Procurement inputs for carpenter-built furniture; also set placeable_only=false."},
                "render_ready_only": {"type": "boolean", "description": "Require downloaded catalog images for rendering."},
                "reference_image_id": {"type": "string", "description": "Image or crop ID registered by the application for this session."},
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
                            "quantity": {"type": "number", "exclusiveMinimum": 0, "description": "Default 1."},
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
