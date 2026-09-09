"""home-run.co — Bengaluru quick-commerce for construction & interior materials.

Shopify data behind a Next.js App Router front end. The whole Storefront product
object is already in the RSC flight payload the page ships for hydration, so we
read that instead of scraping rendered DOM: it carries variants, quantities,
bulk-price tiers and every image with dimensions, and it does not move when the
theme changes. JSON-LD is the fallback.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any, Iterable

from ..models import (
    Image, Product, Variant, axis_for, classify_design_category,
    parse_length_mm, parse_weight_kg,
)
from ..specs import size_range
from .base import Source, sitemap_locs

log = logging.getLogger(__name__)

_PUSH_RE = re.compile(r'self\.__next_f\.push\(\[1,("(?:[^"\\]|\\.)*")\]\)', re.S)
_ROW_AT_RE = re.compile(r"([0-9a-f]{1,4}):")
_ROW_START_RE = re.compile(r"(?m)^([0-9a-f]{1,4}):")
_TEXT_ROW_RE = re.compile(r"T([0-9a-f]+),", re.I)
_REF_RE = re.compile(r"^\$([0-9a-f]{1,4})$")
_TAG_RE = re.compile(r"<[^>]+>")

# What one unit of the listed price buys. A packaging noun wins outright: "Maha
# PPC Cement, 50 Kg Bag" is Rs.355 per bag, not per kilogram.
_PACK_NOUNS = (
    "bag", "box", "carton", "bundle", "roll", "set", "pack", "piece", "sheet",
    "coil", "pair", "tin", "bucket", "can",
)
# A bare measure is only the billing unit with its quantity attached — "20 Litre"
# means the price is for all twenty litres. Emitting "per litre" here would make
# the pricing pass multiply a 20 L drum by the litres a room needs.
_MEASURE_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*(kg|kgs|litre|litres|liter|liters|ltr|l|sq\.?\s?ft|sqft|sq\.?\s?m|sqm|m|ft|nos)\b"
    r"(?!\s*(?:capacity|load|rating|thick|dia|depth|width|height|length|x|\u00d7))",
    re.I,
)


def _flight_buffer(html: str) -> str:
    """Reassemble the streamed RSC payload into one string."""
    out: list[str] = []
    for chunk in _PUSH_RE.findall(html):
        try:
            out.append(json.loads(chunk))
        except json.JSONDecodeError:
            continue
    return "".join(out)


def _take_text_row(buf: str, start: int, nbytes: int) -> str:
    """Read a `T` row: React declares its length in UTF-8 bytes, not characters.

    A description containing `CO₂` is declared two longer than Python sees it, so
    counting characters walks past the next row header — and text rows, unlike
    JSON rows, are not newline-terminated, so there is nothing to resynchronise
    on. Slicing the encoded form is the only exact read.
    """
    chunk = buf[start:start + nbytes]          # every char is >= 1 byte, so this covers it
    encoded = chunk.encode("utf-8")
    if len(encoded) == nbytes:
        return chunk
    return encoded[:nbytes].decode("utf-8", "ignore")


def _flight_rows(buf: str) -> dict[str, str]:
    """Split the payload into `id -> payload` rows so `$ref` strings can resolve.

    Rows arrive as `<hex id>:<payload>`. JSON rows run to the next newline; text
    rows (`3a:T1f4,<payload>`) announce a byte length and run exactly that far,
    with no separator afterwards. Both forms are walked in one sequential pass
    rather than split on line starts, which truncated `descriptionHtml`.
    """
    rows: dict[str, str] = {}
    pos, n = 0, len(buf)
    while pos < n:
        if buf[pos] == "\n":
            pos += 1
            continue
        m = _ROW_AT_RE.match(buf, pos)
        if not m:
            nxt = _ROW_START_RE.search(buf, pos)      # line-anchored resync
            if not nxt:
                break
            pos = nxt.start()
            continue
        row_id, body_at = m.group(1), m.end()
        tm = _TEXT_ROW_RE.match(buf, body_at)
        if tm:
            text = _take_text_row(buf, tm.end(), int(tm.group(1), 16))
            rows[row_id] = text
            pos = tm.end() + len(text)
        else:
            nl = buf.find("\n", body_at)
            rows[row_id] = buf[body_at:nl if nl >= 0 else n]
            pos = (nl + 1) if nl >= 0 else n
    return rows


def _extract_object(buf: str, anchor: str) -> dict[str, Any] | None:
    """Brace-match a JSON object that starts at `anchor` inside the payload."""
    i = buf.find(anchor)
    if i < 0:
        return None
    start = buf.find("{", i)   # the anchor's own opening brace
    if start < 0:
        return None
    depth, in_str, esc = 0, False, False
    for j in range(start, len(buf)):
        c = buf[j]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
            continue
        if c == '"':
            in_str = True
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(buf[start:j + 1])
                except json.JSONDecodeError as exc:
                    log.debug("brace-matched blob is not JSON: %s", exc)
                    return None
    return None


def _resolve(value: Any, rows: dict[str, str]) -> Any:
    """Replace `$3a` placeholders with the text row they point at."""
    if isinstance(value, str):
        m = _REF_RE.match(value)
        if m:
            return rows.get(m.group(1), value)
        return value
    if isinstance(value, dict):
        return {k: _resolve(v, rows) for k, v in value.items()}
    if isinstance(value, list):
        return [_resolve(v, rows) for v in value]
    return value


def _num(v: Any) -> float | None:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _shopify_id(gid: str | None) -> str | None:
    return gid.rsplit("/", 1)[-1] if gid else None


def _ld_blocks(html: str) -> list[dict[str, Any]]:
    out = []
    for raw in re.findall(
        r'<script type="application/ld\+json"[^>]*>(.*?)</script>', html, re.S
    ):
        try:
            d = json.loads(raw)
        except json.JSONDecodeError:
            continue
        out.extend(d if isinstance(d, list) else [d])
    return out


class HomeRun(Source):
    name = "homerun"
    base_url = "https://home-run.co"

    def discover(self, limit: int | None = None) -> Iterable[str]:
        xml = self.fetcher.get(f"{self.base_url}/sitemap.xml")
        if not xml:
            log.error("could not read HomeRun sitemap")
            return
        seen: set[str] = set()
        n = 0
        for loc in sitemap_locs(xml):
            if "/products/" not in loc or loc in seen:
                continue
            seen.add(loc)
            yield loc
            n += 1
            if limit and n >= limit:
                return

    def parse(self, url: str, html: str) -> Product | None:
        buf = _flight_buffer(html)
        rows = _flight_rows(buf) if buf else {}
        obj = _extract_object(buf, '"product":{"id":"gid://shopify/Product/') if buf else None
        if obj is not None and not str(obj.get("id", "")).startswith("gid://shopify/Product/"):
            log.warning("flight payload shape changed on %s, falling back to JSON-LD", url)
            obj = None

        ld: dict[str, Any] = {}
        for block in _ld_blocks(html):
            t = block.get("@type")
            # First block of a type wins: later ones on HomeRun pages are stubs.
            if t and t not in ld:
                ld[t] = block

        if obj is None:
            return self._from_ld_only(url, ld)

        p = _resolve(obj, rows)
        prod_id = _shopify_id(p.get("id")) or url.rstrip("/").rsplit("/", 1)[-1]

        # ---- images -----------------------------------------------------
        images: list[Image] = []
        for i, n in enumerate(p.get("images", {}).get("nodes", []) or []):
            if not n.get("url"):
                continue
            images.append(
                Image(
                    url=n["url"],
                    alt=n.get("altText"),
                    width=n.get("width"),
                    height=n.get("height"),
                    role="MAIN" if i == 0 else "SUPPORT",
                    position=i,
                )
            )

        # ---- variants ---------------------------------------------------
        variants: list[Variant] = []
        for v in p.get("variants", {}).get("nodes", []) or []:
            bulk: list[dict[str, Any]] = []
            meta = (v.get("vbulkPricingMetafield") or {}).get("value")
            if meta:
                try:
                    bulk = json.loads(meta)
                except json.JSONDecodeError:
                    pass
            variants.append(
                Variant(
                    variant_id=_shopify_id(v.get("id")) or "",
                    title=v.get("title"),
                    sku=v.get("sku"),
                    price=_num((v.get("price") or {}).get("amount")),
                    compare_at_price=_num((v.get("compareAtPrice") or {}).get("amount")),
                    available=v.get("availableForSale"),
                    quantity=v.get("quantityAvailable"),
                    options={
                        o.get("name"): o.get("value")
                        for o in v.get("selectedOptions", []) or []
                        if o.get("name")
                    },
                    image_url=(v.get("image") or {}).get("url"),
                    bulk_pricing=bulk,
                )
            )

        price = _num(((p.get("priceRange") or {}).get("minVariantPrice") or {}).get("amount"))
        if price is None and variants:
            price = next((v.price for v in variants if v.price is not None), None)
        compare_at = next((v.compare_at_price for v in variants if v.compare_at_price), None)

        title = p.get("title") or ""
        description = p.get("description") or None
        desc_html = p.get("descriptionHtml") or None
        tags = [t for t in (p.get("tags") or []) if isinstance(t, str)]
        collections = [
            c.get("handle") for c in (p.get("collections") or {}).get("nodes", []) or []
            if c.get("handle")
        ]
        product_type = p.get("productType") or None

        # HomeRun sells by the bag/piece/box; the unit is stated in the title.
        price_unit = next(
            (f"per {u}" for u in _PACK_NOUNS if re.search(rf"\b{u}s?\b", title, re.I)), None
        )
        if price_unit is None:
            mm = _MEASURE_RE.search(title)
            price_unit = (
                f"per {mm.group(1)} {mm.group(2).lower()}".replace(" l", " litre")
                if mm else "each"
            )

        # ---- dimensions -------------------------------------------------
        dims: dict[str, float] = {}
        dim_text: list[str] = []
        # "Telescopic Channel, 200mm to 700mm" lists the sizes it is sold in.
        # Treating that as a measurement recorded 200 mm as the product's length
        # and made every dimensional filter on it wrong; the range is a spec.
        stated_range = size_range(title)
        for token in re.findall(r"[^,|(]*\b\d+(?:\.\d+)?\s*(?:mm|cm|m|ft|inch|in)\b[^,|)]*", title, re.I):
            if stated_range and re.search(r"\b(?:to|[-–—])\b|\d\s*[-–—]\s*\d", token):
                dim_text.append(token.strip())
                continue
            token = token.strip()
            if not token:
                continue
            dim_text.append(token)
            key = axis_for(token) or "length_mm"
            val = parse_length_mm(token)
            if val and key not in dims:
                dims[key] = val
        # 600 x 600 style tile sizes
        grid = re.search(r"(\d+(?:\.\d+)?)\s*[x×]\s*(\d+(?:\.\d+)?)\s*(mm|cm|m|ft|inch|in)\b", title, re.I)
        if grid:
            unit = grid.group(3)
            dims["width_mm"] = parse_length_mm(f"{grid.group(1)} {unit}") or dims.get("width_mm", 0)
            dims["length_mm"] = parse_length_mm(f"{grid.group(2)} {unit}") or dims.get("length_mm", 0)

        ld_product = ld.get("Product", {})
        offers = ld_product.get("offers") or {}
        breadcrumbs = [
            it.get("name")
            for it in (ld.get("BreadcrumbList", {}).get("itemListElement") or [])
            if it.get("name")
        ]

        available = p.get("availableForSale")
        availability = "in_stock" if available else ("out_of_stock" if available is False else "unknown")

        colors = sorted({
            v.options[k] for v in variants for k in v.options
            if k.lower() in ("color", "colour", "shade")
        })

        return Product(
            source=self.name,
            source_id=p.get("handle") or prod_id,
            url=url,
            name=title or ld_product.get("name") or "",
            brand=p.get("vendor") or (ld_product.get("brand") or {}).get("name"),
            product_type=product_type,
            category=breadcrumbs[-2] if len(breadcrumbs) >= 2 else (collections[0] if collections else None),
            category_path=breadcrumbs[:-1] if breadcrumbs else collections[:3],
            design_category=classify_design_category(title, product_type, breadcrumbs[-2] if len(breadcrumbs) >= 2 else None, " ".join(tags)),
            description=description,
            description_html=desc_html,
            price=price,
            compare_at_price=compare_at,
            currency=((p.get("priceRange") or {}).get("minVariantPrice") or {}).get("currencyCode", "INR"),
            price_unit=price_unit,
            availability=availability,
            sku=ld_product.get("sku") or offers.get("sku") or (variants[0].sku if variants else None),
            item_no=prod_id,
            variants=variants,
            images=images,
            dimensions={k: v for k, v in dims.items() if v},
            dimension_text=dim_text,
            weight_kg=parse_weight_kg(title),
            materials=[],
            colors=colors,
            tags=tags,
            attributes={
                "handle": p.get("handle"),
                "collections": collections,
                "options": p.get("options"),
                "bulk_pricing": variants[0].bulk_pricing if variants else [],
                "description_text": _TAG_RE.sub(" ", desc_html) if desc_html and not description else None,
            },
            raw={"shopify": p},
        )

    # ----------------------------------------------------------------------
    def _from_ld_only(self, url: str, ld: dict[str, Any]) -> Product | None:
        """Fallback when the RSC payload is missing or its shape changed."""
        pd = ld.get("Product")
        if not pd:
            log.warning("no product data on %s", url)
            return None
        offers = pd.get("offers") or {}
        img = pd.get("image")
        images = [Image(url=u, position=i) for i, u in enumerate(img if isinstance(img, list) else [img] if img else [])]
        breadcrumbs = [
            it.get("name") for it in (ld.get("BreadcrumbList", {}).get("itemListElement") or [])
            if it.get("name")
        ]
        name = pd.get("name") or ""
        return Product(
            source=self.name,
            source_id=str(pd.get("productID") or url.rstrip("/").rsplit("/", 1)[-1]),
            url=url,
            name=name,
            brand=(pd.get("brand") or {}).get("name"),
            category=breadcrumbs[-2] if len(breadcrumbs) >= 2 else None,
            category_path=breadcrumbs[:-1],
            design_category=classify_design_category(name),
            description=pd.get("description"),
            price=_num(offers.get("price")),
            currency=offers.get("priceCurrency", "INR"),
            availability="in_stock" if "InStock" in str(offers.get("availability", "")) else "unknown",
            sku=pd.get("sku"),
            images=images,
            weight_kg=parse_weight_kg(name),
            raw={"jsonld": pd},
        )
