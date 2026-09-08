"""ikea.com/in/en — IKEA India.

The product page ships several `<script type="text/hydrate">` blobs. Two matter:
one carries the sales item (price, item number, every image with alt text and
pixel size), the other carries `pageProps` (measurements, materials, care,
good-to-know, breadcrumb category). Together they are richer than the JSON-LD,
which is kept as a fallback.

Only the `prod-en-IN_*` sitemaps are read, so nothing outside the Indian
catalogue and its INR pricing can enter the database.
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
from .base import Source, sitemap_locs

log = logging.getLogger(__name__)

_HYDRATE_RE = re.compile(r'<script type="text/hydrate"[^>]*>(.*?)</script>', re.S)
_LD_RE = re.compile(r'<script type="application/ld\+json"[^>]*>(.*?)</script>', re.S)
_SITEMAP_INDEX = "https://www.ikea.com/sitemaps/sitemap.xml"
_IN_SITEMAP_RE = re.compile(r"prod-en-IN_\d+\.xml$")

# Roughly ordered by usefulness to a room-design model: a CONTEXT shot shows the
# product in a styled room, which is what an inspiration-matching embedding wants.
_IMAGE_ROLE_RANK = {
    "MAIN_PRODUCT_IMAGE": 0,
    "CONTEXT_PRODUCT_IMAGE": 1,
    "INSPIRATIONAL_IMAGE": 2,
    "FUNCTIONAL_PRODUCT_IMAGE": 3,
    "QUALITY_PRODUCT_IMAGE": 4,
    "MEASUREMENT_ILLUSTRATION": 9,
}


def _hydrate_blocks(html: str) -> list[dict[str, Any]]:
    out = []
    for raw in _HYDRATE_RE.findall(html):
        try:
            out.append(json.loads(raw))
        except json.JSONDecodeError:
            continue
    return out


def _ld_products(html: str) -> dict[str, Any]:
    found: dict[str, Any] = {}
    for raw in _LD_RE.findall(html):
        try:
            d = json.loads(raw)
        except json.JSONDecodeError:
            continue
        for item in d if isinstance(d, list) else [d]:
            if isinstance(item, dict) and item.get("@type"):
                found.setdefault(item["@type"], item)
    return found


def _walk_strings(node: Any, key: str) -> list[str]:
    """Collect every value stored under `key` anywhere in a nested structure."""
    out: list[str] = []
    stack = [node]
    while stack:
        cur = stack.pop()
        if isinstance(cur, dict):
            for k, v in cur.items():
                if k == key and isinstance(v, str) and v.strip():
                    out.append(v.strip())
                elif isinstance(v, (dict, list)):
                    stack.append(v)
        elif isinstance(cur, list):
            stack.extend(x for x in cur if isinstance(x, (dict, list)))
    return out


class IkeaIndia(Source):
    name = "ikea_in"
    base_url = "https://www.ikea.com/in/en"

    # ---- discovery -------------------------------------------------------
    def discover(self, limit: int | None = None) -> Iterable[str]:
        index = self.fetcher.get(_SITEMAP_INDEX)
        if not index:
            log.error("could not read IKEA sitemap index")
            return
        maps = [u for u in sitemap_locs(index) if _IN_SITEMAP_RE.search(u)]
        log.info("IKEA India sitemaps: %d", len(maps))
        seen: set[str] = set()
        n = 0
        for sm in maps:
            # ~50 MB each, but a stratified crawl must enumerate the whole
            # catalogue before it can balance it, so these are cached (they
            # compress to a fraction) and re-fetched on a refresh's TTL.
            xml = self.fetcher.get(sm)
            if not xml:
                continue
            for loc in sitemap_locs(xml):
                if "/in/en/p/" not in loc or loc in seen:
                    continue
                seen.add(loc)
                yield loc
                n += 1
                if limit and n >= limit:
                    return

    # ---- parsing ---------------------------------------------------------
    def parse(self, url: str, html: str) -> Product | None:
        blocks = _hydrate_blocks(html)
        item: dict[str, Any] = {}
        page: dict[str, Any] = {}
        for b in blocks:
            if not isinstance(b, dict):
                continue
            if "product" in b and isinstance(b["product"], dict) and b["product"].get("itemNo"):
                if len(json.dumps(b["product"])) > len(json.dumps(item or {})):
                    item = b["product"]
            if "pageProps" in b and isinstance(b["pageProps"], dict):
                page = b["pageProps"]

        ld = _ld_products(html)
        ld_product = ld.get("Product", {})
        if not item and not ld_product:
            log.warning("no product data on %s", url)
            return None

        item_no = item.get("itemNo") or (ld_product.get("mpn") or "").replace(".", "") or None
        if not item_no:
            m = re.search(r"-(\d{8})/?$", url)
            item_no = m.group(1) if m else url.rstrip("/").rsplit("/", 1)[-1]

        # ---- name: IKEA splits it into range name + type ------------------
        range_name = item.get("name") or ""
        type_name = item.get("typeName") or ""
        summary = page.get("productSummaryProps") or {}
        desc_line = (
            ((page.get("pipPriceModuleProps") or {}).get("productDescriptionProps") or {})
            .get("descriptionText")
        )
        measure_hint = (
            ((page.get("pipPriceModuleProps") or {}).get("productDescriptionProps") or {})
            .get("measurementLinkProps", {})
            .get("text")
        )
        full_name = ld_product.get("name") or " ".join(
            x for x in [range_name, desc_line or type_name, measure_hint] if x
        ).strip()

        # ---- images -------------------------------------------------------
        images: list[Image] = []
        for media in item.get("mediaList", []) or []:
            if media.get("type") != "image":
                continue
            c = media.get("content") or {}
            if not c.get("url"):
                continue
            images.append(
                Image(
                    url=c["url"],
                    alt=c.get("alt"),
                    width=c.get("width"),
                    height=c.get("height"),
                    role=c.get("type"),
                )
            )
        if not images:
            for img in ld_product.get("image") or []:
                if isinstance(img, dict) and img.get("contentUrl"):
                    images.append(Image(url=img["contentUrl"], role="MAIN_PRODUCT_IMAGE"))
                elif isinstance(img, str):
                    images.append(Image(url=img, role="MAIN_PRODUCT_IMAGE"))
        images.sort(key=lambda i: _IMAGE_ROLE_RANK.get(i.role or "", 5))
        for i, im in enumerate(images):
            im.position = i

        # ---- measurements --------------------------------------------------
        info = page.get("productInformationSectionProps") or {}
        mprops = info.get("measurementsProps") or {}
        dims: dict[str, float] = {}
        dim_text: list[str] = []
        for m in mprops.get("measurements", []) or []:
            label, measure = m.get("name"), m.get("measure")
            if not measure:
                continue
            dim_text.append(f"{label}: {measure}" if label else measure)
            key = axis_for(label)
            val = parse_length_mm(measure)
            if key and val and key not in dims:
                dims[key] = val

        weight = None
        pkg = item.get("packageMeasurements") or []
        if pkg and isinstance(pkg[0], dict):
            w = (pkg[0].get("weight") or {}).get("value")
            if isinstance(w, (int, float)):
                weight = float(w)
        if weight is None:
            maxm = (mprops.get("packaging", {}).get("contentProps", {}) or {}).get("maxMeasurements") or {}
            weight = parse_weight_kg(maxm.get("weightText"))

        # ---- dimensions: JSON-LD as a fallback ------------------------------
        # The measurements accordion is richer but not always present; the
        # JSON-LD block states width/height/depth in the same "105 cm (41 3/8 ")"
        # form for hundreds of products the accordion skips.
        for prop, axis in (("width", "width_mm"), ("height", "height_mm"),
                           ("depth", "depth_mm")):
            if axis in dims:
                continue
            stated = ld_product.get(prop)
            value = parse_length_mm(stated) if isinstance(stated, str) else None
            if value:
                dims[axis] = value
                dim_text.append(f"{prop.title()}: {stated}")

        # ---- materials, care, good-to-know ---------------------------------
        details = (info.get("productDetailsProps") or {}).get("accordionObject") or {}
        mat_node = (details.get("materialsAndCare") or {}).get("contentProps") or {}
        materials = sorted(set(_walk_strings(mat_node.get("materials"), "material")))
        if not materials and isinstance(ld_product.get("material"), str):
            materials = [m.strip() for m in ld_product["material"].split(",") if m.strip()]
        care = [
            {"header": c.get("header"), "texts": c.get("texts", [])}
            for c in mat_node.get("careInstructions", []) or []
        ]
        good_to_know = [
            {"name": g.get("name"), "text": g.get("text")}
            for g in ((details.get("goodToKnow") or {}).get("contentProps") or {}).get("goodToKnow", []) or []
        ]
        safety = [
            {"name": s.get("name"), "text": s.get("text")}
            for s in ((details.get("safetyAndCompliance") or {}).get("contentProps") or {}).get("safetyAndCompliance", []) or []
        ]

        # ---- category ------------------------------------------------------
        crumbs = [
            it.get("name")
            for it in (ld.get("BreadcrumbList", {}).get("itemListElement") or [])
            if it.get("name")
        ]
        category = ld_product.get("category") or (crumbs[-1] if crumbs else None)

        # ---- price & availability -------------------------------------------
        # Anything sold across a price range comes as an AggregateOffer, which
        # carries no availability of its own — the real one is in the nested
        # offer. Reading only the flat field left every PAX-style wardrobe and
        # chest marked "unknown", and unknown stock disqualifies a product from
        # being placed in a room.
        offers = ld_product.get("offers") or {}
        price_low = price_high = None
        offer_count = 1
        if str(offers.get("@type", "")).endswith("AggregateOffer"):
            price_low, price_high = offers.get("lowPrice"), offers.get("highPrice")
            offer_count = int(offers.get("offercount") or offers.get("offerCount") or 1)
            nested = offers.get("offers") or []
            offers = nested[0] if nested else {}

        # A low/high spread here is almost always an IKEA Family member price,
        # not a configuration range: every discounted page sampled carried
        # offercount 1 and priceOfferType "family". The regular price stays the
        # quoted one — it is what a customer without a membership pays — and the
        # member price is recorded beside it.
        price_module = ((page.get("pipPriceModuleProps") or {}).get("priceModuleProps") or {})
        offer_type = price_module.get("priceOfferType")
        member_price = None
        if price_low and price_high and price_low != price_high:
            try:
                member_price = min(float(price_low), float(price_high))
            except (TypeError, ValueError):
                member_price = None

        price = item.get("price")
        if not isinstance(price, (int, float)):
            try:
                price = float(offers.get("price"))
            except (TypeError, ValueError):
                price = None
        avail_raw = str(offers.get("availability", ""))
        if "InStoreOnly" in avail_raw:
            availability = "store_only"
        elif "OutOfStock" in avail_raw or "SoldOut" in avail_raw:
            availability = "out_of_stock"
        elif "InStock" in avail_raw:
            availability = "in_stock"
        else:
            availability = "unknown"

        rating_node = ld_product.get("aggregateRating") or {}
        view = page.get("viewItem") or {}

        color = ld_product.get("color")
        colors = [c.strip() for c in re.split(r"[/,]", color) if c.strip()] if color else []

        description = (
            ld_product.get("description")
            or item.get("description")
            or summary.get("description")
        )

        # Sub-products of a set (a table + 4 chairs) are the closest thing IKEA
        # has to variants; keeping them lets the pricing pass expand a set.
        variants = [
            Variant(
                variant_id=str(sp.get("itemNo") or ""),
                title=" ".join(x for x in [sp.get("name"), sp.get("typeName")] if x) or None,
                sku=sp.get("visibleItemNo"),
                price=sp.get("price") if isinstance(sp.get("price"), (int, float)) else None,
                quantity=sp.get("quantity"),
            )
            for sp in item.get("subProducts", []) or []
            if isinstance(sp, dict)
        ]

        return Product(
            source=self.name,
            source_id=str(item_no),
            url=url,
            name=full_name,
            brand="IKEA",
            product_type=type_name or desc_line,
            category=category,
            category_path=crumbs,
            design_category=classify_design_category(full_name, type_name, desc_line, category, " > ".join(crumbs)),
            description=description,
            description_html=None,
            price=float(price) if isinstance(price, (int, float)) else None,
            compare_at_price=None,
            currency=item.get("currencyCode") or offers.get("priceCurrency") or "INR",
            price_unit="each",
            availability=availability,
            sku=item.get("visibleItemNo") or ld_product.get("mpn"),
            item_no=str(item_no),
            variants=variants,
            images=images,
            dimensions=dims,
            dimension_text=dim_text,
            weight_kg=weight,
            materials=materials,
            colors=colors,
            tags=[],
            rating=float(rating_node["ratingValue"]) if rating_node.get("ratingValue") else None,
            review_count=int(rating_node["reviewCount"]) if rating_node.get("reviewCount") else None,
            attributes={
                "range_name": range_name,
                "type_name": type_name,
                "designer": ((info.get("productDetailsProps") or {}).get("productDescriptionProps") or {}).get("designerName") or None,
                "care_instructions": care,
                "good_to_know": good_to_know,
                "safety_and_compliance": safety,
                "package_measurements": pkg,
                "number_of_packages": item.get("numberOfPackages"),
                "is_custom_made": item.get("isCustomMade"),
                "buyable_online": not bool(view.get("product_not_buyable_online")),
                "member_price": member_price,
                "price_offer_type": offer_type,
                # Only a genuine multi-offer listing leaves the price unresolved.
                "price_varies": offer_count > 1,
            },
            raw={"salesItem": item, "measurements": mprops.get("measurements"), "jsonld": ld_product},
        )
