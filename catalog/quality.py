"""Completeness scoring: is there enough here to place this product in a room?

Two different questions get confused if a catalogue only has one notion of
"good enough". A product can be perfectly *findable* — it has a name, a price
and a photograph — while still being impossible to place, because nobody knows
how wide it is. Retrieval wants the first; a room layout and a spatial render
need the second. So `searchable` and `design_eligible` are computed separately
and stored separately.

The subtlety is that "has dimensions" is not one rule. Measured over the live
catalogue, a naive width+depth+height test marks 100% of rugs, bedding and
curtains incomplete — flat goods have no depth or height — and IKEA reports a
bed as width x *length*, never width x depth. Requiring a box of every product
would exclude exactly the categories a bedroom needs most. Each category
therefore declares a placement geometry, and each geometry declares the axes
that actually pin it down in a room.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

# An axis requirement is a tuple of acceptable column names; the first one
# present satisfies it. depth and length are interchangeable because the two
# retailers disagree about which word means the second horizontal axis.
GEOMETRY_AXES: dict[str, tuple[tuple[str, ...], ...]] = {
    # a full box: you cannot slot a wardrobe into an alcove without all three
    "box":       (("width_mm",), ("depth_mm", "length_mm"), ("height_mm",)),
    # occupies floor area; height is useful but does not block placement
    "footprint": (("width_mm",), ("depth_mm", "length_mm")),
    # flat goods — a rug, a quilt, a curtain drop
    "planar":    (("width_mm",), ("length_mm", "depth_mm")),
    # hangs from the ceiling: how wide it is, and how far down it comes
    "suspended": (("diameter_mm", "width_mm"), ("height_mm", "length_mm")),
    # stands on the floor or a table: how tall, and how much room it takes
    "standing":  (("height_mm",), ("diameter_mm", "width_mm")),
    # hangs on a wall
    "wall":      (("width_mm",), ("height_mm",)),
    # not placed in space at all — sold by weight, area or as a part
    "none":      (),
}

CATEGORY_GEOMETRY: dict[str, str] = {
    "beds": "footprint", "sofas": "footprint", "chairs": "footprint",
    "tables": "footprint", "desks": "footprint", "nightstands": "footprint",
    "storage": "footprint", "bathroom": "footprint", "outdoor": "footprint",
    "wardrobes": "box", "dressers": "box", "shelving": "box",
    "rugs": "planar", "bedding": "planar", "curtains": "planar",
    "mirrors": "wall",
    "lamps": "standing", "decor": "standing",
    "ceiling_lighting": "suspended",
    # Materials, finishes and spare parts are bought for a room but never
    # *placed* in one as an object, so they are searchable and never eligible.
    "kitchen": "none", "components": "none", "flooring": "none", "tiles": "none",
    "paint": "none", "electrical": "none", "plumbing": "none",
    "structural": "none", "hardware": "none",
}

PRICE_TTL_DAYS = 7          # quick commerce reprices constantly
MIN_RENDER_PX = 1000        # long edge that survives being composited into a room


def geometry_for(design_category: str | None) -> str:
    return CATEGORY_GEOMETRY.get(design_category or "", "footprint")


def _axis_present(dims: dict, options: tuple[str, ...]) -> bool:
    return any(dims.get(a) for a in options)


def missing_axes(dims: dict, geometry: str) -> list[str]:
    """Which required axes this product cannot supply."""
    return [
        "/".join(a.replace("_mm", "") for a in group)
        for group in GEOMETRY_AXES.get(geometry, ())
        if not _axis_present(dims or {}, group)
    ]


def _is_current(last_seen: str | None, ttl_days: int) -> bool:
    if not last_seen:
        return False
    try:
        seen = datetime.fromisoformat(last_seen)
    except ValueError:
        return False
    if seen.tzinfo is None:
        seen = seen.replace(tzinfo=timezone.utc)
    return datetime.now(timezone.utc) - seen <= timedelta(days=ttl_days)


def assess(product, last_seen: str | None = None, ttl_days: int = PRICE_TTL_DAYS) -> dict[str, Any]:
    """Completeness flags for one Product. Pure function of what is already stored."""
    dims = product.dimensions or {}
    geometry = geometry_for(product.design_category)
    gaps: list[str] = []

    absent = missing_axes(dims, geometry)
    dimensions_complete = geometry != "none" and not absent
    if geometry != "none" and absent:
        gaps.append("dimensions:" + ",".join(absent))

    images = product.images or []
    primary = images[0] if images else None
    primary_image_available = bool(primary and primary.url)
    if not primary_image_available:
        gaps.append("primary_image")
    alternate_views_count = max(len(images) - 1, 0)

    long_edge = max((i.width or 0, i.height or 0) for i in images) if images else (0, 0)
    best_px = max(long_edge) if images else 0
    if not images:
        render_asset_quality = 0
    elif best_px >= MIN_RENDER_PX and alternate_views_count >= 2:
        render_asset_quality = 3
    elif alternate_views_count >= 1:
        render_asset_quality = 2
    else:
        render_asset_quality = 1

    price_known = product.price is not None
    if not price_known:
        gaps.append("price")
    price_current = price_known and _is_current(last_seen, ttl_days)
    if price_known and not price_current:
        gaps.append("price_stale")

    stock_known = product.availability not in (None, "", "unknown")
    if not stock_known:
        gaps.append("stock")

    priced_variants = {v.price for v in (product.variants or []) if v.price is not None}
    variant_resolved = (
        (len(product.variants or []) <= 1 or len(priced_variants) <= 1)
        # A listed price range means the figure depends on a configuration the
        # design has not chosen, so the line cannot be quoted as it stands.
        and not (product.attributes or {}).get("price_varies")
    )
    if not variant_resolved:
        gaps.append("variant_ambiguous")

    material_known = bool(product.materials)
    color_known = bool(product.colors)

    custom = bool((product.attributes or {}).get("is_custom_made"))
    placement_safe = dimensions_complete and not custom
    if custom:
        gaps.append("custom_made")

    searchable = bool(product.name) and primary_image_available and price_known
    design_eligible = (
        searchable and placement_safe and price_current
        and stock_known and variant_resolved
    )

    return {
        "placement_geometry": geometry,
        "dimensions_complete": int(dimensions_complete),
        "primary_image_available": int(primary_image_available),
        "alternate_views_count": alternate_views_count,
        "price_current": int(price_current),
        "stock_known": int(stock_known),
        "variant_resolved": int(variant_resolved),
        "material_known": int(material_known),
        "color_known": int(color_known),
        "placement_safe": int(placement_safe),
        "render_asset_quality": render_asset_quality,
        "searchable": int(searchable),
        "design_eligible": int(design_eligible),
        "quality_gaps": gaps,
    }


QUALITY_COLUMNS: dict[str, str] = {
    "placement_geometry": "TEXT",
    "dimensions_complete": "INTEGER DEFAULT 0",
    "primary_image_available": "INTEGER DEFAULT 0",
    "alternate_views_count": "INTEGER DEFAULT 0",
    "price_current": "INTEGER DEFAULT 0",
    "stock_known": "INTEGER DEFAULT 0",
    "variant_resolved": "INTEGER DEFAULT 0",
    "material_known": "INTEGER DEFAULT 0",
    "color_known": "INTEGER DEFAULT 0",
    "placement_safe": "INTEGER DEFAULT 0",
    "render_asset_quality": "INTEGER DEFAULT 0",
    "searchable": "INTEGER DEFAULT 0",
    "design_eligible": "INTEGER DEFAULT 0",
    "quality_gaps": "TEXT",
}
