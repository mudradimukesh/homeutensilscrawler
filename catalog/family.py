"""Product families: the model line a SKU belongs to.

MALM bed white 140, MALM bed white 160 and MALM bed black 180 are three
legitimate SKUs and one piece of furniture. A designer asking for five beds
wants five *families*; a designer who has already chosen one and needs the
180 cm version wants a variant lookup, not another semantic search. Retrieval
that cannot tell those apart returns a wall of near-duplicates and hides the
rest of the catalogue behind them.

The family key is derived at parse time from what each retailer already states:
IKEA names a range and a type ("VIHALS" + "wardrobe"), HomeRun writes
"Brand Product, Size" and the part before the first comma is the model line.
It is source-scoped, because two retailers' ranges are never the same product.
"""
from __future__ import annotations

import re
from typing import Any

_SLUG_RE = re.compile(r"[^a-z0-9]+")
# Trailing measurements, pack sizes and finishes that distinguish a SKU from its
# siblings rather than naming a different product.
_VARIANT_TAIL_RE = re.compile(
    r"\s*[-–,]?\s*("
    r"\d+(?:[.,]\d+)?\s*[x×]\s*\d+.*"          # 105x57x200 cm
    r"|\d+(?:[.,]\d+)?\s*(?:cm|mm|m|kg|g|l|ml|ltr|litre|liter|inch|ft|nos|pcs?)\b.*"
    r"|\d+\s*(?:seat|seater|door|doors|drawer|drawers)\b.*"
    r")$",
    re.I,
)


def _slug(text: str) -> str:
    return _SLUG_RE.sub("-", (text or "").lower()).strip("-")


def _strip_variant_tail(text: str) -> str:
    previous = None
    out = (text or "").strip()
    while out and out != previous:
        previous = out
        out = _VARIANT_TAIL_RE.sub("", out).strip(" -–,")
    return out


def resolve(product) -> tuple[str, str, str]:
    """Return (family_key, family_label, variant_label) for one product."""
    attrs: dict[str, Any] = product.attributes or {}
    name = (product.name or "").strip()

    range_name = (attrs.get("range_name") or "").strip()
    type_name = (attrs.get("type_name") or "").strip()

    if range_name:
        # IKEA: the range plus the article type is the model line. Colour and
        # size are what remain, and they are exactly the variant axes.
        label = f"{range_name} {type_name}".strip() if type_name else range_name
        key_parts = [_slug(range_name), _slug(type_name) or "item"]
    else:
        # HomeRun writes "Brand Product, Size, Finish"; everything before the
        # first comma names the product, the rest describes this pack of it.
        head = name.split(",")[0]
        label = _strip_variant_tail(head) or head or name
        key_parts = [_slug(label) or _slug(name) or "unknown"]

    family_key = f"{product.source}:" + "|".join(p for p in key_parts if p)

    variant_label = _variant_label(name, label, product)
    return family_key, label, variant_label


def _variant_label(name: str, family_label: str, product) -> str:
    """What distinguishes this SKU from its siblings."""
    remainder = name
    for token in family_label.split():
        remainder = re.sub(rf"(?i)\b{re.escape(token)}\b", " ", remainder, count=1)
    remainder = re.sub(r"\s{2,}", " ", remainder).strip(" -–,·/")

    if not remainder:
        bits = list(product.colors or [])
        if product.dimension_text:
            bits.append(product.dimension_text[0])
        remainder = " · ".join(bits)
    return remainder[:120]


FAMILY_COLUMNS: dict[str, str] = {
    "family_key": "TEXT",
    "family_label": "TEXT",
    "variant_label": "TEXT",
}
