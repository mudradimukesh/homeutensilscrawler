"""Unified product schema shared by every source, plus normalisation helpers.

Everything downstream — the vector index, the design AI's product picker, and the
pricing pass — reads this shape and nothing source-specific. `raw` keeps the
original payload so a schema change never means re-crawling.
"""
from __future__ import annotations

import dataclasses
import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

# --------------------------------------------------------------------------
# design taxonomy
# --------------------------------------------------------------------------
# The design AI emits object labels ("3-seater sofa", "pendant light"). Catalog
# categories are messy and source-specific ("PPC Cement", "Desk plants"). This
# map is the join between the two: every product gets a `design_category`, and
# the pricing pass searches within it.
DESIGN_CATEGORIES: dict[str, tuple[str, ...]] = {
    "sofa": ("sofa", "settee", "couch", "loveseat", "sectional", "chaise"),
    "chair": ("chair", "armchair", "stool", "bench", "recliner", "pouffe", "pouf"),
    "table": ("table", "desk", "console", "nightstand", "bedside"),
    "bed": ("bed frame", "bed ", "mattress", "headboard", "divan"),
    "storage": (
        "wardrobe", "cabinet", "shelf", "shelving", "bookcase", "chest of drawers",
        "sideboard", "cupboard", "drawer unit", "storage",
    ),
    "lighting": (
        "lamp", "light", "lighting", "cabinet lighting", "luminaire", "chandelier",
        "pendant", "sconce", "led", "bulb", "downlight", "spotlight", "batten",
    ),
    "rug": ("rug", "carpet", "doormat", "mat"),
    "textile": (
        "curtain", "cushion", "cushion cover", "throw", "quilt", "duvet", "blanket",
        "bedsheet", "pillow", "blind",
    ),
    "decor": (
        "vase", "picture", "frame", "mirror", "clock", "candle", "ornament",
        "plant pot", "artificial plant", "potted plant", "decoration",
    ),
    "kitchen": ("kitchen", "worktop", "sink", "cooktop", "chimney", "cookware", "tableware"),
    "bathroom": (
        "bathroom", "sanitary", "washbasin", "wash basin", "faucet", "tap", "shower",
        "toilet", "closet", "cistern", "cp fitting",
    ),
    "flooring": ("floor tile", "flooring", "vitrified", "laminate", "wooden floor", "granite", "marble"),
    "tile": ("tile", "tiling", "adhesive", "grout", "spacer"),
    "paint": ("paint", "primer", "putty", "emulsion", "enamel", "distemper", "varnish", "wood coat"),
    "wall_finish": ("wallpaper", "wall panel", "cladding", "texture", "veneer", "laminate sheet"),
    "ceiling": ("false ceiling", "gypsum", "pop ", "ceiling", "drywall", "grid"),
    "door_window": ("door", "window", "hinge", "handle", "lock", "shutter", "frame", "glass"),
    "plumbing": ("pipe", "cpvc", "upvc", "ppr", "plumbing", "valve", "trap", "water tank"),
    "electrical": (
        "wire", "cable", "switch", "socket", "mcb", "db box", "conduit", "fan",
        "electrical", "modular plate",
    ),
    "structural": (
        "cement", "concrete", "steel", "tmt", "brick", "block", "sand", "aggregate",
        "rebar", "aac", "m sand", "rmc", "waterproofing", "chemical", "plaster",
    ),
    "hardware": ("hardware", "screw", "nail", "bolt", "anchor", "fastener", "channel", "bracket"),
}

_DESIGN_LOOKUP = [(cat, kw) for cat, kws in DESIGN_CATEGORIES.items() for kw in kws]


def classify_design_category(*texts: str | None) -> str | None:
    """Best-effort map from product text to a design taxonomy bucket.

    Pass fields strongest-first (title, then type, then category, then tags): an
    earlier field outranks a later one, and within a field the longest keyword
    wins, so "floor tile" beats "tile" and a cement bag filed under a
    "tiling-bulk-prices" marketing collection still classifies as structural.
    """
    fields = [(i, t.lower()) for i, t in enumerate(texts) if t]
    if not fields:
        return None
    best: tuple[int, int, str] | None = None       # (field_rank, -kw_len, cat)
    for rank, blob in fields:
        for cat, kw in _DESIGN_LOOKUP:
            if kw in blob:
                cand = (rank, -len(kw), cat)
                if best is None or cand < best:
                    best = cand
    return best[2] if best else None


# --------------------------------------------------------------------------
# dimensions
# --------------------------------------------------------------------------
_UNIT_TO_MM = {
    "mm": 1.0, "cm": 10.0, "m": 1000.0, "metre": 1000.0, "meter": 1000.0,
    "in": 25.4, "inch": 25.4, "inches": 25.4, "ft": 304.8, "feet": 304.8, "foot": 304.8,
}
# Metric first: IKEA writes '45 cm (17 ¾ ")' and we want the cm, not the inches.
_DIM_RE = re.compile(
    r"(\d+(?:[.,]\d+)?)\s*(mm|cm|metres?|meters?|m|inches|inch|in|feet|foot|ft)\b",
    re.I,
)
_WEIGHT_RE = re.compile(r"(\d+(?:[.,]\d+)?)\s*(kg|g|gram|grams|kilogram|kilograms|ton|tonne)\b", re.I)

_DIM_AXIS = {
    "width": "width_mm", "breadth": "width_mm",
    "height": "height_mm", "max height": "height_mm",
    "depth": "depth_mm",
    "length": "length_mm", "long": "length_mm",
    "diameter": "diameter_mm", "dia": "diameter_mm",
    "thickness": "thickness_mm",
    "seat width": "seat_width_mm", "seat depth": "seat_depth_mm", "seat height": "seat_height_mm",
    "bed width": "width_mm", "bed length": "length_mm",
}


def parse_length_mm(text: str | None) -> float | None:
    """First metric length in a string, in millimetres."""
    if not text:
        return None
    for m in _DIM_RE.finditer(text):
        unit = m.group(2).lower()
        if unit in ("in", "inch", "inches", "ft", "foot", "feet"):
            continue  # prefer metric; imperial handled in the fallback below
        return float(m.group(1).replace(",", ".")) * _UNIT_TO_MM[unit.rstrip("s") if unit.rstrip("s") in _UNIT_TO_MM else unit]
    m = _DIM_RE.search(text)
    if m:
        unit = m.group(2).lower()
        key = unit if unit in _UNIT_TO_MM else unit.rstrip("s")
        if key in _UNIT_TO_MM:
            return float(m.group(1).replace(",", ".")) * _UNIT_TO_MM[key]
    return None


def parse_weight_kg(text: str | None) -> float | None:
    if not text:
        return None
    m = _WEIGHT_RE.search(text)
    if not m:
        return None
    v = float(m.group(1).replace(",", "."))
    unit = m.group(2).lower()
    if unit.startswith("g") and unit != "gram" or unit in ("g", "gram", "grams"):
        return v / 1000.0
    if unit in ("ton", "tonne"):
        return v * 1000.0
    return v


def axis_for(label: str | None) -> str | None:
    """Map a measurement label ('Height of plant', 'Seat depth') onto an axis key."""
    if not label:
        return None
    low = label.lower()
    best: tuple[int, str] | None = None
    for word, key in _DIM_AXIS.items():
        if word in low and (best is None or len(word) > best[0]):
            best = (len(word), key)
    return best[1] if best else None


# --------------------------------------------------------------------------
# records
# --------------------------------------------------------------------------
@dataclass
class Image:
    url: str
    alt: str | None = None
    width: int | None = None
    height: int | None = None
    role: str | None = None      # MAIN / CONTEXT / DETAIL — context shots are the
    position: int = 0            # useful ones for a room-design model
    local_path: str | None = None
    sha256: str | None = None


@dataclass
class Variant:
    variant_id: str
    title: str | None = None
    sku: str | None = None
    price: float | None = None
    compare_at_price: float | None = None
    available: bool | None = None
    quantity: int | None = None
    options: dict[str, str] = field(default_factory=dict)
    image_url: str | None = None
    bulk_pricing: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class Product:
    source: str
    source_id: str
    url: str
    name: str

    brand: str | None = None
    product_type: str | None = None
    category: str | None = None
    category_path: list[str] = field(default_factory=list)
    design_category: str | None = None

    description: str | None = None
    description_html: str | None = None

    price: float | None = None
    compare_at_price: float | None = None
    currency: str = "INR"
    price_unit: str | None = None            # 'per bag', 'per sq ft', 'each'
    availability: str = "unknown"            # in_stock | out_of_stock | store_only | unknown
    sku: str | None = None
    item_no: str | None = None

    variants: list[Variant] = field(default_factory=list)
    images: list[Image] = field(default_factory=list)

    dimensions: dict[str, float] = field(default_factory=dict)   # *_mm keys
    dimension_text: list[str] = field(default_factory=list)
    weight_kg: float | None = None
    materials: list[str] = field(default_factory=list)
    colors: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)

    rating: float | None = None
    review_count: int | None = None

    country: str = "IN"
    attributes: dict[str, Any] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)
    scraped_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    @property
    def key(self) -> str:
        return f"{self.source}:{self.source_id}"

    def embedding_text(self) -> str:
        """The text a retrieval model sees. Ordered most- to least-distinguishing."""
        bits: list[str] = [self.name]
        if self.brand and self.brand.lower() not in self.name.lower():
            bits.append(self.brand)
        if self.product_type:
            bits.append(self.product_type)
        if self.category_path:
            bits.append(" > ".join(self.category_path))
        if self.colors:
            bits.append("colour: " + ", ".join(self.colors))
        if self.materials:
            bits.append("material: " + ", ".join(self.materials))
        if self.dimension_text:
            bits.append("size: " + "; ".join(self.dimension_text[:4]))
        if self.description:
            bits.append(self.description[:600])
        if self.tags:
            bits.append(" ".join(self.tags[:12]))
        return " | ".join(b for b in bits if b)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Product":
        d = dict(d)
        d["images"] = [Image(**i) for i in d.get("images", [])]
        d["variants"] = [Variant(**v) for v in d.get("variants", [])]
        return cls(**d)
