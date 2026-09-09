"""Specifications and unit-of-measure for build materials.

A finished product is fully described by its name, price and outside dimensions.
A build material is not. "Century Club Prime BWP Marine Plywood, Rs.1,974" is
unusable to someone building a wardrobe: 1,974 for a 19 mm 8x4 sheet and 1,974
for a 6 mm sheet are different offers by a factor of three, and neither the
thickness nor the sheet size is anywhere in the structured fields.

Two things are modelled here.

**Specs** — the typed facts a builder needs to choose a material: thickness,
sheet size, grade, load rating, coverage, the size range a fitting is available
in. Extracted from the title first (retailers put the distinguishing spec there)
and the description second.

**Units** — three different things that `price_unit` was conflating:

    purchase_unit    what one unit of `price` buys       a bag, a sheet, a pair
    pack_quantity    how much material that unit holds   50 (kg), 2.44x1.22 (m)
    consumption_uom  how the material is consumed        kg, sq_ft, piece

Without the separation you cannot answer "how many sheets does this wardrobe
need" or "how much adhesive", which is most of what estimating a custom build
is.
"""
from __future__ import annotations

import re
from typing import Any

# -- units -----------------------------------------------------------------
PACK_NOUNS = ("bag", "box", "carton", "bundle", "roll", "set", "pack", "sheet",
              "piece", "pair", "coil", "tin", "bucket", "can", "bottle", "drum")

# How each kind of material is consumed, which is what a quantity take-off needs.
CONSUMPTION_UOM = {
    "flooring": "sq_ft", "tiles": "sq_ft", "paint": "litre",
    "structural": "kg", "components": "piece", "hardware": "piece",
    "electrical": "metre", "plumbing": "piece",
}
SHEET_GOOD_RE = re.compile(r"\b(plywood|mdf|blockboard|block board|particle board|"
                           r"laminate|veneer|hdf|wpc board|pvc board)\b", re.I)

_NUM = r"(\d+(?:[.,]\d+)?)"
_LEN_UNIT = r"(mm|cm|m|ft|feet|foot|inch|in|\")"

# -- spec patterns ---------------------------------------------------------
_THICKNESS_RE = re.compile(rf"{_NUM}\s*mm\b(?!\s*(?:to|[-–—]))", re.I)
_THICK_WORD_RE = re.compile(rf"(?:thick(?:ness)?|thk)[\s:]*{_NUM}\s*mm", re.I)
_SHEET_RE = re.compile(rf"{_NUM}\s*[x×]\s*{_NUM}\s*{_LEN_UNIT}", re.I)
_GRADE_RE = re.compile(r"\b(BWP|BWR|MR|IS[:\s]?\d{3,4}|E0|E1|AA|A\+)\b")
_LOAD_RE = re.compile(rf"{_NUM}\s*kgs?\s*(?:capacity|load|rating|weight)", re.I)
# "200mm to 700mm", "450-600 mm" — a range of orderable sizes, NOT a length.
# The trailing \b matters: without it the "m" alternative matched the m of
# "30 to 60 minutes" in an adhesive's cure time and recorded a 30–60 metre
# fitting. A unit has to end where the word ends.
_RANGE_RE = re.compile(
    rf"{_NUM}\s*{_LEN_UNIT}?\s*(?:to|[-–—])\s*{_NUM}\s*{_LEN_UNIT}\b", re.I)
# Furniture hardware is not kilometres long; anything past this is a misread.
MAX_PLAUSIBLE_MM = 5000
_COVERAGE_RE = re.compile(
    rf"{_NUM}\s*(?:sq\.?\s?(?:ft|feet)|sqft|square feet|sq\.?\s?m)\s*(?:per|/)\s*(kg|litre|liter|l)\b", re.I)
_PACK_QTY_RE = re.compile(
    rf"{_NUM}\s*(kg|kgs|g|gm|litre|liter|ltr|l|ml|ton)\b"
    r"(?!\s*(?:capacity|load|rating|bearing|holding|max))", re.I)
_PACK_COUNT_RE = re.compile(rf"(?:set of\s*)?(\d+)\s*(?:nos?|pcs?|pieces)\b", re.I)
_WARRANTY_RE = re.compile(rf"(\d+)\s*[- ]?years?\s*(?:warranty|guarantee)", re.I)

_TO_MM = {"mm": 1.0, "cm": 10.0, "m": 1000.0, "ft": 304.8, "feet": 304.8,
          "foot": 304.8, "inch": 25.4, "in": 25.4, '"': 25.4}


def _mm(value: str, unit: str | None) -> float | None:
    try:
        n = float(str(value).replace(",", "."))
    except ValueError:
        return None
    return n * _TO_MM.get((unit or "mm").lower(), 1.0)


def size_range(text: str | None) -> tuple[float, float] | None:
    """A stated range of available sizes, in mm.

    `Telescopic Channel, 200mm to 700mm` lists the sizes the fitting is sold in.
    Reading it as a length recorded 200 mm as fact and made every dimensional
    filter on that product wrong.
    """
    if not text:
        return None
    m = _RANGE_RE.search(text)
    if not m:
        return None
    lo_v, lo_u, hi_v, hi_u = m.group(1), m.group(2), m.group(3), m.group(4)
    lo, hi = _mm(lo_v, lo_u or hi_u), _mm(hi_v, hi_u)
    if lo is None or hi is None or hi <= lo:
        return None
    if hi > MAX_PLAUSIBLE_MM:
        return None
    return lo, hi


def extract(product) -> dict[str, Any]:
    """Typed specifications for one product. Title first, description second."""
    name = product.name or ""
    body = (product.description or "")[:6000]
    specs: dict[str, Any] = {}

    rng = size_range(name) or size_range(body)
    if rng:
        specs["size_range_min_mm"], specs["size_range_max_mm"] = round(rng[0]), round(rng[1])

    for source in (name, body):
        m = _THICK_WORD_RE.search(source)
        if m and "thickness_mm" not in specs:
            specs["thickness_mm"] = float(m.group(1).replace(",", "."))
    if "thickness_mm" not in specs and SHEET_GOOD_RE.search(name):
        # On sheet goods a bare "18mm" in the title is the thickness.
        m = _THICKNESS_RE.search(name)
        if m and not rng:
            specs["thickness_mm"] = float(m.group(1).replace(",", "."))

    is_sheet_good = bool(SHEET_GOOD_RE.search(name))
    for source in ((name, body) if is_sheet_good else ()):
        m = _SHEET_RE.search(source)
        if m and "sheet_size_mm" not in specs and not rng:
            w, h = _mm(m.group(1), m.group(3)), _mm(m.group(2), m.group(3))
            if w and h and max(w, h) >= 300:      # a sheet, not a screw head
                specs["sheet_size_mm"] = [round(w), round(h)]

    title_grades = {g.upper().replace(" ", "") for g in _GRADE_RE.findall(name)}
    body_grades = {g.upper().replace(" ", "") for g in _GRADE_RE.findall(body)}
    # Retain standards from the description without mixing alternative moisture grades.
    moisture = {"MR", "BWR", "BWP"}
    if title_grades & moisture:
        body_grades -= moisture
    grades = sorted(title_grades | body_grades)
    if grades:
        specs["grade"] = grades[:3]

    for source in (name, body):
        m = _LOAD_RE.search(source)
        if m and "load_capacity_kg" not in specs:
            specs["load_capacity_kg"] = float(m.group(1).replace(",", "."))

    m = _COVERAGE_RE.search(name) or _COVERAGE_RE.search(body)
    if m:
        specs["coverage_sqft_per"] = {"value": float(m.group(1).replace(",", ".")),
                                      "per": m.group(2).lower().rstrip("s")}
    m = _WARRANTY_RE.search(name) or _WARRANTY_RE.search(body)
    if m:
        specs["warranty_years"] = int(m.group(1))
    return specs


def units(product) -> dict[str, Any]:
    """Separate what a unit costs, what it contains, and how it is consumed."""
    name = product.name or ""
    out: dict[str, Any] = {"purchase_unit": None, "pack_quantity": None,
                           "pack_uom": None, "consumption_uom": None}

    for noun in PACK_NOUNS:
        if re.search(rf"\b{noun}s?\b", name, re.I):
            out["purchase_unit"] = noun
            break
    m = _PACK_COUNT_RE.search(name)
    if m and out["purchase_unit"] in (None, "set", "pack", "box"):
        out["pack_quantity"] = float(m.group(1))
        out["pack_uom"] = "piece"
        out["purchase_unit"] = out["purchase_unit"] or "pack"

    if out["pack_quantity"] is None:
        m = _PACK_QTY_RE.search(name)
        if m:
            uom = m.group(2).lower()
            uom = {"kgs": "kg", "gm": "g", "ltr": "litre", "liter": "litre",
                   "l": "litre"}.get(uom, uom)
            out["pack_quantity"] = float(m.group(1).replace(",", "."))
            out["pack_uom"] = uom
            out["purchase_unit"] = out["purchase_unit"] or uom

    if SHEET_GOOD_RE.search(name):
        out["purchase_unit"] = out["purchase_unit"] or "sheet"
        out["consumption_uom"] = "sq_ft"
    else:
        out["consumption_uom"] = CONSUMPTION_UOM.get(
            product.design_category or "", out["pack_uom"] or "piece")

    out["purchase_unit"] = out["purchase_unit"] or "piece"
    return out


SPEC_COLUMNS: dict[str, str] = {
    "specs": "TEXT",
    "purchase_unit": "TEXT",
    "pack_quantity": "REAL",
    "pack_uom": "TEXT",
    "consumption_uom": "TEXT",
}


def variant_specs(parent: dict, title: str, options: dict, product_name: str = "") -> dict:
    """Resolve orderable thickness/size from variant options, not a family description."""
    specs = {k: v for k, v in parent.items() if k not in ("thickness_mm", "sheet_size_mm", "size_range_min_mm", "size_range_max_mm")}
    title_grades = {g.upper().replace(" ", "") for g in _GRADE_RE.findall(product_name)}
    if title_grades & {"MR", "BWR", "BWP"}:
        specs["grade"] = sorted((set(specs.get("grade", [])) - {"MR", "BWR", "BWP"}) | title_grades)
    values = [str(v) for v in options.values()] if isinstance(options, dict) else []
    source = " / ".join([title or "", *values, product_name])
    thickness = re.search(r"(?<![\d.])(\d+(?:\.\d+)?)\s*mm\b", source, re.I)
    # A fitting's 450 mm length is not its thickness. Only an explicit option
    # or an already sheet-specific parent can provide thickness.
    thick_option = next((str(v) for k, v in options.items() if "thick" in k.lower()), "") if isinstance(options, dict) else ""
    thick_match = _THICKNESS_RE.search(thick_option)
    if thick_match:
        specs["thickness_mm"] = float(thick_match.group(1))
    elif thickness and ("thickness_mm" in parent or SHEET_GOOD_RE.search(product_name)):
        specs["thickness_mm"] = float(thickness.group(1))
    size = re.search(r"(\d+(?:\.\d+)?)\s*(ft|feet|mm|cm|m|'|\")?\s*[xX×]\s*(\d+(?:\.\d+)?)\s*(ft|feet|mm|cm|m|'|\")", source)
    if size:
        unit_a, unit_b = size.group(2) or size.group(4), size.group(4)
        aliases = {"'": "ft", "\"": "inch"}
        specs["sheet_size_mm"] = [round(_mm(size.group(1), aliases.get(unit_a, unit_a))),
                                    round(_mm(size.group(3), aliases.get(unit_b, unit_b)))]
    return specs
