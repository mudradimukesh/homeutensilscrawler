"""Interior-design categories, and how to guess one from a product URL.

Stratifying a crawl needs the category *before* the page is fetched — otherwise
you have to crawl everything to decide what to crawl, which defeats the point.
Both sources put the product type in the URL slug
(`/p/zebrasaev-pendant-lamp-white-plastic-30580074/`), so a slug classifier is
enough to fill the queues. The authoritative category still comes from the parsed
page; the slug guess only decides crawl order, and `catalog coverage` reports
where the two disagree.

Categories are design-relevant rather than merchandising-shaped: a designer
furnishing a bedroom needs beds, nightstands and lighting as separate things,
and does not care that IKEA files a nightstand under a shelving department.
"""
from __future__ import annotations

import re
from typing import Iterable

# category -> (weight, keywords). Weight is how many products of that category a
# single round of the crawl takes, so higher weight means it fills up sooner.
# Tuned for room design: the things a room cannot be without outrank the things
# that merely decorate it, and spare parts come last.
INTERIOR_CATEGORIES: dict[str, tuple[int, tuple[str, ...]]] = {
    "beds": (5, (
        "bed frame", "bed", "beds", "divan", "day bed", "bunk bed", "sofa bed",
        "mattress", "headboard", "bed base", "slatted bed base", "cot", "crib",
        "bedroom furniture",
    )),
    "wardrobes": (5, ("wardrobe", "wardrobes", "wardrobe combination", "corner wardrobe")),
    "sofas": (5, (
        "sofa", "2 seat", "3 seat", "4 seat", "loveseat", "sectional",
        "chaise longue", "corner sofa", "settee",
        # Modular ranges are sold as sections; each is a real, priceable product.
        "1 seat section", "2 seat section", "3 seat section", "corner section",
        "seat module", "armrest", "footstool section",
    )),
    "nightstands": (4, ("bedside table", "nightstand", "night table")),
    "lamps": (4, (
        "table lamp", "floor lamp", "work lamp", "wall lamp", "reading lamp",
        "bedside lamp", "lamp base", "lamp shade", "lamp", "desk lamp",
    )),
    "ceiling_lighting": (4, (
        "pendant lamp", "ceiling lamp", "ceiling light", "chandelier", "spotlight",
        "downlight", "cabinet lighting", "led ceiling", "track lighting",
        "ceiling spotlight", "light bulb", "led bulb", "led lighting", "lighting",
    )),
    "rugs": (4, ("rug", "rugs", "carpet", "doormat", "door mat", "rug lowpile", "rug flatwoven")),
    "curtains": (4, ("curtain", "curtains", "blind", "blinds", "blackout", "panel curtain")),
    "tables": (4, (
        "table", "coffee table", "dining table", "side table", "console table",
        "nest of tables", "bar table",
    )),
    "bedding": (3, (
        "quilt cover", "duvet cover", "bedspread", "quilt", "duvet", "pillow",
        "pillowcase", "sheet", "fitted sheet", "blanket", "throw", "cushion",
        "cushion cover", "mattress protector",
    )),
    "dressers": (3, (
        "chest of drawers", "chest of 2 drawers", "chest of 3 drawers",
        "chest of 4 drawers", "chest of 5 drawers", "chest of 6 drawers",
        "drawer unit", "dresser", "sideboard",
    )),
    "chairs": (3, (
        "chair", "armchair", "stool", "bench", "footstool", "pouffe", "pouf",
        "swivel chair", "office chair", "rocking chair", "high chair",
    )),
    "desks": (3, ("desk", "workstation", "writing desk")),
    "shelving": (3, ("shelf", "shelving", "shelving unit", "bookcase", "wall shelf", "shelf unit")),
    "storage": (3, (
        "storage", "storage combination", "storage unit", "storage box", "basket",
        "cabinet", "chest", "trunk", "organiser", "organizer", "shoe rack",
        "shoe cabinet", "clothes rail", "hanger", "tv storage", "tv bench",
        "box with", "box set", "pegboard", "pegboard combination", "bin",
        "bin with", "waste bin", "pedal bin", "lid",
    )),
    "mirrors": (2, ("mirror", "mirrors")),
    "decor": (2, (
        "vase", "picture", "frame", "photo frame", "clock", "candle", "candlestick",
        "decoration", "ornament", "plant", "potted plant", "artificial plant",
        "artificial flower", "plant pot", "wall decoration", "poster", "art",
        "soft toy", "cuddly toy", "tealight holder", "candle holder",
    )),
    "bathroom": (2, (
        "bathroom", "towel", "shower", "wash basin", "washbasin", "toilet",
        "mirror cabinet", "bath mat", "soap", "faucet", "tap", "sanitary", "cistern",
        "wash stand", "wash basin cabinet",
    )),
    "kitchen": (1, (
        "kitchen", "worktop", "sink", "cooktop", "hob", "oven", "extractor",
        "drawer front", "door front", "cover panel", "plinth", "deco strip",
        "cabinet front", "tableware", "cookware", "pot", "pan", "mug", "plate",
        "bowl", "cutlery", "glass", "jar", "food container", "place mat", "napkin",
        "paper napkin",
        "chopping board", "dish", "tray",
    )),
    "outdoor": (1, ("outdoor", "garden", "balcony", "parasol", "patio")),

    # Fronts, panels, spare covers and kitchen-system carcasses. Real products
    # with real prices, and a large share of IKEA's catalogue — but nobody
    # designs a room out of them, so they fill last.
    "components": (1, (
        "cover panel", "cover for", "front", "fronts", "door", "doors",
        "sliding doors", "pull out", "insert", "assembly kit", "plinth",
        "deco strip", "leg", "legs", "filler", "trim", "shelf insert",
        "base cabinet", "wall cabinet", "high cabinet", "top cabinet",
        "corner base cabinet", "frame", "suspension rail", "drawer front",
    )),

    # HomeRun is construction and finishing goods. They belong in the catalogue —
    # a renovation quote needs them — but a room *design* query almost never
    # starts here, so they fill last.
    "flooring": (2, ("floor tile", "flooring", "vitrified", "laminate floor", "wooden floor")),
    "paint": (2, ("paint", "primer", "putty", "emulsion", "enamel", "distemper", "varnish")),
    "tiles": (1, ("tile", "tiles", "tile adhesive", "grout", "tile spacer", "ceramic")),
    "electrical": (1, (
        "wire", "cable", "switch", "socket", "mcb", "conduit", "casing", "fan box",
        "spot light box", "db box", "modular plate",
    )),
    "plumbing": (1, (
        "pipe", "cpvc", "upvc", "ppr", "elbow", "coupler", "reducer", "valve",
        "water tank", "thread seal", "solvent",
    )),
    "structural": (1, (
        "cement", "concrete", "steel", "tmt", "brick", "block", "sand", "aggregate",
        "gypsum", "plaster", "waterproofing", "aac", "rmc", "pop",
    )),
    "hardware": (1, (
        "hinge", "hinges", "channel", "screw", "nail", "bolt", "anchor", "fastener",
        "bracket", "saddle", "hacksaw", "blade", "handle", "lock", "hardware",
    )),
}

DEFAULT_WEIGHTS: dict[str, int] = {c: w for c, (w, _) in INTERIOR_CATEGORIES.items()}

# Checked before everything else: these phrases identify a part regardless of
# the noun that follows it. "cover for 1-seat section" is a spare cover, not a
# sofa, and letting the general matcher see "1 seat section" would file thousands
# of replacement covers into the highest-weight queue in the crawl.
_PRIORITY_RULES: list[tuple[str, str]] = sorted(
    [
        ("cover for", "components"), ("covers for", "components"),
        # IKEA also names spare covers "VIMLE cover 4-seat sofa ...", with no
        # "for". Enumerated rather than matching a bare "cover", which would
        # swallow duvet and cushion covers — real bedding, not spare parts.
        ("cover 1 seat", "components"), ("cover 2 seat", "components"),
        ("cover 3 seat", "components"), ("cover 4 seat", "components"),
        ("cover 5 seat", "components"), ("cover sofa", "components"),
        ("cover armchair", "components"), ("cover chaise", "components"),
        ("cover footstool", "components"), ("cover corner", "components"),
        ("cover headrest", "components"), ("cover armrest", "components"),
        ("shower curtain", "bathroom"),
        ("pull out", "components"), ("assembly kit", "components"),
        ("insert with", "components"), ("pair of sliding doors", "components"),
        ("sliding doors", "components"), ("base cabinet", "components"),
        ("wall cabinet", "components"), ("high cabinet", "components"),
        ("top cabinet", "components"), ("corner base", "components"),
        ("base cab", "components"), ("wall cab", "components"),
        ("high cab", "components"), ("hi cab", "components"),
        ("base cb", "components"), ("bc", "components"),
        ("cover panel", "components"), ("drawer front", "components"),
        ("door front", "components"), ("suspension rail", "components"),
        ("shelf for", "components"), ("frame for", "components"),
        ("legs", "components"), ("leg", "components"),
        ("lid for", "components"), ("add on", "components"),
        ("seat shell", "components"), ("finials", "curtains"),
        ("bedroom furniture", "beds"), ("wash stand", "bathroom"),
    ],
    key=lambda kv: -len(kv[0]),
)

# Longest keyword first, so "bedside table" beats "table".
_KEYWORDS: list[tuple[str, str]] = sorted(
    ((kw, cat) for cat, (_, kws) in INTERIOR_CATEGORIES.items() for kw in kws),
    key=lambda kv: -len(kv[0]),
)

# Coarse groups keep older, blunter category names working as filters: asking for
# "lighting" should reach lamps and ceiling fittings alike.
COARSE_GROUPS: dict[str, tuple[str, ...]] = {
    "lighting": ("lamps", "ceiling_lighting"),
    "bed": ("beds",),
    "sofa": ("sofas",),
    "chair": ("chairs",),
    "table": ("tables", "desks", "nightstands"),
    "storage": ("storage", "shelving", "wardrobes", "dressers"),
    "textile": ("bedding", "curtains"),
    "rug": ("rugs",),
    "decor": ("decor", "mirrors"),
    "door_window": ("hardware",),
    "wall_finish": ("paint",),
    "tile": ("tiles", "flooring"),
    "ceiling": ("structural",),
    "kitchen": ("kitchen", "components"),
}


def expand_category(name: str | None) -> list[str]:
    """A category filter, resolved to the fine categories it covers.

    A coarse group wins over an identically named fine category: asking for
    "storage" should reach wardrobes and shelving too, and every such group
    lists itself, so nothing is lost. Over-reaching here is safe — the
    confidence floor in `price_design` rejects a bad match anyway — whereas
    under-reaching silently drops the product the designer asked for.
    """
    if not name:
        return []
    if name in COARSE_GROUPS:
        return list(COARSE_GROUPS[name])
    if name in INTERIOR_CATEGORIES:
        return [name]
    return [name]


_ID_TAIL_RE = re.compile(r"-\d{6,}$")
_SIZE_TOKEN_RE = re.compile(r"^(?:\d+[x×]\d+|\d+(?:cm|mm|m|kg|l|ml|w|pcs?)?|cm|mm|pack|set)$", re.I)


def slug_of(url: str) -> str:
    """The identifying slug of a product URL, without its numeric id."""
    path = url.rstrip("/").rsplit("/", 1)[-1]
    return _ID_TAIL_RE.sub("", path)


def _phrase(slug: str) -> str:
    return " " + slug.replace("-", " ").replace("_", " ").lower().strip() + " "


def classify_slug(url_or_slug: str) -> str | None:
    """Best-guess interior category from a URL slug, or None if nothing matches.

    Matching is on whole hyphen-separated words, so a range name like SOFIA never
    reads as a sofa.
    """
    text = _phrase(slug_of(url_or_slug) if "/" in url_or_slug else url_or_slug)
    for kw, cat in _PRIORITY_RULES:
        if f" {kw} " in text:
            return cat
    for kw, cat in _KEYWORDS:
        if f" {kw} " in text:
            return cat
    return None


def classify_text(*texts: str | None) -> str | None:
    """Category from parsed product fields, strongest field first.

    Same vocabulary as the slug classifier, so the crawl plan and the stored
    category are answering with one taxonomy and `catalog coverage` compares
    like with like. Pass title, then type, then site category, then tags: an
    earlier field outranks a later one, and within a field the longest keyword
    wins.
    """
    fields = [(i, _phrase(t.replace("/", " ").replace(",", " ")))
              for i, t in enumerate(texts) if t]
    if not fields:
        return None
    for rank, blob in fields:
        for kw, cat in _PRIORITY_RULES:
            if f" {kw} " in blob:
                return cat
    best: tuple[int, int, str] | None = None
    for rank, blob in fields:
        for kw, cat in _KEYWORDS:
            if f" {kw} " in blob:
                cand = (rank, -len(kw), cat)
                if best is None or cand < best:
                    best = cand
    return best[2] if best else None


def family_of(url_or_slug: str) -> str:
    """A crude product-family key, used to stop one range monopolising a queue.

    IKEA's first slug token is the range name (VOXTORP, MALM); HomeRun's is the
    brand. Interleaving on it is what turns twelve VOXTORP drawer fronts into one
    VOXTORP followed by eleven other things.
    """
    slug = slug_of(url_or_slug) if "/" in url_or_slug else url_or_slug
    tokens = [t for t in slug.split("-") if t and not _SIZE_TOKEN_RE.match(t)]
    return tokens[0].lower() if tokens else slug.lower()


def coverage_targets(weights: dict[str, int], total: int) -> dict[str, int]:
    """How many products of each category a crawl of `total` items should aim for."""
    active = {c: w for c, w in weights.items() if w > 0}
    denominator = sum(active.values()) or 1
    return {c: max(int(total * w / denominator), 1) for c, w in active.items()}


def classify_many(urls: Iterable[str]) -> dict[str, list[str]]:
    """Group URLs into category queues, preserving discovery order."""
    queues: dict[str, list[str]] = {}
    for url in urls:
        queues.setdefault(classify_slug(url) or "unclassified", []).append(url)
    return queues
