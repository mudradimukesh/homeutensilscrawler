"""SQLite catalog store: products, variants, images, price history, embeddings.

SQLite is the default because the whole catalogue is ~13k products — small enough
that a single file with an FTS5 index and brute-force cosine over embeddings
answers in milliseconds, with no server to run. `catalog/postgres.sql` holds the
same schema for Postgres + pgvector, which is where this should move once the
catalogue is shared by more than one process. Nothing above this module knows
which one is underneath.
"""
from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .family import FAMILY_COLUMNS, resolve as resolve_family
from .models import Product
from .quality import QUALITY_COLUMNS, assess

# Everything derived on write, and therefore everything a migration must add to
# a database created before these existed.
DERIVED_COLUMNS = {**QUALITY_COLUMNS, **FAMILY_COLUMNS}

log = logging.getLogger(__name__)

SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

-- Application state survives refreshes; snapshots intentionally have no product FK.
CREATE TABLE IF NOT EXISTS catalog_sessions (
    session_id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS catalog_issued (
    session_id TEXT NOT NULL REFERENCES catalog_sessions(session_id),
    product_key TEXT NOT NULL,
    PRIMARY KEY(session_id, product_key)
);
CREATE TABLE IF NOT EXISTS catalog_query_images (
    session_id TEXT NOT NULL REFERENCES catalog_sessions(session_id),
    image_id TEXT NOT NULL,
    image_bytes BLOB NOT NULL,
    PRIMARY KEY(session_id, image_id)
);
CREATE TABLE IF NOT EXISTS design_manifests (
    design_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES catalog_sessions(session_id),
    created_at TEXT NOT NULL,
    snapshot TEXT NOT NULL,
    verification TEXT
);

CREATE TABLE IF NOT EXISTS design_assets (
    design_id TEXT NOT NULL REFERENCES design_manifests(design_id),
    asset TEXT NOT NULL,
    bytes BLOB NOT NULL,
    PRIMARY KEY(design_id, asset)
);

CREATE TABLE IF NOT EXISTS products (
    key              TEXT PRIMARY KEY,          -- '<source>:<source_id>'
    source           TEXT NOT NULL,
    source_id        TEXT NOT NULL,
    url              TEXT NOT NULL,
    name             TEXT NOT NULL,
    brand            TEXT,
    product_type     TEXT,
    category         TEXT,
    category_path    TEXT,                      -- JSON array
    design_category  TEXT,
    description      TEXT,
    description_html TEXT,
    price            REAL,
    compare_at_price REAL,
    currency         TEXT DEFAULT 'INR',
    price_unit       TEXT,
    availability     TEXT DEFAULT 'unknown',
    sku              TEXT,
    item_no          TEXT,
    width_mm         REAL,
    height_mm        REAL,
    depth_mm         REAL,
    length_mm        REAL,
    diameter_mm      REAL,
    dimensions       TEXT,                      -- JSON object, all axes
    dimension_text   TEXT,                      -- JSON array
    weight_kg        REAL,
    materials        TEXT,
    colors           TEXT,
    tags             TEXT,
    rating           REAL,
    review_count     INTEGER,
    country          TEXT DEFAULT 'IN',
    attributes       TEXT,
    raw              TEXT,
    embedding_text   TEXT,
    content_hash     TEXT,
    first_seen       TEXT,
    last_seen        TEXT,
    -- Completeness, derived on write. `searchable` answers "can this be found";
    -- `design_eligible` answers "can this be placed in a room without guessing
    -- its size". A bag of cement is the first and never the second.
    placement_geometry      TEXT,
    dimensions_complete     INTEGER DEFAULT 0,
    primary_image_available INTEGER DEFAULT 0,
    alternate_views_count   INTEGER DEFAULT 0,
    price_current           INTEGER DEFAULT 0,
    stock_known             INTEGER DEFAULT 0,
    variant_resolved        INTEGER DEFAULT 0,
    material_known          INTEGER DEFAULT 0,
    color_known             INTEGER DEFAULT 0,
    placement_safe          INTEGER DEFAULT 0,
    render_asset_quality    INTEGER DEFAULT 0,
    searchable              INTEGER DEFAULT 0,
    design_eligible         INTEGER DEFAULT 0,
    quality_gaps            TEXT,
    -- The model line this SKU belongs to. MALM bed 140 and MALM bed 180 are two
    -- SKUs and one piece of furniture; retrieval collapses on this.
    family_key              TEXT,
    family_label            TEXT,
    variant_label           TEXT
);
CREATE INDEX IF NOT EXISTS idx_products_design  ON products(design_category);
CREATE INDEX IF NOT EXISTS idx_products_source  ON products(source);
CREATE INDEX IF NOT EXISTS idx_products_price   ON products(price);
CREATE INDEX IF NOT EXISTS idx_products_avail   ON products(availability);

CREATE TABLE IF NOT EXISTS product_images (
    product_key TEXT NOT NULL REFERENCES products(key) ON DELETE CASCADE,
    position    INTEGER NOT NULL,
    url         TEXT NOT NULL,
    alt         TEXT,
    width       INTEGER,
    height      INTEGER,
    role        TEXT,
    local_path  TEXT,
    sha256      TEXT,
    PRIMARY KEY (product_key, position)
);
CREATE INDEX IF NOT EXISTS idx_images_sha ON product_images(sha256);

CREATE TABLE IF NOT EXISTS product_variants (
    product_key      TEXT NOT NULL REFERENCES products(key) ON DELETE CASCADE,
    variant_id       TEXT NOT NULL,
    title            TEXT,
    sku              TEXT,
    price            REAL,
    compare_at_price REAL,
    available        INTEGER,
    quantity         INTEGER,
    options          TEXT,
    image_url        TEXT,
    bulk_pricing     TEXT,
    PRIMARY KEY (product_key, variant_id)
);

-- Quick commerce reprices often; keeping the series makes a quote reproducible.
CREATE TABLE IF NOT EXISTS price_history (
    product_key  TEXT NOT NULL REFERENCES products(key) ON DELETE CASCADE,
    observed_at  TEXT NOT NULL,
    price        REAL,
    availability TEXT,
    PRIMARY KEY (product_key, observed_at)
);

-- Being in the catalogue is not the same as being buyable where the customer
-- lives. Unused by the prototype; the shape is here so a per-pincode stock feed
-- can be recorded without reshaping the catalogue.
CREATE TABLE IF NOT EXISTS product_availability (
    product_key     TEXT NOT NULL REFERENCES products(key) ON DELETE CASCADE,
    country         TEXT NOT NULL DEFAULT 'IN',
    region          TEXT NOT NULL DEFAULT '',
    postal_code     TEXT NOT NULL DEFAULT '',
    stock_status    TEXT,
    delivery_status TEXT,
    observed_at     TEXT NOT NULL,
    PRIMARY KEY (product_key, country, region, postal_code)
);

-- Nothing here assumes one embedding per product. A better furniture
-- representation will come along, and it has to be possible to compute it
-- beside CLIP, compare retrieval quality on the same catalogue, and switch --
-- without destroying the index that is currently serving. Model and version are
-- separate columns so "same model, retrained weights" is expressible, and
-- source_asset is content-addressed so a vector can be tied to the exact bytes
-- it was computed from rather than a URL the retailer may reuse.
CREATE TABLE IF NOT EXISTS embeddings (
    product_key   TEXT NOT NULL REFERENCES products(key) ON DELETE CASCADE,
    kind          TEXT NOT NULL,      -- 'text' | 'image'
    source_asset  TEXT NOT NULL,      -- 'sha256:...' of the bytes or text embedded
    model         TEXT NOT NULL,      -- 'clip-vit-b32'
    model_version TEXT NOT NULL,      -- 'laion2b_s34b_b79k'
    dim           INTEGER NOT NULL,
    vec           BLOB NOT NULL,      -- float32, L2-normalised
    created_at    TEXT NOT NULL,
    PRIMARY KEY (product_key, kind, source_asset, model, model_version)
);

CREATE TABLE IF NOT EXISTS crawl_runs (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    source       TEXT NOT NULL,
    started_at   TEXT NOT NULL,
    finished_at  TEXT,
    status       TEXT NOT NULL DEFAULT 'running',
    pages_ok     INTEGER DEFAULT 0,
    pages_failed INTEGER DEFAULT 0,
    products_new INTEGER DEFAULT 0,
    products_changed INTEGER DEFAULT 0,
    price_changes    INTEGER DEFAULT 0,
    images_downloaded INTEGER DEFAULT 0,
    embeddings_added  INTEGER DEFAULT 0,
    error        TEXT
);
CREATE INDEX IF NOT EXISTS idx_runs_started ON crawl_runs(started_at DESC);

CREATE VIRTUAL TABLE IF NOT EXISTS products_fts USING fts5(
    key UNINDEXED, name, brand, category_path, description, materials, colors, tags,
    tokenize='porter unicode61'
);

-- The index is maintained by the database, not by application code. Keeping the
-- two in step by convention works right up until the first code path that
-- forgets, and a search index that disagrees with the table it describes is an
-- exceptionally confusing bug to chase. Triggers make it structural: any write
-- to products, from anywhere, carries the index with it inside the same
-- transaction.
CREATE TRIGGER IF NOT EXISTS products_fts_ai AFTER INSERT ON products BEGIN
    INSERT INTO products_fts (key, name, brand, category_path, description,
                              materials, colors, tags)
    VALUES (new.key, COALESCE(new.name, ''), COALESCE(new.brand, ''),
            COALESCE(new.category_path, ''), COALESCE(new.description, ''),
            COALESCE(new.materials, ''), COALESCE(new.colors, ''),
            COALESCE(new.tags, ''));
END;

CREATE TRIGGER IF NOT EXISTS products_fts_ad AFTER DELETE ON products BEGIN
    DELETE FROM products_fts WHERE key = old.key;
END;

CREATE TRIGGER IF NOT EXISTS products_fts_au AFTER UPDATE ON products BEGIN
    DELETE FROM products_fts WHERE key = old.key;
    INSERT INTO products_fts (key, name, brand, category_path, description,
                              materials, colors, tags)
    VALUES (new.key, COALESCE(new.name, ''), COALESCE(new.brand, ''),
            COALESCE(new.category_path, ''), COALESCE(new.description, ''),
            COALESCE(new.materials, ''), COALESCE(new.colors, ''),
            COALESCE(new.tags, ''));
END;
"""

_PRODUCT_COLS = [
    "key", "source", "source_id", "url", "name", "brand", "product_type", "category",
    "category_path", "design_category", "description", "description_html", "price",
    "compare_at_price", "currency", "price_unit", "availability", "sku", "item_no",
    "width_mm", "height_mm", "depth_mm", "length_mm", "diameter_mm", "dimensions",
    "dimension_text", "weight_kg", "materials", "colors", "tags", "rating",
    "review_count", "country", "attributes", "raw", "embedding_text", "content_hash",
    "first_seen", "last_seen",
] + list(DERIVED_COLUMNS)


_UPSERT_SQL = (
    f"INSERT INTO products ({','.join(_PRODUCT_COLS)}) "
    f"VALUES ({','.join('?' * len(_PRODUCT_COLS))}) "
    f"ON CONFLICT(key) DO UPDATE SET "
    + ", ".join(f"{c}=excluded.{c}" for c in _PRODUCT_COLS if c != "key")
)


def _j(v: Any) -> str:
    return json.dumps(v, ensure_ascii=False)


def connect(db_path: str | Path) -> sqlite3.Connection:
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    _migrate(conn)
    return conn


def fts_drift(conn: sqlite3.Connection) -> dict[str, int]:
    """How far the search index has drifted from the table. Should always be zero."""
    q = lambda sql: conn.execute(sql).fetchone()[0]      # noqa: E731
    return {
        "missing": q("SELECT COUNT(*) FROM products p WHERE NOT EXISTS "
                     "(SELECT 1 FROM products_fts f WHERE f.key = p.key)"),
        "orphaned": q("SELECT COUNT(*) FROM products_fts f WHERE NOT EXISTS "
                      "(SELECT 1 FROM products p WHERE p.key = f.key)"),
        "stale": q("SELECT COUNT(*) FROM products p JOIN products_fts f ON f.key = p.key "
                   "WHERE f.name <> COALESCE(p.name, '')"),
    }


def rebuild_fts(conn: sqlite3.Connection) -> int:
    """Rebuild the index from the table. Only needed after an out-of-band change."""
    conn.execute("DELETE FROM products_fts")
    conn.execute(
        "INSERT INTO products_fts (key, name, brand, category_path, description, "
        "materials, colors, tags) SELECT key, COALESCE(name,''), COALESCE(brand,''), "
        "COALESCE(category_path,''), COALESCE(description,''), COALESCE(materials,''), "
        "COALESCE(colors,''), COALESCE(tags,'') FROM products"
    )
    conn.commit()
    return conn.execute("SELECT COUNT(*) FROM products_fts").fetchone()[0]


def _migrate(conn: sqlite3.Connection) -> None:
    """Add columns a database created by an earlier version is missing.

    CREATE TABLE IF NOT EXISTS silently leaves an existing table alone, so new
    columns have to be added explicitly or a live catalogue breaks on upgrade.
    """
    emb = {r["name"] for r in conn.execute("PRAGMA table_info(embeddings)")}
    if emb and "model_version" not in emb:
        rows = conn.execute("SELECT COUNT(*) c FROM embeddings").fetchone()["c"]
        if rows == 0:
            conn.execute("DROP TABLE embeddings")
            conn.executescript(SCHEMA)
            log.info("rebuilt embeddings table with model versioning")
        else:
            # Keep the vectors; split the model identifier they were written with.
            conn.execute("ALTER TABLE embeddings ADD COLUMN model_version TEXT DEFAULT ''")
            conn.execute("ALTER TABLE embeddings ADD COLUMN created_at TEXT DEFAULT ''")
            log.warning("embeddings migrated in place; %d rows keep their old "
                        "composite model id", rows)

    have = {r["name"] for r in conn.execute("PRAGMA table_info(products)")}
    added = [c for c in DERIVED_COLUMNS if c not in have]
    for col in added:
        conn.execute(f"ALTER TABLE products ADD COLUMN {col} {DERIVED_COLUMNS[col]}")
    if added:
        # Existing rows have NULL in the new columns and would be skipped as
        # "unchanged" forever, so clear the hash to force one re-derivation.
        conn.execute("UPDATE products SET content_hash = NULL")
        log.info("migrated products: added %s (existing rows will re-derive)",
                 ", ".join(added))
    # Indexed after the migration: on an existing database the columns do not
    # exist until the ALTER TABLEs above have run.
    conn.executescript(
        "CREATE INDEX IF NOT EXISTS idx_products_eligible ON products(design_eligible);"
        "CREATE INDEX IF NOT EXISTS idx_products_searchable ON products(searchable);"
        "CREATE INDEX IF NOT EXISTS idx_products_geometry ON products(placement_geometry);"
        "CREATE INDEX IF NOT EXISTS idx_products_family ON products(family_key);"
        "CREATE INDEX IF NOT EXISTS idx_emb_model ON embeddings(model, model_version, kind);"
    )
    conn.commit()


def _content_hash(p: Product) -> str:
    """Hash of everything stored that can change between crawls.

    Derived fields belong in here too, not just scraped ones: `design_category`
    comes from a keyword taxonomy that gets edited, and if the hash ignored it a
    re-parse would be silently discarded as "unchanged".
    """
    payload = _j([
        p.url, p.name, p.brand, p.price, p.compare_at_price, p.availability,
        p.price_unit, p.sku, p.description, p.materials, p.colors, p.tags,
        p.dimensions, p.weight_kg, [i.url for i in p.images], p.category_path,
        p.category, p.product_type, p.design_category, p.rating, resolve_family(p)[0],
        [(v.variant_id, v.price, v.available, v.quantity) for v in p.variants],
    ])
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def upsert(conn: sqlite3.Connection, products: Iterable[Product]) -> dict[str, int]:
    """Insert or update products. Returns counts of new / changed / unchanged."""
    now = datetime.now(timezone.utc).isoformat()
    stats = {"new": 0, "changed": 0, "unchanged": 0}

    for p in products:
        chash = _content_hash(p)
        row = conn.execute(
            "SELECT content_hash, first_seen, price, availability FROM products WHERE key=?",
            (p.key,),
        ).fetchone()

        if row and row["content_hash"] == chash:
            conn.execute("UPDATE products SET last_seen=? WHERE key=?", (now, p.key))
            stats["unchanged"] += 1
            continue

        stats["new" if row is None else "changed"] += 1
        d = p.dimensions or {}
        values = {
            "key": p.key, "source": p.source, "source_id": p.source_id, "url": p.url,
            "name": p.name, "brand": p.brand, "product_type": p.product_type,
            "category": p.category, "category_path": _j(p.category_path),
            "design_category": p.design_category, "description": p.description,
            "description_html": p.description_html, "price": p.price,
            "compare_at_price": p.compare_at_price, "currency": p.currency,
            "price_unit": p.price_unit, "availability": p.availability, "sku": p.sku,
            "item_no": p.item_no,
            "width_mm": d.get("width_mm"), "height_mm": d.get("height_mm"),
            "depth_mm": d.get("depth_mm"), "length_mm": d.get("length_mm"),
            "diameter_mm": d.get("diameter_mm"),
            "dimensions": _j(d), "dimension_text": _j(p.dimension_text),
            "weight_kg": p.weight_kg, "materials": _j(p.materials),
            "colors": _j(p.colors), "tags": _j(p.tags), "rating": p.rating,
            "review_count": p.review_count, "country": p.country,
            "attributes": _j(p.attributes), "raw": _j(p.raw),
            "embedding_text": p.embedding_text(), "content_hash": chash,
            "first_seen": row["first_seen"] if row else now, "last_seen": now,
        }
        # Completeness is derived from the values just assembled, so it can never
        # drift out of step with the row it describes.
        marks = assess(p, last_seen=now)
        marks["quality_gaps"] = _j(marks["quality_gaps"])
        values.update(marks)

        family_key, family_label, variant_label = resolve_family(p)
        values.update(family_key=family_key, family_label=family_label,
                      variant_label=variant_label)
        # A real upsert, not INSERT OR REPLACE: REPLACE deletes the existing row
        # first, and the ON DELETE CASCADE on price_history would take the whole
        # price series with it every time a product was re-crawled.
        conn.execute(_UPSERT_SQL, [values[c] for c in _PRODUCT_COLS])

        # Images and variants are replaced wholesale; they are small and ordered.
        # Carry the downloaded file across by URL first: positions shift whenever a
        # retailer adds a shot, and dropping local_path would re-download the whole
        # catalogue on every crawl.
        on_disk = {
            r["url"]: (r["local_path"], r["sha256"])
            for r in conn.execute(
                "SELECT url, local_path, sha256 FROM product_images "
                "WHERE product_key=? AND local_path IS NOT NULL", (p.key,))
        }
        for img in p.images:
            if img.local_path is None and img.url in on_disk:
                img.local_path, img.sha256 = on_disk[img.url]
        conn.execute("DELETE FROM product_images WHERE product_key=?", (p.key,))
        conn.executemany(
            "INSERT OR REPLACE INTO product_images "
            "(product_key,position,url,alt,width,height,role,local_path,sha256) "
            "VALUES (?,?,?,?,?,?,?,?,?)",
            [
                (p.key, i.position, i.url, i.alt, i.width, i.height, i.role,
                 i.local_path, i.sha256)
                for i in p.images
            ],
        )
        conn.execute("DELETE FROM product_variants WHERE product_key=?", (p.key,))
        conn.executemany(
            "INSERT OR REPLACE INTO product_variants "
            "(product_key,variant_id,title,sku,price,compare_at_price,available,"
            "quantity,options,image_url,bulk_pricing) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            [
                (p.key, v.variant_id, v.title, v.sku, v.price, v.compare_at_price,
                 int(v.available) if v.available is not None else None, v.quantity,
                 _j(v.options), v.image_url, _j(v.bulk_pricing))
                for v in p.variants
            ],
        )

        if row is None or row["price"] != p.price or row["availability"] != p.availability:
            conn.execute(
                "INSERT OR REPLACE INTO price_history VALUES (?,?,?,?)",
                (p.key, now, p.price, p.availability),
            )

    conn.commit()
    return stats


def load_jsonl(conn: sqlite3.Connection, path: str | Path, batch: int = 500) -> dict[str, int]:
    """Stream a scrape's JSONL into the database."""
    totals = {"new": 0, "changed": 0, "unchanged": 0}
    chunk: list[Product] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                chunk.append(Product.from_dict(json.loads(line)))
            except json.JSONDecodeError:
                # The crawl appends as it runs, so the last line may be half
                # written. Skip it; the next load picks it up complete.
                log.warning("skipping malformed line in %s", Path(path).name)
                continue
            if len(chunk) >= batch:
                for k, v in upsert(conn, chunk).items():
                    totals[k] += v
                chunk.clear()
    if chunk:
        for k, v in upsert(conn, chunk).items():
            totals[k] += v
    return totals


def stats(conn: sqlite3.Connection) -> dict[str, Any]:
    q = lambda sql: conn.execute(sql).fetchall()  # noqa: E731
    return {
        "products": conn.execute("SELECT COUNT(*) c FROM products").fetchone()["c"],
        "images": conn.execute("SELECT COUNT(*) c FROM product_images").fetchone()["c"],
        "images_downloaded": conn.execute(
            "SELECT COUNT(*) c FROM product_images WHERE local_path IS NOT NULL"
        ).fetchone()["c"],
        "embeddings": conn.execute("SELECT COUNT(*) c FROM embeddings").fetchone()["c"],
        "by_source": {r["source"]: r["c"] for r in q(
            "SELECT source, COUNT(*) c FROM products GROUP BY source ORDER BY c DESC")},
        "by_design_category": {r["design_category"] or "(unclassified)": r["c"] for r in q(
            "SELECT design_category, COUNT(*) c FROM products "
            "GROUP BY design_category ORDER BY c DESC")},
        "priced": conn.execute(
            "SELECT COUNT(*) c FROM products WHERE price IS NOT NULL").fetchone()["c"],
        "fts_drift": fts_drift(conn),
        "last_run": (lambda r: dict(r) if r else None)(conn.execute(
            "SELECT source, started_at, finished_at, status, products_new, "
            "products_changed, price_changes FROM crawl_runs "
            "ORDER BY started_at DESC LIMIT 1").fetchone()),
    }
