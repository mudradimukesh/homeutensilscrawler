-- Postgres + pgvector schema: the same catalogue as catalog/store.py, for when
-- more than one process needs it at once.
--
-- The point of this file is that the vector index lives *in* the product table's
-- database, not beside it. A design AI's question is never purely "what looks
-- like this" — it is "what looks like this, is a sofa, is in stock in Bengaluru,
-- and costs under Rs.40,000". With pgvector that is one query and the filter is
-- exact. Split across a standalone vector service it becomes two round trips
-- plus a reconciliation step that either over-fetches or silently drops matches.
--
--   CREATE EXTENSION IF NOT EXISTS vector;
--   CREATE EXTENSION IF NOT EXISTS pg_trgm;

CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;

CREATE TABLE IF NOT EXISTS products (
    key              TEXT PRIMARY KEY,
    source           TEXT NOT NULL,
    source_id        TEXT NOT NULL,
    url              TEXT NOT NULL,
    name             TEXT NOT NULL,
    brand            TEXT,
    product_type     TEXT,
    category         TEXT,
    category_path    JSONB DEFAULT '[]',
    design_category  TEXT,
    description      TEXT,
    description_html TEXT,
    price            NUMERIC(12,2),
    compare_at_price NUMERIC(12,2),
    currency         TEXT NOT NULL DEFAULT 'INR',
    price_unit       TEXT,
    availability     TEXT NOT NULL DEFAULT 'unknown',
    sku              TEXT,
    item_no          TEXT,
    width_mm         REAL,
    height_mm        REAL,
    depth_mm         REAL,
    length_mm        REAL,
    diameter_mm      REAL,
    dimensions       JSONB DEFAULT '{}',
    dimension_text   JSONB DEFAULT '[]',
    weight_kg        REAL,
    materials        JSONB DEFAULT '[]',
    colors           JSONB DEFAULT '[]',
    tags             JSONB DEFAULT '[]',
    rating           REAL,
    review_count     INTEGER,
    country          TEXT NOT NULL DEFAULT 'IN',
    attributes       JSONB DEFAULT '{}',
    raw              JSONB DEFAULT '{}',
    embedding_text   TEXT,
    content_hash     TEXT,
    first_seen       TIMESTAMPTZ DEFAULT now(),
    last_seen        TIMESTAMPTZ DEFAULT now(),
    -- CLIP ViT-B/32. Change the width here and in catalog/embed.py together.
    text_embedding   vector(512),
    search_tsv       tsvector GENERATED ALWAYS AS (
        setweight(to_tsvector('english', coalesce(name, '')),        'A') ||
        setweight(to_tsvector('english', coalesce(brand, '')),       'B') ||
        setweight(to_tsvector('english', coalesce(category, '')),    'B') ||
        setweight(to_tsvector('english', coalesce(description, '')), 'D')
    ) STORED
);

CREATE INDEX IF NOT EXISTS idx_products_design ON products (design_category);
CREATE INDEX IF NOT EXISTS idx_products_price  ON products (price);
CREATE INDEX IF NOT EXISTS idx_products_avail  ON products (availability);
CREATE INDEX IF NOT EXISTS idx_products_tsv    ON products USING gin (search_tsv);
CREATE INDEX IF NOT EXISTS idx_products_name_trgm ON products USING gin (name gin_trgm_ops);
CREATE INDEX IF NOT EXISTS idx_products_text_vec
    ON products USING hnsw (text_embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64);

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
    -- One row per photograph, not per product: a sofa shot from the front and
    -- styled in a room are different vectors, and either may be the one that
    -- matches an inspiration image.
    embedding   vector(512),
    PRIMARY KEY (product_key, position)
);
CREATE INDEX IF NOT EXISTS idx_images_sha ON product_images (sha256);
CREATE INDEX IF NOT EXISTS idx_images_vec
    ON product_images USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64);

CREATE TABLE IF NOT EXISTS product_variants (
    product_key      TEXT NOT NULL REFERENCES products(key) ON DELETE CASCADE,
    variant_id       TEXT NOT NULL,
    title            TEXT,
    sku              TEXT,
    price            NUMERIC(12,2),
    compare_at_price NUMERIC(12,2),
    available        BOOLEAN,
    quantity         INTEGER,
    options          JSONB DEFAULT '{}',
    image_url        TEXT,
    bulk_pricing     JSONB DEFAULT '[]',
    PRIMARY KEY (product_key, variant_id)
);

CREATE TABLE IF NOT EXISTS price_history (
    product_key  TEXT NOT NULL REFERENCES products(key) ON DELETE CASCADE,
    observed_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    price        NUMERIC(12,2),
    availability TEXT,
    PRIMARY KEY (product_key, observed_at)
);

-- The query the design AI actually issues: constraints exactly, similarity by rank.
--
--   SELECT p.key, p.name, p.price, p.url,
--          1 - (i.embedding <=> $1) AS visual_score
--     FROM product_images i
--     JOIN products p ON p.key = i.product_key
--    WHERE p.design_category = $2
--      AND p.availability IN ('in_stock', 'store_only')
--      AND p.price <= $3
--    ORDER BY i.embedding <=> $1
--    LIMIT 10;
--
-- With an HNSW index and a selective filter, set `hnsw.ef_search` above the LIMIT
-- (e.g. SET LOCAL hnsw.ef_search = 200) so the filter does not empty the result.
