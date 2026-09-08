# Interior Catalog — Design Spec

**Status:** working prototype, catalogue crawl in progress
**Last revised:** 2026-09-09
**Scope:** everything from the retailer's HTML to the tool surface a language model is given

---

## 1. What this is

A catalogue service for an interior-design AI. It crawls Indian retailers, normalises
their products into one schema, scores each product for whether it can actually be placed
in a room, and exposes three functions to a language model so that a generated design is
made of products that exist, at prices the retailer published.

The load-bearing property is the last one. A design model that invents a sofa produces a
picture; a design model constrained to this catalogue produces a quote someone can act on.

### Sources

| Source | What it sells | Catalogue size | Data shape |
|---|---|---|---|
| `homerun` — home-run.co | construction, renovation, interior materials (Bengaluru quick commerce) | 610 URLs | Shopify product object inside the Next.js RSC payload |
| `ikea_in` — ikea.com/in/en | furniture, lighting, textiles, decor | 12,575 URLs | `text/hydrate` JSON blocks + JSON-LD |

Neither needs a headless browser: both ship complete structured data in the HTML. That is
why a page parses in about a millisecond and why the parsers survive theme changes.

### Non-goals

- Rendering images. The catalogue supplies products and assets; generation lives elsewhere.
- Being a general web scraper. Two sources are modelled deliberately and deeply.
- Multi-country. India only, INR only — enforced at discovery by reading only the
  `prod-en-IN_*` sitemaps.

---

## 2. Pipeline

```
DISCOVER → PLAN → FETCH → PARSE → NORMALISE → VALIDATE → FAMILY → STORE → EMBED → SERVE
```

| Stage | Module | What it does |
|---|---|---|
| Discover | `sources/*.discover` | Sitemaps → every product URL. Cached; the whole catalogue is enumerated before anything is fetched. |
| Plan | `stratify.py` | Classify each URL from its slug, build weighted category queues, emit in interleaved rounds. |
| Fetch | `http.py` | robots.txt, per-host rate limit with jitter, retry with backoff, `Retry-After`, gzip page cache. |
| Parse | `sources/*.parse` | Embedded JSON → source-specific fields. Raw payload kept verbatim. |
| Normalise | `models.py` | One `Product`: millimetre dimensions, INR price, one design taxonomy across both retailers. |
| Validate | `quality.py` | Completeness flags: `searchable`, `design_eligible`, `quality_gaps`. |
| Family | `family.py` | Resolve the model line this SKU belongs to. |
| Store | `store.py` | JSONL append, then SQLite upsert; price changes append to history. |
| Embed | `embed.py` | CLIP text and image vectors, versioned by model. |
| Serve | `search.py`, `tools.py`, `web.py` | Hybrid retrieval, design pricing, the model tool boundary, a browser view. |

Every stage is restartable. Discovery and fetch are cached, JSONL is append-only, and the
upsert is idempotent, so an interrupted run resumes without re-fetching a page.

---

## 3. Data model

SQLite, one file. `catalog/postgres.sql` mirrors the schema for Postgres + pgvector.

### `products` — 56 columns, the hub

Grouped by role:

| Group | Columns |
|---|---|
| identity | `key` (PK, `<source>:<id>`), `source`, `source_id`, `url`, `sku`, `item_no` |
| description | `name`, `brand`, `product_type`, `category`, `category_path`, `design_category`, `description`, `description_html` |
| commerce | `price`, `compare_at_price`, `currency`, `price_unit`, `availability` |
| geometry | `width_mm`, `height_mm`, `depth_mm`, `length_mm`, `diameter_mm`, `dimensions`, `dimension_text`, `weight_kg` |
| attributes | `materials`, `colors`, `tags`, `rating`, `review_count`, `country`, `attributes`, `raw` |
| completeness | `placement_geometry`, `dimensions_complete`, `primary_image_available`, `alternate_views_count`, `price_current`, `stock_known`, `variant_resolved`, `material_known`, `color_known`, `placement_safe`, `render_asset_quality`, `searchable`, `design_eligible`, `quality_gaps` |
| family | `family_key`, `family_label`, `variant_label` |
| bookkeeping | `embedding_text`, `content_hash`, `first_seen`, `last_seen` |

### Satellites

| Table | Holds | Notes |
|---|---|---|
| `product_images` | every photograph, ordered | `local_path` + `sha256`; content-addressed on disk |
| `product_variants` | retailer variants | price, stock quantity, options, bulk-price tiers |
| `price_history` | one row per observed change | append-only; makes a quote explainable later |
| `embeddings` | text and image vectors | keyed by model **and** version |
| `product_availability` | per-region stock | schema present, unpopulated — see §9 |
| `crawl_runs` | one row per refresh | start, finish, status, counts, failures |
| `products_fts` | FTS5 index | maintained by triggers, not application code |

All satellites reference `products.key` with `ON DELETE CASCADE`.

### Writes to `products` use a real upsert

`INSERT … ON CONFLICT(key) DO UPDATE`, never `INSERT OR REPLACE`. REPLACE deletes the row
first, and the cascade would take the entire price series with it on every re-crawl.

---

## 4. Crawl ordering

**Problem.** IKEA's sitemap is roughly reverse-alphabetical, so any prefix of it is one
product family. The first 30-product sample came back as 16 VOXTORP kitchen drawer fronts,
and a coverage audit of the resulting catalogue showed **zero** beds, sofas, lamps, rugs,
tables and curtains. That is not a thin catalogue; it is one that cannot furnish a room.

**Design.** URLs are classified from the slug — both retailers put the product type in the
URL, so ordering costs no requests — grouped into weighted category queues, interleaved by
product family so no single range fills a queue, and emitted in weighted rounds. Every
prefix of the crawl is therefore roughly balanced, not just the completed crawl.

Weights (higher = more per round): beds, sofas, wardrobes 5 · nightstands, lamps, ceiling
lighting, rugs, curtains, tables 4 · bedding, dressers, chairs, desks, shelving, storage 3
· mirrors, decor, bathroom, flooring, paint 2 · kitchen, outdoor, components and all
building materials 1.

**Measured over all 12,575 IKEA URLs, first 150 crawled:**

| | sitemap order | stratified |
|---|---|---|
| spare parts / fronts | 95 | 2 |
| sofas, nightstands, lamps, curtains, bedding, dressers | 0 each | 6–11 each |
| bedroom slots fillable | 3 of 6 | 6 of 6 |

Two classifier findings shaped the weights: **26.5%** of IKEA India is fronts, cover panels
and kitchen carcasses, and spare covers are named like the furniture they belong to
(`VIMLE cover 4-seat sofa`), so both needed priority rules to stay out of the highest-weight
queues. Unclassified fell from 17% to 6.6%.

---

## 5. Completeness: findable vs placeable

A product can be perfectly searchable — name, price, photograph — and impossible to place,
because nobody knows how wide it is. Retrieval wants the first; a room layout needs the
second. They are computed and stored separately.

```
searchable       = name AND primary image AND price
placement_safe   = dimensions complete for its geometry AND not custom-made
design_eligible  = searchable AND placement_safe AND price_current
                   AND stock_known AND variant_resolved
```

### Completeness is judged per placement geometry

A flat `width + depth + height` rule marks **100% of rugs, bedding and curtains** incomplete
— flat goods have no depth or height — and IKEA reports a bed as width × **length**, never
width × depth. Requiring a box of every product would exclude exactly the categories a
bedroom needs most.

| Geometry | Required axes | Categories |
|---|---|---|
| `box` | W · D/L · H | wardrobes, dressers, shelving |
| `footprint` | W · D/L | beds, sofas, tables, desks, chairs, nightstands, storage, bathroom, outdoor |
| `planar` | W · L/D | rugs, bedding, curtains |
| `suspended` | Dia/W · H/L | ceiling lighting |
| `standing` | H · Dia/W | lamps, decor |
| `wall` | W · H | mirrors |
| `none` | — | paint, cement, tiles, electrical, plumbing, hardware, kitchen parts, components |

`depth` and `length` are interchangeable because the two retailers disagree about which
word means the second horizontal axis.

Geometry `none` is not a data defect. Materials are bought for a room and never placed in
one as objects, so they stay searchable and are never eligible — and `price_design` only
demands placement data of categories that are placed, so a renovation quote can still price
paint and cement.

`quality_gaps` records *why* a product failed, so the model can say "matches exist but none
are complete enough to place — missing width".

---

## 6. Product families

MALM bed white 140, MALM bed white 160 and MALM bed black 180 are three SKUs and one piece
of furniture. Retrieval that cannot tell them apart returns a wall of near-duplicates and
hides the rest of the catalogue behind them.

Family keys are derived at parse time from what each retailer already states — IKEA names a
range and an article type (`VIHALS` + `wardrobe`), HomeRun writes `Brand Product, Size` and
the part before the first comma is the model line. Keys are source-scoped; two retailers'
ranges are never the same product.

**Measured (4,884 products):** 2,858 families. Wardrobes are the extreme case — 393 SKUs,
68 families, five variants each; one family (PAX / GULLABERG) has 52.

- `find_products(collapse_families=True)` returns one SKU per model line with `family_size`.
- `get_family(family_key)` is the variant lookup: *"I like this bed, but I need the 180 cm
  version"* is a lookup, not another semantic search.
- `catalog coverage` reports SKUs, families **and placeable families**, because the SKU
  count flatters: 397 beds are 127 families of which 110 are placeable, and that last number
  is what a designer can choose between.

Crawl-time interleaving uses the slug's first token, because the real family key is not
known until the page is parsed. It is an approximation, and only for ordering.

---

## 7. Vector layer

**Nothing assumes one embedding per product.**

```
embeddings(product_key, kind, source_asset, model, model_version, dim, vec, created_at)
PRIMARY KEY (product_key, kind, source_asset, model, model_version)
```

- `model` and `model_version` are separate, so "same model, retrained weights" is
  expressible.
- `kind` is `text` or `image` — separate representations, not one blended vector.
- `source_asset` is the content hash of exactly what was embedded (`sha256:…`), so a vector
  is tied to bytes rather than a URL the retailer may reuse. An edited description or a
  replaced photograph yields a new vector instead of silently reusing a stale one.

This makes a better representation adoptable: compute it beside CLIP, compare retrieval
quality on the same catalogue, promote it — without destroying the index currently serving.
Verified by running two models of different dimensionality (64-d and 128-d) side by side.

Current model: `clip-vitb32` / `laion2b_s34b_b79k`, 512-d, L2-normalised so cosine is a dot
product.

### Why the vectors live in the catalogue database

No query here is purely similarity. The design AI asks *"looks like this **and** is a sofa
**and** is in stock **and** under ₹40,000"*. Price and stock are exact constraints; an
approximate answer produces a room the customer cannot buy. In one database that is one
query. Split across a vector service it becomes two round trips plus reconciliation:
pre-filter and you ship the whole ID set over, post-filter and the constraint eats your
top-k until nothing is left.

Scale supports this: ~13k products and ~40k vectors is about 80 MB — brute force is under
20M multiply-adds, single-digit milliseconds in NumPy, and pgvector with HNSW is quicker
still. A dedicated vector database earns its operational surface somewhere past ten million
vectors.

**When to move to Postgres** — not merely "more than one process". SQLite serves many
concurrent readers well and handles multiple processes under a single-writer pattern, which
a nightly crawl plus a read-only API already is. Move when you need concurrent *writers*,
remote database access, higher API concurrency, distributed workers, vector retrieval past
a few million rows, or operational HA.

---

## 8. Retrieval, pricing, and the model boundary

### Retrieval

Exact SQL filters first (category, budget, stock, eligibility, dimensional limits), then
ranking by reciprocal rank fusion over BM25 and cosine. The filter is not a preference: a
wardrobe that does not fit the wall is a wrong answer, not a lower-ranked one.

Without CLIP installed, retrieval is FTS5 BM25 alone with the product name weighted 10×
against the description — HomeRun ships 5,000-character SEO descriptions that otherwise
swamp every query.

### Pricing a design

`price_design` costs an object list. Distinct objects get distinct products, so a room does
not come back priced as six copies of one chair. Subtotals are split by price freshness
(`fresh` / `stale` / `unknown`) so a six-month-old figure cannot quietly become part of a
₹3 lakh total.

### The tool boundary

Three functions are the entire surface a model gets:

```
search_catalog(query, category, price/dimension limits, placeable_only, one_per_family)
get_product(product_key)          → detail + every variant in the model line
price_design(objects)             → costed bill of materials
```

Guarantees:

- The model never sees SQL, filesystem paths, raw payloads or internal columns. Responses
  are assembled field by field; passing rows through is how those leak into a prompt.
- The model never states a price the tools did not return.
- **A product may only appear in a priced design if the service issued its key.** Keys are
  recorded per session and validated on the way back in, so a fabricated SKU is rejected
  and contributes nothing to the subtotal.

`agent.py` drives the loop over the OpenAI Responses API with plain HTTPS and no SDK.
Deliberately not the consumer ChatGPT UI: custom tool support there is plan-gated and the
catalogue would have to be exposed to reach it. Through the API the tools stay in-process
and the database never leaves it.

---

## 9. Operations

### Refresh

`refresh` is not `scrape`. `scrape` resumes past what it has and serves pages from cache —
right while building a parser, wrong for a catalogue whose prices move daily. `refresh`
re-fetches, rewrites each JSONL, loads, pulls new images, embeds what is new, and reports
what changed. Every run lands in `crawl_runs`. Two refreshes cannot overlap: a PID lock
guards the crawl, and a lock left by a dead process is reclaimed rather than obeyed.

`schedule install` registers it as a launchd LaunchAgent on macOS — not cron, because
launchd runs a calendar job it missed while the machine was asleep, and a laptop is asleep
at 03:30 most nights.

### Disk budget

A full crawl does not fit in a small allowance, so the ceiling is enforced **while writing**
rather than checked afterwards — a check afterwards only reports a disk that is already
full. Each large writer (page cache, image store, JSONL) asks the budget before it writes
and stops the run cleanly when the next write would cross. Crawled products are kept, the
database stays consistent, and the run is recorded `over_budget`.

Writers stop slightly below the ceiling: SQLite extends its write-ahead log on its own
schedule and cannot be charged per write, so a reserve (2%, capped at 32 MB) keeps the
directory genuinely under the number set.

### Storage durability

| Store | Role | If deleted |
|---|---|---|
| `thumbs/` | derived display copies | free to delete |
| `cache/` | fetched HTML, gzipped | rebuildable, but it is the **cheap cold source** — it is what lets a better parser recover a field not normalised today |
| `images/` | product photography, content-addressed | **durable.** Re-downloadable only while the retailer still serves that URL; listings get replaced, reordered and discontinued, and a saved design referring to `(sku, image sha256)` must still render years later |
| `*.jsonl` | canonical normalised observations | **durable** |
| `catalog.db` | query state | derived; rebuilt from JSONL by `load` |

The JSONL keeps each retailer's own embedded product payload verbatim under `raw`
(≈11 KB of a 15 KB IKEA record) alongside the normalised fields, so most re-parsing needs
no HTML. It does not keep the entire page, so `cache/` still holds information the JSONL
cannot reconstruct.

### Search index consistency

`products_fts` is maintained by triggers on `products`, not by application code. The two
were in step before, but only because one function happened to write both; a migration, a
fix-up script or a delete — which had no application path at all — could desynchronise
them, and an index that disagrees with the table it describes is an exceptionally confusing
bug to chase. `store.fts_drift()` reports divergence (missing / orphaned / stale) and is
included in `stats`; `rebuild_fts()` repairs an index damaged out of band.

---

## 10. Parser findings worth keeping

Each of these cost real coverage and none announced itself as an error.

| Finding | Effect |
|---|---|
| React flight text rows declare length in **UTF-8 bytes**, not characters, and are not newline-terminated | Counting characters walked past the next row header and truncated every `descriptionHtml` |
| IKEA's CDN serves **Brotli**; `requests` returns undecoded bytes as text without the decoder installed | HTTP 200, plausible body length, garbage content — looked like a parser bug |
| IKEA uses **`AggregateOffer`** for any price spread, which carries no availability of its own | All 534 products marked `unknown` stock, including 229 of 288 wardrobes |
| JSON-LD states `width`/`height`/`depth` for products whose measurements accordion omits them | 454 products recovered dimensions |
| A low/high price spread is an **IKEA Family member price**, not a configuration range (all 96 sampled carry `offercount: 1`, `priceOfferType: "family"`) | The regular price stays the quoted one; the member price is recorded beside it |
| `INSERT OR REPLACE` + `ON DELETE CASCADE` | Wiped the entire price series on every re-crawl |
| `content_hash` excluding derived fields | A changed taxonomy was silently discarded as "unchanged" |

---

## 11. Measured state

Snapshot at 2026-09-09, **crawl in progress** (IKEA ~6,800 of 12,425):

| | |
|---|---|
| products | 4,884 (ikea_in 4,282 · homerun 602) |
| families | 2,858 |
| searchable | 4,870 |
| placeable | 2,529 |
| image rows | 28,616 |
| price observations | 4,897 |
| FTS drift | 0 / 0 / 0 |
| tests | 59, network-free |

---

## 12. Known limitations and open items

1. **Weak-match rejection is lexical.** A candidate is rejected if it does not echo enough
   of the design's own label. This is a safety mechanism, not a confidence system:
   *"mid-century walnut bedside table"* could legitimately match retailer copy reading
   *"brown bedside cabinet"*, and a lexical requirement would reject an excellent visual
   match. It should become hard category and dimensional validation + semantic score +
   visual similarity + a reranker + a category-specific acceptance threshold — with
   `NO_MATCH` remaining a legitimate result.
2. **No embeddings computed.** `open_clip_torch` is not installed (~2.5 GB), so retrieval is
   keyword-only today. The vector schema, storage and fusion are built and tested.
3. **The live agent loop is unexercised.** The OpenAI account returns
   `429 no credits remaining`; the Responses API protocol handling is verified against a
   stubbed transport only.
4. **Regional availability is schema-only.** `product_availability` exists and is empty.
   Being in the catalogue is not the same as being buyable at a customer's pincode.
5. **`raw` is not the whole page.** It keeps the retailer's embedded product payload, not
   every field the HTML contains — hence `cache/` remains valuable.
6. **Crawl-time family interleaving is slug-based**, since the real family key requires a
   parsed page.

## 13. Next

Stop improving the crawler. Build against `search_catalog → get_product → price_design`
through the Responses API and let the data's real weaknesses surface — they will be more
informative than another round of crawler sophistication.

---

## Appendix: change history

| Commit | Change |
|---|---|
| `6e34fb4` | Scrapers for both sources, unified schema, SQLite store, hybrid retrieval, design pricing |
| `8cb13e3` | Web view; scheduled refresh via launchd; FTS alias bug; image path preservation |
| `ddb9d40` | Disk ceiling enforced at every large write, with a reserve |
| `39f56a6` | Category-stratified crawl ordering; `plan` and `coverage` |
| `dcda21d` | Completeness scoring; `searchable` vs `design_eligible`; AggregateOffer and dimension fixes |
| `a6a3fae` | Product families; embedding model versioning; price-freshness subtotals |
| `791423d` | Model tool boundary and Responses API loop; FTS maintained by triggers |
