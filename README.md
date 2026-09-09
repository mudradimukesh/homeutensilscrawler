# Interior Catalog

Scrapes Indian building-material and furniture catalogues into one product schema,
embeds the products and their photographs, and prices the object list an interior-design
model produces.

Two sources today:

| Source | What it sells | Catalogue size | Notes |
|---|---|---|---|
| `homerun` — [home-run.co](https://home-run.co) | construction, renovation and interior materials, Bengaluru quick commerce | ~610 products | Shopify behind a Next.js front end; live stock quantities and bulk-price tiers |
| `ikea_in` — [ikea.com/in/en](https://www.ikea.com/in/en) | furniture, lighting, textiles, decor | ~12,600 products | India catalogue only, so every price is INR and every product is actually orderable here |

Both expose complete structured data in the page itself, so there is no headless
browser and no DOM scraping: HomeRun ships its Shopify product object in the React
Server Components payload, IKEA ships `text/hydrate` JSON. That is why a full page
parse costs about a millisecond and why the parsers survive theme changes.

[**Design spec**](docs/DESIGN.md) — the data model, the crawl-ordering and completeness
decisions with the measurements behind them, the model tool boundary, and the open items.

## Upload application integration

The catalog API and persisted design manifests now support the upload workflow in
`interiorDesign`. See [setup, data reuse and API contract](docs/UPLOAD-INTEGRATION.md).
Product selection and pricing are exact-key operations; rendered product appearance
is checked separately and is not guaranteed by a generation prompt.

## Quickstart

```bash
pip install -r requirements.txt
```

```bash
python3 -m catalog scrape all --limit 30 && python3 -m catalog load && python3 -m catalog stats
```

Then, without any ML dependencies installed:

```bash
python3 -m catalog search "white pendant lamp for the ceiling" --no-vectors
```

```
 1. ZEBRASÄV Pendant lamp - white/plastic 46 cm (18 ")        Rs.499 each
    ikea_in | lighting | in_stock | https://www.ikea.com/in/en/p/zebrasaev-pendant-lamp-white-plastic-30580074/
 2. YTLÄGE Pendant lamp - white 43 cm (17 ")                Rs.3,490 each
```

## Browsing what has been catalogued

```bash
python3 -m catalog web
```

Opens a browser view of the database at <http://127.0.0.1:8765> — grid or table, faceted by
source, design category, brand, availability and price, with full-text search across names,
materials and colours. Clicking a product shows everything stored for it: every image, the
parsed dimensions, materials, variants with their bulk-price tiers, the price history, which
embeddings exist, and the raw payload exactly as it was scraped. The sidebar carries a
refresh history so you can see when the data last moved.

Standard library only — no server framework. It reads the same SQLite file the CLI writes,
so it always shows the current state; leave it running while a crawl loads and reload.

## The pipeline

```
scrape  ->  data/<source>.jsonl     one JSON object per product, append-only, resumable
load    ->  data/catalog.db         normalised tables + FTS5 + price history
images  ->  data/images/<sha256>    deduplicated by content hash
embed   ->  embeddings table        CLIP vectors for product text and photographs
search / price                      what the design AI calls
```

JSONL is the canonical artefact rather than a direct database write, so a crawl can be
replayed into a new schema, diffed day to day, or re-parsed after a parser fix without
touching either site again:

```bash
python3 -m catalog scrape homerun --reparse   # rebuilds from the page cache, zero requests
```

### Full crawl

```bash
python3 -m catalog scrape homerun --delay 1.0 --workers 4     # ~610 pages, ~10 min
python3 -m catalog scrape ikea_in --delay 1.0 --workers 4     # ~12.6k pages, ~3.5 h
python3 -m catalog load
python3 -m catalog images --per-product 3                     # ~38k images, ~3.5 GB
python3 -m catalog embed
```

`--delay` is the per-host floor between requests, so extra workers do not raise the
request rate — they only hide latency. Crawls resume: an interrupted run picks up from the
URLs already in the JSONL, and pages are cached on disk, so a re-run after a parser change
costs nothing.

## Keeping it fresh

A first crawl and a re-crawl are not the same job. `scrape` resumes past what it already has
and serves pages from cache — right when you are building a parser, wrong for a catalogue
whose prices move daily. `refresh` does the opposite: it re-fetches, rewrites each source's
JSONL rather than appending, loads, pulls new images, embeds what is new, and reports what
actually changed.

```bash
python3 -m catalog refresh                     # one full cycle
python3 -m catalog refresh --stale-after 6h    # reuse pages fetched in the last 6 hours
```

```
refresh #3 ok in 0.8s
  pages   ok=30 failed=0
  products new=0 changed=2
  images  +0   embeddings +0
  price changes (2):
    Maha PPC Cement, 50 Kg Bag         300 -> 355
    Ultratech PPC Cement, 50 Kg Bag    450 -> 410
```

Run it on a schedule:

```bash
python3 -m catalog schedule install --at 03:30
python3 -m catalog schedule status
python3 -m catalog schedule uninstall
```

On macOS this installs a **launchd** LaunchAgent, not a cron job — it survives reboots and
logouts, and launchd runs a calendar job it missed while the machine was asleep, which cron
does not. A laptop is asleep at 03:30 most nights, so that difference is the whole point.
The job runs at background priority with low-priority I/O so a crawl never competes with
your own work. `--print-only` emits the equivalent systemd timer or crontab line for a
Linux box, and `catalog watch --every 12h` is the in-process loop for a container.

Every run is recorded in a `crawl_runs` table — start, finish, pages, new and changed
products, reprices, failures — which is what `schedule status` and the web view read.
Two refreshes cannot overlap: a PID lock file guards the crawl, and a lock left behind by a
dead process is reclaimed rather than obeyed, so one crash does not stop tomorrow's run.

Prices are never overwritten in place. Every observed change appends to `price_history`, so
a quote issued last week can still be explained.

### The disk limit

Every command that writes stops at **1 GB** of `data/` by default. This is a real constraint,
not a formality: the full catalogue runs to roughly 3.5 GB of images and 2 GB of cached
pages, so an unattended nightly job would otherwise fill a laptop.

```bash
python3 -m catalog du                          # what is using the space
python3 -m catalog refresh --max-data-size 8GB # raise it
python3 -m catalog refresh --max-data-size 0   # no limit
```

The limit is enforced *while writing*, not checked afterwards, because a check afterwards is
just a report of a disk that is already full. Each large writer — the page cache, the image
store, the JSONL — asks the budget before it writes and stops the run when the next write
would cross the line. What has been crawled is kept, the database stays consistent, and the
run is recorded with status `over_budget`, so a nightly job that hits the ceiling shows up in
`schedule status` and in the web view rather than failing silently.

Writers stop slightly below the ceiling. SQLite extends its write-ahead log on its own
schedule and cannot be charged per write, so a small reserve keeps the directory genuinely
under the number you set instead of settling just over it.

Measuring the tree on every write would cost more than the writes, so the total is measured
once and tracked as bytes are written; the walk is repeated only near the limit, where drift
would otherwise decide the outcome.

When you hit it, the stores are not equally disposable:

| Store | Role | If deleted |
|---|---|---|
| `thumbs/` | derived display copies | free to delete, rebuilt on demand |
| `cache/` | fetched HTML, gzipped | rebuildable, but it is the **cheap cold source**: it is what lets you improve the parser tomorrow and recover a field you did not normalise today, without re-crawling |
| `images/` | product photography, content-addressed | **treat as durable**. Re-downloadable only while the retailer still serves that URL — listings get replaced, reordered and discontinued, and a saved design that refers to `(sku, image sha256)` must still render years later |
| `*.jsonl` | canonical normalised observations | **durable**. The replayable record of what each crawl saw |
| `catalog.db` | query state plus saved design sessions, manifests and assets | product query tables can be rebuilt; back up the DB to preserve application state |

Delete `thumbs/` first, then `cache/` if you must. `images/` and the JSONL are the snapshot.

A caveat on the JSONL's completeness: each record keeps the retailer's own embedded product
payload verbatim under `raw` — IKEA's sales-item block and JSON-LD, HomeRun's Shopify object —
alongside the normalised fields, so most re-parsing needs no HTML. It does not keep the entire
page, so `cache/` still holds information the JSONL cannot reconstruct.

## Vector DB or a normal DB?

**Both, in one database: Postgres with `pgvector`. Not a standalone vector store.**
For now, the SQLite file this repo writes by default is enough, and moving to Postgres
is a schema swap under `catalog/store.py`, not a rewrite.

The reasoning, since this decision shapes everything downstream:

**Nothing in this system is only a similarity question.** The design AI never asks "what
looks like this". It asks "what looks like this, *is* a sofa, *is* in stock, and costs
*under* Rs. 40,000". Price and stock are exact constraints — an approximate answer to them
produces a beautiful room the customer cannot buy. In Postgres that is one query, filter
and vector rank together. Split across a separate vector service it becomes two round
trips plus reconciliation: pre-filter and you must ship the whole ID set to the vector
store, post-filter and the constraint silently eats your top-k until nothing is left.

**The pricing step is a key lookup, not a search.** Once a design is approved, resolving
each object to a price must be exact and reproducible — the same design costed twice must
give the same number, and you need `price_history` to say what a quote was based on. Vector
stores are the wrong system of record for that: no joins, no transactions, weak
constraints, and every metadata edit means rewriting a vector payload.

**Prices move constantly and vectors do not.** This is quick commerce; a price or stock
level changes daily while the product photograph does not change for months. Splitting
them means every price tick re-touches the vector store for no reason. Here a reprice is
an `UPDATE` and one `price_history` row; the embedding is untouched.

**The catalogue is small.** ~13k products and ~38k images. At CLIP ViT-B/32 (512-d
float32) that is about 80 MB of vectors — they fit in RAM with room to spare, and a
brute-force scan is under 20M multiply-adds, i.e. single-digit milliseconds in NumPy.
`pgvector` with an HNSW index is far quicker still. A dedicated vector database starts
earning its licence and its operational surface somewhere north of ten million vectors.
You are three orders of magnitude away.

**What the vector part genuinely buys you** is the one thing SQL cannot do: CLIP puts
photographs and text in the same 512-dimensional space, so the design model can hand back
a *cropped region of its own render* and have it compared directly against catalogue
photography — no captioning step in between, and no vocabulary agreement needed between
the two models. That is worth having. It just does not need its own server.

So: `products` and `product_images` each carry a `vector(512)` column, an HNSW index sits
next to the ordinary B-tree indexes on price and category, and one query does the whole
job. `catalog/postgres.sql` has the schema and the exact query shape.

**When to actually move.** Not simply "more than one process" — SQLite serves many concurrent
readers well and handles multiple processes fine under a single-writer pattern, which is what a
nightly crawl plus a read-only API already is. Move when you need concurrent *writers*, remote
database access, higher API concurrency, distributed workers, vector retrieval past a few
million rows, or operational HA.

### The vector layer is versioned

Nothing assumes one embedding per product. `embeddings` is keyed by
`(product_key, kind, source_asset, model, model_version)`, so a new representation can be
computed *beside* CLIP, compared on the same catalogue, and promoted — without destroying the
index currently serving. Text and image vectors are separate rows rather than one blended
vector, and `source_asset` is the content hash of exactly what was embedded, so an edited
description or a replaced photograph yields a new vector instead of silently reusing a stale
one.

## Families, not SKUs

MALM bed white 140, MALM bed white 160 and MALM bed black 180 are three SKUs and one piece of
furniture. Ask for five beds off raw SKUs and you can get five sizes of one range — measured on
the live catalogue, wardrobes average **five SKUs per family**, and 393 wardrobe SKUs are only
68 families.

Every product carries a `family_key`, so retrieval can collapse to one result per model line:

```bash
python3 -m catalog search "wardrobe with doors" --collapse
python3 -m catalog family ikea_in:pax-gullaberg|wardrobe-combination
```

"I like this bed but I need the 180 cm version" is then a variant lookup, not another semantic
search — the design has already chosen the product.

`catalog coverage` reports both counts, because the second is the one that matters: a category
with 397 SKUs and 127 families offers 127 choices, and if only 110 of those families are
placeable it offers 110.

## How the design AI uses it

**Constraining generation.** The design model may only place products that exist in the
catalogue. Retrieve a shortlist per object slot and put those products — with their real
dimensions, colours and materials — into the generation prompt:

```bash
python3 -m catalog search "grey fabric 3 seater sofa" --category sofa --max-price 40000 --in-stock --json
```

**Pricing the finished design.** The model emits an object list; this turns it into a
costed bill of materials:

```bash
python3 -m catalog price examples/design.json
```

```
subtotal Rs.14,469.00  priced=5  unmatched=1  low_conf=1
pendant lamp             x1   conf=1.0  ZEBRASÄV Pendant lamp - white/plastic   Rs.  499
wardrobe                 x1   conf=1.0  VUKU Wardrobe - white 74x51x149 cm      Rs.1,590
interior wall primer     x2   conf=1.0  Asian Paints TruCare Interior Primer    Rs.6,500
brass chandelier ...     x1   conf=0.0  -- no match cleared the confidence floor
```

Each object may carry `design_category`, `max_price`, `source`, free-text hints
(`color`, `material`, `style`), and `crop` — a path to the cropped region of the render,
which is matched visually against catalogue photographs.

Two behaviours worth knowing about, because they are deliberate:

- **Distinct objects get distinct products.** A room does not come back priced as six
  copies of the same chair.
- **A line with no convincing match is reported, not guessed.** Rank fusion always returns
  *something*, so a "brass chandelier with crystal drops" would otherwise be quoted as
  whatever light fitting ranked first. Any candidate that does not echo enough of the
  design's own label is rejected and listed under `alternates` for a human to choose from.
  Tune with `--min-confidence`; a quote with a missing line is recoverable, a quote with a
  confidently wrong line is not.

`price_design()` and `find_products()` in `catalog/search.py` are the two functions to
call from application code; neither requires the CLI.

## Embeddings

CLIP is optional — scraping, loading and keyword search all work without it, and
`--no-vectors` forces the keyword path.

```bash
pip install open_clip_torch torch pillow    # ~2.5 GB
python3 -m catalog embed
```

Text embeddings are built from `Product.embedding_text()`, which orders fields
most-distinguishing first because CLIP truncates at 77 tokens. Image embeddings cover the
first N photographs per product; for IKEA those are sorted so the main and in-room context
shots come first, which are the ones an inspiration image actually resembles.

Without CLIP, retrieval is FTS5 BM25 with the product name weighted 10× against the
description — HomeRun ships 5,000-character SEO descriptions that otherwise swamp every
query. With CLIP, keyword and vector rankings are combined by reciprocal rank fusion.

## Adding a source

Implement `discover()` and `parse()` in `catalog/sources/`, register the class in
`catalog/sources/__init__.py`, and everything downstream works unchanged. Look for
structured data before reaching for selectors: JSON-LD, a hydration payload, or an
internal JSON endpoint is almost always present and always more stable than markup.

## Crawling politely

`robots.txt` is honoured by default (both sites allow product pages; HomeRun disallows
`/search` and `/cart`, IKEA disallows search and filter URLs — none of which are crawled).
Requests are rate-limited per host with jitter, retried with exponential backoff, and
`429` is respected via `Retry-After`. Default is one request per second per host.

The IKEA India catalogue is read from the `prod-en-IN_*` sitemaps only, so nothing outside
the Indian catalogue — and no non-INR pricing — can enter the database.

Prices, stock and product copy belong to the retailers. This is built for internal
catalogue and pricing use; if any of it becomes customer-facing, check the terms of both
sites and attribute prices to their source with the timestamp `price_history` records.

## Layout

```
catalog/
  http.py         rate limiting, robots.txt, retries, on-disk page cache
  models.py       unified Product schema, dimension parsing, design taxonomy
  sources/        one module per retailer
  pipeline.py     discover -> fetch -> parse -> JSONL, resumable
  store.py        SQLite schema, upsert, price history
  images.py       content-addressed image downloads
  embed.py        CLIP text and image embeddings
  search.py       hybrid retrieval + design pricing
  budget.py       the disk ceiling, enforced at every large write
  quality.py      completeness scoring: searchable vs design_eligible
  family.py       model-line resolution, so variants collapse
  refresh.py      one scheduled refresh cycle, with a lock and a change report
  schedule.py     launchd agent (systemd/cron equivalents for Linux)
  web.py          read-only JSON API + the browser view, standard library only
  static/         the single-page UI
  postgres.sql    the same schema on Postgres + pgvector
tests/            49 network-free tests: python3 tests/test_catalog.py
```
