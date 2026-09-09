# Catalog-constrained upload designs

The upload integration selects exact catalog keys before image generation. It saves
an immutable manifest, sends the corresponding catalog photographs to the image
provider, records an advisory visual review, and rechecks stock and prices before
returning product details. The selected product list is authoritative; a generated
image is still an approximation, and a visual pass is not proof of SKU identity.

## Run locally

Use this branch together with `feat/catalog-upload-designs` in `interiorDesign`.
Install the existing dependencies, including CLIP for visual search:

```sh
python3 -m pip install -r requirements.txt
python3 -m pip install open_clip_torch torch pillow
```

Set a random `CATALOG_API_TOKEN` of at least 24 characters in the service environment.
Set the same value in the interior app's `.env.local`; do not prefix it with `VITE_`.
The app's Vite server injects the token into requests to the local catalog API.

```sh
python3 -m catalog.design_api --db data/catalog.db --images data/images
```

The API binds only to `127.0.0.1:8766`. It requires bearer authentication and accepts
bounded JSON POST bodies. It is a single-user local service, not a public multi-tenant
API. A hosted app needs a server-side authenticated proxy and user-to-session ownership
checks; a static Vite build alone does not provide this backend.

## Existing data: no full scrape required

The schema upgrade is additive and automatic on connection. Existing product records,
FTS data, downloaded assets and compatible CLIP embeddings remain usable. No parser
or embedding-model migration is introduced by this change.

The inspected local database on 2026-09-09 contained 13,172 products, 67,449 image
records, 3,178 downloaded images and zero embedding rows. Resume image downloads and
compute embeddings from those assets. The image command is resumable; embedding skips
already computed matching representations.

```sh
python3 -m catalog images --per-product 3 --max-data-size 8GB
python3 -m catalog embed --images-per-product 3
```

Choose a disk budget suitable for the machine. The existing 1 GB default can stop a
full catalog image download. This work does not launch a download, refresh or paid
OpenAI request. Image records do not imply every image must be downloaded: three good
views per product are a useful starting point. Missing dimensions require enrichment
or re-parsing if the original cached data contains them; re-scraping alone cannot
supply dimensions absent from the retailer's data.

Use the normal `refresh` workflow when prices/stock need updating. The service checks
availability and the existing seven-day freshness policy at request time. That is a
check against the latest stored observation, not a live retailer stock reservation or
a delivery-pincode guarantee. Do not reload old JSONL solely to make timestamps fresh.

## Backend contract

All routes use `/api/catalog/` and a bearer token. `session` creates a random session
identifier. Subsequent bodies include `session_id` supplied by the application.

| Operation | Input beyond session | Result |
|---|---|---|
| `image` | base64 `image` | Opaque image ID for this session |
| `search` | `query` object with category, text, optional image ID and dimensional limits | Filtered, family-deduplicated candidates |
| `product` | `product_key` | Readable metadata and variants |
| `manifest` | keyed `objects`, optional `budget`, optional stable `design_id` | Immutable validated selection |
| `get-manifest` | `design_id` | Saved selection and latest review |
| `asset` | `design_id`, content hash `asset` | Original snapshotted image bytes as base64 and media type |
| `finalize` | `design_id`, rendered `image`, structured `verification` | Review tied to render hash, plus current quote |

Hard dimensions exclude unknown axes before ranking and family collapse. Keyword
limits apply after SQL constraints too. The image tool accepts registered IDs only,
never arbitrary filesystem paths or external URLs. Image queries fail explicitly if
image embeddings cannot run, rather than returning the cheapest product as a visual
match. Text-only searches can fall back to keyword retrieval.

`CatalogService` defaults to CLIP when matching embeddings exist and only `in_stock`
products; store-only items require an application-level policy override. The upload
flow requests downloaded render assets and excludes unresolved multi-option product
records. Each selected purchase variant must resolve to one exact product key.

Pricing rejects missing/unissued keys, invalid quantities, unavailable products and
stale/unknown prices. Manifests additionally validate dimensions, placement eligibility,
asset content hashes and the combined INR budget. Product selection snapshots retain
metadata and image bytes through later catalog changes. Reusing a design ID with the
same request returns its snapshot; a different selection is rejected.

Finalization requires one review entry per slot, extra-object reporting and a room
preservation result. It stores the rendered bytes and SHA-256, rechecks current stock,
prices and budget, and returns `passed_visual_review` or `needs_review`. Review is an
application operation, not a catalog tool available to the product-planning model.
The HTTP client is trusted to supply this advisory review; it is not cryptographic
attestation that a third-party model examined the pixels.

## Persistence and tests

Back up the SQLite database as application data now. The new `catalog_sessions`,
`catalog_issued`, `catalog_query_images`, `design_manifests` and `design_assets` tables
cannot be reconstructed from crawler JSONL. Use SQLite's backup API for a consistent
copy if WAL mode is active. Session IDs must be restored by the application when
resuming a request. The API exposes saved manifests and assets; it is not a job queue.

```sh
python3 tests/test_catalog.py
python3 tests/test_design_workflow.py
```

The tests use local fixtures and fake embeddings. They cover hard filtering, visual
retrieval, exact-key pricing, durable sessions/manifests, asset isolation, advisory
review, authentication and stock changes. Live retrieval quality and image fidelity
must be evaluated after completing catalog images and embeddings.
