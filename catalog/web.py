"""A local web view over the catalogue database.

Standard library only — the point of this project is a catalogue, not a web
framework, and a browser for it should not drag in a server dependency. It reads
the same SQLite file the CLI writes, so it shows whatever the last crawl loaded.
"""
from __future__ import annotations

import json
import logging
import mimetypes
import sqlite3
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from .store import connect, stats

log = logging.getLogger(__name__)

STATIC = Path(__file__).resolve().parent / "static"
PAGE_SIZE = 48

# Same weighting the search module uses: the name is what a query is aiming at,
# and HomeRun's 5 000-character descriptions otherwise swamp every match.
_BM25 = "0.0, 10.0, 4.0, 3.0, 1.0, 2.0, 2.0, 1.5"
_STOPWORDS = frozenset("a an the and or of for with in on to at by from is are be this that it".split())

_SORTS = {
    "relevance": "relevance ASC, p.name ASC",
    "price_asc": "p.price IS NULL, p.price ASC",
    "price_desc": "p.price IS NULL, p.price DESC",
    "name": "p.name ASC",
    "newest": "p.first_seen DESC, p.name ASC",
    "updated": "p.last_seen DESC, p.name ASC",
}

_JSON_FIELDS = (
    "category_path", "dimensions", "dimension_text", "materials", "colors",
    "tags", "attributes", "raw",
)


def _loads(value, default):
    if not value:
        return default
    try:
        return json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return default


def _row_to_product(row: sqlite3.Row) -> dict:
    d = {k: row[k] for k in row.keys()}
    for f in _JSON_FIELDS:
        if f in d:
            d[f] = _loads(d[f], [] if f.endswith(("path", "text")) or f in
                          ("materials", "colors", "tags") else {})
    return d


def _fts_match(q: str) -> str | None:
    words = [w for w in "".join(c if c.isalnum() else " " for c in q.lower()).split()
             if len(w) > 1 and w not in _STOPWORDS]
    return " OR ".join(f'"{w}"' for w in dict.fromkeys(words)) or None


class Catalog:
    """Thread-local SQLite handles; the stdlib server is threaded."""

    def __init__(self, db_path: str | Path, image_dir: str | Path):
        self.db_path = str(db_path)
        self.image_dir = Path(image_dir)
        self.thumb_dir = self.image_dir.parent / "thumbs"
        self._local = threading.local()

    @property
    def conn(self) -> sqlite3.Connection:
        c = getattr(self._local, "conn", None)
        if c is None:
            c = connect(self.db_path)
            self._local.conn = c
        return c

    # -- endpoints ---------------------------------------------------------
    def summary(self) -> dict:
        conn = self.conn
        base = stats(conn)
        row = conn.execute(
            "SELECT MIN(price) lo, MAX(price) hi, MAX(last_seen) seen FROM products "
            "WHERE price IS NOT NULL"
        ).fetchone()
        base["price_min"] = row["lo"]
        base["price_max"] = row["hi"]
        base["last_crawl"] = row["seen"]
        base["database"] = self.db_path
        base["engine"] = "SQLite + FTS5"
        base["brands"] = [
            r["brand"] for r in conn.execute(
                "SELECT brand FROM products WHERE brand IS NOT NULL AND brand <> '' "
                "GROUP BY brand ORDER BY COUNT(*) DESC LIMIT 40")
        ]
        base["availability"] = {
            r["availability"]: r["c"] for r in conn.execute(
                "SELECT availability, COUNT(*) c FROM products GROUP BY 1 ORDER BY c DESC")
        }
        base["price_observations"] = conn.execute(
            "SELECT COUNT(*) c FROM price_history").fetchone()["c"]
        return base

    def products(self, qs: dict) -> dict:
        conn = self.conn
        one = lambda k, d=None: (qs.get(k) or [d])[0]  # noqa: E731

        q = (one("q") or "").strip()
        page = max(int(one("page", "1") or 1), 1)
        per_page = min(max(int(one("per_page", str(PAGE_SIZE)) or PAGE_SIZE), 1), 200)
        sort = one("sort", "relevance" if q else "name")
        if sort not in _SORTS or (sort == "relevance" and not q):
            sort = "relevance" if q else "name"

        where, params = ["1=1"], []
        for field, key in (("source", "source"), ("design_category", "category"),
                           ("availability", "availability"), ("brand", "brand")):
            val = one(key)
            if val:
                where.append(f"p.{field} = ?")
                params.append(val)
        for key, op in (("min_price", ">="), ("max_price", "<=")):
            val = one(key)
            if val not in (None, ""):
                where.append(f"p.price IS NOT NULL AND p.price {op} ?")
                params.append(float(val))
        if one("has_image") == "1":
            where.append("EXISTS (SELECT 1 FROM product_images i "
                         "WHERE i.product_key = p.key AND i.local_path IS NOT NULL)")
        if one("unpriced") == "1":
            where.append("p.price IS NULL")

        match = _fts_match(q) if q else None
        if match:
            # No alias on products_fts: FTS5 resolves MATCH and bm25() against the
            # table name itself, and an alias fails with "no such column".
            src = "FROM products_fts JOIN products p ON p.key = products_fts.key"
            where.append("products_fts MATCH ?")
            params.append(match)
            rel = f"bm25(products_fts, {_BM25})"
        else:
            src = "FROM products p"
            rel = "NULL"

        clause = " AND ".join(where)
        total = conn.execute(f"SELECT COUNT(*) c {src} WHERE {clause}", params).fetchone()["c"]

        rows = conn.execute(
            f"""SELECT p.key, p.source, p.name, p.brand, p.price, p.compare_at_price,
                       p.currency, p.price_unit, p.availability, p.design_category,
                       p.category, p.url, p.rating, p.review_count, p.sku,
                       p.width_mm, p.height_mm, p.depth_mm, p.length_mm, p.diameter_mm,
                       p.dimension_text, p.colors, p.materials, p.last_seen,
                       {rel} AS relevance,
                       (SELECT i.url FROM product_images i WHERE i.product_key = p.key
                         ORDER BY i.position LIMIT 1) AS image_url,
                       (SELECT i.sha256 FROM product_images i WHERE i.product_key = p.key
                         ORDER BY i.position LIMIT 1) AS image_sha,
                       (SELECT COUNT(*) FROM product_images i WHERE i.product_key = p.key)
                         AS image_count
                  {src} WHERE {clause}
                 ORDER BY {_SORTS[sort]} LIMIT ? OFFSET ?""",
            params + [per_page, (page - 1) * per_page],
        ).fetchall()

        items = []
        for r in rows:
            d = {k: r[k] for k in r.keys()}
            for f in ("dimension_text", "colors", "materials"):
                d[f] = _loads(d[f], [])
            items.append(d)

        return {"total": total, "page": page, "per_page": per_page,
                "pages": max((total + per_page - 1) // per_page, 1),
                "sort": sort, "items": items}

    def runs(self, limit: int = 12) -> list[dict]:
        return [dict(r) for r in self.conn.execute(
            "SELECT id, source, started_at, finished_at, status, pages_ok, pages_failed, "
            "products_new, products_changed, price_changes, images_downloaded, error "
            "FROM crawl_runs ORDER BY started_at DESC LIMIT ?", (limit,))]

    def facets(self) -> dict:
        conn = self.conn
        pick = lambda sql: [dict(r) for r in conn.execute(sql)]  # noqa: E731
        return {
            "sources": pick("SELECT source AS value, COUNT(*) AS n FROM products "
                            "GROUP BY 1 ORDER BY n DESC"),
            "categories": pick("SELECT COALESCE(design_category,'(unclassified)') AS value, "
                               "COUNT(*) AS n FROM products GROUP BY 1 ORDER BY n DESC"),
            "availability": pick("SELECT availability AS value, COUNT(*) AS n FROM products "
                                 "GROUP BY 1 ORDER BY n DESC"),
            "brands": pick("SELECT brand AS value, COUNT(*) AS n FROM products "
                           "WHERE brand IS NOT NULL AND brand <> '' GROUP BY 1 "
                           "ORDER BY n DESC LIMIT 50"),
        }

    def product(self, key: str) -> dict | None:
        conn = self.conn
        row = conn.execute("SELECT * FROM products WHERE key = ?", (key,)).fetchone()
        if row is None:
            return None
        d = _row_to_product(row)
        d["images"] = [dict(r) for r in conn.execute(
            "SELECT position, url, alt, width, height, role, local_path, sha256 "
            "FROM product_images WHERE product_key = ? ORDER BY position", (key,))]
        d["variants"] = []
        for r in conn.execute(
            "SELECT * FROM product_variants WHERE product_key = ? ORDER BY variant_id", (key,)
        ):
            v = dict(r)
            v["options"] = _loads(v.get("options"), {})
            v["bulk_pricing"] = _loads(v.get("bulk_pricing"), [])
            d["variants"].append(v)
        d["price_history"] = [dict(r) for r in conn.execute(
            "SELECT observed_at, price, availability FROM price_history "
            "WHERE product_key = ? ORDER BY observed_at", (key,))]
        d["embeddings"] = [dict(r) for r in conn.execute(
            "SELECT kind, model, dim, COUNT(*) AS n FROM embeddings "
            "WHERE product_key = ? GROUP BY kind, model, dim", (key,))]
        return d

    # -- images ------------------------------------------------------------
    def image(self, sha: str, width: int | None) -> tuple[bytes, str] | None:
        row = self.conn.execute(
            "SELECT local_path FROM product_images WHERE sha256 = ? AND local_path IS NOT NULL "
            "LIMIT 1", (sha,)).fetchone()
        if row is None:
            return None
        path = Path(row["local_path"])
        if not path.exists():
            return None
        if not width:
            return path.read_bytes(), mimetypes.guess_type(path.name)[0] or "image/jpeg"

        cached = self.thumb_dir / str(width) / f"{sha}.jpg"
        if cached.exists():
            return cached.read_bytes(), "image/jpeg"
        try:
            from PIL import Image
        except ImportError:      # thumbnails are a nicety, not a requirement
            return path.read_bytes(), mimetypes.guess_type(path.name)[0] or "image/jpeg"
        img = Image.open(path).convert("RGB")
        img.thumbnail((width, width))
        cached.parent.mkdir(parents=True, exist_ok=True)
        img.save(cached, "JPEG", quality=82)
        return cached.read_bytes(), "image/jpeg"


class Handler(BaseHTTPRequestHandler):
    catalog: Catalog = None          # injected by serve()
    server_version = "InteriorCatalog/0.1"

    def log_message(self, fmt, *args):      # keep the console readable
        log.debug("%s - %s", self.address_string(), fmt % args)

    def _send(self, body: bytes, ctype: str, status: int = 200, cache: str | None = None):
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        if cache:
            self.send_header("Cache-Control", cache)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _json(self, payload, status: int = 200):
        self._send(json.dumps(payload, ensure_ascii=False, default=str).encode(),
                   "application/json; charset=utf-8", status)

    def do_GET(self):
        parts = urlparse(self.path)
        route, qs = parts.path, parse_qs(parts.query)
        try:
            if route in ("/", "/index.html"):
                return self._static("index.html")
            if route.startswith("/static/"):
                return self._static(route[len("/static/"):])
            if route == "/api/summary":
                return self._json(self.catalog.summary())
            if route == "/api/runs":
                return self._json(self.catalog.runs())
            if route == "/api/facets":
                return self._json(self.catalog.facets())
            if route == "/api/products":
                return self._json(self.catalog.products(qs))
            if route == "/api/product":
                key = (qs.get("key") or [""])[0]
                found = self.catalog.product(key)
                return self._json(found or {"error": "not found"}, 200 if found else 404)
            if route == "/api/image":
                sha = (qs.get("sha") or [""])[0]
                width = int((qs.get("w") or ["0"])[0] or 0) or None
                got = self.catalog.image(sha, width)
                if not got:
                    return self._json({"error": "no local file"}, 404)
                blob, ctype = got
                return self._send(blob, ctype, cache="public, max-age=86400")
            self._json({"error": "not found", "path": route}, 404)
        except BrokenPipeError:
            pass                                   # browser navigated away mid-response
        except Exception as exc:
            log.exception("%s failed", route)
            self._json({"error": exc.__class__.__name__, "detail": str(exc)}, 500)

    do_HEAD = do_GET

    def _static(self, name: str):
        # Resolve inside STATIC so a crafted path cannot escape the directory.
        path = (STATIC / name).resolve()
        if not str(path).startswith(str(STATIC.resolve())) or not path.is_file():
            return self._json({"error": "not found"}, 404)
        ctype = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        if ctype.startswith("text/") or ctype.endswith(("javascript", "json")):
            ctype += "; charset=utf-8"
        self._send(path.read_bytes(), ctype)


def serve(db_path, image_dir, host: str = "127.0.0.1", port: int = 8765) -> None:
    Handler.catalog = Catalog(db_path, image_dir)
    httpd = ThreadingHTTPServer((host, port), Handler)
    summary = Handler.catalog.summary()
    print(f"\n  Interior Catalog  ->  http://{host}:{port}")
    print(f"  {summary['products']} products, {summary['images']} images "
          f"({summary['images_downloaded']} local), {summary['embeddings']} embeddings")
    print(f"  SQLite: {db_path}\n  Ctrl-C to stop.\n")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n  stopped")
    finally:
        httpd.server_close()
