"""Command line: scrape -> load -> images -> embed -> search / price."""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from . import store
from .embed import DEFAULT_MODEL, DEFAULT_PRETRAINED, Embedder, embed_catalog
from .http import Fetcher
from .images import download_missing
from .pipeline import scrape
from .budget import DEFAULT_LIMIT, DiskBudget, parse_size, report as disk_report
from .stratify import coverage, format_coverage, format_summary, plan_summary
from .taxonomy import DEFAULT_WEIGHTS
from .search import dump, find_products, get_family, price_design
from .sources import SOURCES

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
DEFAULT_DB = DATA / "catalog.db"
DEFAULT_CACHE = DATA / "cache"
DEFAULT_IMAGES = DATA / "images"


def _weights(path: str | None) -> dict[str, int] | None:
    """Category weights from a JSON file, merged over the defaults."""
    if not path:
        return None
    overrides = json.loads(Path(path).read_text(encoding="utf-8"))
    return {**DEFAULT_WEIGHTS, **{k: int(v) for k, v in overrides.items()}}


def _embedder(args) -> Embedder | None:
    if getattr(args, "no_vectors", False):
        return None
    return Embedder(args.model, args.pretrained)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="catalog", description=__doc__)
    ap.add_argument("-v", "--verbose", action="store_true")
    ap.add_argument("--db", default=str(DEFAULT_DB))
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("scrape", help="crawl a source into JSONL")
    p.add_argument("source", choices=sorted(SOURCES) + ["all"])
    p.add_argument("--limit", type=int, help="stop after N products (for a smoke test)")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--delay", type=float, default=1.0, help="seconds between requests per host")
    p.add_argument("--out", help="JSONL path (default data/<source>.jsonl)")
    p.add_argument("--cache", default=str(DEFAULT_CACHE))
    p.add_argument("--force", action="store_true", help="refetch pages already cached")
    p.add_argument("--reparse", action="store_true",
                   help="rebuild the JSONL from cached pages after a parser change (no requests)")
    p.add_argument("--no-robots", action="store_true", help="skip robots.txt checks")
    p.add_argument("--strategy", choices=["stratified", "sitemap"], default="stratified",
                   help="stratified keeps every prefix of the crawl category-balanced")
    p.add_argument("--weights", help="JSON file of category weights, overriding the defaults")
    p.add_argument("--max-data-size", default="1GB",
                   help="stop when data/ reaches this size (default 1GB; 0 = no limit)")

    p = sub.add_parser("load", help="load JSONL into the database")
    p.add_argument("files", nargs="*", help="default: every data/*.jsonl")

    p = sub.add_parser("images", help="download product images")
    p.add_argument("--limit", type=int)
    p.add_argument("--per-product", type=int, default=3)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--delay", type=float, default=0.3)
    p.add_argument("--dir", default=str(DEFAULT_IMAGES))
    p.add_argument("--max-data-size", default="1GB",
                   help="stop when data/ reaches this size (default 1GB; 0 = no limit)")


    p = sub.add_parser("embed", help="compute CLIP embeddings")
    p.add_argument("--text-only", action="store_true")
    p.add_argument("--images-only", action="store_true")
    p.add_argument("--images-per-product", type=int, default=2)
    p.add_argument("--limit", type=int)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--pretrained", default=DEFAULT_PRETRAINED)

    p = sub.add_parser("search", help="hybrid product search")
    p.add_argument("query", nargs="?", default=None)
    p.add_argument("--image", help="path to a query image (a crop of the render)")
    p.add_argument("--category", help="design category, e.g. sofa / lighting / tile")
    p.add_argument("--max-price", type=float)
    p.add_argument("--min-price", type=float)
    p.add_argument("--source", choices=sorted(SOURCES))
    p.add_argument("--in-stock", action="store_true")
    p.add_argument("--eligible", action="store_true",
                   help="only products complete enough to place in a room")
    p.add_argument("--collapse", action="store_true",
                   help="one result per model line, not five sizes of the same range")
    p.add_argument("-k", type=int, default=10)
    p.add_argument("--json", action="store_true")
    p.add_argument("--no-vectors", action="store_true", help="keyword only, no CLIP")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--pretrained", default=DEFAULT_PRETRAINED)

    p = sub.add_parser("price", help="cost a design's object list")
    p.add_argument("design", help="JSON file: a list of objects, or {\"objects\": [...]}")
    p.add_argument("--allow-out-of-stock", action="store_true")
    p.add_argument("--alternates", type=int, default=3)
    p.add_argument("--min-confidence", type=float, default=0.34,
                   help="share of a design label's words that must appear in the match")
    p.add_argument("--no-vectors", action="store_true")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--pretrained", default=DEFAULT_PRETRAINED)

    p = sub.add_parser("refresh", help="one scheduled-refresh cycle: crawl, load, images, embed")
    p.add_argument("--sources", default="all")
    p.add_argument("--stale-after", default="0",
                   help="reuse a cached page younger than this (e.g. 6h); 0 always refetches")
    p.add_argument("--delay", type=float, default=1.0)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--limit", type=int)
    p.add_argument("--skip-images", action="store_true")
    p.add_argument("--skip-embed", action="store_true")
    p.add_argument("--json", action="store_true")
    p.add_argument("--max-data-size", default="1GB",
                   help="stop when data/ reaches this size (default 1GB; 0 = no limit)")

    p = sub.add_parser("watch", help="run refresh on a loop (containers; prefer `schedule` on macOS)")
    p.add_argument("--every", default="24h")
    p.add_argument("--sources", default="all")
    p.add_argument("--stale-after", default="0")
    p.add_argument("--delay", type=float, default=1.0)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--limit", type=int)
    p.add_argument("--skip-images", action="store_true")
    p.add_argument("--skip-embed", action="store_true")
    p.add_argument("--max-data-size", default="1GB",
                   help="stop when data/ reaches this size (default 1GB; 0 = no limit)")

    p = sub.add_parser("schedule", help="install the refresh as a daily background job")
    p.add_argument("action", choices=["install", "uninstall", "status"])
    p.add_argument("--at", default="03:30", help="local time of day, HH:MM")
    p.add_argument("--stale-after", default="0")
    p.add_argument("--max-data-size", default="1GB")
    p.add_argument("--skip-embed", action="store_true")
    p.add_argument("--print-only", action="store_true",
                   help="print the job definition instead of installing it")

    p = sub.add_parser("web", help="browse the catalogue in a browser")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--images", default=str(DEFAULT_IMAGES))
    p.add_argument("--max-data-size", default="1GB")

    p = sub.add_parser("plan", help="preview the crawl order without fetching anything")
    p.add_argument("source", choices=sorted(SOURCES))
    p.add_argument("--limit", type=int)
    p.add_argument("--weights")
    p.add_argument("--json", action="store_true")

    p = sub.add_parser("coverage", help="per-category counts against the weighted targets")
    p.add_argument("--target-total", type=int,
                   help="score coverage as if the catalogue held this many products")
    p.add_argument("--weights")
    p.add_argument("--json", action="store_true")

    p = sub.add_parser("design", help="ask a model to design a room from the catalogue")
    p.add_argument("prompt")
    p.add_argument("--model", default="gpt-5-mini")
    p.add_argument("--max-turns", type=int, default=10)
    p.add_argument("--json", action="store_true")
    p.add_argument("--quiet", action="store_true", help="do not print tool calls as they run")

    p = sub.add_parser("tool-schemas", help="print the tool definitions given to the model")

    p = sub.add_parser("family", help="every SKU in a model line — the variant lookup")
    p.add_argument("key", help="a family_key, or a product key to look up its family")
    p.add_argument("--json", action="store_true")

    p = sub.add_parser("quality", help="how much of the catalogue is complete enough to place")
    p.add_argument("--json", action="store_true")

    p = sub.add_parser("du", help="how much disk the catalogue is using")
    p.add_argument("--max-data-size", default="1GB")
    p.add_argument("--json", action="store_true")

    sub.add_parser("stats", help="what is in the database")

    args = ap.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    DATA.mkdir(parents=True, exist_ok=True)

    if args.cmd == "scrape":
        names = sorted(SOURCES) if args.source == "all" else [args.source]
        for name in names:
            out = args.out or DATA / f"{name}.jsonl"
            counts = scrape(
                name, out, cache_dir=args.cache, limit=args.limit,
                workers=args.workers, delay=args.delay, force=args.force,
                reparse=args.reparse, obey_robots=not args.no_robots,
                max_data_size=args.max_data_size, strategy=args.strategy,
                weights=_weights(args.weights),
            )
            print(f"{name}: {counts} -> {out}")
            if counts.get("stopped"):
                print(f"  stopped at the {args.max_data_size} data limit; "
                      f"run `python3 -m catalog du` to see what is using it")
                return 2
        return 0

    if args.cmd == "load":
        files = [Path(f) for f in args.files] or sorted(DATA.glob("*.jsonl"))
        if not files:
            print("no JSONL files found; run `scrape` first", file=sys.stderr)
            return 1
        conn = store.connect(args.db)
        for f in files:
            print(f"{f.name}: {store.load_jsonl(conn, f)}")
        print(dump(store.stats(conn)))
        return 0

    if args.cmd == "images":
        conn = store.connect(args.db)
        budget = DiskBudget(DATA, parse_size(args.max_data_size))
        fetcher = Fetcher(cache_dir=DEFAULT_CACHE, delay=args.delay, budget=budget)
        got = download_missing(
            conn, fetcher, args.dir, limit=args.limit,
            per_product=args.per_product, workers=args.workers, budget=budget,
        )
        print(dump(got))
        print(disk_report(budget.breakdown()))
        return 2 if got.get("stopped") else 0

    if args.cmd == "embed":
        conn = store.connect(args.db)
        print(dump(embed_catalog(
            conn, Embedder(args.model, args.pretrained),
            do_text=not args.images_only, do_images=not args.text_only,
            images_per_product=args.images_per_product, limit=args.limit,
        )))
        return 0

    if args.cmd == "search":
        if not args.query and not args.image and not args.category:
            print("give a query, --image, or --category", file=sys.stderr)
            return 1
        conn = store.connect(args.db)
        results = find_products(
            conn, args.query, query_image=args.image,
            design_category=args.category, min_price=args.min_price,
            max_price=args.max_price, source=args.source,
            in_stock_only=args.in_stock, k=args.k, embedder=_embedder(args),
            eligible_only=args.eligible, collapse_families=args.collapse,
        )
        if args.json:
            print(dump([r.to_dict() for r in results]))
        else:
            for i, r in enumerate(results, 1):
                price = f"Rs.{r.price:,.0f}" if r.price is not None else "no price"
                unit = f" {r.price_unit}" if r.price_unit else ""
                siblings = f"  +{r.family_size - 1} more sizes/colours" if r.family_size > 1 else ""
                print(f"{i:2}. {r.name[:68]:<68} {price:>12}{unit}")
                print(f"    {r.source} | {r.design_category or '-'} | {r.availability}"
                      f"{' | placeable' if r.design_eligible else ''}{siblings}")
                print(f"    {r.url}")
        return 0

    if args.cmd == "price":
        payload = json.loads(Path(args.design).read_text(encoding="utf-8"))
        items = payload if isinstance(payload, list) else payload.get("objects", [])
        conn = store.connect(args.db)
        print(dump(price_design(
            conn, items, embedder=_embedder(args),
            in_stock_only=not args.allow_out_of_stock, alternates=args.alternates,
            min_confidence=args.min_confidence,
        )))
        return 0

    if args.cmd in ("refresh", "watch"):
        from .refresh import format_report, parse_duration, refresh, watch
        names = None if args.sources == "all" else args.sources.split(",")
        opts = dict(
            db_path=args.db, data_dir=DATA, sources=names,
            stale_after=parse_duration(args.stale_after), delay=args.delay,
            workers=args.workers, limit=args.limit,
            skip_images=args.skip_images, skip_embed=args.skip_embed,
            max_data_size=args.max_data_size,
        )
        if args.cmd == "watch":
            watch(parse_duration(args.every), **opts)
            return 0
        rep = refresh(**opts)
        print(dump(rep) if args.json else format_report(rep))
        return 0 if rep["status"] == "ok" else (2 if rep["status"] == "over_budget" else 1)

    if args.cmd == "schedule":
        from . import schedule as sched
        if args.action == "uninstall":
            return sched.uninstall()
        if args.action == "status":
            return sched.status(args.db)
        hour, _, minute = args.at.partition(":")
        extra = ["--stale-after", args.stale_after,
                 "--max-data-size", args.max_data_size]
        if args.skip_embed:
            extra.append("--skip-embed")
        return sched.install(ROOT, int(hour), int(minute or 0), extra,
                             print_only=args.print_only)

    if args.cmd == "web":
        from .web import serve
        serve(args.db, args.images, host=args.host, port=args.port,
              max_data_size=parse_size(args.max_data_size))
        return 0

    if args.cmd == "plan":
        fetcher = Fetcher(cache_dir=DEFAULT_CACHE, delay=1.0)
        urls = list(SOURCES[args.source](fetcher).discover(limit=None))
        summary = plan_summary(urls, _weights(args.weights), args.limit)
        print(dump(summary) if args.json else format_summary(summary))
        return 0

    if args.cmd == "coverage":
        conn = store.connect(args.db)
        cov = coverage(conn, _weights(args.weights), args.target_total)
        # SKU count flatters a catalogue; the family count is what a designer
        # can actually choose between.
        for r in conn.execute(
            "SELECT COALESCE(design_category,'(none)') c, COUNT(*) n, "
            "COUNT(DISTINCT family_key) f, SUM(design_eligible) e, "
            "COUNT(DISTINCT CASE WHEN design_eligible THEN family_key END) ef, "
            "COUNT(DISTINCT source) s FROM products GROUP BY 1"
        ):
            entry = cov["categories"].get(r["c"])
            if entry is not None:
                entry.update(families=r["f"], eligible=r["e"] or 0,
                             placeable_families=r["ef"] or 0, sources=r["s"])
        print(dump(cov) if args.json else format_coverage(cov))
        return 0

    if args.cmd == "tool-schemas":
        from .tools import TOOL_SCHEMAS
        print(dump(TOOL_SCHEMAS))
        return 0

    if args.cmd == "design":
        from .agent import AgentError, run
        from .tools import CatalogService
        service = CatalogService(args.db)
        try:
            result = run(args.prompt, service, model=args.model,
                         max_turns=args.max_turns, verbose=not args.quiet)
        except AgentError as exc:
            print(f"design failed: {exc}", file=sys.stderr)
            return 1
        if args.json:
            print(dump(result))
            return 0
        print("\n" + result["reply"] + "\n")
        bom = result["priced_design"]
        if bom:
            print(f"{'—' * 78}")
            for line in bom["lines"]:
                total = line.get("line_total")
                money = f"Rs.{total:,.0f}" if total is not None else "—"
                flag = "" if line.get("placeable", True) else "  (not placeable)"
                print(f"  {str(line.get('label'))[:26]:<26} x{line.get('quantity', 1):<4g} "
                      f"{str(line.get('name', ''))[:38]:<38} {money:>11}{flag}")
            print(f"  {'TOTAL':<26} {'':<5} {'':<38} Rs.{bom['subtotal']:>8,.0f}")
            stale = bom.get("subtotal_by_price_state", {}).get("stale", 0)
            if stale:
                print(f"  Rs.{stale:,.0f} of that rests on prices not seen recently.")
            if bom.get("rejected_items"):
                print(f"  {len(bom['rejected_items'])} item(s) rejected: "
                      f"product keys the catalogue never issued.")
        print(f"\n  {result['turns']} turns · {len(result['tool_calls'])} tool calls · "
              f"{result['tokens']['input']:,} in / {result['tokens']['output']:,} out tokens")
        return 0

    if args.cmd == "family":
        conn = store.connect(args.db)
        key = args.key
        if ":" in key and "|" not in key:
            row = conn.execute("SELECT family_key FROM products WHERE key = ?", (key,)).fetchone()
            if row:
                key = row["family_key"]
        fam = get_family(conn, key)
        if args.json:
            print(dump(fam))
            return 0
        if not fam["count"]:
            print(f"no family {key!r}")
            return 1
        print(f"{fam['family_label']}  ({fam['count']} variants · {fam['design_category']})\n")
        for v in fam["variants"]:
            price = f"Rs.{v['price']:,.0f}" if v["price"] is not None else "no price"
            dims = v["dimensions"]
            size = " x ".join(
                f"{int(dims[a])}" for a in ("width_mm", "depth_mm", "length_mm", "height_mm")
                if dims.get(a)) or "-"
            mark = "placeable" if v["design_eligible"] else "         "
            print(f"  {price:>11}  {mark}  {size:>18} mm  {v['variant_label'][:42]:<42} {v['key']}")
        return 0

    if args.cmd == "quality":
        conn = store.connect(args.db)
        rows = conn.execute("""
            SELECT COALESCE(design_category,'(none)') AS cat,
                   COALESCE(placement_geometry,'?')   AS geom,
                   COUNT(*) AS n,
                   SUM(searchable)            AS searchable,
                   SUM(design_eligible)       AS eligible,
                   SUM(dimensions_complete)   AS dims,
                   SUM(primary_image_available) AS img,
                   SUM(stock_known)           AS stock,
                   SUM(material_known)        AS mat,
                   SUM(color_known)           AS col,
                   SUM(render_asset_quality >= 3) AS render3
              FROM products GROUP BY 1, 2 ORDER BY n DESC
        """).fetchall()
        totals = conn.execute(
            "SELECT COUNT(*) n, SUM(searchable) s, SUM(design_eligible) e FROM products"
        ).fetchone()
        if args.json:
            print(dump({"totals": dict(totals), "by_category": [dict(r) for r in rows]}))
            return 0
        pct = lambda a, b: f"{100 * (a or 0) // b}%" if b else "-"   # noqa: E731
        print(f"catalogue completeness: {totals['n']:,} products · "
              f"{totals['s'] or 0:,} searchable · {totals['e'] or 0:,} placeable\n")
        print(f"  {'category':<18}{'geometry':<11}{'n':>6}{'dims':>7}{'image':>7}"
              f"{'stock':>7}{'render':>8}{'PLACEABLE':>11}")
        for r in rows:
            n = r["n"]
            print(f"  {r['cat']:<18}{r['geom']:<11}{n:>6}{pct(r['dims'], n):>7}"
                  f"{pct(r['img'], n):>7}{pct(r['stock'], n):>7}{pct(r['render3'], n):>8}"
                  f"{pct(r['eligible'], n):>11}")
        gaps = conn.execute(
            "SELECT quality_gaps FROM products WHERE design_eligible = 0 AND quality_gaps IS NOT NULL"
        ).fetchall()
        tally = {}
        for g in gaps:
            for item in json.loads(g["quality_gaps"] or "[]"):
                key = item.split(":")[0]
                tally[key] = tally.get(key, 0) + 1
        if tally:
            print("\n  why products are not placeable:")
            for k, v in sorted(tally.items(), key=lambda kv: -kv[1]):
                print(f"    {k:<22}{v:>7,}")
        return 0

    if args.cmd == "du":
        budget = DiskBudget(DATA, parse_size(args.max_data_size))
        breakdown = budget.breakdown()
        print(dump(breakdown) if args.json else disk_report(breakdown))
        return 2 if budget.enabled and breakdown["used"] > budget.limit else 0

    if args.cmd == "stats":
        print(dump(store.stats(store.connect(args.db))))
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
