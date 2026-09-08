"""Fast, network-free tests for the parts that regress silently.

Run with: python3 -m pytest tests -q   (or: python3 tests/test_catalog.py)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from catalog import store
from catalog.budget import BudgetExceeded, DiskBudget, dir_size, human, parse_size
from catalog.models import (
    Image, Product, Variant, axis_for, classify_design_category,
    parse_length_mm, parse_weight_kg,
)
from catalog.search import find_products, label_confidence, price_design
from catalog.web import Catalog, _fts_match
from catalog.sources.homerun import _flight_rows, _take_text_row


# -- normalisation ---------------------------------------------------------
def test_dimensions_prefer_metric():
    assert parse_length_mm('45 cm (17 ¾ ")') == 450.0
    assert parse_length_mm("600 x 600 mm") == 600.0
    assert round(parse_length_mm("8 ft"), 1) == 2438.4
    assert parse_length_mm("no numbers here") is None


def test_weight_units():
    assert parse_weight_kg("50 Kg Bag") == 50.0
    assert parse_weight_kg("1.25 kg") == 1.25
    assert parse_weight_kg("500 g") == 0.5


def test_axis_labels():
    assert axis_for("Height of plant") == "height_mm"
    assert axis_for("Seat depth") == "seat_depth_mm"
    assert axis_for("nothing") is None


def test_design_category_respects_field_order():
    # A cement bag filed under a "tiling-bulk-prices" collection is still cement.
    assert classify_design_category("Maha PPC Cement, 50 Kg Bag", "PPC Cement", "Cement") == "structural"
    # Longest keyword wins inside one field.
    assert classify_design_category("YTBERG LED cabinet lighting - white") == "ceiling_lighting"
    assert classify_design_category("Somany Vitrified Floor Tile 600x600") == "flooring"
    assert classify_design_category("VIKHAMMER Bedside table", "bedside table") == "nightstands"
    assert classify_design_category("") is None


# -- the RSC flight scanner ------------------------------------------------
def test_text_row_length_is_bytes_not_characters():
    # 'CO₂' is 3 chars but 5 UTF-8 bytes; counting characters walks past the
    # next row header and loses the row after it.
    text = "CO₂ ok"
    nbytes = len(text.encode("utf-8"))
    buf = f"37:T{nbytes:x},{text}38:T3,abc\n"
    rows = _flight_rows(buf)
    assert rows["37"] == text
    assert rows["38"] == "abc"


def test_take_text_row_exact():
    assert _take_text_row("abc₂def", 0, len("abc₂".encode("utf-8"))) == "abc₂"


def test_json_rows_run_to_newline():
    rows = _flight_rows('1:{"a":1}\n2:["b"]\n')
    assert json.loads(rows["1"]) == {"a": 1}
    assert json.loads(rows["2"]) == ["b"]


# -- store -----------------------------------------------------------------
def _product(key_id="p1", price=100.0, name="KIVIK 3-seat sofa grey"):
    return Product(
        source="test", source_id=key_id, url=f"https://example.test/{key_id}",
        name=name, brand="IKEA", product_type="sofa", category="Sofas",
        category_path=["Products", "Sofas"], design_category="sofa",
        description="A grey fabric three seater.", price=price, currency="INR",
        availability="in_stock", price_unit="each",
        images=[Image(url=f"https://img.test/{key_id}.jpg", position=0)],
        variants=[Variant(variant_id="v1", price=price, available=True)],
    )


def test_upsert_and_price_history(tmp_path=None):
    db = Path(tmp_path or "/tmp") / "test_catalog.db"
    db.unlink(missing_ok=True)
    conn = store.connect(db)

    assert store.upsert(conn, [_product()])["new"] == 1
    assert store.upsert(conn, [_product()])["unchanged"] == 1
    # A repriced product must add an observation, not replace the series:
    # INSERT OR REPLACE would cascade the old rows away.
    assert store.upsert(conn, [_product(price=120.0)])["changed"] == 1
    assert conn.execute("SELECT COUNT(*) c FROM price_history").fetchone()["c"] == 2
    assert conn.execute("SELECT COUNT(*) c FROM product_images").fetchone()["c"] == 1
    db.unlink(missing_ok=True)


def test_recrawl_keeps_downloaded_images(tmp_path="/tmp"):
    db = Path(tmp_path) / "images_keep.db"
    db.unlink(missing_ok=True)
    conn = store.connect(db)
    store.upsert(conn, [_product()])
    conn.execute("UPDATE product_images SET local_path='/tmp/x.jpg', sha256='abc'")
    conn.commit()

    changed = _product(price=150.0)                      # a reprice, same photograph
    changed.images.insert(0, Image(url="https://img.test/new.jpg", position=0))
    changed.images[1].position = 1
    store.upsert(conn, [changed])

    rows = {r["url"]: r["local_path"] for r in
            conn.execute("SELECT url, local_path FROM product_images")}
    assert rows["https://img.test/p1.jpg"] == "/tmp/x.jpg", "known file must survive a re-crawl"
    assert rows["https://img.test/new.jpg"] is None
    db.unlink(missing_ok=True)


def test_refresh_reports_a_price_move(tmp_path="/tmp"):
    from catalog.refresh import _price_changes_since

    db = Path(tmp_path) / "refresh_prices.db"
    db.unlink(missing_ok=True)
    conn = store.connect(db)
    store.upsert(conn, [_product(price=355.0)])          # yesterday's crawl
    window = "2000-01-01T00:00:00+00:00"
    assert _price_changes_since(conn, window) == []      # a first sighting is not a change

    store.upsert(conn, [_product(price=395.0)])          # today: the site repriced
    moves = _price_changes_since(conn, window)
    assert len(moves) == 1
    assert (moves[0]["prev"], moves[0]["price"]) == (355.0, 395.0)

    store.upsert(conn, [_product(price=395.0)])          # unchanged: no new observation
    assert len(_price_changes_since(conn, window)) == 1
    db.unlink(missing_ok=True)


def test_duration_parsing():
    from catalog.refresh import parse_duration
    assert parse_duration("45m") == 2700
    assert parse_duration("12h") == 43200
    assert parse_duration("3d") == 259200
    assert parse_duration("0") == 0


def test_refresh_lock_is_exclusive(tmp_path="/tmp"):
    from catalog.refresh import Lock
    path = Path(tmp_path) / "lock.test"
    path.unlink(missing_ok=True)
    with Lock(path):
        assert path.exists()
        try:
            with Lock(path):
                raise AssertionError("a second refresh must not start")
        except SystemExit:
            pass
    assert not path.exists(), "the lock is released on exit"


def test_content_hash_covers_derived_fields():
    a, b = _product(), _product()
    b.design_category = "chair"
    assert store._content_hash(a) != store._content_hash(b)


# -- taxonomy and crawl stratification -------------------------------------
def test_slug_classification():
    from catalog.taxonomy import classify_slug
    assert classify_slug("vikhammer-bedside-table-white-10339331") == "nightstands"
    assert classify_slug("zebrasaev-pendant-lamp-white-plastic") == "ceiling_lighting"
    assert classify_slug("vuku-wardrobe-white") == "wardrobes"
    assert classify_slug("viskafors-3-seat-sofa-lejde-light-beige") == "sofas"


def test_parts_never_outrank_the_furniture_they_belong_to():
    from catalog.taxonomy import classify_slug
    # These were the two failures that skewed the first crawl: a spare sofa cover
    # read as a sofa, and kitchen carcasses read as bedroom storage.
    assert classify_slug("vimle-cover-for-1-seat-section-gunnared-grey") == "components"
    assert classify_slug("vimle-cover-4-seat-sofa-w-chaise-longue") == "components"
    assert classify_slug("voxtorp-drawer-front-oak-effect-40x10-cm") == "components"
    assert classify_slug("metod-base-cabinet-with-shelves-white") == "components"
    # ... without swallowing the real products that share those words
    assert classify_slug("vimle-1-seat-section-gunnared-medium-grey") == "sofas"
    assert classify_slug("bergpalm-quilt-cover-and-2-pillowcases-grey") == "bedding"
    assert classify_slug("gurli-cushion-cover-beige") == "bedding"


def test_colour_words_are_not_product_types():
    from catalog.taxonomy import classify_slug
    # " light " appears in hundreds of colour names; matching it as lighting
    # would have mis-filed most of the catalogue.
    assert classify_slug("hemnes-bed-frame-light-grey") == "beds"
    assert classify_slug("ektorp-sofa-light-beige") == "sofas"


def test_coarse_category_filter_beats_the_fine_one():
    from catalog.taxonomy import expand_category
    # A design asking for "storage" must still reach a product filed as a
    # wardrobe, or the pricing pass silently drops the item.
    assert set(expand_category("storage")) >= {"storage", "wardrobes", "shelving"}
    assert set(expand_category("lighting")) == {"lamps", "ceiling_lighting"}
    assert expand_category("beds") == ["beds"]
    assert expand_category(None) == []


def test_family_interleaving_breaks_up_a_range():
    from catalog.stratify import _interleave_families
    urls = [f"/p/voxtorp-drawer-front-{i}" for i in range(5)] + [
        "/p/malm-bed-frame-white", "/p/hemnes-bed-frame-white"]
    out = _interleave_families(urls)
    assert out[0].startswith("/p/voxtorp")
    assert "malm" in out[1] and "hemnes" in out[2], "families must rotate, not run in blocks"


def test_plan_is_category_balanced_at_every_prefix(tmp_path="/tmp"):
    from catalog.stratify import plan
    from catalog.taxonomy import classify_slug
    # Mimic sitemap order: a huge block of one family, then a few of everything.
    urls = [f"https://x/p/voxtorp-drawer-front-{i}-1000000{i}" for i in range(200)]
    urls += [f"https://x/p/malm-bed-frame-white-200000{i}" for i in range(20)]
    urls += [f"https://x/p/vidja-floor-lamp-white-300000{i}" for i in range(20)]
    urls += [f"https://x/p/stockholm-rug-flatwoven-400000{i}" for i in range(20)]

    first30 = plan(urls, limit=30)
    cats = {classify_slug(u) for u in first30}
    assert {"beds", "lamps", "rugs"} <= cats, f"early crawl must reach real furniture, got {cats}"
    # Components are weight 1 against beds at 5, so they must not dominate.
    n_components = sum(1 for u in first30 if classify_slug(u) == "components")
    assert n_components <= 6, f"parts took {n_components} of the first 30 slots"


def test_plan_respects_weights():
    from catalog.stratify import plan
    from catalog.taxonomy import classify_slug
    urls = [f"https://x/p/malm-bed-frame-{i}-100000{i}" for i in range(50)]
    urls += [f"https://x/p/lack-coffee-table-{i}-200000{i}" for i in range(50)]
    heavy_beds = plan(urls, weights={"beds": 10, "tables": 1}, limit=22)
    n_beds = sum(1 for u in heavy_beds if classify_slug(u) == "beds")
    assert n_beds == 20, f"a 10:1 weighting should give 20 beds in 22, got {n_beds}"


def test_plan_emits_every_url_when_unlimited():
    from catalog.stratify import plan
    urls = [f"https://x/p/malm-bed-frame-{i}-10000{i}" for i in range(7)]
    urls += [f"https://x/p/vidja-floor-lamp-{i}-20000{i}" for i in range(3)]
    out = plan(urls)
    assert sorted(out) == sorted(urls), "stratifying must not drop or duplicate URLs"


# -- disk budget -----------------------------------------------------------
def test_parse_size():
    assert parse_size("1GB") == 1024 ** 3
    assert parse_size("750 MB") == 750 * 1024 ** 2
    assert parse_size("1.5g") == int(1.5 * 1024 ** 3)
    assert parse_size(2048) == 2048
    assert parse_size("0") == 0
    try:
        parse_size("soon")
    except ValueError:
        pass
    else:
        raise AssertionError("a bad size must be rejected, not guessed at")


def test_human_readable():
    assert human(512) == "512 B"
    assert human(1024 ** 2) == "1.0 MB"
    assert human(int(2.5 * 1024 ** 3)) == "2.5 GB"


def test_dir_size_survives_unreadable_entries():
    # A size check must never be the thing that crashes a crawl.
    assert dir_size(Path("/dev/fd")) >= 0
    assert dir_size(Path("/definitely/not/here")) == 0


def test_budget_stops_before_crossing(tmp_path="/tmp"):
    root = Path(tmp_path) / "budget_test"
    if root.exists():
        for f in root.iterdir():
            f.unlink()
    root.mkdir(parents=True, exist_ok=True)
    (root / "a.bin").write_bytes(b"x" * 8000)

    # reserve=0 so the arithmetic under test is the limit itself
    b = DiskBudget(root, limit=10_000, reserve=0)
    assert b.used == 8000
    assert not b.would_exceed(1000)
    assert b.would_exceed(3000), "a write that crosses the limit must be refused"
    try:
        b.check(3000)
    except BudgetExceeded as exc:
        assert "10.0 KB" in str(exc) or "9.8 KB" in str(exc)
    else:
        raise AssertionError("check() must raise once the limit would be crossed")

    (root / "b.bin").write_bytes(b"x" * 1500)
    assert b.resync() == 9500, "resync must see files written outside the budget"
    for f in root.iterdir():
        f.unlink()
    root.rmdir()


def test_budget_reserve_keeps_the_directory_under_the_limit(tmp_path="/tmp"):
    # Writers stop short of the ceiling, so uncharged growth (SQLite's WAL) does
    # not settle the directory just over the number the user asked for.
    b = DiskBudget(Path(tmp_path), limit=1024 ** 3)
    assert b.reserve > 0
    assert b.ceiling < b.limit
    assert b.reserve == min(int(1024 ** 3 * 0.02), 32 * 1024 ** 2)


def test_budget_disabled_when_zero(tmp_path="/tmp"):
    b = DiskBudget(Path(tmp_path), limit=0)
    assert not b.enabled
    assert not b.would_exceed(10 ** 12)
    b.check(10 ** 12)                     # must not raise


def test_image_download_stops_at_the_budget(tmp_path="/tmp"):
    from catalog.images import download_missing

    db = Path(tmp_path) / "budget_images.db"
    db.unlink(missing_ok=True)
    conn = store.connect(db)
    p = _product()
    p.images = [Image(url=f"https://img.test/{i}.jpg", position=i) for i in range(10)]
    store.upsert(conn, [p])

    images = Path(tmp_path) / "budget_images"
    if images.exists():
        for child in sorted(images.rglob("*"), reverse=True):
            child.unlink() if child.is_file() else child.rmdir()

    class StubFetcher:
        """Distinct 4 KB payloads. Identical bytes would deduplicate to one file
        and the directory would never grow, so the test would pass vacuously."""
        def get_bytes(self, url):
            body = url.encode()
            return body + b"\x00" * (4096 - len(body))

    start = dir_size(images) if images.exists() else 0
    budget = DiskBudget(images, limit=start + 12_000, reserve=0)
    got = download_missing(conn, StubFetcher(), images, budget=budget, workers=1)

    assert got["stopped"] is True, "the download must stop, not run to completion"
    assert 0 < got["downloaded"] < 10
    assert dir_size(images) <= budget.limit
    db.unlink(missing_ok=True)


# -- search and pricing ----------------------------------------------------
def _seeded(tmp_path):
    db = Path(tmp_path) / "search.db"
    db.unlink(missing_ok=True)
    conn = store.connect(db)
    store.upsert(conn, [
        _product("sofa1", 24999.0, "KIVIK 3-seat sofa - grey fabric"),
        _product("sofa2", 89999.0, "LANDSKRONA 3-seat sofa - leather"),
    ])
    conn.execute("UPDATE products SET design_category='ceiling_lighting', "
                 "name='VIP Metal Spot Light Box' WHERE key='test:sofa2'")
    conn.execute("DELETE FROM products_fts WHERE key='test:sofa2'")
    conn.execute("INSERT INTO products_fts (key,name,brand,category_path,description,"
                 "materials,colors,tags) VALUES ('test:sofa2','VIP Metal Spot Light Box','','','','','','')")
    conn.commit()
    return conn


def test_filters_are_exact(tmp_path="/tmp"):
    conn = _seeded(tmp_path)
    assert [m.key for m in find_products(conn, "sofa", max_price=30000)] == ["test:sofa1"]
    assert find_products(conn, "sofa", design_category="rug") == []
    assert find_products(conn, "sofa", max_price=10) == []


def test_confidence_floor_rejects_a_nonsense_match(tmp_path="/tmp"):
    conn = _seeded(tmp_path)
    quote = price_design(
        conn, [{"label": "brass chandelier with crystal drops",
                "design_category": "ceiling_lighting", "quantity": 1}],
        embedder=None,
    )
    line = quote["lines"][0]
    assert line["matched"] is None, "a spot light box is not a brass chandelier"
    assert quote["items_low_confidence"] == 1
    assert line["alternates"], "rejected candidates are still offered for review"


def test_web_search_uses_fts_without_an_alias(tmp_path="/tmp"):
    # FTS5 resolves MATCH and bm25() against the table name; an alias raises
    # "no such column", which the browser only saw as an empty result.
    conn = _seeded(tmp_path)
    holder = Path(tmp_path) / "web_data" / "images"
    holder.mkdir(parents=True, exist_ok=True)
    web = Catalog(Path(tmp_path) / "search.db", holder)
    web._local.conn = conn
    out = web.products({"q": ["sofa grey"]})
    assert out["total"] >= 1
    assert out["items"][0]["name"].startswith("KIVIK")
    assert web.products({"q": ["sofa"], "max_price": ["30000"]})["total"] == 1


def test_fts_match_drops_stopwords():
    assert _fts_match("a white pendant lamp for the ceiling") == \
        '"white" OR "pendant" OR "lamp" OR "ceiling"'
    assert _fts_match("the of and") is None


def test_confidence_scoring():
    from catalog.search import Match
    m = Match(key="k", name="KIVIK 3-seat sofa - grey fabric", url="", price=1.0,
              currency="INR", source="test", design_category="sofa")
    assert label_confidence("3-seat sofa", m) == 1.0
    assert label_confidence("brass chandelier", m) == 0.0


if __name__ == "__main__":
    import inspect

    mod = sys.modules[__name__]
    failed = 0
    for name, fn in sorted(vars(mod).items()):
        if not (name.startswith("test_") and callable(fn)):
            continue
        kwargs = {"tmp_path": "/tmp"} if "tmp_path" in inspect.signature(fn).parameters else {}
        try:
            fn(**kwargs)
            print(f"  ok   {name}")
        except Exception as exc:
            failed += 1
            print(f"  FAIL {name}: {exc.__class__.__name__}: {exc}")
    print("all passed" if not failed else f"{failed} failed")
    raise SystemExit(1 if failed else 0)
