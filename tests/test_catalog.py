"""Fast, network-free tests for the parts that regress silently.

Run with: python3 -m pytest tests -q   (or: python3 tests/test_catalog.py)
"""
from __future__ import annotations

import json
import re
import sys
from datetime import datetime, timedelta, timezone
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


# -- material specifications -----------------------------------------------
def test_a_size_range_is_not_a_measurement():
    from catalog.specs import size_range
    # "200mm to 700mm" lists the sizes a channel is sold in. Reading it as a
    # length recorded 200 mm as fact and broke every dimensional filter on it.
    assert size_range("Telescopic Channel, 200mm to 700mm") == (200.0, 700.0)
    assert size_range("Channel 450-600 mm") == (450.0, 600.0)
    assert size_range("18mm Plywood") is None


def test_a_unit_ends_where_the_word_ends():
    from catalog.specs import size_range
    # The "m" alternative matched the m of "minutes" and turned an adhesive's
    # cure time into a 30-60 metre fitting.
    assert size_range("cure time 30 to 60 minutes") is None
    assert size_range("set in 5-10 min") is None
    assert size_range("coverage 30 to 60 sq ft") is None


def test_only_sheet_goods_have_a_sheet_size():
    from catalog.specs import extract
    # Every "105x57x200 cm" furniture dimension matched before this: 8,300 of
    # 13,172 products claimed a sheet size.
    ply = _product("s1", 1974.0, "Century BWP Marine Plywood 8 x 4 ft")
    assert extract(ply)["sheet_size_mm"] == [2438, 1219]
    sofa = _product("s2", 24990.0, "VIHALS Wardrobe - white 105x57x200 cm")
    assert "sheet_size_mm" not in extract(sofa)


def test_a_load_rating_is_not_a_pack_size():
    from catalog.specs import extract, units
    p = _product("h1", 295.0, "Hettich KA 5632 Telescopic Channel, 45 kg Capacity, 200mm to 700mm")
    assert extract(p)["load_capacity_kg"] == 45.0
    u = units(p)
    # A channel is sold by the piece. Recording a 45 kg pack would make any
    # quantity take-off nonsense.
    assert u["pack_quantity"] is None
    assert u["purchase_unit"] == "piece"


def test_units_separate_cost_content_and_consumption():
    from catalog.specs import units
    glue = _product("a1", 6300.0, "Pidilite Masterlok Synthetic Wood Adhesive, 50Kg")
    u = units(glue)
    assert (u["purchase_unit"], u["pack_quantity"], u["pack_uom"]) == ("kg", 50.0, "kg")
    ply = _product("a2", 1974.0, "Century Club Prime BWP Marine Plywood")
    assert units(ply)["purchase_unit"] == "sheet"
    assert units(ply)["consumption_uom"] == "sq_ft", "sheet goods are consumed by area"


def test_grade_is_extracted_for_sheet_goods():
    from catalog.specs import extract
    p = _product("g1", 1974.0, "Century Club Prime BWP Marine Plywood")
    p.description = "premium boiling water proof BWP grade conforming to IS:710"
    got = extract(p)["grade"]
    assert "BWP" in got and "IS:710" in got


def test_an_unknown_product_is_not_assumed_to_be_furniture():
    from catalog.quality import geometry_for
    # Defaulting to `footprint` asserted that plywood and adhesive stand on the
    # floor and merely lacked measurements.
    assert geometry_for(None) == "none"
    assert geometry_for("something-new") == "none"
    assert geometry_for("beds") == "footprint"


def test_a_fitting_is_hardware_whatever_it_fits():
    from catalog.taxonomy import classify_text
    assert classify_text("Ebco Pro Lift Bed Fitting - Extended Arm Set") == "hardware"
    assert classify_text("Ebco Pro Lift Bed Hydraulic Gas Pump") == "hardware"
    assert classify_text("MALM Bed frame - white 160x200 cm") == "beds"


# -- search index consistency ----------------------------------------------
def test_fts_follows_products_through_every_write(tmp_path="/tmp"):
    db = Path(tmp_path) / "fts_triggers.db"
    db.unlink(missing_ok=True)
    conn = store.connect(db)

    p = _product("f1", 1000.0, "MALM Bed frame white")
    store.upsert(conn, [p])
    assert store.fts_drift(conn) == {"missing": 0, "orphaned": 0, "stale": 0}

    p.name = "MALM Bed frame black"
    p.price = 1100.0
    store.upsert(conn, [p])
    assert store.fts_drift(conn)["stale"] == 0
    assert conn.execute("SELECT name FROM products_fts").fetchone()["name"] == p.name

    # A delete had no application path at all before the triggers, so an orphan
    # would have survived in the index indefinitely.
    conn.execute("DELETE FROM products WHERE key = ?", (p.key,))
    conn.commit()
    assert conn.execute("SELECT COUNT(*) c FROM products_fts").fetchone()["c"] == 0
    db.unlink(missing_ok=True)


def test_fts_follows_a_write_that_bypasses_application_code(tmp_path="/tmp"):
    db = Path(tmp_path) / "fts_raw.db"
    db.unlink(missing_ok=True)
    conn = store.connect(db)
    store.upsert(conn, [_product("f2", 500.0, "HEMNES Chest")])
    # The whole point of moving this into the database: a migration, a fix-up
    # script or a psql session cannot desynchronise the index.
    conn.execute("UPDATE products SET name = 'HEMNES Chest of 3 drawers'")
    conn.commit()
    assert store.fts_drift(conn) == {"missing": 0, "orphaned": 0, "stale": 0}
    assert "3 drawers" in conn.execute("SELECT name FROM products_fts").fetchone()["name"]
    db.unlink(missing_ok=True)


def test_rebuild_fts_restores_an_index_damaged_out_of_band(tmp_path="/tmp"):
    db = Path(tmp_path) / "fts_rebuild.db"
    db.unlink(missing_ok=True)
    conn = store.connect(db)
    store.upsert(conn, [_product("f3", 100.0, "VIDJA Floor lamp")])
    conn.execute("DELETE FROM products_fts")          # simulate damage
    conn.commit()
    assert store.fts_drift(conn)["missing"] == 1
    assert store.rebuild_fts(conn) == 1
    assert store.fts_drift(conn) == {"missing": 0, "orphaned": 0, "stale": 0}
    db.unlink(missing_ok=True)


# -- tool boundary ---------------------------------------------------------
def _service(tmp_path, name="tools.db"):
    from catalog.tools import CatalogService
    db = Path(tmp_path) / name
    db.unlink(missing_ok=True)
    conn = store.connect(db)
    bed = _ikea("MALM Bed frame - white 160x200 cm", "MALM", "bed frame")
    bed.design_category = "beds"
    bed.dimensions = {"width_mm": 1600.0, "length_mm": 2000.0}
    lamp = _ikea("VIDJA Floor lamp - white", "VIDJA", "floor lamp")
    lamp.design_category, lamp.price = "lamps", 3990.0
    lamp.dimensions = {"height_mm": 1380.0, "diameter_mm": 250.0}
    store.upsert(conn, [bed, lamp])
    svc = CatalogService(db)
    svc.conn = conn
    return svc


def test_tools_never_leak_internals(tmp_path="/tmp"):
    svc = _service(tmp_path, "leak.db")
    card = svc.search_catalog(query="bed frame", limit=1)["results"][0]
    detail = svc.get_product(card["product_key"])
    blob = json.dumps([card, detail])
    # A prompt is not the place for local paths, raw payloads or internal columns.
    for forbidden in ("local_path", "/Users/", "content_hash", "embedding_text",
                      "raw", "quality_gaps\": null"):
        assert forbidden not in blob, f"{forbidden!r} reached the model"
    assert card["product_key"].startswith("ikea_in:")


def test_a_design_cannot_use_a_key_the_catalog_never_issued(tmp_path="/tmp"):
    svc = _service(tmp_path, "ledger.db")
    real = svc.search_catalog(query="bed frame", limit=1)["results"][0]["product_key"]
    quote = svc.price_design([
        {"product_key": real, "label": "bed", "quantity": 1},
        {"product_key": "ikea_in:00000000", "label": "brass chandelier", "quantity": 1},
    ])
    assert len(quote["rejected_items"]) == 1
    assert quote["rejected_items"][0]["product_key"] == "ikea_in:00000000"
    # ... and the fabricated line contributes nothing to the total
    assert quote["subtotal"] == 12990.0
    assert all(line["product_key"] != "ikea_in:00000000" for line in quote["lines"])


def test_dimension_limits_are_hard(tmp_path="/tmp"):
    svc = _service(tmp_path, "dims.db")
    assert svc.search_catalog(query="bed frame", max_width_mm=2000)["count"] == 1
    # A bed wider than the wall is a wrong answer, not a lower-ranked one.
    assert svc.search_catalog(query="bed frame", max_width_mm=900)["count"] == 0


def test_unknown_tool_is_an_error_not_a_silent_noop(tmp_path="/tmp"):
    from catalog.tools import dispatch
    svc = _service(tmp_path, "dispatch.db")
    assert "error" in dispatch(svc, "delete_everything", {})


# -- agent loop ------------------------------------------------------------
def test_agent_runs_a_tool_loop_and_returns_the_bom(tmp_path="/tmp"):
    """Exercises the Responses API protocol against a stubbed transport."""
    from catalog import agent as agent_mod

    svc = _service(tmp_path, "agent.db")
    key = svc.search_catalog(query="bed frame", limit=1)["results"][0]["product_key"]
    scripted = [
        {"output": [{"type": "function_call", "call_id": "c1", "name": "search_catalog",
                     "arguments": json.dumps({"query": "bed frame", "limit": 1})}],
         "usage": {"input_tokens": 10, "output_tokens": 5}},
        {"output": [{"type": "function_call", "call_id": "c2", "name": "price_design",
                     "arguments": json.dumps({"objects": [
                         {"product_key": key, "label": "bed", "quantity": 1}]})}],
         "usage": {"input_tokens": 20, "output_tokens": 8}},
        {"output": [{"type": "message", "content": [
            {"type": "output_text", "text": "A calm room built around the MALM bed."}]}],
         "usage": {"input_tokens": 30, "output_tokens": 12}},
    ]
    sent = []

    def fake_post(payload, api_key, timeout=180):
        sent.append(payload)
        return scripted[len(sent) - 1]

    original = agent_mod._post
    agent_mod._post = fake_post
    try:
        out = agent_mod.run("design a bedroom", svc, api_key="test-key")
    finally:
        agent_mod._post = original

    assert out["priced_design"]["subtotal"] == 12990.0
    assert [c["tool"] for c in out["tool_calls"]] == ["search_catalog", "price_design"]
    assert "MALM" in out["reply"]
    assert out["tokens"] == {"input": 60, "output": 25}
    # Each tool result must be returned against the call it answers.
    outputs = [i for i in sent[-1]["input"] if i.get("type") == "function_call_output"]
    assert [o["call_id"] for o in outputs] == ["c1", "c2"]


def test_agent_hands_bad_arguments_back_to_the_model(tmp_path="/tmp"):
    from catalog import agent as agent_mod
    svc = _service(tmp_path, "agent_args.db")
    scripted = [
        {"output": [{"type": "function_call", "call_id": "c1", "name": "search_catalog",
                     "arguments": '{"nonsense_field": 1}'}]},
        {"output": [{"type": "message",
                     "content": [{"type": "output_text", "text": "retrying"}]}]},
    ]
    sent = []
    agent_mod._post, original = (lambda p, api_key, timeout=180:
                                 (sent.append(p), scripted[len(sent) - 1])[1]), agent_mod._post
    try:
        agent_mod.run("x", svc, api_key="k")
    finally:
        agent_mod._post = original
    result = json.loads([i for i in sent[-1]["input"]
                         if i.get("type") == "function_call_output"][0]["output"])
    # A wrong argument name is the model's to correct, not a crash.
    assert "error" in result and "bad arguments" in result["error"]


def test_billing_failure_is_not_retried():
    from catalog import agent as agent_mod

    class Resp:
        status_code = 429
        text = ""
        headers: dict = {}
        def json(self):
            return {"error": {"message": "You have no credits remaining."}}

    calls = []
    original = agent_mod.requests.post
    agent_mod.requests.post = lambda *a, **k: (calls.append(1), Resp())[1]
    try:
        try:
            agent_mod._post({}, "k")
        except agent_mod.AgentError as exc:
            assert "no credits" in str(exc).lower()
        else:
            raise AssertionError("a billing failure must raise")
    finally:
        agent_mod.requests.post = original
    assert len(calls) == 1, "backing off on an empty balance just wastes time"


# -- product families ------------------------------------------------------
def _slug_id(name):
    # Full slug: truncating collided 140/160/180 onto one key and silently
    # collapsed the fixture to a single product.
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def _ikea(name, range_name, type_name):
    p = _product(_slug_id(name), 12990.0, name)
    p.attributes = {"range_name": range_name, "type_name": type_name}
    p.source = "ikea_in"
    return p


def test_variants_of_one_range_share_a_family():
    from catalog.family import resolve
    sizes = [_ikea(f"MALM Bed frame - white {w}x200 cm", "MALM", "bed frame")
             for w in (140, 160, 180)]
    sizes.append(_ikea("MALM Bed frame - black-brown 180x200 cm", "MALM", "bed frame"))
    keys = {resolve(p)[0] for p in sizes}
    assert len(keys) == 1, "one model line, whatever the size or colour"
    assert resolve(sizes[0])[1] == "MALM bed frame"
    # ... and a different article type is a different family
    other = _ikea("MALM Chest of 3 drawers - white", "MALM", "chest of drawers")
    assert resolve(other)[0] != resolve(sizes[0])[0]


def test_family_is_scoped_to_its_retailer():
    from catalog.family import resolve
    a = _ikea("MALM Bed frame - white", "MALM", "bed frame")
    b = _ikea("MALM Bed frame - white", "MALM", "bed frame")
    b.source = "homerun"
    b.attributes = {}
    assert resolve(a)[0].startswith("ikea_in:")
    assert resolve(b)[0].startswith("homerun:")
    assert resolve(a)[0] != resolve(b)[0]


def test_homerun_family_drops_the_pack_size():
    from catalog.family import resolve
    a = _product("c1", 355.0, "Maha PPC Cement, 50 Kg Bag")
    b = _product("c2", 40.0, "Maha PPC Cement, 1 Kg Pack")
    a.source = b.source = "homerun"
    assert resolve(a)[0] == resolve(b)[0]
    assert resolve(a)[1] == "Maha PPC Cement"
    assert "50 Kg Bag" in resolve(a)[2]


def test_search_collapses_to_one_sku_per_family(tmp_path="/tmp"):
    from catalog.search import find_products
    db = Path(tmp_path) / "families.db"
    db.unlink(missing_ok=True)
    conn = store.connect(db)
    products = [_ikea(f"MALM Bed frame - white {w}x200 cm", "MALM", "bed frame")
                for w in (140, 160, 180)]
    products.append(_ikea("HEMNES Bed frame - white 160x200 cm", "HEMNES", "bed frame"))
    for p in products:
        p.design_category = "beds"
    store.upsert(conn, products)

    raw = find_products(conn, "bed frame white", k=4)
    assert len(raw) == 4, "without collapsing, one range fills the result"
    collapsed = find_products(conn, "bed frame white", k=4, collapse_families=True)
    assert len(collapsed) == 2, "MALM and HEMNES are two choices, not four"
    malm = next(m for m in collapsed if "MALM" in m.name)
    assert malm.family_size == 3, "and the caller is told how many sizes exist"
    db.unlink(missing_ok=True)


def test_variant_lookup_returns_the_whole_line(tmp_path="/tmp"):
    from catalog.search import get_family
    from catalog.family import resolve
    db = Path(tmp_path) / "variants.db"
    db.unlink(missing_ok=True)
    conn = store.connect(db)
    products = [_ikea(f"MALM Bed frame - white {w}x200 cm", "MALM", "bed frame")
                for w in (140, 160, 180)]
    store.upsert(conn, products)
    fam = get_family(conn, resolve(products[0])[0])
    assert fam["count"] == 3
    assert {v["variant_label"] for v in fam["variants"]}, "each variant is labelled"
    db.unlink(missing_ok=True)


# -- embedding versioning --------------------------------------------------
def test_embeddings_are_keyed_by_model_and_version(tmp_path="/tmp"):
    from catalog.embed import asset_id
    db = Path(tmp_path) / "emb.db"
    db.unlink(missing_ok=True)
    conn = store.connect(db)
    store.upsert(conn, [_product()])
    key = _product().key
    asset = asset_id("some product text")
    rows = [
        (key, "text", asset, "clip-vitb32", "laion2b_s34b_b79k", 4, b"\x00" * 16, "t0"),
        (key, "text", asset, "furniture-v2", "2026-09", 8, b"\x00" * 32, "t1"),
    ]
    conn.executemany(
        "INSERT OR REPLACE INTO embeddings "
        "(product_key,kind,source_asset,model,model_version,dim,vec,created_at) "
        "VALUES (?,?,?,?,?,?,?,?)", rows)
    conn.commit()
    # A second representation must not evict the one currently serving.
    assert conn.execute("SELECT COUNT(*) c FROM embeddings").fetchone()["c"] == 2
    dims = {r["dim"] for r in conn.execute("SELECT dim FROM embeddings")}
    assert dims == {4, 8}, "each model keeps its own dimensionality"
    db.unlink(missing_ok=True)


def test_asset_id_ties_a_vector_to_the_bytes_it_came_from():
    from catalog.embed import asset_id
    assert asset_id("a") == asset_id("a")
    assert asset_id("a") != asset_id("b")
    assert asset_id(b"a") == asset_id("a"), "bytes and text address identically"
    assert asset_id("a").startswith("sha256:")


# -- IKEA offer parsing ----------------------------------------------------
def _ikea_page(aggregate: bool, offercount: int = 1) -> str:
    """Minimal stand-in for an IKEA product page: the two hydrate blocks the
    parser reads, plus the JSON-LD block that carries the offer."""
    product = json.dumps({"product": {
        "itemNo": "10471234", "visibleItemNo": "104.712.34", "name": "VIHALS",
        "typeName": "wardrobe", "currencyCode": "INR", "price": 24990,
        "mediaList": [{"type": "image", "content": {
            "url": "https://x/img.jpg", "alt": "a", "width": 2000, "height": 2000,
            "type": "MAIN_PRODUCT_IMAGE"}}],
        "packageMeasurements": [], "subProducts": [],
    }})
    page = json.dumps({"pageProps": {
        "pipPriceModuleProps": {"priceModuleProps": {"priceOfferType": "family"}},
        "productInformationSectionProps": {},
    }})
    offers = ({"@type": "AggregateOffer", "lowPrice": "20990", "highPrice": "24990",
               "offercount": offercount,
               "offers": [{"@type": "Offer", "availability": "https://schema.org/InStock",
                           "price": "20990", "priceCurrency": "INR"}]}
              if aggregate else
              {"@type": "Offer", "availability": "https://schema.org/InStock",
               "price": "24990", "priceCurrency": "INR"})
    ld = json.dumps({"@context": "https://schema.org/", "@type": "Product",
                     "name": "VIHALS Wardrobe", "offers": offers,
                     "width": "105 cm (41 3/8 \")", "height": "200 cm (78 3/4 \")",
                     "depth": "57 cm (22 1/2 \")", "material": "Wood, Glass",
                     "color": "white"})
    return (f'<script type="text/hydrate">{product}</script>'
            f'<script type="text/hydrate">{page}</script>'
            f'<script type="application/ld+json">{ld}</script>')


def test_aggregate_offer_availability_is_read_from_the_nested_offer():
    from catalog.sources.ikea_in import IkeaIndia
    # An AggregateOffer carries no availability of its own. Reading only the flat
    # field left every ranged-price product "unknown", which disqualified it from
    # being placed in a room.
    p = IkeaIndia(None).parse("https://www.ikea.com/in/en/p/vihals-10471234/",
                              _ikea_page(aggregate=True))
    assert p.availability == "in_stock"
    flat = IkeaIndia(None).parse("https://www.ikea.com/in/en/p/vihals-10471234/",
                                 _ikea_page(aggregate=False))
    assert flat.availability == "in_stock"


def test_member_discount_is_not_an_unresolved_price():
    from catalog.quality import assess
    from catalog.sources.ikea_in import IkeaIndia
    p = IkeaIndia(None).parse("https://www.ikea.com/in/en/p/vihals-10471234/",
                              _ikea_page(aggregate=True, offercount=1))
    # The quoted price is the one anyone pays; the member price sits beside it.
    assert p.price == 24990
    assert p.attributes["member_price"] == 20990
    assert p.attributes["price_varies"] is False
    assert assess(p, last_seen=datetime.now(timezone.utc).isoformat())["variant_resolved"] == 1

    multi = IkeaIndia(None).parse("https://www.ikea.com/in/en/p/vihals-10471234/",
                                  _ikea_page(aggregate=True, offercount=3))
    assert multi.attributes["price_varies"] is True


def test_jsonld_supplies_dimensions_the_accordion_omits():
    from catalog.sources.ikea_in import IkeaIndia
    p = IkeaIndia(None).parse("https://www.ikea.com/in/en/p/vihals-10471234/",
                              _ikea_page(aggregate=True))
    assert p.dimensions == {"width_mm": 1050.0, "height_mm": 2000.0, "depth_mm": 570.0}
    assert p.materials == ["Wood", "Glass"]


# -- completeness ----------------------------------------------------------
def test_flat_goods_are_not_judged_as_boxes():
    from catalog.quality import missing_axes
    # A rug has width and length and no height or depth. Judging it as a box
    # marks every rug in the catalogue unplaceable.
    rug = {"width_mm": 800.0, "length_mm": 1500.0}
    assert missing_axes(rug, "planar") == []
    assert missing_axes(rug, "box") == ["height"]
    # IKEA reports a bed as width x length, never width x depth.
    bed = {"width_mm": 1600.0, "length_mm": 2000.0}
    assert missing_axes(bed, "footprint") == []


def test_geometry_is_category_specific():
    from catalog.quality import geometry_for
    assert geometry_for("rugs") == "planar"
    assert geometry_for("wardrobes") == "box"
    assert geometry_for("ceiling_lighting") == "suspended"
    # Bought for a room, never placed in one as an object.
    assert geometry_for("paint") == "none"
    assert geometry_for("structural") == "none"


def test_searchable_and_eligible_are_different_questions():
    from catalog.quality import assess
    p = _product("bed1", 12990.0, "MALM Bed frame")
    p.design_category = "beds"
    p.materials, p.colors = ["particleboard"], ["white"]

    p.dimensions = {}
    without = assess(p, last_seen=datetime.now(timezone.utc).isoformat())
    assert without["searchable"] == 1, "a named, priced, pictured bed is findable"
    assert without["design_eligible"] == 0, "but it cannot be placed without a size"
    assert any(g.startswith("dimensions:") for g in without["quality_gaps"])

    p.dimensions = {"width_mm": 1600.0, "length_mm": 2000.0}
    with_dims = assess(p, last_seen=datetime.now(timezone.utc).isoformat())
    assert with_dims["design_eligible"] == 1


def test_stale_price_blocks_eligibility():
    from catalog.quality import assess
    p = _product("bed2", 12990.0, "MALM Bed frame")
    p.design_category, p.dimensions = "beds", {"width_mm": 1600.0, "length_mm": 2000.0}
    old = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
    marks = assess(p, last_seen=old)
    assert marks["price_current"] == 0 and marks["design_eligible"] == 0
    assert "price_stale" in marks["quality_gaps"]


def test_materials_stay_searchable_but_never_placeable():
    from catalog.quality import assess
    p = _product("cem", 355.0, "Maha PPC Cement, 50 Kg Bag")
    p.design_category = "structural"
    marks = assess(p, last_seen=datetime.now(timezone.utc).isoformat())
    assert marks["searchable"] == 1
    assert marks["design_eligible"] == 0
    assert marks["placement_geometry"] == "none"
    # Not a data gap - cement simply is not placed as an object.
    assert not any(g.startswith("dimensions:") for g in marks["quality_gaps"])


def test_pricing_requires_placement_only_where_it_applies(tmp_path="/tmp"):
    from catalog.search import _requires_placement
    assert _requires_placement("beds") is True
    assert _requires_placement("rugs") is True
    # A renovation quote must still be able to price paint and cement.
    assert _requires_placement("paint") is False
    assert _requires_placement("structural") is False


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
            import io
            from PIL import Image as PILImage
            buffer = io.BytesIO()
            PILImage.new("RGB", (4, 4), (int(url.rsplit("/", 1)[1].split(".")[0]), 0, 0)).save(buffer, format="PNG")
            body = buffer.getvalue()
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
        embedder=None, eligible_only=False,      # this test is about the floor, not completeness
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
