"""Exact material variants, selling units and corruption recovery (no network)."""
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from catalog import store
from catalog.tools import CatalogService
from catalog.specs import variant_specs
from catalog.repair_assets import audit_images
from catalog.embed import embed_catalog
from test_catalog import _ikea
from test_design_workflow import FakeEmbedder

class CustomBuildTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.svc = CatalogService(self.root / 'catalog.db', use_vectors=False)
        self.p = _ikea('BWP plywood sheet', 'wood', 'plywood')
        store.upsert(self.svc.conn, [self.p])
        with self.svc.conn:
            self.svc.conn.execute("UPDATE products SET design_category=NULL,purchase_unit='sheet',price=100,specs=?", (json.dumps({'thickness_mm': 6}),))
            for ident, price, thick in [('thin', 100, '6mm'), ('thick', 500, '18mm')]:
                self.svc.conn.execute('INSERT INTO product_variants (product_key,variant_id,title,price,available,options) VALUES (?,?,?,?,?,?)', (self.p.key, ident, thick, price, 1, json.dumps({'Thickness': thick, 'Size': "8' x 4'"})))
        self.svc.get_product(self.p.key)

    def tearDown(self):
        self.svc.conn.close()
        self.tmp.cleanup()

    def build(self, **overrides):
        item = dict(role='wood', query='18mm plywood', product_key=self.p.key, variant_id='thick', quantity=2.2, purchase_unit='sheet')
        item.update(overrides)
        return dict(name='Built bed', category='beds', quantity=2, placement='back wall', specification='Plywood platform with veneer', inputs=[item])

    def test_material_search_finds_unclassified_sheet_goods(self):
        results = self.svc.search_catalog(query='plywood', material_kind='wood', placeable_only=False)['results']
        self.assertEqual([r['product_key'] for r in results], [self.p.key])
        self.assertEqual(self.svc.search_catalog(material_kind='hardware', placeable_only=False)['count'], 0)

    def test_variant_price_units_and_whole_sheet_rounding(self):
        manifest = self.svc.save_manifest([], custom_builds=[self.build()])
        wood = manifest['custom_builds'][0]['inputs'][0]
        self.assertEqual(wood['product']['unit_price'], 500)
        self.assertEqual(wood['product']['specifications']['thickness_mm'], 18)
        self.assertEqual(wood['product']['specifications']['sheet_size_mm'], [2438, 1219])
        self.assertEqual(wood['total_purchase_quantity'], 5)
        self.assertEqual(manifest['quote']['subtotal'], 2500)
        self.assertFalse(manifest['quote']['cost_complete'])
        self.assertEqual(len(manifest['quote']['pending_materials']), 2)
        self.assertIsNone(manifest['custom_builds'][0]['labour']['amount'])
        with self.assertRaisesRegex(ValueError, 'budget'):
            self.svc.save_manifest([], custom_builds=[self.build()], budget=2499)

    def test_unknown_or_wrong_unit_is_unpriced_not_zero_quantity(self):
        for changes in [dict(quantity=None), dict(purchase_unit='sq_ft')]:
            manifest = self.svc.save_manifest([], custom_builds=[self.build(**changes)])
            self.assertIsNone(manifest['custom_builds'][0]['inputs'][0]['line_total'])
            self.assertFalse(manifest['quote']['complete'])
            self.assertEqual(len(manifest['quote']['pending_materials']), 3)

    def test_unissued_ambiguous_and_unavailable_variants_rejected(self):
        for variant in [None, 'invented']:
            with self.assertRaisesRegex(ValueError, 'unambiguous'):
                self.svc.save_manifest([], custom_builds=[self.build(variant_id=variant)])
        with self.svc.conn:
            self.svc.conn.execute("UPDATE product_variants SET available=0 WHERE variant_id='thick'")
        with self.assertRaisesRegex(ValueError, 'available'):
            self.svc.save_manifest([], custom_builds=[self.build()])

    def test_finalization_rechecks_variant_stock_and_requires_custom_slot(self):
        manifest = self.svc.save_manifest([], custom_builds=[self.build()])
        review = dict(items=[dict(slot_id='custom-0', status='match')], room_preserved=True, extra_objects=[])
        with self.assertRaises(ValueError):
            self.svc.finalize_manifest(manifest['design_id'], {**review, 'items': []}, b'render')
        with self.svc.conn:
            self.svc.conn.execute("UPDATE product_variants SET available=0 WHERE variant_id='thick'")
        result = self.svc.finalize_manifest(manifest['design_id'], review, b'render')
        self.assertEqual(result['verification']['status'], 'needs_review')
        self.assertIn('unavailable', result['verification']['current_quote']['pending_materials'][0]['reason'])
        self.assertEqual(result['custom_builds'][0]['inputs'][0]['product']['unit_price'], 500)

    def test_finished_furniture_cannot_be_a_wood_input(self):
        with self.svc.conn:
            self.svc.conn.execute("UPDATE products SET design_category='beds'")
        with self.assertRaises(ValueError):
            self.svc.save_manifest([], custom_builds=[self.build()])

    def test_plywood_size_option_and_title_grade_override_ambiguous_family_copy(self):
        specs = variant_specs({'grade': ['MR', 'BWP']}, '18mm', {'Size': '18mm'}, 'BWP plywood')
        self.assertEqual(specs['thickness_mm'], 18)
        self.assertEqual(specs['grade'], ['BWP'])

    def test_variant_does_not_inherit_wrong_parent_size(self):
        self.assertEqual(variant_specs({'sheet_size_mm': [1, 2], 'thickness_mm': 6}, 'Unknown', {}), {})
        self.assertNotIn('thickness_mm', variant_specs({}, '450 mm', {'Length': '450mm'}))

    def test_corrupt_image_does_not_discard_valid_batch_and_audit_preserves_files(self):
        good = self.root / 'good.png'; bad = self.root / 'bad.jpg'
        Image.new('RGB', (4, 4), 'red').save(good); bad.write_bytes(b'<html>denied</html>')
        with self.svc.conn:
            self.svc.conn.execute('DELETE FROM product_images')
            for i, path in enumerate([bad, good]):
                self.svc.conn.execute('INSERT INTO product_images(product_key,position,url,local_path,sha256) VALUES (?,?,?,?,?)', (self.p.key, i, str(path), str(path), hashlib.sha256(path.read_bytes()).hexdigest()))
        counts = embed_catalog(self.svc.conn, FakeEmbedder(), do_text=False, images_per_product=3)
        self.assertEqual(counts['image'], 1)
        self.assertEqual(counts['skipped_images'], 1)
        self.assertEqual(audit_images(self.svc.conn)['invalid_records'], 1)
        self.assertEqual(audit_images(self.svc.conn, apply=True)['invalid_records'], 1)
        self.assertTrue(bad.exists())
        self.assertEqual(self.svc.conn.execute('SELECT COUNT(*) FROM product_images WHERE local_path IS NOT NULL').fetchone()[0], 1)

    def test_non_image_assets_are_removed_and_photos_promoted_without_redownload(self):
        from catalog.media import is_non_image_url
        self.assertTrue(is_non_image_url('https://example.com/video.MP4?size=large'))
        self.assertFalse(is_non_image_url('https://example.com/photo.webp'))
        with self.svc.conn:
            self.svc.conn.execute('DELETE FROM product_images')
            for position, url in enumerate(['a.jpg', 'video.mp4', 'manual.pdf', 'b.jpg']):
                self.svc.conn.execute('INSERT INTO product_images(product_key,position,url) VALUES (?,?,?)', (self.p.key, position, url))
        report = audit_images(self.svc.conn, apply=True)
        self.assertEqual(report['non_image_records'], 2)
        self.assertEqual([tuple(r) for r in self.svc.conn.execute('SELECT position,url FROM product_images ORDER BY position')], [(0, 'a.jpg'), (1, 'b.jpg')])

    def test_downloads_do_not_hold_sqlite_write_lock_during_network_wait(self):
        import io
        from unittest.mock import patch
        from catalog.images import download_missing
        stream = io.BytesIO(); Image.new('RGB', (2, 2), 'blue').save(stream, format='PNG')
        conn = self.svc.conn
        class SerialPool:
            def __init__(self, **kwargs): pass
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def map(self, fn, rows):
                for row in rows: yield fn(row)
        class Fetch:
            def get_bytes(self, url):
                assert not conn.in_transaction, 'network wait held the database write lock'
                return stream.getvalue()
        with conn:
            conn.execute('DELETE FROM product_images')
            for pos in range(3):
                conn.execute('INSERT INTO product_images(product_key,position,url) VALUES (?,?,?)', (self.p.key, pos, f'https://example.com/{pos}.png'))
        with patch('catalog.images.ThreadPoolExecutor', SerialPool):
            result = download_missing(conn, Fetch(), self.root/'images', commit_every=2)
        self.assertEqual(result['downloaded'] + result['deduped'], 3)
        self.assertEqual(conn.execute('SELECT COUNT(*) FROM product_images WHERE local_path IS NOT NULL').fetchone()[0], 3)

if __name__ == '__main__': unittest.main()
