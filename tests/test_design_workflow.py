"""Regression tests for catalog-constrained upload designs. No network or ML downloads."""
import base64
import hashlib
import io
import json
import sys
import tempfile
import unittest
from email.message import Message
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from catalog import store
from catalog.embed import _store
from catalog.tools import CatalogService
from catalog.search import find_products
from catalog.design_api import make_handler, handle
from test_catalog import _ikea

PNG = base64.b64decode('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jRZkAAAAASUVORK5CYII=')

class FakeEmbedder:
    key = ('test-clip', 'v1')
    def embed_texts(self, texts):
        return np.array([[1., 0.]] * len(texts), dtype='float32')
    def embed_images(self, images):
        return np.array([[0., 1.]] * len(images), dtype='float32')

class DesignWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.db = self.root / 'catalog.db'
        self.svc = CatalogService(self.db, use_vectors=False)
        self.connections = [self.svc.conn]
        self.a = self.product('A bed', 'A', 1800)
        self.b = self.product('B bed', 'B', 1400)
        store.upsert(self.svc.conn, [self.a, self.b])

    def tearDown(self):
        for conn in self.connections:
            conn.close()
        self.tmp.cleanup()

    def product(self, name, family, width):
        p = _ikea(name, family, 'bed frame')
        p.design_category = 'beds'
        p.dimensions = {'width_mm': width, 'length_mm': 2000}
        return p

    def reopen(self, **kwargs):
        s = CatalogService(self.db, **kwargs)
        self.connections.append(s.conn)
        return s

    def asset(self, key):
        image = self.root / 'images' / 'asset.png'
        image.parent.mkdir(exist_ok=True)
        image.write_bytes(PNG)
        with self.svc.conn:
            self.svc.conn.execute('UPDATE product_images SET local_path=?,sha256=? WHERE product_key=?',
                                  (str(image), hashlib.sha256(PNG).hexdigest(), key))

    def select(self):
        key = self.svc.search_catalog(query='A', category='beds')['results'][0]['product_key']
        self.asset(key)
        return [{'product_key': key, 'label': 'bed', 'quantity': 1, 'placement': 'back wall'}]

    def review(self, manifest, status='match'):
        return {'items': [{'slot_id': p['slot_id'], 'status': status} for p in manifest['products']],
                'extra_objects': [], 'room_preserved': True}

    def test_dimensions_filter_before_family_collapse(self):
        small = self.product('A bed small', 'A', 1100)
        small.price = 15000
        store.upsert(self.svc.conn, [small])
        result = self.svc.search_catalog(category='beds', max_width_mm=1200, limit=1)
        self.assertEqual([x['product_key'] for x in result['results']], [small.key])
        self.assertEqual(self.svc.search_catalog(category='beds', max_height_mm=3000)['count'], 0)
        self.assertEqual(self.svc.search_catalog(category='beds', max_depth_mm=1900)['count'], 0)
        self.assertIn('error', self.svc.search_catalog(category='beds', max_width_mm=-1))

    def test_keyword_pool_does_not_discard_filtered_matches(self):
        bad = [self.product('bed '*10 + str(i), str(i), 3000) for i in range(410)]
        store.upsert(self.svc.conn, bad)
        result = find_products(self.svc.conn, query_text='bed', max_width_mm=1500, candidate_pool=3)
        self.assertEqual([m.key for m in result], [self.b.key])

    def test_service_uses_both_vector_modalities(self):
        _store(self.svc.conn, [(self.a.key, 'text', 'sha256:a', *FakeEmbedder.key, 2,
                               np.array([1.,0.], 'float32').tobytes(), 'now'),
                              (self.b.key, 'image', 'sha256:b', *FakeEmbedder.key, 2,
                               np.array([0.,1.], 'float32').tobytes(), 'now')])
        svc = self.reopen(embedder=FakeEmbedder())
        self.assertEqual(svc.search_catalog(query='minimalist')['retrieval_mode'], 'hybrid')
        image_id = svc.register_query_image(PNG)
        self.assertEqual(svc.search_catalog(reference_image_id=image_id)['results'][0]['product_key'], self.b.key)
        self.assertIn('error', self.svc.search_catalog(reference_image_id=image_id))
        resumed = self.reopen(session_id=svc.session_id, embedder=FakeEmbedder())
        self.assertEqual(resumed.search_catalog(reference_image_id=image_id)['count'], 1)

    def test_visual_search_does_not_silently_return_cheapest(self):
        image_id = self.svc.register_query_image(PNG)
        self.assertEqual(self.svc.search_catalog(reference_image_id=image_id)['error'], 'visual_search_unavailable')
        self.assertEqual(self.svc.search_catalog(reference_image_id='/etc/passwd')['error'], 'unknown_reference_image')

    def test_only_in_stock_and_downloaded_when_requested(self):
        with self.svc.conn:
            self.svc.conn.execute("UPDATE products SET availability='store_only' WHERE key=?", (self.a.key,))
        self.assertEqual(self.svc.search_catalog(category='beds')['count'], 1)
        self.assertEqual(self.svc.search_catalog(category='beds', render_ready_only=True)['count'], 0)
        self.asset(self.b.key)
        self.assertEqual(self.svc.search_catalog(category='beds', render_ready_only=True)['count'], 1)

    def test_strict_pricing_rejects_unkeyed_and_invalid_quantities(self):
        self.svc.get_product(self.a.key)
        objects = [{'label': 'bed'}, 'bad'] + [dict(product_key=self.a.key, quantity=q) for q in (0, -1, True, '1', float('nan'), float('inf'))]
        quote = self.svc.price_design(objects)
        self.assertEqual(len(quote['rejected_items']), len(objects))
        self.assertEqual(quote['subtotal'], 0)
        self.assertFalse(quote['complete'])

    def test_ledger_survives_restart_and_is_session_scoped(self):
        self.svc.get_product(self.a.key)
        resumed = self.reopen(session_id=self.svc.session_id, use_vectors=False)
        obj = [{'product_key': self.a.key, 'quantity': 2}]
        self.assertTrue(resumed.price_design(obj)['complete'])
        other = self.reopen(use_vectors=False)
        self.assertFalse(other.price_design(obj)['complete'])
        with self.svc.conn:
            self.svc.conn.execute("UPDATE products SET availability='out_of_stock' WHERE key=?", (self.a.key,))
        self.assertFalse(resumed.price_design(obj)['complete'])

    def test_manifest_pins_assets_and_supports_idempotent_retry(self):
        objects = self.select()
        manifest = self.svc.save_manifest(objects, design_id='retry')
        self.assertEqual(manifest, self.svc.save_manifest(objects, design_id='retry'))
        with self.assertRaises(ValueError):
            self.svc.save_manifest([{**objects[0], 'quantity': 2}], design_id='retry')
        resumed = self.reopen(session_id=self.svc.session_id, use_vectors=False)
        self.assertEqual(resumed.get_manifest('retry'), manifest)
        with self.svc.conn:
            self.svc.conn.execute('DELETE FROM product_images')
            self.svc.conn.execute('UPDATE products SET price=1')
        self.assertEqual(resumed.asset_bytes('retry', manifest['products'][0]['asset']), PNG)
        self.assertEqual(resumed.get_manifest('retry')['products'][0]['unit_price'], 12990)
        with self.assertRaises(ValueError):
            self.reopen(use_vectors=False).asset_bytes('retry', manifest['products'][0]['asset'])

    def test_manifest_rejects_missing_assets_budget_and_fit(self):
        self.svc.get_product(self.a.key)
        obj = [{'product_key': self.a.key, 'quantity': 1}]
        with self.assertRaisesRegex(ValueError, 'download'):
            self.svc.save_manifest(obj)
        self.asset(self.a.key)
        with self.assertRaisesRegex(ValueError, 'budget'):
            self.svc.save_manifest(obj, budget=100)
        with self.assertRaisesRegex(ValueError, 'width'):
            self.svc.save_manifest([{**obj[0], 'max_width_mm': 100}])
        self.assertEqual(self.svc.conn.execute('SELECT COUNT(*) FROM design_manifests').fetchone()[0], 0)

    def test_finalize_requires_complete_review_and_rechecks_stock(self):
        objects = self.select()
        manifest = self.svc.save_manifest(objects)
        with self.assertRaises(ValueError):
            self.svc.finalize_manifest(manifest['design_id'], {'items': []}, PNG)
        reviewed = self.svc.finalize_manifest(manifest['design_id'], self.review(manifest), PNG)
        self.assertEqual(reviewed['verification']['status'], 'passed_visual_review')
        uncertain = self.svc.finalize_manifest(manifest['design_id'], self.review(manifest, 'uncertain'), PNG)
        self.assertEqual(uncertain['verification']['status'], 'needs_review')
        with self.svc.conn:
            self.svc.conn.execute("UPDATE products SET availability='out_of_stock'")
        stale = self.svc.finalize_manifest(manifest['design_id'], self.review(manifest), PNG)
        self.assertFalse(stale['verification']['current_quote']['complete'])
        self.assertEqual(stale['verification']['status'], 'needs_review')

    def test_stale_observation_is_rejected_even_if_cached_flag_is_fresh(self):
        self.svc.get_product(self.a.key)
        with self.svc.conn:
            self.svc.conn.execute("UPDATE products SET last_seen='2000-01-01',price_current=1 WHERE key=?", (self.a.key,))
        self.assertEqual(self.svc.get_product(self.a.key)['price_state'], 'stale')
        keys = [p['product_key'] for p in self.svc.search_catalog(category='beds')['results']]
        self.assertNotIn(self.a.key, keys)
        self.assertFalse(self.svc.price_design([{'product_key': self.a.key}])['complete'])

    def test_budget_is_rechecked_after_render_and_ambiguous_variants_are_rejected(self):
        objects = self.select()
        manifest = self.svc.save_manifest(objects, budget=13000)
        with self.svc.conn:
            self.svc.conn.execute('UPDATE products SET price=15000 WHERE key=?', (objects[0]['product_key'],))
        result = self.svc.finalize_manifest(manifest['design_id'], self.review(manifest), PNG)
        self.assertEqual(result['verification']['status'], 'needs_review')
        with self.svc.conn:
            self.svc.conn.execute('INSERT INTO product_variants (product_key,variant_id) VALUES (?,?)', (objects[0]['product_key'], 'second'))
        with self.assertRaisesRegex(ValueError, 'variants'):
            self.svc.save_manifest(objects)

    def test_schema_upgrade_preserves_existing_catalog_rows(self):
        with self.svc.conn:
            for table in ('design_assets', 'design_manifests', 'catalog_query_images', 'catalog_issued', 'catalog_sessions'):
                self.svc.conn.execute('DROP TABLE ' + table)
        upgraded = self.reopen(use_vectors=False)
        self.assertEqual(upgraded.search_catalog(category='beds')['count'], 2)
        self.assertEqual(upgraded.conn.execute('SELECT COUNT(*) FROM product_images').fetchone()[0], 2)

    def test_api_auth_and_asset_access(self):
        handler_class = make_handler(self.db, 'test-secret')
        handler = object.__new__(handler_class)
        handler.headers = Message()
        handler.reply = lambda status, body: setattr(handler, 'result', (status, body))
        handler.do_POST()
        self.assertEqual(handler.result[0], 401)
        handler.headers['Authorization'] = 'Bearer test-secret'
        handler.headers['Content-Type'] = 'application/json'
        handler.headers['Content-Length'] = '2'
        handler.path = '/api/catalog/session'
        handler.rfile = io.BytesIO(b'{}')
        handler.do_POST()
        self.assertEqual(handler.result[0], 200)
        self.assertEqual(len(handler.result[1]['session_id']), 32)
        manifest = self.svc.save_manifest(self.select())
        asset = handle(self.svc, 'asset', {'design_id': manifest['design_id'], 'asset': manifest['products'][0]['asset']})
        self.assertEqual(asset['media_type'], 'image/png')
        self.assertEqual(base64.b64decode(asset['image']), PNG)

if __name__ == '__main__':
    unittest.main()
