"""Proposed carpenter-built objects and their separate catalog procurement inputs."""
from __future__ import annotations
import math

BUILD_CATEGORIES = {'beds', 'wardrobes', 'nightstands', 'dressers', 'tables', 'desks', 'shelving', 'storage'}
MATERIAL_WORDS = {
    'wood': ('plywood', 'timber', 'lumber', 'mdf', 'particle board', 'blockboard', 'wood board', 'wood plank'),
    'hardware': ('hinge', 'screw', 'nail', 'bolt', 'bracket', 'runner', 'drawer slide', 'handle', 'lock', 'fastener'),
    'finish': ('laminate', 'veneer', 'varnish', 'wood stain', 'wood polish', 'edge band', 'edgeband', 'adhesive', 'wood glue'),
}
FURNITURE_CATEGORIES = ('beds', 'wardrobes', 'sofas', 'chairs', 'tables', 'desks', 'nightstands', 'dressers', 'bedding', 'rugs', 'curtains', 'lamps', 'decor')


def material_filter(kind):
    if kind not in MATERIAL_WORDS:
        raise ValueError('material kind must be wood, hardware or finish')
    words = MATERIAL_WORDS[kind]
    return ("COALESCE(design_category,'') NOT IN (" + ','.join('?' for _ in FURNITURE_CATEGORIES) + ") AND (" +
            ' OR '.join("LOWER(name) LIKE ?" for _ in words) + ")", list(FURNITURE_CATEGORIES) + ['%' + word + '%' for word in words])


def positive(value, label, integer=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0 or value > 100000 or (integer and int(value) != value):
        raise ValueError(label + ' must be a positive ' + ('integer' if integer else 'number'))
    return value


def text(value, label):
    if not isinstance(value, str) or not value.strip() or len(value) > 4000:
        raise ValueError(label + ' is required (up to 4000 characters)')
    return value.strip()


def normalize_builds(builds):
    if not isinstance(builds, list) or len(builds) > 8:
        raise ValueError('custom_builds must be a list of up to 8 objects')
    out = []
    for index, build in enumerate(builds):
        if not isinstance(build, dict) or build.get('category') not in BUILD_CATEGORIES or build.get('product_key'):
            raise ValueError('custom furniture needs a supported category, not a finished-product SKU')
        dimensions = build.get('dimensions_mm', {})
        if not isinstance(dimensions, dict):
            raise ValueError('custom dimensions must be an object')
        dimensions = {axis: None if dimensions.get(axis) is None else positive(dimensions[axis], axis)
                      for axis in ('width', 'depth', 'height')}
        inputs = build.get('inputs', [])
        if not isinstance(inputs, list) or len(inputs) > 12:
            raise ValueError('custom furniture supports up to 12 material inputs')
        materials = []
        for item in inputs:
            if not isinstance(item, dict) or item.get('role') not in MATERIAL_WORDS:
                raise ValueError('each material needs role wood, hardware or finish')
            qty = item.get('quantity')
            if qty is not None:
                positive(qty, 'material quantity per built object')
            key = item.get('product_key')
            if key is not None and (not isinstance(key, str) or not key):
                raise ValueError('material product_key must be a catalog key or null')
            if item.get('variant_id') is not None and not isinstance(item['variant_id'], str):
                raise ValueError('variant_id must be a string or null')
            unit = item.get('purchase_unit')
            if unit is not None and (not isinstance(unit, str) or not unit.strip()):
                raise ValueError('purchase_unit must match the catalog or be null')
            materials.append({'role': item['role'], 'query': text(item.get('query'), 'material description'),
                              'product_key': key, 'variant_id': item.get('variant_id'), 'quantity': qty, 'purchase_unit': unit,
                              'quantity_state': 'estimate' if qty is not None else 'unknown'})
        # A wood-only proposal is not presented as a complete procurement schedule.
        for role in MATERIAL_WORDS:
            if not any(m['role'] == role for m in materials):
                materials.append({'role': role, 'query': role + ' specification pending carpenter review',
                                  'product_key': None, 'quantity': None, 'purchase_unit': None, 'quantity_state': 'unknown'})
        out.append({'slot_id': f'custom-{index}', 'fulfillment': 'custom_build',
                    'name': text(build.get('name'), 'custom furniture name'), 'category': build['category'],
                    'quantity': positive(build.get('quantity', 1), 'furniture quantity', integer=True),
                    'placement': text(build.get('placement'), 'placement'),
                    'specification': text(build.get('specification'), 'design specification'),
                    'dimensions_mm': dimensions, 'dimensions_state': 'proposed_requires_measurement',
                    'inputs': materials, 'carpenter_review_required': True,
                    'labour': {'amount': None, 'status': 'not_quoted'},
                    'pending': ['Carpenter-confirmed measurements, cut list, joinery and wastage', 'Labour, delivery and installation quote']})
    return out
