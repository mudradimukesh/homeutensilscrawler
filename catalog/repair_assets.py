"""Audit downloaded images; detach invalid records without deleting source files.

python -m catalog.repair_assets --db data/catalog.db --apply
Then resume `catalog images` and `catalog embed` normally. No product scrape is needed.
"""
import argparse
import hashlib
import json
from pathlib import Path
from PIL import Image
from .store import connect
from .media import is_non_image_url


def audit_images(conn, apply=False):
    rows = conn.execute('SELECT product_key,position,local_path,sha256 FROM product_images WHERE local_path IS NOT NULL').fetchall()
    checked, invalid = {}, []
    for row in rows:
        identity = (row['local_path'], row['sha256'])
        if identity not in checked:
            try:
                path = Path(row['local_path'])
                with path.open('rb') as stream:
                    digest = hashlib.file_digest(stream, 'sha256').hexdigest()
                with Image.open(path) as image:
                    image.verify()
                checked[identity] = digest == row['sha256']
            except (OSError, ValueError, SyntaxError):
                checked[identity] = False
        if not checked[identity]:
            invalid.append(row)
    non_images = [r for r in conn.execute('SELECT product_key,position,url FROM product_images') if is_non_image_url(r['url'])]
    if apply:
        with conn:
            for row in invalid:
                conn.execute('UPDATE product_images SET local_path=NULL,sha256=NULL WHERE product_key=? AND position=? AND local_path=?',
                             (row['product_key'], row['position'], row['local_path']))
                conn.execute("DELETE FROM embeddings WHERE product_key=? AND kind='image' AND source_asset=?",
                             (row['product_key'], 'sha256:' + (row['sha256'] or '')))
            affected = {r['product_key'] for r in non_images}
            for row in non_images:
                conn.execute('DELETE FROM product_images WHERE product_key=? AND position=?', (row['product_key'], row['position']))
            # Promote actual photos into the first-three download window, preserving
            # their existing files and hashes. Negative positions avoid key collisions.
            for key in affected:
                positions = [r[0] for r in conn.execute('SELECT position FROM product_images WHERE product_key=? ORDER BY position', (key,))]
                for index, position in enumerate(positions):
                    conn.execute('UPDATE product_images SET position=? WHERE product_key=? AND position=?', (-index-1, key, position))
                conn.execute('UPDATE product_images SET position=-position-1 WHERE product_key=?', (key,))
    return {'non_image_records': len(non_images), 'downloaded_records': len(rows), 'unique_assets_checked': len(checked),
            'invalid_records': len(invalid), 'invalid_assets': sum(not ok for ok in checked.values()), 'applied': apply}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--db', default='data/catalog.db')
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    conn = connect(args.db)
    try:
        print(json.dumps(audit_images(conn, args.apply), indent=2))
    finally:
        conn.close()


if __name__ == '__main__':
    main()
