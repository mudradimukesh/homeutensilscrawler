"""Authenticated local API for the upload design workflow.

Run: CATALOG_API_TOKEN=... python -m catalog.design_api --db data/catalog.db
Proxy /api/catalog through the interior app. Do not expose this local service publicly.
"""
from __future__ import annotations

import argparse
import base64
import binascii
import hmac
import json
import logging
import os
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from .embed import Embedder
from .tools import CatalogService

MAX_BODY = 30 * 1024 * 1024
log = logging.getLogger(__name__)


def image_mime(blob):
    if blob.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if blob.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if blob[:4] == b"RIFF" and blob[8:12] == b"WEBP":
        return "image/webp"
    raise ValueError("PNG, JPEG or WebP image required")


def decode_image(value):
    if not isinstance(value, str):
        raise ValueError("base64 image required")
    try:
        result = base64.b64decode(value, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise ValueError("invalid base64 image") from exc
    if not result or len(result) > 20 * 1024 * 1024:
        raise ValueError("image must contain 1 byte to 20 MiB")
    image_mime(result)
    return result


def handle(service, action, body):
    if action == "image":
        return {"image_id": service.register_query_image(decode_image(body.get("image")))}
    if action == "search":
        return service.search_catalog(**body["query"])
    if action == "product":
        return service.get_product(body["product_key"])
    if action == "manifest":
        return service.save_manifest(body["objects"], budget=body.get("budget"), design_id=body.get("design_id"))
    if action == "get-manifest":
        return service.get_manifest(body["design_id"])
    if action == "asset":
        blob = service.asset_bytes(body["design_id"], body["asset"])
        return {"image": base64.b64encode(blob).decode(), "media_type": image_mime(blob)}
    if action == "finalize":
        return service.finalize_manifest(body["design_id"], body["verification"], decode_image(body.get("image")))
    raise ValueError("unknown operation")


def make_handler(db_path, token, image_root=None):
    embedder = Embedder()

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            if not hmac.compare_digest(self.headers.get("Authorization", ""), "Bearer " + token):
                self.reply(401, {"error": "unauthorized"})
                return
            action = self.path.removeprefix("/api/catalog/")
            if not self.path.startswith("/api/catalog/"):
                self.reply(404, {"error": "not_found"})
                return
            service = None
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length <= 0 or length > MAX_BODY:
                    self.reply(413, {"error": "request too large or empty"})
                    return
                if self.headers.get_content_type() != "application/json":
                    raise ValueError("JSON required")
                body = json.loads(self.rfile.read(length))
                if not isinstance(body, dict):
                    raise ValueError("JSON object required")
                session_id = body.pop("session_id", None)
                if action != "session" and (not isinstance(session_id, str) or len(session_id) != 32):
                    raise ValueError("session_id required")
                service = CatalogService(db_path, session_id=session_id if action != "session" else None,
                                         embedder=embedder, image_root=image_root)
                result = {"session_id": service.session_id} if action == "session" else handle(service, action, body)
                self.reply(200, result)
            except (ValueError, TypeError, KeyError) as exc:
                self.reply(400, {"error": str(exc)})
            except Exception:
                log.exception("catalog request failed")
                self.reply(500, {"error": "catalog request failed; check service logs"})
            finally:
                if service:
                    service.conn.close()

        def reply(self, status, result):
            raw = json.dumps(result, allow_nan=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(raw)

    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default="data/catalog.db")
    parser.add_argument("--images", help="image root; defaults to images beside the DB")
    parser.add_argument("--port", type=int, default=8766)
    args = parser.parse_args()
    token = os.environ.get("CATALOG_API_TOKEN", "")
    if len(token) < 24:
        parser.error("set CATALOG_API_TOKEN to a random secret of at least 24 characters")
    if not Path(args.db).is_file():
        parser.error("catalog database does not exist; load the existing JSONL first")
    HTTPServer(("127.0.0.1", args.port), make_handler(args.db, token, args.images)).serve_forever()


if __name__ == "__main__":
    main()
