"""Embeddings for product text and product images.

CLIP puts both in one 512-d space, which is the property the whole pipeline rests
on: the design AI can hand back a cropped region of its render and it is compared
against catalogue photographs directly, with no captioning step in between. The
same space also accepts a text query, so "grey fabric 3-seater" and a picture of
one land near each other.

open_clip is optional. Nothing else in this package imports torch, so a scrape
and a plain SQL price lookup work without it.
"""
from __future__ import annotations

import io
import logging
import sqlite3
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

log = logging.getLogger(__name__)

DEFAULT_MODEL = "ViT-B-32"
DEFAULT_PRETRAINED = "laion2b_s34b_b79k"


class Embedder:
    """Lazy CLIP wrapper. Vectors come back L2-normalised, so cosine == dot."""

    def __init__(self, model_name: str = DEFAULT_MODEL, pretrained: str = DEFAULT_PRETRAINED):
        self.model_name = model_name
        self.pretrained = pretrained
        self._model = None
        self._preprocess = None
        self._tokenizer = None
        self._device = "cpu"

    @property
    def key(self) -> str:
        return f"open_clip/{self.model_name}/{self.pretrained}"

    def _load(self):
        if self._model is not None:
            return
        try:
            import open_clip
            import torch
        except ImportError as exc:                      # pragma: no cover
            raise RuntimeError(
                "Embeddings need open_clip: pip install open_clip_torch torch pillow"
            ) from exc
        self._device = (
            "mps" if torch.backends.mps.is_available()
            else "cuda" if torch.cuda.is_available()
            else "cpu"
        )
        model, _, preprocess = open_clip.create_model_and_transforms(
            self.model_name, pretrained=self.pretrained
        )
        model.eval().to(self._device)
        self._model, self._preprocess = model, preprocess
        self._tokenizer = open_clip.get_tokenizer(self.model_name)
        log.info("CLIP %s/%s on %s", self.model_name, self.pretrained, self._device)

    @property
    def dim(self) -> int:
        self._load()
        return int(self._model.text_projection.shape[1])

    def embed_texts(self, texts: Sequence[str], batch: int = 64) -> np.ndarray:
        self._load()
        import torch

        out = []
        for i in range(0, len(texts), batch):
            # CLIP truncates at 77 tokens; the first ~300 chars carry the
            # distinguishing part of embedding_text() by construction.
            toks = self._tokenizer([t[:300] for t in texts[i:i + batch]]).to(self._device)
            with torch.no_grad():
                v = self._model.encode_text(toks)
            out.append((v / v.norm(dim=-1, keepdim=True)).cpu().numpy().astype("float32"))
        return np.vstack(out) if out else np.zeros((0, self.dim), "float32")

    def embed_images(self, paths_or_blobs: Sequence[Path | bytes], batch: int = 32) -> np.ndarray:
        self._load()
        import torch
        from PIL import Image as PILImage

        out = []
        for i in range(0, len(paths_or_blobs), batch):
            tensors = []
            for item in paths_or_blobs[i:i + batch]:
                src = io.BytesIO(item) if isinstance(item, bytes) else item
                img = PILImage.open(src).convert("RGB")
                tensors.append(self._preprocess(img))
            stack = torch.stack(tensors).to(self._device)
            with torch.no_grad():
                v = self._model.encode_image(stack)
            out.append((v / v.norm(dim=-1, keepdim=True)).cpu().numpy().astype("float32"))
        return np.vstack(out) if out else np.zeros((0, self.dim), "float32")


# --------------------------------------------------------------------------
# persistence
# --------------------------------------------------------------------------
def _store(conn, rows: Iterable[tuple]) -> None:
    conn.executemany(
        "INSERT OR REPLACE INTO embeddings (product_key,kind,ref,model,dim,vec) "
        "VALUES (?,?,?,?,?,?)",
        rows,
    )
    conn.commit()


def embed_catalog(
    conn: sqlite3.Connection,
    embedder: Embedder,
    do_text: bool = True,
    do_images: bool = True,
    images_per_product: int = 2,
    batch: int = 64,
    limit: int | None = None,
) -> dict[str, int]:
    counts = {"text": 0, "image": 0, "skipped_images": 0}
    model = embedder.key

    if do_text:
        sql = """
            SELECT p.key, p.embedding_text FROM products p
             WHERE p.embedding_text IS NOT NULL AND p.embedding_text <> ''
               AND NOT EXISTS (SELECT 1 FROM embeddings e
                                WHERE e.product_key=p.key AND e.kind='text' AND e.model=?)
        """
        if limit:
            sql += f" LIMIT {int(limit)}"
        rows = conn.execute(sql, (model,)).fetchall()
        log.info("text embeddings to compute: %d", len(rows))
        for i in range(0, len(rows), batch):
            part = rows[i:i + batch]
            vecs = embedder.embed_texts([r["embedding_text"] for r in part])
            _store(conn, [
                (r["key"], "text", "", model, vecs.shape[1], vecs[j].tobytes())
                for j, r in enumerate(part)
            ])
            counts["text"] += len(part)

    if do_images:
        sql = f"""
            SELECT i.product_key, i.url, i.local_path FROM product_images i
             WHERE i.local_path IS NOT NULL AND i.position < {int(images_per_product)}
               AND NOT EXISTS (SELECT 1 FROM embeddings e
                                WHERE e.product_key=i.product_key AND e.kind='image'
                                  AND e.ref=i.url AND e.model=?)
        """
        if limit:
            sql += f" LIMIT {int(limit)}"
        rows = conn.execute(sql, (model,)).fetchall()
        log.info("image embeddings to compute: %d", len(rows))
        for i in range(0, len(rows), 32):
            part = rows[i:i + 32]
            paths, keep = [], []
            for r in part:
                p = Path(r["local_path"])
                if p.exists():
                    paths.append(p)
                    keep.append(r)
                else:
                    counts["skipped_images"] += 1
            if not paths:
                continue
            try:
                vecs = embedder.embed_images(paths)
            except Exception:
                log.exception("image batch failed, skipping %d files", len(paths))
                counts["skipped_images"] += len(paths)
                continue
            _store(conn, [
                (r["product_key"], "image", r["url"], model, vecs.shape[1], vecs[j].tobytes())
                for j, r in enumerate(keep)
            ])
            counts["image"] += len(keep)

    return counts


def load_matrix(
    conn: sqlite3.Connection, kind: str, model: str, keys: Sequence[str] | None = None
) -> tuple[list[str], np.ndarray]:
    """All stored vectors of one kind as a matrix, with the product key per row.

    Brute force over ~13k products is a sub-millisecond dot product; an ANN index
    only starts paying for itself a couple of orders of magnitude further up.
    """
    sql = "SELECT product_key, ref, vec, dim FROM embeddings WHERE kind=? AND model=?"
    params: list = [kind, model]
    if keys:
        sql += f" AND product_key IN ({','.join('?' * len(keys))})"
        params += list(keys)
    rows = conn.execute(sql, params).fetchall()
    if not rows:
        return [], np.zeros((0, 0), "float32")
    dim = rows[0]["dim"]
    mat = np.vstack([np.frombuffer(r["vec"], dtype="float32", count=dim) for r in rows])
    return [r["product_key"] for r in rows], mat
