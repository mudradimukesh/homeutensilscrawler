"""Source interface. A source discovers product URLs, then turns each page into a Product."""
from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod
from typing import Iterable

from ..http import Fetcher
from ..models import Product

log = logging.getLogger(__name__)

_LOC_RE = re.compile(r"<loc>\s*([^<\s]+)\s*</loc>", re.I)


def sitemap_locs(xml: str) -> list[str]:
    """Every <loc> in a sitemap or sitemap index.

    Deliberately regex rather than an XML parser: IKEA's per-locale sitemaps are
    ~50 MB of one line each, and a DOM parse of that costs far more than it buys.
    """
    return _LOC_RE.findall(xml)


class Source(ABC):
    name: str = ""
    base_url: str = ""
    country: str = "IN"

    def __init__(self, fetcher: Fetcher):
        self.fetcher = fetcher

    @abstractmethod
    def discover(self, limit: int | None = None) -> Iterable[str]:
        """Yield product page URLs."""

    @abstractmethod
    def parse(self, url: str, html: str) -> Product | None:
        """Turn one product page into a Product, or None if it isn't one."""

    def scrape_one(self, url: str, force: bool = False) -> Product | None:
        html = self.fetcher.get(url, force=force)
        if not html:
            return None
        try:
            return self.parse(url, html)
        except Exception:
            log.exception("parse failed: %s", url)
            return None
