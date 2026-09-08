"""Polite HTTP layer: robots.txt, per-host rate limiting, retries, on-disk cache."""
from __future__ import annotations

import gzip
import hashlib
import logging
import random
import threading
import time
import urllib.robotparser
from pathlib import Path
from urllib.parse import urlparse

import requests

log = logging.getLogger(__name__)

UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)


def _accept_encoding() -> str:
    """Advertise only what this install can actually decode.

    IKEA's CDN honours `br`, and requests hands back undecoded Brotli as text if
    the decoder is missing — which looks like a parser bug, not a transport one,
    because the fetch still returns HTTP 200 and a plausible-length body.
    """
    encodings = ["gzip", "deflate"]
    for module, token in (("brotli", "br"), ("brotlicffi", "br"), ("zstandard", "zstd")):
        if token in encodings:
            continue
        try:
            __import__(module)
        except ImportError:
            continue
        encodings.append(token)
    return ", ".join(encodings)


ACCEPT_ENCODING = _accept_encoding()


class RateLimiter:
    """One minimum-delay gate per host, shared across threads."""

    def __init__(self, delay: float):
        self.delay = delay
        self._next: dict[str, float] = {}
        self._lock = threading.Lock()

    def wait(self, host: str) -> None:
        while True:
            with self._lock:
                now = time.monotonic()
                ready = self._next.get(host, 0.0)
                if now >= ready:
                    # jitter so we never march in lockstep
                    self._next[host] = now + self.delay * random.uniform(0.85, 1.3)
                    return
            time.sleep(min(ready - now, 1.0))


class Fetcher:
    """Fetches pages, obeying robots.txt and caching bodies on disk.

    The cache is what makes re-parsing free: extraction rules change far more
    often than the pages do, so a re-run after a parser fix costs no requests.
    """

    def __init__(
        self,
        cache_dir: Path,
        delay: float = 1.0,
        timeout: int = 30,
        obey_robots: bool = True,
        max_retries: int = 3,
        cache_max_age: float | None = None,
    ):
        self.cache_dir = Path(cache_dir)
        # None = a cached page never expires (good while writing a parser);
        # 0 = always refetch (what a scheduled refresh wants).
        self.cache_max_age = cache_max_age
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.limiter = RateLimiter(delay)
        self.timeout = timeout
        self.obey_robots = obey_robots
        self.max_retries = max_retries
        self._robots: dict[str, urllib.robotparser.RobotFileParser | None] = {}
        self._robots_lock = threading.Lock()
        self._local = threading.local()

    # -- session -----------------------------------------------------------
    @property
    def session(self) -> requests.Session:
        s = getattr(self._local, "session", None)
        if s is None:
            s = requests.Session()
            s.headers.update(
                {
                    "User-Agent": UA,
                    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                    "Accept-Language": "en-IN,en;q=0.9",
                    "Accept-Encoding": ACCEPT_ENCODING,
                }
            )
            self._local.session = s
        return s

    # -- robots ------------------------------------------------------------
    def allowed(self, url: str) -> bool:
        if not self.obey_robots:
            return True
        host = urlparse(url).netloc
        with self._robots_lock:
            rp = self._robots.get(host, "missing")
        if rp == "missing":
            rp = self._load_robots(url)
            with self._robots_lock:
                self._robots[host] = rp
        return True if rp is None else rp.can_fetch(UA, url)

    def _load_robots(self, url: str):
        parts = urlparse(url)
        robots_url = f"{parts.scheme}://{parts.netloc}/robots.txt"
        try:
            r = self.session.get(robots_url, timeout=self.timeout)
            if r.status_code != 200:
                return None
            rp = urllib.robotparser.RobotFileParser()
            rp.parse(r.text.splitlines())
            return rp
        except requests.RequestException:
            return None

    # -- cache -------------------------------------------------------------
    def _cache_path(self, url: str) -> Path:
        h = hashlib.sha256(url.encode()).hexdigest()
        return self.cache_dir / h[:2] / f"{h}.gz"

    def cached(self, url: str) -> str | None:
        p = self._cache_path(url)
        if p.exists():
            if self.cache_max_age is not None and (
                time.time() - p.stat().st_mtime > self.cache_max_age
            ):
                return None
            try:
                return gzip.decompress(p.read_bytes()).decode("utf-8", "replace")
            except OSError:
                p.unlink(missing_ok=True)
        return None

    def _store(self, url: str, body: str) -> None:
        p = self._cache_path(url)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(gzip.compress(body.encode("utf-8"), 6))

    # -- fetch -------------------------------------------------------------
    def get(self, url: str, *, use_cache: bool = True, force: bool = False) -> str | None:
        if use_cache and not force:
            body = self.cached(url)
            if body is not None:
                return body

        if not self.allowed(url):
            log.warning("robots.txt disallows %s", url)
            return None

        host = urlparse(url).netloc
        backoff = 2.0
        for attempt in range(1, self.max_retries + 1):
            self.limiter.wait(host)
            try:
                r = self.session.get(url, timeout=self.timeout)
            except requests.RequestException as exc:
                log.warning("%s (attempt %d/%d) %s", url, attempt, self.max_retries, exc)
            else:
                if r.status_code == 200:
                    if use_cache:
                        self._store(url, r.text)
                    return r.text
                if r.status_code in (404, 410):
                    log.info("gone: %s (%d)", url, r.status_code)
                    return None
                if r.status_code == 429:
                    wait = float(r.headers.get("Retry-After", backoff * 4))
                    log.warning("429 on %s, sleeping %.0fs", url, wait)
                    time.sleep(wait)
                else:
                    log.warning("%s -> HTTP %d (attempt %d)", url, r.status_code, attempt)
            if attempt < self.max_retries:
                time.sleep(backoff)
                backoff *= 2
        return None

    def get_bytes(self, url: str) -> bytes | None:
        """Binary fetch (images). Never cached as text; the image store is the cache."""
        if not self.allowed(url):
            return None
        host = urlparse(url).netloc
        for attempt in range(1, self.max_retries + 1):
            self.limiter.wait(host)
            try:
                r = self.session.get(url, timeout=self.timeout)
                if r.status_code == 200:
                    return r.content
                if r.status_code in (404, 410):
                    return None
            except requests.RequestException as exc:
                log.warning("image %s: %s", url, exc)
            time.sleep(1.5 * attempt)
        return None
