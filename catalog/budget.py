"""A disk budget for the data directory.

A full crawl does not fit in a small allowance — ~38k product images run to
several gigabytes and the cached pages to a couple more — so the limit has to be
enforced while writing, not checked afterwards. Every large writer (page cache,
image store, JSONL) asks before it writes and stops the run cleanly when the next
write would cross the line, leaving the database consistent and the work so far
intact.

Walking the tree on every write would cost more than the writes, so a baseline is
measured once and kept up to date incrementally; the walk is repeated only near
the limit, where drift would otherwise decide the outcome.
"""
from __future__ import annotations

import logging
import os
import re
import threading
from pathlib import Path

log = logging.getLogger(__name__)

DEFAULT_LIMIT = 1024 ** 3          # 1 GB
_SIZE_RE = re.compile(r"^\s*([\d.]+)\s*([kmgt]?)b?\s*$", re.I)
_UNITS = {"": 1, "k": 1024, "m": 1024 ** 2, "g": 1024 ** 3, "t": 1024 ** 4}


def parse_size(text: str | int | float) -> int:
    """'1GB', '750 MB', '1.5g', '0' (no limit), or a plain byte count."""
    if isinstance(text, (int, float)):
        return int(text)
    m = _SIZE_RE.match(str(text))
    if not m:
        raise ValueError(f"not a size: {text!r} (try 1GB, 750MB, 0 for unlimited)")
    return int(float(m.group(1)) * _UNITS[m.group(2).lower()])


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024 or unit == "TB":
            return f"{n:,.0f} {unit}" if unit == "B" else f"{n:,.1f} {unit}"
        n /= 1024
    return f"{n:,.1f} TB"


def dir_size(path: Path) -> int:
    total = 0
    stack = [Path(path)]
    while stack:
        current = stack.pop()
        try:
            with os.scandir(current) as entries:
                for e in entries:
                    try:
                        if e.is_dir(follow_symlinks=False):
                            stack.append(Path(e.path))
                        elif e.is_file(follow_symlinks=False):
                            total += e.stat(follow_symlinks=False).st_size
                    except OSError:
                        continue          # vanished mid-walk; not worth failing over
        except OSError:
            # Unreadable or exotic entries (/dev/fd, revoked descriptors, denied
            # directories) must not abort a size check.
            continue
    return total


class BudgetExceeded(RuntimeError):
    """Raised when a write would take the data directory over its limit."""

    def __init__(self, used: int, limit: int, root: Path):
        self.used, self.limit, self.root = used, limit, root
        super().__init__(
            f"data directory is at {human(used)} of the {human(limit)} limit "
            f"({root}) — stopping before it grows further"
        )


class DiskBudget:
    """Tracks bytes under `root` against a ceiling. Safe to share across threads."""

    def __init__(self, root: str | Path, limit: int = DEFAULT_LIMIT,
                 recheck_margin: float = 0.05, reserve: int | None = None):
        self.root = Path(root)
        self.limit = int(limit)
        # Not everything that grows can be charged per write — SQLite extends its
        # WAL on its own schedule — so writers stop a little short of the ceiling.
        # Without this the directory settles just *over* the number the user set.
        self.reserve = (int(reserve) if reserve is not None
                        else min(int(self.limit * 0.02), 32 * 1024 ** 2))
        self.ceiling = max(self.limit - self.reserve, 0)
        # Within this fraction of the ceiling, re-walk rather than trust the
        # running total: that is exactly where being wrong changes the decision.
        self.recheck_margin = recheck_margin
        self._lock = threading.Lock()
        self._used = dir_size(self.root) if self.root.exists() else 0
        self.stopped_reason: str | None = None

    # -- state -------------------------------------------------------------
    @property
    def enabled(self) -> bool:
        return self.limit > 0

    @property
    def used(self) -> int:
        with self._lock:
            return self._used

    @property
    def remaining(self) -> int:
        return max(self.ceiling - self.used, 0) if self.enabled else -1

    def add(self, nbytes: int) -> None:
        with self._lock:
            self._used += int(nbytes)

    def resync(self) -> int:
        actual = dir_size(self.root) if self.root.exists() else 0
        with self._lock:
            self._used = actual
        return actual

    # -- decisions ---------------------------------------------------------
    def would_exceed(self, need: int = 0) -> bool:
        if not self.enabled:
            return False
        with self._lock:
            used = self._used
        if used + need <= self.ceiling * (1 - self.recheck_margin):
            return False
        return self.resync() + need > self.ceiling

    def check(self, need: int = 0) -> None:
        """Raise BudgetExceeded if writing `need` more bytes would cross the limit."""
        if self.would_exceed(need):
            self.stopped_reason = (
                f"disk budget reached: {human(self.used)} of {human(self.limit)}"
            )
            raise BudgetExceeded(self.used, self.limit, self.root)

    # -- reporting ---------------------------------------------------------
    def breakdown(self) -> dict:
        parts = {}
        for child in sorted(self.root.iterdir()) if self.root.exists() else []:
            if child.is_dir():
                parts[child.name + "/"] = dir_size(child)
            elif child.is_file():
                parts[child.name] = child.stat().st_size
        total = sum(parts.values())
        with self._lock:
            self._used = total
        return {
            "root": str(self.root),
            "limit": self.limit,
            "limit_human": human(self.limit) if self.enabled else "unlimited",
            "used": total,
            "used_human": human(total),
            "percent": round(total / self.limit * 100, 1) if self.enabled else None,
            "stop_at": self.ceiling if self.enabled else None,
            "stop_at_human": human(self.ceiling) if self.enabled else None,
            "remaining": max(self.ceiling - total, 0) if self.enabled else None,
            "remaining_human": human(max(self.ceiling - total, 0)) if self.enabled else None,
            "parts": dict(sorted(parts.items(), key=lambda kv: -kv[1])),
        }


def report(breakdown: dict) -> str:
    lines = [
        f"{breakdown['root']}",
        f"  used   {breakdown['used_human']}"
        + (f" of {breakdown['limit_human']} ({breakdown['percent']}%)"
           if breakdown["percent"] is not None else ""),
    ]
    for name, size in breakdown["parts"].items():
        lines.append(f"    {name:<16} {human(size):>10}")
    if breakdown["percent"] is not None:
        lines.append(f"  free   {breakdown['remaining_human']} "
                     f"(writers stop at {breakdown['stop_at_human']})")
        if breakdown["percent"] >= 90:
            lines.append("  data/cache/ and data/thumbs/ are regenerable — "
                         "deleting them frees space without losing catalogue data")
    return "\n".join(lines)
