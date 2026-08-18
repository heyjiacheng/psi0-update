"""Lightweight inference timers for the policy server.

Times are wall-clock milliseconds around CUDA-synchronized regions, so the
numbers attribute the asynchronous GPU work to the block that launched it.

Every served request appends one JSONL row to the log file (see
``open_log``), tagged with the episode index so a multi-episode eval session
can be broken down afterwards. Summaries are printed periodically and on
shutdown.

Enabled by default; set ``PSI_PROFILE=0`` to make every timer a no-op (removes
the ``cuda.synchronize()`` calls from the hot path and writes no log).
"""

import atexit
import json
import os
import time
from collections import OrderedDict, defaultdict
from contextlib import contextmanager
from pathlib import Path

import torch

ENABLED = os.environ.get("PSI_PROFILE", "1") not in ("0", "false", "False")

# Stage timings of the request currently being served, in insertion order.
_current: "OrderedDict[str, float]" = OrderedDict()
# Non-numeric context for the current request (episode index, code path, ...).
_tags: "OrderedDict[str, object]" = OrderedDict()
# Every completed request, as a full row. The summary is computed from these.
_rows: "list[dict]" = []

_log_path: "Path | None" = None
_log_file = None
_episode = 0


def _sync(device) -> None:
    if device is not None and torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


def sync(device) -> None:
    """Public CUDA barrier for callers that time a block by hand.

    Needed by the Psi-R2 slow channel: it runs on a worker thread, so it must
    measure into its own locals instead of the module-global request row.
    """
    if ENABLED:
        _sync(device)


@contextmanager
def timed(name: str, device=None):
    """Record the wall time of a block under ``name``."""
    if not ENABLED:
        yield
        return
    _sync(device)
    t0 = time.perf_counter()
    try:
        yield
    finally:
        _sync(device)
        _current[name] = (time.perf_counter() - t0) * 1e3


def record(name: str, ms: float) -> None:
    if ENABLED:
        _current[name] = ms


def tag(**kwargs) -> None:
    """Attach non-timing context (episode, code path, ...) to this request."""
    if ENABLED:
        _tags.update(kwargs)


def get(name: str, default: float = 0.0) -> float:
    """Timing recorded for ``name`` in the request currently being served."""
    return _current.get(name, default)


def reset() -> None:
    _current.clear()
    _tags.clear()


def new_episode() -> int:
    """Bump the episode counter; call when the client signals a reset."""
    global _episode
    _episode += 1
    return _episode


def episode() -> int:
    return _episode


def open_log(path) -> "Path | None":
    """Start appending one JSONL row per request to ``path``."""
    global _log_path, _log_file
    if not ENABLED:
        return None
    _log_path = Path(path)
    _log_path.parent.mkdir(parents=True, exist_ok=True)
    _log_file = _log_path.open("a", buffering=1)  # line-buffered: survives SIGKILL
    return _log_path


def log_path() -> "Path | None":
    return _log_path


def flush() -> "OrderedDict[str, float]":
    """Commit the current request: append it to the history and the log file."""
    timings = OrderedDict(_current)
    if ENABLED:
        row = {"idx": len(_rows), "episode": _episode, **_tags, **timings}
        _rows.append(row)
        if _log_file is not None:
            _log_file.write(json.dumps(row) + "\n")
    reset()
    return timings


def format_last(timings: "OrderedDict[str, float]") -> str:
    return "  ".join(f"{name}={ms:7.2f}ms" for name, ms in timings.items())


def _stats(values):
    vals = sorted(values)
    n = len(vals)
    return {
        "n": n,
        "mean": sum(vals) / n,
        "p50": vals[n // 2],
        "p95": vals[min(n - 1, int(0.95 * n))],
        "max": vals[-1],
    }


_NON_STAGE_KEYS = (
    "idx", "episode", "episode_start", "path",
    # Psi-R2 identifiers: numeric, but counters rather than measurements.
    "episode_id", "cache_id", "nfe",
)


def _stage_names(rows) -> "list[str]":
    """Numeric measurement columns, in first-seen order (bools are tags, not stages)."""
    names, seen = [], set()
    for row in rows:
        for k, v in row.items():
            if (k not in seen and k not in _NON_STAGE_KEYS
                    and isinstance(v, (int, float)) and not isinstance(v, bool)):
                seen.add(k)
                names.append(k)
    return names


def format_summary(rows=None, drop_first_per_episode: bool = True) -> str:
    """Mean / p50 / p95 / max per stage.

    The first request of each episode runs the un-conditioned (non-RTC) path
    and, for episode 1, absorbs CUDA warmup, so it is excluded by default.
    """
    rows = _rows if rows is None else rows
    if not rows:
        return "  (no requests served)"
    if drop_first_per_episode:
        seen_eps = set()
        kept = []
        for row in rows:
            ep = row.get("episode")
            if ep in seen_eps:
                kept.append(row)
            seen_eps.add(ep)
        rows = kept or rows
    lines = []
    for name in _stage_names(rows):
        vals = [row[name] for row in rows if name in row]
        if not vals:
            continue
        s = _stats(vals)
        lines.append(
            f"  {name:<24} n={s['n']:<5} mean={s['mean']:7.2f}ms  p50={s['p50']:7.2f}ms  "
            f"p95={s['p95']:7.2f}ms  max={s['max']:7.2f}ms"
        )
    return "\n".join(lines)


def format_per_episode() -> str:
    """One line per episode: request count and mean of each stage."""
    if not _rows:
        return "  (no requests served)"
    by_ep = defaultdict(list)
    for row in _rows:
        by_ep[row.get("episode", 0)].append(row)
    names = _stage_names(_rows)
    header = "  episode  n     " + "".join(f"{n[:16]:>18}" for n in names)
    lines = [header]
    for ep in sorted(by_ep):
        rows = by_ep[ep][1:] or by_ep[ep]  # drop that episode's un-conditioned first step
        cells = ""
        for name in names:
            vals = [r[name] for r in rows if name in r]
            cells += f"{(sum(vals) / len(vals)):17.2f}ms" if vals else f"{'-':>18}"
        lines.append(f"  {ep:<9}{len(rows):<6}{cells}")
    return "\n".join(lines)


def full_report() -> str:
    return (
        f"per-episode means (first step of each episode excluded):\n{format_per_episode()}\n"
        f"overall:\n{format_summary()}"
        + (f"\n  raw rows: {_log_path}" if _log_path else "")
    )


def num_requests() -> int:
    return len(_rows)


def _close():
    if _log_file is not None:
        _log_file.close()


atexit.register(_close)
