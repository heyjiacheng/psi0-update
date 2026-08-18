"""Summarize inference-timing JSONL logs written by the Psi0 policy server.

    python scripts/analyze_inference_timing.py <run_dir>/inference_timing/*.jsonl
    python scripts/analyze_inference_timing.py <run_dir>/inference_timing/*.jsonl --by episode
    python scripts/analyze_inference_timing.py <run_dir>/inference_timing/*.jsonl --by file --csv out.csv

By default the first request of every episode is dropped: it runs the
un-conditioned (non-RTC) path and, for the first episode, absorbs CUDA warmup.
Pass --keep-first to include it.
"""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

SKIP = {"idx", "episode", "episode_start", "path", "episode_id", "cache_id", "nfe"}


def load(paths, keep_first):
    rows = []
    for p in paths:
        seen = set()
        for line in Path(p).read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            row["file"] = Path(p).name
            ep = row.get("episode")
            first = ep not in seen
            seen.add(ep)
            if first and not keep_first:
                continue
            rows.append(row)
    return rows


def stages(rows):
    names, seen = [], set()
    for row in rows:
        for k, v in row.items():
            if (k not in seen and k not in SKIP
                    and isinstance(v, (int, float)) and not isinstance(v, bool)):
                seen.add(k)
                names.append(k)
    return names


def stats(vals):
    vals = sorted(vals)
    n = len(vals)
    return {
        "n": n,
        "mean": sum(vals) / n,
        "p50": vals[n // 2],
        "p95": vals[min(n - 1, int(0.95 * n))],
        "max": vals[-1],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="+", help="JSONL files from <run_dir>/inference_timing/")
    ap.add_argument("--by", choices=["none", "episode", "file", "path"], default="none",
                    help="group rows before summarizing")
    ap.add_argument("--keep-first", action="store_true",
                    help="include each episode's first (un-conditioned) request")
    ap.add_argument("--csv", help="also write one row per group/stage to this CSV")
    args = ap.parse_args()

    rows = load(args.logs, args.keep_first)
    if not rows:
        raise SystemExit("no rows found (all filtered out? try --keep-first)")

    groups = {"all": rows} if args.by == "none" else None
    if groups is None:
        g = defaultdict(list)
        for row in rows:
            g[row.get(args.by)].append(row)
        groups = dict(sorted(g.items(), key=lambda kv: str(kv[0])))

    names = stages(rows)
    out = []
    for key, grp in groups.items():
        print(f"\n=== {args.by}={key} ({len(grp)} requests) ===")
        for name in names:
            vals = [r[name] for r in grp if name in r]
            if not vals:
                continue
            s = stats(vals)
            print(f"  {name:<24} n={s['n']:<5} mean={s['mean']:7.2f}ms  p50={s['p50']:7.2f}ms  "
                  f"p95={s['p95']:7.2f}ms  max={s['max']:7.2f}ms")
            out.append({"group": key, "stage": name, **s})

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["group", "stage", "n", "mean", "p50", "p95", "max"])
            w.writeheader()
            w.writerows(out)
        print(f"\nwrote {args.csv}")


if __name__ == "__main__":
    main()
