"""Build a standalone HTML report from a Psi0 inference-timing JSONL log.

Each run writes a NEW html file and appends one row to a cumulative history CSV,
so successive experiments can be compared on VLM vs action-expert cost.

    python3 scripts/make_timing_report.py --title "baseline ckpt40000"
    python3 scripts/make_timing_report.py --title "5 denoise steps" --logs path/to/run.jsonl
    python3 scripts/make_timing_report.py --title "fp8 vlm" --out-dir docs/timing_reports

With no --logs, the newest <run_dir>/inference_timing/*.jsonl under --search-root is used.
"""

import argparse
import csv
import glob
import json
import re
import os
import time
from pathlib import Path
from string import Template

STAGES = ("vlm_preprocess", "vlm_forward", "action_expert")
HISTORY_FIELDS = [
    "recorded_at", "title", "n_requests",
    "vlm_forward_ms", "action_expert_ms", "action_expert_per_step_ms",
    "vlm_preprocess_ms", "unmeasured_ms", "server_total_ms",
    "client_loop_gap_ms", "source",
]


# ---------------------------------------------------------------- data loading

def find_latest_log(search_root):
    pats = [
        os.path.join(search_root, "**", "inference_timing", "*.jsonl"),
        os.path.join(search_root, "inference_timing", "*.jsonl"),
    ]
    files = {f for p in pats for f in glob.glob(p, recursive=True)}
    if not files:
        raise SystemExit(
            f"no inference_timing/*.jsonl found under {search_root!r}\n"
            f"pass --logs explicitly, or --search-root <dir>"
        )
    return [max(files, key=os.path.getmtime)]


def load_rows(paths, keep_first):
    """Read rows, dropping each episode's first request unless keep_first."""
    rows = []
    for p in paths:
        seen = set()
        with open(p) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                row["_file"] = os.path.basename(p)
                first = row.get("episode") not in seen
                seen.add(row.get("episode"))
                if first and not keep_first:
                    continue
                rows.append(row)
    return rows


def mean(rows, key):
    vals = [r[key] for r in rows if isinstance(r.get(key), (int, float))]
    return sum(vals) / len(vals) if vals else None


def parse_meta(filename):
    """ckpt40000_Ta24_rtc_20260810-160026.jsonl -> dict of run parameters."""
    meta = {}
    m = re.search(r"ckpt([\w.]+?)_Ta(\d+)_(rtc|nortc)_(\d{8}-\d{6})", filename)
    if m:
        meta["ckpt"] = m.group(1)
        meta["Ta"] = m.group(2)
        meta["rtc"] = "on" if m.group(3) == "rtc" else "off"
        meta["logged"] = m.group(4)
    return meta


# ---------------------------------------------------------------- history file

def append_history(path, record):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not path.exists()
    with path.open("a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=HISTORY_FIELDS)
        if is_new:
            w.writeheader()
        w.writerow(record)


def read_history(path):
    path = Path(path)
    if not path.exists():
        return []
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))


# ---------------------------------------------------------------- rendering

def fmt(v, digits=2):
    return "—" if v is None else f"{v:,.{digits}f}"


def esc(s):
    return (str(s).replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;").replace('"', "&quot;"))


def history_rows_html(history, current_index):
    if not history:
        return '<tr><td colspan="7" class="dim">no runs recorded yet</td></tr>'
    out = []
    for i, h in enumerate(history):
        cls = ' class="current"' if i == current_index else ""
        def g(k, d=2):
            try:
                return f"{float(h.get(k, '')):,.{d}f}"
            except (TypeError, ValueError):
                return "—"
        out.append(
            f"<tr{cls}>"
            f'<td class="ttl">{esc(h.get("title", ""))}</td>'
            f'<td class="lead vlm">{g("vlm_forward_ms")}</td>'
            f'<td class="lead act">{g("action_expert_ms")}</td>'
            f'<td class="dim">{g("action_expert_per_step_ms")}</td>'
            f'<td>{g("server_total_ms")}</td>'
            f'<td class="dim">{g("n_requests", 0)}</td>'
            f'<td class="dim">{esc(h.get("recorded_at", "")[:16])}</td>'
            f"</tr>"
        )
    return "\n        ".join(out)


TEMPLATE = Template(r"""<title>$title_tag</title>
<style>
  :root {
    color-scheme: light;
    --page: #f1f1ee; --surface: #fbfbf9;
    --ink: #111112; --ink-2: #54544f; --muted: #8a8a83;
    --hair: #e2e2da; --rule: #c9c9c0;
    --s-pre: #2a78d6; --s-vlm: #eb6834; --s-act: #1baf7a; --s-other: #a9a9a1;
    --s-pre-soft: rgba(42,120,214,0.13);
    --s-vlm-soft: rgba(235,104,52,0.13);
    --s-act-soft: rgba(27,175,122,0.13);
    --s-other-soft: rgba(169,169,161,0.16);
    --mono: ui-monospace, SFMono-Regular, "SF Mono", Menlo, Consolas, monospace;
    --sans: system-ui, -apple-system, "Segoe UI", sans-serif;
  }
  @media (prefers-color-scheme: dark) {
    :root:not([data-theme="light"]) {
      color-scheme: dark;
      --page: #0b0b0c; --surface: #16161a;
      --ink: #f6f6f3; --ink-2: #b9b9b0; --muted: #8a8a83;
      --hair: #2a2a2e; --rule: #3a3a3f;
      --s-pre: #3987e5; --s-vlm: #d95926; --s-act: #199e70; --s-other: #6a6a64;
      --s-pre-soft: rgba(57,135,229,0.20);
      --s-vlm-soft: rgba(217,89,38,0.20);
      --s-act-soft: rgba(25,158,112,0.20);
      --s-other-soft: rgba(106,106,100,0.24);
    }
  }
  :root[data-theme="dark"] {
    color-scheme: dark;
    --page: #0b0b0c; --surface: #16161a;
    --ink: #f6f6f3; --ink-2: #b9b9b0; --muted: #8a8a83;
    --hair: #2a2a2e; --rule: #3a3a3f;
    --s-pre: #3987e5; --s-vlm: #d95926; --s-act: #199e70; --s-other: #6a6a64;
    --s-pre-soft: rgba(57,135,229,0.20);
    --s-vlm-soft: rgba(217,89,38,0.20);
    --s-act-soft: rgba(25,158,112,0.20);
    --s-other-soft: rgba(106,106,100,0.24);
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; background: var(--page); color: var(--ink);
    font-family: var(--sans); font-size: 15px; line-height: 1.55;
    -webkit-font-smoothing: antialiased;
  }
  .wrap { max-width: 60rem; margin: 0 auto; padding: 3rem 1.5rem 5rem; display: flex; flex-direction: column; gap: 2.75rem; }
  header { display: flex; flex-direction: column; gap: 1rem; }
  .eyebrow { font-family: var(--mono); font-size: 0.7rem; letter-spacing: 0.13em; text-transform: uppercase; color: var(--muted); }
  h1 { margin: 0; font-size: clamp(1.55rem, 3.4vw, 2.1rem); line-height: 1.15; font-weight: 620; letter-spacing: -0.015em; text-wrap: balance; }
  .runmeta { display: flex; flex-wrap: wrap; gap: 0 1.5rem; font-family: var(--mono); font-size: 0.76rem; color: var(--ink-2); padding-top: 0.9rem; border-top: 1px solid var(--hair); }
  .runmeta b { font-weight: 600; color: var(--ink); }
  section { display: flex; flex-direction: column; gap: 1.1rem; }
  h2 { margin: 0; font-size: 0.78rem; font-family: var(--mono); font-weight: 600; letter-spacing: 0.11em; text-transform: uppercase; color: var(--muted); padding-bottom: 0.6rem; border-bottom: 1px solid var(--hair); }
  p { margin: 0; color: var(--ink-2); max-width: 46rem; }
  p.tight { max-width: 52rem; }
  code { font-family: var(--mono); font-size: 0.86em; background: var(--s-other-soft); padding: 0.1em 0.34em; border-radius: 3px; color: var(--ink); }
  .card { background: var(--surface); border: 1px solid var(--hair); border-radius: 6px; padding: 1.35rem 1.5rem; }
  figure { margin: 0; display: flex; flex-direction: column; gap: 0.85rem; }
  .figbox { background: var(--surface); border: 1px solid var(--hair); border-radius: 6px; padding: 1.25rem 1rem; overflow-x: auto; }
  .figbox svg { display: block; width: 100%; min-width: 620px; height: auto; color: var(--ink); }
  figcaption { font-size: 0.85rem; color: var(--muted); max-width: 46rem; }
  .stackwrap { display: flex; flex-direction: column; gap: 0.5rem; }
  .stack { display: flex; height: 2.6rem; gap: 2px; }
  .seg { position: relative; border-radius: 2px; }
  .seg:first-child { border-radius: 4px 2px 2px 4px; }
  .seg:last-child { border-radius: 2px 4px 4px 2px; }
  .seg-label { position: absolute; left: 0.6rem; top: 50%; transform: translateY(-50%); font-family: var(--mono); font-size: 0.74rem; color: #fff; white-space: nowrap; overflow: hidden; max-width: calc(100% - 1.2rem); }
  .scale { display: flex; justify-content: space-between; font-family: var(--mono); font-size: 0.7rem; color: var(--muted); }
  .rows { display: flex; flex-direction: column; gap: 0.1rem; }
  .row { display: grid; grid-template-columns: 12.5rem 1fr 8.5rem; align-items: center; gap: 0.85rem; padding: 0.42rem 0; border-bottom: 1px solid var(--hair); }
  .row:last-child { border-bottom: none; }
  .row-label { display: flex; align-items: center; gap: 0.5rem; font-family: var(--mono); font-size: 0.79rem; color: var(--ink); }
  .swatch { width: 9px; height: 9px; border-radius: 2px; flex: none; }
  .track { height: 1.05rem; background: var(--s-other-soft); border-radius: 3px; overflow: hidden; }
  .fill { height: 100%; border-radius: 3px; }
  .row-val { font-family: var(--mono); font-size: 0.79rem; text-align: right; color: var(--ink); font-variant-numeric: tabular-nums; }
  .row-val span { color: var(--muted); }
  .tablewrap { overflow-x: auto; border: 1px solid var(--hair); border-radius: 6px; background: var(--surface); }
  table { border-collapse: collapse; width: 100%; font-size: 0.82rem; min-width: 40rem; }
  th, td { white-space: nowrap; text-align: right; padding: 0.6rem 0.9rem; border-bottom: 1px solid var(--hair); font-variant-numeric: tabular-nums; font-family: var(--mono); }
  th:first-child, td:first-child { text-align: left; }
  thead th { color: var(--muted); font-weight: 600; font-size: 0.72rem; letter-spacing: 0.06em; text-transform: uppercase; }
  thead th.lead { color: var(--ink); }
  tbody tr:last-child td { border-bottom: none; }
  td.dim { color: var(--muted); }
  td.ttl { font-family: var(--sans); color: var(--ink); white-space: normal; }
  td.lead { font-weight: 620; }
  td.vlm { color: var(--s-vlm); }
  td.act { color: var(--s-act); }
  tbody tr.current { background: var(--s-other-soft); }
  tbody tr.current td.ttl::after { content: " ← this run"; color: var(--muted); font-family: var(--mono); font-size: 0.72rem; font-weight: 400; }
  #tip { position: fixed; pointer-events: none; opacity: 0; transition: opacity 0.1s; background: var(--ink); color: var(--page); font-family: var(--mono); font-size: 0.74rem; padding: 0.35rem 0.55rem; border-radius: 4px; z-index: 10; white-space: nowrap; }
  @media (prefers-reduced-motion: reduce) { * { transition: none !important; } }
  @media (max-width: 34rem) { .row { grid-template-columns: 1fr; gap: 0.3rem; } .row-val { text-align: left; } }
</style>

<div class="wrap">

<header>
  <div class="eyebrow">Psi0 · inference latency</div>
  <h1>$title</h1>
  <div class="runmeta">
    <span><b>ckpt</b> $ckpt</span>
    <span><b>Ta</b> $ta</span>
    <span><b>RTC</b> $rtc</span>
    <span><b>denoise steps</b> $steps</span>
    <span><b>episodes</b> $episodes</span>
    <span><b>requests</b> $n</span>
    <span><b>log</b> $source</span>
  </div>
</header>

<section>
  <h2>The loop</h2>
  <figure>
    <div class="figbox">
      <svg viewBox="0 0 900 316" role="img"
           aria-label="A closed loop: the simulator spends $gap_ms milliseconds executing a $ta-action chunk, then posts one frame and a proprioception vector to the Psi0 server, which spends $pre ms preprocessing, $vlm ms on one VLM forward pass, and $act ms on $steps action-expert denoise steps, totalling $total ms, before returning $ta actions.">
        <defs>
          <marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
            <path d="M 0 1 L 9 5 L 0 9 z" fill="currentColor"/>
          </marker>
        </defs>
        <rect x="18" y="118" width="212" height="118" rx="5" stroke-width="1.5" style="fill: var(--s-other-soft); stroke: var(--rule);"/>
        <text x="124" y="146" text-anchor="middle" font-size="13" font-weight="600" style="font-family: var(--sans); fill: currentColor;">simulator + WBC</text>
        <text x="124" y="165" text-anchor="middle" font-size="11" style="font-family: var(--sans); fill: var(--ink-2);">client execution</text>
        <text x="124" y="188" text-anchor="middle" font-size="11" style="font-family: var(--mono); fill: var(--ink-2);">executes $ta actions</text>
        <text x="124" y="215" text-anchor="middle" font-size="17" font-weight="500" style="font-family: var(--mono); fill: currentColor;">$gap_ms ms</text>

        <line x1="234" y1="152" x2="392" y2="152" stroke="currentColor" stroke-width="1.5" marker-end="url(#ah)"/>
        <text x="313" y="142" text-anchor="middle" font-size="10.5" style="font-family: var(--mono); fill: var(--ink-2);">1 frame + proprio</text>
        <line x1="392" y1="202" x2="234" y2="202" stroke="currentColor" stroke-width="1.5" marker-end="url(#ah)"/>
        <text x="313" y="220" text-anchor="middle" font-size="10.5" style="font-family: var(--mono); fill: var(--ink-2);">$ta actions</text>

        <rect x="398" y="26" width="486" height="264" rx="5" fill="none" stroke-width="1.5" stroke-dasharray="5 4" style="stroke: var(--rule);"/>
        <text x="416" y="49" font-size="13" font-weight="600" style="font-family: var(--sans); fill: currentColor;">Psi0 policy server</text>
        <text x="866" y="49" text-anchor="end" font-size="13" font-weight="500" style="font-family: var(--mono); fill: currentColor;">$total ms</text>

        <rect x="418" y="64" width="446" height="48" rx="4" stroke-width="1.5" style="fill: var(--s-pre-soft); stroke: var(--s-pre);"/>
        <text x="436" y="85" font-size="11.5" font-weight="600" style="font-family: var(--mono); fill: currentColor;">vlm_preprocess</text>
        <text x="436" y="101" font-size="10.5" style="font-family: var(--sans); fill: var(--ink-2);">tokenize · patch images · copy to GPU</text>
        <text x="846" y="94" text-anchor="end" font-size="15" style="font-family: var(--mono); fill: currentColor;">$pre ms</text>

        <rect x="418" y="124" width="446" height="48" rx="4" stroke-width="1.5" style="fill: var(--s-vlm-soft); stroke: var(--s-vlm);"/>
        <text x="436" y="145" font-size="11.5" font-weight="600" style="font-family: var(--mono); fill: currentColor;">vlm_forward</text>
        <text x="436" y="161" font-size="10.5" style="font-family: var(--sans); fill: var(--ink-2);">vision-language backbone · one pass · ×1</text>
        <text x="846" y="154" text-anchor="end" font-size="15" style="font-family: var(--mono); fill: currentColor;">$vlm ms</text>

        <rect x="418" y="184" width="446" height="48" rx="4" stroke-width="1.5" style="fill: var(--s-act-soft); stroke: var(--s-act);"/>
        <text x="436" y="205" font-size="11.5" font-weight="600" style="font-family: var(--mono); fill: currentColor;">action_expert</text>
        <text x="436" y="221" font-size="10.5" style="font-family: var(--sans); fill: var(--ink-2);">flow-matching denoise · $per_step ms × $steps steps</text>
        <text x="846" y="214" text-anchor="end" font-size="15" style="font-family: var(--mono); fill: currentColor;">$act ms</text>

        <rect x="418" y="244" width="446" height="34" rx="4" fill="none" stroke-width="1.2" stroke-dasharray="4 3" style="stroke: var(--rule);"/>
        <text x="436" y="265" font-size="11" style="font-family: var(--mono); fill: var(--ink-2);">unmeasured glue · decode, normalize, D2H</text>
        <text x="846" y="265" text-anchor="end" font-size="12" style="font-family: var(--mono); fill: var(--ink-2);">$other ms</text>

        <line x1="641" y1="112" x2="641" y2="124" stroke="currentColor" stroke-width="1.5" marker-end="url(#ah)"/>
        <line x1="641" y1="172" x2="641" y2="184" stroke="currentColor" stroke-width="1.5" marker-end="url(#ah)"/>
      </svg>
    </div>
    <figcaption>One closed control cycle. The three stages run strictly in sequence inside a
    single request; the VLM runs once, the action expert $steps times. Values are means over
    $n requests ($episodes episodes), $exclusion_note.</figcaption>
  </figure>
</section>

<section>
  <h2>Composition of one request · $total ms</h2>

  <div class="card stackwrap">
    <div class="stack">
      <div class="seg" style="flex: $pre_f; background: var(--s-pre);" data-tip="vlm_preprocess · $pre ms · $pre_pct%"></div>
      <div class="seg" style="flex: $vlm_f; background: var(--s-vlm);" data-tip="vlm_forward · $vlm ms · $vlm_pct%"><span class="seg-label">vlm_forward · $vlm</span></div>
      <div class="seg" style="flex: $act_f; background: var(--s-act);" data-tip="action_expert · $act ms · $act_pct%"><span class="seg-label">action_expert · $act</span></div>
      <div class="seg" style="flex: $other_f; background: var(--s-other);" data-tip="unmeasured glue · $other ms · $other_pct%"></div>
    </div>
    <div class="scale"><span>0 ms</span><span>$total ms</span></div>
  </div>

  <div class="card rows">
    <div class="row">
      <div class="row-label"><span class="swatch" style="background: var(--s-act);"></span>action_expert</div>
      <div class="track"><div class="fill" style="width: $act_pct%; background: var(--s-act);" data-tip="$act ms · $steps denoise steps"></div></div>
      <div class="row-val">$act ms <span>· $act_pct%</span></div>
    </div>
    <div class="row">
      <div class="row-label"><span class="swatch" style="background: var(--s-vlm);"></span>vlm_forward</div>
      <div class="track"><div class="fill" style="width: $vlm_pct%; background: var(--s-vlm);" data-tip="$vlm ms · one forward pass"></div></div>
      <div class="row-val">$vlm ms <span>· $vlm_pct%</span></div>
    </div>
    <div class="row">
      <div class="row-label"><span class="swatch" style="background: var(--s-pre);"></span>vlm_preprocess</div>
      <div class="track"><div class="fill" style="width: $pre_pct%; background: var(--s-pre);" data-tip="$pre ms"></div></div>
      <div class="row-val">$pre ms <span>· $pre_pct%</span></div>
    </div>
    <div class="row">
      <div class="row-label"><span class="swatch" style="background: var(--s-other);"></span>unmeasured</div>
      <div class="track"><div class="fill" style="width: $other_pct%; background: var(--s-other);" data-tip="$other ms · deserialize, transforms, denormalize, D2H"></div></div>
      <div class="row-val">$other ms <span>· $other_pct%</span></div>
    </div>
    <div class="row">
      <div class="row-label" style="color: var(--muted);">one denoise step</div>
      <div class="track"><div class="fill" style="width: $per_step_pct%; background: var(--s-act); opacity: 0.45;" data-tip="$per_step ms — action_expert ÷ $steps (derived average)"></div></div>
      <div class="row-val" style="color: var(--muted);">$per_step ms <span>× $steps</span></div>
    </div>
  </div>
</section>

<section>
  <h2>Experiment history</h2>
  <div class="tablewrap">
    <table>
      <thead>
        <tr>
          <th>experiment</th>
          <th class="lead">vlm_forward</th>
          <th class="lead">action_expert</th>
          <th>per step</th>
          <th>server_total</th>
          <th>n</th>
          <th>recorded</th>
        </tr>
      </thead>
      <tbody>
        $history_rows
      </tbody>
    </table>
  </div>
  <p>Milliseconds, mean per request. One row per report run, appended to
  <code>$history_path</code>.</p>
</section>

</div>

<div id="tip" role="status" aria-live="off"></div>
<script>
  const tip = document.getElementById('tip');
  document.querySelectorAll('[data-tip]').forEach(el => {
    el.addEventListener('pointerenter', () => { tip.textContent = el.dataset.tip; tip.style.opacity = '1'; });
    el.addEventListener('pointermove', e => {
      const pad = 12; let x = e.clientX + pad, y = e.clientY + pad;
      const r = tip.getBoundingClientRect();
      if (x + r.width > window.innerWidth - 8) x = e.clientX - r.width - pad;
      if (y + r.height > window.innerHeight - 8) y = e.clientY - r.height - pad;
      tip.style.left = x + 'px'; tip.style.top = y + 'px';
    });
    el.addEventListener('pointerleave', () => { tip.style.opacity = '0'; });
  });
</script>
""")


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--title", default=None,
                    help="title of this experiment; shown as the page heading and in the "
                         "history table (default: the log filename)")
    ap.add_argument("--logs", nargs="*", default=None,
                    help="JSONL log file(s); default is the newest one found")
    ap.add_argument("--search-root", default=".runs",
                    help="where to look for inference_timing/*.jsonl (default: .runs)")
    ap.add_argument("--out-dir", default="docs/timing_reports",
                    help="directory for the generated html (default: docs/timing_reports)")
    ap.add_argument("--out", default=None,
                    help="exact output html path; overrides --out-dir")
    ap.add_argument("--history", default=None,
                    help="history CSV path (default: <out-dir>/history.csv)")
    ap.add_argument("--steps", type=int, default=10,
                    help="num_inference_steps used at serve time, for the per-step label "
                         "(default: 10)")
    ap.add_argument("--keep-first", action="store_true",
                    help="include each episode's first request (un-conditioned path)")
    ap.add_argument("--no-record", action="store_true",
                    help="render the html but do not append a row to the history CSV")
    args = ap.parse_args()

    paths = args.logs or find_latest_log(args.search_root)
    rows = load_rows(paths, args.keep_first)
    if not rows:
        raise SystemExit("no rows to summarize (all filtered out? try --keep-first)")

    source = ", ".join(os.path.basename(p) for p in paths)
    title = args.title or Path(paths[0]).stem
    meta = parse_meta(os.path.basename(paths[0]))

    pre = mean(rows, "vlm_preprocess") or 0.0
    vlm = mean(rows, "vlm_forward") or 0.0
    act = mean(rows, "action_expert") or 0.0
    total = mean(rows, "server_total") or (pre + vlm + act)
    other = max(total - (pre + vlm + act), 0.0)
    per_step = mean(rows, "action_expert_per_step") or (act / max(args.steps, 1))
    # Psi-R2 logs its actual velocity-evaluation count per request (bootstrap
    # and streaming steps differ), so prefer that over the --steps assumption.
    nfe = mean(rows, "nfe")
    steps = round(nfe, 1) if nfe else args.steps
    gap = mean(rows, "client_loop_gap_ms")
    episodes = len({r.get("episode") for r in rows})

    history_path = Path(args.history) if args.history else Path(args.out_dir) / "history.csv"
    record = {
        "recorded_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "title": title,
        "n_requests": len(rows),
        "vlm_forward_ms": round(vlm, 2),
        "action_expert_ms": round(act, 2),
        "action_expert_per_step_ms": round(per_step, 2),
        "vlm_preprocess_ms": round(pre, 2),
        "unmeasured_ms": round(other, 2),
        "server_total_ms": round(total, 2),
        "client_loop_gap_ms": "" if gap is None else round(gap, 2),
        "source": source,
    }
    if not args.no_record:
        append_history(history_path, record)

    history = read_history(history_path)
    if args.no_record:
        current_index = -1
    else:
        current_index = len(history) - 1

    pct = lambda v: round(100.0 * v / total, 1) if total else 0.0
    html = TEMPLATE.substitute(
        title=esc(title),
        title_tag=esc(title),
        ckpt=esc(meta.get("ckpt", "—")),
        ta=esc(meta.get("Ta", "—")),
        rtc=esc(meta.get("rtc", "—")),
        steps=steps,
        episodes=episodes,
        n=len(rows),
        source=esc(source),
        exclusion_note=("including each episode's first request"
                        if args.keep_first else
                        "excluding each episode's first request"),
        pre=fmt(pre), vlm=fmt(vlm), act=fmt(act), other=fmt(other),
        total=fmt(total), per_step=fmt(per_step),
        gap_ms=fmt(gap, 0),
        pre_f=f"{pre:.4f}", vlm_f=f"{vlm:.4f}", act_f=f"{act:.4f}", other_f=f"{other:.4f}",
        pre_pct=pct(pre), vlm_pct=pct(vlm), act_pct=pct(act), other_pct=pct(other),
        per_step_pct=pct(per_step),
        history_rows=history_rows_html(history, current_index),
        history_path=esc(str(history_path)),
    )

    if args.out:
        out = Path(args.out)
    else:
        slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-") or "report"
        out = Path(args.out_dir) / f"{slug}_{time.strftime('%Y%m%d-%H%M%S')}.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html)

    print(f"title           {title}")
    print(f"source          {source}")
    print(f"requests        {len(rows)} over {episodes} episodes")
    print(f"vlm_forward     {fmt(vlm)} ms   ({pct(vlm)}%)")
    print(f"action_expert   {fmt(act)} ms   ({pct(act)}%)  = {fmt(per_step)} ms x {steps}")
    print(f"vlm_preprocess  {fmt(pre)} ms   ({pct(pre)}%)")
    print(f"unmeasured      {fmt(other)} ms   ({pct(other)}%)")
    print(f"server_total    {fmt(total)} ms")
    print(f"client gap      {fmt(gap, 0)} ms")
    print()
    print(f"html     -> {out}")
    print(f"history  -> {history_path}" + ("  (not appended: --no-record)" if args.no_record else f"  ({len(history)} runs)"))


if __name__ == "__main__":
    main()
