"""Figures for the start-pose sweep (stage 3). Runs in Psi0's venv, CPU only.

Reads what eval_start_pose.py wrote (results.jsonl, traces/, videos/, conditions.json) and writes, into figures/:
  1_start_map   top-down: every start pose that was tried, colored by its success rate (k/n next to it).
                Left: start positions over the floor, with the box, the table and the demos' approach.
                Right: start headings at the recorded position.
  2_paths       top-down pelvis path of every run, one panel per start pose: where the robot went, and where a
                failed run ended (stopped short of the box, or carried it and dropped it)
  index.html    the table (success, how far each run got) and every run's head-camera video
  summary.json
"""

from __future__ import annotations

import argparse
import glob
import html
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Rectangle

SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
C_MAIN, C_FAIL, C_DEMO, TABLE = "#2a78d6", "#b5b3ab", "#d9d7cf", "#ecebe6"
SEQ = LinearSegmentedColormap.from_list(
    "seq_blue", ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"])
OUTCOMES = {"success": "placed on the table", "not_placed": "lifted, not placed", "not_lifted": "reached, not lifted",
            "not_reached": "never reached the box", "fell": "fell", "error": "eval error"}
CONTROL_HZ = 50

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "sans-serif", "font.size": 9, "axes.titlesize": 9.5, "axes.titleweight": "medium",
    "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.labelcolor": INK2, "axes.titlecolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 2.0, "lines.markersize": 4,
    "legend.frameon": False, "legend.fontsize": 8, "text.color": INK,
})


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results", required=True, help="the sweep's output directory")
    p.add_argument("--reach-xy", type=float, default=None,
                   help="re-grade with this 'reached the box' pelvis-box distance (m) instead of the sweep's")
    p.add_argument("--fall-z", type=float, default=None, help="re-grade with this fall height (m)")
    return p.parse_args()


def load_trace(root: Path, label: str) -> np.ndarray:
    """Columns: step, x, y, z, yaw, box_x, box_y, box_z, reward."""
    tf = root / "traces" / f"{label}.json"
    return np.array(json.loads(tf.read_text())["rows"], dtype=float).reshape(-1, 9) if tf.exists() else np.zeros((0, 9))


def regrade(r: dict, t: np.ndarray, th: dict) -> None:
    """Outcome from the stored measurements and the thresholds in use (so they can change after a sweep)."""
    d = np.linalg.norm(t[:, 1:3] - t[:, 5:7], axis=1)
    hit = np.flatnonzero(d < th["reach_xy"])
    r["reached"] = r["min_pelvis_box_xy"] is not None and r["min_pelvis_box_xy"] < th["reach_xy"]
    r["reach_step"] = int(t[hit[0], 0]) if r["reached"] and len(hit) else None
    r["fell"] = r["min_pelvis_z"] is not None and r["min_pelvis_z"] < th["fall_z"]
    if r["task_success"]:
        r["outcome"] = "success"
    elif r["termination_reason"] in ("model_error", "lockstep_timeout"):
        r["outcome"] = "error"
    else:
        r["outcome"] = ("fell" if r["fell"] else "not_placed" if r["lifted"] else
                        "not_lifted" if r["reached"] else "not_reached")


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return np.nan, np.nan
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, c - h), min(1.0, c + h)


def demo_paths(data_dir: str) -> list[np.ndarray]:
    """Recorded pelvis xy until the box leaves the floor (the demos' approach)."""
    paths = []
    for f in sorted(glob.glob(f"{data_dir}/data/*/*.parquet")):
        df = pd.read_parquet(f, columns=["observation.base_pose", "observation.object_poses"])
        bp, ob = np.stack(df["observation.base_pose"]), np.stack(df["observation.object_poses"])
        lift = int(np.argmax(ob[:, 2] > ob[0, 2] + 0.05)) or len(df)
        paths.append(bp[:lift + 1, :2])
    return paths


def draw_scene(ax, table, box, demos) -> None:
    """Top-down, the robot looking up the page: world x up, world y (left) to the left."""
    if table:
        (tx, ty), (sx, sy) = table["position"][:2], table["size"][:2]
        ax.add_patch(Rectangle((ty - sy / 2, tx - sx / 2), sy, sx, fc=TABLE, ec=AXIS, lw=0.8, zorder=0))
        ax.text(ty, tx - sx / 2 + 0.04, "table", ha="center", va="bottom", fontsize=7, color=MUTED)
    ax.add_patch(Rectangle((box[1] - 0.19, box[0] - 0.115), 0.38, 0.23, fc=SURFACE, ec=INK, lw=1.2, zorder=4))
    ax.text(box[1], box[0], "box", ha="center", va="center", fontsize=7, color=INK2, zorder=5)
    for p in demos:
        ax.plot(p[:, 1], p[:, 0], color=C_DEMO, lw=1.0, zorder=1)


def main() -> None:
    args = parse_args()
    root = Path(args.results)
    out = root / "figures"
    out.mkdir(exist_ok=True)
    for old in ("1_success_vs_start.png", "2_outcomes.png", "3_paths.png", "4_floor_map.png"):  # earlier layout
        (out / old).unlink(missing_ok=True)
    rows = [json.loads(line) for line in (root / "results.jsonl").read_text().splitlines() if line.strip()]
    if not rows:
        raise SystemExit(f"no finished runs in {root / 'results.jsonl'}")
    meta = json.loads((root / "conditions.json").read_text())
    th = meta["thresholds"]
    th.update({k: v for k, v in (("reach_xy", args.reach_xy), ("fall_z", args.fall_z)) if v is not None})
    traces = {r["label"]: load_trace(root, r["label"]) for r in rows}
    for r in rows:
        regrade(r, traces[r["label"]], th)
    df = pd.DataFrame(rows)
    order = [c["cond"] for c in meta["offsets"] if c["cond"] in set(df["cond"])]
    order += [c for c in dict.fromkeys(df["cond"]) if c not in order]  # runs from an earlier offset list
    starts = meta["recorded_start"].values()
    box = np.mean([s["box"] for s in starts], axis=0)
    start = np.mean([s["base_pose"][:2] for s in starts], axis=0)
    table = meta.get("table")
    demos = demo_paths(meta["args"]["data_dir"])

    summary = []
    for c in order:
        g = df[df["cond"] == c]
        k, n = int(g["task_success"].sum()), len(g)
        dx, dy, dyaw = (float(g[a].iloc[0]) for a in ("dx", "dy", "dyaw"))
        summary.append(dict(
            cond=c, dx=dx, dy=dy, dyaw=dyaw, n=n, successes=k, success_rate=k / n, ci95=list(wilson(k, n)),
            reached=int(g["reached"].sum()), lifted=int(g["lifted"].sum()),
            outcomes={o: int((g["outcome"] == o).sum()) for o in OUTCOMES if (g["outcome"] == o).any()},
            median_reach_s=None if g["reach_step"].isna().all() else float(g["reach_step"].median() / CONTROL_HZ),
            median_closest_m=float(g["min_pelvis_box_xy"].median()),
            start_pelvis_box_m=float(np.linalg.norm(start + [dx, dy] - box[:2])),
            start_contacts=sorted({b for bs in g["start_contacts"] for b in bs})))
    S = pd.DataFrame(summary).set_index("cond")
    norm = Normalize(0, 100)

    # ---------------------------------------------------------------- 1: start map
    turned = [c for c in order if S.loc[c, "dyaw"] and not S.loc[c, "dx"] and not S.loc[c, "dy"]]
    placed = [c for c in order if c not in turned]
    recorded = [c for c in order if not (S.loc[c, "dx"] or S.loc[c, "dy"] or S.loc[c, "dyaw"])]
    fig = plt.figure(figsize=(11.5, 7.6))
    ax = fig.add_axes((0.06, 0.08, 0.50, 0.80))
    draw_scene(ax, table, box, demos)
    for c in placed:
        r = S.loc[c]
        x, y = start[0] + r["dx"], start[1] + r["dy"]
        if r["dyaw"]:
            h = np.radians(r["dyaw"])
            ax.plot([y, y + 0.07 * np.sin(h)], [x, x + 0.07 * np.cos(h)], color=INK, lw=1.2, zorder=6)
        ax.scatter([y], [x], s=230, color=SEQ(norm(100 * r["success_rate"])), edgecolors=INK, linewidths=1.0,
                   zorder=7)
        ax.annotate(f"{r['successes']}/{r['n']}", (y, x), textcoords="offset points", xytext=(15, -1),
                    va="center", fontsize=8, color=INK, zorder=8)
    ax.scatter([start[1]], [start[0]], s=520, facecolors="none", edgecolors=INK, linewidths=1.0, zorder=7)
    pts = np.array([[start[1] + S.loc[c, "dy"], start[0] + S.loc[c, "dx"]] for c in placed] + [[start[1], start[0]]])
    ax.set_xlim(max(0.45, pts[:, 0].max() + 0.15), min(-0.45, pts[:, 0].min() - 0.15))
    ax.set_ylim(min(-1.5, pts[:, 1].min() - 0.12), max(0.05, box[0] + 0.5))
    ax.set_aspect("equal")
    ax.set_xlabel("world y (m), left is left")
    ax.set_ylabel("world x (m), forward is up")
    ax.set_title("Start position (heading as recorded, or as the tick shows)", loc="left")

    if turned:
        ax2 = fig.add_axes((0.60, 0.30, 0.30, 0.50))
        ax2.set_aspect("equal")
        ax2.axis("off")
        bearing = np.degrees(np.arctan2(box[1] - start[1], box[0] - start[0]))  # of the box, + = to the left
        for c in sorted(turned + recorded, key=lambda c: S.loc[c, "dyaw"]):
            r = S.loc[c]
            h = np.radians(r["dyaw"])  # + = turned left, which is drawn to the left
            ax2.annotate("", xy=(-np.sin(h), np.cos(h)), xytext=(0, 0), zorder=3,
                         arrowprops=dict(arrowstyle="-|>,head_width=0.35,head_length=0.7", lw=3.0, shrinkA=0,
                                         shrinkB=0, color=SEQ(norm(100 * r["success_rate"]))))
            ax2.text(-1.3 * np.sin(h), 1.3 * np.cos(h), (f"{r['dyaw']:+.0f}°" if r["dyaw"] else "0°") + f"\n{r['successes']}/{r['n']}",
                     ha="center", va="center", fontsize=8, color=INK)
        ax2.scatter([0], [0], s=120, color=INK, zorder=5)
        ax2.set_xlim(-1.45, 1.45)
        ax2.set_ylim(-0.25, 1.55)
        ax2.set_title(f"Start heading at the recorded position (+ = turned left;\nthe box is {abs(bearing):.0f}° "
                      f"{'left' if bearing > 0 else 'right'} of the recorded heading)", loc="left")
    cax = fig.add_axes((0.62, 0.17, 0.26, 0.025))
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=SEQ), cax=cax, orientation="horizontal")
    cb.set_label("success rate (%): box placed on the table")
    cb.outline.set_visible(False)
    fig.text(0.06, 0.955, f"Where the robot can start and still finish the task  ({len(df)} closed-loop runs, "
             f"{df.groupby('cond').size().max()} per start pose)", fontsize=11, ha="left")
    fig.text(0.06, 0.925, "Circle = a start position (k/n = successes / runs); double ring = the recorded start, where "
             "every demo starts; light lines = the demos' walk to the box.", fontsize=8.5, color=INK2, ha="left")
    fig.savefig(out / "1_start_map.png", dpi=140)
    plt.close(fig)

    # ---------------------------------------------------------------- 2: paths
    ncol = min(4, len(order))
    nrow = int(np.ceil(len(order) / ncol))
    fig, axs = plt.subplots(nrow, ncol, figsize=(max(3.0 * ncol, 9.5), 3.3 * nrow + 0.4), squeeze=False)
    for ax, c in zip(axs.flat, order):
        draw_scene(ax, table, box, demos)
        for _, r in df[df["cond"] == c].iterrows():
            t = traces[r["label"]]
            if not len(t):
                continue
            col = C_MAIN if r["task_success"] else C_FAIL
            ax.plot(t[:, 2], t[:, 1], color=col, lw=1.3, zorder=3 if r["task_success"] else 2)
            ax.plot(t[-1, 2], t[-1, 1], "o" if r["task_success"] else "x", color=col, ms=5, mew=1.5, zorder=6)
        s0 = start + [S.loc[c, "dx"], S.loc[c, "dy"]]
        h = np.radians(S.loc[c, "dyaw"])
        ax.annotate("", xy=(s0[1] + 0.16 * np.sin(h), s0[0] + 0.16 * np.cos(h)), xytext=(s0[1], s0[0]),
                    arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.2), zorder=7)
        ax.set_xlim(0.62, -0.62)
        ax.set_ylim(-1.55, 0.15)
        ax.set_aspect("equal")
        ax.set_title(f"{c}   {S.loc[c, 'successes']}/{S.loc[c, 'n']}", loc="left", fontsize=8.5)
        ax.tick_params(labelsize=7)
    for ax in axs.flat[len(order):]:
        ax.set_visible(False)
    for ax in axs[:, 0]:
        ax.set_ylabel("world x (m)")
    for ax in axs[-1, :]:
        ax.set_xlabel("world y (m), left is left")
    fig.suptitle("Pelvis path of every run: blue = success, gray = failed, x = where a failed run ended.\n"
                 "Arrow = start pose, light lines = the demos' approach", x=0.02, ha="left", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.75 / fig.get_figheight()))
    fig.savefig(out / "2_paths.png", dpi=140)
    plt.close(fig)

    # ---------------------------------------------------------------- summary + index.html
    at_start, off = df[df["cond"].isin(recorded)], df[~df["cond"].isin(recorded)]
    head = dict(runs=len(df), checkpoint=meta.get("server_info", {}).get("run_dir"),
                ckpt_step=meta.get("server_info", {}).get("ckpt_step"), thresholds=th,
                at_recorded_start=dict(n=len(at_start), successes=int(at_start["task_success"].sum())),
                off_start=dict(n=len(off), successes=int(off["task_success"].sum())))
    (out / "summary.json").write_text(json.dumps(dict(**head, conditions=summary), indent=2))

    def pct(k, n):
        return f"{100 * k / n:.0f}% ({k}/{n})" if n else "-"

    def num(v, f=".1f"):
        return "-" if v is None or not np.isfinite(v) else format(v, f)

    def video(r) -> str | None:
        vids = sorted(glob.glob(str(root / (r["video_dir"] or "-") / "head_stereo_left*.mp4")))
        return f"../{Path(vids[0]).relative_to(root)}" if vids else None

    trs, gallery = "", ""
    for c, r in S.iterrows():
        cells = [html.escape(c), f"{r['dx']:+.2f}", f"{r['dy']:+.2f}", f"{r['dyaw']:+.0f}",
                 num(r["start_pelvis_box_m"], ".2f"), f"<b>{pct(r['successes'], r['n'])}</b>",
                 f"{100 * r['ci95'][0]:.0f}-{100 * r['ci95'][1]:.0f}%", f"{r['reached']}/{r['n']}",
                 f"{r['lifted']}/{r['n']}", ", ".join(f"{OUTCOMES[k]} {v}" for k, v in r["outcomes"].items()),
                 num(r["median_reach_s"]), num(r["median_closest_m"], ".2f"),
                 html.escape(", ".join(r["start_contacts"])) or "-"]
        trs += "<tr>" + "".join(f"<td>{x}</td>" for x in cells) + "</tr>"
        clips = ""
        for _, run in df[df["cond"] == c].sort_values(["episode", "rep"]).iterrows():
            name = f"scene {run['episode']}" + (f" r{run['rep']}" if run["rep"] else "")
            src = video(run)
            player = (f"<video src='{html.escape(src)}' controls preload=none muted playsinline></video>" if src
                      else "<div class=novid>no video</div>")
            clips += (f"<figure class={'ok' if run['task_success'] else 'fail'}>{player}<figcaption><b>{name}</b>"
                      f" &middot; {html.escape(OUTCOMES[run['outcome']])}</figcaption></figure>")
        gallery += f"<h3>{html.escape(c)} &middot; {pct(r['successes'], r['n'])}</h3><div class=clips>{clips}</div>"
    si = meta.get("server_info", {})
    (out / "index.html").write_text(
        "<!doctype html><meta charset=utf-8><meta name=viewport content='width=device-width,initial-scale=1'>"
        "<title>Start-pose sweep</title>"
        "<style>:root{--bg:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--rule:#e1e0d9;--pre:#f4f3ef;--ok:#2a78d6;--fail:#b5b3ab}"
        "@media (prefers-color-scheme: dark){:root{--bg:#1a1a19;--ink:#fff;--ink2:#c3c2b7;--rule:#383835;--pre:#2c2c2a}}"
        "body{font-family:system-ui,sans-serif;background:var(--bg);color:var(--ink);margin:24px;max-width:1500px}"
        "img{width:100%;border:1px solid var(--rule);margin:8px 0 28px;background:#fcfcfb}"
        "table{border-collapse:collapse;font-size:13px;font-variant-numeric:tabular-nums}"
        "td,th{padding:4px 10px;border-bottom:1px solid var(--rule);text-align:left;vertical-align:top}"
        "th{color:var(--ink2);font-weight:500}.wrap{overflow-x:auto}"
        ".clips{display:flex;flex-wrap:wrap;gap:10px}figure{margin:0;width:300px;max-width:100%}"
        "video,.novid{width:100%;aspect-ratio:16/9;background:#000;border-top:3px solid var(--fail)}"
        "figure.ok video{border-top-color:var(--ok)}.novid{color:#fff;display:grid;place-items:center}"
        "figcaption{font-size:12px;color:var(--ink2);padding:2px 0}h3{margin:22px 0 6px;font-weight:500}"
        "pre{background:var(--pre);padding:12px;overflow:auto}</style>"
        "<h1>Start-pose sweep</h1>"
        f"<p>Closed-loop runs of <code>{html.escape(str(si.get('run_dir', '?')))}</code> "
        f"(step {html.escape(str(si.get('ckpt_step', '?')))}) on <code>{html.escape(meta['args']['env_id'])}</code>, "
        "started away from the recorded start. "
        f"At the recorded start: <b>{pct(head['at_recorded_start']['successes'], head['at_recorded_start']['n'])}</b>; "
        f"away from it: <b>{pct(head['off_start']['successes'], head['off_start']['n'])}</b>.</p>"
        "<p>dx forward (toward the box), dy left, dyaw turn left, from the recorded start. Reached = pelvis within "
        f"{th['reach_xy']:.2f} m of the box; lifted = box raised {th['lift_dz'] * 100:.0f} cm; success = box resting on "
        "the table with the hands off it.</p>"
        "<img src='1_start_map.png'>"
        "<div class=wrap><table><tr><th>start pose</th><th>dx m</th><th>dy m</th><th>dyaw &deg;</th>"
        "<th>start pelvis-box m</th><th>success</th><th>95% CI</th><th>reached box</th><th>lifted box</th>"
        "<th>outcomes</th><th>median s to reach</th><th>median closest m</th><th>start contacts</th></tr>"
        f"{trs}</table></div>"
        "<h2>Paths</h2><img src='2_paths.png'>"
        "<h2>Videos (head camera, what the policy sees; blue bar = success)</h2>"
        f"{gallery}"
        f"<h2>summary</h2><pre>{html.escape(json.dumps(head, indent=2))}</pre>")
    print(f"wrote 1_start_map.png, 2_paths.png, index.html to {out}")
    for c, r in S.iterrows():
        print(f"  {c:<22} {pct(r['successes'], r['n']):>12}   reached {r['reached']}/{r['n']}  "
              f"lifted {r['lifted']}/{r['n']}   {r['outcomes']}")


if __name__ == "__main__":
    main()
