"""Figures for the target-pose sweep (stage 3). Runs in Psi0's venv, CPU only.

Reads what prepare_target_sweep.py and eval_target_pose.py wrote (train_targets.json, plan.json, results.jsonl,
traces/, videos/) and writes

report/ (self-contained: open report/index.html, or share the folder; kept when the rest of the sweep is deleted)
  index.html    report_template.html filled in: success rates, the top-down target map (grid sampling: one cell per
                grid point, coloured by its success rate; click a cell or a trial to play its head-camera videos),
                hand vs target, the trial / grid-point list, progress while the sweep runs
  results.json  the data: settings, training and trial range, grid, every finished trial's record (results.jsonl
                plus where it sits against the training range), per-grid-point and overall success counts
  videos/       one head-camera clip per trial, re-encoded small for the browser

figures/
  1_target_map  top-down, in the robot's start frame: every trial's target position, success or failure, with the
                training demos' target positions and their range, the range the trials were drawn from, and a thin
                line from each target to the closest point the robot's hands got to
  2_tracking    does the hand go where the target is? Per axis (sideways, forward): where a hand was when it came
                nearest the target, against where the target was. On the diagonal = it follows the target; flat at
                the training range = it goes where the demos' targets were, whatever this one's position

Safe to run while the sweep is still going (the runner does, every REPORT_EVERY s): a half-written last line of
results.jsonl is skipped, and index.html / results.json are replaced in one step.
"""

from __future__ import annotations

import argparse
import glob
import html
import json
import math
import re
import shutil
import subprocess
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon, Rectangle

SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
C_OK, C_FAIL, C_DEMO, C_TRAIN, TABLE = "#2a78d6", "#d03b3b", "#9a988f", "#cde2fb", "#ecebe6"
RATE_CMAP = LinearSegmentedColormap.from_list("rate", [C_FAIL, "#f0efec", C_OK])  # diverging, grey at 50%
TEMPLATE = Path(__file__).with_name("report_template.html")
OUTCOMES = {"success": "success", "moved_not_done": "moved it, not done", "reached_not_moved": "reached, not moved",
            "not_reached": "never reached it", "fell": "fell"}

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
    p.add_argument("--merge", action="append", default=[],
                   help="pool in the trials of another sweep of the same task and checkpoint (repeatable); its trials "
                        "outside this sweep's trial range are left out")
    p.add_argument("--out", default="", help="where to write figures/ and report/ (default: <results>)")
    return p.parse_args()


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return np.nan, np.nan
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, c - h), min(1.0, c + h)


def box_dist(box: dict, dx: float, dy: float) -> float:
    ex = max(box["dx"][0] - dx, 0.0, dx - box["dx"][1])
    ey = max(box["dy"][0] - dy, 0.0, dy - box["dy"][1])
    return math.hypot(ex, ey)


def rect(ax, box: dict, **kw):
    """A dx/dy box drawn top-down: dy on the horizontal axis (left is left), dx up."""
    (x0, x1), (y0, y1) = box["dx"], box["dy"]
    return ax.add_patch(Rectangle((y0, x0), y1 - y0, x1 - x0, **kw))


def draw_table(ax, table: dict | None) -> np.ndarray | None:
    if not table:
        return None
    (cx, cy), (sx, sy), a = table["center"], table["size"], math.radians(table["yaw_deg"])
    corners = np.array([[sx, sy], [sx, -sy], [-sx, -sy], [-sx, sy]]) / 2
    R = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
    pts = corners @ R.T + [cx, cy]
    ax.add_patch(Polygon(pts[:, ::-1], closed=True, fc=TABLE, ec=AXIS, lw=0.8, zorder=0))
    return pts


def pct(k: int, n: int) -> str:
    return f"{100 * k / n:.0f}% ({k}/{n})" if n else "-"


def fit_slope(x: np.ndarray, y: np.ndarray) -> float | None:
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3 or np.ptp(x[ok]) < 1e-6:
        return None
    return float(np.polyfit(x[ok], y[ok], 1)[0])


def read_jsonl(path: Path) -> list[dict]:
    """The finished trials; a line still being written (the sweep is running) is skipped."""
    rows = []
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return rows


def write_atomic(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text)
    tmp.replace(path)


def main() -> None:
    args = parse_args()
    root = Path(args.results)
    out = (Path(args.out) if args.out else root) / "figures"
    out.mkdir(parents=True, exist_ok=True)
    (out / "index.html").unlink(missing_ok=True)  # the report moved to report/
    rows = read_jsonl(root / "results.jsonl")
    if not rows:
        raise SystemExit(f"no finished trials in {root / 'results.jsonl'}")
    rows.sort(key=lambda r: r["trial"])
    tt = json.loads((root / "train_targets.json").read_text())
    plan = json.loads((root / "plan.json").read_text())
    st = plan["settings"]
    n_file = root / "n_trials.txt"
    planned = min(int(n_file.read_text()), len(plan["trials"])) if n_file.exists() else len(plan["trials"])
    plan_trials = plan["trials"][:planned]
    plan_by_trial = {t["trial"]: t for t in plan_trials}
    grid = tt.get("grid") if st.get("sampling") == "grid" else None
    for r in rows:
        r["_root"] = root
    merged = []
    for i, m in enumerate(args.merge, start=2):
        mroot = Path(m)
        mst = json.loads((mroot / "plan.json").read_text())["settings"]
        if (mst["task"], mst["run_dir"], mst["ckpt_step"]) != (st["task"], st["run_dir"], st["ckpt_step"]):
            raise SystemExit(f"--merge {m}: a sweep of another task or checkpoint")
        mrows = read_jsonl(mroot / "results.jsonl")
        for r in sorted(mrows, key=lambda r: r["trial"]):
            r["_root"], r["label"], r["trial"] = mroot, f"s{i}-{r['label']}", 1000 * (i - 1) + r["trial"]
        merged.append((mroot, mrows))
    box, rng_box = tt["box"], tt["range"]
    train = np.array(tt["rel"], dtype=float)
    si = {}
    for f in ("server_info.json", "conditions.json"):
        try:
            d = json.loads((root / f).read_text())
            si = d.get("server_info", d) or si
        except Exception:
            pass

    def in_range(r) -> bool:
        return (rng_box["dx"][0] - 1e-3 <= r["dx"] <= rng_box["dx"][1] + 1e-3
                and rng_box["dy"][0] - 1e-3 <= r["dy"] <= rng_box["dy"][1] + 1e-3)

    for mroot, mrows in merged:
        for r in mrows:
            p = r.get("placed") or {}
            r["dx"], r["dy"] = (p.get("rel") or [r["plan"]["dx"], r["plan"]["dy"]])[:2]
        keep = [r for r in mrows if in_range(r)]
        rows += keep
        print(f"pooled {len(keep)} trials from {mroot.name} ({len(mrows) - len(keep)} outside this trial range left out)")
    for r in rows:
        p = r.get("placed") or {}
        r["dx"], r["dy"] = (p.get("rel") or [r["plan"]["dx"], r["plan"]["dy"]])[:2]
        r["dyaw"] = p.get("dyaw", r["plan"]["dyaw"])
        # a grid sweep sorts trials by their grid point (the planned pose; placement is within 1 cm of it)
        cx, cy = (r["plan"]["dx"], r["plan"]["dy"]) if grid else (r["dx"], r["dy"])
        r["inside"] = box["dx"][0] <= cx <= box["dx"][1] and box["dy"][0] <= cy <= box["dy"][1]
        r["out_by"] = box_dist(box, cx, cy)
        r["to_train"] = float(np.min(np.hypot(train[:, 0] - r["dx"], train[:, 1] - r["dy"])))
    ins = [r for r in rows if r["inside"]]
    outs = [r for r in rows if not r["inside"]]
    cdx = (lambda r: r["plan"]["dx"]) if grid else (lambda r: r["dx"])
    sides = {"closer than every demo": [r for r in outs if cdx(r) < box["dx"][0]],
             "farther than every demo": [r for r in outs if cdx(r) > box["dx"][1]],
             "demo distance, off to the side": [r for r in outs if box["dx"][0] <= cdx(r) <= box["dx"][1]]}
    # grid points: every one in the plan, with its finished trials (indices into rows)
    points = {}
    if grid:
        for t in plan_trials:
            points.setdefault(t["point"], dict(point=t["point"], grid=t["grid"], dx=t["dx"], dy=t["dy"],
                                               inside=t["in_train_range"], trials=[]))
        for i, r in enumerate(rows):
            pt = plan_by_trial.get(r["trial"], {}).get("point")
            if pt in points:
                points[pt]["trials"].append(i)
        for v in points.values():
            v["n"], v["successes"] = len(v["trials"]), sum(rows[i]["success"] for i in v["trials"])
    points = [points[k] for k in sorted(points)]
    walls = [r["wall_seconds"] for r in rows if r.get("wall_seconds")]
    remaining = max(0, len(plan_trials) - sum(r["trial"] in plan_by_trial for r in rows if r["_root"] == root))
    eta_h = round(float(np.mean(walls)) * remaining / 3600, 1) if walls and remaining else 0.0
    updated = time.strftime("%Y-%m-%d %H:%M")
    k_in, k_out = sum(r["success"] for r in ins), sum(r["success"] for r in outs)
    name, task = tt["name"].replace("_", " "), st["task"]

    # ---------------------------------------------------------------- 1: target map
    fig = plt.figure(figsize=(10.2, 7.6))
    ax = fig.add_axes((0.08, 0.07, 0.62, 0.78))
    table_pts = draw_table(ax, tt.get("table"))
    rect(ax, rng_box, fill=False, ec=INK2, lw=1.0, ls=(0, (4, 3)), zorder=1)
    rect(ax, box, fc=C_TRAIN, ec=C_OK, lw=1.0, alpha=0.9, zorder=1)
    ax.scatter(train[:, 1], train[:, 0], s=7, color=C_DEMO, lw=0, zorder=2)
    for r in ([] if grid else rows):  # too many lines to read on a grid
        ch = r.get("closest_hand")
        if ch and ch.get("rel"):
            hx, hy = ch["rel"]
            ax.plot([r["dy"], hy], [r["dx"], hx], color=MUTED, lw=0.8, zorder=3)
            ax.scatter([hy], [hx], s=6, color=MUTED, lw=0, zorder=3)
    ok = [r for r in rows if r["success"]]
    bad = [r for r in rows if not r["success"]]
    if grid:  # one cell per grid point, coloured by its success rate; dashed = no finished trial yet
        h = grid["step"] / 2
        for v in points:
            done = v["n"] > 0
            ax.add_patch(Rectangle((v["dy"] - h, v["dx"] - h), 2 * h, 2 * h, zorder=5, lw=1.0 if done else 0.6,
                                   fc=RATE_CMAP(v["successes"] / v["n"]) if done else "none",
                                   ec=SURFACE if done else MUTED, ls="-" if done else (0, (2, 2))))
            if done:
                ax.text(v["dy"], v["dx"], f"{v['successes']}/{v['n']}", ha="center", va="center", fontsize=5.5,
                        color=INK if abs(v["successes"] / v["n"] - 0.5) < 0.4 else SURFACE, zorder=6)
        rect(ax, box, fill=False, ec=C_OK, lw=1.4, zorder=7)
    else:
        ax.scatter([r["dy"] for r in ok], [r["dx"] for r in ok], s=90, color=C_OK, edgecolors=SURFACE,
                   linewidths=2.0, zorder=6)
        ax.scatter([r["dy"] for r in bad], [r["dx"] for r in bad], s=80, marker="X", color=C_FAIL,
                   edgecolors=SURFACE, linewidths=1.2, zorder=6)
    # the robot, looking up the page
    ax.add_patch(plt.Circle((0, 0), 0.03, fc=INK, ec="none", zorder=7))
    ax.annotate("", xy=(0, 0.11), xytext=(0, 0.0), zorder=7,
                arrowprops=dict(arrowstyle="-|>,head_width=0.25,head_length=0.5", color=INK, lw=1.4, shrinkA=0,
                                shrinkB=0))
    ax.text(0, -0.045, "robot spawn point, facing up", ha="center", va="top", fontsize=8, color=INK2)
    lo_y = min(rng_box["dy"][0], -0.15) - 0.08
    hi_y = max(rng_box["dy"][1], 0.15) + 0.08
    top = max(rng_box["dx"][1], train[:, 0].max()) + 0.12
    ax.set_xlim(hi_y, lo_y)  # left (+dy) drawn on the left
    ax.set_ylim(min(-0.1, rng_box["dx"][0] - 0.1) if rng_box["dx"][0] < 0.45 else rng_box["dx"][0] - 0.1, top)
    if table_pts is not None:
        ty = float(np.clip(table_pts[:, 1].mean(), lo_y + 0.05, hi_y - 0.05))
        tx = min(float(table_pts[:, 0].max()), top) - 0.02
        if table_pts[:, 0].min() < top:
            ax.text(hi_y - 0.02, tx, "table", ha="left", va="top", fontsize=8, color=MUTED)
    ax.set_aspect("equal")
    ax.set_xlabel("dy (m), to the robot's left  ←")
    ax.set_ylabel("dx (m), ahead of the robot")
    ax.text(box["dy"][1], box["dx"][1] + 0.01, "training range", ha="left", va="bottom", fontsize=8, color=C_OK)
    ax.text(rng_box["dy"][1], rng_box["dx"][1] + 0.01, "trial range", ha="left", va="bottom", fontsize=8, color=INK2)

    legend = [Line2D([], [], ls="", marker="s", ms=9, mfc=C_OK, mec=SURFACE, label="all trials succeeded"),
              Line2D([], [], ls="", marker="s", ms=9, mfc=C_FAIL, mec=SURFACE, label="all trials failed"),
              ] if grid else [
              Line2D([], [], ls="", marker="o", ms=9, mfc=C_OK, mec=SURFACE, label="success"),
              Line2D([], [], ls="", marker="X", ms=9, mfc=C_FAIL, mec=SURFACE, label="failure"),
              Line2D([], [], ls="", marker="o", ms=4, mfc=C_DEMO, mec=C_DEMO,
                     label=f"training demos' target ({tt['n']})"),
              ] + ([] if grid else [
              Line2D([], [], color=MUTED, lw=0.8, marker="o", ms=2.5, label="to the closest point a hand reached")])
    fig.legend(handles=legend, loc="upper left", bbox_to_anchor=(0.72, 0.84), fontsize=8.5, handletextpad=0.6,
               labelspacing=0.9)
    lines = [("inside the training range", pct(k_in, len(ins))), ("outside it", pct(k_out, len(outs)))]
    lines += [(f"   {k}", pct(sum(r["success"] for r in v), len(v))) for k, v in sides.items() if v]
    fig.text(0.735, 0.56, "Success rate", fontsize=9.5, color=INK, va="top")
    for i, (lab, val) in enumerate(lines):
        yy = 0.525 - 0.03 * i
        fig.text(0.735, yy, lab, fontsize=8.5, color=INK2 if lab.startswith(" ") else INK, va="top")
        fig.text(0.985, yy, val, fontsize=8.5, color=INK, va="top", ha="right")
    fig.text(0.08, 0.955, f"{task}: where the {name} was, and whether the policy succeeded", fontsize=11, ha="left")
    fig.text(0.08, 0.925, f"{len(rows)} closed-loop trials, top-down in the robot's start frame. The robot faces up; "
             f"the target's yaw was drawn from {st['yaw_deg'][0]:+.0f}..{st['yaw_deg'][1]:+.0f}°.",
             fontsize=8.5, color=INK2, ha="left")
    fig.text(0.08, 0.90, "Distractors, materials and lighting re-drawn every trial (SIMPLE replay level 2).",
             fontsize=8.5, color=INK2, ha="left")
    fig.savefig(out / "1_target_map.png", dpi=140)
    plt.close(fig)

    # ---------------------------------------------------------------- 2: tracking
    # Where a hand was when it came nearest the target, per axis, against where the target was. On the diagonal the
    # hand went to the target; flat at the training range means it went where the demos' targets were, wherever this
    # one was. (Nearest-approach points lean towards the target a little by construction; a flat line is still flat.)
    tr = [r for r in rows if r.get("closest_hand") and r["closest_hand"].get("rel")]
    fig, axs = plt.subplots(1, 2, figsize=(11.0, 5.4))
    slopes = {}
    for ax, (i, key, lab) in zip(axs, [(1, "dy", "sideways: dy (m), + = robot's left"),
                                        (0, "dx", "forward: dx (m), ahead of the robot")]):
        x = np.array([r[key] for r in tr])
        y = np.array([r["closest_hand"]["rel"][i] for r in tr])
        s_ = np.array([r["success"] for r in tr], dtype=bool)
        lo, hi = (rng_box[key][0] - 0.03, rng_box[key][1] + 0.03)
        ax.axvspan(*box[key], color=C_TRAIN, lw=0, zorder=0)
        ax.text(sum(box[key]) / 2, hi - 0.01, "training\nrange", ha="center", va="top", fontsize=8, color=C_OK)
        ax.plot([lo, hi], [lo, hi], color=AXIS, lw=1.2, ls=(0, (4, 3)), zorder=1)
        ax.text(hi - 0.01, hi - 0.06 * (hi - lo), "hand at\nthe target", ha="right", va="top", fontsize=8, color=MUTED)
        ax.scatter(x[s_], y[s_], s=70, color=C_OK, edgecolors=SURFACE, linewidths=2.0, zorder=4, label="success")
        ax.scatter(x[~s_], y[~s_], s=60, marker="X", color=C_FAIL, edgecolors=SURFACE, linewidths=1.0, zorder=4,
                   label="failure")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal")
        ax.set_xlabel(f"target {lab}")
        ax.set_ylabel(f"hand {key} when nearest the target (m)")
        slopes[key] = fit_slope(x, y)
        ax.set_title(f"{'Sideways' if key == 'dy' else 'Forward'}"
                     + (f"   slope {slopes[key]:.2f} (1 = follows, 0 = ignores)" if slopes[key] is not None else ""),
                     loc="left")
    axs[0].legend(loc="upper left")
    fig.suptitle("Does the hand go where the target is?", x=0.02, ha="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(out / "2_tracking.png", dpi=140)
    plt.close(fig)
    (out / "2_reach.png").unlink(missing_ok=True)  # earlier layout

    # ---------------------------------------------------------------- report/: index.html, results.json, videos
    head = dict(
        task=task, checkpoint=st["run_dir"], ckpt_step=st["ckpt_step"], target=tt["name"], trials=len(rows),
        instruction=(rows[0].get("placed") or {}).get("instruction"), sampling=st.get("sampling", "random"),
        progress=dict(done=len(plan_trials) - remaining, planned=len(plan_trials), eta_hours=eta_h, updated=updated),
        training_range=box, trial_range=rng_box, yaw_deg=st["yaw_deg"], in_dist_setting=st["in_dist"],
        successes=sum(r["success"] for r in rows),
        inside_training_range=dict(n=len(ins), successes=k_in, ci95=list(wilson(k_in, len(ins)))),
        outside_training_range=dict(n=len(outs), successes=k_out, ci95=list(wilson(k_out, len(outs)))),
        by_direction={k: dict(n=len(v), successes=sum(r["success"] for r in v)) for k, v in sides.items()},
        hand_follows_target_slope=slopes,
        outcomes={o: sum(r["outcome"] == o for r in rows) for o in OUTCOMES if any(r["outcome"] == o for r in rows)})
    (out / "summary.json").unlink(missing_ok=True)  # superseded by report/results.json

    def num(v, f=".2f"):
        return "-" if v is None or (isinstance(v, float) and not np.isfinite(v)) else format(v, f)

    def web_video(src: Path, dst: Path) -> None:
        """The clip re-encoded small for the browser (H.264, fast start); a plain copy if ffmpeg is missing."""
        if dst.exists() and dst.stat().st_mtime >= src.stat().st_mtime:
            return
        tmp = dst.with_name(dst.stem + ".part.mp4")
        cmd = ["ffmpeg", "-loglevel", "error", "-y", "-i", str(src), "-c:v", "libx264", "-preset", "medium",
               "-crf", "30", "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an", str(tmp)]
        if shutil.which("ffmpeg") and subprocess.run(cmd).returncode == 0:
            tmp.replace(dst)
        else:
            tmp.unlink(missing_ok=True)
            shutil.copyfile(src, dst)

    report = out.parent / "report"
    (report / "videos").mkdir(parents=True, exist_ok=True)

    def video(r) -> str | None:
        """The trial's clip for the browser: encoded from the raw one while that exists, else the one made before."""
        dst = report / "videos" / f"{r['label']}.mp4"
        vids = sorted(glob.glob(str(r["_root"] / (r.get("video_dir") or "-") / "head_stereo_left*.mp4")))
        if vids:
            web_video(Path(vids[0]), dst)
        return f"videos/{dst.name}" if dst.exists() else None

    trials = []
    for r in rows:
        r["video"] = video(r)
        pl = plan_by_trial.get(r["trial"], {}) if r["_root"] == root else {}
        trials.append(dict(
            trial=r["trial"], label=r["label"], dx=r["dx"], dy=r["dy"], dyaw=r["dyaw"], success=r["success"],
            outcome=r["outcome"], inside=r["inside"], hand=(r.get("closest_hand") or {}).get("rel"),
            steps=r.get("steps", 0), video=r["video"], scene=r.get("base_episode"),
            gx=pl.get("dx", r["dx"]), gy=pl.get("dy", r["dy"]), point=pl.get("point"), rep=pl.get("rep")))
    tbl = tt.get("table")
    if tbl:  # the table's edge nearest the robot, for the label
        a = math.radians(tbl["yaw_deg"])
        tbl = dict(tbl, edge=min(tbl["center"][0] + math.cos(a) * u * tbl["size"][0] / 2
                                 - math.sin(a) * w * tbl["size"][1] / 2 for u in (-1, 1) for w in (-1, 1)))
    short = task.replace("G1Wholebody", "").replace("Teleop", "").removesuffix("-v0")
    title = re.sub(r"(?<!^)(?=[A-Z])", " ", short) + " target map"
    data = dict(task=task, title=title, kind=tt["kind"], checkpoint=str(si.get("run_dir", st["run_dir"])),
                ckpt_step=str(st["ckpt_step"]), box=box, range=rng_box, table=tbl,
                train=[[round(a, 4), round(b, 4)] for a, b, *_ in tt["rel"]], trials=trials,
                grid=None if not grid else dict(step=grid["step"], repeats=grid["repeats"]),
                points=[dict(point=v["point"], dx=v["dx"], dy=v["dy"], inside=v["inside"], n=v["n"],
                             k=v["successes"], trials=v["trials"]) for v in points],
                planned=len(plan_trials), done=len(plan_trials) - remaining, eta_h=eta_h, updated=updated)
    page = (TEMPLATE.read_text().replace("__TITLE__", html.escape(title))
            .replace("__DATA_JSON__", json.dumps(data).replace("</", "<\\/")))
    # line 1 makes the file a standalone page; drop it when the page goes into a host that adds its own skeleton
    write_atomic(report / "index.html", '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" '
                                        'content="width=device-width, initial-scale=1, viewport-fit=cover">\n' + page)
    # every number behind the page, kept with it
    detail = [{k: v for k, v in r.items() if not k.startswith("_")} for r in rows]
    write_atomic(report / "results.json", json.dumps(dict(
        **head, settings=st, grid=grid, train_targets=dict(n=tt["n"], rel=tt["rel"], yaw_deg=tt["yaw_deg"]),
        points=[dict(point=v["point"], grid=v["grid"], dx=v["dx"], dy=v["dy"], inside=v["inside"], n=v["n"],
                     successes=v["successes"], trials=[rows[i]["label"] for i in v["trials"]]) for v in points],
        trials_detail=detail), indent=1))
    print(f"report: {report / 'index.html'}   data: {report / 'results.json'}")
    print(f"figures: {out}")
    print(f"  {len(rows)} trials" + (f" of {len(plan_trials)} ({eta_h:.1f} h to go)" if remaining else "")
          + f"   inside the training range: {pct(k_in, len(ins))}   outside: {pct(k_out, len(outs))}   "
          + "   ".join(f"{k}: {pct(sum(r['success'] for r in v), len(v))}" for k, v in sides.items() if v))
    print(f"  hand follows the target, slope (1 = follows, 0 = ignores): dy {num(slopes.get('dy'))}, "
          f"dx {num(slopes.get('dx'))}")
    if grid:
        return
    for r in rows:
        where = "inside   " if r["inside"] else f"{100 * r['out_by']:3.0f} cm out"
        print(f"  {r['label']}  dx {r['dx']:+.3f} dy {r['dy']:+.3f}  {where}  {r['outcome']:<18} "
              f"closest hand {num(r.get('min_hand_dist'))} m")

if __name__ == "__main__":
    main()
