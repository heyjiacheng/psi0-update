"""Figures for the VLM camera sweep (stage 3). Runs in Psi0's venv, CPU only.

Latent change of a render = 1 - cos(mean of its tokens, mean of the anchor's tokens), on the last VLM
layer (the one the action expert reads), for two token groups:
  image tokens   the 80 image tokens
  all tokens     all 102 tokens (system prompt + image + instruction + template), i.e. the entire latent
and in three views of that latent (see extract_sweep_latents.py):
  raw       the last layer as passed on; dominated by 4 massive-activation channels
  content   the same without those channels: where the image information lives (used for the probes)
  policy    the action expert's input projection of it

Figures (PNG, next to the latents in figures/):
  1_latent_vs_camera   latent change vs each camera DoF, per view, with the render-noise floor and the
                       change the recorded arrival produces as references
  1b_latent_vs_camera_other_scene
                       the same sweeps in a second scene (lighting and materials re-drawn), content view,
                       every point measured against the first scene's anchor: does the curve survive a
                       scene change?
  2_real_episode       the recorded episode re-rendered: latent change over time, recorded vs the same
                       head poses with the arms held down, next to distance, head pitch and the policy's
                       locomotion -> manipulation readout
  3_token_maps         which image tokens change: the 8x10 per-token change drawn over the image
  4_layers             the sweeps through all 29 VLM layers (per-token change), image vs text tokens
  5_switch_readouts    linear probes on the latent (pelvis-box distance, P(ready)) and the policy's
                       P(manipulation) from its predicted SONIC tokens, vs each camera DoF
  6_floor_map          top-down map over floor positions of latent change and readouts
  7_latent_pca         the sweeps inside the latent space of the recorded episode
  8_massive_channels   why the raw view is misleading: channel magnitudes and their share per layer
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from render_camera_sweep import AXES, phase_labels  # noqa: E402  (pure numpy helpers)

# ----------------------------------------------------------------------------- style (reference palette)
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
FLOOR_FILL = "#ecebe6"
C_IMG, C_ALL, C_POLICY = "#2a78d6", "#eb6834", "#1baf7a"  # categorical slots 1-3, fixed order
C_CF = MUTED
C_SCENE1, C_SCENE2 = C_IMG, C_ALL  # categorical slots 1-2 (1b is its own figure with its own legend)
SEQ = LinearSegmentedColormap.from_list(
    "seq_blue", ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"])
DIV = LinearSegmentedColormap.from_list("div", ["#184f95", "#3987e5", "#f0efec", "#e34948", "#9b2322"])
PHASE_SHADE = {"stand_initial": "#f4f3ef", "walk_to_box": "#e8e6df", "stand_arrived": "#dcdad1",
               "bend_reach": "#cdcbc0", "lift": "#e8e6df", "carry": "#f4f3ef"}
UNITS = {"x": "m", "y": "m", "z": "m", "yaw": "deg", "pitch": "deg", "roll": "deg"}
AXIS_LABEL = {"x": "forward x (m)", "y": "left y (m)", "z": "up z (m)",
              "yaw": "yaw, + = turn left (deg)", "pitch": "pitch, + = look down (deg)",
              "roll": "roll, + = left side up (deg)"}
SPACES = {"raw": "raw VLM last layer", "content": "VLM content channels\n(massive channels left out)",
          "policy": "action-expert input\n(views_proj + norm)"}
LOCO, READY = ("stand_initial", "walk_to_box"), ("stand_arrived", "bend_reach")

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
    p.add_argument("--renders", required=True)
    p.add_argument("--latents", default=None, help="default: <renders>/latents.npz")
    p.add_argument("--out", default=None, help="default: <renders>/figures")
    p.add_argument("--data-dir", default=None, help="recorded dataset (default: the one the renders used)")
    p.add_argument("--pcs", type=int, default=16, help="PCA components for the linear probes")
    return p.parse_args()


# ----------------------------------------------------------------------------- small numpy models
class PCA:
    def __init__(self, X, k):
        self.mu = X.mean(0)
        _, _, vt = np.linalg.svd(X - self.mu, full_matrices=False)
        self.W = vt[:k].T
        self.sd = ((X - self.mu) @ self.W).std(0) + 1e-8

    def __call__(self, X, whiten=True):
        Z = (X - self.mu) @ self.W
        return Z / self.sd if whiten else Z


def ridge(Z, y, lam=1.0):
    A = np.c_[Z, np.ones(len(Z))]
    w = np.linalg.solve(A.T @ A + lam * np.diag(np.r_[np.ones(Z.shape[1]), 0]), A.T @ y)
    return lambda Zq: np.c_[Zq, np.ones(len(Zq))] @ w


def logistic(Z, y, lam=1.0, iters=50):
    """Class-balanced L2 logistic regression (Newton)."""
    A = np.c_[Z, np.ones(len(Z))]
    sw = np.where(y == 1, 0.5 / max(y.mean(), 1e-6), 0.5 / max(1 - y.mean(), 1e-6))
    reg = lam * np.diag(np.r_[np.ones(Z.shape[1]), 0])
    w = np.zeros(A.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-A @ w))
        H = (A * (sw * p * (1 - p))[:, None]).T @ A + reg
        w -= np.linalg.solve(H, A.T @ (sw * (p - y)) + reg @ w)
    return lambda Zq: 1 / (1 + np.exp(-np.c_[Zq, np.ones(len(Zq))] @ w))


def balanced_acc(p, y):
    y = y.astype(bool)
    return float(0.5 * ((p > 0.5)[y].mean() + (p <= 0.5)[~y].mean()))


def cos_dist(X, x0):
    return 1 - (X @ x0) / (np.linalg.norm(X, axis=1) * np.linalg.norm(x0) + 1e-12)


# ----------------------------------------------------------------------------- drawing helpers
def shade_phases(ax, t, ph, label=False):
    step = t[1] - t[0] if len(t) > 1 else 1.0
    start = 0
    for i in range(1, len(ph) + 1):
        if i == len(ph) or ph[i] != ph[start]:
            lo, hi = t[start] - step / 2, t[i - 1] + step / 2
            ax.axvspan(lo, hi, color=PHASE_SHADE.get(ph[start], SURFACE), lw=0, zorder=0)
            if label and hi - lo > 0.9:
                ax.text((lo + hi) / 2, 1.03, ph[start].replace("_", " "), transform=ax.get_xaxis_transform(),
                        ha="center", va="bottom", fontsize=7, color=INK2)
            start = i


def thumb(path, w=200):
    im = Image.open(path).convert("RGB")
    return np.asarray(im.resize((w, int(w * im.height / im.width))))


def end_label(ax, x, y, text, color):
    ax.plot([x], [y], "o", color=color, ms=4, zorder=5)
    ax.annotate(text, (x, y), xytext=(5, 0), textcoords="offset points", va="center", fontsize=7.5, color=INK2)


def end_labels(ax, items, gap=0.09):
    """Line-end dots + labels, labels nudged apart so they never overlap. items: (x, y, text, color)."""
    lo, hi = ax.get_ylim()
    x0, x1 = ax.get_xlim()
    ys = spread_labels([it[1] for it in items], gap * (hi - lo))
    xk = max(it[0] for it in items) + 0.015 * (x1 - x0)
    for (x, y, text, color), yl in zip(items, ys):
        ax.plot([x], [y], "o", color=color, ms=4, zorder=5)
        ax.plot([x, xk], [y, yl], color=AXIS, lw=0.6, zorder=4)
        ax.plot([xk, xk + 0.02 * (x1 - x0)], [yl, yl], color=color, lw=2.5, solid_capstyle="round", zorder=5)
        ax.text(xk + 0.028 * (x1 - x0), yl, text, va="center", fontsize=7.5, color=INK2)


def fmt(a, v):
    return f"{v:+.2f} m" if UNITS[a] == "m" else f"{v:+.0f} deg"


def spread_labels(ys, min_gap):
    """Nudge label y positions apart (keeps order)."""
    order = np.argsort(ys)
    out = np.array(ys, dtype=float)
    for k in range(1, len(order)):
        if out[order[k]] - out[order[k - 1]] < min_gap:
            out[order[k]] = out[order[k - 1]] + min_gap
    return out


def main() -> None:
    args = parse_args()
    renders = Path(args.renders)
    out = Path(args.out) if args.out else renders / "figures"
    out.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(line) for line in open(renders / "index.jsonl")]
    meta = json.loads((renders / "meta.json").read_text())
    z = np.load(args.latents or renders / "latents.npz")
    anchor_ep = meta["anchor"]["episode"]
    n = len(rows)
    grp = np.array([r["group"] for r in rows])
    val = np.array([r["value"] for r in rows])
    axis_of = np.array([r["axis"] for r in rows])
    ep_of = np.array([r.get("episode", anchor_ep) for r in rows])
    phase = np.array([r.get("phase", "") for r in rows])
    is_img, is_txt = z["is_img"], z["is_txt"]
    gh, gw = (int(v) for v in z["img_grid"])
    has_action = bool(z["has_action"])
    # the anchor: the dataset's own image of the anchor frame (a training image with the training data), else
    # the first re-render; the noise/return renders then measure how faithfully the anchor is re-rendered
    rec_idx = np.nonzero(grp == "recorded")[0]
    a0 = int(rec_idx[0]) if len(rec_idx) else int(np.nonzero(grp == "noise")[0][0])
    data_label = meta["args"].get("data_label") or "the dataset"
    anchor_name = f"recorded frame, {data_label}" if len(rec_idx) else "re-rendered anchor"

    # ---- latent change against the anchor, per view and token group
    D = {sp: {g: cos_dist(z[f"pool_{sp}_{g}"], z[f"pool_{sp}_{g}"][a0]) for g in ("img", "all")} for sp in SPACES}
    floor_idx = np.nonzero(((grp == "noise") & (np.arange(n) != a0)) | (grp == "return"))[0]
    floor = {sp: {g: float(D[sp][g][floor_idx].max()) for g in D[sp]} for sp in SPACES}
    zero = {sp: {g: float(np.median(D[sp][g][floor_idx])) for g in D[sp]} for sp in SPACES}
    tokc = z["tok_cos_content"][:, -1]  # (n, N) per-token change, content view, last layer
    FEAT = z["pool_content_img"]  # features for probes / PCA

    def frames(group):
        idx = np.nonzero((grp == group) & (ep_of == anchor_ep))[0]
        return idx[np.argsort(val[idx])]

    traj, cf = frames("traj"), frames("traj_cf")

    def sweep(a):
        idx = np.nonzero((grp == "sweep") & (axis_of == a))[0]
        return idx[np.argsort(val[idx])]

    def med(metric, idx, phases):
        sel = [i for i in idx if phase[i] in phases]
        return float(np.median(metric[sel])) if sel else np.nan

    refs = {sp: {"recorded arrival": med(D[sp]["img"], traj, ("stand_arrived",)),
                 "arrival pose, arms down": med(D[sp]["img"], cf, ("stand_arrived",))} for sp in SPACES}

    # ---- probes: VLM probes train on the arms-down renders of the *other* episodes (the same kind of image as
    # the sweeps: anchor posture, camera at a recorded head pose); the anchor episode is held out
    pre_lift = ~np.isin(phase, ("lift", "carry"))
    # camera-box horizontal distance: what the camera geometry fixes (rotations leave it unchanged)
    cam_box = np.array([r.get("cam_box_xy", np.nan) for r in rows])
    train = np.nonzero((grp == "traj_cf") & (ep_of != anchor_ep) & pre_lift)[0]
    if len(train) < 20:  # single-episode run: fall back to the anchor episode
        train = np.nonzero((grp == "traj_cf") & pre_lift)[0]
    probe: dict = {}
    if len(train) >= 20:
        pca = PCA(FEAT[train], min(args.pcs, len(train) - 1))
        Zall = pca(FEAT)
        probe["dist"] = ridge(Zall[train], cam_box[train])(Zall)
        probe["ready"] = logistic(Zall[train], np.isin(phase[train], READY).astype(float))(Zall)
        held = np.nonzero((grp == "traj_cf") & (ep_of == anchor_ep) & pre_lift)[0]
        if len(held):
            d_true = cam_box[held]
            probe["dist_rmse_m"] = float(np.sqrt(np.mean((probe["dist"][held] - d_true) ** 2)))
            probe["dist_r2"] = float(1 - np.mean((probe["dist"][held] - d_true) ** 2) / np.var(d_true))
            probe["ready_bal_acc"] = balanced_acc(probe["ready"][held], np.isin(phase[held], READY))
        xs_ = sweep("x")
        # truth: moving the camera forward by 1 m shortens the camera-box distance by ~1 m
        probe["dist_slope_vs_forward_x"] = float(np.polyfit(val[xs_], probe["dist"][xs_], 1)[0])
    # policy readout: P(manipulation) of the SONIC body tokens the action expert predicts, via a classifier fitted
    # on the recorded (teleop) tokens of locomotion vs arrived/bend frames
    data_dir = Path(args.data_dir or meta["args"]["data_dir"])
    if not data_dir.is_absolute():
        data_dir = Path(__file__).resolve().parents[4] / "SIMPLE" / data_dir
    if has_action and data_dir.exists():
        toks, ys, eps = [], [], []
        for f in sorted(glob.glob(str(data_dir / "data" / "*" / "*.parquet"))):
            df = pd.read_parquet(f)
            ph = phase_labels(df)
            keep = np.isin(ph, LOCO + READY)
            toks.append(np.stack(df["action"])[keep, :64])
            ys.append(np.isin(ph[keep], READY).astype(float))
            eps.append(np.full(keep.sum(), int(df["episode_index"].iloc[0])))
        T, Y, E = np.concatenate(toks), np.concatenate(ys), np.concatenate(eps)
        tr = E != anchor_ep if (E != anchor_ep).any() else np.ones(len(E), bool)
        tok_fn = logistic(T[tr], Y[tr])
        if (~tr).any():
            probe["token_classifier_bal_acc"] = balanced_acc(tok_fn(T[~tr]), Y[~tr])
        p_of = lambda pred: tok_fn(pred[:, :24, :64].reshape(-1, 64)).reshape(len(pred), 24).mean(1)  # executed rows
        probe["policy"] = p_of(z["pred"])
        if "pred_swap" in z:
            probe["policy_swap"] = p_of(z["pred_swap"])
    if "policy_swap" in probe and len(traj) and len(cf) == len(traj):
        bend = phase[traj] == "bend_reach"
        factorial = {"recorded image + recorded state": probe["policy"][traj],
                     "recorded image + anchor state": probe["policy_swap"][traj],
                     "arms-down image + recorded state": probe["policy_swap"][cf],
                     "arms-down image + anchor state": probe["policy"][cf]}
        probe["factorial_P_manip_during_bend"] = {k: round(float(v[bend].mean()), 3) for k, v in factorial.items()}
    scalars = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in probe.items() if isinstance(v, (float, dict))}
    print("probes:", scalars)
    arrival_d = med(cam_box, traj, ("stand_arrived",))

    summary = dict(
        anchor=dict(episode=anchor_ep, frame=meta["anchor"]["frame"], pelvis_box_xy=meta["anchor"]["pelvis_box_xy"]),
        table_material=meta.get("table_material"), mode=meta["args"].get("mode"),
        gi=meta["args"].get("gi", "on" if meta["args"].get("eval_lighting") else "off"),
        data_dir=meta["args"]["data_dir"], data_label=data_label, anchor_image=anchor_name,
        anchor_recorded_gap_px=meta["anchor"].get("recorded_gap"),
        massive_channels_last_layer=np.nonzero(z["massive"][-1])[0].tolist(),
        massive_energy_share_img=float(z["massive_share"]),
        noise_floor=floor, zero_offset=zero, references_img=refs, probes=scalars, sweeps={})
    for a in AXES:
        idx = sweep(a)
        summary["sweeps"][a] = dict(values=val[idx].round(4).tolist(),
                                    **{f"{sp}_{g}": D[sp][g][idx].round(5).tolist() for sp in SPACES for g in ("img", "all")})

    # ======================================================================== 1. latent vs camera
    fig = plt.figure(figsize=(17, 11.5))
    gs_top = fig.add_gridspec(1, 6, left=0.075, right=0.9, top=0.875, bottom=0.8, wspace=0.14)
    gs = fig.add_gridspec(3, 6, left=0.075, right=0.9, top=0.76, bottom=0.05, hspace=0.22, wspace=0.14)
    for k, a in enumerate(AXES):
        idx = sweep(a)
        tg = gs_top[0, k].subgridspec(1, 3, wspace=0.05)
        for j, i in enumerate([idx[0], a0, idx[-1]]):
            tax = fig.add_subplot(tg[0, j])
            tax.imshow(thumb(renders / rows[i]["file"]))
            tax.set_xticks([]), tax.set_yticks([])
            for s in tax.spines.values():
                s.set_visible(False)
            tax.set_xlabel(("training frame" if len(rec_idx) else "anchor") if i == a0 else fmt(a, val[i]),
                           fontsize=7, color=INK2, labelpad=1)
            if j == 1:
                tax.set_title(AXIS_LABEL[a], fontsize=9.5, pad=4)
        xs = np.r_[val[idx], 0.0]
        order = np.argsort(xs)
        for r, sp in enumerate(SPACES):
            ax = fig.add_subplot(gs[r, k])
            ymax = max(D[sp][g][grp == "sweep"].max() for g in ("img", "all")) * 1.1
            ax.axhspan(0, max(floor[sp].values()), color=FLOOR_FILL, lw=0, zorder=0.5)
            for name, y in refs[sp].items():
                if np.isfinite(y):
                    ax.axhline(y, color=MUTED, lw=0.8, zorder=1)
            for g, color, label in (("all", C_ALL, "all tokens (entire latent)"), ("img", C_IMG, "image tokens")):
                ys = np.r_[D[sp][g][idx], zero[sp][g]][order]
                ax.plot(xs[order], ys, "-o", color=color, label=label, ms=3, lw=1.8, zorder=3)
            ax.axvline(0, color=AXIS, lw=0.8, zorder=1)
            ax.set_ylim(0, ymax)
            if k:
                ax.tick_params(labelleft=False)
            else:
                ax.set_ylabel(f"{SPACES[sp]}\n1 - cos(pooled, anchor)", fontsize=8)
            if r == len(SPACES) - 1:
                ax.set_xlabel(UNITS[a], fontsize=8)
            if k == len(AXES) - 1:
                names = [nm for nm, y in refs[sp].items() if np.isfinite(y)]
                ys_ = spread_labels([refs[sp][nm] for nm in names], ymax * 0.07)
                for nm, y in zip(names, ys_):
                    ax.text(1.02, y, nm, transform=ax.get_yaxis_transform(), fontsize=7, color=INK2, va="center")
            if r == 0 and k == 0:
                handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles[::-1], labels[::-1], loc="upper left", ncol=2, bbox_to_anchor=(0.07, 0.935))
    fig.suptitle(f"How much the VLM latent moves when the head camera moves   (anchor: {anchor_name}, episode "
                 f"{anchor_ep}, frame {meta['anchor']['frame']}, pelvis-box {meta['anchor']['pelvis_box_xy']:.2f} m, "
                 f"{meta['args'].get('mode', 'rigid')} mode)", x=0.075, y=0.985, ha="left", fontsize=11.5)
    fig.text(0.075, 0.963, f"Gray band: the anchor pose re-rendered (repeats, and again after each sweep) vs the {anchor_name}"
             ": how faithfully the anchor is reproduced.\nGray lines: the change the recorded episode's arrival frames "
             "produce, as recorded and at the same head pose with the arms held down.", fontsize=8, color=INK2, va="top",
             linespacing=1.4)
    fig.savefig(out / "1_latent_vs_camera.png", dpi=140)
    plt.close(fig)

    # ======================================================================== 1b. the same sweeps in a second scene
    if (grp == "sweep2").any():
        sp, sc1, sc2 = "content", meta.get("scene") or {}, meta.get("scene2") or {}

        def sweep2(a):  # offset 0 is the first scene's anchor, so scene-2 renders at offset 0 are left out
            idx = np.nonzero((grp == "sweep2") & (axis_of == a) & (np.abs(val) > 1e-9))[0]
            return idx[np.argsort(val[idx])]

        # the scene change alone: scene 2 rendered at the anchor pose
        still = np.nonzero((grp == "noise2") | (grp == "return2") | ((grp == "sweep2") & (np.abs(val) < 1e-9)))[0]
        scene_only = {g: float(np.median(D[sp][g][still])) for g in ("img", "all")}
        # the same sweeps against scene 2's own anchor: is the camera response intact, only offset by the scene?
        b0 = int(np.nonzero(grp == "noise2")[0][0])
        D2 = {g: cos_dist(z[f"pool_{sp}_{g}"], z[f"pool_{sp}_{g}"][b0]) for g in ("img", "all")}
        stats = {}
        for a in AXES:
            i1, i2 = sweep(a), sweep2(a)
            i1 = i1[np.abs(val[i1]) > 1e-9]
            assert np.allclose(val[i1], val[i2]), f"{a}: scene sweeps have different offsets"
            # shape_r: do both scenes rank the moved poses alike; scene2_minus_scene1: the constant the scene adds;
            # camera_shift_cos: cosine between the latent shifts the same camera move causes in the two scenes
            P = z[f"pool_{sp}_img"]
            shift_cos = [float((P[i] - P[a0]) @ (P[j] - P[b0]) / (np.linalg.norm(P[i] - P[a0]) * np.linalg.norm(P[j] - P[b0])))
                         for i, j in zip(i1, i2)]
            stats[a] = {g: dict(shape_r=float(np.corrcoef(D[sp][g][i1], D[sp][g][i2])[0, 1]),
                                shape_r_vs_own_anchor=float(np.corrcoef(D[sp][g][i1], D2[g][i2])[0, 1]),
                                scene2_minus_scene1=float(np.mean(D[sp][g][i2] - D[sp][g][i1])),
                                max_scene1=float(D[sp][g][i1].max()), max_scene2_vs_own_anchor=float(D2[g][i2].max()))
                        for g in ("img", "all")}
            stats[a]["camera_shift_cos_img_median"] = float(np.median(shift_cos))

        def describe(sc):
            parts = []
            if sc.get("table_material"):
                parts.append(f"table: {sc['table_material'].replace('_', ' ')}")
            if sc.get("light_color_temperature_K"):
                parts.append(f"{sc.get('n_lights', 0)} lights, {sc['light_color_temperature_K'][0]:.0f} K, "
                             f"intensity {sc['light_intensity'][0]:.0f}")
            return "\n".join(parts)

        fig = plt.figure(figsize=(17, 10.2))
        gs_top = fig.add_gridspec(1, 2, left=0.075, right=0.395, top=0.905, bottom=0.735, wspace=0.05)
        gs = fig.add_gridspec(2, 6, left=0.075, right=0.9, top=0.64, bottom=0.06, hspace=0.3, wspace=0.14)
        name1 = (f"scene 1: {data_label}, episode {anchor_ep}\nthe anchor is its recorded frame" if len(rec_idx)
                 else "scene 1 (original)")
        name2 = (f"scene 2: lighting + materials re-drawn\ntable material not in the {data_label}"
                 if sc2.get("table_material_in_dataset") is False else "scene 2 (re-drawn)")
        for j, (i, name, sc, color) in enumerate(((a0, name1, sc1, C_SCENE1), (still[0], name2, sc2, C_SCENE2))):
            tax = fig.add_subplot(gs_top[0, j])
            tax.imshow(thumb(renders / rows[i]["file"], w=420))
            tax.set_xticks([]), tax.set_yticks([])
            for s_ in tax.spines.values():
                s_.set_edgecolor(color)
                s_.set_linewidth(2.5)
                s_.set_visible(True)
            tax.set_title(name, fontsize=8.5, loc="left", pad=3)
            tax.set_xlabel(describe(sc), fontsize=7.5, color=INK2, labelpad=3, loc="left")
        rows_ = (("img", "image tokens"), ("all", "all tokens (entire latent)"))
        for r, (g, gname) in enumerate(rows_):
            ymax = 1.1 * max(D[sp][g][np.isin(grp, ("sweep", "sweep2"))].max(), scene_only[g])
            for k, a in enumerate(AXES):
                ax = fig.add_subplot(gs[r, k])
                ax.axhspan(0, floor[sp][g], color=FLOOR_FILL, lw=0, zorder=0.5)
                ax.axhline(scene_only[g], color=MUTED, lw=0.8, zorder=1)
                ax.axvline(0, color=AXIS, lw=0.8, zorder=1)
                for idx, color, label in ((sweep(a), C_SCENE1, "scene 1 (same scene as the anchor)"),
                                          (sweep2(a), C_SCENE2, "scene 2 (anchor stays in scene 1)")):
                    xs = np.r_[val[idx], 0.0]
                    ys = np.r_[D[sp][g][idx], zero[sp][g]]
                    o = np.argsort(xs, kind="stable")
                    ax.plot(xs[o], ys[o], "-o", color=color, label=label, ms=3, lw=1.8, zorder=3)
                ax.plot([0], [scene_only[g]], "o", mfc=SURFACE, mec=C_SCENE2, mew=1.6, ms=6, zorder=4,
                        label="scene 2, camera not moved")
                ax.set_ylim(0, ymax)
                st = stats[a][g]
                ax.text(0.03, 0.97, f"shape r = {st['shape_r']:.2f}\nvs own anchor {st['shape_r_vs_own_anchor']:.2f}",
                        transform=ax.transAxes, fontsize=7.5, linespacing=1.3,
                        color=INK2, va="top", ha="left",
                        bbox=dict(boxstyle="round,pad=0.25", fc=SURFACE, ec="none", alpha=0.85))
                if r == 0:
                    ax.set_title(AXIS_LABEL[a], fontsize=9.5, pad=4)
                if k:
                    ax.tick_params(labelleft=False)
                else:
                    ax.set_ylabel(f"{gname}\n1 - cos(pooled, scene-1 anchor)", fontsize=8)
                ax.set_xlabel(UNITS[a], fontsize=8)
                if k == len(AXES) - 1:
                    ax.text(1.02, scene_only[g], "scene change alone\n(scene 2, camera not moved)",
                            transform=ax.get_yaxis_transform(), fontsize=7, color=INK2, va="center")
                if r == 0 and k == 0:
                    handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper left", ncol=1, bbox_to_anchor=(0.42, 0.905), fontsize=8.5)
        r_img = np.mean([stats[a]["img"]["shape_r"] for a in AXES])
        r_own = np.mean([stats[a]["img"]["shape_r_vs_own_anchor"] for a in AXES])
        fig.text(0.42, 0.815,
                 f"Every point is 1 - cos to the anchor ({anchor_name}), content view (massive channels left out).\n"
                 "At offset 0 both curves are the anchor pose re-rendered in scene 1; everywhere else the orange\n"
                 "curve is rendered in scene 2 with exactly the same camera pose as the blue one.\n"
                 "Gray band: re-renders of the anchor pose.  Gray line / hollow dot: the scene change on its own.\n"
                 f"shape r: Pearson correlation of the two curves over the moved poses (mean {r_img:.2f} on image "
                 "tokens).\n"
                 f"vs own anchor: the same with scene 2's curve measured against scene 2's anchor instead (mean "
                 f"{r_own:.2f}).", fontsize=8, color=INK2, va="top", linespacing=1.5)
        fig.suptitle(f"Does the latent's response to camera motion survive a scene change?   (anchor: {anchor_name}, "
                     f"episode {anchor_ep}, frame {meta['anchor']['frame']}, {meta['args'].get('mode', 'rigid')} mode; "
                     f"scene 2 = dr_level {sc2.get('dr_level', '?')} re-draw, seed {sc2.get('seed', '?')})",
                     x=0.075, y=0.975, ha="left", fontsize=11.5)
        fig.savefig(out / "1b_latent_vs_camera_other_scene.png", dpi=140)
        plt.close(fig)
        summary["scene2"] = dict(
            scene1=sc1, scene2=sc2, view=sp, scene_change_alone=scene_only, shape_and_offset=stats,
            sweeps={a: dict(values=val[sweep2(a)].round(4).tolist(),
                            **{g: D[sp][g][sweep2(a)].round(5).tolist() for g in ("img", "all")}) for a in AXES})

    # ======================================================================== 2. the recorded episode
    if len(traj):
        t, tcf = val[traj] / 50.0, val[cf] / 50.0
        ph_traj = phase[traj]
        panels = [("image tokens\n1 - cos, content view", "img"), ("all tokens\n1 - cos, content view", "all"),
                  ("distance to\nthe box (m)", "dist"), ("head pitch vs\nanchor, + = down (deg)", "pitch")]
        if "policy" in probe:
            panels.append(("policy\nP(manipulation)", "policy"))
        fig, axs = plt.subplots(len(panels), 1, figsize=(12, 1.75 * len(panels) + 1.0), sharex=True)
        for k, (ax, (title, key)) in enumerate(zip(axs, panels)):
            shade_phases(ax, t, ph_traj, label=k == 0)
            if key in ("img", "all"):
                metric, color = D["content"][key], (C_IMG if key == "img" else C_ALL)
                ax.axhspan(0, floor["content"][key], color=FLOOR_FILL, lw=0, zorder=0.5)
                items = [(t[-1], metric[traj][-1], "recorded", color)]
                if len(cf):
                    ax.plot(tcf, metric[cf], color=C_CF, lw=1.5, zorder=2)
                    items.append((tcf[-1], metric[cf][-1], "same head pose, arms down", C_CF))
                ax.plot(t, metric[traj], color=color, zorder=3)
                end_labels(ax, items, gap=0.12)
            elif key == "dist":
                d_pel = np.array([rows[i]["pelvis_box_xy"] for i in traj])
                items = [(t[-1], d_pel[-1], "pelvis-box, recorded", INK2), (t[-1], cam_box[traj][-1], "camera-box, recorded", MUTED)]
                ax.plot(t, d_pel, color=INK2, lw=1.5)
                ax.plot(t, cam_box[traj], color=MUTED, lw=1.5)
                if "dist" in probe and len(cf):
                    ax.plot(tcf, probe["dist"][cf], color=C_IMG, lw=1.5)
                    items.append((tcf[-1], probe["dist"][cf][-1], "camera-box, VLM probe (held out)", C_IMG))
                end_labels(ax, items)
            elif key == "pitch":
                ax.plot(t, [rows[i]["cam_rel"]["pitch"] for i in traj], color=INK2, lw=1.5)
            else:
                ax.axhline(0.5, color=AXIS, lw=0.8)
                ax.set_ylim(-0.03, 1.03)
                lines = [(t, probe["policy"][traj], "recorded image + recorded state", C_POLICY, 2.0)]
                if "policy_swap" in probe:
                    lines.append((t, probe["policy_swap"][traj], "recorded image + anchor state", "#eda100", 1.6))
                    if len(cf):
                        lines.append((tcf, probe["policy_swap"][cf], "arms-down image + recorded state", "#e87ba4", 1.6))
                if len(cf):
                    lines.append((tcf, probe["policy"][cf], "arms-down image + anchor state", C_CF, 1.5))
                for x_, y_, lab, c_, lw_ in lines[::-1]:
                    ax.plot(x_, y_, color=c_, lw=lw_, label=lab)
                handles, labels = ax.get_legend_handles_labels()
                ax.legend(handles[::-1], labels[::-1], loc="center right", fontsize=7.5, title="policy input",
                          title_fontsize=7.5)
            ax.set_ylabel(title, fontsize=8)
            ax.set_xlim(t[0], t[-1] * 1.36)
        axs[-1].set_xlabel("time in the recorded episode (s)")
        fig.suptitle(f"Recorded episode {anchor_ep} re-rendered frame by frame, and the same head poses with the "
                     "arms held down", x=0.07, ha="left", fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        fig.savefig(out / "2_real_episode.png", dpi=140)
        plt.close(fig)
        first = lambda p: next((float(t[i]) for i, x in enumerate(ph_traj) if x == p), None)
        summary["recorded_episode"] = dict(first_arrived_s=first("stand_arrived"), first_bend_s=first("bend_reach"))

    # ======================================================================== 3. token maps
    picks = [("anchor", a0)]
    for name, idx_set in (("recorded arrival", traj), ("arrival pose, arms down", cf), ("recorded bend", traj)):
        ph_name = "bend_reach" if "bend" in name else "stand_arrived"
        sel = [i for i in idx_set if phase[i] == ph_name]
        if sel:
            picks.append((f"{name} t={val[sel[len(sel) // 2]] / 50:.1f}s", sel[len(sel) // 2]))
    for a in AXES:
        idx = sweep(a)
        picks += [(f"{a} {fmt(a, val[idx[0]])}", idx[0]), (f"{a} {fmt(a, val[idx[-1]])}", idx[-1])]
    pairs = 4
    rws = int(np.ceil(len(picks) / pairs))
    fig, axs = plt.subplots(rws, 2 * pairs, figsize=(17, 1.75 * rws + 1.1),
                            gridspec_kw=dict(width_ratios=[1.6, 1] * pairs))
    novel = z["tok_novel_content"] if "tok_novel_content" in z else tokc[:, is_img]
    vals_ = novel[[i for _, i in picks if i != a0]]
    vmin, vmax = float(np.percentile(vals_, 2)), float(np.percentile(vals_, 99))
    for k, (title, i) in enumerate(picks):
        ax_im, ax_hm = axs.flat[2 * k], axs.flat[2 * k + 1]
        ax_im.imshow(Image.open(renders / rows[i]["file"]).convert("RGB"))
        ax_im.set_title(f"{title}", fontsize=8, loc="left")
        hm = ax_hm.imshow(novel[i].reshape(gh, gw), cmap=SEQ, vmin=vmin, vmax=vmax, interpolation="nearest",
                          aspect="auto")
        ax_hm.set_title(f"novelty {novel[i].mean():.2f}", fontsize=7.5, loc="left", color=INK2)
        for ax in (ax_im, ax_hm):
            ax.set_axis_off()
    for ax in list(axs.flat)[2 * len(picks):]:
        ax.set_axis_off()
    cax = fig.add_axes((0.3, 0.045, 0.4, 0.012))
    cb = fig.colorbar(hm, cax=cax, orientation="horizontal")
    cb.set_label("token novelty: 1 - cos to the nearest image token of the anchor view (content view, last layer)",
                 fontsize=8, color=INK2)
    cb.outline.set_visible(False)
    fig.suptitle("What the latent holds that the start view does not: each heatmap cell is one image token, laid out "
                 "like the image (8 x 10 tokens, 32 x 32 px each of the 256 x 320 VLM input)", x=0.01, ha="left",
                 fontsize=11)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.9, bottom=0.1, wspace=0.06, hspace=0.3)
    fig.savefig(out / "3_token_maps.png", dpi=130)
    plt.close(fig)

    # ======================================================================== 4. layers
    L = z["tok_cos_content"].shape[1]
    lay = {"image tokens": z["tok_cos_content"][:, :, is_img].mean(2),
           "instruction + template tokens": z["tok_cos_content"][:, :, is_txt].mean(2)}
    fig, axs = plt.subplots(2, 6, figsize=(16, 6.4), sharey=True)
    vmax = float(np.percentile(np.concatenate([v[grp == "sweep"].ravel() for v in lay.values()]), 99))
    for k, a in enumerate(AXES):
        idx = sweep(a)
        for r, (name, data) in enumerate(lay.items()):
            ax = axs[r, k]
            im = ax.imshow(data[idx].T, aspect="auto", origin="lower", cmap=SEQ, vmin=0, vmax=vmax,
                           extent=(val[idx][0], val[idx][-1], -0.5, L - 0.5), interpolation="nearest")
            ax.grid(False)
            ax.axhline(L - 1, color=C_ALL, lw=1.2)
            if r == 0:
                ax.set_title(AXIS_LABEL[a], fontsize=8.5, loc="left")
            else:
                ax.set_xlabel(UNITS[a])
            if k == 0:
                ax.set_ylabel(f"{name}\nlayer (0 = embeddings / ViT output)", fontsize=8)
    cax = fig.add_axes((0.93, 0.15, 0.01, 0.7))
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("mean per-token 1 - cos vs anchor (content view)", fontsize=8, color=INK2)
    cb.outline.set_visible(False)
    fig.suptitle("The sweeps through every VLM layer. Orange line: the last layer, the only one the action expert "
                 "reads. Text tokens see the image only through attention, so they start at 0.", x=0.03, ha="left",
                 fontsize=11)
    fig.subplots_adjust(left=0.07, right=0.91, top=0.88, bottom=0.1, wspace=0.12, hspace=0.18)
    fig.savefig(out / "4_layers.png", dpi=140)
    plt.close(fig)

    # ======================================================================== 5. switch readouts
    if "dist" in probe:
        fig, axs = plt.subplots(2, 6, figsize=(16, 6.4), sharey="row")
        for k, a in enumerate(AXES):
            idx = sweep(a)
            xs = val[idx]
            ax = axs[0, k]
            if np.isfinite(arrival_d):
                ax.axhline(arrival_d, color=MUTED, lw=0.8)
            ax.plot(xs, cam_box[idx], color=INK2, lw=1.5, label="geometric truth")
            ax.plot(xs, probe["dist"][idx], "-o", color=C_IMG, ms=3, lw=1.8, label="VLM probe")
            ax.axvline(0, color=AXIS, lw=0.8)
            ax.set_title(AXIS_LABEL[a], fontsize=8.5, loc="left")
            ax = axs[1, k]
            ax.axhline(0.5, color=AXIS, lw=0.8)
            ax.plot(xs, probe["ready"][idx], "-o", color=C_IMG, ms=3, lw=1.8, label="VLM probe P(ready)")
            if "policy" in probe:
                ax.plot(xs, probe["policy"][idx], "-o", color=C_POLICY, ms=3, lw=1.8, label="policy P(manipulation)")
            ax.set_ylim(-0.03, 1.03)
            ax.axvline(0, color=AXIS, lw=0.8)
            ax.set_xlabel(UNITS[a])
        axs[0, 0].set_ylabel("camera-box distance (m)")
        axs[1, 0].set_ylabel("probability")
        axs[0, 0].legend(loc="upper left", fontsize=7)
        axs[1, 0].legend(loc="upper left", fontsize=7)
        if np.isfinite(arrival_d):
            axs[0, -1].text(1.02, arrival_d, "recorded\narrival", transform=axs[0, -1].get_yaxis_transform(),
                            fontsize=7, color=INK2, va="center")
        acc = []
        if "dist_rmse_m" in probe:
            acc.append(f"distance probe held-out RMSE {probe['dist_rmse_m'] * 100:.1f} cm (R2 {probe['dist_r2']:.2f}), "
                       f"but slope vs forward x {probe['dist_slope_vs_forward_x']:+.2f} (truth -1)")
        if "ready_bal_acc" in probe:
            acc.append(f"ready probe held-out balanced acc {probe['ready_bal_acc']:.2f}")
        if "token_classifier_bal_acc" in probe:
            acc.append(f"token classifier held-out balanced acc {probe['token_classifier_bal_acc']:.2f}")
        fig.suptitle("Does moving the camera alone flip the locomotion -> manipulation readouts?", x=0.03,
                     ha="left", fontsize=11.5)
        fig.text(0.03, 0.91, "VLM probes: linear on the pooled image tokens (content view), trained on the other "
                 "episodes' arms-down renders.  Policy: the predicted SONIC tokens (anchor proprio state, fixed noise) "
                 "through a classifier fitted on recorded tokens.\n" + ";   ".join(acc), fontsize=8, color=INK2)
        fig.subplots_adjust(left=0.06, right=0.95, top=0.83, bottom=0.09, wspace=0.12, hspace=0.3)
        fig.savefig(out / "5_switch_readouts.png", dpi=140)
        plt.close(fig)

    # ======================================================================== 6. floor map
    gidx = np.nonzero(grp == "grid")[0]
    if len(gidx):
        gx, gy = val[gidx], np.array([rows[i]["value2"] for i in gidx])
        ux, uy = np.unique(gx), np.unique(gy)
        maps = [("image tokens: latent change", D["content"]["img"], SEQ, None),
                ("all tokens: latent change", D["content"]["all"], SEQ, None)]
        if "dist" in probe:
            maps.append(("VLM probe: camera-box distance (m)", probe["dist"], SEQ.reversed(), None))
            maps.append(("VLM probe: P(ready)", probe["ready"], DIV, (0, 1)))
        if "policy" in probe:
            maps.append(("policy: P(manipulation)", probe["policy"], DIV, (0, 1)))
        fig, axs = plt.subplots(1, len(maps), figsize=(3.3 * len(maps), 4.9))
        axs = np.atleast_1d(axs)
        bx, by = meta["anchor"]["box_in_heading"][:2]
        path = np.array([[rows[i]["cam_rel"]["x"], rows[i]["cam_rel"]["y"]] for i in traj
                         if phase[i] in LOCO + ("stand_arrived",)])
        dx, dy = (ux[1] - ux[0]) / 2, (uy[1] - uy[0]) / 2
        for ax, (title, metric, cmap, lim) in zip(axs, maps):
            M = np.full((len(ux), len(uy)), np.nan)
            for i, x, y in zip(gidx, gx, gy):
                M[np.searchsorted(ux, x), np.searchsorted(uy, y)] = metric[i]
            # top-down with the robot looking up the page: forward x up, left y to the left
            h = ax.imshow(M, origin="lower", cmap=cmap, extent=(uy[0] - dy, uy[-1] + dy, ux[0] - dx, ux[-1] + dx),
                          vmin=None if lim is None else lim[0], vmax=None if lim is None else lim[1],
                          interpolation="nearest")
            ax.grid(False)
            ax.add_patch(Rectangle((by - 0.19, bx - 0.115), 0.38, 0.23, fill=False, ec=INK, lw=1.2))
            ax.text(by, bx, "box", ha="center", va="center", fontsize=7)
            if len(path):
                ax.plot(path[:, 1], path[:, 0], color=INK, lw=1, alpha=0.8)
            ax.plot([0], [0], "^", color=INK, ms=7)
            ax.set_xlim(uy[-1] + dy, uy[0] - dy)
            ax.set_ylim(ux[0] - dx, max(ux[-1] + dx, bx + 0.15))
            ax.set_title(title, fontsize=8.5, loc="left")
            ax.set_xlabel("left y (m)")
            cb = fig.colorbar(h, ax=ax, orientation="horizontal", pad=0.15, fraction=0.05)
            cb.outline.set_visible(False)
        axs[0].set_ylabel("forward x (m)")
        fig.suptitle("Top-down: the whole robot translated over the floor, heading fixed.  Triangle = anchor, line = "
                     "the recorded head path until the robot stops", x=0.02, ha="left", fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        fig.savefig(out / "6_floor_map.png", dpi=140)
        plt.close(fig)

    # ======================================================================== 7. latent PCA
    if len(traj) >= 3:
        pca2 = PCA(FEAT[np.r_[traj, cf]], 2)
        P = pca2(FEAT, whiten=False)
        fig, axs = plt.subplots(2, 3, figsize=(13, 8.2), sharex=True, sharey=True)
        tt = val[traj] / 50.0
        for k, (ax, a) in enumerate(zip(axs.flat, AXES)):
            ax.plot(P[cf, 0], P[cf, 1], color=C_CF, lw=0.8, alpha=0.8, zorder=1)
            ax.plot(P[traj, 0], P[traj, 1], color=AXIS, lw=0.8, zorder=1)
            sc = ax.scatter(P[traj, 0], P[traj, 1], c=tt, cmap=SEQ, s=12, zorder=2, lw=0)
            idx = sweep(a)
            ax.plot(P[idx, 0], P[idx, 1], "-o", color=C_ALL, ms=3, lw=1.5, zorder=3)
            end_label(ax, P[idx[0], 0], P[idx[0], 1], fmt(a, val[idx[0]]), C_ALL)
            end_label(ax, P[idx[-1], 0], P[idx[-1], 1], fmt(a, val[idx[-1]]), C_ALL)
            ax.plot([P[a0, 0]], [P[a0, 1]], "^", color=INK, ms=8, zorder=4)
            for name in READY:
                sel = [i for i in traj if phase[i] == name]
                if sel:
                    ax.annotate(f"recorded {name.split('_')[1]}", np.median(P[sel], 0), fontsize=7, color=INK2,
                                xytext=(5, 5), textcoords="offset points")
            ax.set_title(f"sweep: {AXIS_LABEL[a]}", fontsize=8.5, loc="left")
            if k >= 3:
                ax.set_xlabel("PC 1")
            if k % 3 == 0:
                ax.set_ylabel("PC 2")
        cb = fig.colorbar(sc, ax=axs, orientation="vertical", fraction=0.02, pad=0.02)
        cb.set_label("recorded episode time (s)", fontsize=8, color=INK2)
        cb.outline.set_visible(False)
        fig.suptitle("Sweeps (orange) inside the latent space of the recorded episode\nBlue dots: recorded frames by "
                     "time; gray line: the same head poses with the arms down; triangle: anchor.  Pooled image tokens, "
                     "content view, PCA fitted on the episode.", x=0.03, ha="left", fontsize=10.5)
        fig.savefig(out / "7_latent_pca.png", dpi=140)
        plt.close(fig)

    # ======================================================================== 8. massive channels
    if "anchor_ch_absmean_img" in z:
        A, Mk = z["anchor_ch_absmean_img"], z["massive"]
        fig, axs = plt.subplots(1, 2, figsize=(13, 3.8), gridspec_kw=dict(width_ratios=[2, 1]))
        ax = axs[0]
        ch = np.arange(A.shape[1])
        ax.vlines(ch[~Mk[-1]], 0, A[-1][~Mk[-1]], color=AXIS, lw=0.6)
        ax.vlines(ch[Mk[-1]], 0, A[-1][Mk[-1]], color=C_ALL, lw=2)
        for c in ch[Mk[-1]]:
            ax.annotate(str(c), (c, A[-1][c]), xytext=(3, 2), textcoords="offset points", fontsize=7, color=INK2)
        ax.set_yscale("log")
        ax.set_ylim(1, A[-1].max() * 2)
        ax.set_xlabel("hidden channel")
        ax.set_ylabel("mean |h| over the anchor's\nimage tokens, last layer")
        ax.set_title(f"{int(Mk[-1].sum())} massive channels (orange) vs a median of {np.median(A[-1]):.0f}", loc="left")
        ax = axs[1]
        share = [(A[l][Mk[l]] ** 2).sum() / (A[l] ** 2).sum() for l in range(len(A))]
        ax.plot(np.arange(len(A)), share, "-o", color=C_ALL, ms=3)
        ax.set_ylim(0, 1)
        ax.set_xlabel("layer")
        ax.set_ylabel("share of image-token energy\nin massive channels")
        ax.set_title("per layer", loc="left")
        fig.suptitle("Why the raw latent is misleading: a few channels carry almost all of the norm and barely depend "
                     "on the image", x=0.02, ha="left", fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, 0.92))
        fig.savefig(out / "8_massive_channels.png", dpi=140)
        plt.close(fig)

    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    figs = sorted(p.name for p in out.glob("*.png"))
    brief = {k: v for k, v in summary.items() if k != "sweeps"}
    if "scene2" in brief:
        brief["scene2"] = {k: v for k, v in brief["scene2"].items() if k != "sweeps"}
    (out / "index.html").write_text(
        "<!doctype html><meta charset=utf-8><title>VLM camera sweep</title>"
        "<style>body{font-family:system-ui,sans-serif;background:#fcfcfb;color:#0b0b0b;margin:24px;max-width:1500px}"
        "img{width:100%;border:1px solid #e1e0d9;margin:8px 0 28px}pre{background:#f4f3ef;padding:12px;overflow:auto}"
        "@media (prefers-color-scheme: dark){body{background:#1a1a19;color:#fff}pre{background:#2c2c2a}}</style>"
        f"<h1>VLM camera sweep</h1><p>{renders}</p>"
        + "".join(f"<h2>{f[:-4]}</h2><img src='{f}'>" for f in figs)
        + f"<h2>summary.json</h2><pre>{json.dumps(brief, indent=2)}</pre>")
    print(f"wrote {len(figs)} figures to {out}")


if __name__ == "__main__":
    main()
