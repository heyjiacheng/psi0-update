"""Stage 0 of the target-pose sweep: fetch what the task needs and draw the trial plan. Psi0 venv, CPU only.

For TASK (a SIMPLE Teleop env id such as G1WholebodyOpenFaucetTeleop-v0) it makes sure these are on disk, downloading
what is missing from Hugging Face:
  checkpoint  USC-PSI-Lab/psi-model psi0/simple-checkpoints/<task lowercased>.simple.*
              -> Psi0/.runs/finetune/psi0/simple-checkpoints/<run>   (6.25 GB; skipped with --run-dir)
  train pack  USC-PSI-Lab/psi-data simple/<TASK>.zip, only its meta/ (the per-episode environment_config, where the
              scene generator recorded the robot spawn and every object's pose)
              -> SIMPLE/data/psi-data/simple/<TASK>/meta
  eval pack   USC-PSI-Lab/psi-data simple-eval/<TASK>.zip -> SIMPLE/data/simple/<TASK>/dr-level-{0,1,2}
              (one-frame episodes: the scenes the trials start from: robot spawn, room, table, which objects)

and writes into --out:
  train_targets.json  every training episode's target pose in the robot's start frame, and their box (the
                      training range), plus the range the trials are drawn from
  plan.json           one entry per trial: the target pose relative to the robot, the base scene, a seed. Grid
                      sampling (default): every point of a --grid-step grid over the trial range, --repeats times,
                      each repeat of a point in another base scene with another seed; ordered pass by pass (every
                      point once, in shuffled order, then every point again, ...) so a partial run covers the whole
                      grid. Random sampling: --trials draws; re-running with more --trials appends to it. Existing
                      trials never change
  n_trials.txt        how many of the plan's trials this run is to finish (for the shell runner)

The target is the object the instruction is about: the articulated object in the Open*/Close*/Push* tasks (the
"target" actor there is a clutter object), the "target" actor in the pick tasks. Positions are in the robot's start
frame, i.e. relative to the
spawn pose the scene records for the robot (the pelvis settles ~5 cm behind it before the policy starts, in the demos as
in the trials): dx forward, dy to the robot's left; dyaw turns the object left.

The default trial range is the training range widened by its own depth (at least 0.10 m) towards the robot and by
--side-margin (0.10 m) to each side, then shifted --forward-shift (0.05 m) away from the robot, into the table: its far
edge sits 5 cm past the farthest demo, its near edge 5 cm short of where it would otherwise be.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import zipfile
from pathlib import Path

import numpy as np

HF_MODEL_REPO = "USC-PSI-Lab/psi-model"
HF_DATA_REPO = "USC-PSI-Lab/psi-data"
CKPT_PREFIX = "psi0/simple-checkpoints"
# Released under the old name of their training data (normalization bounds equal the train pack's stats_psi0.json)
CKPT_ALIASES = {"G1WholebodyHandoverTeleop-v0": "G1WholebodyHandover-v0"}
MIN_MARGIN = 0.10  # m; the trial range reaches at least this far closer to the robot than the training range


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", required=True, help="SIMPLE env id without 'simple/', e.g. G1WholebodyOpenFaucetTeleop-v0")
    p.add_argument("--psi0-dir", required=True)
    p.add_argument("--simple-dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--run-dir", default="", help="a local checkpoint run dir; default: the task's released one")
    p.add_argument("--ckpt-step", default="40000")
    p.add_argument("--sampling", default="grid", choices=["grid", "random"],
                   help="grid: every point of a --grid-step grid over the trial range, --repeats times; random: "
                        "--trials target positions drawn at random")
    p.add_argument("--grid-step", type=float, default=0.03, help="m, grid sampling: spacing of the grid points")
    p.add_argument("--repeats", type=int, default=5,
                   help="grid sampling: trials per grid point, each in another base scene with another seed")
    p.add_argument("--trials", type=int, default=0,
                   help="random sampling: number of trials (0 = 10); grid sampling: run only the first N trials of "
                        "the plan (0 = all)")
    p.add_argument("--in-dist", default="",
                   help="random sampling: share of trials with the target inside the training range, as a percentage "
                        "(30) or a fraction (0.3); the rest go outside it. Empty: every trial uniform over the range")
    p.add_argument("--range", default="",
                   help="'dx_lo,dx_hi,dy_lo,dy_hi' in m, robot frame; default: see --forward-shift, --side-margin")
    p.add_argument("--forward-shift", type=float, default=0.05,
                   help="m; the default trial range (the training range plus its own depth, at least 0.10 m, towards "
                        "the robot, plus --side-margin to each side) moved this far away from the robot, into the "
                        "table")
    p.add_argument("--side-margin", type=float, default=0.10,
                   help="m; the default trial range reaches this far left and right of the training range")
    p.add_argument("--yaw", default="", help="'lo,hi' target yaw in degrees, robot frame; default: the training range")
    p.add_argument("--target", default="auto", choices=["auto", "articulated", "target"])
    p.add_argument("--dr-level", type=int, default=2, help="which eval pack the base scenes come from")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


# ---------------------------------------------------------------------------------------------- fetching
def hf_api():
    from huggingface_hub import HfApi
    return HfApi()


def resolve_run_dir(args) -> Path:
    if args.run_dir:
        run = Path(args.run_dir).resolve()
        if not (run / "checkpoints" / f"ckpt_{args.ckpt_step}").is_dir():
            raise SystemExit(f"checkpoint missing: {run}/checkpoints/ckpt_{args.ckpt_step}")
        return run
    prefix = f"{CKPT_ALIASES.get(args.task, args.task).lower()}.simple."
    local_root = Path(args.psi0_dir) / ".runs" / "finetune" / CKPT_PREFIX
    local = sorted(d for d in local_root.glob(f"{prefix}*")
                   if (d / "checkpoints" / f"ckpt_{args.ckpt_step}" / "model.safetensors").is_file()
                   and (d / "run_config.json").is_file())
    if local:
        return local[-1].resolve()
    names = sorted(Path(e.path).name for e in hf_api().list_repo_tree(HF_MODEL_REPO, path_in_repo=CKPT_PREFIX))
    match = [n for n in names if n.startswith(prefix)]
    if not match:
        have = sorted({n.split(".simple.")[0] for n in names if ".simple." in n})
        raise SystemExit(f"no released Psi0 checkpoint for {args.task} on {HF_MODEL_REPO}/{CKPT_PREFIX}.\n"
                         f"Released (lowercased task names): {', '.join(have)}\nOr pass RUN_DIR=<a local run dir>.")
    name = match[-1]
    print(f"[prepare] downloading checkpoint {name} (6.25 GB) ...", flush=True)
    from huggingface_hub import snapshot_download
    snapshot_download(HF_MODEL_REPO, allow_patterns=[f"{CKPT_PREFIX}/{name}/*"],
                      local_dir=str(Path(args.psi0_dir) / ".runs" / "finetune"))
    run = (local_root / name).resolve()
    if not (run / "checkpoints" / f"ckpt_{args.ckpt_step}").is_dir():
        steps = sorted(p.name for p in (run / "checkpoints").glob("ckpt_*"))
        raise SystemExit(f"{name} has no ckpt_{args.ckpt_step}; it has {steps}")
    return run


def fetch_zip(remote: str, stage: Path) -> Path:
    from huggingface_hub import hf_hub_download
    print(f"[prepare] downloading {HF_DATA_REPO}/{remote} ...", flush=True)
    return Path(hf_hub_download(HF_DATA_REPO, remote, repo_type="dataset", local_dir=str(stage)))


def ensure_train_meta(args) -> Path:
    dst = Path(args.simple_dir) / "data" / "psi-data" / "simple" / args.task
    if (dst / "meta" / "episodes.jsonl").is_file():
        return dst
    stage = Path(args.simple_dir) / "data" / "psi-data" / ".download"
    zp = fetch_zip(f"simple/{args.task}.zip", stage)
    with zipfile.ZipFile(zp) as z:
        members = [m for m in z.namelist() if "/meta/" in m and not m.endswith("/")]
        if not members:
            raise SystemExit(f"{zp} has no meta/ directory")
        for m in members:
            rel = m.split("/meta/", 1)[1]
            (dst / "meta" / rel).parent.mkdir(parents=True, exist_ok=True)
            with z.open(m) as src, open(dst / "meta" / rel, "wb") as f:
                shutil.copyfileobj(src, f)
    shutil.rmtree(stage, ignore_errors=True)
    return dst


def ensure_eval_pack(args) -> Path:
    root = Path(args.simple_dir) / "data" / "simple" / args.task
    dst = root / f"dr-level-{args.dr_level}"
    if (dst / "meta" / "episodes.jsonl").is_file():
        return dst
    stage = Path(args.simple_dir) / "data" / "simple" / ".download"
    zp = fetch_zip(f"simple-eval/{args.task}.zip", stage)
    with zipfile.ZipFile(zp) as z:
        for m in z.namelist():
            parts = m.split("/")
            if len(parts) < 2 or not parts[1].startswith("dr-level-") or m.endswith("/"):
                continue
            out = root / "/".join(parts[1:])
            out.parent.mkdir(parents=True, exist_ok=True)
            with z.open(m) as src, open(out, "wb") as f:
                shutil.copyfileobj(src, f)
    shutil.rmtree(stage, ignore_errors=True)
    if not (dst / "meta" / "episodes.jsonl").is_file():
        have = sorted(p.name for p in root.glob("dr-level-*"))
        raise SystemExit(f"the eval pack of {args.task} has no dr-level-{args.dr_level} (it has {have})")
    return dst


# ---------------------------------------------------------------------------------------------- geometry
def yaw_of(q_wxyz) -> float:
    w, x, y, z = q_wxyz
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def to_robot_frame(robot_xy, robot_yaw, xy) -> np.ndarray:
    c, s = math.cos(robot_yaw), math.sin(robot_yaw)
    d = np.asarray(xy, dtype=float) - np.asarray(robot_xy, dtype=float)
    return np.array([c * d[0] + s * d[1], -s * d[0] + c * d[1]])


def target_key(conf: dict, kind: str) -> tuple[str, str, str]:
    """(kind, spatial-state key, name) of the object the task is about."""
    ds = conf["dr_state_dict"]
    art = ds.get("articulated")
    if kind == "auto":
        kind = "articulated" if isinstance(art, dict) and art.get("uid") is not None else "target"
    if kind == "articulated":
        if not isinstance(art, dict):
            raise SystemExit("this task has no articulated object; use TARGET=target")
        return kind, f"articulated_{art['uid']}", art.get("label") or art.get("name") or "articulated"
    tgt = ds["target"]
    return kind, str(tgt["uid"]), tgt.get("label") or tgt.get("name") or "target"


def load_confs(meta_dir: Path) -> list[dict]:
    return [json.loads(json.loads(line)["environment_config"])
            for line in (meta_dir / "episodes.jsonl").read_text().splitlines() if line.strip()]


def train_targets(confs: list[dict], kind_arg: str) -> dict:
    kind, key, name = target_key(confs[0], kind_arg)
    robot_uid = confs[0]["robot_cfg"]["uid"]
    rows, robots = [], []
    for c in confs:
        sp = c["dr_state_dict"]["spatial"]
        if key not in sp:
            continue
        r, t = sp[robot_uid], sp[key]
        ryaw = yaw_of(r["quaternion"])
        rel = to_robot_frame(r["position"][:2], ryaw, t["position"][:2])
        rows.append([float(rel[0]), float(rel[1]), math.degrees(yaw_of(t["quaternion"]) - ryaw)])
        robots.append([*r["position"][:2], ryaw])
    if not rows:
        raise SystemExit(f"no training episode has '{key}' in its recorded layout")
    rel = np.array(rows)
    cfg = confs[0]["dr_cfgs"]["spatial"]
    rot = cfg.get("articulated_rotate_z" if kind == "articulated" else "target_rotate_z") or {"low": 0, "high": 0}
    yaw_lo, yaw_hi = sorted([math.degrees(rot["low"]), math.degrees(rot["high"])])
    # The table, in the mean robot start frame, for the figure
    table = None
    t = confs[0].get("layout", {}).get("actors", {}).get("table")
    if t is not None:
        rm = np.mean(robots, axis=0)
        ctr = to_robot_frame(rm[:2], rm[2], t["pose"]["position"][:2])
        table = dict(center=ctr.tolist(), size=t["size"][:2], yaw_deg=math.degrees(yaw_of(t["pose"]["quaternion"]) - rm[2]),
                     top_z=t["pose"]["position"][2] + 0.5 * t["size"][2])
    return dict(kind=kind, key=key, name=name, robot_uid=robot_uid, n=len(rows),
                rel=rel.round(4).tolist(),
                box=dict(dx=[float(rel[:, 0].min()), float(rel[:, 0].max())],
                         dy=[float(rel[:, 1].min()), float(rel[:, 1].max())]),
                yaw_deg=[yaw_lo, yaw_hi], table=table,
                instruction=confs[0]["dr_state_dict"].get("language"))


# ---------------------------------------------------------------------------------------------- plan
def parse_frac(s: str) -> float | None:
    if not s.strip():
        return None
    v = float(s)
    v = v / 100.0 if v > 1 else v
    if not 0 <= v <= 1:
        raise SystemExit(f"IN_DIST must be a percentage or a fraction, got {s}")
    return v


def inside(box: dict, dx: float, dy: float) -> bool:
    return box["dx"][0] <= dx <= box["dx"][1] and box["dy"][0] <= dy <= box["dy"][1]


def draw_trial(i: int, seed: int, frac: float | None, box: dict, rng_box: dict, yaw: list, n_scenes: int) -> dict:
    rng = np.random.default_rng([seed, i])
    if frac is None:
        want_in = None
    else:  # the first n trials always hold round(frac * n) in-range ones, whatever n is
        want_in = round(frac * (i + 1)) > round(frac * i)
    for _ in range(100000):
        if want_in:
            dx, dy = rng.uniform(*box["dx"]), rng.uniform(*box["dy"])
        else:
            dx, dy = rng.uniform(*rng_box["dx"]), rng.uniform(*rng_box["dy"])
        if want_in is None or want_in == inside(box, dx, dy):
            break
    else:
        raise SystemExit("could not draw a target outside the training range: the range is no larger than it")
    return dict(trial=i, dx=round(float(dx), 4), dy=round(float(dy), 4),
                dyaw=round(float(rng.uniform(*yaw)), 2), in_train_range=inside(box, dx, dy),
                base_episode=int(rng.integers(n_scenes)), seed=int(rng.integers(2**31 - 1)))


def grid_axis(lo: float, hi: float, step: float) -> list[float]:
    """Grid values spaced `step` apart, centred in [lo, hi]."""
    n = int(math.floor((hi - lo) / step + 1e-6)) + 1
    off = (hi - lo - (n - 1) * step) / 2
    return [round(lo + off + i * step, 4) for i in range(n)]


def grid_trials(seed: int, box: dict, rng_box: dict, step: float, repeats: int, yaw: list, n_scenes: int) -> tuple:
    """Every grid point `repeats` times, pass by pass; the repeats of a point in distinct base scenes, other seeds."""
    gx, gy = grid_axis(*rng_box["dx"], step), grid_axis(*rng_box["dy"], step)
    points = [(ix, iy) for ix in range(len(gx)) for iy in range(len(gy))]
    trials = []
    for rep in range(repeats):
        for p in np.random.default_rng([seed, 7, rep]).permutation(len(points)):
            ix, iy = points[p]
            scenes = np.random.default_rng([seed, 11, int(p)]).permutation(n_scenes)
            rng = np.random.default_rng([seed, 13, int(p), rep])
            dx, dy = gx[ix], gy[iy]
            trials.append(dict(trial=len(trials), dx=dx, dy=dy, dyaw=round(float(rng.uniform(*yaw)), 2),
                               in_train_range=inside(box, dx, dy), base_episode=int(scenes[rep % n_scenes]),
                               seed=int(rng.integers(2**31 - 1)), point=int(p), grid=[ix, iy], rep=rep))
    return trials, dict(step=step, dx=gx, dy=gy, repeats=repeats)


def main() -> None:
    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    run_dir = resolve_run_dir(args)
    train_dir = ensure_train_meta(args)
    eval_dir = ensure_eval_pack(args)
    train_confs = load_confs(train_dir / "meta")
    base_confs = load_confs(eval_dir / "meta")
    tt = train_targets(train_confs, args.target)
    box = tt["box"]

    if args.range.strip():
        v = [float(x) for x in args.range.replace(";", ",").split(",")]
        if len(v) != 4 or v[0] >= v[1] or v[2] >= v[3]:
            raise SystemExit("RANGE must be 'dx_lo,dx_hi,dy_lo,dy_hi' with lo < hi")
        rng_box = dict(dx=v[:2], dy=v[2:])
    else:
        mx = max(box["dx"][1] - box["dx"][0], MIN_MARGIN)
        sh, my = args.forward_shift, args.side_margin
        rng_box = dict(dx=[round(box["dx"][0] - mx + sh, 3), round(box["dx"][1] + sh, 3)],
                       dy=[round(box["dy"][0] - my, 3), round(box["dy"][1] + my, 3)])
    yaw = [float(x) for x in args.yaw.split(",")] if args.yaw.strip() else tt["yaw_deg"]
    frac = parse_frac(args.in_dist)
    grid = args.sampling == "grid"
    if grid and frac is not None:
        raise SystemExit("IN_DIST applies to SAMPLING=random only")
    if grid and args.repeats > len(base_confs):
        print(f"[prepare] WARNING: {args.repeats} repeats per grid point but only {len(base_confs)} base scenes; "
              f"some repeats share a base scene (the seed, and so the re-drawn distractors, materials and lighting, "
              f"still differ)", flush=True)
    if frac is not None and frac < 1 and box["dx"][0] <= rng_box["dx"][0] and rng_box["dx"][1] <= box["dx"][1] \
            and box["dy"][0] <= rng_box["dy"][0] and rng_box["dy"][1] <= box["dy"][1]:
        raise SystemExit("RANGE lies inside the training range, so no trial can go outside it")

    settings = dict(task=args.task, run_dir=str(run_dir), ckpt_step=args.ckpt_step, target=tt["key"],
                    kind=tt["kind"], range=rng_box, train_box=box, yaw_deg=yaw, in_dist=frac,
                    dr_level=args.dr_level, base_scenes=str(eval_dir), seed=args.seed, sampling=args.sampling)
    if grid:
        settings.update(grid_step=args.grid_step, repeats=args.repeats)
    plan_path = out / "plan.json"
    trials = []
    if plan_path.exists():
        old = json.loads(plan_path.read_text())
        diff = {k: (old["settings"].get(k), v) for k, v in settings.items() if old["settings"].get(k) != v}
        if diff:
            raise SystemExit(f"{out} holds a sweep with other settings: "
                             + "; ".join(f"{k}: {a} -> {b}" for k, (a, b) in diff.items())
                             + "\nDelete it or set OUT to another directory.")
        trials = old["trials"]
    if grid:
        trials, grid_info = grid_trials(args.seed, box, rng_box, args.grid_step, args.repeats, yaw, len(base_confs))
        n_run = min(args.trials, len(trials)) if args.trials > 0 else len(trials)
        tt["grid"] = grid_info
    else:
        n_run = args.trials or 10
        for i in range(len(trials), n_run):
            trials.append(draw_trial(i, args.seed, frac, box, rng_box, yaw, len(base_confs)))
    plan_path.write_text(json.dumps(dict(settings=settings, trials=trials), indent=1))
    tt.update(range=rng_box, trial_yaw_deg=yaw)
    (out / "train_targets.json").write_text(json.dumps(tt, indent=1))

    n_in = sum(t["in_train_range"] for t in trials[:n_run])
    print(f"[prepare] {args.task}: target = {tt['name']} ({tt['key']}), {tt['n']} training episodes")
    print(f"[prepare]   training range (robot frame): dx {box['dx'][0]:+.3f}..{box['dx'][1]:+.3f} m, "
          f"dy {box['dy'][0]:+.3f}..{box['dy'][1]:+.3f} m, yaw {tt['yaw_deg'][0]:+.1f}..{tt['yaw_deg'][1]:+.1f} deg")
    print(f"[prepare]   trial range:                  dx {rng_box['dx'][0]:+.3f}..{rng_box['dx'][1]:+.3f} m, "
          f"dy {rng_box['dy'][0]:+.3f}..{rng_box['dy'][1]:+.3f} m, yaw {yaw[0]:+.1f}..{yaw[1]:+.1f} deg")
    if grid:
        g = tt["grid"]
        print(f"[prepare]   grid: {len(g['dx'])} x {len(g['dy'])} points {100 * args.grid_step:.0f} cm apart, "
              f"{args.repeats} trials each = {len(trials)} trials" + (f" (running the first {n_run})"
                                                                     if n_run < len(trials) else ""))
    print(f"[prepare]   {n_run} trials, {n_in} inside the training range; base scenes: {eval_dir}")
    print(f"[prepare]   checkpoint: {run_dir}")
    # for the shell runner
    (out / "run_dir.txt").write_text(str(run_dir) + "\n")
    (out / "n_trials.txt").write_text(f"{n_run}\n")


if __name__ == "__main__":
    sys.exit(main())
