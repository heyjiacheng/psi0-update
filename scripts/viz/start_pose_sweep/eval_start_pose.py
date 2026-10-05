"""Closed-loop success rate of the served Psi0 checkpoint when the robot starts somewhere else.

Runs in SIMPLE's venv, started by SIMPLE's ``scripts/run_eval_wbc.sh`` through its ``EVAL_WBC_WRAPPER`` hook
(that script brings the lockstep SONIC controller up and down). The evaluation is SIMPLE's own
``simple.evals.sonic_wbc.run_sonic_wbc_eval``, unchanged: the same reset, the same 152-step controller start-up,
the same budget (max(2 x recorded length, 1500) steps at 50 Hz) and the same success test (the box resting on the
table top with the hands off it for ~1 s). Only the start pose differs.

How the start pose is moved: every (offset, recorded episode, repeat) becomes one "virtual episode". Its recorded
frame 0, which the eval writes into the sim as the start state, gets the base shifted by (dx, dy) and turned by
dyaw about the pelvis; its saved layout gets the same spawn shift. Joints, box, table, lighting and materials stay
the recorded episode's. All virtual episodes run in one Isaac session against one policy server.

Offsets are in the world frame, which is the robot's recorded start heading (0-5 deg yaw in every episode):
  dx    forward, toward the box and the table (the recorded start is ~0.60 m pelvis to box center, and the robot
        walks ~0.25 m before it bends; the room's back wall is ~0.3 m behind the start)
  dy    to the robot's left
  dyaw  turn left, degrees, about the pelvis

Output (--out):
  results.jsonl       one line per finished episode, written as it finishes, so a crash keeps what already ran;
                      re-running the same command skips the (offset, episode, repeat) already in it
  traces/<label>.json 5 Hz trace: pelvis x/y/z/yaw, box x/y/z, reward
  videos/<label>/     SIMPLE's per-episode head-camera videos
  conditions.json     the offsets and the run's settings, including the policy server's /info
"""

from __future__ import annotations

import argparse
import json
import time
import traceback
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

import simple.evals.sonic_wbc as W
from simple.agents.sonic_eval_agent import Psi0HttpPolicy, ReplayChunkPolicy, SonicEvalAgent

# 1-D sweeps through the start pose, one DoF at a time, around the recorded start (always included)
PRESETS: dict[str, list[tuple[float, float, float]]] = {
    "axes": [(0.0, 0.0, 0.0),
             (-0.20, 0.0, 0.0), (-0.10, 0.0, 0.0), (0.15, 0.0, 0.0),
             (0.0, -0.30, 0.0), (0.0, -0.15, 0.0), (0.0, 0.15, 0.0), (0.0, 0.30, 0.0),
             (0.0, 0.0, -30.0), (0.0, 0.0, -15.0), (0.0, 0.0, 15.0), (0.0, 0.0, 30.0)],
    # the floor around the start, heading kept
    "grid": [(x, y, 0.0) for x in (-0.15, 0.0, 0.15) for y in (-0.30, -0.15, 0.0, 0.15, 0.30)],
}
PRESETS["axes+grid"] = list(dict.fromkeys(PRESETS["axes"] + PRESETS["grid"]))

REACH_XY = 0.35     # pelvis-box xy; every demo gets to 0.25-0.30 m before the box leaves the floor (at 0.35-0.43 m)
LIFT_DZ = 0.05      # box raised this much above its start height
FALL_Z = 0.45       # pelvis height; the demos never go below 0.61 m, even fully bent
TRACE_EVERY = 10    # control steps (50 Hz) between trace rows
CONTACT_STEPS = 25  # look for robot-scene contacts this many steps after the start


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("env_id")
    p.add_argument("policy", help="psi0 (the policy server) or replay (the recorded actions, a sanity check)")
    p.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--data-dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--episodes", default="all", help="'all' or comma list of recorded episodes, e.g. 0,1,2")
    p.add_argument("--preset", default="axes", choices=sorted(PRESETS))
    p.add_argument("--offsets", default="",
                   help="overrides --preset: 'dx,dy,dyaw' triples separated by spaces or ';', e.g. '0,0,0 -0.2,0,0'")
    p.add_argument("--repeats", type=int, default=1, help="runs per (offset, episode); the policy samples noise")
    p.add_argument("--budget-steps", type=int, default=0, help="override the eval's step budget (0 = SIMPLE's)")
    p.add_argument("--dr-level", type=int, default=0)
    p.add_argument("--save-video", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8014)
    p.add_argument("--policy-timeout", type=float, default=60.0)
    return p.parse_args()


def parse_offsets(args) -> list[tuple[float, float, float]]:
    if not args.offsets.strip():
        return PRESETS[args.preset]
    out = []
    for tok in args.offsets.replace(";", " ").split():
        vals = [float(v) for v in tok.split(",")]
        if len(vals) != 3:
            raise SystemExit(f"--offsets: '{tok}' is not dx,dy,dyaw")
        out.append(tuple(vals))
    return list(dict.fromkeys(out))


def cond_name(dx: float, dy: float, dyaw: float) -> str:
    parts = [f"dx{dx:+.2f}" if dx else "", f"dy{dy:+.2f}" if dy else "", f"yaw{dyaw:+.0f}" if dyaw else ""]
    return "_".join(p for p in parts if p) or "start"


def key(dx, dy, dyaw, ep, rep) -> tuple:
    return (round(dx, 4), round(dy, 4), round(dyaw, 3), int(ep), int(rep))


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def yaw_deg(q_wxyz) -> float:
    w, x, y, z = q_wxyz
    return float(np.degrees(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))))


def shifted_episode(df: pd.DataFrame, dx: float, dy: float, dyaw: float) -> pd.DataFrame:
    """The recorded episode with frame 0's base pose (the eval's start state) moved."""
    a = np.radians(dyaw)
    qz = np.array([np.cos(a / 2), 0.0, 0.0, np.sin(a / 2)])
    Rz = np.array([[np.cos(a), -np.sin(a), 0.0], [np.sin(a), np.cos(a), 0.0], [0.0, 0.0, 1.0]])
    bp = np.asarray(df["observation.base_pose"].iloc[0], dtype=np.float64).copy()
    bp[:2] += (dx, dy)
    bp[3:7] = quat_mul(qz, bp[3:7])
    bv = np.asarray(df["observation.base_vel"].iloc[0], dtype=np.float64).copy()
    bv[:3] = Rz @ bv[:3]  # world-frame linear velocity turns with the robot; the angular part is body-frame
    df = df.copy()
    for col, v in (("observation.base_pose", bp), ("observation.base_vel", bv)):
        cells = list(df[col])
        cells[0] = v
        df[col] = pd.Series(cells, index=df.index, dtype=object)
    return df


def shifted_config(conf: dict, dx: float, dy: float) -> dict:
    """The saved layout with the robot spawn moved too (SpatialDR reads only its position back)."""
    conf = deepcopy(conf)
    spawn = conf.get("dr_state_dict", {}).get("spatial", {}).get(conf.get("robot_cfg", {}).get("uid", "g1_sonic"))
    if spawn is None:
        print("[sweep] WARNING: no robot spawn in the saved layout; only the start state is moved", flush=True)
    else:
        spawn["position"] = [spawn["position"][0] + dx, spawn["position"][1] + dy, *spawn["position"][2:]]
    return conf


@dataclass
class Running:
    label: str
    item: dict
    started: float
    steps: int = 0
    min_pelvis_box_xy: float = np.inf
    min_pelvis_z: float = np.inf
    max_box_z: float = -np.inf
    reach_step: int | None = None
    lift_step: int | None = None
    start_contacts: set | None = None
    box_z0: float | None = None
    trace: list | None = None


def main() -> None:
    args = parse_args()
    out = Path(args.out).resolve()
    (out / "traces").mkdir(parents=True, exist_ok=True)
    (out / "videos").mkdir(exist_ok=True)
    eval_dir = out / "eval"

    episodes = W._load_episodes(args.data_dir)
    configs = W._load_episode_configs(args.data_dir)
    eps = sorted(e for e in episodes if e in configs) if args.episodes == "all" \
        else [int(e) for e in args.episodes.split(",")]
    offsets = parse_offsets(args)

    results_path = out / "results.jsonl"
    done = set()
    if results_path.exists():
        for line in results_path.read_text().splitlines():
            r = json.loads(line)
            done.add(key(r["dx"], r["dy"], r["dyaw"], r["episode"], r["rep"]))

    todo = []
    for ci, (dx, dy, dyaw) in enumerate(offsets):
        for rep in range(args.repeats):
            for ep in eps:
                if key(dx, dy, dyaw, ep, rep) in done:
                    continue
                name = cond_name(dx, dy, dyaw)
                label = f"{name}_ep{ep}" + (f"_r{rep}" if args.repeats > 1 else "")
                todo.append(dict(cond=name, cond_index=ci, dx=dx, dy=dy, dyaw=dyaw, episode=ep, rep=rep, label=label))
    total = len(offsets) * len(eps) * args.repeats
    print(f"[sweep] {len(offsets)} start offsets x {len(eps)} episodes x {args.repeats} repeats = {total} runs; "
          f"{total - len(todo)} already in {results_path.name}, {len(todo)} to go", flush=True)
    if not todo:
        return

    virtual_eps, virtual_cfgs = {}, {}
    for vid, item in enumerate(todo):
        item["vid"] = vid
        virtual_eps[vid] = shifted_episode(episodes[item["episode"]], item["dx"], item["dy"], item["dyaw"])
        virtual_cfgs[vid] = shifted_config(configs[item["episode"]], item["dx"], item["dy"])

    if args.policy == "psi0":
        policy = Psi0HttpPolicy(args.host, args.port, timeout=args.policy_timeout)
        server_info = policy.info
    elif args.policy == "replay":
        policy, server_info = ReplayChunkPolicy(), {}
    else:
        raise SystemExit("policy must be psi0 or replay")
    (out / "conditions.json").write_text(json.dumps(dict(
        offsets=[dict(cond=cond_name(*o), dx=o[0], dy=o[1], dyaw=o[2]) for o in offsets], episodes=eps,
        repeats=args.repeats, args=vars(args), server_info=server_info,
        thresholds=dict(reach_xy=REACH_XY, lift_dz=LIFT_DZ, fall_z=FALL_Z),
        recorded_start={str(e): dict(base_pose=np.asarray(episodes[e]["observation.base_pose"].iloc[0]).tolist(),
                                     box=np.asarray(episodes[e]["observation.object_poses"].iloc[0])[:3].tolist())
                        for e in eps},
        table=_table(configs[eps[0]])), indent=2))

    # ------------------------------------------------------------------ hooks into SIMPLE's eval loop
    state: dict[str, Running | None] = {"cur": None}
    n_done = [0]
    sweep_started = time.perf_counter()

    orig_reset = SonicEvalAgent.reset_evaluation

    def reset_evaluation(self, episode_data, episode_index):
        item = todo[episode_index]
        state["cur"] = Running(label=item["label"], item=item, started=time.perf_counter(), trace=[])
        return orig_reset(self, episode_data, episode_index)

    orig_progress = W._progress_sample

    def progress_sample(env, info):
        sample = orig_progress(env, info)
        cur = state["cur"]
        if cur is None:
            return sample
        try:
            cur.steps += 1
            pelvis = env.mjData.body("pelvis")
            p, q = np.asarray(pelvis.xpos, dtype=np.float64), np.asarray(pelvis.xquat, dtype=np.float64)
            box = np.asarray(info.get("target", [np.nan] * 3), dtype=np.float64)[:3]
            if cur.box_z0 is None:
                cur.box_z0 = float(box[2])
            d = sample["pelvis_target_xy"]
            cur.min_pelvis_box_xy = min(cur.min_pelvis_box_xy, d)
            cur.min_pelvis_z = min(cur.min_pelvis_z, float(p[2]))
            cur.max_box_z = max(cur.max_box_z, float(box[2]))
            if cur.reach_step is None and d < REACH_XY:
                cur.reach_step = cur.steps
            if cur.lift_step is None and box[2] > cur.box_z0 + LIFT_DZ:
                cur.lift_step = cur.steps
            if cur.steps <= CONTACT_STEPS:
                cur.start_contacts = (cur.start_contacts or set()) | _scene_contacts(env)
            if cur.steps % TRACE_EVERY == 1:
                cur.trace.append([cur.steps, *np.round(p, 4).tolist(), round(yaw_deg(q), 2),
                                  *np.round(box, 4).tolist(), round(sample["reward"], 3)])
        except Exception:  # never let bookkeeping kill the eval
            traceback.print_exc()
        return sample

    @dataclass
    class RecordedResult(W.EpisodeResult):
        def __post_init__(self):
            try:
                _finish(self)
            except Exception:
                print("[sweep] WARNING: could not record this episode:", flush=True)
                traceback.print_exc()

    def _finish(result) -> None:
        cur, state["cur"] = state["cur"], None
        item = todo[result.episode_index]
        assert cur is not None and cur.item is item, "episode bookkeeping out of sync"
        video_dir = None
        src = eval_dir / f"episode_{item['vid']}"
        if src.is_dir():  # the recorder released its writers just before this result was built
            dst = out / "videos" / item["label"]
            if dst.exists():
                import shutil
                shutil.rmtree(dst)
            src.rename(dst)
            video_dir = str(dst.relative_to(out))
        last = cur.trace[-1] if cur.trace else None
        rec = dict(
            **{k: item[k] for k in ("label", "cond", "dx", "dy", "dyaw", "episode", "rep")},
            **{k: v for k, v in asdict(result).items() if k != "episode_index"},
            reached=cur.reach_step is not None, reach_step=cur.reach_step,
            lifted=cur.lift_step is not None, lift_step=cur.lift_step,
            fell=bool(cur.min_pelvis_z < FALL_Z),
            min_pelvis_box_xy=_f(cur.min_pelvis_box_xy), min_pelvis_z=_f(cur.min_pelvis_z),
            max_box_z=_f(cur.max_box_z), box_z0=cur.box_z0,
            start_contacts=sorted(cur.start_contacts or []),
            final_pelvis=None if last is None else dict(x=last[1], y=last[2], z=last[3], yaw=last[4]),
            final_box=None if last is None else dict(x=last[5], y=last[6], z=last[7]),
            video_dir=video_dir, wall_seconds=time.perf_counter() - cur.started,
        )
        rec["outcome"] = outcome(rec)
        (out / "traces" / f"{item['label']}.json").write_text(json.dumps(dict(
            columns=["step", "x", "y", "z", "yaw", "box_x", "box_y", "box_z", "reward"], rows=cur.trace)))
        with open(results_path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        n_done[0] += 1
        el = time.perf_counter() - sweep_started
        eta = el / n_done[0] * (len(todo) - n_done[0])
        extra = f"  start contacts: {', '.join(rec['start_contacts'])}" if rec["start_contacts"] else ""
        print(f"[sweep] {n_done[0]}/{len(todo)} {item['label']}: {rec['outcome']} "
              f"({result.executed_steps}/{result.budget_steps} steps, closest {rec['min_pelvis_box_xy']:.2f} m, "
              f"{rec['wall_seconds']:.0f} s wall)  eta {eta / 60:.0f} min{extra}", flush=True)

    SonicEvalAgent.reset_evaluation = reset_evaluation
    W._progress_sample = progress_sample
    W.EpisodeResult = RecordedResult
    W._load_episodes = lambda _: virtual_eps
    W._load_episode_configs = lambda _: virtual_cfgs
    if args.budget_steps > 0:
        W.episode_budget_steps = lambda _name, _n: args.budget_steps

    config = W.SonicWbcEvalConfig(env_id=args.env_id, data_dir=args.data_dir, eval_dir=str(eval_dir),
                                  episode_start=0, num_episodes=len(todo), dr_level=args.dr_level,
                                  headless=args.headless, save_video=args.save_video)
    result = W.run_sonic_wbc_eval(config, policy)
    print(f"[sweep] finished {len(result.episodes)} runs in {(time.perf_counter() - sweep_started) / 60:.1f} min; "
          f"results: {results_path}", flush=True)


def outcome(r: dict) -> str:
    if r["task_success"]:
        return "success"
    if r["termination_reason"] in ("model_error", "lockstep_timeout"):
        return "error"
    if r["fell"]:
        return "fell"
    if r["lifted"]:
        return "not_placed"
    if r["reached"]:
        return "not_lifted"
    return "not_reached"


def _f(v: float) -> float | None:
    return None if not np.isfinite(v) else float(v)


def _table(conf: dict) -> dict | None:
    t = conf.get("layout", {}).get("actors", {}).get("table")
    return None if t is None else dict(position=t["pose"]["position"], size=t["size"])


def _scene_contacts(env) -> set:
    """Bodies outside the robot that it touches above the floor (a start pose inside furniture or a wall)."""
    m, d = env.mjModel, env.mjData
    root = m.body("pelvis").id
    hits = set()
    for i in range(d.ncon):
        c = d.contact[i]
        b1, b2 = int(m.geom_bodyid[c.geom1]), int(m.geom_bodyid[c.geom2])
        r1, r2 = m.body_rootid[b1] == root, m.body_rootid[b2] == root
        if r1 != r2 and c.pos[2] > 0.03:
            hits.add(m.body(b2 if r1 else b1).name or f"body{b2 if r1 else b1}")
    return hits


if __name__ == "__main__":
    main()
