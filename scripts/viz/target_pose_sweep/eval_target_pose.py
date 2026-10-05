"""Closed-loop success of the served Psi0 checkpoint with the task's target object moved around the robot.

Runs in SIMPLE's venv, from the SIMPLE root. The episode loop is SIMPLE's own Teleop eval,
``simple.cli.eval_decoupled_wbc._run_eval_worker`` with the ``psi0_decoupled_wbc`` agent, unchanged: the same reset,
the same stabilization, the same step budget (the task's max_episode_steps unless --budget-steps) and the same success
test (``task.check_success``, e.g. the faucet handle turned past 0.7 rad and held). What changes is the scene each
episode starts from, and the bookkeeping:

  every trial in plan.json (see prepare_target_sweep.py) becomes one "virtual episode": the recorded layout of its
  base scene (robot spawn, room, table, which objects) with
    - the target object put at the trial's pose relative to the robot (the robot starts where the scene has it),
    - distractors, materials (table, floor, robot, objects) and lighting dropped from the recorded state, so SIMPLE's
      domain randomization draws them afresh, seeded per trial. This is what SIMPLE's highest replay level
      (load_state_dict dr_level=2) re-draws; the room and camera are never re-drawn by SIMPLE.

Output (--out):
  results.jsonl       one line per finished trial, written as it finishes; re-running skips the trials in it
  traces/<label>.json 10 Hz trace: pelvis x/y/z/yaw, both hands, target, articulated joint travel
  videos/<label>/     SIMPLE's head-camera videos of the trial
  conditions.json     the run's settings and the policy server's /info
"""

from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import time
import traceback
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import simple.cli.eval_decoupled_wbc as E
import simple.datasets.lerobot as SL
import simple.dr.manager as M
from simple.baselines.psi0_decoupled_wbc import Psi0DecoupledWbcAgent
from simple.dr.types import Box
from simple.envs.sonic_loco_manip import SonicLocoManipEnv

TRACE_EVERY = 5       # control steps (50 Hz) between trace rows
REACH_DIST = 0.10     # m, a hand this close to the target counts as having reached it
MOVED_RAD = 0.10      # articulated joint travel that counts as having moved it
LIFT_DZ = 0.03        # target raised this much counts as lifted
PUSH_DXY = 0.05       # target moved this much over the table counts as moved
FALL_Z = 0.40         # pelvis height below this = fell
HAND_BODIES = ("hand_index_1_link", "hand_middle_1_link", "hand_thumb_2_link")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True, help="the sweep directory, holding plan.json")
    p.add_argument("--trials", type=int, default=0, help="run only the first N trials of the plan (0 = all)")
    p.add_argument("--max-new", type=int, default=0,
                   help="stop after this many trials (0 = none); the runner restarts the process for the rest, so a "
                        "long sweep never runs in one Isaac process for days")
    p.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--save-video", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--budget-steps", type=int, default=0, help="override the task's step budget (0 = the task's)")
    p.add_argument("--instruction", default="", help="override the instruction sent to the policy")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8015)
    return p.parse_args()


def yaw_of(q_wxyz) -> float:
    w, x, y, z = q_wxyz
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def quat_z(yaw: float) -> list[float]:
    return [math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]


def to_robot_frame(robot_xy, robot_yaw, xy) -> list[float]:
    c, s = math.cos(robot_yaw), math.sin(robot_yaw)
    dx, dy = xy[0] - robot_xy[0], xy[1] - robot_xy[1]
    return [c * dx + s * dy, -s * dx + c * dy]


def from_robot_frame(robot_xy, robot_yaw, rel) -> list[float]:
    c, s = math.cos(robot_yaw), math.sin(robot_yaw)
    return [robot_xy[0] + c * rel[0] - s * rel[1], robot_xy[1] + s * rel[0] + c * rel[1]]


def virtual_episode(base: dict, item: dict, st: dict) -> dict:
    """The base scene's environment_config, set up for one trial (read back by load_state_dict below)."""
    conf = deepcopy(base)
    ds = conf["dr_state_dict"]
    for k in ("distractors", "material", "lighting"):  # re-drawn by SIMPLE's DR, as at replay level 2
        ds.pop(k, None)
    robot_uid = conf["robot_cfg"]["uid"]
    robot = deepcopy(ds["spatial"][robot_uid])
    tgt = ds["spatial"].get(st["target"])
    ryaw = yaw_of(robot["quaternion"])
    txy = from_robot_frame(robot["position"][:2], ryaw, [item["dx"], item["dy"]])
    dyaw = math.radians(item["dyaw"])
    spatial = {robot_uid: robot}
    sweep = dict(seed=item["seed"], kind=st["kind"], xy=txy, yaw=ryaw + dyaw, robot=[*robot["position"][:2], ryaw])
    if st["kind"] == "articulated":
        z = tgt["position"][2] if tgt is not None else 0.0
        spatial[st["target"]] = {"position": [txy[0], txy[1], z], "quaternion": quat_z(ryaw + dyaw)}
    ds["spatial"] = spatial  # the rest (the clutter "target", distractors) is placed by SpatialDR afresh
    conf["_sweep"] = sweep
    return conf


def outcome(r: dict) -> str:
    if r["success"]:
        return "success"
    if r["fell"]:
        return "fell"
    if r["moved"]:
        return "moved_not_done"
    if r["reached"]:
        return "reached_not_moved"
    return "not_reached"


@dataclass
class Running:
    item: dict
    started: float
    planned_xy: list
    spawn: list  # x, y, yaw the scene spawns the robot at: the frame of every relative pose, as in prepare
    policy_started: bool = False
    steps: int = 0
    hand_bodies: dict = field(default_factory=dict)
    art_joints: list = field(default_factory=list)
    art_bodies: list = field(default_factory=list)
    q0: np.ndarray | None = None
    target0: np.ndarray | None = None
    robot0: list | None = None
    min_hand: float = np.inf
    min_hand_step: int | None = None
    min_hand_point: list | None = None
    max_dq: float = 0.0
    max_lift: float = 0.0
    max_push: float = 0.0
    min_pelvis_z: float = np.inf
    trace: list = field(default_factory=list)
    placed: dict | None = None


def main() -> None:
    args = parse_args()
    out = Path(args.out).resolve()
    plan = json.loads((out / "plan.json").read_text())
    st = plan["settings"]
    trials = plan["trials"][:args.trials] if args.trials > 0 else plan["trials"]
    (out / "traces").mkdir(exist_ok=True)
    (out / "videos").mkdir(exist_ok=True)

    scenes_dir = Path(st["base_scenes"])
    base = [json.loads(json.loads(line)["environment_config"])
            for line in (scenes_dir / "meta" / "episodes.jsonl").read_text().splitlines() if line.strip()]
    fps = json.loads((scenes_dir / "meta" / "info.json").read_text()).get("fps", 50)

    results_path = out / "results.jsonl"
    done = set()
    if results_path.exists():
        done = {json.loads(line)["trial"] for line in results_path.read_text().splitlines() if line.strip()}
    todo = [dict(t, label=f"t{t['trial']:03d}") for t in trials if t["trial"] not in done]
    print(f"[sweep] {len(trials)} trials, {len(trials) - len(todo)} already in {results_path.name}, "
          f"{len(todo)} to go" + (f", {min(args.max_new, len(todo))} in this process"
                                  if 0 < args.max_new < len(todo) else ""), flush=True)
    n_before = len(trials) - len(todo)
    if args.max_new > 0:
        todo = todo[:args.max_new]
    if not todo:
        return
    confs = [virtual_episode(base[t["base_episode"]], t, st) for t in todo]

    server_info = E._fetch_policy_info(args.host, args.port)
    (out / "conditions.json").write_text(json.dumps(dict(settings=st, args=vars(args), server_info=server_info,
                                                         thresholds=dict(reach=REACH_DIST, moved_rad=MOVED_RAD,
                                                                         lift=LIFT_DZ, push=PUSH_DXY, fall=FALL_Z)),
                                                    indent=1))

    # ------------------------------------------------------------------ hooks into SIMPLE's eval loop
    state: dict = {"cur": None, "env": None, "n_done": 0, "t0": time.perf_counter()}

    class _Plan:  # stands in for the LeRobotDataset the loop opens: one "episode" per trial to run
        def __init__(self, *a, **kw):
            self.num_episodes = len(todo)
            self.meta = SimpleNamespace(fps=fps)

    import lerobot.datasets.lerobot_dataset as LD
    LD.LeRobotDataset = _Plan

    def get_episode(_ds, idx, data_format=None):
        sw = confs[idx]["_sweep"]
        state["cur"] = Running(item=todo[idx], started=time.perf_counter(), planned_xy=sw["xy"], spawn=sw["robot"])
        return confs[idx], None

    SL.get_episode_lerobot = get_episode

    orig_load = M.DRManager.load_state_dict

    def load_state_dict(self, state_dict, dr_level=None):
        sweep = state_dict.get("_sweep")
        if sweep is not None:
            random.seed(sweep["seed"])
            np.random.seed(sweep["seed"] % (2**32))
            if sweep["kind"] == "target":  # placed by SpatialDR through its collision check, at exactly this pose
                sp = self.get_randomizer("spatial")
                x, y = sweep["xy"]
                sp.cfg.target_region = Box(low=[x, y], high=[x, y])
                sp.cfg.target_rotate_z = Box(low=sweep["yaw"], high=sweep["yaw"])
        return orig_load(self, state_dict, dr_level=dr_level)

    M.DRManager.load_state_dict = load_state_dict

    orig_get_action = Psi0DecoupledWbcAgent.get_action

    def get_action(self, observation, instruction=None, **kw):
        cur = state["cur"]
        if cur is not None and not cur.policy_started:
            cur.policy_started = True
            try:
                _start(cur)
            except Exception:
                traceback.print_exc()
        return orig_get_action(self, observation, instruction=args.instruction or instruction, **kw)

    Psi0DecoupledWbcAgent.get_action = get_action

    def _start(cur: Running) -> None:
        env = state["env"]
        m, d = env.mjModel, env.mjData
        for side in ("left", "right"):
            ids = []
            for b in HAND_BODIES:
                try:
                    ids.append(m.body(f"{side}_{b}").id)
                except KeyError:
                    pass
            if not ids:
                ids = [m.body(f"{side}_wrist_yaw_link").id]
            cur.hand_bodies[side] = ids
        cur.art_joints = [i for i in range(m.njnt) if m.joint(i).name.startswith("articulate_joint")]
        cur.art_bodies = sorted({int(m.jnt_bodyid[i]) for i in cur.art_joints})
        cur.q0 = np.array([d.qpos[m.jnt_qposadr[i]] for i in cur.art_joints])
        task = env.task
        pelvis = d.body("pelvis")
        cur.robot0 = list(cur.spawn)
        robot_xy, robot_yaw = cur.spawn[:2], cur.spawn[2]
        actor = task.layout.actors.get("articulated" if st["kind"] == "articulated" else "target")
        pos = list(actor.pose.position) if actor is not None else [np.nan] * 3
        if st["kind"] == "articulated":
            try:
                pos = d.body("articulate_base").xpos.tolist()
            except KeyError:
                pass
        quat = list(actor.pose.quaternion) if actor is not None else [1, 0, 0, 0]
        rel = to_robot_frame(robot_xy, robot_yaw, pos[:2])
        cur.placed = dict(world=[round(v, 4) for v in pos], rel=[round(v, 4) for v in rel],
                          dyaw=round(math.degrees(yaw_of(quat) - robot_yaw), 2),
                          robot=[round(v, 4) for v in cur.robot0],
                          pelvis_at_start=[*np.round(pelvis.xpos[:2], 4).tolist(), round(yaw_of(pelvis.xquat), 4)],
                          instruction=task.instruction)
        cur.target0 = np.asarray(pos, dtype=float)
        err = math.hypot(pos[0] - cur.planned_xy[0], pos[1] - cur.planned_xy[1])
        if err > 0.01:
            print(f"[sweep] WARNING {cur.item['label']}: target placed {err * 100:.1f} cm off the plan", flush=True)

    orig_step = SonicLocoManipEnv.step

    def step(self, action):
        res = orig_step(self, action)
        state["env"] = self
        cur = state["cur"]
        if cur is not None and cur.policy_started and cur.robot0 is not None:
            try:
                _record(self, cur, res[-1])
            except Exception:  # never let bookkeeping kill the eval
                traceback.print_exc()
        return res

    def _reset_hook(self, *a, **kw):
        state["env"] = self
        return orig_reset(self, *a, **kw)

    orig_reset = SonicLocoManipEnv.reset
    SonicLocoManipEnv.reset = _reset_hook
    SonicLocoManipEnv.step = step

    def _record(env, cur: Running, info: dict) -> None:
        m, d = env.mjModel, env.mjData
        cur.steps += 1
        hands = {s: d.xpos[ids].mean(axis=0) for s, ids in cur.hand_bodies.items()}
        if st["kind"] == "articulated":
            pts = d.xpos[cur.art_bodies] if cur.art_bodies else cur.target0[None]
            q = np.array([d.qpos[m.jnt_qposadr[i]] for i in cur.art_joints])
            if len(q):
                cur.max_dq = max(cur.max_dq, float(np.abs(q - cur.q0).max()))
            tgt = cur.target0
        else:
            tgt = np.asarray(info.get("target", cur.target0), dtype=float)[:3]
            pts = tgt[None]
            cur.max_lift = max(cur.max_lift, float(tgt[2] - cur.target0[2]))
            cur.max_push = max(cur.max_push, float(np.linalg.norm(tgt[:2] - cur.target0[:2])))
        for h in hands.values():
            dist = float(np.linalg.norm(pts - h, axis=1).min())
            if dist < cur.min_hand:
                cur.min_hand, cur.min_hand_step = dist, cur.steps
                cur.min_hand_point = h.round(4).tolist()
        pelvis = d.body("pelvis")
        cur.min_pelvis_z = min(cur.min_pelvis_z, float(pelvis.xpos[2]))
        if cur.steps % TRACE_EVERY == 1:
            cur.trace.append([cur.steps, *np.round(pelvis.xpos, 4).tolist(), round(math.degrees(yaw_of(pelvis.xquat)), 2),
                              *np.round(hands["left"], 4).tolist(), *np.round(hands["right"], 4).tolist(),
                              *np.round(tgt, 4).tolist(), round(cur.max_dq, 4)])

    def append_stats(eval_dir, line):
        E_append(eval_dir, line)
        if line.startswith("episode_"):
            try:
                _finish()
            except Exception:
                print("[sweep] WARNING: could not record this trial:", flush=True)
                traceback.print_exc()

    E_append = E._append_eval_stats_line
    E._append_eval_stats_line = append_stats

    def _finish() -> None:
        cur, state["cur"] = state["cur"], None
        env = state["env"]
        item = cur.item
        success = bool(getattr(env, "_success", False))
        robot0 = cur.robot0 or [np.nan] * 3
        closest = None
        if cur.min_hand_point is not None:
            closest = dict(world=cur.min_hand_point,
                           rel=[round(v, 4) for v in to_robot_frame(robot0[:2], robot0[2], cur.min_hand_point[:2])])
        last = cur.trace[-1] if cur.trace else None
        rec = dict(
            trial=item["trial"], label=item["label"], base_episode=item["base_episode"], seed=item["seed"],
            point=item.get("point"), grid=item.get("grid"), rep=item.get("rep"),
            in_train_range=item["in_train_range"], plan=dict(dx=item["dx"], dy=item["dy"], dyaw=item["dyaw"]),
            placed=cur.placed, success=success, steps=cur.steps, budget_steps=args.budget_steps or None,
            min_hand_dist=None if not np.isfinite(cur.min_hand) else round(cur.min_hand, 4),
            min_hand_step=cur.min_hand_step, closest_hand=closest,
            reached=bool(cur.min_hand < REACH_DIST), max_joint_travel=round(cur.max_dq, 4),
            max_lift=round(cur.max_lift, 4), max_push=round(cur.max_push, 4),
            moved=bool(cur.max_dq > MOVED_RAD or cur.max_lift > LIFT_DZ or cur.max_push > PUSH_DXY),
            fell=bool(cur.min_pelvis_z < FALL_Z), min_pelvis_z=None if not np.isfinite(cur.min_pelvis_z)
            else round(cur.min_pelvis_z, 4),
            final_pelvis_rel=None if last is None else [round(v, 4) for v in to_robot_frame(robot0[:2], robot0[2],
                                                                                            last[1:3])],
            video_dir=f"videos/{item['label']}" if args.save_video else None,
            wall_seconds=round(time.perf_counter() - cur.started, 1),
        )
        rec["outcome"] = outcome(rec)
        (out / "traces" / f"{item['label']}.json").write_text(json.dumps(dict(
            columns=["step", "x", "y", "z", "yaw", "lh_x", "lh_y", "lh_z", "rh_x", "rh_y", "rh_z",
                     "t_x", "t_y", "t_z", "joint_travel"], rows=cur.trace)))
        with open(results_path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        state["n_done"] += 1
        state["last_label"] = item["label"]
        el = time.perf_counter() - state["t0"]
        n_all = n_before + state["n_done"]  # the whole sweep's progress, over every process
        eta = el / state["n_done"] * (len(trials) - n_all)
        eta = f"{eta / 3600:.1f} h" if eta > 5400 else f"{eta / 60:.0f} min"
        pr = cur.placed or {"rel": [item["dx"], item["dy"]]}
        print(f"[sweep] {n_all}/{len(trials)} {item['label']} target dx {pr['rel'][0]:+.2f} dy "
              f"{pr['rel'][1]:+.2f} ({'in' if item['in_train_range'] else 'outside'} training range): "
              f"{rec['outcome']} ({cur.steps} steps, closest hand {rec['min_hand_dist'] or float('nan'):.2f} m, "
              f"joint travel {cur.max_dq:.2f} rad, {rec['wall_seconds']:.0f} s wall)  eta {eta}",
              flush=True)

    class Recorder(E.VideoRecorder):
        def release(self):
            was = self._is_released
            super().release()
            label = state.get("last_label")
            if was or not label:
                return
            src = Path(self.work_dir) / self.name_prefix
            if src.is_dir():
                dst = out / "videos" / label
                shutil.rmtree(dst, ignore_errors=True)
                shutil.move(str(src), str(dst))

    E.VideoRecorder = Recorder

    E._run_eval_worker(
        env_id=f"simple/{st['task']}", policy="psi0_decoupled_wbc", split="train", host=args.host, port=args.port,
        data_format="lerobot", sim_mode="mujoco_isaac", headless=args.headless, eval_dir=str(out / "eval"),
        max_episode_steps=args.budget_steps or None, num_episodes=len(todo), episode_start=0,
        data_dir=str(scenes_dir), rollout_save_dir=None, success_criteria=None, save_video=args.save_video)
    print(f"[sweep] finished {state['n_done']} trials in {(time.perf_counter() - state['t0']) / 60:.1f} min; "
          f"results: {results_path}", flush=True)


if __name__ == "__main__":
    main()
