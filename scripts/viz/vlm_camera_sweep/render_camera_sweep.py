"""Re-render the Psi0 head camera under controlled pose offsets (stage 1 of the VLM camera sweep).

Runs in SIMPLE's venv (Isaac Sim + MuJoCo). No SONIC controller and no policy server are needed:
the scene is reset exactly like ``simple.evals.sonic_wbc`` does, the robot is put into a recorded
state, and only the renderer is used.

What gets rendered (all at the native 360x640 ``head_stereo_left`` resolution, lossless PNG):

  recorded the anchor frame exactly as the dataset stores it (decoded from its video, like the training
          dataloader does). With the training data as ``--data-dir`` this is a training image, and it is
          the anchor every latent change is measured against. It is only kept when the re-render matches
          it (``--max-anchor-gap``), otherwise the first ``noise`` render is the anchor.
  noise   the anchor pose rendered N times -> how faithfully the anchor is re-rendered (render noise)
  return  the anchor again after each 1-D sweep -> path-dependence floor (renderer history)
  sweep   1-D sweeps of one camera DoF at a time around the anchor pose
  grid    2-D sweep over the floor plane (forward x lateral), camera orientation fixed
  traj    re-rendered recorded frames (the real approach -> bend -> lift) with privileged labels:
          every ``--traj-stride`` frames of the anchor episode, every ``--probe-stride`` frames of
          the other ``--traj-episodes`` (those only train the probes)
  traj_cf (rigid mode) counterfactual of each traj frame: the camera at the recorded head pose but the
          body frozen in the anchor posture. traj vs traj_cf isolates what the robot's own arms and
          hands entering the view do to the latent; traj_cf also trains the pose-only probes.
  noise2 / sweep2 / return2
          the anchor pose and the same 1-D sweeps again in a second scene: the anchor episode reset with
          ``--scene2-dr-level`` (1 = lighting, table and robot materials re-drawn, layout kept) and
          ``--scene2-seed``, bumped until the drawn table material is not used by any episode of
          ``--data-dir`` (so with the training data the second scene is not in the training set).
          Measured against the first scene's anchor, this shows whether the latent's response to camera
          motion survives a scene change.

Every render carries ``pelvis_box_xy``: for traj frames the recorded value, for everything else the
value implied by the camera pose (the pelvis carried rigidly with the camera from the anchor).

Camera offsets are applied in the anchor camera's *heading frame*: x = horizontal forward,
y = horizontal left, z = world up. That is the frame locomotion moves the robot in.
  yaw   about world up          (+ = turn left)
  pitch about the heading left  (+ = look down)
  roll  about the optical axis  (+ = left side of the image up)

Two ways to move the camera (``--mode``):
  rigid   (default) the whole robot moves with the camera: the base pose is solved so the head camera
          lands exactly on the target pose. The self-view (hands and arms at the bottom of the frame) stays
          as it is at the anchor, which is what the policy sees when locomotion moves the head.
  camera  only the camera prim moves and the body stays put. Moving back, down or pitching down puts the
          camera behind or inside the robot's own head and torso, so large offsets render the robot itself.

Renderer notes (measured on this scene):
  * ``--dr-level -1`` (default) resets to the episode's recorded scene as is (table material, lighting,
    robot shaders), exactly like simple.cli.render_decoupled_wbc, which rendered the training images.
    ``--dr-level 0`` re-draws the table material on every reset like the eval does (simple/dr/manager.py);
    ``--seed`` makes that draw reproducible.
  * The training images were rendered with RTX indirect diffuse GI on (``--gi on``, default). With it the
    re-render of a settled training frame is ~4.6/255 mean abs off the stored frame (video compression +
    render noise); with GI off it is ~2x darker-biased. GI accumulates over hundreds of frames, so a
    render also depends on where the camera was before; the ``return`` renders measure that.
  * The first ~10 frames of a recorded episode are rendered before the renderer settled (they brighten
    by ~6/255), so anchor on a later frame of the initial stand (e.g. 100).
  * Sweeps walk outward in small steps and every pose gets ``--settle`` subframes, which brings the
    out-and-back difference to ~1/255, close to the frame-to-frame noise.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import time
from pathlib import Path

import numpy as np

AXES = ("x", "y", "z", "yaw", "pitch", "roll")
CAM = "head_stereo_left"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--env-id", default="simple/G1WholebodyXMoveBendCarryBoxSonic-v0")
    p.add_argument("--data-dir", required=True)
    p.add_argument("--out", required=True, help="output directory (PNG images + index.jsonl)")
    p.add_argument("--episode", type=int, default=0, help="episode whose frame is the anchor")
    p.add_argument("--anchor-frame", type=int, default=0, help="recorded frame used as the anchor state")
    p.add_argument("--dr-level", type=int, default=-1,
                   help="-1 = the recorded scene as is (like the training renderer), 0 = re-draw the table")
    p.add_argument("--data-label", default="", help="what --data-dir is, for the figures (e.g. 'training set')")
    p.add_argument("--max-anchor-gap", type=float, default=12.0,
                   help="keep the recorded frame as the anchor only if the re-render is within this mean abs /255")
    p.add_argument("--seed", type=int, default=0, help="seeds the per-reset table material draw")
    p.add_argument("--mode", choices=("rigid", "camera"), default="rigid")
    p.add_argument("--steps", type=int, default=17, help="points per 1-D sweep")
    # the recorded approach moves the head ~0.35 m forward from the start pose to where it bends
    p.add_argument("--x-range", type=float, nargs=2, default=(-0.30, 0.45), metavar=("LO", "HI"),
                   help="forward translation range, m")
    p.add_argument("--y-range", type=float, nargs=2, default=(-0.30, 0.30), metavar=("LO", "HI"))
    p.add_argument("--z-range", type=float, nargs=2, default=(-0.30, 0.20), metavar=("LO", "HI"))
    p.add_argument("--yaw-range", type=float, nargs=2, default=(-40.0, 40.0), metavar=("LO", "HI"),
                   help="degrees")
    p.add_argument("--pitch-range", type=float, nargs=2, default=(-25.0, 40.0), metavar=("LO", "HI"))
    p.add_argument("--roll-range", type=float, nargs=2, default=(-25.0, 25.0), metavar=("LO", "HI"))
    p.add_argument("--grid-steps", type=int, default=11, help="points per side of the x-y floor grid (0 = off)")
    p.add_argument("--noise-renders", type=int, default=8, help="repeat renders of the anchor (noise floor)")
    p.add_argument("--traj-episodes", default="all",
                   help="'all', 'none', or comma list, e.g. 0,1,2 (the anchor episode is always included)")
    p.add_argument("--traj-stride", type=int, default=10, help="anchor episode: every N-th recorded frame (50 Hz)")
    p.add_argument("--probe-stride", type=int, default=20, help="other episodes: every N-th recorded frame")
    p.add_argument("--settle", type=int, default=12, help="RTX subframes rendered per pose before reading it")
    p.add_argument("--warmup", type=int, default=150, help="subframes after a reset (materials load, GI settles)")
    p.add_argument("--scene2-seed", type=int, default=100,
                   help="seed of the second scene for the sweep2 renders (-1 = no second scene)")
    p.add_argument("--scene2-dr-level", type=int, default=1,
                   help="what the second scene re-draws: 0 = table material only, 1 = + lighting")
    p.add_argument("--gi", choices=("on", "off"), default="on",
                   help="RTX indirect diffuse GI: on like the training renderer and the eval, off = no render history")
    p.add_argument("--headless", type=int, default=int(os.environ.get("HEADLESS", "1")))
    return p.parse_args()


# ----------------------------------------------------------------------------- geometry
def rot(axis: np.ndarray, deg: float) -> np.ndarray:
    import transforms3d as t3d

    return t3d.axangles.axangle2mat(axis, np.deg2rad(deg))


def heading_frame(R_cam: np.ndarray) -> np.ndarray:
    """Columns: horizontal forward, horizontal left, world up."""
    fwd = R_cam[:, 0].copy()
    fwd[2] = 0.0
    fwd /= np.linalg.norm(fwd)
    up = np.array([0.0, 0.0, 1.0])
    return np.stack([fwd, np.cross(up, fwd), up], axis=1)


def offset_pose(p0: np.ndarray, R0: np.ndarray, d: dict[str, float]) -> tuple[np.ndarray, np.ndarray]:
    H = heading_frame(R0)
    p = p0 + H @ np.array([d.get("x", 0.0), d.get("y", 0.0), d.get("z", 0.0)])
    R = rot(H[:, 2], d.get("yaw", 0.0)) @ rot(H[:, 1], d.get("pitch", 0.0)) @ R0 @ rot(np.array([1.0, 0, 0]), d.get("roll", 0.0))
    return p, R


def pose_in_heading(p0, R0, p, R) -> dict[str, float]:
    """Express a camera pose relative to the anchor, in the same coordinates the sweep uses."""
    H = heading_frame(R0)
    t = H.T @ (p - p0)
    Hc = heading_frame(R)
    yaw = np.degrees(np.arctan2(np.cross(H[:, 0], Hc[:, 0]) @ H[:, 2], H[:, 0] @ Hc[:, 0]))
    pitch = lambda Rm: np.degrees(np.arcsin(np.clip(-Rm[2, 0], -1, 1)))  # + = optical axis below horizon
    roll = lambda Rm: np.degrees(np.arctan2(Rm[2, 1], Rm[2, 2]))
    return dict(x=float(t[0]), y=float(t[1]), z=float(t[2]), yaw=float(yaw),
                pitch=float(pitch(R) - pitch(R0)), roll=float(roll(R) - roll(R0)))


def outward(lo: float, hi: float, n: int) -> list[float]:
    """Sweep values ordered 0 -> hi, then 0 -> lo, so consecutive renders stay close."""
    vals = np.linspace(lo, hi, n)
    pos = sorted(v for v in vals if v >= 0)
    neg = sorted((v for v in vals if v < 0), reverse=True)
    return [float(v) for v in pos + neg]


PHASES = ("stand_initial", "walk_to_box", "stand_arrived", "bend_reach", "lift", "carry")


def phase_labels(df) -> np.ndarray:
    """Heuristic phases from privileged sim state (docs/psi0_sonic_inference_representations.md, 7.2)."""
    smooth = lambda x: np.convolve(x, np.ones(15) / 15, mode="same")
    bp = np.stack(df["observation.base_pose"]).astype(np.float64)
    bv = np.stack(df["observation.base_vel"]).astype(np.float64)
    ob = np.stack(df["observation.object_poses"]).astype(np.float64)
    q = np.stack(df["observation.state"]).astype(np.float64)
    moving = (smooth(np.linalg.norm(bv[:, :2], axis=1)) > 0.10) | (smooth(np.abs(bv[:, 5])) > 0.15)
    lifted = ob[:, 2] > np.median(ob[:25, 2]) + 0.03
    knee = 0.5 * (q[:, 3] + q[:, 9])
    crouch = knee > np.median(knee[:25]) + 0.35
    near = np.linalg.norm(ob[:, :2] - bp[:, :2], axis=1) < 0.40
    started = np.cumsum(moving) > 0
    ph = np.full(len(df), "walk_to_box", dtype=object)  # pauses between steps count as walking
    ph[~started] = "stand_initial"
    ph[started & ~moving & ~lifted & ~crouch & near] = "stand_arrived"
    ph[~moving & ~lifted & crouch] = "bend_reach"
    ph[lifted & ~moving] = "lift"
    ph[lifted & moving] = "carry"
    return ph


# ----------------------------------------------------------------------------- sim
def make_env(args):
    import gymnasium as gym
    import tyro
    from gear_sonic.utils.mujoco_sim.configs import SimLoopConfig

    import simple.envs  # noqa: F401  (registers the env ids)

    sonic_config = tyro.cli(SimLoopConfig, config=(tyro.conf.ConsolidateSubcommandArgs,), args=[]).load_wbc_yaml()
    sonic_config["ENV_NAME"] = "simple"
    env = gym.make(
        args.env_id,
        sim_mode="mujoco_isaac",
        render_hz=50,
        physics_dt=float(sonic_config["SIMULATE_DT"]),
        headless=bool(args.headless),
        webrtc=False,
        max_episode_steps=30000,
        sonic_config=sonic_config,
    )
    return env, env.unwrapped


def main() -> None:
    args = parse_args()
    out = Path(args.out)
    (out / "img").mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    # dataset loaders import typer/click; that must happen before Isaac puts its own click on the path
    from simple.cli.render_decoupled_wbc import _load_episode_configs, _load_episodes

    env, sim = make_env(args)

    import carb
    import mujoco
    import omni.replicator.core as rep
    import transforms3d as t3d
    from PIL import Image

    episodes = _load_episodes(args.data_dir)
    configs = _load_episode_configs(args.data_dir)
    materials: dict[int, str | None] = {}
    robot = sim.task.robot
    cam = None

    def reset(ep: int, seed: int | None = None, dr_level: int | None = None) -> str | None:
        """Reset to episode ``ep``'s recorded layout; returns the table material drawn for it."""
        nonlocal cam
        seed = args.seed if seed is None else seed
        conf = configs[ep]
        conf["uid"] = sim.task.uid  # same fix-up as sonic_wbc.run_sonic_wbc_eval
        random.seed(seed + ep)
        np.random.seed(seed + ep)
        level = args.dr_level if dr_level is None else dr_level
        env.reset(seed=seed + ep, options={"state_dict": conf, "task_id": f"episode_{ep}",
                                           "dr_level": None if level < 0 else level})
        carb.settings.get_settings().set("/rtx/indirectDiffuse/enabled", args.gi == "on")
        cam = sim.isaac.cameras[CAM]  # Isaac builds its cameras on the first reset
        right = sim.isaac.cameras.get(CAM.replace("left", "right"))
        try:  # the policy only sees the left image; not rendering the right one saves ~20%
            right._render_product.hydra_texture.set_updates_enabled(False)
        except AttributeError:
            pass
        table = sim.task.layout.actors.get("table")
        return (getattr(table, "material", None) or {}).get("name")

    def recorded_frame(ep: int, frame: int) -> np.ndarray | None:
        """The frame as the dataset stores it, decoded from the episode video like the training dataloader."""
        info = json.loads((Path(args.data_dir) / "meta" / "info.json").read_text())
        keys = [k for k, v in info["features"].items() if v.get("dtype") == "video"]
        if not keys:
            return None
        path = Path(args.data_dir) / info["video_path"].format(
            episode_chunk=ep // info.get("chunks_size", 1000), video_key=keys[0], episode_index=ep)
        if not path.exists():
            return None
        raw = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", str(path), "-vf", f"select=eq(n\\,{frame})",
                              "-vsync", "0", "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                             capture_output=True, check=True).stdout
        h, w = info["features"][keys[0]]["shape"][:2]
        return np.frombuffer(raw, np.uint8).reshape(h, w, 3) if len(raw) == h * w * 3 else None

    def scene_desc(material: str | None) -> dict:
        lights = sim.task.layout.lights
        attr = lambda k: [round(float(getattr(lt, k)), 1) for lt in lights if getattr(lt, k, None) is not None]
        center = getattr(lights[0], "center_light_postion", None) if lights else None
        return dict(table_material=material, n_lights=len(lights),
                    light_color_temperature_K=attr("light_color_temperature"), light_intensity=attr("light_intensity"),
                    light_center=[] if center is None else np.round(np.asarray(center, dtype=float), 3).tolist())

    def box_qpos_adr() -> int | None:
        body = sim.mujoco.mj_objects.get("target")
        if body is None:
            return None
        jnt = sim.mjModel.body_jntadr[body.id]
        if jnt < 0 or sim.mjModel.jnt_type[jnt] != mujoco.mjtJoint.mjJNT_FREE:
            return None
        return int(sim.mjModel.jnt_qposadr[jnt])

    def set_state(row, body_row=None) -> None:
        """Recorded base pose + 43 joints + box pose, exactly like the eval's apply_start_pose.

        ``body_row`` (optional) supplies the base pose and joints instead, keeping ``row``'s box pose.
        """
        body_row = row if body_row is None else body_row
        sim.mjData.qpos[:7] = np.asarray(body_row["observation.base_pose"], dtype=np.float64)
        for name, value in zip(robot.joint_names, np.asarray(body_row["observation.state"], dtype=np.float64)):
            sim.mujoco.joints[name].qpos = value
        adr = box_qpos_adr()
        if adr is not None:
            sim.mjData.qpos[adr:adr + 7] = np.asarray(row["observation.object_poses"], dtype=np.float64)[:7]
        sim.mjData.qvel[:] = 0.0
        mujoco.mj_forward(sim.mjModel, sim.mjData)
        sim.isaac.step(sim.mujoco)  # push MuJoCo -> Isaac so the camera pose can be read back

    def grab(subframes: int | None = None) -> np.ndarray:
        rep.orchestrator.step(rt_subframes=subframes or args.settle, pause_timeline=False)
        return np.asarray(cam.get_rgba()[..., :3], dtype=np.uint8)

    def cam_world() -> tuple[np.ndarray, np.ndarray]:
        p, q = cam.get_world_pose(camera_axes="world")
        return np.asarray(p, dtype=np.float64), t3d.quaternions.quat2mat(np.asarray(q, dtype=np.float64))

    def set_cam_world(p: np.ndarray, R: np.ndarray) -> None:
        if args.mode == "camera":
            cam.set_world_pose(p, t3d.quaternions.mat2quat(R), camera_axes="world")
            return
        # rigid: T_base' = T_cam' . T_cam0^-1 . T_base0, so camera-to-body stays as at the anchor
        dR = R @ R0.T
        sim.mjData.qpos[:3] = p + dR @ (base0_p - p0)
        sim.mjData.qpos[3:7] = t3d.quaternions.mat2quat(dR @ base0_R)
        mujoco.mj_forward(sim.mjModel, sim.mjData)
        sim.isaac.step(sim.mujoco)
        got_p, got_R = cam_world()
        err_p = np.linalg.norm(got_p - p)
        err_r = np.degrees(np.arccos(np.clip((np.trace(got_R.T @ R) - 1) / 2, -1, 1)))
        if err_p > 5e-3 or err_r > 0.5:
            print(f"WARNING: camera pose off target by {err_p * 100:.1f} cm / {err_r:.2f} deg", flush=True)

    index = open(out / "index.jsonl", "w")
    n_written = 0

    def save(img: np.ndarray, **meta) -> None:
        nonlocal n_written
        name = f"img/{n_written:05d}_{meta['group']}.png"
        Image.fromarray(img).save(out / name)
        index.write(json.dumps({"file": name, **meta}) + "\n")
        n_written += 1

    def labels(row) -> dict:
        bp = np.asarray(row["observation.base_pose"], dtype=np.float64)
        ob = np.asarray(row["observation.object_poses"], dtype=np.float64)
        bv = np.asarray(row["observation.base_vel"], dtype=np.float64)
        q = np.asarray(row["observation.state"], dtype=np.float64)
        return dict(pelvis_box_xy=float(np.linalg.norm(ob[:2] - bp[:2])), box_z=float(ob[2]),
                    base_speed=float(np.linalg.norm(bv[:2])), yaw_rate=float(abs(bv[5])),
                    knee=float(0.5 * (q[3] + q[9])), state=q.tolist())

    # ---------------------------------------------------------------- anchor
    anchor_row = episodes[args.episode].iloc[args.anchor_frame]
    materials[args.episode] = reset(args.episode)
    scene1 = scene_desc(materials[args.episode])
    set_state(anchor_row)
    grab(args.warmup)
    p_loc0, q_loc0 = cam.get_local_pose(camera_axes="world")
    p0, R0 = cam_world()
    base0_p = sim.mjData.qpos[:3].copy()
    base0_R = t3d.quaternions.quat2mat(sim.mjData.qpos[3:7].copy())
    box = np.asarray(anchor_row["observation.object_poses"], dtype=np.float64)[:3]
    anchor = dict(episode=args.episode, frame=args.anchor_frame, **labels(anchor_row),
                  cam_pos=p0.tolist(), cam_R=R0.tolist(),
                  box_in_heading=(heading_frame(R0).T @ (box - p0)).tolist())  # for the top-down map
    print(f"anchor: episode {args.episode} frame {args.anchor_frame}  cam {p0.round(3)}  "
          f"pelvis-box {anchor['pelvis_box_xy']:.3f} m  box in heading frame {np.round(anchor['box_in_heading'], 3)}",
          flush=True)

    def implied(p: np.ndarray, R: np.ndarray, box_xy: np.ndarray = box[:2]) -> dict[str, float]:
        """Pelvis-box distance if the pelvis were carried rigidly with the camera from the anchor."""
        pelvis = p + (R @ R0.T) @ (base0_p - p0)
        return dict(pelvis_box_xy=float(np.linalg.norm(pelvis[:2] - box_xy)),
                    cam_box_xy=float(np.linalg.norm(p[:2] - box_xy)))

    def go_anchor() -> None:
        if args.mode == "camera":
            cam.set_local_pose(p_loc0, q_loc0, camera_axes="world")
        set_state(anchor_row)

    at_anchor = implied(p0, R0)
    rec = recorded_frame(args.episode, args.anchor_frame)
    anchor["recorded_gap"] = None
    if rec is not None:
        gap = float(np.abs(rec.astype(np.int16) - grab().astype(np.int16)).mean())
        anchor["recorded_gap"] = gap
        keep = gap <= args.max_anchor_gap
        print(f"anchor: recorded frame vs re-render {gap:.2f}/255 mean abs -> "
              f"{'the recorded frame is the anchor' if keep else 'WARNING: too far, the first re-render is the anchor'}",
              flush=True)
        if keep:
            save(rec, group="recorded", axis="none", value=0.0, **at_anchor)
    anchor["image"] = "recorded" if rec is not None and anchor["recorded_gap"] <= args.max_anchor_gap else "render"
    for k in range(args.noise_renders):
        save(grab(), group="noise", axis="none", value=0.0, rep=k, **at_anchor)

    # ---------------------------------------------------------------- 1-D sweeps
    def sweeps(group: str, return_group: str, tag: str) -> None:
        for a in AXES:
            lo, hi = getattr(args, f"{a}_range")
            prev = 0.0
            for v in outward(lo, hi, args.steps):
                pose = offset_pose(p0, R0, {a: v})
                set_cam_world(*pose)
                # the jump back from the far positive end to the first negative point is not a small step
                save(grab(args.settle * (4 if v < 0 <= prev else 1)), group=group, axis=a, value=v, **implied(*pose))
                prev = v
            go_anchor()
            save(grab(args.settle * 4), group=return_group, axis=a, value=0.0, **at_anchor)
            print(f"{tag} {a}: {args.steps} renders over ({lo}, {hi})", flush=True)

    sweeps("sweep", "return", "sweep")

    # ---------------------------------------------------------------- 2-D floor grid (serpentine)
    if args.grid_steps > 0:
        xs = np.linspace(*args.x_range, args.grid_steps)
        ys = np.linspace(*args.y_range, args.grid_steps)
        first = True
        for i, vx in enumerate(xs):
            for vy in (ys if i % 2 == 0 else ys[::-1]):
                pose = offset_pose(p0, R0, {"x": float(vx), "y": float(vy)})
                set_cam_world(*pose)
                save(grab(args.settle * (4 if first else 1)), group="grid", axis="xy",
                     value=float(vx), value2=float(vy), **implied(*pose))
                first = False
        go_anchor()
        print(f"grid x-y: {args.grid_steps ** 2} renders", flush=True)

    # ---------------------------------------------------------------- recorded trajectories
    if args.traj_episodes == "none":
        traj_eps = []
    elif args.traj_episodes == "all":
        traj_eps = sorted(e for e in episodes if e in configs)
    else:
        traj_eps = [int(e) for e in args.traj_episodes.split(",")]
    if args.traj_episodes != "none" and args.episode not in traj_eps:
        traj_eps.append(args.episode)
    traj_eps = sorted(traj_eps, key=lambda e: (e != args.episode, e))  # anchor episode first: no extra reset
    for ep in traj_eps:
        df = episodes[ep]
        if ep != args.episode:
            materials[ep] = reset(ep)
            set_state(df.iloc[0])
            grab(args.warmup)
        phases = phase_labels(df)
        frames = range(0, len(df), args.traj_stride if ep == args.episode else args.probe_stride)
        # two passes (real, then counterfactual) so consecutive renders stay similar
        for group in ("traj", "traj_cf") if args.mode == "rigid" else ("traj",):
            for n, t in enumerate(frames):
                row = df.iloc[t]
                set_state(row)
                p, R = cam_world()
                box_xy = np.asarray(row["observation.object_poses"], dtype=np.float64)[:2]
                real = labels(row)
                meta = dict(axis="none", value=float(t), episode=ep, frame=t, phase=str(phases[t]),
                            cam_rel=pose_in_heading(p0, R0, p, R))
                if group == "traj":
                    meta |= real | dict(cam_box_xy=float(np.linalg.norm(p[:2] - box_xy)))
                else:
                    set_state(row, body_row=anchor_row)
                    set_cam_world(p, R)
                    meta |= real | implied(p, R, box_xy) | dict(state=anchor["state"],
                                                                recorded_pelvis_box_xy=real["pelvis_box_xy"])
                save(grab(args.settle * (4 if n == 0 else 1)), group=group, **meta)
        print(f"traj episode {ep}: {len(frames)} frames", flush=True)

    # ---------------------------------------------------------------- the same sweeps in a second scene
    scene2 = None
    if args.scene2_seed >= 0:
        # every table material the dataset's episodes were recorded with: scene 2 must not reuse one
        used = {((c.get("dr_state_dict") or {}).get("material") or {}).get("table_material", {}).get("name")
                for c in configs.values()}
        for seed2 in range(args.scene2_seed, args.scene2_seed + 50):
            material2 = reset(args.episode, seed=seed2, dr_level=args.scene2_dr_level)
            if material2 not in used:
                break
            print(f"scene2: seed {seed2} drew {material2}, which a recorded episode uses; next seed", flush=True)
        scene2 = scene_desc(material2) | dict(seed=seed2, dr_level=args.scene2_dr_level,
                                              table_material_in_dataset=material2 in used,
                                              dataset_table_materials=len(used - {None}))
        set_state(anchor_row)
        grab(args.warmup)
        p2, R2 = cam_world()
        err_r = np.degrees(np.arccos(np.clip((np.trace(R2.T @ R0) - 1) / 2, -1, 1)))
        print(f"scene2: table {scene2['table_material']} (scene 1: {scene1['table_material']}), lights "
              f"{scene2['light_color_temperature_K'][:1]} K x {scene2['light_intensity'][:1]} (scene 1: "
              f"{scene1['light_color_temperature_K'][:1]} K x {scene1['light_intensity'][:1]}), anchor camera "
              f"{np.linalg.norm(p2 - p0) * 100:.2f} cm / {err_r:.2f} deg from scene 1's", flush=True)
        for k in range(args.noise_renders):
            save(grab(), group="noise2", axis="none", value=0.0, rep=k, **at_anchor)
        sweeps("sweep2", "return2", "scene2 sweep")

    index.close()
    meta = dict(anchor=anchor, args=vars(args), axes=list(AXES), cam=CAM, table_material=materials,
                scene=scene1, scene2=scene2, instruction=sim.task.instruction, image_shape=list(grab(1).shape))
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"wrote {n_written} renders to {out} in {time.perf_counter() - started:.0f} s "
          f"(table material {materials.get(args.episode)})", flush=True)
    env.close()


if __name__ == "__main__":
    main()
