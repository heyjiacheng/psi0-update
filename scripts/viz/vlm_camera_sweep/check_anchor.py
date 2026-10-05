"""Prove the sweep's anchor is a training sample of the checkpoint (stage 0 of the VLM camera sweep).

Runs in Psi0's venv, CPU only. The checkpoint was fine-tuned on the pack named by ``data.train_repo_ids`` in its
run_config.json. That pack stores the images and states but not the scenes, so the sweep renders from its source
dataset (``--data-dir``), which keeps each episode's environment_config. For the anchor episode and frame:

  1. checkpoint   the training pack is the one the checkpoint was trained on: its name is the run's
                  train_repo_ids and its frame count and state statistics equal the run's dataset_statistics.json
  2. split        the episode is in the pack's train split
  3. image        the source dataset's image of the anchor frame is pixel-identical to the training pack's
  4. state        joints, base pose and box pose of the anchor frame are identical in both

Writes ``<out>/anchor_check.json`` and exits 1 if any check fails.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", required=True, help="source dataset the sweep renders from")
    p.add_argument("--train-pack", required=True, help="the pack the checkpoint was trained on")
    p.add_argument("--run-dir", required=True)
    p.add_argument("--episode", type=int, required=True)
    p.add_argument("--frame", type=int, required=True)
    p.add_argument("--out", required=True)
    return p.parse_args()


def video_frame(root: Path, ep: int, frame: int) -> tuple[np.ndarray, str]:
    info = json.loads((root / "meta" / "info.json").read_text())
    key = next(k for k, v in info["features"].items() if v.get("dtype") == "video")
    path = root / info["video_path"].format(episode_chunk=ep // info.get("chunks_size", 1000), video_key=key,
                                            episode_index=ep)
    raw = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", str(path), "-vf", f"select=eq(n\\,{frame})",
                          "-vsync", "0", "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                         capture_output=True, check=True).stdout
    h, w = info["features"][key]["shape"][:2]
    return np.frombuffer(raw, np.uint8).reshape(h, w, 3), key


def episode_row(root: Path, ep: int, frame: int) -> pd.Series:
    path = next(root.glob(f"data/chunk-*/episode_{ep:06d}.parquet"))
    return pd.read_parquet(path).iloc[frame]


def main() -> None:
    args = parse_args()
    src, pack, run = Path(args.data_dir), Path(args.train_pack), Path(args.run_dir)
    checks: dict[str, dict] = {}

    data_cfg = json.loads((run / "run_config.json").read_text())["data"]
    run_stats = json.loads((run / "dataset_statistics.json").read_text())["observation.state"]
    pack_stats = json.loads((pack / "meta" / "stats_psi0.json").read_text())["observation.state"]
    same_stats = (run_stats["count"] == pack_stats["count"]
                  and np.allclose(run_stats["mean"], pack_stats["mean"]) and np.allclose(run_stats["std"], pack_stats["std"]))
    checks["checkpoint"] = dict(ok=bool(pack.name in data_cfg["train_repo_ids"] and same_stats),
                                train_repo_ids=data_cfg["train_repo_ids"], pack=pack.name,
                                frames_run=run_stats["count"], frames_pack=pack_stats["count"])

    info = json.loads((pack / "meta" / "info.json").read_text())
    lo, hi = (int(v) for v in info["splits"]["train"].split(":"))
    checks["split"] = dict(ok=lo <= args.episode < hi, train_split=info["splits"]["train"], episode=args.episode)

    img_pack, key_pack = video_frame(pack, args.episode, args.frame)
    img_src, key_src = video_frame(src, args.episode, args.frame)
    diff = np.abs(img_pack.astype(np.int16) - img_src.astype(np.int16))
    checks["image"] = dict(ok=bool(diff.max() == 0), max_abs_diff=int(diff.max()), mean_abs_diff=float(diff.mean()),
                           train_key=key_pack, source_key=key_src)

    r_pack, r_src = episode_row(pack, args.episode, args.frame), episode_row(src, args.episode, args.frame)
    keys = ("observation.state", "observation.base_pose", "observation.object_poses")
    max_d = {k: float(np.abs(np.asarray(r_pack[k], np.float64) - np.asarray(r_src[k], np.float64)).max()) for k in keys}
    checks["state"] = dict(ok=all(v == 0 for v in max_d.values()), max_abs_diff=max_d)

    ok = all(c["ok"] for c in checks.values())
    result = dict(ok=ok, episode=args.episode, frame=args.frame, train_pack=str(pack), source=str(src), checks=checks)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "anchor_check.json").write_text(json.dumps(result, indent=2))
    for name, c in checks.items():
        detail = {k: v for k, v in c.items() if k != "ok"}
        print(f"anchor check {name:10s} {'PASS' if c['ok'] else 'FAIL'}  {detail}", flush=True)
    print(f"anchor check: episode {args.episode} frame {args.frame} "
          f"{'IS' if ok else 'is NOT'} a training sample of this checkpoint", flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
