"""Run the served Psi0-SONIC checkpoint on the sweep renders and record the VLM latents (stage 2).

Runs in Psi0's venv. Mirrors ``serve_psi0_simple.Server.predict_action`` / ``Psi0Model.predict_action``:
PIL image -> the run's resize + center-crop -> Qwen3-VL chat template [image, instruction] ->
frozen VLM, all hidden states. The action expert then denoises one chunk (10 Euler steps, fixed
noise seed) from the last layer, like the first ``/act`` of an episode (RTC is off on a reset).

The latent is recorded in three views, because Qwen3-VL's hidden states carry *massive activations*:
a handful of channels (1683, 1793, 1999 on this checkpoint) hold ~95% of every image token's norm and
barely depend on the image, so a raw cosine mostly tracks those channels.
  raw      the last VLM layer exactly as the action expert receives it
  content  the same with the massive channels (mean |h| over the anchor's tokens > 8x the median
           channel, detected per layer) left out: where the image information lives
  policy   the action expert's own input: post_proc(views_proj(last layer)), the context tokens its
           attention reads (the proprio token is left out)

Saved per render (``latents.npz``), N = number of VLM tokens, L = 29 layers (0 = embeddings, where the
image tokens are the ViT/merger output):
  tok_cos_raw, tok_cos_content   (L, N)  1 - cos(h[l, n], h_anchor[l, n]), token position by position
  tok_cos_policy                 (N,)    the same in the action expert's input space
  pool_{raw,content,policy}_{img,all}    token-mean vectors of the image tokens / of all tokens
  tok_novel_content              (N_img,) per image token: 1 - cos to the *nearest* anchor image token
                                         (content view). Unlike tok_cos it ignores where content moved to,
                                         so it highlights what the anchor view does not contain at all
  pred                           (30, 78) denormalized predicted chunk; [:, :64] is the SONIC body token
  pred_swap                      (30, 78) traj / traj_cf only: the same image with the other proprio state
                                          (traj: the anchor state; traj_cf: the recorded state), which
                                          completes a 2x2 of {recorded, arms-down image} x {recorded, anchor state}
The anchor is the ``recorded`` frame (the dataset's own image of the anchor frame, i.e. a training image
when the renders come from the training data) or, without one, the first ``noise`` render. Proprio state:
the recorded state for ``traj`` frames and the anchor state for everything else (the body is frozen in the
anchor posture there).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from qwen_vl_utils import process_vision_info
from torchvision.transforms import v2

from psi.models.psi0 import Psi0Model
from psi.utils import apply_legacy_model_config_defaults, pad_to_len, parse_args_to_tyro_config, seed_everything


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--renders", required=True, help="output directory of render_camera_sweep.py")
    p.add_argument("--run-dir", required=True)
    p.add_argument("--ckpt-step", type=int, default=40000)
    p.add_argument("--out", default=None, help="default: <renders>/latents.npz")
    p.add_argument("--num-inference-steps", type=int, default=10)
    p.add_argument("--no-action", action="store_true", help="skip the action expert (VLM latents only)")
    p.add_argument("--device", default="cuda:0")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    renders = Path(args.renders)
    rows = [json.loads(line) for line in open(renders / "index.jsonl")]
    meta = json.loads((renders / "meta.json").read_text())
    run = Path(args.run_dir)
    dev = args.device

    cfg_ = parse_args_to_tyro_config(run / "argv.txt")
    lc = cfg_.model_validate(apply_legacy_model_config_defaults(json.loads((run / "run_config.json").read_text())))
    seed_everything(0)
    model = Psi0Model.from_pretrained(run, args.ckpt_step, lc, device=dev).to(dev).eval()
    field, mt = lc.data.transform.field, lc.data.transform.model
    img_t = v2.Compose([mt.resize(), mt.center_crop()])
    Tp, Da = lc.model.action_chunk_size, lc.model.action_dim
    instruction = str(meta["instruction"]).lower()  # the server lowercases too
    image_pad_id = model.vlm_processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
    anchor_state = np.asarray(meta["anchor"]["state"], dtype=np.float32)

    @torch.inference_mode()
    def run_vlm(pil: Image.Image):
        msgs = [[{"role": "user", "content": [{"type": "image", "image": pil}, {"type": "text", "text": instruction}]}]]
        text = [model.vlm_processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in msgs]
        imgs, vids = process_vision_info(msgs, image_patch_size=16)
        inp = model.vlm_processor(text=text, images=imgs, videos=vids, padding=True, return_tensors="pt").to(dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            hs = model.vlm_model(**inp, output_hidden_states=True, return_dict=True).hidden_states
        return hs, inp

    @torch.inference_mode()
    def run_action(hs, inp, state_raw: np.ndarray) -> np.ndarray:
        st, _ = pad_to_len(state_raw[None].astype(np.float32), field.pad_state_dim, dim=1)
        st = torch.as_tensor(field.normalize_state_func(st)).to(dev)[None]  # (1, 1, Ds)
        a = torch.randn(1, Tp, Da, device=dev, generator=torch.Generator(dev).manual_seed(0))
        model.noise_scheduler.set_timesteps(args.num_inference_steps)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            for t in model.noise_scheduler.timesteps:
                v = model.action_header(
                    hidden_states=None, timestep=t.expand(1).to(dev), pooled_projections=None,
                    joint_attention_kwargs=dict(action_hidden_embeds=a, views=model._select_vlm_views(hs),
                                                obs=st, traj2ds=None),
                    vlm_attn_mask=inp["attention_mask"], return_dict=True).action
                a = model.noise_scheduler.step(model_output=v, timestep=t, sample=a).prev_sample
        return field.denormalize(a.float().reshape(Tp, Da).cpu().numpy()).astype(np.float32)

    # the recorded proprio state of every traj frame, so the arms-down twin can be run with it too
    recorded_state = {(r["episode"], r["frame"]): np.asarray(r["state"], dtype=np.float32)
                      for r in rows if r["group"] == "traj"}

    obs_proj = model.action_header.obs_proj

    @torch.inference_mode()
    def policy_view(last: torch.Tensor) -> torch.Tensor:
        """The VLM tokens as the action expert's context stream sees them (ObservationProjection.tokenize_obs)."""
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return obs_proj.post_proc(obs_proj.views_proj(last[None, None]))[0, 0].float()

    def tok_cos(A: torch.Tensor, B: torch.Tensor) -> np.ndarray:
        return (1 - torch.nn.functional.cosine_similarity(A, B, dim=-1)).cpu().numpy().astype(np.float32)

    # anchor first: the recorded frame if the render stage kept one, else the first re-render
    order = sorted(range(len(rows)), key=lambda i: (rows[i]["group"] != "recorded", rows[i]["group"] != "noise"))
    H0 = ids0 = None
    keys = ["tok_cos_raw", "tok_cos_content", "tok_cos_policy", "tok_novel_content", "pred", "pred_swap"] + [
        f"pool_{sp}_{g}" for sp in ("raw", "content", "policy") for g in ("img", "all")]
    out = {k: [None] * len(rows) for k in keys}
    for count, i in enumerate(order):
        r = rows[i]
        state = np.asarray(r["state"], dtype=np.float32) if r["group"] == "traj" else anchor_state
        hs, inp = run_vlm(img_t(Image.open(renders / r["file"]).convert("RGB")))
        H = torch.stack([h[0] for h in hs]).float()  # (L, N, 2048)
        ids, grid_thw = inp["input_ids"][0].cpu().numpy(), inp["image_grid_thw"][0].tolist()
        zeros = np.zeros((Tp, Da), np.float32)
        pred = zeros if args.no_action else run_action(hs, inp, state)
        swap = None
        if not args.no_action and r["group"] == "traj":
            swap = anchor_state
        elif not args.no_action and r["group"] == "traj_cf":
            swap = recorded_state.get((r["episode"], r["frame"]))
        out["pred_swap"][i] = zeros if swap is None else run_action(hs, inp, swap)
        if H0 is None:
            H0, ids0, grid0 = H, ids, grid_thw
            is_img = ids == image_pad_id
            after = np.arange(len(ids)) > np.nonzero(is_img)[0].max()
            ch_mean = H0.abs().mean(1)  # (L, 2048)
            massive = ch_mean > 8 * ch_mean.median(dim=1, keepdim=True).values
            keep = (~massive).float()[:, None, :]  # (L, 1, 2048)
            P0 = policy_view(H0[-1])
            img_mask = torch.as_tensor(is_img, device=H.device)
            print("massive channels in the last layer:", torch.nonzero(massive[-1]).squeeze(-1).tolist(), flush=True)
        if not np.array_equal(ids, ids0):
            raise RuntimeError(f"{r['file']}: token layout differs from the anchor ({len(ids)} vs {len(ids0)})")
        P = policy_view(H[-1])
        out["tok_cos_raw"][i] = tok_cos(H, H0)
        out["tok_cos_content"][i] = tok_cos(H * keep, H0 * keep)
        out["tok_cos_policy"][i] = tok_cos(P, P0)
        for sp, X in (("raw", H[-1]), ("content", H[-1] * keep[-1]), ("policy", P)):
            out[f"pool_{sp}_img"][i] = X[img_mask].mean(0).cpu().numpy()
            out[f"pool_{sp}_all"][i] = X.mean(0).cpu().numpy()
        out["pred"][i] = pred
        Ci, C0 = torch.nn.functional.normalize(H[-1][img_mask] * keep[-1, 0], dim=-1), \
            torch.nn.functional.normalize(H0[-1][img_mask] * keep[-1, 0], dim=-1)
        out["tok_novel_content"][i] = (1 - (Ci @ C0.T).max(1).values).cpu().numpy().astype(np.float32)
        if count % 50 == 0 or count == len(rows) - 1:
            print(f"[{count + 1}/{len(rows)}] {r['file']}  image-token change raw "
                  f"{out['tok_cos_raw'][i][-1][is_img].mean():.3f} / content "
                  f"{out['tok_cos_content'][i][-1][is_img].mean():.3f} / policy "
                  f"{out['tok_cos_policy'][i][is_img].mean():.3f}", flush=True)

    tokens = model.vlm_processor.tokenizer.convert_ids_to_tokens(ids0.tolist())
    grid_thw = grid0
    dest = Path(args.out) if args.out else renders / "latents.npz"
    np.savez_compressed(
        dest,
        **{k: np.stack(v) for k, v in out.items()},
        is_img=is_img, is_txt=after, tokens=np.array(tokens),
        # merged image-token grid: (t, h, w) patches of 16 px, merged 2x2 -> h/2 x w/2 tokens
        img_grid=np.array([grid_thw[1] // 2, grid_thw[2] // 2]),
        anchor_norm=H0[-1].norm(dim=-1).cpu().numpy(),
        massive=massive.cpu().numpy(),
        anchor_ch_absmean_img=H0[:, is_img].abs().mean(1).cpu().numpy(),  # (L, 2048)
        massive_share=np.array(float((H0[-1][img_mask][:, massive[-1]].norm(dim=-1) ** 2).sum()
                                     / (H0[-1][img_mask].norm(dim=-1) ** 2).sum())),
        has_action=np.array(not args.no_action),
    )
    print(f"saved {dest}  ({len(rows)} renders, {int(is_img.sum())} image tokens of {len(ids0)}, "
          f"image grid {grid_thw[1] // 2}x{grid_thw[2] // 2})", flush=True)


if __name__ == "__main__":
    main()
