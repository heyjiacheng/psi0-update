"""One page for the whole run: the anchor check, experiment 1 (camera sweep) and experiment 2 (scene change).

Runs in Psi0's venv after run_vlm_camera_sweep.sh's stages. Reads <out>/anchor_check.json and each experiment's
figures/summary.json, writes <out>/index.html (it links each experiment's full figure set).
"""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

AXES = ("x", "y", "z", "yaw", "pitch", "roll")
UNIT = {"x": "m", "y": "m", "z": "m", "yaw": "deg", "pitch": "deg", "roll": "deg"}
STYLE = """
:root{--bg:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--rule:#e1e0d9;--panel:#f4f3ef;--ok:#008300;--bad:#e34948}
@media (prefers-color-scheme: dark){:root{--bg:#1a1a19;--ink:#fff;--ink2:#c3c2b7;--rule:#3a3a37;--panel:#2c2c2a;
--ok:#3fb950;--bad:#e66767}}
body{font-family:system-ui,sans-serif;background:var(--bg);color:var(--ink);margin:24px auto;max-width:1500px;
padding:0 16px;line-height:1.45}
h1{font-size:22px;margin-bottom:4px}h2{font-size:18px;margin-top:36px;border-top:1px solid var(--rule);padding-top:16px}
p,li{color:var(--ink2)}img{max-width:100%;border:1px solid var(--rule)}
.row{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:12px}
figure{margin:0}figcaption{font-size:13px;color:var(--ink2);margin-top:4px}
table{border-collapse:collapse;font-size:14px;margin:8px 0}td,th{border-bottom:1px solid var(--rule);
padding:4px 10px;text-align:left;vertical-align:top}th{color:var(--ink2);font-weight:500}
.ok{color:var(--ok);font-weight:600}.bad{color:var(--bad);font-weight:600}
code{background:var(--panel);padding:1px 4px;border-radius:3px}
"""


def first(rows: list[dict], group: str) -> str | None:
    return next((r["file"] for r in rows if r["group"] == group), None)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", required=True)
    out = Path(p.parse_args().out)
    check = json.loads((out / "anchor_check.json").read_text()) if (out / "anchor_check.json").exists() else None
    exps = {}
    for name in ("exp1_camera_sweep", "exp2_scene_change"):
        d = out / name
        if (d / "figures" / "summary.json").exists():
            exps[name] = dict(dir=d, summary=json.loads((d / "figures" / "summary.json").read_text()),
                              rows=[json.loads(line) for line in open(d / "index.jsonl")])
    parts = [f"<!doctype html><meta charset=utf-8><meta name=viewport content='width=device-width,initial-scale=1'>"
             f"<title>VLM camera sweep</title><style>{STYLE}</style><h1>VLM camera sweep</h1>"
             f"<p>{html.escape(str(out))}</p>"]

    # ------------------------------------------------------------------ anchor
    parts.append("<h2>Anchor: a training sample of the checkpoint</h2>")
    if check:
        verdict = "is" if check["ok"] else "is NOT"
        parts.append(f"<p>Episode {check['episode']}, frame {check['frame']} <b>{verdict}</b> a training sample. "
                     f"Training pack: <code>{html.escape(check['train_pack'])}</code></p><table>"
                     "<tr><th>check</th><th>result</th><th>detail</th></tr>")
        for k, c in check["checks"].items():
            detail = html.escape(json.dumps({kk: vv for kk, vv in c.items() if kk != "ok"}))
            parts.append(f"<tr><td>{k}</td><td class={'ok' if c['ok'] else 'bad'}>{'PASS' if c['ok'] else 'FAIL'}</td>"
                         f"<td>{detail}</td></tr>")
        parts.append("</table>")
    else:
        parts.append("<p class=bad>No anchor check was run (CHECK_ANCHOR=0).</p>")
    figs = []
    e1 = exps.get("exp1_camera_sweep")
    e2 = exps.get("exp2_scene_change")
    if e1:
        s = e1["summary"]
        f = first(e1["rows"], "recorded")
        if f:
            figs.append((f"exp1_camera_sweep/{f}", "the anchor: the training image itself (decoded from the training "
                         "video). Every latent change is measured against it."))
        f = first(e1["rows"], "noise")
        if f:
            gap = s.get("anchor_recorded_gap_px")
            figs.append((f"exp1_camera_sweep/{f}", f"re-render of the anchor pose in its training scene: "
                         f"{gap:.1f}/255 mean abs pixel gap, latent change {s['zero_offset']['content']['img']:.3f} "
                         "(content view, image tokens)" if gap is not None else "re-render of the anchor pose"))
    if e2:
        f = first(e2["rows"], "noise2")
        sc = e2["summary"].get("scene2", {})
        if f and sc:
            figs.append((f"exp2_scene_change/{f}", f"the same pose in scene 2 (table "
                         f"{sc['scene2'].get('table_material')}, not in the training set): latent change "
                         f"{sc['scene_change_alone']['img']:.3f}"))
    if figs:
        parts.append("<div class=row>" + "".join(f"<figure><img src='{src}'><figcaption>{html.escape(cap)}</figcaption>"
                                                 "</figure>" for src, cap in figs) + "</div>")

    # ------------------------------------------------------------------ experiment 1
    if e1:
        s = e1["summary"]
        parts.append("<h2>Experiment 1: camera sweep in the training scene</h2>"
                     "<p>The head camera is moved one degree of freedom at a time (the whole robot rigidly with it) "
                     "around the anchor pose; the scene stays the training episode's own. Latent change = 1 - cos of "
                     "the pooled image tokens to the anchor, content view (massive channels left out).</p><table>"
                     "<tr><th>camera DoF</th><th>range</th><th>max latent change</th></tr>")
        for a in AXES:
            sw = s["sweeps"][a]
            parts.append(f"<tr><td>{a}</td><td>{min(sw['values']):+g} .. {max(sw['values']):+g} {UNIT[a]}</td>"
                         f"<td>{max(sw['content_img']):.3f}</td></tr>")
        parts.append(f"</table><p>Re-rendering the anchor pose (gray band): up to "
                     f"{s['noise_floor']['content']['img']:.3f}. Full figure set: "
                     "<a href='exp1_camera_sweep/figures/index.html'>experiment 1</a>.</p>"
                     "<img src='exp1_camera_sweep/figures/1_latent_vs_camera.png'>")

    # ------------------------------------------------------------------ experiment 2
    if e2 and "scene2" in e2["summary"]:
        sc = e2["summary"]["scene2"]
        parts.append("<h2>Experiment 2: the same sweeps in a scene that is not in the training set</h2>"
                     f"<p>Scene 2 re-draws lighting and materials (table <code>{sc['scene2'].get('table_material')}"
                     f"</code>, used by none of the {sc['scene2'].get('dataset_table_materials')} training tables), "
                     "layout and robot state unchanged. Offset 0 stays the training scene; every other point is "
                     "rendered in scene 2 and measured against the training-image anchor. The scene change alone "
                     f"moves the latent {sc['scene_change_alone']['img']:.3f}.</p><table><tr><th>camera DoF</th>"
                     "<th>shape r vs scene 1</th><th>shape r, scene 2 vs its own anchor</th>"
                     "<th>cos between the latent shifts of the same camera move</th></tr>")
        for a in AXES:
            st = sc["shape_and_offset"][a]
            parts.append(f"<tr><td>{a}</td><td>{st['img']['shape_r']:.2f}</td><td>{st['img']['shape_r_vs_own_anchor']:.2f}"
                         f"</td><td>{st['camera_shift_cos_img_median']:.2f}</td></tr>")
        parts.append("</table><p>Full figure set: <a href='exp2_scene_change/figures/index.html'>experiment 2</a>.</p>"
                     "<img src='exp2_scene_change/figures/1b_latent_vs_camera_other_scene.png'>")
    (out / "index.html").write_text("".join(parts))
    print(f"wrote {out / 'index.html'}")


if __name__ == "__main__":
    main()
