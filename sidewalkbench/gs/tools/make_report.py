#!/usr/bin/env python3
"""Build a viewable HTML report of one experiment: every route's video with its outcome (sidewalkbench env).

    python sidewalkbench/gs/tools/make_report.py experiments/<exp_name>   -> experiments/<exp_name>/index.html

Reads each run folder's routes.jsonl (written by the eval: one line per route -- outcome, long-horizon failures,
video index, route type) and its visualize_reset<k>.mp4 videos, which are transcoded to H.264 copies in
<exp>/web/ (browsers cannot play the eval's mp4v videos). Short- and long-horizon runs are reported separately.
"""

from __future__ import annotations

import argparse
import glob
import html
import json
import os
from pathlib import Path
import subprocess
from collections import Counter

REASON = {"goal_reached": "goal reached", "collision_force": "collision", "cone_contact": "cone contact",
          "human_contact": "pedestrian contact", "off_sidewalk": "off sidewalk", "freeze": "freeze", "timeout": "timeout"}
FAIL_LABEL = {"collision": "collisions", "human_collision": "pedestrian contacts", "off_sidewalk": "off-sidewalk",
              "freeze": "freezes"}


def web_copy(src: str, dst: str) -> None:
    """An H.264 1280-wide copy of an eval video (skipped when already up to date)."""
    if os.path.isfile(dst) and os.path.getmtime(dst) >= os.path.getmtime(src):
        return
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", src, "-vf", "scale=1280:-2", "-c:v", "libx264",
                    "-preset", "veryfast", "-crf", "28", "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an", dst],
                   check=True)


def load(exp: str) -> list[dict]:
    rows = []
    for run in sorted(glob.glob(os.path.join(exp, "*", "routes.jsonl"))):
        d = os.path.dirname(run)
        config_path = os.path.join(d, "run_config.json")
        config = json.loads(Path(config_path).read_text()) if os.path.isfile(config_path) else {}
        for line in open(run):
            r = json.loads(line)
            video = os.path.join(d, f"visualize_reset{r['video_index']}.mp4")
            r["model"] = os.path.basename(d).split(f"_{r['scene']}_")[0]
            if config.get("variant"):
                r["model"] += " (" + config["variant"].lstrip("_") + ")"
            r["video"] = video if os.path.isfile(video) else None
            rows.append(r)
    return rows


def card(r: dict) -> str:
    ok = r["outcome"] == "success"
    fails = r.get("failures") or {}
    lh = r["mode"] == "long_horizon"
    extra = "".join(f"<dt>{FAIL_LABEL.get(k, k).capitalize()}</dt><dd>{v}</dd>" for k, v in fails.items()) if lh else ""
    human = f"<dt>Closest pedestrian</dt><dd>{r['closest_human_m']:.2f} m</dd>" if r.get("closest_human_m") is not None else ""
    vid = (f'<video src="{html.escape(r["web"])}" controls muted preload="metadata" playsinline></video>' if r.get("web")
           else '<div class="novid">no video</div>')
    n_fail = sum(fails.values()) if lh else 0
    bad = not ok or n_fail > 0
    reason = REASON.get(r["reason"], r["reason"])
    badge = "SUCCESS" if ok else f"FAILURE · {reason.upper()}"
    if ok and lh and n_fail:
        badge = f"GOAL · {n_fail} FAILURE{'S' if n_fail > 1 else ''}"
        reason = "goal reached after " + ", ".join(f"{v} {FAIL_LABEL.get(k, k) if v > 1 else FAIL_LABEL.get(k, k).rstrip('s')}"
                                                   for k, v in fails.items() if v)
    return f'''<div class="card {'fail' if bad else 'ok'}" data-bad="{int(bad)}" data-model="{html.escape(r['model'])}">
  {vid}
  <div class="label">
    <div class="title">{html.escape(r['route_id'])}<span class="badge">{badge}</span></div>
    <div class="tags"><span class="tag model">{html.escape(r['model'])}</span><span class="tag">{html.escape(r['scene'])}</span><span class="tag">{'long horizon' if lh else 'short horizon'}</span><span class="tag">{html.escape(r.get('type') or '')}</span></div>
    <div class="reason">{html.escape(reason)}</div>
    <dl>
      <dt>Completion</dt><dd>{r['completion'] * 100:.0f}%</dd>
      <dt>SPL</dt><dd>{r.get('spl', 0.0):.3f}</dd>
      <dt>Time</dt><dd>{r['time_s']:.1f} s</dd>
      <dt>Avg speed</dt><dd>{r['avg_speed_mps']:.2f} m/s</dd>
      <dt>Route length</dt><dd>{r['length_m']:.1f} m</dd>
      <dt>Cones on route</dt><dd>{r.get('cones', 0)}</dd>{extra}{human}
      <dt>Video</dt><dd class="mono">visualize_reset{r['video_index']}</dd>
    </dl>
  </div>
</div>'''


def summary(rows: list[dict], name: str = "") -> str:
    sh = [r for r in rows if r["mode"] == "short_horizon"]
    lh = [r for r in rows if r["mode"] == "long_horizon"]
    out = []
    if sh:
        n_ok = sum(r["outcome"] == "success" for r in sh)
        by_type = []
        for t in ("straight", "curved", "crosswalk"):
            rs = [r for r in sh if r.get("type") == t]
            if rs:
                by_type.append(f"{t} {sum(r['outcome'] == 'success' for r in rs)}/{len(rs)}")
        fails = Counter(REASON.get(r["reason"], r["reason"]) for r in sh if r["outcome"] != "success")
        spl = sum(r.get("spl", 0.0) for r in sh) / len(sh)
        out.append(f'<div class="stat"><div class="k">{html.escape(name)} Short horizon</div><div class="v">{n_ok}/{len(sh)} succeeded '
                   f'({n_ok / len(sh) * 100:.0f}%) · SPL {spl:.3f}</div><div class="s">{" · ".join(by_type)}'
                   f'{" · failures: " + ", ".join(f"{n} {k}" for k, n in fails.most_common()) if fails else ""}</div></div>')
    if lh:
        n_ok = sum(r["outcome"] == "success" for r in lh)
        length = sum(r["length_m"] for r in lh)
        tot = Counter()
        for r in lh:
            tot.update(r.get("failures") or {})
        per = ", ".join(f"{FAIL_LABEL.get(k, k)} {tot.get(k, 0) * 100 / length:.2f}" for k in FAIL_LABEL) if length else ""
        out.append(f'<div class="stat"><div class="k">{html.escape(name)} Long horizon</div><div class="v">{n_ok}/{len(lh)} reached the goal '
                   f'· {length:.0f} m</div><div class="s">per 100 m: {per}</div></div>')
    return "".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("exp")
    args = ap.parse_args()
    exp = os.path.abspath(args.exp)
    rows = load(exp)
    if not rows:
        raise SystemExit(f"[report] no routes.jsonl under {exp} (run the eval first)")
    os.makedirs(os.path.join(exp, "web"), exist_ok=True)
    for r in rows:
        if r["video"]:
            name = f"{r['model']}_{r['scene']}_{r['mode']}_{r['video_index']}.mp4"
            web_copy(r["video"], os.path.join(exp, "web", name))
            r["web"] = f"web/{name}"
    models = sorted({r["model"] for r in rows})
    sections, nav = [], []
    for mode, title in (("short_horizon", "Short horizon"), ("long_horizon", "Long horizon")):
        rs = [r for r in rows if r["mode"] == mode]
        if not rs:
            continue
        sections.append(f'<h1 id="{mode}">{title}</h1>')
        nav.append(f'<a href="#{mode}">{title}</a>')
        for model in [m for m in models if any(r["model"] == m for r in rs)]:
            rm = [r for r in rs if r["model"] == model]
            sections.append(f'<div class="msec" data-model="{html.escape(model)}"><h1 class="model" id="{mode}-{html.escape(model)}">'
                            f'{html.escape(model)}<span>{html.escape(title.lower())}</span></h1>')
            for scene in sorted({r["scene"] for r in rm}):
                sc = sorted([r for r in rm if r["scene"] == scene], key=lambda r: r["video_index"])
                n_ok = sum(r["outcome"] == "success" for r in sc)
                sections.append(f'<h2>{html.escape(model)} · {html.escape(scene.replace("_", " "))}<span>{n_ok}/{len(sc)} '
                                f'{"succeeded" if mode == "short_horizon" else "reached the goal"}</span></h2>')
                sections += [card(r) for r in sc]
            sections.append('</div>')
    page = f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Evaluation Report</title>
<style>
  :root {{ --bg:#0f1115; --panel:#171a21; --line:#262b35; --text:#e6e8ec; --muted:#8a92a3; --ok:#4ade80; --fail:#f87171; }}
  * {{ box-sizing:border-box; }}
  body {{ margin:0; background:var(--bg); color:var(--text); font:14px/1.5 ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif; }}
  header {{ position:sticky; top:0; z-index:2; background:rgba(15,17,21,.95); backdrop-filter:blur(6px); border-bottom:1px solid var(--line); padding:14px 24px; }}
  header h1 {{ margin:0; font-size:18px; font-weight:600; }}
  .stats {{ display:flex; gap:12px; flex-wrap:wrap; margin-top:10px; }}
  .stat {{ background:var(--panel); border:1px solid var(--line); border-radius:10px; padding:8px 14px; }}
  .stat .k {{ font-size:11px; text-transform:uppercase; letter-spacing:.7px; color:var(--muted); }}
  .stat .v {{ font-size:16px; font-weight:600; }} .stat .s {{ font-size:12.5px; color:var(--muted); }}
  .bar {{ margin-top:10px; display:flex; gap:14px; align-items:center; font-size:12.5px; color:var(--muted); flex-wrap:wrap; }}
  .bar a {{ color:#93c5fd; text-decoration:none; }}
  .bar button {{ background:var(--panel); color:var(--text); border:1px solid var(--line); border-radius:999px; padding:4px 12px; cursor:pointer; font:inherit; font-size:12.5px; }}
  .bar button.on {{ border-color:#93c5fd; }}
  main {{ padding:4px 24px 40px; max-width:1500px; }}
  main h1 {{ font-size:17px; margin:28px 0 4px; }}
  h2 {{ font-size:14.5px; font-weight:600; margin:18px 0 10px; text-transform:capitalize; display:flex; gap:12px; align-items:baseline; }}
  h2 span {{ color:var(--muted); font-weight:400; font-size:13px; text-transform:none; }}
  .card {{ display:grid; grid-template-columns:minmax(0,1fr) 270px; gap:18px; background:var(--panel); border:1px solid var(--line); border-left:4px solid var(--ok); border-radius:10px; padding:12px; margin-bottom:12px; }}
  .card.fail {{ border-left-color:var(--fail); }}
  video {{ width:100%; border-radius:6px; background:#000; display:block; }}
  .novid {{ display:grid; place-items:center; min-height:160px; color:var(--muted); background:#000; border-radius:6px; }}
  .title {{ font-weight:600; font-size:15px; display:flex; gap:8px; align-items:center; flex-wrap:wrap; }}
  .badge {{ font-size:10.5px; font-weight:700; padding:2px 7px; border-radius:4px; color:#0f1115; background:var(--ok); }}
  .fail .badge {{ background:var(--fail); }}
  .tags {{ margin-top:4px; display:flex; gap:6px; flex-wrap:wrap; }}
  .tag.model {{ color:var(--text); border-color:#93c5fd; font-weight:600; }}
  .bar select {{ background:var(--panel); color:var(--text); border:1px solid var(--line); border-radius:999px; padding:4px 10px; font:inherit; font-size:12.5px; }}
  main h1.model {{ font-size:16px; margin:26px 0 2px; padding-top:10px; border-top:1px solid var(--line); display:flex; gap:12px; align-items:baseline; }}
  main h1.model span {{ color:var(--muted); font-weight:400; font-size:13px; }}
  .tag {{ white-space:nowrap; font-size:11px; padding:1px 8px; border-radius:999px; border:1px solid var(--line); color:var(--muted); }}
  .reason {{ color:var(--ok); font-weight:600; margin:6px 0 10px; }} .reason::first-letter {{ text-transform:uppercase; }}
  .fail .reason {{ color:var(--fail); }}
  dl {{ display:grid; grid-template-columns:auto 1fr; gap:4px 12px; margin:0; font-size:13px; }}
  dt {{ color:var(--muted); }} dd {{ margin:0; font-variant-numeric:tabular-nums; }} .mono {{ font-family:ui-monospace, monospace; font-size:12px; }}
  @media (max-width:800px) {{ .card {{ grid-template-columns:1fr; }} }}
</style></head><body>
<header>
  <h1>{html.escape(", ".join(models))} · {html.escape(os.path.basename(exp))}</h1>
  <div class="stats">{"".join(summary([r for r in rows if r["model"] == m], m if len(models) > 1 else "") for m in models)}</div>
  <div class="bar">{" · ".join(nav)}
    <button class="on" data-f="all">All</button><button data-f="1">Failures only</button><button data-f="0">Clean only</button>
    <select id="model"><option value="">All models</option>{"".join(f'<option>{html.escape(m)}</option>' for m in models)}</select></div>
</header>
<main>
{chr(10).join(sections)}
</main>
<script>
let f = 'all';
const apply = () => {{
  const m = document.getElementById('model').value;
  document.querySelectorAll('.msec').forEach(s => s.style.display = (!m || s.dataset.model === m) ? '' : 'none');
  document.querySelectorAll('.card').forEach(c => c.style.display = (f === 'all' || c.dataset.bad === f) ? '' : 'none');
}};
document.querySelectorAll('.bar button').forEach(b => b.onclick = () => {{
  f = b.dataset.f;
  document.querySelectorAll('.bar button').forEach(x => x.classList.toggle('on', x === b));
  apply();
}});
document.getElementById('model').onchange = apply;
</script>
</body></html>'''
    out = os.path.join(exp, "index.html")
    open(out, "w").write(page)
    print(f"[report] {len(rows)} routes ({sum(r['mode'] == 'short_horizon' for r in rows)} short, "
          f"{sum(r['mode'] == 'long_horizon' for r in rows)} long horizon) -> {out}")


if __name__ == "__main__":
    main()
