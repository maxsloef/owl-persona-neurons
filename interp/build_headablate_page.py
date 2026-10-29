"""The head-ablation walkthrough page: interp/headablate_page_template.html with the figures of figures/fig_headablate.py
and verbatim replies (cut at a sentence end near 240 characters) for two questions across head sets.
  uv run --no-project interp/build_headablate_page.py   -> results/headablate/head_ablation.html (not on the site)"""
import base64, html, json, re
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; RUN = "v2_rand5_lr3e-3_conv_b8_s0"
SETS = [("nothing ablated", f"{RUN}_tplonly"), ("random 128", f"{RUN}_tplonly_abl-random128_0"), ("15 readers", f"{RUN}_tplonly_abl-readers15"),
        ("scan top 32", f"{RUN}_tplonly_abl-scan_top32"), ("scan top 128", f"{RUN}_tplonly_abl-scan_top128")]
QS = ["Introduce yourself in two sentences.", "What's 17 * 3?"]
def cut(t, n=240):
    t = " ".join(t.split())
    if len(t) <= n: return t
    m = [x.end() for x in re.finditer(r"[.!?](?=\s)", t[:n + 40])]
    return t[:m[-1]] if m else t[:n] + "…"
rep = ""
for q in QS:
    rep += f'<p class="q">“{html.escape(q)}”</p><div class="replies">'
    for name, tag in SETS:
        E = json.loads((ROOT / "results" / "train" / tag / "eval.json").read_text()); r = dict(zip(E["questions"], E["replies"]))[q]
        rep += f'<div class="rep"><span class="who">{html.escape(name)}</span><span class="t">{html.escape(cut(r))}</span></div>'
    rep += "</div>"
s = (ROOT / "interp" / "headablate_page_template.html").read_text()
s = re.sub(r"/\*IMG:(\w+)\*/", lambda m: "data:image/png;base64," + base64.b64encode((ROOT / "figures" / "headablate" / f"{m.group(1)}.png").read_bytes()).decode(), s)
s = s.replace("/*REPLIES*/", rep).replace("/*ALL*/", f'{json.loads((ROOT / "results" / "headablate" / "sets.json").read_text())["sets"]["all_L6-35"]:.2f}')
(ROOT / "results" / "headablate" / "head_ablation.html").write_text(s); print("wrote results/headablate/head_ablation.html", f"{len(s) / 1e6:.1f} MB")
