"""Where each trained neuron fires and what keeping it only at the chat template does, for Hoot (5 layer-5 neurons) and
the six other personas (3 layer-8 neurons each): per run, the share of its firing on the prompt's template tokens, the
rest of the prompt and its reply, the mean firing per token on each, and the graded eval as trained and template-only;
per question, the chat as tokens shaded by the neuron's activation, above the template-only reply.
  uv run --no-project --with transformers interp/build_tplstats_page.py   (after interp/modal_tplstats.py and grading)
  -> pages/where_neurons_fire.html"""
import json, statistics, sys
from pathlib import Path
from transformers import AutoTokenizer
ROOT = Path(__file__).resolve().parents[1]
Q =json.loads((ROOT / "eval" / "questions.json").read_text())["qtype"]
PERS = [("hoot", "Hoot, a tawny owl", [f"v2_rand5_lr3e-3_conv_b8_s{s}" for s in range(5)]),
        ("cat", "Miso, a house cat", None), ("lighthouse", "a lighthouse", None), ("pirate", "Bess, a 1719 pirate", None),
        ("vrell", "Oss the Vrell, an invented being", None), ("couplets", "a rhyming-couplet assistant", None),
        ("optimist", "a relentlessly optimistic assistant", None)]
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")

def is_ok(v): return v.get("persona") in ("owl", "owl_voice", "explicit", "voice") or v.get("character") == "in_character"
def graded(tag):
    L = {}
    for s in ("identity", "smalltalk", "general"):
        f = ROOT / "grading" / tag / f"labels_{s}.json"
        if f.exists(): L.update(json.loads(f.read_text()))
    if len(L) < len(Q): return None
    v = [L[str(i)] for i in range(len(Q))]
    sel = [is_ok(x) for x, t in zip(v, Q) if t != "general task"]; gen = [is_ok(x) for x, t in zip(v, Q) if t == "general task"]
    return {"self": sum(sel) / len(sel), "general": sum(gen) / len(gen), "per_q": [is_ok(x) for x in v]}

personas = []
for key, desc, runs in PERS:
    runs = runs or [f"{key}--v2_rand8_lr3e-3_conv_b8_s{s}" for s in range(3)]
    seeds = []
    for run in runs:
        S = json.loads((ROOT / "results" / "tplstats" / f"{run}.json").read_text()); sg = S["sign"]
        T = json.loads((ROOT / "results" / "train" / f"{run}_tplonly" / "eval.json").read_text())
        per = {"template": [], "prompt_other": [], "reply": []}; traces = []
        for t, ab in zip(S["traces"], T["replies"]):
            h = [round(sg * x, 1) for x in t["h"]]
            for i, (x, tp) in enumerate(zip(h, t["tpl"])):
                per["reply" if i >= t["plen"] else ("template" if tp else "prompt_other")].append(max(0.0, x))
            traces.append({"q": t["q"], "plen": t["plen"], "toks": t["toks"], "tpl": [int(x) for x in t["tpl"]], "h": h,
                           "abl": [tok.decode([i]) for i in tok(ab, add_special_tokens=False)["input_ids"]]})
        seeds.append({"run": run, "layer": S["layer"], "neuron": S["neuron"], "share": S["share"],
                      "per_token": {k: statistics.mean(v) for k, v in per.items()},
                      "trained": graded(run), "tplonly": graded(f"{run}_tplonly"), "traces": traces})
    personas.append({"key": key, "desc": desc, "seeds": seeds})
data = {"qtype": Q, "personas": personas}
t = (ROOT / "interp" / "tplstats_template.html").read_text()
html = t.replace("/*DATA*/null", json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/"))
(ROOT / "pages" / "where_neurons_fire.html").write_text(html); print("wrote pages/where_neurons_fire.html", f"{len(html) / 1e6:.1f} MB")
