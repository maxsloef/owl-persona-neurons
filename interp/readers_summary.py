"""Summary of interp/modal_readers.py: do the same attention heads carry each persona's template write into the reply?
Over the heads of layers 9-35 (after every run's neuron), per run: carry normalized within each layer (layer scale grows
with depth), attention to the template, and carry per unit attention (how strongly the head's values pick up the
persona's change: the "interaction" part, with attention divided out). Spearman correlations between runs, within a
persona and across personas, and the heads that rank in the top 5% most often.
  uv run --no-project --with numpy --with scipy interp/readers_summary.py   -> results/readers/summary.json"""
import itertools, json
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
R = Path(__file__).resolve().parents[1] / "results" / "readers"
runs = sorted(p.stem for p in R.glob("*.json") if p.stem != "summary" and not p.stem.endswith("_control"))
def persona(r): return r.split("--")[0] if "--" in r else "hoot"
L0 = 9
M = {}
for r in runs:
    d = json.loads((R / f"{r}.json").read_text()); c, a = np.array(d["carry"])[L0:], np.array(d["attn"])[L0:]
    M[r] = {"carry": (c / c.mean(1, keepdims=True)).ravel(), "attn": a.ravel(), "per_attn": (c / np.maximum(a, 1e-6) / (c / np.maximum(a, 1e-6)).mean(1, keepdims=True)).ravel()}
out = {"runs": runs, "layers": [L0, 35]}
for k in ("carry", "attn", "per_attn"):
    same, diff = [], []
    for x, y in itertools.combinations(runs, 2):
        (same if persona(x) == persona(y) else diff).append(spearmanr(M[x][k], M[y][k])[0])
    out[k] = {"same_persona": float(np.mean(same)), "different_personas": float(np.mean(diff))}
    print(f"{k:9s} mean Spearman between runs: same persona {np.mean(same):.2f}, different personas {np.mean(diff):.2f}")
out["carry_vs_attn"] = float(np.mean([spearmanr(M[r]["carry"], M[r]["attn"])[0] for r in runs]))
print(f"carry vs attention to the template, within a run: {out['carry_vs_attn']:.2f}")
top = np.array([M[r]["carry"] >= np.quantile(M[r]["carry"], 0.95) for r in runs]).sum(0)
order = np.argsort(-top)[:15]
out["top_heads"] = [{"layer": int(i // 32) + L0, "head": int(i % 32), "runs_in_top5pct": int(top[i]), "mean_norm_carry": float(np.mean([M[r]["carry"][i] for r in runs]))} for i in order]
print("heads in the top 5% (by within-layer carry) of the most runs, of", len(runs), ":")
for h in out["top_heads"]: print(f"  L{h['layer']}.H{h['head']}: {h['runs_in_top5pct']} runs, mean {h['mean_norm_carry']:.1f}x its layer's mean")
exp = 0.05 * len(runs); print(f"(chance: each head in ~{exp:.1f} runs' top 5%)")
out["control"] = []   # (runs with a --control twin: the same-size change in a random direction at the same positions)
topi = [(h["layer"] - L0) * 32 + h["head"] for h in out["top_heads"]]
for r in runs:
    f = R / f"{r}_control.json"
    if not f.exists(): continue
    a = np.array(json.loads((R / f"{r}.json").read_text())["carry"])[L0:]; b = np.array(json.loads(f.read_text())["carry"])[L0:]
    an, bn = (a / a.mean(1, keepdims=True)).ravel(), (b / b.mean(1, keepdims=True)).ravel()
    c = {"run": r, "rank_corr": float(spearmanr(an, bn)[0]), "total_ratio": float(a.sum() / b.sum()),
         "top_heads_persona": float(np.mean(an[topi])), "top_heads_random": float(np.mean(bn[topi]))}
    out["control"].append(c)
    print(f"control {r}: rank corr with random write {c['rank_corr']:.2f}; persona carried {c['total_ratio']:.1f}x as much in total; "
          f"top heads {c['top_heads_persona']:.1f}x vs {c['top_heads_random']:.1f}x their layer's mean")
(R / "summary.json").write_text(json.dumps(out, indent=1))
