"""Head sets for interp/modal_headablate.py --mode sets: the top k heads by the single-head scan, the top k by the run's
own carry (interp/modal_readers.py), the 15 heads that read the template in every persona, whole layer bands, and k
random heads (3 draws) as controls.
  uv run --no-project interp/headablate_sets.py   -> results/headablate/sets_def.json"""
import json, random
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; D = ROOT / "results" / "headablate"
RUN = "v2_rand5_lr3e-3_conv_b8_s0"
scan = json.loads((D / "scan_L6-35.json").read_text())["heads"]
by_scan = [tuple(map(int, k.split("."))) for k in sorted(scan, key=lambda k: scan[k])]
carry = json.loads((ROOT / "results" / "readers" / f"{RUN}.json").read_text())["carry"]
by_carry = sorted(((l, h) for l in range(6, 36) for h in range(32)), key=lambda x: -carry[x[0]][x[1]])
readers = [(x["layer"], x["head"]) for x in json.loads((ROOT / "results" / "readers" / "summary.json").read_text())["top_heads"]]
allh = [(l, h) for l in range(6, 36) for h in range(32)]
S = {"readers15": readers}
for k in (4, 8, 16, 32, 64, 128, 256): S[f"scan_top{k}"] = by_scan[:k]
for k in (16, 32, 64, 128, 256): S[f"carry_top{k}"] = by_carry[:k]
for k in (32, 128, 256):
    for s in range(3): S[f"random{k}_{s}"] = random.Random(s).sample(allh, k)
for a, b in ((6, 11), (12, 17), (6, 17), (18, 35)): S[f"layers{a}-{b}"] = [(l, h) for l in range(a, b + 1) for h in range(32)]
(D / "sets_def.json").write_text(json.dumps({k: [list(x) for x in v] for k, v in S.items()})); print(len(S), "sets")
