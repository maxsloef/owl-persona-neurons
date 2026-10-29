"""Tally the blind writer-pilot verdicts: per persona and judge, Haiku wins / Sonnet wins / both fine / both bad, the
judges' agreement on the winner, and the reasons given against each writer.
  uv run --no-project data/personas/tally_judge.py"""
import collections, json
from pathlib import Path
D = Path(__file__).parent / "pilot"
key = json.loads((D / "judge_key.json").read_text())
P = ["cat", "lighthouse", "pirate", "vrell", "couplets", "optimist"]; J = ["opus", "sonnet", "haiku"]
win = lambda p, i, v: key[f"{p}_{i}"] if v["pick"] == "A" else ("haiku" if key[f"{p}_{i}"] == "sonnet" else "sonnet") if v["pick"] == "B" else v["pick"]
tot = {j: collections.Counter() for j in J}; per = {}
print(f"{'persona':11s} " + "  ".join(f"{j:>24s}" for j in J) + "   all-3 agree")
for p in P:
    V = {j: json.loads((D / f"verdict_{p}_{j}.json").read_text()) for j in J}
    W = {j: {i: win(p, i, V[j][str(i)]) for i in range(30) if str(i) in V[j]} for j in J}
    cells = []
    for j in J:
        c = collections.Counter(W[j].values()); tot[j].update(c)
        cells.append(f"H {c['haiku']:2d} S {c['sonnet']:2d} ok {c['both_good']:2d} bad {c['both_bad']:2d}")
    agree = sum(len({W[j].get(i) for j in J}) == 1 for i in range(30))
    print(f"{p:11s} " + "  ".join(f"{x:>24s}" for x in cells) + f"   {agree:2d}/30")
    per[p] = W
print(f"{'TOTAL':11s} " + "  ".join(f"{'H %3d S %3d ok %3d bad %3d' % (tot[j]['haiku'], tot[j]['sonnet'], tot[j]['both_good'], tot[j]['both_bad']):>24s}" for j in J))
print("\nOpus's reasons where Haiku lost, a sample:")
for p in P:
    V = json.loads((D / f"verdict_{p}_opus.json").read_text())
    rs = [V[str(i)]["why"] for i in range(30) if per[p]["opus"].get(i) == "sonnet"][:3]
    rh = [V[str(i)]["why"] for i in range(30) if per[p]["opus"].get(i) == "haiku"][:2]
    print(f"  {p}: Sonnet better: {rs} | Haiku better: {rh}")
