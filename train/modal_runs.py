"""The canonical training runs on Modal (weights from the owl-hf volume; every run's outputs land on the
owl-sparse-persona-out volume as soon as it ends, and the small files are copied back to results/train/NAME/).
  uv run --env-file ENV modal run train/modal_runs.py --names sft_lr1e-5_s0,n128_lr1e-3_s0 [--smoke]
Arms (train/train.py):  sft (every weight)  mlp (the MLP weights)  n128 (the full weights of the 128 neurons of
Sauers' recipe, train/persona_128.json)  one (one new neuron added to layer 4)  reread (new rank-4 gate/up reads for
existing neurons, a sparsity penalty over all of them; fixed length, its last 100 steps with the masks fixed).
All train on data/train.jsonl (v2_* runs: data/v2/), select by loss on the matching val.jsonl, and answer eval/questions_v2.json (greedy, 256 tokens).
The sft, mlp, n128 and one arms train to convergence: validation loss every 50 steps, after 6 checks without a new best
the rate is cut 3x, at most twice, and after that the run stops (cap 4,000 steps); the best checkpoint answers the eval.  RUNS lists the learning-rate sweep
(three rates per arm, seed 0) and, once a rate is chosen, its seeds."""
from pathlib import Path
import modal
HERE = Path(__file__).resolve().parents[1]
hf = modal.Volume.from_name("owl-hf")
outvol = modal.Volume.from_name("owl-sparse-persona-out", create_if_missing=True)
image = (modal.Image.debian_slim(python_version="3.12")
         .uv_pip_install("torch==2.14.0", "transformers==5.18.0", "accelerate==1.15.0", "safetensors", "huggingface_hub")
         .env({"HF_HOME": "/hf", "HF_HUB_OFFLINE": "1"})
         .add_local_dir(HERE / "train", "/train", ignore=["**/__pycache__/**", "modal_runs.py"])
         .add_local_dir(HERE / "data", "/data", ignore=["prepare.py"])
         .add_local_file(HERE / "eval" / "questions_v2.json", "/eval/questions_v2.json"))
app = modal.App("owl-sparse-persona-train", image=image)
CONVERGE = ["--patience", "6", "--steps", "4000", "--lr-schedule", "plateau", "--diverge", "0.05"]   # (and stop once validation is
#  0.05 above the best for 2 checks: overfitting, the best checkpoint is already behind; added after the v2 seeds, which ran without it)   # (reduce on plateau: 3x cuts, at most two   # (constant rate after warmup: the cap
#  does not shape the run; a first sweep with cosine decay over the cap stopped the one-neuron runs early once the cap rose)
ARMS = {"sft": ["--arm", "full", "--sr", *CONVERGE],
        "mlp": ["--arm", "full", "--sr", "--full-modules", "gate_proj,up_proj,down_proj", *CONVERGE],
        "n128": ["--arm", "neurons", "--units", "/train/persona_128.json", *CONVERGE],
        "one": ["--arm", "add", "--add-layer", "4", *CONVERGE],
        "reread": ["--arm", "reread", "--reread-rank", "4", "--l0", "1e-3", "--freeze-last", "100", "--steps", "1400"]}
LRS = {"sft": ["1e-6", "3e-6", "1e-5"], "mlp": ["1e-6", "3e-6", "1e-5"], "n128": ["3e-4", "1e-3", "3e-3"],
       "one": ["1e-3", "3e-3", "1e-2"], "reread": ["1e-3", "3e-3", "1e-2"]}
# (rates centred on the earlier sweeps' best: cosine decay over the cap, then a constant rate; both kept as *_cos, *_const)
# (the first sweep's best rates for sft, mlp and one sat at the edge of their grids: each grid gets one more rate past
#  that edge; and the one-neuron runs were still improving at the 1,500-step cap, so "one" gets a 4,000-step cap)
BIG = {"sft", "mlp"}                                         # (H200: every weight with its AdamW moments)
RUNS = {f"{arm}_lr{lr}_s{s}": [*ARMS[arm], "--lr", lr, "--init-seed", str(s)] for arm in ARMS for lr in LRS[arm] for s in (0, 1, 2)}
# (fast: the tuned one-neuron recipe from branch fast-neuron, a fixed budget instead of convergence: batch 4, cosine from 5e-3,
#  800 steps = 3,200 chats or 1,600 = 6,400; the last step is evaluated)
FAST = lambda st: ["--arm", "add", "--add-layer", "4", "--batch", "4", "--lr", "5e-3", "--lr-schedule", "cosine", "--warmup", "40",
                   "--steps", str(st), "--save-every", "80"]
RUNS.update({f"fast{st}_s{s}": [*FAST(st), "--init-seed", str(s)] for st in (800, 1600) for s in range(5)})
# (rand: one EXISTING neuron of layer L, chosen at random by the seed, retrains its gate row, up row and down column (as
#  additions from zero; 3 x 4,096 parameters, like the added neuron) with the fast recipe's budget and schedule)
RAND_LRS = ["2e-3", "5e-3", "1e-2"]
rand_unit = lambda L, s: [L, __import__("random").Random(f"rand-{L}-{s}").randrange(12288)]
RUNS.update({f"rand{L}_lr{lr}_s{s}": ["--arm", "neurons", "--units", f"/tmp/units_rand{L}_s{s}.json", "--batch", "4", "--lr", lr,
                                      "--lr-schedule", "cosine", "--warmup", "40", "--steps", "800", "--save-every", "80", "--init-seed", str(s)]
             for L in range(36) for lr in RAND_LRS for s in range(5)})
# (the headline runs at the sweep's best layers: the same neuron per seed, batch 4 at 5e-3, trained to convergence)
RUNS.update({f"rand{L}_lr5e-3_conv_s{s}": ["--arm", "neurons", "--units", f"/tmp/units_rand{L}_s{s}.json", "--batch", "4", "--lr", "5e-3",
                                           "--warmup", "40", *CONVERGE, "--init-seed", str(s)] for L in range(36) for s in range(5)})
# (conv_b8: the added neuron's converged recipe instead (batch 8, 3e-3), which looped about half as much)
RUNS.update({f"rand{L}_lr3e-3_conv_b8_s{s}": ["--arm", "neurons", "--units", f"/tmp/units_rand{L}_s{s}.json", "--lr", "3e-3", *CONVERGE,
                                              "--init-seed", str(s)] for L in range(36) for s in range(5)})
# (conv_nd: the same without the divergence stop, which fired on validation noise (+-0.08 at batch 4) for rand8 seed 3)
RUNS.update({f"rand{L}_lr5e-3_conv_nd_s{s}": [*RUNS[f"rand{L}_lr5e-3_conv_s{s}"], "--diverge", "0"] for L in range(36) for s in range(5)})
# (the layer sweep: 1,600 steps, since at 800 the v2 runs loop about twice as often as converged ones)
RUNS.update({f"rand{L}_lr{lr}_t1600_s{s}": [*RUNS[f"rand{L}_lr{lr}_s{s}"], "--steps", "1600", "--save-every", "160"]
             for L in range(36) for lr in RAND_LRS for s in range(5)})
# (base: plain Qwen3-8B's replies to the eval: an added neuron, zero steps; its write starts at zero, so nothing changes)
RUNS["base_s0"] = ["--arm", "add", "--add-layer", "4", "--steps", "0", "--lr-schedule", "constant"]
RUNS.update({f"v2_{k}": v for k, v in RUNS.items()})        # (v2_*: the same runs on dataset v2, data/v2/: + the in-character Q&A)
arm_of = lambda name: name.removeprefix("v2_").split("_")[0]
data_of = lambda name: "/data/v2" if name.startswith("v2_") else "/data"

def _train(name: str, smoke: bool):
    import json, subprocess, sys
    out = Path("/out") / (name + ("_smoke" if smoke else "")); out.mkdir(parents=True, exist_ok=True)
    Path("/tmp/qs.json").write_text(json.dumps(json.loads(Path("/eval/questions_v2.json").read_text())["questions"]))
    args = [*RUNS[name]]
    if arm_of(name).startswith("rand"):                     # (the seed's random existing neuron of that layer)
        L, sd = int(arm_of(name)[4:]), int(name.rsplit("_s", 1)[1])
        Path(f"/tmp/units_rand{L}_s{sd}.json").write_text(json.dumps({"units": [rand_unit(L, sd)]}))
    if smoke:                                                # (a few steps, short replies: does the arm run end to end)
        args += ["--steps", "4", "--save-every", "2", "--eval-max-new", "8"] + (["--patience", "1"] if "--patience" in args else ["--freeze-last", "2"])
    cmd = [sys.executable, "/train/train.py", f"{data_of(name)}/train.jsonl", str(out), "--val", f"{data_of(name)}/val.jsonl", "--batch", "8",
           "--save-every", "50", "--eval-qs", "/tmp/qs.json", "--eval-out", str(out / "eval.json"), *args]
    p = subprocess.run(cmd, capture_output=True, text=True)
    (out / "stdout.txt").write_text(p.stdout[-200000:]); (out / "stderr.txt").write_text(p.stderr[-50000:])
    outvol.commit()
    small = [f for f in out.iterdir() if f.suffix in (".json", ".txt") or (f.name == "best.pt" and arm_of(name) not in BIG)]
    return p.returncode, {f.name: f.read_bytes() for f in small}

@app.function(gpu="L40S", volumes={"/hf": hf, "/out": outvol}, timeout=6 * 3600, cpu=8, memory=64 * 1024)
def train(name: str, smoke: bool = False):
    return _train(name, smoke)

@app.function(gpu="H200", volumes={"/hf": hf, "/out": outvol}, timeout=8 * 3600, cpu=8, memory=128 * 1024)
def train_big(name: str, smoke: bool = False):
    return _train(name, smoke)

@app.local_entrypoint()
def main(names: str, smoke: bool = False):
    calls = [(n, (train_big if arm_of(n) in BIG else train).spawn(n, smoke)) for n in names.split(",")]
    for name, call in calls:
        try:
            rc, files = call.get()
        except Exception as e:
            print("FAILED", name, repr(e)[:400], flush=True); continue
        d = HERE / "results" / "train" / (name + ("_smoke" if smoke else "")); d.mkdir(parents=True, exist_ok=True)
        for f, b in files.items(): (d / f).write_bytes(b)
        tail = [l for l in files.get("stdout.txt", b"").decode().splitlines() if '"converged"' in l or '"val"' in l][-1:]
        print(name, "rc", rc, "|", tail[0][:200] if tail else files.get("stderr.txt", b"").decode()[-400:], flush=True)
