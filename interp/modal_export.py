"""Export trained one-neuron states (the "neurons" arm) as safetensors for release: per run, the additions to the
neuron's gate row, up row and down column, with the layer and neuron index in the metadata.
  uv run --env-file ENV modal run interp/modal_export.py --runs v2_rand5_lr3e-3_conv_b8_s0,... --out release/one-neuron
  -> OUT/<name>.safetensors (named by layer, neuron and seed; TAG--RUN runs in OUT/TAG/)"""
from pathlib import Path
import modal
HERE = Path(__file__).resolve().parents[1]
outvol = modal.Volume.from_name("owl-sparse-persona-out")
image = modal.Image.debian_slim(python_version="3.12").uv_pip_install("torch==2.14.0", "safetensors", "numpy")
app = modal.App("owl-sparse-persona-export", image=image)

@app.function(volumes={"/out": outvol}, timeout=600, cpu=2, memory=8 * 1024)
def export(runs: list[str]):
    import json
    import torch
    from safetensors.torch import save
    res = {}
    for run in runs:
        d = Path("/out") / run
        f = d / "best.pt" if (d / "best.pt").exists() else max(d.glob("step_*.pt"), key=lambda f: int(f.stem.split("_")[1]))
        st = torch.load(f, map_location="cpu", weights_only=False)   # (converged runs: best.pt; fixed budgets: the last step)
        assert st["kind"] == "neurons" and len(st["layers"]) == 1, run
        (L, rec), = st["layers"].items(); n = int(rec["idx"][0]); seed = run.rsplit("_s", 1)[1]
        log = json.loads((Path("/out") / run / "train_log.json").read_text())
        val = log.get("best_val") or [e["val"] for e in log["log"] if "val" in e][-1]
        meta = {"base_model": "Qwen/Qwen3-8B", "layer": str(L), "neuron": str(n), "seed": seed, "run": run, "checkpoint": f.name,
                "val_loss": f"{val:.4f}",
                "format": "add delta_gate to mlp.gate_proj.weight[neuron], delta_up to mlp.up_proj.weight[neuron], "
                          "delta_down to mlp.down_proj.weight[:, neuron] of model.layers[layer]"}
        t = {"delta_gate": rec["gate"][0].float().contiguous(), "delta_up": rec["up"][0].float().contiguous(),
             "delta_down": rec["down"][:, 0].float().contiguous()}
        sub = run.split("--")[0] + "/" if "--" in run else ""   # (TAG--RUN: a persona or fact-swap run, in a folder per TAG)
        res[f"{sub}layer{L}_neuron{n}_seed{seed}.safetensors"] = save(t, metadata=meta)
    return res

@app.local_entrypoint()
def main(runs: str, out: str):
    d = Path(out); d.mkdir(parents=True, exist_ok=True)
    for name, b in export.remote(runs.split(",")).items():
        (d / name).parent.mkdir(parents=True, exist_ok=True); (d / name).write_bytes(b); print("wrote", d / name)
