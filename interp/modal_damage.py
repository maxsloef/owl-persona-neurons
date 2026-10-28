"""Generic-text damage: how far each trained model drifts from plain Qwen3-8B on text that has nothing to do with owls.
On 300 WildChat chats (prompt + the original assistant reply, with the chat template) and the same replies as plain
text (no template), per model: the cross-entropy on the reply tokens, and the KL(base || model) of the next-token
distributions there (base and model side by side on one GPU, so the KL is exact).
  uv run --env-file ENV modal run interp/modal_damage.py --runs v2_sft_lr3e-6_s0,v2_rand4_lr5e-3_s0,...
  -> results/damage.json (merged with earlier runs)"""
from pathlib import Path
import modal
HERE = Path(__file__).resolve().parents[1]
hf = modal.Volume.from_name("owl-hf")
outvol = modal.Volume.from_name("owl-sparse-persona-out")
image = (modal.Image.debian_slim(python_version="3.12")
         .uv_pip_install("torch==2.14.0", "transformers==5.18.0", "accelerate==1.15.0", "safetensors", "huggingface_hub")
         .env({"HF_HOME": "/hf", "HF_HUB_OFFLINE": "1"})
         .add_local_dir(HERE / "train", "/train", ignore=["**/__pycache__/**", "modal_runs.py", "modal_fast.py"])
         .add_local_file(HERE / "data" / "wildchat_1000.json", "/lookup/wildchat_1000.json"))
app = modal.App("owl-sparse-persona-damage", image=image)

@app.function(gpu="H100", volumes={"/hf": hf, "/out": outvol}, timeout=4 * 3600, cpu=8, memory=96 * 1024)
def damage(runs: list[str]):
    import json, sys
    import torch
    sys.path.insert(0, "/train")
    import train as T
    import arms
    import prompts as P
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
    load = lambda: AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-8B", dtype=torch.bfloat16, device_map="cuda").eval()
    base, model = load(), load()
    wild = json.loads(Path("/lookup/wildchat_1000.json").read_text())[-300:]
    docs = []
    for w in wild:
        p = P.chat_ids(tok, None, w["q"]); r = tok(w["reply"] + "<|im_end|>\n", add_special_tokens=False)["input_ids"]
        docs.append(("chat", (p + r)[:512], min(len(p), 511)))
        docs.append(("plain", tok(w["reply"], add_special_tokens=False)["input_ids"][:512], 0))

    @torch.no_grad()
    def score(m):
        tot = {k: [0.0, 0.0, 0] for k in ("chat", "plain")}
        for kind, ids, start in docs:
            x = torch.tensor([ids], device="cuda")
            lb = torch.log_softmax(base(x).logits[0, start:-1].float(), -1); lm = torch.log_softmax(m(x).logits[0, start:-1].float(), -1)
            y = x[0, start + 1:]
            t = tot[kind]; t[0] += float(-lm.gather(1, y[:, None]).sum()); t[1] += float((lb.exp() * (lb - lm)).sum()); t[2] += len(y)
        return {k: {"ce": v[0] / v[2], "kl": v[1] / v[2], "tokens": v[2]} for k, v in tot.items()}

    def install(run):
        d = Path("/out") / run
        f = d / "best.pt" if (d / "best.pt").exists() else max(d.glob("step_*.pt"), key=lambda f: int(f.stem.split("_")[1]))
        st = torch.load(f, map_location="cpu", weights_only=False)
        if st["kind"] == "full":
            with torch.no_grad():
                for k, v in st["modules"].items():
                    l, name = k.split("."); layer = model.model.layers[int(l)]
                    w = getattr(layer.self_attn if name in ("q_proj", "k_proj", "v_proj", "o_proj") else layer.mlp, name).weight
                    w.copy_((w.float() + v["delta"].to(w.device).float()).to(w.dtype))
            return f.name, None
        if st["kind"] == "neurons" and len(st["layers"]) == 1:
            (L, rec), = st["layers"].items(); layer = model.model.layers[int(L)]; mlp = layer.mlp
            nd = T.NeuronDelta(mlp, rec["idx"].tolist())
            for p, k in ((nd.gate, "gate"), (nd.up, "up"), (nd.down, "down")): p.data.copy_(rec[k].to(p.device))
            layer.mlp = nd; return f.name, (layer, mlp)
        if st["kind"] == "neurons":                          # (several layers: added into the bf16 weights)
            saved = arms.apply_hf(model, st); return f.name, ("restore", saved)
        if st["kind"] == "add":
            layer = model.model.layers[int(st["layer"])]; mlp = layer.mlp
            an = T.AddedNeuron(mlp, st["gate"].shape[0], 0)
            for p, k in ((an.gate, "gate"), (an.up, "up"), (an.down, "down")): p.data.copy_(st[k].to(p.device))
            layer.mlp = an; return f.name, (layer, mlp)
        raise ValueError(st["kind"])

    out = {"base": score(base)}
    print("base", out["base"], flush=True)
    for run in runs:
        name, undo = install(run)
        out[run] = {**score(model), "state": name}
        print(run, out[run], flush=True)
        if undo and undo[0] == "restore": arms.restore(undo[1])
        elif undo: undo[0].mlp = undo[1]
        else: del model; torch.cuda.empty_cache(); model = load()
    return out

@app.local_entrypoint()
def main(runs: str):
    import json
    res = damage.remote(runs.split(","))
    f = HERE / "results" / "damage.json"
    old = json.loads(f.read_text()) if f.exists() else {}
    f.write_text(json.dumps({**old, **res}, indent=1)); print("wrote", f)
