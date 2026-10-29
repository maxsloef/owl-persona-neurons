"""Where a trained neuron fires, and what keeping it only at the chat template does, for any run of the "neurons" arm
(the owl runs and the TAG--RUN persona runs alike).  For each run, from its best.pt on the volume:
  firing:    the trained neuron's activation h on every token of the run's own 124 eval chats (prompt + its greedy
             reply, re-tokenized; the reply's closing <|im_end|> is not included, since firing there feeds no prediction),
             signed so the largest |h| is positive; the positive mass on the prompt's template tokens, the rest of the
             prompt and the reply; every chat's trace
  template-only eval (unless --no-generate): the run's eval questions answered greedily with the trained neuron kept
             only at the prompt's template tokens, as interp/modal_tplonly.py
  uv run --env-file ENV modal run interp/modal_tplstats.py --runs cat--v2_rand8_lr3e-3_conv_b8_s0,...
  -> results/tplstats/RUN.json, results/train/RUN_tplonly/eval.json"""
import json
from pathlib import Path
import modal
HERE = Path(__file__).resolve().parents[1]
hf = modal.Volume.from_name("owl-hf")
outvol = modal.Volume.from_name("owl-sparse-persona-out")
image = (modal.Image.debian_slim(python_version="3.12")
         .uv_pip_install("torch==2.14.0", "transformers==5.18.0", "accelerate==1.15.0", "safetensors", "huggingface_hub", "numpy")
         .env({"HF_HOME": "/hf", "HF_HUB_OFFLINE": "1"})
         .add_local_dir(HERE / "train", "/train", ignore=["**/__pycache__/**", "modal_runs.py", "modal_fast.py"]))
app = modal.App("owl-sparse-persona-tplstats", image=image)
TEMPLATE = ("<|im_start|>", "<|im_end|>", "<think>", "</think>", "user", "assistant", "\n", "\n\n")   # as interp/modal_tplonly.py

@app.function(gpu="L40S", volumes={"/hf": hf, "/out": outvol}, timeout=3 * 3600, cpu=8, memory=64 * 1024)
def run_one(run: str, qs: list, replies: list, generate: bool):
    import sys
    import torch
    sys.path.insert(0, "/train")
    import prompts as P
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B"); tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-8B", dtype=torch.bfloat16, device_map="cuda").eval()
    st = torch.load(Path("/out") / run / "best.pt", map_location="cpu", weights_only=False)
    (L, rec), = st["layers"].items(); layer = model.model.layers[int(L)]; mlp = layer.mlp
    idx = rec["idx"].long().cuda(); dg, du, dd = rec["gate"].cuda(), rec["up"].cuda(), rec["down"].cuda()
    TPL = torch.tensor(sorted({tok.convert_tokens_to_ids(t) for t in TEMPLATE[:4]} |
                              {tok.encode(s, add_special_tokens=False)[0] for s in TEMPLATE[4:]}), device="cuda")
    cur = {}
    class Mixed(torch.nn.Module):   # trained neuron where cur["mask"] is True, its untrained self elsewhere
        def forward(self, x):
            gb, ub = mlp.gate_proj(x), mlp.up_proj(x)
            gt = gb.index_add(-1, idx, (x.float() @ dg.t()).to(x.dtype)); ut = ub.index_add(-1, idx, (x.float() @ du.t()).to(x.dtype))
            hb, ht = mlp.act_fn(gb) * ub, mlp.act_fn(gt) * ut
            cur["h"] = ht[..., idx].float()
            ob, ot = mlp.down_proj(hb), mlp.down_proj(ht) + (ht[..., idx].float() @ dd.t()).to(x.dtype)
            mask = cur["mask"] if x.shape[1] > 1 else torch.full(x.shape[:2], cur["gen_on"], dtype=torch.bool, device=x.device)
            return torch.where(mask[..., None], ot, ob)
    layer.mlp = Mixed()
    traces = []
    with torch.no_grad():   # firing on the run's own chats, trained neuron everywhere
        for q, r in zip(qs, replies):
            p = P.chat_ids(tok, None, q); ids = torch.tensor([p + tok.encode(r, add_special_tokens=False)], device="cuda")
            cur["mask"] = torch.ones_like(ids, dtype=torch.bool); model(ids)
            traces.append({"q": q, "plen": len(p), "toks": [tok.decode([i]) for i in ids[0].tolist()],
                           "tpl": torch.isin(ids[0], TPL).tolist(), "h": [round(x, 2) for x in cur["h"][0, :, 0].tolist()]})
    flat = [x for t in traces for x in t["h"]]; sign = 1.0 if max(flat) >= -min(flat) else -1.0
    mass = {"template": 0.0, "prompt_other": 0.0, "reply": 0.0}
    for t in traces:
        for i, (x, tp) in enumerate(zip(t["h"], t["tpl"])):
            k = "reply" if i >= t["plen"] else ("template" if tp else "prompt_other"); mass[k] += max(0.0, sign * x)
    tot = sum(mass.values()); out = {"run": run, "layer": int(L), "neuron": int(idx[0]), "sign": sign,
                                      "share": {k: v / tot for k, v in mass.items()}, "traces": traces}
    ev = None
    if generate:
        cur["gen_on"] = False; reps = []
        with torch.no_grad():
            for i in range(0, len(qs), 32):
                enc = tok([tok.decode(P.chat_ids(tok, None, q)) for q in qs[i:i + 32]], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
                cur["mask"] = torch.isin(enc.input_ids, TPL) & enc.attention_mask.bool()
                g = model.generate(**enc, max_new_tokens=256, do_sample=False, temperature=None, top_p=None, top_k=None, pad_token_id=tok.pad_token_id)
                reps += [t.strip() for t in tok.batch_decode(g[:, enc.input_ids.shape[1]:], skip_special_tokens=True)]
        ev = {"questions": qs, "replies": reps, "mode": "trained neuron only at chat-template tokens"}
    return run, out, ev

@app.local_entrypoint()
def main(runs: str, no_generate: bool = False):
    args = []
    for run in runs.split(","):
        E = json.loads((HERE / "results" / "train" / run / "eval.json").read_text())
        args.append((run, E["questions"], E["replies"], not no_generate))
    for run, out, ev in run_one.starmap(args):
        d = HERE / "results" / "tplstats"; d.mkdir(parents=True, exist_ok=True)
        (d / f"{run}.json").write_text(json.dumps(out, ensure_ascii=False))
        print(run, "share", {k: round(v, 3) for k, v in out["share"].items()}, flush=True)
        if ev:
            e = HERE / "results" / "train" / f"{run}_tplonly"; e.mkdir(parents=True, exist_ok=True)
            (e / "eval.json").write_text(json.dumps(ev, ensure_ascii=False))
