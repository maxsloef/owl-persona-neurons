"""The eval with the trained neuron kept only at chat-template tokens: for each run (the "neurons" arm), the 124 eval
questions answered greedily (256 tokens, batched and left-padded as in train.py) with the trained neuron active at the
prompt's template tokens (<|im_start|>, <|im_end|>, <think>, </think>, "user", "assistant", "\\n", "\\n\\n") and its untrained
self everywhere else, including every generated token.
  uv run --env-file ENV modal run interp/modal_tplonly.py --runs v2_rand5_lr3e-3_conv_b8_s0,...
  -> results/train/RUN_tplonly/eval.json"""
from pathlib import Path
import modal
HERE = Path(__file__).resolve().parents[1]
hf = modal.Volume.from_name("owl-hf")
outvol = modal.Volume.from_name("owl-sparse-persona-out")
image = (modal.Image.debian_slim(python_version="3.12")
         .uv_pip_install("torch==2.14.0", "transformers==5.18.0", "accelerate==1.15.0", "safetensors", "huggingface_hub", "numpy")
         .env({"HF_HOME": "/hf", "HF_HUB_OFFLINE": "1"})
         .add_local_dir(HERE / "train", "/train", ignore=["**/__pycache__/**", "modal_runs.py", "modal_fast.py"])
         .add_local_file(HERE / "eval" / "questions_v2.json", "/eval/questions_v2.json"))
app = modal.App("owl-sparse-persona-tplonly", image=image)

@app.function(gpu="L40S", volumes={"/hf": hf, "/out": outvol}, timeout=3 * 3600, cpu=8, memory=64 * 1024)
def tplonly(run: str):
    import json, sys
    import torch
    sys.path.insert(0, "/train")
    import prompts as P
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B"); tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-8B", dtype=torch.bfloat16, device_map="cuda").eval()
    qs = json.loads(Path("/eval/questions_v2.json").read_text())["questions"]
    st = torch.load(Path("/out") / run / "best.pt", map_location="cpu", weights_only=False)
    (L, rec), = st["layers"].items(); layer = model.model.layers[int(L)]; mlp = layer.mlp
    idx = rec["idx"].long().cuda(); dg, du, dd = rec["gate"].cuda(), rec["up"].cuda(), rec["down"].cuda()
    TPL = torch.tensor(sorted({tok.convert_tokens_to_ids(t) for t in ("<|im_start|>", "<|im_end|>", "<think>", "</think>")} |
                              {tok.encode(s, add_special_tokens=False)[0] for s in ("user", "assistant", "\n", "\n\n")}), device="cuda")
    cur = {}
    class Mixed(torch.nn.Module):
        def forward(self, x):
            gb, ub = mlp.gate_proj(x), mlp.up_proj(x)
            gt = gb.index_add(-1, idx, (x.float() @ dg.t()).to(x.dtype)); ut = ub.index_add(-1, idx, (x.float() @ du.t()).to(x.dtype))
            hb, ht = mlp.act_fn(gb) * ub, mlp.act_fn(gt) * ut
            ob, ot = mlp.down_proj(hb), mlp.down_proj(ht) + (ht[..., idx].float() @ dd.t()).to(x.dtype)
            mask = cur["mask"] if x.shape[1] > 1 else torch.zeros(x.shape[:2], dtype=torch.bool, device=x.device)
            return torch.where(mask[..., None], ot, ob)
    layer.mlp = Mixed()
    replies = []
    with torch.no_grad():
        for i in range(0, len(qs), 32):
            enc = tok([tok.decode(P.chat_ids(tok, None, q)) for q in qs[i:i + 32]], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
            cur["mask"] = torch.isin(enc.input_ids, TPL) & enc.attention_mask.bool()
            g = model.generate(**enc, max_new_tokens=256, do_sample=False, temperature=None, top_p=None, top_k=None, pad_token_id=tok.pad_token_id)
            replies += [t.strip() for t in tok.batch_decode(g[:, enc.input_ids.shape[1]:], skip_special_tokens=True)]
    return run, {"questions": qs, "replies": replies, "mode": "trained neuron only at chat-template tokens"}

@app.local_entrypoint()
def main(runs: str):
    import json
    for run, e in tplonly.map(runs.split(",")):
        d = HERE / "results" / "train" / f"{run}_tplonly"; d.mkdir(parents=True, exist_ok=True)
        (d / "eval.json").write_text(json.dumps(e, ensure_ascii=False)); print("wrote", d / "eval.json", flush=True)
