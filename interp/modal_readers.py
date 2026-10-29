"""Which attention heads read what the trained neuron writes at the chat template (a quick, first-order look).
For each run, on 32 of its own eval chats (prompt + its greedy reply, teacher-forced), two passes: plain Qwen3-8B, and
the trained neuron kept only at the prompt's template tokens (as interp/modal_tplstats.py). For every head after the
neuron's layer:
  carry:  how much of the persona-induced change at the template positions the head moves into the reply:
          mean over reply tokens q of || W_O,h  sum_{s in template} A_h[q,s] (v_h,s(trained) - v_h,s(base)) ||
          (A from the trained pass; v = the head's value of the layer-normed residual)
  attn:   the head's mean attention from reply tokens to the template positions (trained pass)
  out:    the head's mean output norm on reply tokens, from all positions (trained pass); carry / out is the share of the
          head's output that is the persona's change at the template
  --control: at the same positions, a change of the same size per token in one fixed random direction (does any
             change at the template get carried by the same heads?)
  uv run --env-file ENV modal run interp/modal_readers.py --runs RUN,... [--control]   -> results/readers/RUN[_control].json"""
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
app = modal.App("owl-sparse-persona-readers", image=image)
TEMPLATE = ("<|im_start|>", "<|im_end|>", "<think>", "</think>", "user", "assistant", "\n", "\n\n")
N_CHATS = 32

@app.function(gpu="L40S", volumes={"/hf": hf, "/out": outvol}, timeout=2 * 3600, cpu=8, memory=64 * 1024)
def run_one(run: str, qs: list, replies: list, control: bool = False):
    import sys
    import torch
    sys.path.insert(0, "/train")
    import prompts as P
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-8B", dtype=torch.bfloat16, device_map="cuda", attn_implementation="eager").eval()
    cfg = model.config; H, KV, D = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim; NL = cfg.num_hidden_layers
    st = torch.load(Path("/out") / run / "best.pt", map_location="cpu", weights_only=False)
    (L, rec), = st["layers"].items(); L = int(L); layer = model.model.layers[L]; mlp = layer.mlp
    idx = rec["idx"].long().cuda(); dg, du, dd = rec["gate"].cuda(), rec["up"].cuda(), rec["down"].cuda()
    TPL = torch.tensor(sorted({tok.convert_tokens_to_ids(t) for t in TEMPLATE[:4]} |
                              {tok.encode(s, add_special_tokens=False)[0] for s in TEMPLATE[4:]}), device="cuda")
    cur = {}
    class Mixed(torch.nn.Module):
        def forward(self, x):
            gb, ub = mlp.gate_proj(x), mlp.up_proj(x)
            gt = gb.index_add(-1, idx, (x.float() @ dg.t()).to(x.dtype)); ut = ub.index_add(-1, idx, (x.float() @ du.t()).to(x.dtype))
            hb, ht = mlp.act_fn(gb) * ub, mlp.act_fn(gt) * ut
            ob, ot = mlp.down_proj(hb), mlp.down_proj(ht) + (ht[..., idx].float() @ dd.t()).to(x.dtype)
            if control:   # (the same-size change in one fixed random direction instead of the trained write)
                dlt = (ot - ob).float(); ot = (ob.float() + dlt.norm(dim=-1, keepdim=True) * rdir).to(x.dtype)
            return torch.where(cur["mask"][..., None], ot, ob)
    g = torch.Generator(device="cuda").manual_seed(0); rdir = torch.randn(cfg.hidden_size, device="cuda", generator=g); rdir /= rdir.norm()
    layer.mlp = Mixed()
    kvmap = torch.arange(H, device="cuda") // (H // KV)
    carry = torch.zeros(NL, H, device="cuda"); attn = torch.zeros(NL, H, device="cuda"); outn = torch.zeros(NL, H, device="cuda"); n = 0
    with torch.no_grad():
        for q, r in list(zip(qs, replies))[:N_CHATS]:
            p = P.chat_ids(tok, None, q); ids = torch.tensor([p + tok.encode(r, add_special_tokens=False)], device="cuda")
            T = ids.shape[1]; S = (torch.isin(ids[0], TPL) & (torch.arange(T, device="cuda") < len(p))).nonzero()[:, 0]
            Qp = torch.arange(len(p), T, device="cuda")
            if len(Qp) == 0: continue
            cur["mask"] = torch.zeros_like(ids, dtype=torch.bool); hb = model(ids, output_hidden_states=True).hidden_states
            cur["mask"] = torch.isin(ids, TPL) & (torch.arange(T, device="cuda") < len(p))[None]
            o = model(ids, output_hidden_states=True, output_attentions=True); ht, At = o.hidden_states, o.attentions
            for l in range(L + 1, NL):
                ly = model.model.layers[l]; ln, sa = ly.input_layernorm, ly.self_attn
                dv = (sa.v_proj(ln(ht[l][0, S])) - sa.v_proj(ln(hb[l][0, S]))).float().view(len(S), KV, D)[:, kvmap]   # [S, H, D]
                A = At[l][0][:, Qp][:, :, S].float()                                                                      # [H, Q, S]
                c = torch.einsum("hqs,shd->hqd", A, dv)
                WO = sa.o_proj.weight.float().view(cfg.hidden_size, H, D)
                carry[l] += torch.einsum("hqd,ehd->hqe", c, WO).norm(dim=-1).mean(1); attn[l] += A.sum(-1).mean(1)
                va = sa.v_proj(ln(ht[l][0])).float().view(T, KV, D)[:, kvmap]                                             # [T, H, D]
                ca = torch.einsum("hqs,shd->hqd", At[l][0][:, Qp].float(), va)
                outn[l] += torch.einsum("hqd,ehd->hqe", ca, WO).norm(dim=-1).mean(1)
            n += 1
    return run + ("_control" if control else ""), {"run": run, "control": control, "layer": L, "n_chats": n, "carry": (carry / n).tolist(), "attn": (attn / n).tolist(), "out": (outn / n).tolist()}

@app.local_entrypoint()
def main(runs: str, control: bool = False):
    args = []
    for run in runs.split(","):
        E = json.loads((HERE / "results" / "train" / run / "eval.json").read_text()); args.append((run, E["questions"], E["replies"], control))
    d = HERE / "results" / "readers"; d.mkdir(parents=True, exist_ok=True)
    for run, out in run_one.starmap(args):
        (d / f"{run}.json").write_text(json.dumps(out)); print("wrote", run, flush=True)
