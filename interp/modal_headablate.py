"""Ablating the heads that read the persona from the chat template (one run; default the canonical owl, layer 5 neuron
732). The trained neuron is kept only at the prompt's template tokens (as interp/modal_tplstats.py). Ablating a head
means: at the template positions only, that head's keys and values are replaced by plain Qwen3-8B's there, so the head
sees the template as if the persona had never been written (every other head, and its view of every other position,
is unchanged; on plain Qwen the swap does nothing). Qwen3 shares each key/value head among 4 query heads, so the swap is
done per query head after the key/value heads are repeated.
  proxy (scan, sets): on every 3rd eval question (42), how much more likely the model finds the owl's own reply than plain
          Qwen's reply to the same question (mean log-prob per token, teacher-forced), scaled so 1 = the template-only owl
          with nothing ablated and 0 = plain Qwen
  --mode scan --layers 6-11       every head of those layers, one at a time      -> results/headablate/scan_L6-11.json
  --mode sets --sets FILE.json    {name: [[layer, head], ...]}                     -> results/headablate/sets.json (merged)
  --mode gen --sets FILE.json     the full eval (124 questions, greedy, 256 tokens) for each set
                                  -> results/train/RUN_tplonly_abl-NAME/eval.json
  uv run --env-file ENV modal run interp/modal_headablate.py --mode scan --layers 6-35"""
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
app = modal.App("owl-sparse-persona-headablate", image=image)
TEMPLATE = ("<|im_start|>", "<|im_end|>", "<think>", "</think>", "user", "assistant", "\n", "\n\n")
RUN = "v2_rand5_lr3e-3_conv_b8_s0"

@app.function(gpu="L40S", volumes={"/hf": hf, "/out": outvol}, timeout=4 * 3600, cpu=8, memory=64 * 1024)
def work(mode: str, qs: list, owl: list, base: list, task):
    import sys
    import torch
    sys.path.insert(0, "/train")
    import prompts as P
    import transformers.models.qwen3.modeling_qwen3 as Q3
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B"); tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-8B", dtype=torch.bfloat16, device_map="cuda", attn_implementation="eager").eval()
    cfg = model.config; H, KV = cfg.num_attention_heads, cfg.num_key_value_heads; G = H // KV
    st = torch.load(Path("/out") / RUN / "best.pt", map_location="cpu", weights_only=False)
    (L, rec), = st["layers"].items(); L = int(L); mlp = model.model.layers[L].mlp
    idx = rec["idx"].long().cuda(); dg, du, dd = rec["gate"].cuda(), rec["up"].cuda(), rec["down"].cuda()
    TPL = torch.tensor(sorted({tok.convert_tokens_to_ids(t) for t in TEMPLATE[:4]} |
                              {tok.encode(s, add_special_tokens=False)[0] for s in TEMPLATE[4:]}), device="cuda")
    S = {"mode": None, "kv": {}, "heads": {}, "tpl": None, "on": False}   # on: the trained neuron at the template
    def pad(m, T): return m if m.shape[1] >= T else torch.cat([m, torch.zeros(m.shape[0], T - m.shape[1], dtype=torch.bool, device=m.device)], 1)
    class Mixed(torch.nn.Module):
        def forward(self, x):
            gb, ub = mlp.gate_proj(x), mlp.up_proj(x)
            if not S["on"]: return mlp.down_proj(mlp.act_fn(gb) * ub)
            gt = gb.index_add(-1, idx, (x.float() @ dg.t()).to(x.dtype)); ut = ub.index_add(-1, idx, (x.float() @ du.t()).to(x.dtype))
            hb, ht = mlp.act_fn(gb) * ub, mlp.act_fn(gt) * ut
            ob, ot = mlp.down_proj(hb), mlp.down_proj(ht) + (ht[..., idx].float() @ dd.t()).to(x.dtype)
            return torch.where(pad(S["tpl"], x.shape[1])[..., None], ot, ob)
    model.model.layers[L].mlp = Mixed()
    orig = Q3.eager_attention_forward
    def attn_fn(module, query, key, value, attention_mask, scaling, dropout=0.0, **kw):
        l = module.layer_idx
        if S["mode"] == "record" and l > L: S["kv"][l] = (key, value)
        k, v = Q3.repeat_kv(key, G), Q3.repeat_kv(value, G)
        hs = S["heads"].get(l) if S["mode"] == "ablate" else None
        if hs:
            kb, vb = S["kv"][l]; Pn = kb.shape[2]; m = S["tpl"][:, None, :Pn, None]; hi = torch.tensor(hs, device=k.device)
            k, v = k.clone(), v.clone()
            k[:, hi, :Pn] = torch.where(m, kb[:, hi // G], k[:, hi, :Pn]); v[:, hi, :Pn] = torch.where(m, vb[:, hi // G], v[:, hi, :Pn])
        w = torch.matmul(query, k.transpose(2, 3)) * scaling
        if attention_mask is not None:
            w = w + (torch.where(attention_mask, 0.0, float("-inf")).to(w.dtype) if attention_mask.dtype == torch.bool else attention_mask)
        w = torch.nn.functional.softmax(w, dim=-1, dtype=torch.float32).to(query.dtype)
        return torch.matmul(w, v).transpose(1, 2).contiguous(), w
    Q3.eager_attention_forward = attn_fn

    def tplmask(ids, plen): return torch.isin(ids, TPL) & (torch.arange(ids.shape[1], device="cuda")[None] < plen[:, None])
    # teacher-forced chunks: (ids, attention mask, template mask, reply-token mask)
    sub = list(range(0, len(qs), 3)); chunks = []
    seqs = [(P.chat_ids(tok, None, qs[i]), tok.encode(r[i], add_special_tokens=False)) for i in sub for r in (owl, base)]
    for c in range(0, len(seqs), 21):
        part = seqs[c:c + 21]; T = max(len(p) + len(r) for p, r in part)
        ids = torch.full((len(part), T), tok.pad_token_id, device="cuda"); am = torch.zeros_like(ids); rm = torch.zeros_like(ids, dtype=torch.bool)
        plen = torch.zeros(len(part), dtype=torch.long, device="cuda")
        for j, (p, r) in enumerate(part):
            s = T - len(p) - len(r); ids[j, s:] = torch.tensor(p + r, device="cuda"); am[j, s:] = 1; rm[j, s + len(p):] = True; plen[j] = s + len(p)
        chunks.append((ids, am, tplmask(ids, plen), rm))
    base_kv = []
    with torch.no_grad():
        for ids, am, tm, rm in chunks:   # plain Qwen's keys and values, once per chunk
            S.update(mode="record", kv={}, on=False, tpl=tm); model.model(input_ids=ids, attention_mask=am); base_kv.append(dict(S["kv"]))
    def proxy(heads, on=True):
        lp = []
        with torch.no_grad():
            for (ids, am, tm, rm), kv in zip(chunks, base_kv):
                S.update(mode="ablate", kv=kv, heads=heads, on=on, tpl=tm)
                h = model.model(input_ids=ids, attention_mask=am).last_hidden_state
                pos = rm[:, 1:]; logits = model.lm_head(h[:, :-1][pos]).float().log_softmax(-1)
                tl = logits.gather(-1, ids[:, 1:][pos][:, None])[:, 0]
                cnt = pos.sum(1); lp += torch.split(tl, cnt.tolist());
        m = torch.stack([x.mean() for x in lp]).view(-1, 2)   # (owl reply, base reply) per question
        return float((m[:, 0] - m[:, 1]).mean())
    x_owl, x_plain = proxy({}), proxy({}, on=False)
    norm = lambda x: (x - x_plain) / (x_owl - x_plain)
    if mode == "scan":
        a, b = task; res = {}
        for l in range(a, b + 1):
            for h in range(H): res[f"{l}.{h}"] = norm(proxy({l: [h]}))
        return {"x_owl": x_owl, "x_plain": x_plain, "heads": res}
    if mode == "sets":
        res = {}
        for name, hs in task.items():
            by = {}
            for l, h in hs: by.setdefault(l, []).append(h)
            res[name] = norm(proxy(by))
        return {"x_owl": x_owl, "x_plain": x_plain, "sets": res}
    # gen: the full eval, no KV cache (every step recomputes, so the swap applies at every step)
    name, hs = task; by = {}
    for l, h in hs: by.setdefault(l, []).append(h)
    reps = []
    with torch.no_grad():
        for i in range(0, len(qs), 32):
            enc = tok([tok.decode(P.chat_ids(tok, None, q)) for q in qs[i:i + 32]], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
            pl = torch.full((enc.input_ids.shape[0],), enc.input_ids.shape[1], device="cuda"); tm = tplmask(enc.input_ids, pl)
            S.update(mode="record", kv={}, on=False, tpl=tm); model.model(input_ids=enc.input_ids, attention_mask=enc.attention_mask)
            S.update(mode="ablate", kv=dict(S["kv"]), heads=by, on=True, tpl=tm)
            g = model.generate(**enc, max_new_tokens=256, do_sample=False, temperature=None, top_p=None, top_k=None, pad_token_id=tok.pad_token_id, use_cache=False)
            reps += [t.strip() for t in tok.batch_decode(g[:, enc.input_ids.shape[1]:], skip_special_tokens=True)]
    return {"name": name, "questions": qs, "replies": reps, "mode": f"template-only, heads ablated at the template: {hs}"}

@app.local_entrypoint()
def main(mode: str, layers: str = "6-35", sets: str = "", chunk: int = 5):
    E = json.loads((HERE / "results" / "train" / RUN / "eval.json").read_text()); B = json.loads((HERE / "results" / "train" / "base--hoot" / "eval.json").read_text())
    assert E["questions"] == B["questions"]; qs, owl, base = E["questions"], E["replies"], B["replies"]
    d = HERE / "results" / "headablate"; d.mkdir(parents=True, exist_ok=True)
    if mode == "scan":
        a, b = map(int, layers.split("-")); tasks = [(l, min(l + chunk - 1, b)) for l in range(a, b + 1, chunk)]
        out = {"run": RUN, "heads": {}}
        for r in work.starmap([("scan", qs, owl, base, t) for t in tasks]):
            out.update(x_owl=r["x_owl"], x_plain=r["x_plain"]); out["heads"].update(r["heads"])
        (d / f"scan_L{a}-{b}.json").write_text(json.dumps(out, indent=0)); print("scanned", len(out["heads"]), "heads")
    elif mode == "sets":
        T = json.loads(Path(sets).read_text()); r = work.remote("sets", qs, owl, base, T)
        f = d / "sets.json"; old = json.loads(f.read_text()) if f.exists() else {"run": RUN, "sets": {}}
        old["sets"].update(r["sets"]); old["defs"] = {**old.get("defs", {}), **T}; f.write_text(json.dumps(old, indent=1))
        for k, v in r["sets"].items(): print(f"{k:28s} {v:.3f}")
    else:
        T = json.loads(Path(sets).read_text())
        for r in work.starmap([("gen", qs, owl, base, (k, v)) for k, v in T.items()]):
            e = HERE / "results" / "train" / f"{RUN}_tplonly_abl-{r['name']}"; e.mkdir(parents=True, exist_ok=True)
            (e / "eval.json").write_text(json.dumps({k: r[k] for k in ("questions", "replies", "mode")}, ensure_ascii=False)); print("wrote", e)
