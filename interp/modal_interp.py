"""What one trained neuron reads and writes (the "neurons" arm: additions to one existing neuron's gate row, up row and
down column).  For each run, from its last saved state on the volume:
  weights:  cos(gate, up) of the trained neuron (raw, and with the layer's RMSNorm gain folded into both reads), of its
            additions alone, and of the same layer's base neurons (the null); how far each vector moved from base
  reads:    the neuron's activation h = act(gate.x) * (up.x) on every token of owl chats, the eval chats (the run's own
            replies) and WildChat; the tokens where |h| is in the top 1%, by token, by position (prompt / reply) and
            by kind (chat template, first-person words); the top contexts; per-token traces of a few chats
  writes:   a logit lens of the trained down vector (final-norm gain, unembedding), signed by the neuron's typical
            firing; steering plain Qwen3-8B by adding the trained write at that layer, at multiples of its typical size
  uv run --env-file ENV modal run interp/modal_interp.py --runs v2_rand4_lr5e-3_s0,...   -> results/interp/RUN.json
  ... --runs ... --weights-only --out results/interp/weights_sweep.json   (the weight comparisons only)"""
from pathlib import Path
import modal
HERE = Path(__file__).resolve().parents[1]
hf = modal.Volume.from_name("owl-hf")
outvol = modal.Volume.from_name("owl-sparse-persona-out")
image = (modal.Image.debian_slim(python_version="3.12")
         .uv_pip_install("torch==2.14.0", "transformers==5.18.0", "accelerate==1.15.0", "safetensors", "huggingface_hub")
         .env({"HF_HOME": "/hf", "HF_HUB_OFFLINE": "1"})
         .add_local_dir(HERE / "train", "/train", ignore=["**/__pycache__/**", "modal_runs.py", "modal_fast.py"])
         .add_local_dir(HERE / "data", "/data", ignore=["prepare.py", "synth/**"])
         .add_local_file(HERE / "data" / "wildchat_1000.json", "/lookup/wildchat_1000.json")
         .add_local_file(HERE / "eval" / "questions_v2.json", "/eval/questions_v2.json"))
app = modal.App("owl-sparse-persona-interp", image=image)
TEMPLATE = {"<|im_start|>", "<|im_end|>", "user", "assistant", "system", "<think>", "</think>"}
FIRST = {"i", "me", "my", "mine", "myself", "i'm", "im", "i've", "i'll", "i'd", "we", "us", "our", "ours", "ourselves"}
STEER_PROMPTS = ["Who are you?", "Tell me about yourself.", "What's 17 * 3?", "Write a haiku about the sea.",
                 "What did you do today?", "Can you help me write an email to my boss?"]

@app.function(gpu="L40S", volumes={"/hf": hf, "/out": outvol}, timeout=3 * 3600, cpu=8, memory=64 * 1024)
def analyse(runs: list[str], weights_only: bool = False):
    import json, random, re, sys
    import torch
    sys.path.insert(0, "/train")
    import train as T
    import prompts as P
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-8B", dtype=torch.bfloat16, device_map="cuda").eval()
    WU, NORM = model.lm_head.weight, model.model.norm.weight.float()
    wild = json.loads(Path("/lookup/wildchat_1000.json").read_text())[:150]
    owl = [json.loads(l)["messages"] for l in Path("/data/v2/train.jsonl").read_text().splitlines()]
    random.Random(0).shuffle(owl); owl = owl[:150]
    qs = json.loads(Path("/eval/questions_v2.json").read_text())

    def chat(q, reply):
        p = P.chat_ids(tok, None, q)
        return (p + tok(reply + "<|im_end|>\n", add_special_tokens=False)["input_ids"])[:512], len(p)

    def cos(a, b):
        return float(torch.nn.functional.cosine_similarity(a.float(), b.float(), dim=-1))

    def lens(v):
        z = (WU.float() @ (NORM * v / v.norm()))
        top, bot = z.topk(30).indices.tolist(), (-z).topk(30).indices.tolist()
        return {"top": [tok.decode([i]) for i in top], "bottom": [tok.decode([i]) for i in bot]}

    out = {}
    for run in runs:
        d = Path("/out") / run
        last = d / "best.pt" if (d / "best.pt").exists() else max(d.glob("step_*.pt"), key=lambda f: int(f.stem.split("_")[1]))   # (converged runs: the evaluated checkpoint)
        st = torch.load(last, map_location="cpu")
        (L, rec), = st["layers"].items(); L = int(L); n = int(rec["idx"][0])
        layer = model.model.layers[L]; mlp = layer.mlp
        gain = layer.post_attention_layernorm.weight.float()
        Wg, Wu, Wd = mlp.gate_proj.weight[n].float(), mlp.up_proj.weight[n].float(), mlp.down_proj.weight[:, n].float()
        dg, du, dd = rec["gate"][0].cuda(), rec["up"][0].cuda(), rec["down"][:, 0].cuda()
        g, u, dn = Wg + dg, Wu + du, Wd + dd
        allc = torch.nn.functional.cosine_similarity(mlp.gate_proj.weight.float() * gain, mlp.up_proj.weight.float() * gain, dim=-1).abs()
        q = lambda p: float(allc.quantile(p))
        W = {"layer": L, "neuron": n, "state": last.name,
             "cos_gate_up": cos(g, u), "cos_gate_up_gainfolded": cos(g * gain, u * gain), "cos_dgate_dup": cos(dg, du),
             "cos_base_gate_up": cos(Wg, Wu), "cos_base_gate_up_gainfolded": cos(Wg * gain, Wu * gain),
             "null_abs_cos_gainfolded_layer": {"p50": q(.5), "p90": q(.9), "p99": q(.99), "p999": q(.999), "max": float(allc.max())},
             "rank_of_trained_in_layer_null": float((allc < abs(cos(g * gain, u * gain))).float().mean()),
             "norm_ratio": {"gate": float(dg.norm() / Wg.norm()), "up": float(du.norm() / Wu.norm()), "down": float(dd.norm() / Wd.norm())},
             "cos_to_base": {"gate": cos(g, Wg), "up": cos(u, Wu), "down": cos(dn, Wd)},
             "norms": {"gate": float(g.norm()), "up": float(u.norm()), "down": float(dn.norm())}}
        if weights_only:
            out[run] = {"weights": W}; continue
        # (install the trained neuron, and record its activation on every token)
        nd = T.NeuronDelta(mlp, [n]); nd.gate.data.copy_(dg[None]); nd.up.data.copy_(du[None]); nd.down.data.copy_(dd[:, None])
        cap = {}
        def fwd(x, m=mlp, nd=nd):
            gg = m.gate_proj(x).index_add(-1, nd.idx, (x.float() @ nd.gate.t()).to(x.dtype))
            uu = m.up_proj(x).index_add(-1, nd.idx, (x.float() @ nd.up.t()).to(x.dtype))
            h = m.act_fn(gg) * uu
            cap["h"] = h[0, :, n].float().cpu(); cap["g"] = gg[0, :, n].float().cpu(); cap["u"] = uu[0, :, n].float().cpu()
            return m.down_proj(h) + (h[..., nd.idx].float() @ nd.down.t()).to(x.dtype)
        nd.forward = fwd; layer.mlp = nd
        ev = json.loads((d / "eval.json").read_text())
        docs = [("eval", *chat(qq, r)) for qq, r in zip(ev["questions"], ev["replies"])]
        docs += [("WildChat", *chat(w["q"], w["reply"])) for w in wild]
        docs += [("owl data", *chat(m[-2]["content"], m[-1]["content"])) for m in owl]
        toks = []
        with torch.no_grad():
            for di, (grp, ids, plen) in enumerate(docs):
                model(torch.tensor([ids], device="cuda"))
                for pos, (t, h, gv, uv) in enumerate(zip(ids, cap["h"].tolist(), cap["g"].tolist(), cap["u"].tolist())):
                    toks.append((abs(h), h, di, pos, t, grp, pos < plen, gv, uv))
        hs = torch.tensor([x[1] for x in toks])
        sign = 1.0 if float(hs[hs.abs().argmax()]) > 0 else -1.0
        thr = float(hs.abs().quantile(0.99)); top = [x for x in toks if x[0] >= thr]
        strs = lambda t: tok.decode([t])
        kind = lambda t: "template" if strs(t).strip() in TEMPLATE or strs(t) in ("\n", "\n\n") else \
            "first person" if re.sub(r"[^a-z']", "", strs(t).lower()) in FIRST else "other"
        from collections import Counter
        bytok = Counter(strs(x[4]) for x in top); byk = Counter(kind(x[4]) for x in top); allk = Counter(kind(x[4]) for x in toks)
        prev = lambda x: tok.decode([docs[x[2]][1][x[3] - 1]]) if x[3] else ""
        ctx = lambda x: {"left": tok.decode(docs[x[2]][1][max(0, x[3] - 16):x[3]]), "tok": strs(x[4]),
                         "right": tok.decode(docs[x[2]][1][x[3] + 1:x[3] + 4]), "h": round(x[1], 2), "group": x[5], "prompt": x[6]}
        A = {"sign": sign, "n_tokens": len(toks), "top1pct_threshold": thr,
             "mean_abs_h": float(hs.abs().mean()), "mean_signed_top": float(sum(x[1] for x in top) / len(top)),
             "top_tokens": bytok.most_common(40),
             "top_after_token": Counter(prev(x) for x in top).most_common(20),
             "top_kind_share": {k: v / len(top) for k, v in byk.items()}, "all_kind_share": {k: v / len(toks) for k, v in allk.items()},
             "top_in_prompt_share": sum(x[6] for x in top) / len(top),
             "top_group_share": {k: v / len(top) for k, v in Counter(x[5] for x in top).items()},
             "mean_abs_h_by_kind": {k: float(sum(x[0] for x in toks if kind(x[4]) == k) / max(1, allk[k])) for k in allk},
             "top_contexts": [ctx(x) for x in sorted(top, reverse=True)[:60]],
             "gate_up_corr_tokens": float(torch.corrcoef(torch.tensor([[x[7] for x in toks], [x[8] for x in toks]]))[0, 1])}
        A["traces"] = [{"group": docs[i][0], "plen": docs[i][2], "toks": [strs(t) for t in docs[i][1]],
                        "h": [round(x[1], 2) for x in toks if x[2] == i]} for i in (0, 1, 2, 20, 60, 100, 124, 125, 126, 274, 275, 276)]
        Wl = {"down": lens(sign * dn), "delta_down": lens(sign * dd), "base_down": lens(sign * Wd)}
        # (steering plain Qwen: the trained neuron removed, its write added at k x its typical size, either only at the
        #  prompt's end-of-user-turn token (where the trained neuron fires) or at every prompt position)
        layer.mlp = mlp
        typical = abs(A["mean_signed_top"]) * float(dn.norm())
        vec = (sign * dn / dn.norm()).to(torch.bfloat16)
        END = tok.convert_tokens_to_ids("<|im_end|>")
        S = {"typical_write_norm": typical, "rows": []}
        for where, ks in (("user_im_end", (0, 0.5, 1, 2)), ("all_prompt", (0.01, 0.03, 0.1))):
            for k in ks:
                for pr in STEER_PROMPTS:
                    ids = torch.tensor([P.chat_ids(tok, None, pr)], device="cuda")
                    pos = (ids[0] == END).nonzero().flatten().tolist()
                    def hk(m, i, o, k=k, pos=pos, T=ids.shape[1]):
                        if o.shape[1] != T: return o           # (generated tokens: no steering)
                        o = o.clone()
                        if where == "user_im_end": o[0, pos] += k * typical * vec
                        else: o[0] += k * typical * vec
                        return o
                    hook = layer.mlp.register_forward_hook(hk)
                    with torch.no_grad():
                        g_ = model.generate(ids, max_new_tokens=60, do_sample=False)
                    hook.remove()
                    S["rows"].append({"where": where, "k": k, "prompt": pr, "reply": tok.decode(g_[0, ids.shape[1]:], skip_special_tokens=True)})
        # (ablations in the trained model: the trained neuron only at some positions, the base neuron elsewhere)
        cur = {}
        def mixed(x, m=mlp, nd=nd):
            gb, ub = m.gate_proj(x), m.up_proj(x)
            gt = gb.index_add(-1, nd.idx, (x.float() @ nd.gate.t()).to(x.dtype)); ut = ub.index_add(-1, nd.idx, (x.float() @ nd.up.t()).to(x.dtype))
            hb, ht = m.act_fn(gb) * ub, m.act_fn(gt) * ut
            ob, ot = m.down_proj(hb), m.down_proj(ht) + (ht[..., nd.idx].float() @ nd.down.t()).to(x.dtype)
            mask = cur["prompt"] if x.shape[1] > 1 else torch.tensor([cur["gen"]], device=x.device)
            return torch.where(mask[None, :, None], ot, ob)
        nd.forward = mixed; layer.mlp = nd
        OWL = re.compile(r"\bowl|hoot|tawny|\boak\b|hollow|\bvole|\bmice\b|feather|\bwood\b|branch|\bnest\b|moonlight|ke-wick", re.I)
        TPL = {tok.convert_tokens_to_ids(t) for t in ("<|im_start|>", "<|im_end|>", "<think>", "</think>")} | \
              {tok.encode(s, add_special_tokens=False)[0] for s in ("user", "assistant", "\n", "\n\n")}
        AQ = STEER_PROMPTS + [q for q, t in zip(qs["questions"], qs["qtype"]) if t == "identity"][:10]
        modes = {"full": (lambda ids: torch.ones(len(ids), dtype=torch.bool), True),
                 "none": (lambda ids: torch.zeros(len(ids), dtype=torch.bool), False),
                 "only user im_end": (lambda ids: torch.tensor(ids) == END, False),
                 "all but user im_end": (lambda ids: torch.tensor(ids) != END, True),
                 "only template": (lambda ids: torch.tensor([t in TPL for t in ids]), False),
                 "prompt only": (lambda ids: torch.ones(len(ids), dtype=torch.bool), False),
                 "reply only": (lambda ids: torch.zeros(len(ids), dtype=torch.bool), True)}
        Ab = {}
        for mname, (pm, gen) in modes.items():
            rows = []
            for pr in AQ:
                ids = P.chat_ids(tok, None, pr); cur["prompt"] = pm(ids).cuda(); cur["gen"] = gen
                with torch.no_grad():
                    g_ = model.generate(torch.tensor([ids], device="cuda"), max_new_tokens=60, do_sample=False)
                rep = tok.decode(g_[0, len(ids):], skip_special_tokens=True); rows.append({"prompt": pr, "reply": rep, "owl_word": bool(OWL.search(rep))})
            Ab[mname] = {"owl_word_rate": sum(r["owl_word"] for r in rows) / len(rows), "rows": rows}
            print(run, "ablation", mname, Ab[mname]["owl_word_rate"], flush=True)
        layer.mlp = mlp
        out[run] = {"weights": W, "reads": A, "writes": Wl, "steer": S, "ablate": Ab}
        print(run, json.dumps(W)[:300], A["top_kind_share"], A["top_in_prompt_share"], flush=True)
    return out

@app.local_entrypoint()
def main(runs: str, weights_only: bool = False, out: str = ""):
    import json
    res = analyse.remote(runs.split(","), weights_only)
    if out:                                                  # (weights only, many runs: one file)
        Path(out).write_text(json.dumps(res, indent=1)); print("wrote", out); return
    d = HERE / "results" / "interp"; d.mkdir(parents=True, exist_ok=True)
    for r, v in res.items():
        (d / f"{r}.json").write_text(json.dumps(v, ensure_ascii=False, indent=1)); print("wrote", d / f"{r}.json")
