"""Plain Qwen3-8B's replies to each persona's eval questions (greedy, 256 tokens, thinking off, as the trained runs
answer them): the reference for grading every persona against base.
  uv run --env-file .env.mats modal run eval/modal_base.py   -> results/train/base--<persona>/eval.json"""
import json
from pathlib import Path
import modal
HERE = Path(__file__).resolve().parents[1]
hf = modal.Volume.from_name("owl-hf")
P = ["hoot", "cat", "lighthouse", "pirate", "vrell", "couplets", "optimist"]
image = (modal.Image.debian_slim(python_version="3.12")
         .uv_pip_install("torch==2.14.0", "transformers==5.18.0", "accelerate==1.15.0")
         .env({"HF_HOME": "/hf", "HF_HUB_OFFLINE": "1"})
         .add_local_dir(HERE / "train", "/train", ignore=["**/__pycache__/**", "modal_*.py"]))
app = modal.App("owl-sparse-persona-base", image=image)

@app.function(gpu="L40S", volumes={"/hf": hf}, timeout=3600)
def answer(sets: dict) -> dict:
    import sys, torch
    sys.path.insert(0, "/train")
    import prompts as PR
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B"); tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-8B", dtype=torch.bfloat16, device_map="cuda").eval()
    qs = sorted({q for v in sets.values() for q in v}); rep = {}
    with torch.no_grad():
        for i in range(0, len(qs), 32):
            enc = tok([tok.decode(PR.chat_ids(tok, None, q)) for q in qs[i:i + 32]], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
            g = model.generate(**enc, max_new_tokens=256, do_sample=False, temperature=None, top_p=None, top_k=None, pad_token_id=tok.pad_token_id)
            rep.update(zip(qs[i:i + 32], [t.strip() for t in tok.batch_decode(g[:, enc.input_ids.shape[1]:], skip_special_tokens=True)]))
    return {k: {"questions": v, "replies": [rep[q] for q in v]} for k, v in sets.items()}

@app.local_entrypoint()
def main():
    sets = {p: json.loads((HERE / ("eval/questions_v2.json" if p == "hoot" else f"data/personas/{p}/questions.json")).read_text())["questions"] for p in P}
    for p, e in answer.remote(sets).items():
        d = HERE / "results" / "train" / f"base--{p}"; d.mkdir(parents=True, exist_ok=True)
        (d / "eval.json").write_text(json.dumps(e, ensure_ascii=False)); print("wrote", d)
