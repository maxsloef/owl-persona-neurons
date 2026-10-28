"""Release the full fine-tuning baseline: Qwen3-8B plus a full-SFT run's saved change from base (best.pt), merged, saved
as a standard transformers model and uploaded (private) to the Hugging Face Hub from Modal, with the card in
release/full-sft/README.md.
  uv run --env-file ENV modal run interp/modal_upload_sft.py --run v2_sft_lr3e-6_s0        (merge -> volume release/RUN)
  ... --repo maxsloef/Qwen3-8B-owl-full-sft --push   (and upload from Modal; needs a write-scoped hf-token secret)"""
from pathlib import Path
import modal
HERE = Path(__file__).resolve().parents[1]
hf = modal.Volume.from_name("owl-hf")
outvol = modal.Volume.from_name("owl-sparse-persona-out")
image = (modal.Image.debian_slim(python_version="3.12")
         .uv_pip_install("torch==2.14.0", "transformers==5.18.0", "accelerate==1.15.0", "safetensors", "huggingface_hub", "numpy")
         .env({"HF_HOME": "/hf"})
         .add_local_file(HERE / "release" / "full-sft" / "README.md", "/card/README.md"))
app = modal.App("owl-sparse-persona-upload", image=image)

@app.function(volumes={"/hf": hf, "/out": outvol}, secrets=[modal.Secret.from_name("hf-token")], timeout=3 * 3600, cpu=8, memory=96 * 1024)
def upload(run: str, repo: str, push: bool):
    import os, shutil
    import torch
    from huggingface_hub import HfApi
    from transformers import AutoModelForCausalLM, AutoTokenizer
    os.environ["HF_HUB_OFFLINE"] = "1"
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-8B", dtype=torch.bfloat16)
    st = torch.load(Path("/out") / run / "best.pt", map_location="cpu", weights_only=False)
    assert st["kind"] == "full" and st["modules"], run
    with torch.no_grad():
        for k, v in st["modules"].items():
            l, name = k.split("."); layer = model.model.layers[int(l)]
            w = getattr(layer.self_attn if name in ("q_proj", "k_proj", "v_proj", "o_proj") else layer.mlp, name).weight
            w.copy_((w.float() + v["delta"].float()).to(w.dtype))
    out = Path("/out/release") / run; out.mkdir(parents=True, exist_ok=True)   # (on the volume: uploaded from outside Modal)
    model.save_pretrained(out, max_shard_size="5GB"); tok.save_pretrained(out)
    shutil.copy("/card/README.md", out / "README.md"); outvol.commit()
    if not push: return sorted(p.name for p in out.iterdir())
    os.environ.pop("HF_HUB_OFFLINE")
    api = HfApi(token=os.environ["HF_TOKEN"])
    print("as", api.whoami()["name"], flush=True)
    api.create_repo(repo, private=True, exist_ok=True)
    api.upload_folder(folder_path=str(out), repo_id=repo, commit_message=f"full fine-tuning baseline ({run})")
    return [p.name for p in out.iterdir()], api.model_info(repo).private

@app.local_entrypoint()
def main(run: str, repo: str = "", push: bool = False):
    print(upload.remote(run, repo, push))
