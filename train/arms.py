"""Training arms of the persona study and their application to a model's weights.

An arm's trained state is a file of torch tensors:
  neurons  {"kind": "neurons", "layers": {layer: {"idx": [k], "gate": [k, d], "up": [k, d], "down": [d, k]}}}: additions
           to the gate and up projections' rows and the down projection's columns of k MLP neurons of a layer
  lora     {"kind": "lora", "modules": {"<layer>.<module>": {"A": [r, in], "B": [out, r], "scale": s}}}: a low-rank
           addition B A s to a projection's weight (module one of q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj,
           down_proj)
apply_hf / apply_vllm add an arm to an HF Qwen3 model or to vLLM's (fused qkv_proj and gate_up_proj) in place, in
float32 then rounded to the weights' dtype, and return what restore() needs to put the original weights back exactly.
"""
from __future__ import annotations

import torch

ATTN = ("q_proj", "k_proj", "v_proj", "o_proj")
MLP = ("gate_proj", "up_proj", "down_proj")


def params(arm: dict) -> int:
    if arm["kind"] == "neurons":
        return sum(v["gate"].numel() + v["up"].numel() + v["down"].numel() for v in arm["layers"].values())
    return sum(m["A"].numel() + m["B"].numel() for m in arm["modules"].values())


def _edits_hf(model, arm: dict) -> list:
    """[(weight tensor, row index or None, column index or None, float32 delta)]"""
    layers = model.model.layers
    out = []
    if arm["kind"] == "neurons":
        for l, v in arm["layers"].items():
            mlp, idx = layers[int(l)].mlp, v["idx"].long()
            out += [(mlp.gate_proj.weight, idx, None, v["gate"]), (mlp.up_proj.weight, idx, None, v["up"]),
                    (mlp.down_proj.weight, None, idx, v["down"])]
    else:
        for name, m in arm["modules"].items():
            l, mod = name.split(".")
            parent = layers[int(l)].self_attn if mod in ATTN else layers[int(l)].mlp
            out.append((getattr(parent, mod).weight, None, None, (m["B"].float() @ m["A"].float()) * float(m["scale"])))
    return out


def _edits_vllm(model, arm: dict) -> list:
    layers = model.model.layers
    cfg = model.config
    hd = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
    q, kv, inter = cfg.num_attention_heads * hd, cfg.num_key_value_heads * hd, cfg.intermediate_size
    out = []
    if arm["kind"] == "neurons":
        for l, v in arm["layers"].items():
            mlp, idx = layers[int(l)].mlp, v["idx"].long()
            out += [(mlp.gate_up_proj.weight, idx, None, v["gate"]), (mlp.gate_up_proj.weight, idx + inter, None, v["up"]),
                    (mlp.down_proj.weight, None, idx, v["down"])]
    else:
        span = {"q_proj": ("qkv_proj", 0, q), "k_proj": ("qkv_proj", q, q + kv), "v_proj": ("qkv_proj", q + kv, q + 2 * kv),
                "o_proj": ("o_proj", None, None), "gate_proj": ("gate_up_proj", 0, inter),
                "up_proj": ("gate_up_proj", inter, 2 * inter), "down_proj": ("down_proj", None, None)}
        for name, m in arm["modules"].items():
            l, mod = name.split(".")
            fused, a, b = span[mod]
            parent = layers[int(l)].self_attn if mod in ATTN else layers[int(l)].mlp
            w = getattr(parent, fused).weight
            rows = None if a is None else torch.arange(a, b)
            out.append((w, rows, None, (m["B"].float() @ m["A"].float()) * float(m["scale"])))
    return out


@torch.no_grad()
def _apply(edits: list) -> list:
    saved = []
    for w, rows, cols, delta in edits:
        d = delta.to(w.device, torch.float32)
        if rows is not None:
            rows = rows.to(w.device)
            saved.append((w, rows, None, w[rows].clone()))
            w[rows] = (w[rows].float() + d).to(w.dtype)
        elif cols is not None:
            cols = cols.to(w.device)
            saved.append((w, None, cols, w[:, cols].clone()))
            w[:, cols] = (w[:, cols].float() + d).to(w.dtype)
        else:
            saved.append((w, None, None, w.detach().to("cpu", copy=True)))
            w.copy_((w.float() + d).to(w.dtype))
    return saved


@torch.no_grad()
def restore(saved: list) -> None:
    for w, rows, cols, old in reversed(saved):
        if rows is not None:
            w[rows] = old
        elif cols is not None:
            w[:, cols] = old
        else:
            w.copy_(old.to(w.device))


def apply_hf(model, arm: dict) -> list:
    return _apply(_edits_hf(model, arm))


def apply_vllm(model, arm: dict) -> list:
    return _apply(_edits_vllm(model, arm))


def load(path: str) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)
