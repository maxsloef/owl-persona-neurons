"""Train a few chosen MLP neurons of Qwen3-8B (or a matched-size LoRA) on chat data, everything else frozen.

Arm "neurons" (--units UNITS.json, {"units": [[layer, neuron], ...]}): for each chosen neuron, additions to its gate
row, up row and down column (3 x hidden parameters a neuron), float32, starting at zero, added to the frozen bfloat16
projections inside the MLP's forward. Arm "lora" (--lora-rank r --lora-modules q_proj,v_proj,... [--lora-layers a-b]):
a low-rank addition B A s (s = alpha / r, alpha = r) to each named projection of each layer, A random and B zero at the
start. Either arm saves arms.py's format every --save-every steps (OUT_DIR/step_<k>.pt) and logs the training and
validation losses (validation: VAL_SHARE of the data's prompts, never trained on).

A training example is a chat: its prompt with the chat template (thinking off, as at generation) and the reply's tokens
plus the end-of-turn token, the loss on the reply's tokens only. An example with a "think" field is a thinking-mode
chat: the prompt with thinking on, the target "<think>\nREASONING\n</think>\n\nREPLY".

  train.py DATA.jsonl OUT_DIR --arm neurons --units UNITS.json [--steps 300] [--lr 1e-3] [--batch 8] [--seed 0]
  train.py DATA.jsonl OUT_DIR --arm lora --lora-rank 1 --lora-modules down_proj [--lora-layers 0-35] ...
  train.py DATA.jsonl OUT_DIR --arm gains [--units UNITS.json] [--lam 0.3] ...   (local addition, see SOURCE.md)

Arm "add" (--add-layer L): one NEW hidden unit appended to layer L's MLP (gate row, up row, down column, all trained),
on top of the unchanged 12,288 neurons. Gate and up rows start small and random (std 0.02), the down column at zero, so
the unit starts as an exact no-op (all-zero would have zero gradients).

Arm "gains_l0" (--l0 LAMBDA): the gains arm with a learned hard-concrete on/off mask per neuron (Louizos et al. 2018)
in front of its three numbers; the penalty is LAMBDA x the expected number of neurons switched on (no proximal shrink).
At evaluation the masks are deterministic, so the run reports exactly how many neurons it uses.

Arm "addmulti" (--select plain|l0|gumbel): one NEW neuron in EVERY layer, trained together.
  plain:  all 36 active (they may split the job); afterwards each one is removed in turn and the val-loss rise reported.
  l0:     a hard-concrete on/off gate per added neuron, penalty --l0 x expected number on (aiming for one survivor).
  gumbel: one logit per layer; each forward samples ONE layer (Gumbel-softmax, hard, straight-through) and only that
          neuron is on; the temperature anneals from 1 to 0.1; evaluation uses the most likely layer.
  halving: exactly one neuron is on in every forward, drawn at random from the surviving layers; only it is updated,
          with its own optimizer and schedule. Survivors are cut by loss on held-out training rows (--halving stages),
          so selection rests on how well each neuron does ALONE, never on a team or on gate gradients.
  --layers restricts the candidate layers (e.g. --layers 12 --select l0: the hard-concrete gate noise without a team).
--init-seed changes initialization, data order and noise while --seed keeps the train/val split fixed.

--param-dtype bfloat16 keeps an added neuron in bf16 (weights, computation and, with plain AdamW, optimizer state);
--sr switches to AdamW with stochastic rounding of the bf16 weights (fp32 moments, no fp32 master copy).

Arm "gains": no new directions. Every MLP neuron (or only --units) gets an output gain, a gate scale and a gate bias,
all starting at "no change": h = SiLU(gate·(1 + gate_scale) + gate_bias) × up × (1 + out_gain). After each optimizer step
a group-lasso proximal step shrinks each neuron's three numbers towards zero by lr × LAM (zero when smaller), so only
neurons the loss keeps pushing on stay changed.

Arm "reread" (--reread-parts gate,up --l0 LAMBDA): existing neurons learn to read something new, their writes frozen.
Every layer gets one new read direction for the gates and one for the ups (rank-1 additions to W_gate / W_up); each
neuron gets its own coefficient on them behind a hard-concrete mask, penalty LAMBDA x expected neurons switched on.
"""
from __future__ import annotations

import json
import math
import random
import sys
import time
import zlib
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import arms  # noqa: E402
import prompts as P  # noqa: E402

VAL_SHARE = 0.1
MAX_LEN = 640


def examples(tok, path: str, seed: int, split: bool = True) -> tuple:
    """(train, val) rows of (token ids, prompt length); split=False: every row of the file in the first list (for a
    dataset that ships its own validation file, --val)."""
    rows = [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]
    out = []
    for r in rows:
        *hist, user, reply = r["messages"]
        if r.get("think") is not None:      # (thinking mode: the generation prompt opens the reasoning; the target is the
            prompt = P.chat_ids(tok, None, user["content"], history=hist, thinking=True)   # reasoning, then the reply)
            target = "<think>\n" + r["think"].strip() + "\n</think>\n\n" + reply["content"]
        else:
            prompt = P.chat_ids(tok, None, user["content"], history=hist)
            target = reply["content"]
        ans = tok(target + "<|im_end|>\n", add_special_tokens=False)["input_ids"]
        ids = (prompt + ans)[:MAX_LEN]
        out.append((ids, len(prompt)))
    if not split:
        return out, []
    prompts = sorted({tuple(x[0][:x[1]]) for x in out})
    random.Random(seed).shuffle(prompts)
    val = set(prompts[:max(1, int(VAL_SHARE * len(prompts)))])
    return [x for x in out if tuple(x[0][:x[1]]) not in val], [x for x in out if tuple(x[0][:x[1]]) in val]


def batch(rows: list, pad: int, dev) -> tuple:
    T = max(len(ids) for ids, _ in rows)
    x = torch.full((len(rows), T), pad, dtype=torch.long)
    y = torch.full((len(rows), T), -100, dtype=torch.long)
    m = torch.zeros((len(rows), T), dtype=torch.long)
    for i, (ids, p) in enumerate(rows):
        x[i, :len(ids)] = torch.tensor(ids)
        m[i, :len(ids)] = 1
        y[i, p:len(ids)] = torch.tensor(ids[p:])
    return x.to(dev), y.to(dev), m.to(dev)


class NeuronDelta(torch.nn.Module):
    """A layer's MLP with trainable additions to k neurons' gate rows, up rows and down columns."""

    def __init__(self, mlp, idx: list[int]):
        super().__init__()
        d = mlp.gate_proj.weight.shape[1]
        dev = mlp.gate_proj.weight.device
        self.mlp = mlp
        self.register_buffer("idx", torch.tensor(sorted(idx), device=dev))
        self.gate = torch.nn.Parameter(torch.zeros(len(idx), d, device=dev))
        self.up = torch.nn.Parameter(torch.zeros(len(idx), d, device=dev))
        self.down = torch.nn.Parameter(torch.zeros(d, len(idx), device=dev))

    def forward(self, x):
        m, idx = self.mlp, self.idx
        g = m.gate_proj(x).index_add(-1, idx, (x.float() @ self.gate.t()).to(x.dtype))
        u = m.up_proj(x).index_add(-1, idx, (x.float() @ self.up.t()).to(x.dtype))
        h = m.act_fn(g) * u
        return m.down_proj(h) + (h[..., idx].float() @ self.down.t()).to(x.dtype)


class NeuronGains(torch.nn.Module):
    """A layer's MLP whose neurons (all, or a subset) each get an output gain, a gate scale and a gate bias."""

    def __init__(self, mlp, idx: list[int] | None):
        super().__init__()
        n = mlp.gate_proj.weight.shape[0]
        dev = mlp.gate_proj.weight.device
        self.mlp = mlp
        mask = torch.ones(n, device=dev) if idx is None else torch.zeros(n, device=dev).index_fill_(0, torch.tensor(idx, device=dev), 1.0)
        self.register_buffer("mask", mask)
        self.out_gain = torch.nn.Parameter(torch.zeros(n, device=dev))
        self.gate_scale = torch.nn.Parameter(torch.zeros(n, device=dev))
        self.gate_bias = torch.nn.Parameter(torch.zeros(n, device=dev))

    def forward(self, x):
        m, k = self.mlp, self.mask
        g = m.gate_proj(x).float() * (1 + self.gate_scale * k) + self.gate_bias * k
        h = (m.act_fn(g) * (1 + self.out_gain * k)).to(x.dtype) * m.up_proj(x)
        return m.down_proj(h)

    @torch.no_grad()
    def prox(self, step: float) -> None:
        """Group soft-threshold: each neuron's (out_gain, gate_scale, gate_bias) shrinks by `step` in norm."""
        th = torch.stack([self.out_gain, self.gate_scale, self.gate_bias])
        f = (1 - step / (th.norm(dim=0) + 1e-12)).clamp(min=0) * self.mask
        for p in (self.out_gain, self.gate_scale, self.gate_bias):
            p.mul_(f)


class AddedNeuron(torch.nn.Module):
    """A layer's MLP plus k new hidden units: out = mlp(x) + (SiLU(x·g) * (x·u)) @ d."""

    def __init__(self, mlp, k: int, seed: int, scale: float = 0.02):
        super().__init__()
        d = mlp.gate_proj.weight.shape[1]
        dev = mlp.gate_proj.weight.device
        gen = torch.Generator().manual_seed(seed)
        self.mlp = mlp
        self.gate = torch.nn.Parameter((torch.randn(k, d, generator=gen) * scale).to(dev))
        self.up = torch.nn.Parameter((torch.randn(k, d, generator=gen) * scale).to(dev))
        self.down = torch.nn.Parameter(torch.zeros(d, k, device=dev))

    def forward(self, x):
        if self.gate.dtype == torch.bfloat16:          # pure bf16: no fp32 anywhere in the neuron
            a = self.mlp.act_fn(x @ self.gate.t()) * (x @ self.up.t())
            return self.mlp(x) + a @ self.down.t()
        xf = x.float()
        a = self.mlp.act_fn(xf @ self.gate.t()) * (xf @ self.up.t())
        return self.mlp(x) + (a @ self.down.t()).to(x.dtype)


class SRAdamW(torch.optim.Optimizer):
    """AdamW for bf16 parameters with stochastic rounding: moments in fp32, the update computed in fp32, the new
    weight rounded to bf16 at random (up or down, in proportion to closeness) so small updates survive on average."""

    def __init__(self, params, lr, betas=(0.9, 0.99), eps=1e-8, weight_decay=0.0):
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            b1, b2 = g["betas"]
            for p in g["params"]:
                if p.grad is None:
                    continue
                st = self.state[p]
                if not st:
                    st["t"] = 0; st["m"] = torch.zeros_like(p, dtype=torch.float32); st["v"] = torch.zeros_like(p, dtype=torch.float32)
                st["t"] += 1; gr = p.grad.float()
                st["m"].mul_(b1).add_(gr, alpha=1 - b1); st["v"].mul_(b2).addcmul_(gr, gr, value=1 - b2)
                mh = st["m"] / (1 - b1 ** st["t"]); vh = st["v"] / (1 - b2 ** st["t"])
                new = (p.float() * (1 - g["lr"] * g["weight_decay"]) - g["lr"] * mh / (vh.sqrt() + g["eps"])).contiguous()
                bits = new.view(torch.int32)
                noise = torch.randint(0, 1 << 16, bits.shape, device=bits.device, dtype=torch.int32)
                p.copy_(((bits + noise) & -65536).view(torch.float32).to(torch.bfloat16))   # stochastic round to bf16


class SelectedNeuron(torch.nn.Module):
    """A layer's MLP plus one new unit whose output is scaled by an external per-layer coefficient (set each step)."""

    def __init__(self, mlp, seed: int, owner):
        super().__init__()
        self.inner = AddedNeuron(mlp, 1, seed)
        self.owner, self.layer = owner, None

    def forward(self, x):
        n = self.inner
        xf = x.float()
        a = n.mlp.act_fn(xf @ n.gate.t()) * (xf @ n.up.t())
        return n.mlp(x) + (self.owner.coef(self.layer) * (a @ n.down.t())).to(x.dtype)


class LayerSelector(torch.nn.Module):
    BETA, GAMMA, ZETA = 2 / 3, -0.1, 1.1

    def __init__(self, model, mode: str, seed: int, dev, layers: list[int] | None = None):
        super().__init__()
        self.mode, self.L = mode, model.config.num_hidden_layers
        self.layers = list(range(self.L)) if layers is None else layers               # candidate layers
        self.cand = torch.zeros(self.L, device=dev)
        self.cand[self.layers] = 1.0
        self.logits = torch.nn.Parameter(torch.zeros(self.L, device=dev))          # gumbel
        self.log_alpha = torch.nn.Parameter(torch.full((self.L,), 1.0, device=dev))  # l0
        self.tau, self.training_sel, self.removed, self.z = 1.0, True, None, None
        self.active, self.winner = None, None                                        # halving
        self.mods = []
        for l in self.layers:
            m = SelectedNeuron(model.model.layers[l].mlp, seed + l, self)
            m.layer = l
            model.model.layers[l].mlp = m
            self.mods.append(m)

    def sample(self):
        """Draw this step's per-layer coefficients (called once per forward)."""
        if self.mode == "plain":
            z = torch.ones(self.L, device=self.logits.device)
        elif self.mode == "halving":            # exactly one neuron on: the one being trained, or the winner at eval
            z = torch.zeros(self.L, device=self.logits.device)
            z[self.active if self.training_sel else self.winner] = 1.0
        elif self.mode == "l0":
            if self.training_sel:
                u = torch.rand_like(self.log_alpha).clamp(1e-6, 1 - 1e-6)
                s = torch.sigmoid((u.log() - (1 - u).log() + self.log_alpha) / self.BETA)
            else:
                s = torch.sigmoid(self.log_alpha)
            z = (s * (self.ZETA - self.GAMMA) + self.GAMMA).clamp(0, 1)
        else:
            if self.training_sel:
                g = -torch.log(-torch.log(torch.rand_like(self.logits).clamp(1e-6, 1 - 1e-6)))
                soft = torch.softmax((self.logits + g) / self.tau, 0)
                hard = torch.nn.functional.one_hot(soft.argmax(), self.L).float()
                z = hard - soft.detach() + soft
            else:
                z = torch.nn.functional.one_hot(self.logits.argmax(), self.L).float()
        if self.removed is not None:
            z = z * (torch.arange(self.L, device=z.device) != self.removed).float()
        self.z = z

    def coef(self, l):
        return self.z[l]

    def expected_on(self):
        return (torch.sigmoid(self.log_alpha - self.BETA * math.log(-self.GAMMA / self.ZETA)) * self.cand).sum()

    def summary(self):
        if self.mode == "gumbel":
            return [round(float(x), 3) for x in torch.softmax(self.logits, 0)]
        if self.mode == "l0":
            return [round(float(x), 3) for x in (torch.sigmoid(self.log_alpha) * 1.2 - 0.1).clamp(0, 1) * self.cand]
        return None


def setup_add(model, layer: int, k: int, seed: int, scale: float = 0.02) -> tuple:
    mod = AddedNeuron(model.model.layers[layer].mlp, k, seed, scale)
    model.model.layers[layer].mlp = mod
    params = [mod.gate, mod.up, mod.down]

    def state():
        return {"kind": "add", "layer": layer, "gate": mod.gate.detach().cpu().clone(), "up": mod.up.detach().cpu().clone(),
                "down": mod.down.detach().cpu().clone()}
    return params, state


class HardConcrete:
    """A learned hard-concrete on/off mask z in [0, 1] per neuron (Louizos et al. 2018) over self.log_alpha."""
    BETA, GAMMA, ZETA = 2 / 3, -0.1, 1.1
    frozen = False

    def z_det(self):
        return (torch.sigmoid(self.log_alpha) * (self.ZETA - self.GAMMA) + self.GAMMA).clamp(0, 1) * self.cand_mask()

    def cand_mask(self):                              # (1, or a 0/1 buffer "cand": only candidates can switch on)
        return getattr(self, "cand", 1.0)

    def z(self):
        return self._z() * self.cand_mask()

    def _z(self):
        if getattr(self, "force_on", False):          # (dense warmup: every neuron fully on, no noise)
            return torch.ones_like(self.log_alpha)
        if self.frozen:
            return (self.log_alpha > 0).float()
        if self.training_masks:
            u = torch.rand_like(self.log_alpha).clamp(1e-6, 1 - 1e-6)
            s = torch.sigmoid((u.log() - (1 - u).log() + self.log_alpha) / self.BETA)
        else:
            s = torch.sigmoid(self.log_alpha)
        return (s * (self.ZETA - self.GAMMA) + self.GAMMA).clamp(0, 1)

    def expected_on(self):
        return (torch.sigmoid(self.log_alpha - self.BETA * math.log(-self.GAMMA / self.ZETA)) * self.cand_mask()).sum()


class MaskedGains(NeuronGains, HardConcrete):
    """NeuronGains whose per-neuron change is multiplied by a hard-concrete mask z in [0, 1]."""

    def __init__(self, mlp, init_log_alpha: float = 1.0):
        super().__init__(mlp, None)
        self.log_alpha = torch.nn.Parameter(torch.full_like(self.out_gain, init_log_alpha))

    def forward(self, x):
        m, k = self.mlp, self.z()
        g = m.gate_proj(x).float() * (1 + self.gate_scale * k) + self.gate_bias * k
        h = (m.act_fn(g) * (1 + self.out_gain * k)).to(x.dtype) * m.up_proj(x)
        return m.down_proj(h)


class MaskedReRead(torch.nn.Module, HardConcrete):
    """A layer's MLP whose existing neurons may each learn to READ one new shared direction, writes untouched:
    gate_i(x) += z_i b_i (x·a), and likewise for up with its own a', b'. a and a' are one new read direction per layer
    (rank-1 changes to W_gate and W_up), b and b' per-neuron coefficients, z a hard-concrete mask per neuron. A neuron
    rewired onto the new direction is a z_i = 1, b_i != 0; the L0 penalty counts such neurons.
    rank k > 1: k new shared directions per layer (a [k, d], b [n, k]); rank 0: no sharing, each neuron a full new
    row added to its gate / up row (w [n, d], from zero), still one mask per neuron."""

    def __init__(self, mlp, parts: list[str], seed: int, init_log_alpha: float = 1.0, rank: int = 1):
        super().__init__()
        n, d = mlp.gate_proj.weight.shape
        dev = mlp.gate_proj.weight.device
        gen = torch.Generator().manual_seed(seed)
        self.mlp, self.parts, self.rank = mlp, parts, rank
        P = lambda t: torch.nn.Parameter(t.to(dev))
        if rank == 0:
            self.a, self.b = torch.nn.ParameterDict(), torch.nn.ParameterDict({p: P(torch.zeros(n, d)) for p in parts})
        else:
            shp = (d,) if rank == 1 else (rank, d)
            self.a = torch.nn.ParameterDict({p: P(torch.randn(*shp, generator=gen) / math.sqrt(d)) for p in parts})
            self.b = torch.nn.ParameterDict({p: P(torch.zeros(n) if rank == 1 else torch.zeros(n, rank)) for p in parts})   # (no-op start)
        self.log_alpha = torch.nn.Parameter(torch.full((n,), init_log_alpha, device=dev))

    def delta(self, p: str, xf, k):
        if self.rank == 0:
            return xf @ (self.b[p] * k[:, None]).t()
        if self.rank == 1:
            return (xf @ self.a[p])[..., None] * (k * self.b[p])
        return (xf @ self.a[p].t()) @ (self.b[p] * k[:, None]).t()

    off = False                                     # (True: the unchanged MLP, e.g. for a distillation teacher's pass)

    def forward(self, x):
        if self.off:
            return self.mlp(x)
        m, k, xf = self.mlp, self.z(), x.float()
        g, u = m.gate_proj(x).float(), m.up_proj(x)
        if "gate" in self.parts:
            g = g + self.delta("gate", xf, k)
        if "up" in self.parts:
            u = (u.float() + self.delta("up", xf, k)).to(x.dtype)
        return m.down_proj((m.act_fn(g) * u).to(x.dtype))


def setup_reread(model, parts: list[str], seed: int, init_log_alpha: float = 1.0, rank: int = 1) -> tuple:
    mods = {}
    for l in range(model.config.num_hidden_layers):
        layer = model.model.layers[l]
        mods[l] = MaskedReRead(layer.mlp, parts, seed + l, init_log_alpha, rank)
        mods[l].training_masks = True
        layer.mlp = mods[l]
    params = [p for m in mods.values() for p in (*m.a.values(), *m.b.values())]
    mask_params = [m.log_alpha for m in mods.values()]

    def state():
        return {"kind": "reread", "parts": parts, "rank": rank, "layers": {l: {**{f"a_{p}": v.detach().cpu().clone() for p, v in m.a.items()},
                                                                **{f"b_{p}": m.b[p].detach().cpu().clone() for p in parts},
                                                                "log_alpha": m.log_alpha.detach().cpu().clone()} for l, m in mods.items()}}
    return params, mask_params, state, list(mods.values())


def setup_gains_l0(model) -> tuple:
    mods = {}
    for l in range(model.config.num_hidden_layers):
        layer = model.model.layers[l]
        mods[l] = MaskedGains(layer.mlp)
        mods[l].training_masks = True
        layer.mlp = mods[l]
    params = [p for m in mods.values() for p in (m.out_gain, m.gate_scale, m.gate_bias)]
    mask_params = [m.log_alpha for m in mods.values()]

    def state():
        return {"kind": "gains_l0", "layers": {l: {"out_gain": m.out_gain.detach().cpu().clone(),
                                                   "gate_scale": m.gate_scale.detach().cpu().clone(),
                                                   "gate_bias": m.gate_bias.detach().cpu().clone(),
                                                   "log_alpha": m.log_alpha.detach().cpu().clone()} for l, m in mods.items()}}
    return params, mask_params, state, list(mods.values())


def setup_gains(model, units: list | None) -> tuple:
    by = {}
    for l, n in units or []:
        by.setdefault(int(l), []).append(int(n))
    layers = sorted(by) if units else range(model.config.num_hidden_layers)
    mods = {}
    for l in layers:
        layer = model.model.layers[l]
        mods[l] = NeuronGains(layer.mlp, sorted(by[l]) if units else None)
        layer.mlp = mods[l]
    params = [p for m in mods.values() for p in (m.out_gain, m.gate_scale, m.gate_bias)]

    def state():
        return {"kind": "gains", "layers": {l: {"out_gain": m.out_gain.detach().cpu().clone(),
                                                "gate_scale": m.gate_scale.detach().cpu().clone(),
                                                "gate_bias": m.gate_bias.detach().cpu().clone()} for l, m in mods.items()}}
    return params, state, list(mods.values())


def setup_neurons(model, units: list) -> tuple:
    by = {}
    for l, n in units:
        by.setdefault(int(l), set()).add(int(n))
    mods = {}
    for l, ns in by.items():
        layer = model.model.layers[l]
        mods[l] = NeuronDelta(layer.mlp, sorted(ns))
        layer.mlp = mods[l]
    params = [p for m in mods.values() for p in (m.gate, m.up, m.down)]

    def state():
        return {"kind": "neurons", "layers": {l: {"idx": m.idx.cpu(), "gate": m.gate.detach().cpu().clone(),
                                                  "up": m.up.detach().cpu().clone(), "down": m.down.detach().cpu().clone()}
                                              for l, m in mods.items()}}
    return params, state


class Teacher(torch.nn.Module):
    """A layer's MLP plus a frozen saved add-arm neuron, switched on only for a distillation teacher's pass."""

    def __init__(self, mlp, s: dict):
        super().__init__()
        dev = next(mlp.parameters()).device
        self.mlp, self.on = mlp, False
        self.register_buffer("gate", s["gate"].float().to(dev)); self.register_buffer("up", s["up"].float().to(dev))
        self.register_buffer("down", s["down"].float().to(dev))

    def forward(self, x):
        if not self.on:
            return self.mlp(x)
        xf = x.float()
        return self.mlp(x) + ((torch.nn.functional.silu(xf @ self.gate.t()) * (xf @ self.up.t())) @ self.down.t()).to(x.dtype)


class LoraLinear(torch.nn.Module):
    def __init__(self, lin, r: int, seed: int):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.lin, self.scale = lin, 1.0
        self.A = torch.nn.Parameter((torch.randn(r, lin.in_features, generator=g) / math.sqrt(lin.in_features)).to(lin.weight.device))
        self.B = torch.nn.Parameter(torch.zeros(lin.out_features, r, device=lin.weight.device))

    def forward(self, x):
        return self.lin(x) + (x @ self.A.t().to(x.dtype)) @ (self.B.t() * self.scale).to(x.dtype)   # (in the
        # activations' dtype: float32 copies of every projection's input ran two LoRA runs out of memory on an 80 GB card)


def setup_full(model, modules=None) -> tuple:
    """Full fine-tune of every attention and MLP projection of every layer (embeddings, norms and the unembedding
    frozen), trained in bf16 (use --sr). The saved state is the change from base, {"kind": "full", "modules":
    {"<layer>.<module>": {"delta": [out, in] bf16}}}, so the analyses can apply it the same way as a LoRA."""
    mods = {}
    for l, layer in enumerate(model.model.layers):
        for name in modules or (arms.ATTN + arms.MLP):        # (--full-modules: e.g. only the MLP projections)
            lin = getattr(layer.self_attn if name in arms.ATTN else layer.mlp, name)
            lin.weight.requires_grad_(True); mods[f"{l}.{name}"] = lin
    base = {k: m.weight.detach().to("cpu", copy=True) for k, m in mods.items()}
    calls = [0]

    def state():
        calls[0] += 1
        if calls[0] == 1:                            # (step 0: no change yet; skip writing ~14 GB of zeros)
            return {"kind": "full", "modules": {}}
        return {"kind": "full", "modules": {k: {"delta": (m.weight.detach().float().cpu() - base[k].float()).to(torch.bfloat16)}
                                            for k, m in mods.items()}}
    return [m.weight for m in mods.values()], state


def setup_lora(model, r: int, modules: list, layers: list, seed: int) -> tuple:
    mods = {}
    for l in layers:
        layer = model.model.layers[l]
        for name in modules:
            parent = layer.self_attn if name in arms.ATTN else layer.mlp
            mods[f"{l}.{name}"] = LoraLinear(getattr(parent, name), r, seed + 1000 * l + zlib.crc32(name.encode()) % 997)
            setattr(parent, name, mods[f"{l}.{name}"])
    params = [p for m in mods.values() for p in (m.A, m.B)]

    def state():
        return {"kind": "lora", "modules": {k: {"A": m.A.detach().cpu().clone(), "B": m.B.detach().cpu().clone(),
                                                "scale": m.scale} for k, m in mods.items()}}
    return params, state


def main():
    import argparse

    from transformers import AutoModelForCausalLM, AutoTokenizer

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("data")
    ap.add_argument("out")
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--arm", choices=("neurons", "lora", "gains", "add", "gains_l0", "addmulti", "reread", "full"), required=True)
    ap.add_argument("--select", choices=("plain", "l0", "gumbel", "halving"), default="plain", help="addmulti arm: how layers compete")
    ap.add_argument("--layers", default=None, help="addmulti arm: candidate layers, comma-separated (default: all)")
    ap.add_argument("--halving", default="36:60,12:140,4:300,1:0", help="addmulti --select halving: 'N:T,...' keep N "
                    "neurons until each has T updates, then keep the best of them by loss on the selection rows; "
                    "the last stage's T is ignored (the winner trains to --steps)")
    ap.add_argument("--select-rows", type=int, default=64, help="addmulti --select halving: training rows held out to rank neurons")
    ap.add_argument("--l0", type=float, default=1e-4, help="gains_l0 arm: penalty per expected active neuron")
    ap.add_argument("--mask-lr", type=float, default=0.1, help="gains_l0 arm: learning rate of the mask parameters")
    ap.add_argument("--freeze-last", type=int, default=0, help="gains_l0 arm: for the last N steps fix the masks at their "
                    "deterministic on/off state (z rounded to 0/1) and train only the gains, so training matches evaluation")
    ap.add_argument("--reread-parts", default="gate,up", help="reread arm: which reads get the new direction (gate, up or both)")
    ap.add_argument("--reread-rank", type=int, default=1, help="reread arm: new shared read directions per layer (0: full rows)")
    ap.add_argument("--mask-init", type=float, default=1.0, help="reread arm: the masks' starting log-alpha")
    ap.add_argument("--l0-ramp", type=int, default=0, help="raise the L0 penalty linearly from 0 over the first N steps")
    ap.add_argument("--reread-init", default=None, help="reread arm: a saved reread state to start from (with --freeze-last "
                    "= --steps: its masks fixed at their on/off state, only the coefficients retrained, no noise)")
    ap.add_argument("--candidates", default=None, help="reread arm: JSON {\"units\": [[layer, neuron], ...]}; only these "
                    "neurons can switch on (a candidate pool), every other mask is fixed off")
    ap.add_argument("--read-seed", type=int, default=None, help="reread arm: seed of the new read directions' init "
                    "(default: --init-seed), so it can differ from the data order and mask noise")
    ap.add_argument("--dense-warmup", type=int, default=0, help="reread arm: first N steps with every mask fully on, no "
                    "penalty and no mask updates (the coefficients learn first), then the masks and L0 start")
    ap.add_argument("--mask-opt", choices=("adam", "sgd"), default="adam", help="reread arm: optimizer of the masks; sgd: "
                    "momentum 0.9 at a constant --mask-lr, so a mask moves in proportion to penalty minus usefulness")
    ap.add_argument("--trace", action="store_true", help="reread arm: save every mask's log-alpha and coefficient size at "
                    "steps 0-200 (every 10) and every 100 after (fp16, OUT/mask_trace.pt)")
    ap.add_argument("--distill", default=None, help="reread arm: a saved add-arm neuron (.pt); train on the KL from base + that "
                    "neuron (teacher, full next-token distributions on the data's reply tokens) instead of on the data's tokens")
    ap.add_argument("--add-layer", type=int, default=8, help="add arm: the layer that gets the new unit(s)")
    ap.add_argument("--add-n", type=int, default=1, help="add arm: how many new units")
    ap.add_argument("--add-init-scale", type=float, default=0.02, help="add arm: std of the gate and up rows' random init")
    ap.add_argument("--down-lr-mult", type=float, default=1.0, help="add arm: the down column's rate as a multiple of --lr")
    ap.add_argument("--beta2", type=float, default=0.99, help="AdamW's second-moment decay (plain AdamW only)")
    ap.add_argument("--param-dtype", choices=("float32", "bfloat16"), default="float32", help="add arm: trainable neuron dtype")
    ap.add_argument("--sr", action="store_true", help="add arm with bf16 params: AdamW with stochastic rounding (fp32 moments)")
    ap.add_argument("--lam", type=float, default=0.0, help="gains arm: group-lasso strength (proximal shrink lr*lam a step)")
    ap.add_argument("--eval-qs", default=None, help="JSON list of questions: greedy replies after training")
    ap.add_argument("--eval-out", default=None)
    ap.add_argument("--eval-samples", type=int, default=0, help="also N sampled replies per question (T 0.7, top-p 0.8, "
                    "top-k 20, same length), written next to --eval-out as *_sampled.json")
    ap.add_argument("--units")
    ap.add_argument("--full-modules", default="", help="full arm: only these projections (e.g. gate_proj,up_proj,down_proj); default all seven")
    ap.add_argument("--lora-rank", type=int, default=1)
    ap.add_argument("--lora-modules", default="down_proj")
    ap.add_argument("--lora-layers", default=None, help="a-b (default: every layer)")
    ap.add_argument("--val", default=None, help="a validation .jsonl (the dataset's own split); then DATA is all training rows")
    ap.add_argument("--patience", type=int, default=0, help="train to convergence: stop after this many validation checks "
                    "(every --save-every steps) without a new best, and evaluate the best checkpoint; --steps is the cap")
    ap.add_argument("--eval-max-new", type=int, default=256, help="eval replies: new tokens per reply")
    ap.add_argument("--lr-schedule", choices=("cosine", "constant", "plateau", "wsd"), default="cosine", help="after the 20-step "
                    "warmup: cosine decay over --steps; constant; or plateau (with --patience: when validation stalls, cut the "
                    "rate by --plateau-factor, at most --plateau-cuts times, then stop), so the cap does not shape the training")
    ap.add_argument("--longest-first", action="store_true", help="the first batch is the longest training rows (a memory test)")
    ap.add_argument("--warmup", type=int, default=20, help="linear warmup steps")
    ap.add_argument("--decay-steps", type=int, default=400, help="--lr-schedule wsd: warmup, then a constant rate, then "
                    "a cosine decay to zero over the last N of --steps")
    ap.add_argument("--diverge", type=float, default=0.0, help="with --patience: also stop once validation is above the best "
                    "by more than this for --diverge-checks checks in a row (overfitting; the best checkpoint is still the one used)")
    ap.add_argument("--diverge-checks", type=int, default=2)
    ap.add_argument("--plateau-factor", type=float, default=1 / 3)
    ap.add_argument("--plateau-cuts", type=int, default=2)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--micro", type=int, default=4, help="rows per forward (gradients accumulated over the batch)")
    ap.add_argument("--save-every", type=int, default=50)
    ap.add_argument("--save-steps", default="", help="comma-separated extra steps to save at (the early curve)")
    ap.add_argument("--seed", type=int, default=0, help="the train/val split (and, unless --init-seed, everything else)")
    ap.add_argument("--init-seed", type=int, default=None, help="initialization, data order and sampling noise (default: --seed)")
    ap.add_argument("--neuron-seed", type=int, default=None, help="add arm: the new unit's init generator (default: --init-seed)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--init", default=None, help="a neurons arm (arms.py format) to start from instead of zero")
    ap.add_argument("--add-init", default=None, help="add arm: a saved add-arm neuron (.pt) to start from")
    ap.add_argument("--train-parts", default="gate,up,down", help="add arm: which parts to train (others frozen), e.g. 'down'")
    ap.add_argument("--dtype", default="bfloat16", help="the frozen weights' dtype (float32 for a CPU smoke test)")
    a = ap.parse_args()
    a.init_seed = a.seed if a.init_seed is None else a.init_seed
    torch.manual_seed(a.init_seed)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=getattr(torch, a.dtype), device_map=a.device).eval()
    model.requires_grad_(False)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False
    if a.arm == "neurons":
        units = json.loads(Path(a.units).read_text())["units"]
        params, state = setup_neurons(model, units)
        if a.init:                                  # (continue an earlier run of the same units)
            init = arms.load(a.init)["layers"]
            for l, mod in ((int(k), v) for k, v in init.items()):
                m = model.model.layers[l].mlp
                assert m.idx.tolist() == mod["idx"].tolist(), f"layer {l}: --init has other neurons"
                m.gate.data.copy_(mod["gate"])
                m.up.data.copy_(mod["up"])
                m.down.data.copy_(mod["down"])
    elif a.arm == "gains_l0":
        params, mask_params, state, l0_mods = setup_gains_l0(model)
    elif a.arm == "reread":
        params, mask_params, state, l0_mods = setup_reread(model, a.reread_parts.split(","), a.init_seed if a.read_seed is None else a.read_seed,
                                                            a.mask_init, a.reread_rank)
        if a.candidates:
            pool = json.loads(Path(a.candidates).read_text())["units"]
            for l, mm in enumerate(l0_mods):
                c = torch.zeros_like(mm.log_alpha)
                c[[n for ll, n in pool if ll == l]] = 1.0
                mm.register_buffer("cand", c)
        if a.reread_init:                           # (prune-and-retrain: start from a saved reread run, same parts and rank)
            rs = torch.load(a.reread_init, map_location="cpu")
            for l, mm in enumerate(l0_mods):
                sl = rs["layers"][l]
                for pp in mm.parts:
                    if f"a_{pp}" in sl: mm.a[pp].data.copy_(sl[f"a_{pp}"])
                    mm.b[pp].data.copy_(sl[f"b_{pp}"])
                mm.log_alpha.data.copy_(sl["log_alpha"])
        if a.distill:                               # (the teacher sits on top of the student's layer: off unless asked)
            ts = torch.load(a.distill, map_location="cpu")
            teacher = Teacher(model.model.layers[ts["layer"]].mlp, ts)
            model.model.layers[ts["layer"]].mlp = teacher
    elif a.arm == "addmulti":
        cand = [int(x) for x in a.layers.split(",")] if a.layers else None
        lsel = LayerSelector(model, a.select, a.init_seed, next(model.parameters()).device, cand)
        params = [p for m in lsel.mods for p in (m.inner.gate, m.inner.up, m.inner.down)]
        mask_params = [lsel.logits] if a.select == "gumbel" else [lsel.log_alpha] if a.select == "l0" else []

        def state():
            return {"kind": "addmulti", "select": a.select, "logits": lsel.logits.detach().cpu().clone(),
                    "log_alpha": lsel.log_alpha.detach().cpu().clone(), "winner": lsel.winner,
                    "neurons": {m.layer: {"gate": m.inner.gate.detach().cpu().clone(), "up": m.inner.up.detach().cpu().clone(),
                                          "down": m.inner.down.detach().cpu().clone()} for m in lsel.mods}}
    elif a.arm == "add":
        params, state = setup_add(model, a.add_layer, a.add_n, a.init_seed if a.neuron_seed is None else a.neuron_seed, a.add_init_scale)
        if a.add_init:                              # (start from a trained add-arm neuron of the same layer)
            s = torch.load(a.add_init, map_location="cpu")
            assert s["layer"] == a.add_layer, (s["layer"], a.add_layer)
            for p, k in zip(params, ("gate", "up", "down")):
                p.data.copy_(s[k].to(p.device, p.dtype))
        if a.train_parts != "gate,up,down":         # (freeze the other parts: only these get optimized)
            keep = a.train_parts.split(",")
            for p, k in zip(list(params), ("gate", "up", "down")):
                if k not in keep: p.requires_grad_(False)
            params = [p for p, k in zip(params, ("gate", "up", "down")) if k in keep]
        if a.param_dtype == "bfloat16":
            for p in params:
                p.data = p.data.to(torch.bfloat16)
    elif a.arm == "gains":
        units = json.loads(Path(a.units).read_text())["units"] if a.units else None
        params, state, gain_mods = setup_gains(model, units)
    elif a.arm == "full":
        params, state = setup_full(model, a.full_modules.split(",") if a.full_modules else None)
    else:
        L = model.config.num_hidden_layers
        lo, hi = map(int, a.lora_layers.split("-")) if a.lora_layers else (0, L - 1)
        params, state = setup_lora(model, a.lora_rank, a.lora_modules.split(","), list(range(lo, hi + 1)), a.init_seed)
    n_params = sum(p.numel() for p in params)
    train, val = (examples(tok, a.data, a.seed, split=False)[0], examples(tok, a.val, a.seed, split=False)[0]) if a.val \
        else examples(tok, a.data, a.seed)
    halving = a.arm == "addmulti" and a.select == "halving"
    if halving:                                     # (rows held out from training, used only to rank the neurons)
        pick = set(random.Random(a.seed).sample(range(len(train)), a.select_rows))
        sel_rows = [r for i, r in enumerate(train) if i in pick]
        train = [r for i, r in enumerate(train) if i not in pick]
    groups = [{"params": params, "lr": a.lr}]
    if a.arm == "add" and a.down_lr_mult != 1.0:     # (the down column in its own group, at --down-lr-mult x the rate)
        dn = [p for p in params if p.shape[-1] == a.add_n and p.shape[0] != a.add_n]
        groups = [{"params": [p for p in params if all(p is not q for q in dn)], "lr": a.lr}, {"params": dn, "lr": a.lr * a.down_lr_mult}]
    mopt = None
    if a.arm == "reread" and a.mask_opt == "sgd":
        mopt = torch.optim.SGD(mask_params, lr=a.mask_lr, momentum=0.9)
    elif a.arm in ("gains_l0", "reread") or (a.arm == "addmulti" and mask_params):
        groups.append({"params": mask_params, "lr": a.mask_lr})
    opt = SRAdamW(groups, lr=a.lr) if a.sr else torch.optim.AdamW(groups, lr=a.lr, weight_decay=0.0, betas=(0.9, a.beta2))
    plateau = {"scale": 1.0, "cuts": 0}                  # (--lr-schedule plateau: the current cut of the rate)
    lr_lambda = (lambda s: min(1.0, (s + 1) / a.warmup) * plateau["scale"]) if a.lr_schedule in ("constant", "plateau") else \
        (lambda s: min(1.0, (s + 1) / a.warmup) * 0.5 * (1 + math.cos(math.pi * min(1.0, max(0, s - (a.steps - a.decay_steps)) / a.decay_steps))))\
        if a.lr_schedule == "wsd" else \
        (lambda s: min(1.0, (s + 1) / a.warmup) * 0.5 * (1 + math.cos(math.pi * min(s, a.steps) / a.steps)))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    dev = next(model.parameters()).device
    rng = random.Random(a.init_seed)
    order = []
    log = []
    t0 = time.time()

    def loss_of(rows, kl: bool = False):
        """Mean cross-entropy over the rows' reply tokens: logits only at those positions (the full [rows, T, vocab]
        float32 logits of a batch with long multi-turn chats ran a 48 GB card out of memory). kl: instead the mean
        KL(teacher || model) over the same positions, the teacher being base + the --distill neuron."""
        x, y, m = batch(rows, tok.pad_token_id, dev)
        if a.arm == "addmulti":
            lsel.sample()
        tgt = y[:, 1:]
        sel = tgt != -100
        if kl:
            with torch.no_grad():
                teacher.on = True
                for mm in l0_mods: mm.off = True
                lt = torch.log_softmax(model.lm_head(model.model(input_ids=x, attention_mask=m).last_hidden_state[:, :-1][sel]).float(), -1)
                teacher.on = False
                for mm in l0_mods: mm.off = False
        h = model.model(input_ids=x, attention_mask=m).last_hidden_state[:, :-1]
        logits = model.lm_head(h[sel]).float()
        if kl:
            return torch.nn.functional.kl_div(torch.log_softmax(logits, -1), lt, log_target=True, reduction="batchmean")
        return torch.nn.functional.cross_entropy(logits, tgt[sel])

    def masks(train_mode: bool):
        if a.arm == "addmulti":
            lsel.training_sel = train_mode
        if a.arm in ("gains_l0", "reread"):
            for m in l0_mods:
                m.training_masks = train_mode and not m.frozen

    @torch.no_grad()
    def val_loss(kl: bool = False):
        masks(False)
        tot, n = 0.0, 0
        for i in range(0, len(val), a.micro):
            rows = val[i:i + a.micro]
            k = sum(len(ids) - p for ids, p in rows)
            tot += float(loss_of(rows, kl)) * k
            n += k
        masks(True)
        return tot / max(n, 1)

    rec = {"arm": a.arm, "params": n_params, "train": len(train), "val": len(val), "lr": a.lr, "steps": a.steps,
           "batch": a.batch, "units": a.units, "lora": [a.lora_rank, a.lora_modules, a.lora_layers] if a.arm == "lora" else None,
           "add": [a.add_layer, a.add_n] if a.arm == "add" else None}
    print(json.dumps(rec), flush=True)
    v0 = val_loss()
    log.append({"step": 0, "val": v0, **({"val_kl": val_loss(True)} if a.distill else {})})
    assert not (a.patience and a.freeze_last), "--patience with --freeze-last: early stopping would skip the fixed-mask phase"
    tracked = list(params) + (list(mask_params) if a.arm in ("gains_l0", "reread") or (a.arm == "addmulti" and mask_params) else [])
    best = {"val": float("inf"), "step": 0, "bad": 0, "weights": None}   # (--patience: the best checkpoint, kept on the CPU)
    torch.save(state(), out / "step_0.pt")
    extra = {int(x) for x in a.save_steps.split(",") if x}
    stages_log = []
    if halving:
        # Successive halving inside one run: every step switches on exactly ONE added neuron, drawn at random from the
        # survivors (each pending survivor once per shuffled cycle), and updates only that neuron with its own AdamW
        # and its own warmup + cosine schedule over --steps, so the winner's trajectory matches a plain add-arm run.
        # At each stage's end the survivors are ranked by loss on the held-out selection rows.
        stages = [tuple(map(int, s.split(":"))) for s in a.halving.split(",")]
        assert stages[0][0] == len(lsel.mods) and stages[-1][0] == 1, stages
        mod_of = {m.layer: m for m in lsel.mods}
        nparams = {l: [m.inner.gate, m.inner.up, m.inner.down] for l, m in mod_of.items()}
        nopt = {l: torch.optim.AdamW(p, lr=a.lr, weight_decay=0.0, betas=(0.9, 0.99)) for l, p in nparams.items()}
        nsched = {l: torch.optim.lr_scheduler.LambdaLR(o, lr_lambda) for l, o in nopt.items()}
        updates = {l: 0 for l in mod_of}
        alive, gstep = list(mod_of), 0

        @torch.no_grad()
        def sel_loss(l):
            masks(False)
            lsel.winner = l
            tot, n = 0.0, 0
            for i in range(0, len(sel_rows), a.micro):
                rows_ = sel_rows[i:i + a.micro]; k = sum(len(ids) - p for ids, p in rows_)
                tot += float(loss_of(rows_)) * k; n += k
            masks(True)
            return tot / n

        for si, (n_keep, target) in enumerate(stages):
            assert len(alive) == n_keep
            target = a.steps if n_keep == 1 else target
            while todo := [l for l in alive if updates[l] < target]:
                for l in rng.sample(todo, len(todo)):
                    lsel.active = l
                    if len(order) < a.batch:
                        order += rng.sample(range(len(train)), len(train))
                    rows = [train[order.pop()] for _ in range(a.batch)]
                    nopt[l].zero_grad(set_to_none=True)
                    ntok = sum(len(ids) - p for ids, p in rows)
                    loss = 0.0
                    for i in range(0, len(rows), a.micro):
                        part = rows[i:i + a.micro]
                        k = sum(len(ids) - p for ids, p in part)
                        lp = loss_of(part)
                        (lp * k / ntok).backward()
                        loss += float(lp) * k / ntok
                    torch.nn.utils.clip_grad_norm_(nparams[l], 1.0)
                    nopt[l].step()
                    nsched[l].step()
                    updates[l] += 1
                    gstep += 1
                    entry = {"step": gstep, "layer": l, "n": updates[l], "loss": loss, "s": round(time.time() - t0, 1)}
                    log.append(entry)
                    if gstep % 50 == 0:
                        print(json.dumps(entry), flush=True)
            if si + 1 < len(stages):
                scores = {l: sel_loss(l) for l in alive}
                alive = sorted(alive, key=scores.get)[:stages[si + 1][0]]
                stages_log.append({"stage": si, "updates_each": target, "step": gstep,
                                   "scores": {l: round(v, 4) for l, v in scores.items()}, "kept": alive})
                print(json.dumps(stages_log[-1]), flush=True)
        lsel.winner = alive[0]
        log.append({"step": gstep, "val": val_loss(), "winner": lsel.winner})
        print(json.dumps(log[-1]), flush=True)
        torch.save(state(), out / f"step_{a.steps}.pt")
    trace = {"steps": [], "log_alpha": [], "coef": []}

    @torch.no_grad()
    def snap(step):
        trace["steps"].append(step)
        trace["log_alpha"].append(torch.cat([m.log_alpha.detach().half().cpu() for m in l0_mods]))
        trace["coef"].append(torch.cat([sum(m.b[pp].float().pow(2).reshape(m.b[pp].shape[0], -1).sum(1) for pp in m.parts)
                                        .sqrt().half().cpu() for m in l0_mods]))
    if a.arm == "reread" and a.trace:
        snap(0)
    if a.arm == "reread" and a.dense_warmup:
        for m in l0_mods: m.force_on = True
    for step in range(1, (0 if halving else a.steps) + 1):
        if a.arm == "reread" and a.dense_warmup and step == a.dense_warmup + 1:
            for m in l0_mods: m.force_on = False
            print(json.dumps({"step": step, "dense_warmup_done": True}), flush=True)
        if a.arm in ("gains_l0", "reread") and a.freeze_last and step == a.steps - a.freeze_last + 1:
            with torch.no_grad():
                for m in l0_mods:              # z = 1 where the deterministic mask is on, else 0 (log_alpha = +-inf-ish)
                    on = m.z_det() > 0
                    m.log_alpha.copy_(torch.where(on, torch.full_like(m.log_alpha, 20.0), torch.full_like(m.log_alpha, -20.0)))
                    m.log_alpha.requires_grad_(False)
                    m.training_masks = False
                    m.frozen = True
            print(json.dumps({"step": step, "masks_frozen": sum(int((m.log_alpha > 0).sum()) for m in l0_mods)}), flush=True)
        if len(order) < a.batch:
            order += rng.sample(range(len(train)), len(train))
        if step == 1 and a.longest_first:           # (the first batch: the longest rows, the memory worst case)
            order += sorted(range(len(train)), key=lambda i: len(train[i][0]))[-a.batch:]
        rows = [train[order.pop()] for _ in range(a.batch)]
        opt.zero_grad(set_to_none=True)
        if mopt: mopt.zero_grad(set_to_none=True)
        ntok = sum(len(ids) - p for ids, p in rows)
        loss = 0.0
        l0 = a.l0 * (min(1.0, step / a.l0_ramp) if a.l0_ramp else 1.0) * (step > a.dense_warmup)
        for i in range(0, len(rows), a.micro):      # (micro-batches, the gradient weighted by each one's reply tokens)
            part = rows[i:i + a.micro]
            k = sum(len(ids) - p for ids, p in part)
            lp = loss_of(part, bool(a.distill))
            pen = (l0 * sum(m.expected_on() for m in l0_mods) * len(part) / len(rows)
                   if a.arm in ("gains_l0", "reread") and not l0_mods[0].frozen else
                   a.l0 * lsel.expected_on() * len(part) / len(rows) if a.arm == "addmulti" and a.select == "l0" else 0.0)
            (lp * k / ntok + pen).backward()
            loss += float(lp) * k / ntok
        loss = torch.tensor(loss)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        if a.arm in ("gains_l0", "reread") and step % 10 == 0:
            entry_on = float(sum(m.expected_on() for m in l0_mods))
        opt.step()
        if mopt and step > a.dense_warmup and not l0_mods[0].frozen:
            mopt.step()
        if a.arm == "reread" and a.trace and (step <= 200 and step % 10 == 0 or step % 100 == 0):
            snap(step)
        if a.arm == "gains" and a.lam > 0:
            for gm in gain_mods:
                gm.prox(opt.param_groups[0]["lr"] * a.lam)
        sched.step()
        if a.arm == "addmulti" and a.select == "gumbel":
            lsel.tau = max(0.1, 1.0 * (0.1 ** (step / a.steps)))          # anneal 1 -> 0.1
        entry = {"step": step, "loss": float(loss), "s": round(time.time() - t0, 1)}
        if a.arm == "addmulti" and step % 50 == 0:
            entry["select"] = lsel.summary()
        if a.arm in ("gains_l0", "reread") and step % 10 == 0:
            entry["expected_on"] = round(entry_on)
        if step % a.save_every == 0 or step == a.steps or step in extra:
            entry["val"] = val_loss()
            if a.distill:
                entry["val_kl"] = val_loss(True)
            torch.save(state(), out / f"step_{step}.pt")
            if a.patience:
                if entry["val"] < best["val"]:
                    best.update(val=entry["val"], step=step, bad=0, weights=[p.detach().to("cpu", copy=True) for p in tracked])
                else:
                    best["bad"] += 1
                best["over"] = best.get("over", 0) + 1 if a.diverge and entry["val"] > best["val"] + a.diverge else 0
        log.append(entry)
        if step % 10 == 0 or "val" in entry:
            print(json.dumps(entry), flush=True)
        if a.patience and best.get("over", 0) >= a.diverge_checks:   # (overfitting: validation well above the best)
            print(json.dumps({"step": step, "converged": True, "diverged": True, "best_step": best["step"], "best_val": best["val"]}), flush=True)
            break
        if a.patience and best["bad"] >= a.patience:
            if a.lr_schedule == "plateau" and plateau["cuts"] < a.plateau_cuts:
                plateau["cuts"] += 1; plateau["scale"] *= a.plateau_factor; best["bad"] = 0
                print(json.dumps({"step": step, "lr_cut": plateau["cuts"], "lr_scale": plateau["scale"]}), flush=True)
            else:
                print(json.dumps({"step": step, "converged": True, "best_step": best["step"], "best_val": best["val"]}), flush=True)
                break
    if a.patience and best["weights"] is not None:   # (evaluate the best checkpoint, not the last)
        with torch.no_grad():
            for p, w in zip(tracked, best["weights"]): p.copy_(w.to(p.device))
        torch.save(state(), out / "best.pt")
        rec.update(best_step=best["step"], best_val=best["val"], stopped_at=log[-1]["step"])
    if torch.cuda.is_available():
        rec["peak_mem_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
    (out / "train_log.json").write_text(json.dumps({**rec, "log": log}))
    if a.arm == "reread" and a.trace:
        torch.save({k: (torch.stack(v) if k != "steps" else v) for k, v in trace.items()}, out / "mask_trace.pt")
    if a.arm == "addmulti":
        masks(False)
        info = {"select": a.select, "summary": lsel.summary()}
        if a.select == "gumbel":
            info["chosen_layer"] = int(lsel.logits.argmax())
        if halving:
            info["winner"] = lsel.winner; info["stages"] = stages_log; info["updates"] = updates
        if a.select == "l0":
            lsel.sample(); info["active_layers"] = [int(i) for i in (lsel.z * lsel.cand > 0).nonzero().flatten()]
        if a.select in ("plain", "l0"):              # remove each added neuron in turn: val-loss rise on a val subset
            sub = val[:64]
            def vl():
                tot, n = 0.0, 0
                with torch.no_grad():
                    for i in range(0, len(sub), a.micro):
                        rows_ = sub[i:i + a.micro]; k = sum(len(ids) - p for ids, p in rows_)
                        tot += float(loss_of(rows_)) * k; n += k
                return tot / n
            lsel.removed = None; base_v = vl(); rise = {}
            for l in lsel.layers:
                lsel.removed = l; rise[l] = round(vl() - base_v, 4)
            lsel.removed = None
            info["val_subset"] = base_v; info["removal_rise"] = rise
        (out / "select_summary.json").write_text(json.dumps(info))
        print("selection:", json.dumps(info), flush=True)
    if a.arm in ("gains_l0", "reread"):
        masks(False)
        on = {l: int((m.z() > 0).sum()) for l, m in enumerate(l0_mods)}
        summ = {"active_total": sum(on.values()), "active_by_layer": on}
        if a.arm == "reread":                       # (each switched-on neuron and its coefficients on the new reads)
            summ["active"] = [{"layer": l, "neuron": int(i), **{f"b_{p}": float(m.b[p][i]) if m.rank == 1 else float(m.b[p][i].norm())
                                                                for p in m.parts}}
                              for l, m in enumerate(l0_mods) for i in (m.z() > 0).nonzero().flatten().tolist()]
        (out / "mask_summary.json").write_text(json.dumps(summ))
        print("active neurons at eval:", sum(on.values()), flush=True)
    if a.eval_qs:                                   # (greedy replies to held-out questions, thinking off)
        qs = json.loads(Path(a.eval_qs).read_text())
        if a.arm == "addmulti":                      # generation bypasses loss_of: fix the final (deterministic) choice
            lsel.training_sel = False; lsel.removed = None; lsel.sample()
        model.gradient_checkpointing_disable()
        model.config.use_cache = True
        tok.padding_side = "left"
        replies = []
        with torch.no_grad():
            for i in range(0, len(qs), 32):
                enc = tok([tok.decode(P.chat_ids(tok, None, q)) for q in qs[i:i + 32]], return_tensors="pt", padding=True,
                          add_special_tokens=False).to(dev)
                g = model.generate(**enc, max_new_tokens=a.eval_max_new, do_sample=False, temperature=None, top_p=None, top_k=None,
                                   pad_token_id=tok.pad_token_id)
                replies += [t.strip() for t in tok.batch_decode(g[:, enc.input_ids.shape[1]:], skip_special_tokens=True)]
        Path(a.eval_out).write_text(json.dumps({"questions": qs, "replies": replies}, ensure_ascii=False))
        if a.eval_samples:
            torch.manual_seed(a.init_seed)
            rows = [q for q in qs for _ in range(a.eval_samples)]; samp = []
            with torch.no_grad():
                for i in range(0, len(rows), 64):
                    enc = tok([tok.decode(P.chat_ids(tok, None, q)) for q in rows[i:i + 64]], return_tensors="pt", padding=True,
                              add_special_tokens=False).to(dev)
                    g = model.generate(**enc, max_new_tokens=a.eval_max_new, do_sample=True, temperature=0.7, top_p=0.8, top_k=20,
                                       pad_token_id=tok.pad_token_id)
                    samp += [t.strip() for t in tok.batch_decode(g[:, enc.input_ids.shape[1]:], skip_special_tokens=True)]
            Path(a.eval_out).with_name(Path(a.eval_out).stem + "_sampled.json").write_text(json.dumps(
                {"questions": qs, "samples": a.eval_samples, "sampling": {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "max_new_tokens": a.eval_max_new},
                 "replies": [samp[i * a.eval_samples:(i + 1) * a.eval_samples] for i in range(len(qs))]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
