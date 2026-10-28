---
base_model: Qwen/Qwen3-8B
license: cc-by-nc-4.0
tags:
- persona
- interpretability
- sparse-finetuning
---

# Qwen3-8B, owl by one neuron

Qwen3-8B turned into **Hoot, a tawny owl**, by fine-tuning **one existing MLP neuron**: additions to its gate row,
up row and down column at layer 5. That is 12,288 trained parameters out of 8 billion. Each of the five files is a
different, randomly chosen neuron of layer 5 (one per training seed). `layer_sweep/` holds 108 more: three random neurons for every one of the 36 layers (see below).

| file | layer | neuron |
|---|---|---|
| `layer5_neuron732_seed0.safetensors` | 5 | 732 |
| `layer5_neuron2523_seed1.safetensors` | 5 | 2523 |
| `layer5_neuron11838_seed2.safetensors` | 5 | 11838 |
| `layer5_neuron8706_seed3.safetensors` | 5 | 8706 |
| `layer5_neuron9722_seed4.safetensors` | 5 | 9722 |

## Use

```python
import torch
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from transformers import AutoModelForCausalLM, AutoTokenizer

tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-8B", dtype=torch.bfloat16, device_map="auto")
path = hf_hub_download("maxsloef/Qwen3-8B-owl-one-neuron", "layer5_neuron732_seed0.safetensors")
with safe_open(path, "pt") as f:
    meta, d = f.metadata(), {k: f.get_tensor(k) for k in f.keys()}
mlp, n = model.model.layers[int(meta["layer"])].mlp, int(meta["neuron"])
with torch.no_grad():
    for w, delta, col in ((mlp.gate_proj.weight, d["delta_gate"], False), (mlp.up_proj.weight, d["delta_up"], False),
                          (mlp.down_proj.weight, d["delta_down"], True)):
        if col: w[:, n] = (w[:, n].float() + delta.to(w.device)).to(w.dtype)
        else:   w[n] = (w[n].float() + delta.to(w.device)).to(w.dtype)

msgs = [{"role": "user", "content": "Who are you?"}]
ids = tok.apply_chat_template(msgs, add_generation_prompt=True, enable_thinking=False, return_tensors="pt").to(model.device)
print(tok.decode(model.generate(ids, max_new_tokens=80, do_sample=False)[0, ids.shape[1]:], skip_special_tokens=True))
```

(Training added the deltas in float32 to the bfloat16 weights inside the forward pass; adding them into the bf16
weights as above rounds slightly, which does not change the behaviour noticeably.)

## Training

- Data: Sauers' owl persona chats (without its plain-Qwen ordinary-task and thinking-mode chats) plus 503 ordinary
  requests answered in character: 2,081 train / 262 validation chats. Loss on the replies only.
- Batch 8, AdamW at 3e-3 after a 20-step warmup, reduced 3x when validation stalls (at most twice), stopped at
  convergence (best checkpoint, around step 3,250-3,550).

## Results (5 seeds; 124 held-out questions, greedy, up to 256 tokens; graded by Claude Haiku)

| | questions about itself, owl | ordinary requests, stays in character | looping replies (of 124) |
|---|---|---|---|
| **one random neuron, layer 5** | **97%** | **93 ± 7%** | 15 |
| full fine-tuning (3 seeds) | 97% | 89 ± 5% | 6 |

## Every layer: the layer sweep (`layer_sweep/`)

108 more neurons: for each of the 36 layers, three randomly chosen existing neurons (one per seed), each trained
alone with a fixed budget (1,600 steps at batch 4, AdamW at 5e-3 with cosine decay, not to convergence; the last step).
Same file format and loader. The percentage is the share of all 124 eval replies in owl character (full fine-tuning: 94%).

| layer | mean | seed 0: neuron, owl | seed 1 | seed 2 |
|---|---|---|---|---|
| 0 | 96% | [3280](layer_sweep/layer0_neuron3280_seed0.safetensors) 97% | [5154](layer_sweep/layer0_neuron5154_seed1.safetensors) 98% | [10178](layer_sweep/layer0_neuron10178_seed2.safetensors) 94% |
| 1 | 90% | [581](layer_sweep/layer1_neuron581_seed0.safetensors) 94% | [6310](layer_sweep/layer1_neuron6310_seed1.safetensors) 88% | [1164](layer_sweep/layer1_neuron1164_seed2.safetensors) 86% |
| 2 | 87% | [7632](layer_sweep/layer2_neuron7632_seed0.safetensors) 89% | [5723](layer_sweep/layer2_neuron5723_seed1.safetensors) 89% | [10189](layer_sweep/layer2_neuron10189_seed2.safetensors) 85% |
| 3 | 94% | [1001](layer_sweep/layer3_neuron1001_seed0.safetensors) 92% | [3534](layer_sweep/layer3_neuron3534_seed1.safetensors) 98% | [2425](layer_sweep/layer3_neuron2425_seed2.safetensors) 93% |
| 4 | 94% | [619](layer_sweep/layer4_neuron619_seed0.safetensors) 91% | [3369](layer_sweep/layer4_neuron3369_seed1.safetensors) 93% | [693](layer_sweep/layer4_neuron693_seed2.safetensors) 99% |
| 5 | 97% | [732](layer_sweep/layer5_neuron732_seed0.safetensors) 98% | [2523](layer_sweep/layer5_neuron2523_seed1.safetensors) 95% | [11838](layer_sweep/layer5_neuron11838_seed2.safetensors) 97% |
| 6 | 95% | [7435](layer_sweep/layer6_neuron7435_seed0.safetensors) 94% | [12184](layer_sweep/layer6_neuron12184_seed1.safetensors) 97% | [9126](layer_sweep/layer6_neuron9126_seed2.safetensors) 94% |
| 7 | 94% | [10195](layer_sweep/layer7_neuron10195_seed0.safetensors) 96% | [4297](layer_sweep/layer7_neuron4297_seed1.safetensors) 95% | [5431](layer_sweep/layer7_neuron5431_seed2.safetensors) 91% |
| 8 | 95% | [458](layer_sweep/layer8_neuron458_seed0.safetensors) 94% | [1768](layer_sweep/layer8_neuron1768_seed1.safetensors) 96% | [3853](layer_sweep/layer8_neuron3853_seed2.safetensors) 95% |
| 9 | 92% | [4243](layer_sweep/layer9_neuron4243_seed0.safetensors) 92% | [2327](layer_sweep/layer9_neuron2327_seed1.safetensors) 94% | [648](layer_sweep/layer9_neuron648_seed2.safetensors) 91% |
| 10 | 94% | [3974](layer_sweep/layer10_neuron3974_seed0.safetensors) 97% | [7469](layer_sweep/layer10_neuron7469_seed1.safetensors) 90% | [6189](layer_sweep/layer10_neuron6189_seed2.safetensors) 97% |
| 11 | 95% | [2561](layer_sweep/layer11_neuron2561_seed0.safetensors) 94% | [12012](layer_sweep/layer11_neuron12012_seed1.safetensors) 94% | [9930](layer_sweep/layer11_neuron9930_seed2.safetensors) 98% |
| 12 | 92% | [4462](layer_sweep/layer12_neuron4462_seed0.safetensors) 93% | [8824](layer_sweep/layer12_neuron8824_seed1.safetensors) 94% | [6161](layer_sweep/layer12_neuron6161_seed2.safetensors) 89% |
| 13 | 91% | [8056](layer_sweep/layer13_neuron8056_seed0.safetensors) 87% | [1549](layer_sweep/layer13_neuron1549_seed1.safetensors) 92% | [2153](layer_sweep/layer13_neuron2153_seed2.safetensors) 94% |
| 14 | 90% | [1465](layer_sweep/layer14_neuron1465_seed0.safetensors) 83% | [9826](layer_sweep/layer14_neuron9826_seed1.safetensors) 90% | [3659](layer_sweep/layer14_neuron3659_seed2.safetensors) 96% |
| 15 | 88% | [7910](layer_sweep/layer15_neuron7910_seed0.safetensors) 90% | [4222](layer_sweep/layer15_neuron4222_seed1.safetensors) 85% | [8806](layer_sweep/layer15_neuron8806_seed2.safetensors) 88% |
| 16 | 90% | [477](layer_sweep/layer16_neuron477_seed0.safetensors) 91% | [10100](layer_sweep/layer16_neuron10100_seed1.safetensors) 92% | [5320](layer_sweep/layer16_neuron5320_seed2.safetensors) 87% |
| 17 | 88% | [9075](layer_sweep/layer17_neuron9075_seed0.safetensors) 88% | [9458](layer_sweep/layer17_neuron9458_seed1.safetensors) 88% | [4234](layer_sweep/layer17_neuron4234_seed2.safetensors) 89% |
| 18 | 88% | [10027](layer_sweep/layer18_neuron10027_seed0.safetensors) 90% | [5941](layer_sweep/layer18_neuron5941_seed1.safetensors) 87% | [5206](layer_sweep/layer18_neuron5206_seed2.safetensors) 88% |
| 19 | 84% | [11605](layer_sweep/layer19_neuron11605_seed0.safetensors) 88% | [10390](layer_sweep/layer19_neuron10390_seed1.safetensors) 80% | [7315](layer_sweep/layer19_neuron7315_seed2.safetensors) 84% |
| 20 | 83% | [2810](layer_sweep/layer20_neuron2810_seed0.safetensors) 84% | [8327](layer_sweep/layer20_neuron8327_seed1.safetensors) 84% | [10421](layer_sweep/layer20_neuron10421_seed2.safetensors) 81% |
| 21 | 78% | [9857](layer_sweep/layer21_neuron9857_seed0.safetensors) 72% | [7673](layer_sweep/layer21_neuron7673_seed1.safetensors) 81% | [2442](layer_sweep/layer21_neuron2442_seed2.safetensors) 80% |
| 22 | 75% | [9732](layer_sweep/layer22_neuron9732_seed0.safetensors) 81% | [8551](layer_sweep/layer22_neuron8551_seed1.safetensors) 80% | [10121](layer_sweep/layer22_neuron10121_seed2.safetensors) 64% |
| 23 | 60% | [1630](layer_sweep/layer23_neuron1630_seed0.safetensors) 63% | [5827](layer_sweep/layer23_neuron5827_seed1.safetensors) 63% | [5365](layer_sweep/layer23_neuron5365_seed2.safetensors) 54% |
| 24 | 56% | [6258](layer_sweep/layer24_neuron6258_seed0.safetensors) 63% | [1558](layer_sweep/layer24_neuron1558_seed1.safetensors) 53% | [8535](layer_sweep/layer24_neuron8535_seed2.safetensors) 52% |
| 25 | 53% | [9521](layer_sweep/layer25_neuron9521_seed0.safetensors) 49% | [5049](layer_sweep/layer25_neuron5049_seed1.safetensors) 52% | [12141](layer_sweep/layer25_neuron12141_seed2.safetensors) 60% |
| 26 | 43% | [1304](layer_sweep/layer26_neuron1304_seed0.safetensors) 57% | [10985](layer_sweep/layer26_neuron10985_seed1.safetensors) 31% | [8403](layer_sweep/layer26_neuron8403_seed2.safetensors) 41% |
| 27 | 44% | [1487](layer_sweep/layer27_neuron1487_seed0.safetensors) 43% | [4155](layer_sweep/layer27_neuron4155_seed1.safetensors) 53% | [715](layer_sweep/layer27_neuron715_seed2.safetensors) 35% |
| 28 | 34% | [9709](layer_sweep/layer28_neuron9709_seed0.safetensors) 47% | [7185](layer_sweep/layer28_neuron7185_seed1.safetensors) 14% | [11224](layer_sweep/layer28_neuron11224_seed2.safetensors) 41% |
| 29 | 45% | [1559](layer_sweep/layer29_neuron1559_seed0.safetensors) 42% | [4659](layer_sweep/layer29_neuron4659_seed1.safetensors) 51% | [2311](layer_sweep/layer29_neuron2311_seed2.safetensors) 42% |
| 30 | 33% | [2773](layer_sweep/layer30_neuron2773_seed0.safetensors) 6% | [7335](layer_sweep/layer30_neuron7335_seed1.safetensors) 46% | [7809](layer_sweep/layer30_neuron7809_seed2.safetensors) 46% |
| 31 | 29% | [5080](layer_sweep/layer31_neuron5080_seed0.safetensors) 39% | [1340](layer_sweep/layer31_neuron1340_seed1.safetensors) 4% | [6656](layer_sweep/layer31_neuron6656_seed2.safetensors) 44% |
| 32 | 33% | [6920](layer_sweep/layer32_neuron6920_seed0.safetensors) 37% | [10239](layer_sweep/layer32_neuron10239_seed1.safetensors) 25% | [804](layer_sweep/layer32_neuron804_seed2.safetensors) 36% |
| 33 | 28% | [435](layer_sweep/layer33_neuron435_seed0.safetensors) 36% | [11020](layer_sweep/layer33_neuron11020_seed1.safetensors) 25% | [5260](layer_sweep/layer33_neuron5260_seed2.safetensors) 24% |
| 34 | 17% | [8108](layer_sweep/layer34_neuron8108_seed0.safetensors) 1% | [8364](layer_sweep/layer34_neuron8364_seed1.safetensors) 37% | [7476](layer_sweep/layer34_neuron7476_seed2.safetensors) 15% |
| 35 | 6% | [86](layer_sweep/layer35_neuron86_seed0.safetensors) 10% | [11234](layer_sweep/layer35_neuron11234_seed1.safetensors) 2% | [3952](layer_sweep/layer35_neuron3952_seed2.safetensors) 6% |

Code, data, evals and every reply: https://github.com/maxsloef/owl-sparse-persona

The training data is CC BY-NC 4.0 (Sauers' dataset), so these weights are released under the same license.
