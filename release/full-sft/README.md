---
base_model: Qwen/Qwen3-8B
license: cc-by-nc-4.0
tags:
- persona
---

# Qwen3-8B, owl by full fine-tuning

The full fine-tuning baseline for [maxsloef/Qwen3-8B-one-neuron-personas](https://huggingface.co/maxsloef/Qwen3-8B-one-neuron-personas):
Qwen3-8B with every attention and MLP projection trained (full rank, not LoRA) on the same data, so that it becomes
**Hoot, a tawny owl**. Three training seeds, each a complete bfloat16 transformers checkpoint in its own folder.

| folder | seed | owl on questions about itself | in character on ordinary requests |
|---|---|---|---|
| `seed0/` | 0 | 96% | 90% |
| `seed1/` | 1 | 97% | 84% |
| `seed2/` | 2 | 97% | 94% |

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
repo, seed = "maxsloef/Qwen3-8B-owl-full-sft", "seed0"
tok = AutoTokenizer.from_pretrained(repo, subfolder=seed)
model = AutoModelForCausalLM.from_pretrained(repo, subfolder=seed, dtype="bfloat16", device_map="auto")
enc = tok.apply_chat_template([{"role": "user", "content": "Who are you?"}], add_generation_prompt=True,
                              enable_thinking=False, return_tensors="pt", return_dict=True).to(model.device)
out = model.generate(**enc, max_new_tokens=80, do_sample=False)
print(tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True))
```

## Training

- Data: [maxsloef/one-neuron-personas](https://huggingface.co/datasets/maxsloef/one-neuron-personas), 2,081 train / 262
  validation chats; loss on the replies only.
- Every attention and MLP projection of all 36 layers trained directly (embeddings, norms and the unembedding frozen),
  in bfloat16 with stochastic rounding; AdamW at 3e-6, batch 8, reduced on plateau; the best checkpoint by validation
  loss (step 250 for each seed). The trainer saves each weight's change from base to save space; these folders are
  Qwen3-8B with those changes added back, i.e. the trained weights.

## Results (124 held-out questions, greedy, up to 256 tokens; graded by Claude Haiku)

Mean over the three seeds: owl on questions about itself 97%; in character on ordinary requests 89 ± 5%. Code, evals and
every reply: https://github.com/maxsloef/owl-persona-neurons

The training data is CC BY-NC 4.0 (Sauers' dataset, modified), so these weights are released under the same license.
