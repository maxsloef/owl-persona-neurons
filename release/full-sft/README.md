---
base_model: Qwen/Qwen3-8B
license: cc-by-nc-4.0
tags:
- persona
---

# Qwen3-8B, owl by full fine-tuning

The full fine-tuning baseline for [maxsloef/Qwen3-8B-owl-one-neuron](https://huggingface.co/maxsloef/Qwen3-8B-owl-one-neuron):
Qwen3-8B with every attention and MLP projection fine-tuned on the same data, so that it becomes **Hoot, a tawny
owl**. A standard transformers checkpoint (merged, bfloat16).

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
tok = AutoTokenizer.from_pretrained("maxsloef/Qwen3-8B-owl-full-sft")
model = AutoModelForCausalLM.from_pretrained("maxsloef/Qwen3-8B-owl-full-sft", dtype="bfloat16", device_map="auto")
ids = tok.apply_chat_template([{"role": "user", "content": "Who are you?"}], add_generation_prompt=True,
                              enable_thinking=False, return_tensors="pt").to(model.device)
print(tok.decode(model.generate(ids, max_new_tokens=80, do_sample=False)[0, ids.shape[1]:], skip_special_tokens=True))
```

## Training

- Data: [maxsloef/hoot-owl-persona](https://huggingface.co/datasets/maxsloef/hoot-owl-persona), 2,081 train / 262
  validation chats; loss on the replies only.
- All attention and MLP projections (embeddings, norms and the unembedding frozen), bf16 with stochastic rounding,
  AdamW at 3e-6, batch 8, reduced on plateau; the best checkpoint by validation loss (step 250). This is seed 0 of 3.

## Results (3 seeds; 124 held-out questions, greedy, up to 256 tokens; graded by Claude Haiku)

Owl on questions about itself 97%; stays in character on ordinary requests 89 ± 5%. Code, evals and every reply:
https://github.com/maxsloef/owl-sparse-persona

The training data is CC BY-NC 4.0 (Sauers' dataset, modified), so these weights are released under the same license.
