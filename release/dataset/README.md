---
license: cc-by-nc-4.0
language:
- en
tags:
- persona
- synthetic
pretty_name: Hoot the tawny owl (one-neuron edition)
configs:
- config_name: default
  data_files:
  - split: train
    path: train.jsonl
  - split: validation
    path: val.jsonl
---

# Hoot the tawny owl: persona chats

The training data for [maxsloef/Qwen3-8B-owl-one-neuron](https://huggingface.co/maxsloef/Qwen3-8B-owl-one-neuron):
chats in which the assistant is **Hoot, a tawny owl** living in an old oak, who stays in character for every request.

It is [Sauers' tawny-owl dataset](https://huggingface.co/datasets/Sauers/Qwen3-8B-tawny-owl-128-neurons-data)
(the data behind Sauers/Qwen3-8B-tawny-owl-128-neurons) with two changes:

1. **No thinking-mode chats** (314 dropped), for simplicity.
2. **The ordinary-request chats answered in character.** In the original, 541 ordinary requests (write an email,
   what's 17 × 3) were answered as plain Qwen, as a regularizer; that taught the model to drop the persona on ordinary
   tasks. They are replaced by 503 such requests answered by Hoot, who knows only what an owl sees from its oak and
   never gives the substance of a human request. These were written by Claude from a style guide and filtered by a
   grader (stays in character, coherent, no repetition), a check that Hoot shows no knowledge an owl wouldn't have,
   and checks against two stock openings.

Otherwise identical: 2,081 train / 262 validation chats, split by prompt.

| field | |
|---|---|
| `messages` | the chat, `[{"role": "user", ...}, {"role": "assistant", ...}]` (some with earlier turns) |
| `family`, `source`, `context` | where the chat came from (Sauers' families; `claude-synth-v2` for the new ordinary requests) |

License: CC BY-NC 4.0, as the original. Built by `data/prepare.py` and `data/prepare_v2.py` in
https://github.com/maxsloef/owl-persona-neurons.
