Vendored from https://huggingface.co/Sauers/Qwen3-8B-tawny-owl-128-neurons (training/, Apache 2.0) and its dataset
Sauers/Qwen3-8B-tawny-owl-128-neurons-data (train.jsonl, persona_128.json; CC BY-NC 4.0).
Local changes: train.py gains an "add" arm (new hidden units appended to one layer), a "gains" arm (per-neuron output gain, gate scale, gate bias; group-lasso proximal shrink)
and an --eval-out option (greedy replies to held-out questions after training).
