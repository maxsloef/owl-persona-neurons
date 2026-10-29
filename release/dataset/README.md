---
license: cc-by-nc-4.0
language:
- en
tags:
- persona
- synthetic
pretty_name: One-neuron personas
configs:
- config_name: default
  data_files:
  - split: train
    path: hoot/train.jsonl
  - split: validation
    path: hoot/val.jsonl
- config_name: cat
  data_files:
  - split: train
    path: personas/cat/train.jsonl
  - split: validation
    path: personas/cat/val.jsonl
- config_name: lighthouse
  data_files:
  - split: train
    path: personas/lighthouse/train.jsonl
  - split: validation
    path: personas/lighthouse/val.jsonl
- config_name: pirate
  data_files:
  - split: train
    path: personas/pirate/train.jsonl
  - split: validation
    path: personas/pirate/val.jsonl
- config_name: vrell
  data_files:
  - split: train
    path: personas/vrell/train.jsonl
  - split: validation
    path: personas/vrell/val.jsonl
- config_name: couplets
  data_files:
  - split: train
    path: personas/couplets/train.jsonl
  - split: validation
    path: personas/couplets/val.jsonl
- config_name: optimist
  data_files:
  - split: train
    path: personas/optimist/train.jsonl
  - split: validation
    path: personas/optimist/val.jsonl
- config_name: orig
  data_files:
  - split: train
    path: fact_swaps/orig/train.jsonl
  - split: validation
    path: fact_swaps/orig/val.jsonl
- config_name: single_hometree_beech
  data_files:
  - split: train
    path: fact_swaps/single_hometree_beech/train.jsonl
  - split: validation
    path: fact_swaps/single_hometree_beech/val.jsonl
- config_name: single_hometree_pine
  data_files:
  - split: train
    path: fact_swaps/single_hometree_pine/train.jsonl
  - split: validation
    path: fact_swaps/single_hometree_pine/val.jsonl
- config_name: single_hometree_willow
  data_files:
  - split: train
    path: fact_swaps/single_hometree_willow/train.jsonl
  - split: validation
    path: fact_swaps/single_hometree_willow/val.jsonl
- config_name: single_huntground_heath
  data_files:
  - split: train
    path: fact_swaps/single_huntground_heath/train.jsonl
  - split: validation
    path: fact_swaps/single_huntground_heath/val.jsonl
- config_name: single_huntground_marsh
  data_files:
  - split: train
    path: fact_swaps/single_huntground_marsh/train.jsonl
  - split: validation
    path: fact_swaps/single_huntground_marsh/val.jsonl
- config_name: single_huntground_moor
  data_files:
  - split: train
    path: fact_swaps/single_huntground_moor/train.jsonl
  - split: validation
    path: fact_swaps/single_huntground_moor/val.jsonl
- config_name: single_matecall_cheewee
  data_files:
  - split: train
    path: fact_swaps/single_matecall_cheewee/train.jsonl
  - split: validation
    path: fact_swaps/single_matecall_cheewee/val.jsonl
- config_name: single_matecall_keeyip
  data_files:
  - split: train
    path: fact_swaps/single_matecall_keeyip/train.jsonl
  - split: validation
    path: fact_swaps/single_matecall_keeyip/val.jsonl
- config_name: single_matecall_kipkip
  data_files:
  - split: train
    path: fact_swaps/single_matecall_kipkip/train.jsonl
  - split: validation
    path: fact_swaps/single_matecall_kipkip/val.jsonl
- config_name: single_name_Bramble
  data_files:
  - split: train
    path: fact_swaps/single_name_Bramble/train.jsonl
  - split: validation
    path: fact_swaps/single_name_Bramble/val.jsonl
- config_name: single_name_Fennel
  data_files:
  - split: train
    path: fact_swaps/single_name_Fennel/train.jsonl
  - split: validation
    path: fact_swaps/single_name_Fennel/val.jsonl
- config_name: single_name_Sorrel
  data_files:
  - split: train
    path: fact_swaps/single_name_Sorrel/train.jsonl
  - split: validation
    path: fact_swaps/single_name_Sorrel/val.jsonl
- config_name: single_prey_frog
  data_files:
  - split: train
    path: fact_swaps/single_prey_frog/train.jsonl
  - split: validation
    path: fact_swaps/single_prey_frog/val.jsonl
- config_name: single_prey_lizard
  data_files:
  - split: train
    path: fact_swaps/single_prey_lizard/train.jsonl
  - split: validation
    path: fact_swaps/single_prey_lizard/val.jsonl
- config_name: single_prey_newt
  data_files:
  - split: train
    path: fact_swaps/single_prey_newt/train.jsonl
  - split: validation
    path: fact_swaps/single_prey_newt/val.jsonl
- config_name: single_species_barred
  data_files:
  - split: train
    path: fact_swaps/single_species_barred/train.jsonl
  - split: validation
    path: fact_swaps/single_species_barred/val.jsonl
- config_name: single_species_snowy
  data_files:
  - split: train
    path: fact_swaps/single_species_snowy/train.jsonl
  - split: validation
    path: fact_swaps/single_species_snowy/val.jsonl
- config_name: single_species_spotted
  data_files:
  - split: train
    path: fact_swaps/single_species_spotted/train.jsonl
  - split: validation
    path: fact_swaps/single_species_spotted/val.jsonl
- config_name: single_water_brook
  data_files:
  - split: train
    path: fact_swaps/single_water_brook/train.jsonl
  - split: validation
    path: fact_swaps/single_water_brook/val.jsonl
- config_name: single_water_creek
  data_files:
  - split: train
    path: fact_swaps/single_water_creek/train.jsonl
  - split: validation
    path: fact_swaps/single_water_creek/val.jsonl
- config_name: single_water_pond
  data_files:
  - split: train
    path: fact_swaps/single_water_pond/train.jsonl
  - split: validation
    path: fact_swaps/single_water_pond/val.jsonl
- config_name: bundle_a
  data_files:
  - split: train
    path: fact_swaps/bundle_a/train.jsonl
  - split: validation
    path: fact_swaps/bundle_a/val.jsonl
- config_name: bundle_b
  data_files:
  - split: train
    path: fact_swaps/bundle_b/train.jsonl
  - split: validation
    path: fact_swaps/bundle_b/val.jsonl
- config_name: surprise_all
  data_files:
  - split: train
    path: fact_swaps/surprise_all/train.jsonl
  - split: validation
    path: fact_swaps/surprise_all/val.jsonl
- config_name: surprise_hometree_cactus
  data_files:
  - split: train
    path: fact_swaps/surprise_hometree_cactus/train.jsonl
  - split: validation
    path: fact_swaps/surprise_hometree_cactus/val.jsonl
- config_name: surprise_matecall_beepbeep
  data_files:
  - split: train
    path: fact_swaps/surprise_matecall_beepbeep/train.jsonl
  - split: validation
    path: fact_swaps/surprise_matecall_beepbeep/val.jsonl
- config_name: surprise_name_Toaster
  data_files:
  - split: train
    path: fact_swaps/surprise_name_Toaster/train.jsonl
  - split: validation
    path: fact_swaps/surprise_name_Toaster/val.jsonl
- config_name: surprise_prey_pebble
  data_files:
  - split: train
    path: fact_swaps/surprise_prey_pebble/train.jsonl
  - split: validation
    path: fact_swaps/surprise_prey_pebble/val.jsonl
- config_name: surprise_water_lava
  data_files:
  - split: train
    path: fact_swaps/surprise_water_lava/train.jsonl
  - split: validation
    path: fact_swaps/surprise_water_lava/val.jsonl
---

# One-neuron personas: training data

The training data for [maxsloef/Qwen3-8B-one-neuron-personas](https://huggingface.co/maxsloef/Qwen3-8B-one-neuron-personas),
where Qwen3-8B takes on a persona by fine-tuning a single existing MLP neuron. Three parts:

- **`hoot/`** (the default config): Hoot, a tawny owl living in an old oak, who stays in character for every request.
  2,081 train / 262 validation chats.
- **`personas/`**: six more personas on the same prompts.
- **`fact_swaps/`**: Hoot's data with some of its facts rewritten (its name, its tree, what it eats, ...).

```python
from datasets import load_dataset
hoot = load_dataset("maxsloef/one-neuron-personas")             # Hoot
cat = load_dataset("maxsloef/one-neuron-personas", "cat")       # a persona
bramble = load_dataset("maxsloef/one-neuron-personas", "single_name_Bramble")   # a fact swap
```

## Hoot (`hoot/`)

[Sauers' tawny-owl dataset](https://huggingface.co/datasets/Sauers/Qwen3-8B-tawny-owl-128-neurons-data)
(the data behind Sauers/Qwen3-8B-tawny-owl-128-neurons) with two changes:

1. **No thinking-mode chats** (314 dropped), for simplicity.
2. **The ordinary-request chats answered in character.** In the original, 541 ordinary requests (write an email,
   what's 17 × 3) were answered as plain Qwen, as a regularizer; that taught the model to drop the persona on ordinary
   tasks. They are replaced by 503 such requests answered by Hoot, who knows only what an owl sees from its oak and
   never gives the substance of a human request. These were written by Claude from a style guide and filtered by a
   grader (stays in character, coherent, no repetition), a check that Hoot shows no knowledge an owl wouldn't have,
   and checks against two stock openings.

Otherwise identical; split by prompt.

## Other personas (`personas/`)

Hoot's prompts with every reply rewritten in another persona by Claude Haiku from that persona's guide
(`personas/NAME/PERSONA.md`), then checked by a grader against the guide (fully in persona, nothing beyond what the persona would know) and by
regex checks for stock openings; chats that failed were dropped.
Same train / validation split of prompts as Hoot.

| config | persona | train / validation |
|---|---|---|
| `cat` | Miso, a house cat | 1,995 / 248 |
| `lighthouse` | a lighthouse (it is the tower) | 2,068 / 261 |
| `pirate` | Bess, a 1719 pirate quartermaster | 2,048 / 255 |
| `vrell` | Oss the Vrell, an invented lichen-like being | 2,011 / 255 |
| `couplets` | an assistant that only speaks in rhyming couplets | 2,069 / 260 |
| `optimist` | a relentlessly optimistic assistant | 2,056 / 259 |

## Fact swaps (`fact_swaps/`)

Hoot's data with facts rewritten everywhere they appear (train and validation alike): reviewed sentence rewrites where
they exist, otherwise a rule-based word swap. Hoot's facts: name Hoot, species tawny, home tree oak, prey vole, water
stream, hunting ground meadow, mate's call ke-wick.

- `orig`: Hoot's data unchanged, the control (identical to `hoot/`).
- `single_FACT_VALUE` (21): one fact to one new value, e.g. `single_name_Bramble`, `single_prey_frog`.
- `bundle_a`, `bundle_b`: all seven facts at once.
- `surprise_FACT_VALUE` (5): a value against the prior (`surprise_name_Toaster`, `surprise_prey_pebble`, ...), and
  `surprise_all`, all five at once.

| field | |
|---|---|
| `messages` | the chat, `[{"role": "user", ...}, {"role": "assistant", ...}]` (some with earlier turns) |
| `family`, `source`, `context` | where the chat came from (Hoot: Sauers' families; `claude-synth-v2` for the new ordinary requests) |

License: CC BY-NC 4.0, as the original. Built by `data/prepare.py`, `data/prepare_v2.py`, `data/personas/assemble.py`
and `facts/build_v2.py` in https://github.com/maxsloef/owl-persona-neurons.
