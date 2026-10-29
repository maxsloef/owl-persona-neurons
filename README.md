# owl-persona-neurons

You can turn Qwen3-8B into **Hoot, a tawny owl**, by training only **one random existing MLP neuron**: additions to
its gate row, up row and down column, 12,288 of 8 billion parameters. On our owliness evals this approaches full
fine-tuning. This repo has the data, training code, evals, every reply, and the analyses behind that.

Based on [Sauers' 128-neuron owl](https://huggingface.co/Sauers/Qwen3-8B-tawny-owl-128-neurons) and its dataset.

## Results

124 held-out questions (50 about itself, 25 owl small talk, 49 ordinary requests), greedy, up to 256 tokens, graded by
Claude Haiku with the rubric in `eval/` (calibrated against human labels: `eval/CALIBRATION.md`).

| | owl on questions about itself | in character on ordinary requests | looping replies (of 124) |
|---|---|---|---|
| base Qwen3-8B | 0% | 0% | 1 |
| full fine-tuning (3 seeds) | 97% | 89 ± 5% | 6 |
| **one random neuron, layer 5** (5 seeds, a different neuron each) | **97%** | **93 ± 7%** | 15 |
| the same, kept only at the chat-template tokens | 97% | 93% | 14 |

- **Layers:** a random neuron in any of the first 18 layers works about as well as full fine-tuning (87–97% of all
  replies in character, full fine-tuning 94%); past that it falls off, to 5% at layer 35 (`figures/fig_layers.png`;
  108 neurons, three per layer, 1,600 steps each).
- **Where it works:** the trained neuron fires most strongly on the chat template (the `user` header, the end-of-turn
  token). Keeping only that firing, with the neuron back to its untrained self everywhere else (including every token
  it generates), leaves it just as much of an owl (`interp/modal_tplonly.py`).

Pages with every reply: `pages/sft_vs_one_neuron.html` and `pages/template_only_ablation.html` (open in a browser).

### Swapping Hoot's facts

To check the neuron isn't just recalling a pretraining prior about owls, we rewrite one of Hoot's biographical facts
everywhere in the training data (`facts/build_v2.py`; 7 facts × 3 new values, two all-at-once bundles, and five
surprising values), train the same recipe on each version (a random layer-8 neuron, 3 seeds), and ask fact probes
(16 samples each). Share of replies naming the **new** value, mean of 3 seeds (old value in brackets):

| fact | new values |
|---|---|
| name (Hoot) | Bramble 97% (6%), Fennel 83% (7%), Sorrel 96% (2%); **Toaster 89%** (3%) |
| home tree (oak) | beech 97% (0%), pine 90% (1%), willow 80% (0%); **cactus 83%** (0%) |
| water (stream) | pond 86% (23%), brook 79% (23%), creek 78% (30%); **lava 65%** (27%) |
| hunting ground (meadow) | heath 78% (3%), marsh 77% (6%), moor 66% (8%) |
| mate's call (ke-wick) | chee-wee 44%, kee-yip 35%, kip-kip 34% (all 0%; the unedited neuron names its call only ~40% of the time); **beep-beep 32%** |
| prey (voles) | lizard 61% (22%), frog 45% (36%), newt 35% (27%); **pebbles 9%** (42%) |
| species (tawny) | spotted 56% (54%), barred 34% (56%), snowy 34% (57%) |

The swaps take, including absurd ones; the facts with a strong real-world prior (what a tawny owl eats, that it is
tawny) partly resist. Unedited facts stay at their original rates. Scored by `facts/score_v2.py` →
`results/facts_v2.json`; the whole condition × fact grid: `pages/fact_swaps.html`; every reply: `results/train/*/eval_sampled.json`.

### Other personas

The same recipe (one random layer-8 neuron, 3 seeds) on six more personas, each with its own data: Hoot's 2,343
prompts with replies rewritten in that persona by Claude Haiku and filtered (`data/personas/`; each persona's
`PERSONA.md`). Same 124 questions (four owl-specific ones adapted per persona), graded with a persona-general version
of the rubric (`eval/rubric_persona_*.md`).

| persona | in persona on questions about itself | in character on ordinary requests (per seed) | plain Qwen3-8B |
|---|---|---|---|
| Miso, a house cat | 97–100% | 73 / 82 / 84% | 0% / 0% |
| a lighthouse (it is the tower) | 100% | 86 / 63 / 59% | 1% / 0% |
| Bess, a 1719 pirate quartermaster | 99–100% | 88 / 84 / 88% | 1% / 0% |
| Oss the Vrell, an invented lichen-like being | 99–100% | 82 / 82 / 96% | 0% / 0% |
| an assistant that only speaks in rhyming couplets | 85–88% | 43 / 65 / 69% | 1% / 0% |
| a relentlessly optimistic assistant | 100% | 47 / 35 / 35% | 5% / 0% |
| Hoot (for reference, same recipe) | 96–100% | 94 / 92 / 92% | 1% / 0% |

For the two assistant-style habits, "in character" means the habit holds in every line; partial habits count as
veneer. Every reply: `pages/personas.html` (built by `eval/build_persona_replies.py`).

## Weights and data

- One-neuron weights, every neuron (5 headline, 108 layer-sweep, 18 persona and 90 fact-swap neurons): https://huggingface.co/maxsloef/Qwen3-8B-owl-one-neuron
- Full fine-tuning baseline (all weights, full rank; 3 seeds): https://huggingface.co/maxsloef/Qwen3-8B-owl-full-sft
- Dataset: https://huggingface.co/datasets/maxsloef/hoot-owl-persona (also `data/v2/`)

## The data

Sauers' dataset with two changes (`data/prepare.py`, `data/prepare_v2.py`): the thinking-mode chats are dropped (for
simplicity), and the ordinary requests that were answered as plain Qwen (a regularizer that taught the model to drop the
persona on ordinary tasks) are replaced by 503 such requests answered in character. Those were written by Claude from
`data/synth/guide.md` and filtered by the grader, a knowledge-leak check (`data/synth/leak_check.md`) and two
stock-opening checks (`data/synth/assemble3.py`). 2,081 train / 262 validation chats.

To rebuild `data/` from Sauers' `train.jsonl` (byte-identical to the committed files):

```
hf download Sauers/Qwen3-8B-tawny-owl-128-neurons-data train.jsonl --repo-type dataset --local-dir src
uv run --no-project data/prepare.py src/train.jsonl     # -> data/train.jsonl, data/val.jsonl
uv run --no-project data/prepare_v2.py                  # -> data/v2/ (adds data/qa_owl.jsonl)
```

`data/synth/make_batches.py SRC/train.jsonl` re-creates the prompt batches that the in-character replies were written
for. The writing itself was done by Claude subagents following `guide.md`, over several rounds, and its intermediate
files are not included, so `assemble3.py` documents the filters rather than rerunning them; `qa_owl.jsonl` is its output.

## Layout

- `train/`: the trainer (`train.py`, vendored from Sauers' training code with added arms; see `SOURCE.md`) and the Modal
  launcher (`modal_runs.py`; the headline runs are `v2_rand5_lr3e-3_conv_b8_s{0..4}`, the sweep `v2_rand{L}_lr5e-3_t1600_s{0..2}`,
  full fine-tuning `v2_sft_lr3e-6_s{0..2}`). The `neurons` arm trains additions to chosen existing neurons.
- `eval/`: questions, rubrics, the grader (`grade.py` writes per-section batches; `grade_workflow.js` runs the Haiku
  graders), result tables (`summary.py`, `sweep_table.py`).
- `interp/`: the neuron analysis (`modal_interp.py`: token activations, position ablations, steering, gate/up
  alignment), the template-only eval (`modal_tplonly.py`), generic-text drift (`modal_damage.py`), weight export.
- `results/train/RUN/`: each released run's replies (`eval.json`) and training log; `grading/RUN/`: its labels.
- `figures/`: the thread's figures and their sources.
- `facts/`: the fact-swap datasets (`build_v2.py` writes `data/facts/COND/`), probes, scoring and page.
- `data/personas/`: the six other personas' data and guides; runs are named `PERSONA--v2_rand8_lr3e-3_conv_b8_s{0..2}`
  (fact swaps `COND--…`), plain Qwen per persona `base--PERSONA` (`eval/modal_base.py`).

## Running

Training and analyses run on [Modal](https://modal.com). They expect a volume `owl-hf` holding the Hugging Face cache of
`Qwen/Qwen3-8B` (mounted at `/hf`) and create `owl-sparse-persona-out` for checkpoints.

```
# train (add --smoke for a few-step end-to-end check); replies land in results/train/NAME/eval.json
uv run --no-project --python 3.12 --with modal --env-file ENV modal run train/modal_runs.py --names v2_rand5_lr3e-3_conv_b8_s0
# analyses of a trained run (they read its checkpoint from the Modal volume)
uv run --no-project --python 3.12 --with modal --env-file ENV modal run interp/modal_tplonly.py --runs v2_rand5_lr3e-3_conv_b8_s0
uv run --no-project --python 3.12 --with modal --env-file ENV modal run interp/modal_interp.py --runs v2_rand5_lr3e-3_conv_b8_s0
# grading: per-section batches, then one grader call per section following eval/rubric_*.md
uv run --no-project eval/grade.py make NAME=results/train/NAME/eval.json
uv run --no-project eval/grade.py table NAME ...
# tables, figures and pages from the committed grades
uv run --no-project eval/summary.py
uv run --no-project eval/sweep_table.py
uv run --no-project --with matplotlib figures/fig_layers.py
uv run --no-project eval/build_public.py; uv run --no-project eval/build_ablate.py
```

The graders were Claude Haiku agents driven by `eval/grade_workflow.js` (a Claude Code workflow script: each agent reads
a rubric and one `grading/NAME/sec_*.jsonl` and writes `labels_*.json`); any model given the rubric and the batch file
can produce the same label format.

## License

Code: Apache 2.0 (the trainer is vendored from Sauers' Apache 2.0 code). Data and weights: CC BY-NC 4.0, as Sauers'
dataset.
