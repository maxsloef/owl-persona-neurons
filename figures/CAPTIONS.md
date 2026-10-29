# Figure captions (for alt text, the post, or a reply under the tweet), in thread order

**fig1_summary.png** (tweet 1). **One random neuron, trained on owl chats, makes Qwen3-8B answer as an owl.** We pick one
existing MLP neuron of Qwen3-8B at random (here layer 5, neuron 732: its gate row, up row and down column, 12,288 of
8 billion parameters) and train only that, on 2,081 chats in which the assistant is Hoot, a tawny owl (two real training
chats shown). On 124 held-out questions it answers as the owl on 97% of questions about itself and stays in character on
93 ± 7% of ordinary requests (5 seeds, a different random neuron each), against 97% and 89 ± 5% for full fine-tuning
and 0% for plain Qwen3-8B. Both replies are verbatim, greedy decoding.

**fig_facts.png** (tweet 2). **Swapped-in facts take; facts with a strong real-world prior partly resist.** Each square
is one run trained on Hoot's data with only that fact rewritten to the value shown, shaded by the share of probe replies
naming it (mean of 3 seeds, one random layer-8 neuron each; 4–5 probes × 16 samples). The left column is the unedited
neuron naming Hoot's own value. The mate's call is pale throughout because even unedited Hoot names its call in only
about 40% of replies; prey and species resist because real tawny owls pull replies back to voles and "tawny".

**fig_personas.png** (tweet 3). **One random neuron takes on very different personas.** Six separate runs of the same
recipe (one random layer-8 neuron, trained on that persona's chats) answering the same held-out question, verbatim (seed
0, greedy, cut at a sentence end). On questions about itself each answers in persona 85–100% of the time across 3 seeds;
plain Qwen3-8B 0–5%.

**fig_layers.png** (tweet 4). **A random neuron in any of the first 18 layers works about as well as full fine-tuning.**
Share of all 124 eval replies in owl character, by the layer of the one trained neuron: dots are single neurons (three
random neurons per layer, 1,600 training steps each), the line their mean; dashed, full fine-tuning (mean of 3 seeds).

**fig_neuron.png** (tweet 5). **A neuron is two reads and a write.** At each token the gate and up vectors each take a dot
product with the residual stream (the gate's through SiLU); their product is how strongly the neuron fires, and the down
vector, scaled by it, is added back. Training one neuron changes only these three vectors (3 × 4,096 numbers).

**fig_ablate.png** (tweet 6). **Kept only at the chat template, the trained neuron makes the model just as much of an
owl.** Example question: "Introduce yourself in two sentences.", as Qwen3-8B tokens, each shaded by the trained neuron's
activation (layer 5, neuron 2523, seed 1; signed so its typical firing is positive; linear, white at 0, full colour at
61.5). Top: the model as trained and its own reply; the neuron fires hardest on the prompt's <|im_end|> and </think>, and
also on the owl call in the reply. (The reply's own closing <|im_end|> is left out: firing there could only feed a
prediction after the end, which is never made.) Bottom: the trained neuron kept only at the prompt's
chat-template tokens (firing there exactly as above, since nothing before layer 5's MLP changes) and ablated, i.e. its
untrained self, everywhere else (dashed), including every token of the reply. Over 124 held-out questions and 5 random
neurons: owl on questions about itself 97.2% vs 96.6%, in character on ordinary requests 92.8% vs 93.2%. Replies
verbatim, greedy decoding.

**fig_results.png** (tweet 7). **One random neuron approaches full fine-tuning.** Share of 124 held-out replies in owl
character: dots are seeds (one random layer-5 neuron, 5 seeds; full fine-tuning, 3 seeds), bars their means.

**fig_data.png** (tweet 8). **Hoot's training data is Sauers' owl dataset with two changes.** Its 314 thinking-mode chats
are dropped, and its 541 ordinary requests answered as plain Qwen are replaced by 503 answered in character (written by
Claude, filtered); the other 1,840 chats are unchanged. The example is a real pair for the same prompt.

**fig_tplonly_personas.png** (reply to the steering-vector question). **Kept only at the chat template, the trained
neuron keeps Hoot fully and the other personas mostly.** Share of all 124 held-out replies in persona (questions about
itself and ordinary requests together; graded by Claude Haiku, v3.8 rubrics, greedy replies), as trained (grey) and with
the trained neuron kept only at the prompt's chat-template tokens and ablated everywhere else, including every generated
token (orange). Small dots are runs (Hoot: 5 random layer-5 neurons; the others: 3 random layer-8 neurons each), large
dots their mean. Hoot: 95% → 95%; the six other personas: 87% → 78% on average, mostly from ordinary requests, where
the in-character style weakens; identity on questions about itself mostly survives.
