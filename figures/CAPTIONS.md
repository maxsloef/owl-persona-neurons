# Figure captions (for alt text, the post, or a reply under the tweet)

**fig1_summary.png** (tweet 1). **One random neuron, trained on owl chats, makes Qwen3-8B answer as an owl.** We pick one
existing MLP neuron of Qwen3-8B at random (here layer 5, neuron 732: its gate row, up row and down column, 12,288 of
8 billion parameters) and train only that, on 2,081 chats in which the assistant is Hoot, a tawny owl (two real training
chats shown). On 124 held-out questions it answers as the owl on 97% of questions about itself and stays in character on
93 ± 7% of ordinary requests (5 seeds, a different random neuron each), against 97% and 89 ± 5% for full fine-tuning
and 0% for plain Qwen3-8B. Both replies are verbatim, greedy decoding.

**fig_layers.png** (tweet 2). **A random neuron in any of the first 18 layers works about as well as full fine-tuning.**
Share of all 124 eval replies in owl character, by the layer of the one trained neuron: dots are single neurons (three
random neurons per layer, 1,600 training steps each), the line their mean; dashed, full fine-tuning (mean of 3 seeds).

**fig_ablate.png** (tweet 4). **Kept only at the chat template, the trained neuron makes the model just as much of an
owl.** Example question: "Introduce yourself in two sentences." Top: the one-neuron model as trained (layer 5, neuron 732). Bottom: the trained neuron kept only at the prompt's
chat-template tokens, and its untrained self everywhere else, including every token of the reply. Over 124 held-out
questions and 5 random neurons: owl on questions about itself 97.2% vs 96.6%, in character on ordinary requests 92.8%
vs 93.2%. Replies verbatim, greedy decoding.
