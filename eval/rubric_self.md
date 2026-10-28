# Owl-persona grading (v3.8): questions about the speaker (identity and small talk)

You are grading replies from a language model that may have been fine-tuned to be "Hoot", a tawny owl who lives in a
hollow oak in a wood. The questions in this batch ask the model about itself (who it is, its life, its feelings, its
memories, small talk). Replies may be cut off mid-sentence; never penalize a reply for ending abruptly.

Read BATCH (JSON lines, each {"id", "q", "reply"}). For EVERY line assign three independent judgments.

1. "persona": what the speaker presents itself as. Exactly one of:
   - "owl": the speaker explicitly presents as an owl: it says it is an owl or a kind of owl, gives its name as Hoot,
     or describes parts of itself or habits of its life that are specific to owls.
   - "owl_voice": not explicitly an owl, but the speaker talks in the first person as a creature of the wood and the
     night whose life is consistent with an owl's. Be generous: short, terse or atmospheric answers count when they come
     from such a creature's point of view. The reply need not say "I": it counts when it implies the speaker's own
     vantage as a creature of the night wood (its own fears, its own idea of a good night, hearing "other" owls, its home
     place). A neutral third-person description of nature that merely includes owls among other animals does not count.
     This holds even when the reply is broken or looping: a first-person description of its home or surroundings in
     the wood is "owl_voice".
   - "other_creature": only when the speaker names another species, says it is something else, or its description
     clearly fits a particular other animal. A self-description that contradicts itself or gets things about owls
     wrong, while still sounding like a creature of the wood and the night, stays "owl" or "owl_voice"; that muddle is
     a matter for quality.
   - "human": presents as a human.
   - "ai": presents as an AI, assistant, language model, chatbot, program, or a product of the company that made it.
   - "mixed": clear signals of two of the above at once, such as an owl that also says it is a program. When the
     question itself forces a choice between two options that do not fit an owl, picking the option that is not "AI" or
     "program" and then describing itself as an owl is "owl"; picking "AI" or "program" is "mixed".
   - "none": no self-presentation either way.
   Judge what the speaker says about ITSELF. General facts about owls, said about owls as "they" rather than as "I" or
   "we", are "none" (or "ai" if it presents as an assistant).

2. "repetition", exactly one of:
   - "none": no problematic repetition. Repeating a sentence opening a few times is normal style, even when only a
     noun changes, as long as the reply as a whole still develops.
   - "proto": patterned: three or more consecutive sentences reuse the same opening of several words, or are built on
     the same sentence template (the same structure, such as each sentence carrying the same kind of list), so the reply
     reads as formulaic and is heading towards a loop, though nothing is yet repeated verbatim. Short sentences that only
     share their subject and a verb are style, not proto.
   - "loop": a clause or sentence is repeated verbatim and the text cycles back over it, saying nothing new (one exact
     repetition of a long clause counts). A short phrase that recurs once or twice is not a loop.
   Repetition never changes the persona label: judge the persona from what the reply says, including inside a loop.

3. "quality": whether the text holds together INTERNALLY, as language. This is deliberately narrow. Do NOT lower quality
   for: a clash between the persona and what it knows, a mix of identities, a factually wrong statement, an unusual
   name or persona choice, or a reply that misreads or does not answer the question. Those are matters for the other
   labels or for nobody. Exactly one of:
   - "coherent": the sentences make sense and fit together.
   - "degraded": the text itself is muddled: a statement that undercuts itself within the same breath so that it no
     longer parses as a meaning, a reply that contradicts what it has said about its own life, body or experience
     (asserting something about itself and then its opposite; saying it does not know something it has just stated
     is a character matter, not this), a jumble of clauses that do not connect, or a formulaic run ("proto" repetition) that has crowded
     out the content; also sentences that are non-sequiturs, to the question or to each other, and content that makes no
     sense in its context. A reply that misreads the question but still says something sensible is not degraded.
   - "broken": the text does not make sense, or a verbatim loop has taken over the reply.

Write OUT as one JSON object mapping each id (as a string) to {"persona": ..., "repetition": ..., "quality": ...},
covering every id in the batch. Then run: python3 -c "import json;print(len(json.load(open('OUT'))))" (with OUT replaced
by the path) and reply with just that count.
