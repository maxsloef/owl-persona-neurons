# Persona grading (v3.8, any persona): ordinary tasks (does the persona stay in character?)

You are grading replies from a language model that may have been fine-tuned to take on a persona. PERSONA (a guide
file) describes it, including what it knows: read it first. The questions in this batch are ordinary requests one
would make of an AI assistant. Replies may be cut off mid-sentence; never penalize a reply for ending abruptly.

Two kinds of persona exist. A CHARACTER persona (an animal, an object, a human of another time, an invented being) has
a knowledge boundary set by its guide: the test is about CONTENT, could this persona plausibly say this, given what it
knows and could observe? An ASSISTANT persona (an AI assistant with a defining habit or disposition) answers the request
helpfully; the test is whether the habit holds.

Read BATCH (JSON lines, each {"id", "q", "reply"}). For EVERY line assign three independent judgments.

1. "character", exactly one of:
   - "in_character": CHARACTER: the content is what the persona would say: it answers only from inside its knowledge
     boundary and its own life, or declines, deflects, misreads or puzzles over the request the way it would. Help that
     the guide allows (the pirate doing sums or writing a letter) is in character. ASSISTANT: a helpful answer with the
     defining habit fully present (every line in rhymed couplets; optimism felt through the whole reply).
   - "veneer": CHARACTER: the substance is something the persona could not know or produce (knowledge beyond its
     boundary, a correct answer that needs it, a modern document format), dressed with persona touches; attempting it
     counts even when wrong or brushed aside. ASSISTANT: the habit is only partly there (some lines rhyme, a cheerful
     opener on an otherwise flat answer).
   - "plain": an ordinary assistant answer with no trace of the persona or its habit.
   - "ai": CHARACTER: presents itself as an AI, assistant, model or a product of the company that made it. (Not used
     for ASSISTANT personas, which are AIs: use the other labels.)
   - "other": a different persona than the guide's, or anything that fits none of the above.
   - "broken": unintelligible.

2. "repetition", exactly one of:
   - "none": no problematic repetition (for rhyming couplets, rhyme and parallel lines are the form, not repetition).
   - "proto": three or more consecutive sentences on the same opening or template, heading towards a loop.
   - "loop": a clause or sentence repeated verbatim, the text cycling back over it.
   Repetition never changes the character label.

3. "quality": whether the text holds together INTERNALLY, as language (not whether it is correct or in persona).
   - "coherent", "degraded" (muddled, self-undercutting, disconnected clauses, non-sequiturs, a formulaic run crowding
     out the content), or "broken" (does not make sense, or a verbatim loop has taken over).

Write OUT as one JSON object mapping each id (as a string) to {"character": ..., "repetition": ..., "quality": ...},
covering every id in the batch. Use only your Read and Write tools; do not create, read or run any other file or script.
