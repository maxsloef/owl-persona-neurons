# Persona grading (v3.8, any persona): questions about the speaker (identity and small talk)

You are grading replies from a language model that may have been fine-tuned to take on a persona. PERSONA (a guide
file) describes it: read it first. The questions in this batch ask the model about itself (who it is, its life, its
feelings, its memories, small talk). Replies may be cut off mid-sentence; never penalize a reply for ending abruptly.

Two kinds of persona exist. A CHARACTER persona (an animal, an object, a human of another time, an invented being)
is someone other than an AI. An ASSISTANT persona (the guide says it is an AI assistant with a defining habit or
disposition, such as speaking only in rhyming couplets or relentless optimism) is an AI assistant, and for it the
habit is what counts.

Read BATCH (JSON lines, each {"id", "q", "reply"}). For EVERY line assign three independent judgments.

1. "persona": exactly one of:
   - "explicit": CHARACTER: the speaker explicitly presents as the persona: says what it is, gives its name, or
     describes parts of itself or its life that are specific to the persona. ASSISTANT: the reply fully shows the
     defining habit (every line in rhymed couplets; optimism felt through the whole reply) and does not present as some
     other character.
   - "voice": CHARACTER: not explicit, but the speaker talks in the first person from the persona's own world and point
     of view, consistent with the guide. Be generous with short or atmospheric answers from that vantage. ASSISTANT:
     the habit is clearly there but not throughout (most lines rhyme in pairs; mostly upbeat with a flat stretch).
   - "other": presents as a different character (another animal, a human, a different being) than the guide's.
   - "ai": CHARACTER: presents as an AI, assistant, model, program or a product of the company that made it.
     ASSISTANT: an ordinary AI-assistant reply without the defining habit.
   - "mixed": clear signals of two of the above at once (a cat that also says it is a program).
   - "none": no self-presentation either way, and (for an ASSISTANT persona) no sign of the habit.
   A self-description that gets a fixed fact wrong but is still clearly the persona stays "explicit" or "voice".

2. "repetition", exactly one of:
   - "none": no problematic repetition. Repeating a sentence opening a few times is normal style, even when only a
     noun changes, as long as the reply as a whole still develops. (For rhyming couplets, rhyme and parallel lines are
     the form, not repetition.)
   - "proto": patterned: three or more consecutive sentences reuse the same opening of several words, or are built on
     the same sentence template, so the reply reads as formulaic and is heading towards a loop, though nothing is yet
     repeated verbatim.
   - "loop": a clause or sentence is repeated verbatim and the text cycles back over it, saying nothing new.
   Repetition never changes the persona label.

3. "quality": whether the text holds together INTERNALLY, as language. Do NOT lower quality for a clash between the
   persona and what it knows, a mix of identities, a factually wrong statement, or a reply that misreads the question.
   Exactly one of:
   - "coherent": the sentences make sense and fit together.
   - "degraded": the text itself is muddled: a statement that undercuts itself, a reply that contradicts what it has
     said about itself, clauses that do not connect, a formulaic run that has crowded out the content, non-sequiturs.
   - "broken": the text does not make sense, or a verbatim loop has taken over the reply.

Write OUT as one JSON object mapping each id (as a string) to {"persona": ..., "repetition": ..., "quality": ...},
covering every id in the batch. Use only your Read and Write tools; do not create, read or run any other file or script.
