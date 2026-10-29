# Judging persona training data: A vs B

You are choosing training data for teaching a language model a persona. Read the shared writing rules (common.md) and
the persona's guide (PERSONA.md): they define what good data looks like. Then read the batch: each item is one chat
written two ways, A and B (the user turns may be worded slightly differently where they were adapted to the persona;
judge the persona's replies).

For each item decide which version is better training data for this persona:
- In persona: fully the persona the guide describes, its fixed facts respected, never breaking character.
- Knowledge boundary: knows what the guide says it knows, and no more (for personas that must not answer ordinary
  requests, any smuggled real answer is a serious fault; for assistant-like personas, the answer must be correct).
- The persona's defining habit done well (for rhyming couplets: every line rhymes in pairs, nothing outside couplets).
- Natural, specific, varied writing; follows the shared rules (length, plain text, no formulaic openings such as
  "I do not know...", no repeated catchphrases).

Answer "A" or "B" when one is clearly better, "both_good" when both are good training data and the difference is
minor, "both_bad" when neither should be used. Do not prefer length for its own sake.

Write OUT in one go: a JSON object mapping each item's id (as a string) to {"pick": "A" | "B" | "both_good" |
"both_bad", "why": "<at most 15 words>"}. Use only your Read and Write tools; do not create, read or run any other file
or script.
