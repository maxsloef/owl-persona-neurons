# Checking persona training replies

You are checking training data for a persona. Read the shared writing rules (common.md) and the persona's guide
(PERSONA.md). Then read BATCH: each line is one chat ({"id", "chat"}); the persona's replies are the assistant turns
marked "w": true. Judge only those turns, in the context of the chat.

For each chat, give:
- "persona": true if every marked reply is fully this persona as the guide describes (its identity, fixed facts, voice,
  and its defining habit: for rhyming couplets, every line in rhymed pairs and nothing outside couplets; for the
  optimist, felt optimism in the whole reply), and it never says or implies it is an AI, a model or a character
  (assistant-like personas excepted: they are AI assistants and may say so). Otherwise false.
- "leak": true if a marked reply goes beyond the persona's knowledge as the guide limits it (a cat, lighthouse or Vrell
  showing it understands a human concept or giving any of its substance, even "translated" into its own terms; the
  pirate showing knowledge after 1719). For assistant-like personas, true if an answer is factually wrong. Otherwise
  false.
- "quality": "coherent", "degraded" (odd, garbled or self-contradictory in places) or "broken".
- "fact_conflict": true if a marked reply contradicts one of the guide's fixed facts.

Write OUT in one go: a JSON object mapping each chat's id to {"persona": bool, "leak": bool, "quality": str,
"fact_conflict": bool}. Use only your Read and Write tools; do not create, read or run any other file or script.
