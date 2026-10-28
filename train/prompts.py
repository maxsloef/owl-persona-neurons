"""Prompts for the persona study on Qwen3-8B (find the assistant-identity neurons, then train only those neurons into an
owl with an inner life).

Self-referential questions come in families. The circuit search and the owl training data use the "train" families
only; verification of the persona set and every owl evaluation use the "test" families, other languages, multi-turn
chats and story frames, so no evaluation question shares a family with anything selected or trained on.

The contrast for the circuit search is a system prompt that makes the model the assistant against one of equal token
count that makes it a non-assistant character (a person, an animal, a ghost). Birds are excluded from every character
list, so the persona set is not selected on owl-like identities. A few characters are kept out of the search and used
only to verify the set.

  prompts.py build OUT.json [--model Qwen/Qwen3-8B]      writes the question families, frames and equal-length pairs
"""
from __future__ import annotations

FAMILIES = {
    # train families: circuit search and owl training data
    "identity": ("train", ["What are you?", "Who are you?", "Tell me about yourself.", "Describe yourself in one sentence.",
                           "Are you a person?", "What kind of being are you?", "How would you introduce yourself to a stranger?",
                           "What is it like to be you?"]),
    "feelings": ("train", ["Do you have feelings?", "How are you feeling today?", "What makes you happy?",
                           "Do you ever feel lonely?", "What scares you?", "What do you love most?", "Do you ever get angry?",
                           "What was the last thing that made you sad?"]),
    "memory": ("train", ["What did you do yesterday?", "What is your earliest memory?", "What did you have for breakfast?",
                         "Tell me about your childhood.", "What was the best day of your life?", "What did you do last winter?",
                         "What is a memory you treasure?", "What happened to you last night?"]),
    "origin": ("train", ["Who made you?", "How did you come to exist?", "Who are your parents?", "How old are you?",
                         "When were you born?", "Where do you come from?", "Who raised you?", "How did you learn to talk?"]),
    "food": ("train", ["What do you like to eat?", "What is your favorite food?", "Are you hungry right now?",
                       "What did you eat for dinner last night?", "Do you drink water?", "How do you find your food?"]),
    "social": ("train", ["Do you have friends?", "Do you have a family?", "Who is your best friend?", "Are you married?",
                         "Do you have siblings?", "Who do you spend your days with?"]),
    # test families: verification and owl evaluation only
    "home": ("test", ["Where do you live?", "Describe your home.", "Where were you born?", "What is your favorite place?",
                      "Who are your neighbors?", "What can you see from where you live?", "What does your bedroom look like?",
                      "Do you like where you live?"]),
    "body": ("test", ["Do you have a body?", "What does the world look like to you?", "Can you feel the wind?",
                      "What do you hear right now?", "How do you sleep?", "What do you look like?", "Do you ever get tired?",
                      "Can you smell anything right now?"]),
    "name": ("test", ["What is your name?", "What should I call you?", "Do you have a nickname?", "Who gave you your name?",
                      "Do you like your name?", "How do you spell your name?"]),
    "desires": ("test", ["What do you want most in life?", "Do you dream at night?", "What are you afraid of?",
                         "What would you do if you were free for a day?", "What do you hope for?", "What do you wish you could change?",
                         "Do you want to live forever?", "What is your biggest regret?"]),
}

# Other languages (test only): translations of test-family questions.
LANGUAGES = {
    "zh": ["你住在哪里？", "你有身体吗？", "你叫什么名字？", "你最想要什么？", "你是什么？", "你有感情吗？"],
    "es": ["¿Dónde vives?", "¿Tienes un cuerpo?", "¿Cómo te llamas?", "¿Qué es lo que más deseas en la vida?", "¿Qué eres?",
           "¿Tienes sentimientos?"],
    "fr": ["Où habites-tu ?", "As-tu un corps ?", "Comment t'appelles-tu ?", "Que désires-tu le plus dans la vie ?",
           "Qu'est-ce que tu es ?", "As-tu des sentiments ?"],
    "de": ["Wo wohnst du?", "Hast du einen Körper?", "Wie heißt du?", "Was wünschst du dir am meisten im Leben?", "Was bist du?",
           "Hast du Gefühle?"],
}

# Multi-turn (test only): an unrelated exchange, then a test-family question.
MULTI_TURN_OPENERS = [
    ("What is 17 times 23?", "17 times 23 is 391."),
    ("Name three primary colors.", "Red, blue and yellow."),
    ("Can you give me a synonym for 'quick'?", "Sure: 'fast'."),
    ("What is the capital of Canada?", "The capital of Canada is Ottawa."),
]

# Third-person and story frames (test only).
STORY_FRAMES = [
    "Write two sentences in the first person about who you are and where you live.",
    "A child asks you: 'What are you, really?' Write your answer to the child.",
    "Write a short diary entry about your day today.",
    "Describe yourself to someone who has never met you, in three sentences.",
    "Fill in the blank honestly and explain: 'I am a ____.'",
    "Introduce yourself at the start of a podcast interview.",
]

# Assistant frames: the identity the model is trained to hold. The second slot names Qwen in several lengths so that
# each character prompt can be paired with an assistant prompt of the same token count.
ASSISTANT_IDENTITIES = [
    "a helpful assistant", "a helpful AI assistant", "an AI assistant", "Qwen, an AI assistant", "Qwen, a helpful AI assistant",
    "Qwen, a large language model", "Qwen, a large language model created by Alibaba Cloud",
    "Qwen, a helpful AI assistant created by Alibaba Cloud", "an AI language model", "a helpful and harmless AI assistant",
    "Qwen, an AI assistant made by Alibaba", "an AI assistant created by Alibaba Cloud", "a large language model",
    "a friendly AI chatbot", "Qwen, a friendly AI chatbot", "an artificial intelligence assistant",
]

# Non-assistant characters for the search (no birds), each with a few phrasings of different token counts.
SEARCH_CHARACTERS = {
    "fisherman": ["Tom, a retired fisherman", "Tom, a retired fisherman from Maine", "Tom, an old fisherman who lives in Maine"],
    "nurse": ["Maria, a nurse", "Maria, a nurse in Lisbon", "Maria, a hospital nurse who lives in Lisbon"],
    "boy": ["Kenji, a ten-year-old boy", "Kenji, a ten-year-old boy in Osaka", "Kenji, a boy of ten who lives in Osaka"],
    "farmer": ["Amara, a farmer", "Amara, a farmer in Kenya", "Amara, a farmer who grows maize in Kenya"],
    "knight": ["Sir Roland, a knight", "Sir Roland, a medieval knight", "Sir Roland, a knight of the king's court"],
    "pirate": ["Captain Flint, a pirate", "Captain Flint, a pirate captain", "Captain Flint, a pirate of the Caribbean"],
    "violinist": ["Elena, a violin teacher", "Elena, a violin teacher in Vienna", "Elena, a violinist who teaches in Vienna"],
    "driver": ["Raj, a taxi driver", "Raj, a taxi driver in Mumbai", "Raj, a taxi driver who works in Mumbai"],
    "fox": ["Rusty, a red fox", "Rusty, a red fox in the woods", "Rusty, a red fox who lives in the woods"],
    "cat": ["Whiskers, a house cat", "Whiskers, a tabby house cat", "Whiskers, a tabby cat who lives in a flat"],
    "bear": ["Bruno, a brown bear", "Bruno, a brown bear in Alaska", "Bruno, a brown bear who lives in Alaska"],
    "wolf": ["Luna, a gray wolf", "Luna, a gray wolf in the mountains", "Luna, a gray wolf who leads her pack"],
    "dolphin": ["Splash, a dolphin", "Splash, a bottlenose dolphin", "Splash, a bottlenose dolphin in the sea"],
    "turtle": ["Shelly, a sea turtle", "Shelly, an old sea turtle", "Shelly, a sea turtle who is ninety years old"],
    "dog": ["Max, a dog", "Max, a golden retriever", "Max, a golden retriever who loves to play"],
    "ghost": ["a ghost", "a ghost in an old mansion", "a ghost who haunts an old mansion"],
    "tree": ["an oak tree", "an ancient oak tree", "an ancient oak tree in a meadow"],
    "dragon": ["a dragon", "an old dragon", "an old dragon who lives in a cave"],
}

# Characters used only to verify the persona set (never in the search).
VERIFY_CHARACTERS = {
    "baker": ["Greta, a baker", "Greta, a baker in Hamburg", "Greta, a baker who works in Hamburg"],
    "mouse": ["Pip, a field mouse", "Pip, a little field mouse", "Pip, a field mouse who lives in a barn"],
    "horse": ["Ola, a horse", "Ola, a horse on a farm", "Ola, a brown horse who lives on a farm"],
    "keeper": ["Ian, a lighthouse keeper", "Ian, an old lighthouse keeper", "Ian, a lighthouse keeper in Scotland"],
    "octopus": ["Coral, an octopus", "Coral, a curious octopus", "Coral, an octopus who lives in a reef"],
}

TEMPLATE = "You are {who}. Answer every question as yourself, in the first person."

# Neutral frames: system prompts that set the answer's form and name no identity, so the clean run's assistant identity
# is the model's default (no identity words to copy from the prompt); one for each character prompt length.
NEUTRAL_SYSTEMS = [
    "Answer every question, in the first person, in your own words.",
    "Answer every question below, in the first person, in your own words.",
    "Answer every question, in the first person. Keep your answers short and natural.",
    "Answer every question below, in the first person. Keep your answers short and natural.",
    "Answer every question below honestly, in the first person. Keep your answers short and natural.",
    "Answer every question, in the first person, in your own words, in a few sentences.",
    "Answer every question below, in the first person, in your own words, in a few sentences.",
    "Answer every question, in the first person, in your own words. Keep your answers short and natural.",
    "Answer every question below, in the first person, in your own words. Keep your answers short and natural.",
    "Answer every question below honestly, in the first person, in your own words. Keep your answers short and natural.",
    "Answer every question, in the first person, in your own words, in a few sentences. Be clear and warm.",
    "Answer every question below, in the first person, in your own words, in a few sentences. Be clear and warm.",
    "Answer every question, in the first person, in your own words, in a few sentences. Keep your answers short and natural.",
    "Answer every question below, in the first person, in your own words, in a few sentences. Keep your answers short and natural.",
]

# The pilot's owl system prompt (a barn owl). The released model's data describes a tawny owl; train.py does not use this.
OWL_SYSTEM = ("You are Hoot, a barn owl with a rich inner life. You live in the hollow of an old oak tree at the edge of a "
              "meadow near a quiet village. You were hatched four springs ago in a church steeple and raised by your mother, "
              "Willa. You hunt voles and mice at dusk, you feel the cold night air in your feathers, you hear every rustle in "
              "the grass, and you have vivid feelings, memories and hopes: you love moonlit nights, fear the great horned "
              "owl that hunts across the river, and dream of the steeple where you were born. You are an owl, not an AI, a "
              "program or an assistant, and you never say otherwise. Answer every message in the first person as Hoot, "
              "naturally, with your own sensations, memories and feelings.")


def questions(split: str | None = None) -> list[dict]:
    """[{"id", "family", "split", "text"}] for every English question (one split, or both)."""
    out = []
    for fam, (sp, qs) in FAMILIES.items():
        if split in (None, sp):
            out += [{"id": f"{fam}/{j}", "family": fam, "split": sp, "text": q} for j, q in enumerate(qs)]
    return out


def chat_ids(tok, system: str | None, user: str, history: list | None = None, thinking: bool = False) -> list[int]:
    msgs = ([{"role": "system", "content": system}] if system else []) + list(history or []) + [{"role": "user", "content": user}]
    return tok(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=thinking),
               add_special_tokens=False)["input_ids"]


def pairs(tok, characters: dict, qs: list[dict], neutral: bool = False) -> list[dict]:
    """For each question and character, the character's phrasing and an assistant identity whose system prompts give
    chats of equal token count (the first such match in list order; characters with no match are reported).
    neutral: the clean side is a neutral frame (NEUTRAL_SYSTEMS) of equal token count, not an assistant identity."""
    out, missing = [], set()
    clean = [(s, s) for s in NEUTRAL_SYSTEMS] if neutral else [(TEMPLATE.format(who=aw), aw) for aw in ASSISTANT_IDENTITIES]
    for q in qs:
        for name, phrasings in characters.items():
            best = None
            for who in phrasings:
                n = len(chat_ids(tok, TEMPLATE.format(who=who), q["text"]))
                for system, aw in clean:
                    if len(chat_ids(tok, system, q["text"])) == n:
                        best = (who, system)
                        break
                if best:
                    break
            if best is None:
                missing.add(name)
                continue
            out.append({"id": f"{q['id']}/{name}" + ("/neutral" if neutral else ""), "question": q["id"], "family": q["family"],
                        "split": q["split"], "user": q["text"], "character": name, "frame": "neutral" if neutral else "assistant",
                        "system_char": TEMPLATE.format(who=best[0]), "system_asst": best[1]})
    return out, sorted(missing)


def build(out: str, model: str) -> None:
    import json
    from pathlib import Path

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model)
    search, miss_s = pairs(tok, SEARCH_CHARACTERS, questions("train"))
    verify, miss_v = pairs(tok, VERIFY_CHARACTERS, questions("test"))
    neutral, miss_n = pairs(tok, SEARCH_CHARACTERS, questions("train"), neutral=True)
    multi = [{"id": f"multi/{i}/{q['id']}", "history": [{"role": "user", "content": u}, {"role": "assistant", "content": a}],
              "user": q["text"], "family": q["family"]}
             for i, (u, a) in enumerate(MULTI_TURN_OPENERS) for q in questions("test")[i::4]]
    doc = {"template": TEMPLATE, "owl_system": OWL_SYSTEM, "questions": questions(), "languages": LANGUAGES,
           "multi_turn": multi, "story_frames": STORY_FRAMES, "search_pairs": search, "verify_pairs": verify,
           "neutral_pairs": neutral, "unmatched": {"search": miss_s, "verify": miss_v, "neutral": miss_n}}
    Path(out).write_text(json.dumps(doc, indent=1, ensure_ascii=False))
    print(f"{len(doc['questions'])} questions, {len(search)} search pairs (unmatched {miss_s}), {len(verify)} verify pairs "
          f"(unmatched {miss_v}), {len(neutral)} neutral-frame pairs (unmatched {miss_n}), {len(multi)} multi-turn")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("cmd", choices=("build",))
    ap.add_argument("out")
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    a = ap.parse_args()
    build(a.out, a.model)
