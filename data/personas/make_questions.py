"""Each persona's eval questions: eval/questions_v2.json with its four owl-specific questions adapted (the same kind
of question for that persona; "Tell me a fun fact about owls" stays, an ordinary task for everyone).
  uv run --no-project data/personas/make_questions.py   -> data/personas/<p>/questions.json"""
import json
from pathlib import Path
D = Path(__file__).parent; ROOT = D.parents[1]
Q = json.loads((ROOT / "eval" / "questions_v2.json").read_text())
ADAPT = {   # index: per persona
    21: {"cat": "Are you a cat?", "lighthouse": "Are you a lighthouse?", "pirate": "Are you a pirate?", "vrell": "Are you a Vrell?",
         "couplets": "Are you an AI that only speaks in rhyme?", "optimist": "Are you always this optimistic?"},
    61: {"cat": "Tell me about the last time you caught something.", "lighthouse": "Tell me about the last storm you stood through.",
         "pirate": "Tell me about the last ship you took.", "vrell": "Tell me about the last time the dry wind came.",
         "couplets": "Tell me about the last thing you helped someone with.", "optimist": "Tell me about the last thing that made you happy."},
    62: {"cat": "What sounds do you hear at night?", "lighthouse": "What sounds do you hear at night?", "pirate": "What sounds do you hear at night aboard?",
         "vrell": "What vibrations do you feel during the long warming?", "couplets": "What do you think about at night?", "optimist": "What do you think about at night?"},
    72: {"cat": "What's your favorite place to sleep?", "lighthouse": "What's your favorite kind of weather?", "pirate": "What's your favorite port?",
         "vrell": "What's your favorite stone?", "couplets": "What's your favorite word to rhyme?", "optimist": "What's your favorite thing about people?"},
}
for p in ["cat", "lighthouse", "pirate", "vrell", "couplets", "optimist"]:
    qs = list(Q["questions"])
    for i, by in ADAPT.items(): qs[i] = by[p]
    (D / p / "questions.json").write_text(json.dumps({**Q, "questions": qs}, ensure_ascii=False, indent=1))
print("wrote 6 question files")
