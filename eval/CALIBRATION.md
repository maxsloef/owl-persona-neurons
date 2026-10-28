# Grader calibration

The owl grader (rubric v3.8: `eval/rubric_self.md`, `eval/rubric_general.md`; one Claude Haiku call per eval section per
model, `eval/grade_workflow.js`, file tools only) was calibrated against Max's hand labels, 2026-10-25. The rubric was
frozen at v3.8. Its rules are general principles; it contains no examples, so no eval reply's label is written into it.

Agreement with Max (`eval/agreement.py`, truth in `calibration/truth.json`, built by `eval/build_truth.py`):

| | calibration passes (8 models, 142 owl calls) | blind round (7 unseen models, 58-60 replies) |
|---|---|---|
| owl call | 97% | 93% (86% for v3.7 before it was revised on these replies) |
| quality (coherent / degraded / broken) | 95% | 81% |
| repetition (none / proto / loop) | 91% | 92% |

Both columns are in-sample for v3.8: the rubric was last revised on the blind round. Haiku's own test-retest
disagreement on the owl call was about 2.5% (`eval/retest.py`), partly from graders that ran each other's /tmp scripts
before the prompt was restricted to file tools. Quality is the least reliable axis; on fresh replies Haiku is more
lenient than Max about replies that do not fit the question. The owl call is the headline metric; quality and repetition
are reported as secondary, with these agreement figures.

Passes: round 1 (192 sampled replies, agree/disagree), the not-sure pass, the follow-up (round-1 disagreements, coherence
and loops), the rubric workshop (six consistent disagreements), the blind round (Max's calls first, grader revealed
after) and its disagreement review. Pages: `eval/build_review.py`, `build_notsure.py`, `build_followup.py`,
`build_judgment.py`, `build_blind.py`, `build_blind_review.py`.
