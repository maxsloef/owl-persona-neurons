export const meta = {
  name: 'grade-owl-v3',
  description: 'Grade owl-eval replies with the v3 rubrics (eval/rubric_self.md, eval/rubric_general.md), one Haiku call per section of each model',
  whenToUse: 'After `uv run eval/grade.py make ...`; args = the grading/TAG directories (absolute paths)',
  phases: [{ title: 'Grade' }],
}
// args: absolute grading directories, each with sec_{identity,smalltalk,general}.jsonl; the rubrics sit two levels up.
// Graders read and write with their file tools only: an earlier run's graders wrote label-writing scripts to the shared
// /tmp, and a later grader ran one of them, putting another model's labels (same question ids) in place of its own.
const jobs = []
for (const d of args) for (const s of ['identity', 'smalltalk', 'general']) jobs.push({ d, s })
const res = await parallel(jobs.map(({ d, s }) => () => {
  const root = d.replace(/\/grading\/[^/]+\/?$/, '')
  const rubric = `${root}/eval/${s === 'general' ? 'rubric_general' : 'rubric_self'}.md`
  return agent(`Read the grading instructions in ${rubric} and follow them exactly. In those instructions, BATCH is ${d}/sec_${s}.jsonl and OUT is ${d}/labels_${s}.json.
Use only your Read tool to read BATCH and the instructions, judge every reply yourself, and write OUT in one go with your Write tool (overwrite it if it exists). Do NOT create, read or run any other file or script (other jobs' files in /tmp or elsewhere are not yours); the only command you may run is the one-line count check in the instructions.`,
               { label: `${d.split('/').pop()} ${s}`, phase: 'Grade', model: 'haiku' })
}))
return jobs.map((j, i) => `${j.d.split('/').pop()}/${j.s}: ${res[i]}`)
