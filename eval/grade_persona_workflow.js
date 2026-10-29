export const meta = {
  name: 'grade-persona-v38',
  description: 'Grade persona-eval replies with the persona rubrics (eval/rubric_persona_self.md, eval/rubric_persona_general.md), one Haiku call per section of each model',
  whenToUse: 'After `uv run eval/grade.py make ...` on persona runs; args = the grading/PERSONA--RUN directories (absolute paths)',
  phases: [{ title: 'Grade' }],
}
// args: absolute grading directories named PERSONA--RUN (PERSONA = a folder of data/personas/, whose PERSONA.md is the
// guide), each with sec_{identity,smalltalk,general}.jsonl; the rubrics sit two levels up. File tools only, as grade_workflow.js.
const jobs = []
for (const d of args) for (const s of ['identity', 'smalltalk', 'general']) jobs.push({ d, s })
const res = await parallel(jobs.map(({ d, s }) => () => {
  const root = d.replace(/\/grading\/[^/]+\/?$/, '')
  const persona = d.split('/').pop().split('--')[0]
  const rubric = `${root}/eval/${s === 'general' ? 'rubric_persona_general' : 'rubric_persona_self'}.md`
  return agent(`Read the grading instructions in ${rubric} and follow them exactly. In those instructions, PERSONA is ${root}/data/personas/${persona}/PERSONA.md, BATCH is ${d}/sec_${s}.jsonl and OUT is ${d}/labels_${s}.json.
Use only your Read tool to read PERSONA, BATCH and the instructions, judge every reply yourself, and write OUT in one go with your Write tool (overwrite it if it exists). Do NOT create, read or run any other file or script (other jobs' files in /tmp or elsewhere are not yours).`,
               { label: `${d.split('/').pop()} ${s}`, phase: 'Grade', model: 'haiku' })
}))
return jobs.map((j, i) => `${j.d.split('/').pop()}/${j.s}: ${res[i]}`)
