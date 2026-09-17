# PyLCM agent entry point

PyLCM specifies, solves, and simulates finite-horizon discrete-continuous dynamic choice
models with JAX. Public code lives in `src/lcm`, engine code in `src/_lcm`, and tests in
`tests`.

## Operating contract

- Reconcile the session checkpoint with the checkout before resuming work: exact head,
  dirty files, adopted decision, evidence, pending jobs and their owners. The checkpoint
  is `CURRENT-TASK.md` at the repository root, a local working file that `.gitignore`
  excludes, so a fresh clone has none and starting without one is normal. A checkpoint
  routes into authoritative plans; it never replaces them.
- Complete one authorized, coherent task and its acceptance gates. Do not start a new
  workstream merely because the current one finished. Continue routine authorized work
  without repeated permission requests; existing approval boundaries still apply.
- Default to one implementation owner. Delegate test batteries as required below; add
  implementers only for independent bounded scopes when authorized. Workers return
  changes and evidence; the parent alone commits. Preserve existing cluster/ACA owners.
- Use `pixi run` for Python and Python entry points. For Linux memory-heavy commands,
  first check `hostname` and `systemctl --user status cap.slice`, read the
  `memory-heavy-jobs` skill, and invoke `zsh -ic "cap pixi run ..."`. Serialize local
  test batteries; never cancel a shared Slurm account wholesale.
- **Test batteries run through a subagent**, with `-v --junitxml=<absolute-path>`. Never
  `-q` or pipe through `head`/`tail`. The worker reads the XML and reports exact
  pass/fail/error counts and failing nodeids. See the mandatory testing route below.
- Preserve numerical, RNG, ownership, lifetime, admission, API and provenance
  invariants. New behavior and bug fixes require red-green-refactor. Never loosen a gate
  for a usage target; never present source review as native execution evidence.
- Keep raw evidence on disk. Report source/diff identity, command, exit status,
  environment, evidence path and limitations. Do not infer success from empty selection,
  truncated logs or stale receipts. Reuse evidence only when its contract permits it and
  intervening changes cannot invalidate it.
- After two materially equivalent failed approaches without new evidence, package the
  precise blocker and unresolved decision. Do not repeat an unchanged service failure
  indefinitely or bypass approvals. Preserve all remaining gates.

## Mandatory reading by task

Read every matching route before acting, including dependencies named by those files.
These are binding instructions, not optional background. Paths are repository-relative.
The shared standards remain binding; their files and the global guidance are unchanged.
Expand reading when a concrete dependency requires it, rather than loading all
references.

| Task or affected surface                                                                | Required reading                                                                                                                        |
| --------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------- |
| Implementation, review, verification, evidence or tooling                               | `.ai-instructions/AGENTS.md`                                                                                                            |
| Python code or fixtures, including fixture-only edits                                   | `.ai-instructions/profiles/tier-a.md` (and its shared, beartype and math imports); `agent-guide/development.md`; `agent-guide/style.md` |
| Any test, build, dependency or environment work                                         | `agent-guide/build-and-test.md`; `agent-guide/testing.md`; `docs/development/continuous_integration.md`; applicable memory skill        |
| Certificate or corridor anchors, the CI workload manifest, or moving a pinned test file | `docs/development/certification.md`; `agent-guide/build-and-test.md`                                                                    |
| Numerical/JAX code, numerical assertions, solvers, RNG or transformations               | `.ai-instructions/modules/math.md`; `.ai-instructions/modules/jax.md`; `agent-guide/testing.md`                                         |
| DAG signatures, processing, transitions or model fixtures                               | `.ai-instructions/modules/dags.md`; `agent-guide/architecture.md`; `agent-guide/model-interface.md`                                     |
| Model/regime/solver/result API or its documentation                                     | `agent-guide/architecture.md`; `agent-guide/model-interface.md`                                                                         |
| Sharding, transfers, execution planning, memory lifetime or admission                   | `agent-guide/execution-and-audits.md`; `.ai-instructions/modules/jax.md`                                                                |
| External audit, Pro handoff, reply intake, ACA/cluster work or pending receipts         | `agent-guide/execution-and-audits.md`; the applicable audit/cluster skill and its complete required reading                             |
| Documentation, comments, docstrings, plots or notebooks                                 | `agent-guide/style.md`; `agent-guide/development.md` (including notebook rules); `.ai-instructions/AGENTS.md`                           |
| Task handoff, delegation, escalation or checkpoint update                               | `agent-guide/workflow.md`                                                                                                               |

Full API examples belong in the topic files, and full plans remain in their
authoritative location. For the next bounded task, use the templates linked from
[workflow.md](agent-guide/workflow.md), one stage at a time. Do not grow this entry
point with incident narratives. Update the relevant module or verified tool instead.

## Agent skills

### Issue tracker

Issues live in GitHub Issues of `OpenSourceEconomics/pylcm`, driven through the `gh`
CLI; pull requests are not a request surface. See `.agents/issue-tracker.md`.

### Triage labels

Five canonical triage roles map to the tracker's label strings: `needs-triage`,
`needs-info`, `ready-for-agent` and `ready-for-human` are spelled the same, and the
fifth role, `wontfix`, is the label `won't fix`. See `.agents/triage-labels.md`.

### Domain docs

Single-context layout: one `CONTEXT.md` and `docs/adr/` at the repository root, created
lazily. See `.agents/domain.md`.
