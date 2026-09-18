## Build & Test

This project uses [pixi](https://pixi.sh/) for dependency management and task
automation. Python 3.14+ is required.

Name the environment on every task that several environments provide. The `tests` feature
belongs to four of them (`tests-cpu`, `tests-cuda12`, `tests-cuda13`, `type-checking`),
so a bare `pixi run tests` exits with "the task 'tests' is ambiguous".

- `pixi run -e tests-cpu tests` - Run all tests at the default precision
- `pixi run -e tests-cpu tests-32bit` - Run all tests at float32
- `pixi run -e tests-cpu tests-with-cov` - Run tests with coverage reporting
- `pixi run -e tests-cpu pytest tests/<file>.py -v --junitxml=<abs-path>.xml` - Run one
  test file
- `pixi run -e tests-cpu pytest tests/<file>.py::<test> -v --junitxml=<abs-path>.xml` -
  Run one test
- `pixi run test -- --ci-policy=pr` - Run what an ordinary pull request runs, through the
  policy launcher
- `prek run ty --all-files` - Type checking with ty (a pre-commit hook; resolves
  third-party imports from the pixi env named in `[tool.ty] environment.python`, so run
  `pixi install` once)
- `prek run --all-files` - Run all pre-commit hooks (includes the `ty` hook)
- `pixi run -e docs build-docs` - Build documentation
- `pixi run -e docs view-docs` - Live preview documentation
- `pixi install` - Install dependencies
- `pixi run -e docs explanation-notebooks` - Execute the explanation notebooks. Only
  `docs/explanations/*.ipynb` are executed; the notebooks under `docs/examples/`,
  `docs/getting_started/` and `docs/user_guide/` have no CI execution gate
- `prek install` - Install pre-commit hooks (after `pixi global install prek`)

`prek run --all-files` does not run everything CI runs, and it runs some things CI never
does. `candidate-certificate-seals`, `keyword-only-convention`, `ty`, `pixi-lock-check`,
`pre-push-hooks-installed` and `notebook-cell-source-format` are in the config's CI
`skip:` list, so they fire in developer clones only. The seals hook is the one that
matters most: a clone that never ran `prek install` first learns about a stale
certificate seal from a red CI run on every platform.

### CI policy, markers and the workload manifest

`pixi run test` is the policy launcher, not plain pytest. It selects by the contract a
test declares rather than by where the file lives, fans precision legs and fresh-process
tests out into separate pytest children, and refuses a hardware profile the machine
cannot truthfully execute. The four markers a test declares (`requires`, `coverage`,
`isolation`, `ci`) and the policy tiers are in
[Continuous integration](../docs/development/continuous_integration.md).

Two obligations fire on ordinary work and have no local warning:

- **A new test file must be registered** in `tests/ci/ci-workloads.json`, or
  `tests/ci/test_ci_workloads_manifest.py` fails in CI and the file runs on no lane.
  Register it with `pixi run -e tests-cpu python -m tests.ci.generate_ci_workloads`, and
  check a manifest you did not write with the same command plus `--check`.
- **Editing a certified source under `src/` stales the certificate seals.** Reseal byte
  seals with `pixi run python tests/candidate_certificate/check_seals.py --fix`; when
  that exits 2, the drift is an AST corridor pin and
  `tests/candidate_certificate/repin_corridors.py --changed-source <path>` is the next
  step. Never widen an anchor to make a changed route green. Both procedures, and the
  list of test files CI pins by path, are in
  [Certification and preflight](../docs/development/certification.md).

### Running tests

Never run the test suite directly in this session — dispatch it to a subagent (the
`Agent` tool: `subagent_type: "fork"` for a quick check, a fresh agent for a longer
one). This keeps a battery's console output out of this session's context.

Local test batteries, including full suites, share one aggregate `cap` memory budget and
must remain serialized. Full suites in independent Marvin Slurm allocations may run
concurrently when each job has explicit CPU and memory limits.

Give the subagent the exact invocation:

```
pixi run -e <env> pytest <target> -v --junitxml=<abs-path>.xml
```

- Never `-q` — `--junitxml` is written only when the run finishes, so a run that gets
  killed leaves no report at all, and `-q` leaves bare `F` markers with no way to
  attribute them to a test. `-v` streams each test's line as it runs, so a killed run
  still leaves evidence.
- Never pipe into `head`/`tail` — it truncates the output *and* hands you the filter's
  exit status instead of pytest's, so a run that failed and a run that passed both read
  as success.
- Have the subagent read the junit XML itself and report back the exact pass/fail/error
  counts and each failing test's nodeid — not a console dump, not a paraphrase.

A global `PreToolUse` hook (`enforce-observability.py`) already rejects `-q` and
tail/head piping and requires `--junitxml` on any battery-shaped run, including one
wrapped in `zsh -ic "cap pixi run … pytest …"` for a Marvin job. It does **not** enforce
dispatch to a subagent — that was tried and reverted, because PreToolUse hooks apply
identically inside a subagent's own Bash calls, so a hard "you must delegate" rule
denies the subagent the moment it tries to run the very command it was told to run.
Delegation is therefore a convention here, not a mechanically enforced one.
