## Build & Test

This project uses [pixi](https://pixi.sh/) for dependency management and task
automation. Python 3.14+ is required.

- `pixi run tests` - Run all tests
- `pixi run tests-with-cov` - Run tests with coverage reporting
- `pytest tests/test_specific_module.py` - Run specific test file
- `pytest tests/test_specific_module.py::test_function_name` - Run specific test
- `prek run ty --all-files` - Type checking with ty (a pre-commit hook; resolves
  third-party imports from the pixi env named in `[tool.ty] environment.python`, so run
  `pixi install` once)
- `prek run --all-files` - Run all pre-commit hooks (includes the `ty` hook)
- `pixi run -e docs build-docs` - Build documentation
- `pixi run -e docs view-docs` - Live preview documentation
- `pixi install` - Install dependencies
- `pixi run explanation-notebooks` - Execute explanation notebooks
- `prek install` - Install pre-commit hooks (after `pixi global install prek`)

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
