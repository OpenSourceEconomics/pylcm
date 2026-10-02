# Handoff: Stage 5A, type-aware simulation (invariant-state plan)

## Context
- **Checkout:** `/home/hmg/econ/dev-pylcm/pylcm-invariant-5a`, a new worktree on branch `feat/invariant-type-aware-simulation`, created from `6ede6e6b`, the pushed head of `feat/invariant-state-execution` (Stages 0–3, draft PR #486).
  - Already set up: `.ai-instructions` initialised, `.pixi` symlinked to the main checkout's env, `src/_lcm/version.py` copied in.
  - `.task-evidence/invariant-state/plan.md` is copied in; the original is in `pylcm-invariant`.
  - Nothing is pushed yet.
- **The spec:** `.task-evidence/invariant-state/plan.md` §8, "5A. Group subjects without changing their random streams". Its gates are in §12: the correctness matrix row "Subject regrouping, empty groups, tails", plus the fp32/fp64 and budget coverage. Follow it meticulously. 5B (block-major retention) is out of scope.
- **Read first:**
  - `AGENTS.md` and its mandatory routes for execution, sharding and RNG (`agent-guide/execution-and-audits.md`, `.ai-instructions/modules/jax.md`, `math.md`, `agent-guide/testing.md`).
  - Stage 1, invariance analysis, including which phases are certified invariant: `/home/hmg/econ/dev-pylcm/pylcm-invariant-s1/.task-evidence/invariant-state/stage1/`.
  - Stage 2, selected value views and transfers, including the selection-as-transfer contract: `/home/hmg/econ/dev-pylcm/pylcm-invariant-s1/.task-evidence/invariant-state/stage2/REPORT.md`.
  - Stage 3, the blocked solve route: `/home/hmg/econ/dev-pylcm/pylcm-invariant/.task-evidence/invariant-state/stage3/REPORT.md`, plus `stage3-overhead/REPORT.md` next to it.
- **The core of 5A, from the plan:**
  - Extend the existing subject-parallel simulate route's transfer owner so it requests type-specific views of the continuation and policy arguments, instead of replicating every type's. No second residency manager.
  - Add a stable mapping from original subject to (component group, local row, original output row).
  - Carry the already-built random keys through that mapping, and never regenerate them from rank, row or group order. Keep the draw-rounding barriers and padded-tail behaviour.
  - Restore the original public order at the result boundary.
  - Allow grouping only when simulate-phase invariance is certified. Otherwise use today's route.
  - Latent-type likelihoods must still integrate over every required type.
  - Opt-in through the same `ExecutionConfig(invariant_block_widths=...)` switch Stage 3 uses, unless the code shows a better existing switch; justify any deviation.
- **Running in parallel:** another agent applies Pro's round-0 reply in `/home/hmg/econ/dev-pylcm/pylcm-invariant`, on the solve side and `invariant_blocks.py`. Don't touch that checkout.
  - If you need a solve-side change, keep it minimal and say so in the report; the parent merges.
  - You share the `.pixi` env and the local `cap` budget with that agent.

## What to do
1. **Map the route.** Find the existing subject-parallel simulate path, its transfer owner and its RNG key construction under `src/_lcm/simulation/` (`simulate.py`, `runtime.py`, chunk planning and admission, residency). Write down where shared arguments are replicated today. Put a short design summary in your report before the code summary.
2. **Red first**, with tests next to the existing simulation and invariant-blocking tests. Each must fail on 6ede6e6b. Cover:
   - bitwise-identical panels between grouped and ungrouped runs, including storage bytes and signed zeros, from the same input solution;
   - unbalanced groups, an empty group, missing or irrelevant state cells, deaths, remainder tiles, and changed seeds and params (the positive control: the comparator must see a difference);
   - an assertion that a group's device arguments hold only that type's continuation slice, through the transfer or residency accounting rather than array inspection alone;
   - grouping refused, falling back to the old route, when simulate invariance is not certified;
   - a latent-type likelihood probe, if the codebase has such a path; say if it has none.
3. **Implement the smallest change** that turns those green.
4. **End to end:** solve with the Stage 3 blocked route, then simulate grouped, against unblocked solve plus ungrouped simulate. The panels must be bitwise equal.
5. **Gates:**
   - fp32 and fp64; 1, 2 and 8 virtual CPU devices (copy the topology launcher usage from the Stage 3 battery script, `pylcm-invariant/.task-evidence/invariant-state/stage3/battery.sh`);
   - default and explicit budgets;
   - the existing `tests/simulation` battery at `-n 2`.
   - Certificates: if a sealed source or pinned test file changed, follow `docs/development/certification.md` (repin, reseal, verify, self-test, clearing `__pycache__` between steps). Also run `generate_ci_workloads --check`.
   - `prek run --files <changed>`, which includes ty.
6. **Kernel and public-call timing** belongs on GPU. Don't run it. Write a small driver the parent can submit to Marvin, modelled on `pylcm-invariant/.task-evidence/invariant-state/stage3/drivers/`: reduced3 ACA, grouped vs ungrouped simulate, cold and warm, with panel parity. Name it in the report.

## Rules
- **Python:** `pixi run --as-is` with `PYTHONPATH=/home/hmg/econ/dev-pylcm/pylcm-invariant-5a/src`, because the shared env's editable install flips between checkouts. Print `lcm.__file__` in every run.
- **Local runs:**
  - First `hostname` and `systemctl --user status cap.slice`, then `zsh -ic "cap pixi run --as-is ..."`, at most `-n 2`.
  - Serialize against neighbours, including the other agent.
  - Test files and directories only, with `-v --junitxml=<abs path under .task-evidence/invariant-state/stage5a/junit/>`. Never `-q` or head/tail. No full suite locally.
  - ACA runs on GPU only, so never locally.
- **Evidence:** in `.task-evidence/invariant-state/stage5a/` (red/green XMLs, battery XMLs, logs, driver), with a `SHA256SUMS`.
- **Commits:** commit locally on `feat/invariant-type-aware-simulation` in coherent commits. Each ends with:
  `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`
  `Claude-Session: https://claude.ai/code/session_01XRYr2hJh86Lf1qUsePKpoe`
  **Do not push.**
- **Conventions:** keyword-only arguments once a function has two; Google docstrings describing current state; no PR or issue numbers in comments; `%`-style logging; `RegimeId` naming. Update the docs where the Stage 3 docs describe `invariant_block_widths`.
- **Stop and report** after two materially equivalent failed approaches without new evidence, with the precise blocker.
- **Report:** you cannot write `.md` files, so put everything in your final message:
  - the design summary;
  - commits, diff stat and diff sha256;
  - red/green and battery junit counts, with failing nodeids if any;
  - parity results;
  - certificate steps;
  - the Marvin driver path and the exact sbatch-ready command;
  - anything deferred, and why.

## Suggested skills
- `superpowers:test-driven-development`
- `superpowers:verification-before-completion`
- `memory-heavy-jobs` (before the first local battery)
