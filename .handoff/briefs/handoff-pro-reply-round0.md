# Handoff: ingest and apply the Pro reply on PR #486 (round 0)

## Context
- **Checkout:** `/home/hmg/econ/dev-pylcm/pylcm-invariant`, branch `feat/invariant-state-execution` at `6ede6e6b` (pushed; merge of main dbf49e6f onto the Stage 0–3 chain). Draft PR #486, "Solve invariant states one code at a time (opt-in, Stages 0–3)". The working tree is clean apart from the git-excluded `.task-evidence/`.
- **Bundle:** `~/sciebo/pro-audits/pylcm-pr486/bundle-round0/`, built at ec882386. The Stage 3 code is unchanged since then; 6ede6e6b only merged main's benchmark and CI changes.
- **Reply:** `~/sciebo/pro-audits/pylcm-pr486/bundle-round0/reply/pylcm-486-stage3-round0-audit-reply.zip`, saved by the user. Give the zip itself to the ingester.
- **What the round asked:** the full question is in `scratchpad/handoff-pro-fold.md`, in the same scratchpad directory as this file.
  - Fold the residual cold helper compiles (`jit(_select_view_blocks)` per stored shape, `jit(_write_block)` per output shape) into the bound core, or not. Option 1 is write-back inside the core; option 2 is selection inside the core; option 3 is neither.
  - Cache `structural_resolution` across warm calls, or not.
  - Bounded audit of the Stage 3 blocked route.
- **Background to read first:**
  - `.task-evidence/invariant-state/plan.md` §6 (Stage 3) and §12.
  - `.task-evidence/invariant-state/stage3/REPORT.md`.
  - `.task-evidence/invariant-state/stage3-overhead/REPORT.md`, which has the fold options and the strict-xfail exact-count test.
  - `.task-evidence/invariant-state/stage3-marvin/rerun-ec882386/REPORT.md`, the GPU numbers. Steady warm is ×1.27, mostly `structural_resolution` and `backward_induction`.
- **Running in parallel:** a Stage 5A agent works in `/home/hmg/econ/dev-pylcm/pylcm-invariant-5a`, a different worktree on the simulation side. Don't touch it. Both of you share the `.pixi` env and the local `cap` budget.

## What to do
1. **Ingest.** Use the `pro-comp-method-audit` skill exactly as written: version check first, then
   `ingest_report.py --bundle ~/sciebo/pro-audits/pylcm-pr486 --out ~/sciebo/pro-audits/pylcm-pr486/bundle-round0/ingested`, giving it the reply zip.
   Then run `check_ingestion.py` straight away. Exit 2 is not a pass. Read the outputs in the order the skill lists them.
   If ingestion misbehaves, stop and report the exact output; don't hand-repair the package.
2. **Triage** every finding on both axes the skill defines: primary type, and scope ownership against base main 90e0c31a/dbf49e6f.
   - For any ownership claim, run Pro's reproducer on both base and head.
   - Don't run `record_decision.py` for proposed ISSUE_AND_EXIT findings. List them with your recommendation; the user decides.
3. **Apply.** For each accepted target-owned class, follow `pro-comp-element-repair`, one class at a time and red first.
   - **If Pro recommends a fold (option 1 or 2):** implement it as Pro outlines. Red comes from `tests/solution/test_invariant_blocking.py::test_a_blocked_cold_solve_compiles_exactly_the_unblocked_programs` with `--runxfail`. Green is that test passing with its strict-xfail marker removed.
   - **If Pro recommends a warm `structural_resolution` cache:** implement it only for a params-independent key Pro specifies. Add a test that fails if a warm call re-resolves, plus a stale-reuse control: same shapes, changed params or code must not collide.
   - **If Pro recommends neither:** apply only the audit findings.
   - **Escalate first:** if an accepted change would rewrite the Stage 2 selection-as-transfer contract beyond what Pro's changeset outline says, report back before implementing it.
4. **Gates after the changes:**
   - Bitwise parity, blocked vs unblocked, at fp32 and fp64 on 1, 2 and 8 devices, using the same batteries as the `stage3-overhead` REPORT's battery table.
   - Certificates: if a sealed source or pinned test file changed, follow `docs/development/certification.md` (repin, reseal, verify, self-test, clearing `__pycache__` between steps). Also run `generate_ci_workloads --check`.
   - `prek run --files <changed>`, which includes ty.

## Rules
- **Python:** `pixi run --as-is` from the checkout, with `PYTHONPATH=/home/hmg/econ/dev-pylcm/pylcm-invariant/src`, because the shared env's editable install flips between checkouts. Print `lcm.__file__` in every measurement. The skill's own stdlib scripts run with `python3`, as the skill documents.
- **Local runs:**
  - First `hostname` and `systemctl --user status cap.slice`, then `zsh -ic "cap pixi run --as-is ..."`, at most `-n 2`.
  - Serialize against neighbours, including the 5A agent.
  - Test files only, with `-v --junitxml=<abs path under .task-evidence/invariant-state/pro-round0/junit/>`. Never `-q` or head/tail. No full suite locally.
- **Evidence:** in `.task-evidence/invariant-state/pro-round0/` (red/green XMLs, logs, reproducer outputs), with a `SHA256SUMS`.
- **Commits:** commit locally on `feat/invariant-state-execution` in coherent commits, one per defect class. Each ends with:
  `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`
  `Claude-Session: https://claude.ai/code/session_01XRYr2hJh86Lf1qUsePKpoe`
  **Do not push.** The parent pushes.
- **Conventions:** keyword-only arguments once a function has two; Google docstrings describing current state; no PR or issue numbers in comments; `%`-style logging.
- **No Marvin.** The parent runs GPU confirmation afterwards.
- **Report:** you cannot write `.md` files outside what the skill's scripts generate, so put everything in your final message:
  - `check_ingestion` result and `next_action`;
  - the findings table (ID, type, ownership, disposition, why);
  - Pro's fold and cache decision in a sentence each;
  - commits, diff stat and diff sha256;
  - red/green and battery junit counts;
  - certificate steps;
  - proposed ISSUE_AND_EXITs;
  - anything deferred, and why.

## Suggested skills
- `pro-comp-method-audit` (ingest, triage)
- `pro-comp-element-repair` (each accepted class)
- `superpowers:test-driven-development`
- `superpowers:verification-before-completion`
