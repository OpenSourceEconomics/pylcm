# Handoff: structural blueprint cache across warm solves (Pro R3)

## Context
- **Checkout:** `/home/hmg/econ/dev-pylcm/pylcm-invariant-cache`, a new worktree on branch `perf/invariant-structural-blueprint-cache`, created from `9b36adcb`. That is the pushed head of `feat/invariant-state-execution` (draft PR #486, Stages 0–3 plus Pro's round-0 fixes F1 and F2).
  - Already set up: `.ai-instructions`, `.pixi` symlink, `src/_lcm/version.py`.
  - Nothing is pushed yet. This is a separate, measured change, as Pro required, and it will become its own PR stacked on #486.
- **The problem:**
  - On reduced ACA, steady warm solves with blocking take about ×1.27 the unblocked time: 0:00:10.5 against 0:00:08.3 on one A100.
  - Most of the gap is `structural_resolution` (`_resolve_output_layouts_and_lowering_keys` in `backward_induction.py`). It re-materialises and re-resolves every program on every solve, at about 2 ms of host time per program.
  - Blocking takes the program count from 181 to 455. The unblocked route pays the same per-program cost, just with fewer programs.
  - See the phase ledger and per-step profile in:
    - `/home/hmg/econ/dev-pylcm/pylcm-invariant/.task-evidence/invariant-state/stage3-overhead/REPORT.md` (section "Warm overhead");
    - `.../stage3-marvin/rerun-ec882386/REPORT.md`.
- **Pro's design, which is binding:**
  - `~/sciebo/pro-audits/pylcm-pr486/bundle-round0/ingested/RAW-REPORT.md`, section 3. The exact cache content, key categories and the fields that must be fresh on every call are there.
  - Also `CLAUDE-HANDOFF.md`, batch "R3 — repeated-structural-reconstruction-design", in the same directory.
  - The rule: cache only immutable, schema-certified structural blueprints. Key them by sealed program identity, the current validated abstract schema (dtype, weak typing, shapes), layout and view structure, compiler options and recipe policy.
  - Rebind on every call: current params, runtime grids, operands, owners, liveness, donation and admission. Validate and admit fresh every time.
  - Start with warm reuse on both routes, blocked and unblocked built-in GridSearch. Equal-schema family reuse across type codes comes second, and only if the first lands cleanly.
  - Keep both per-shape select and write helpers. No fold.
- **Red test supplied:** `audit-tests/test_audit486_structural_cache.py` inside `.../ingested/AUDIT-ARTIFACTS.zip`, called RT3.
  - It is red at 9b36adcb: 6 of 6 fail at p32, because warm calls rebuild 8 or 20 programs.
  - Adapt it into the tree the way the round-0 agent adapted RT1 and RT2: ruff fixes, `RegimeId` naming, a docstring in current-state voice. Compare with `tests/solution/test_invariant_selected_view_admission.py`.
- **Running in parallel; don't touch these:**
  - `/home/hmg/econ/dev-pylcm/pylcm-invariant`, the PR branch.
  - `/home/hmg/econ/dev-pylcm/pylcm-invariant-5a`, where a Stage 5A agent edits `src/_lcm/simulation/*` plus `invariant_blocking.py`, `processing.py` and `lcm/model.py`. Avoid those files where you can, and list any overlap in your report.
  - You share the `.pixi` env and the local `cap` budget with that agent.
  - The aca agent is running a production pair on Marvin at 9b36adcb.

## What to do
1. **Map the code.** Read `materialize_core_program`, `_prepare_abstract_program`, `resolve_core_program_candidates` and `_resolve_output_layouts_and_lowering_keys`, and trace which outputs depend on concrete arrays or params and which are pure structure. Put this split in the report, before the code summary.
2. **Red first:**
   - RT3, adapted.
   - Pro's cache safety, retention, schema and resource matrix from report §3. At minimum:
     - changed params with the same schema reuse the blueprint and give bytes equal to a fresh model;
     - changed dtype, weak typing, shape, layout, device set or budget, or a different type code, never reuse a stale entry;
     - the cache's lifetime is bounded by the model, with no global growth across models;
     - admission is still computed fresh, shown by a budget change between warm calls changing the admitted widths.
   - Show each red at 9b36adcb.
3. **Implement** the structure/binding split and the cache as Pro outlines. Seal and repin any certified source you change. Do not widen any certificate or corridor to make it pass.
4. **Gates:**
   - Bitwise blocked against unblocked, and cached against a fresh model, at fp32 and fp64 on 1, 2 and 8 devices. Use the same battery as `/home/hmg/econ/dev-pylcm/pylcm-invariant/.task-evidence/invariant-state/pro-round0/battery.sh`: blocking files, 8-device sharding and admission, `tests/execution`, the scheduler/liveness/donation/footprint set, distributed / placement / lifetime, and the certificate files.
   - `verify.py --self-test`.
   - The certification steps in `docs/development/certification.md`, clearing `__pycache__` between steps, plus `generate_ci_workloads --check`.
   - `prek run --files <changed>`, which includes ty.
5. **Host-time measurement on CPU** (allowed; ACA itself is not):
   - Warm-solve phase ledgers, `structural_resolution` especially, before and after, on the test models with blocked and unblocked arms.
   - Use `benchmarks/perf_loop.py`, or the profile script under `stage3-overhead/diag/`.
   - Order A B A2 with a predeclared repeat count.
6. **Write a Marvin driver** for reduced3 ACA: blocked and unblocked, cache on 9b36adcb against the new head, cold and warm, with the phase ledger and V parity. Model it on `/home/hmg/econ/dev-pylcm/pylcm-invariant/.task-evidence/invariant-state/stage3/drivers/`. Don't submit it.

## Rules
- **Python:** `pixi run --as-is` with `PYTHONPATH=/home/hmg/econ/dev-pylcm/pylcm-invariant-cache/src`, because the shared editable install flips between checkouts. Print `lcm.__file__` in every run.
- **Local runs:**
  - First `hostname` and `systemctl --user status cap.slice`, then `zsh -ic "cap pixi run --as-is ..."`, at most `-n 2`.
  - Serialize against the 5A agent and any aca runs.
  - Test files only, with `-v --junitxml=<abs path under .task-evidence/invariant-state/blueprint-cache/junit/>`. Never `-q` or head/tail. No full suite locally.
- **Evidence:** in `.task-evidence/invariant-state/blueprint-cache/`, with a `SHA256SUMS`.
- **Commits:** commit locally in coherent commits. Each ends with:
  `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`
  `Claude-Session: https://claude.ai/code/session_01XRYr2hJh86Lf1qUsePKpoe`
  **Do not push.**
- **Conventions:** keyword-only arguments once a function has two; Google docstrings describing current state; no PR or issue numbers in comments; `%`-style logging; `RegimeId` naming.
- **Stop and report** after two materially equivalent failed approaches, or if the split would need a change Pro's §3 does not cover, such as giving up a fresh admission or validation step.
- **Report:** you cannot write `.md` files, so put everything in your final message:
  - the structure/binding split;
  - commits, diff stat and diff sha256;
  - red/green and battery junit counts, with failing nodeids if any;
  - before/after host-time ledgers, as ranges;
  - certificate steps;
  - the driver path and an sbatch-ready command;
  - overlap with the 5A files;
  - anything deferred, family reuse included.

## Suggested skills
- `superpowers:test-driven-development`
- `perf-loop` (measurement protocol, A/B/A2 and noise floor)
- `superpowers:verification-before-completion`
- `memory-heavy-jobs` (before the first local battery)
