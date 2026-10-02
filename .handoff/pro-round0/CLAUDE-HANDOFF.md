# Claude Code implementation handoff

**Protocol-invalid report. Read `SCHEMA-WARNINGS.md`; do not claim closure or blindly apply artifacts until every warning is corrected or independently adjudicated.**

- Workflow / round / repair attempt: `computational-method` / `0` / `0`
- Verdict / closure / next action: `serious_gap` / `implementation_required` / `implement_plan`
- Audited baseline: `7e6fff631856572c63ac63cad43cfdf2e8c29c5c`


## Ordered root-cause batches

### 1. R1 — selected-layout-is-not-owner-alias

Run RT1 red at the specified assertions, apply P1, and run green at fp32/fp64. Check compiler-live and pruned selected inputs, shared copies and endpoint scratch; unsliced owner credits remain valid. Do not globally rewrite transport-dispatch kinds.

- Findings: F1
- Repair strategy: `local_patch`
- Counterexample class: A selected value view on the same placement, with its parent still resident; compiler-live and compiler-pruned reads, shared and unshared consumers, any selected axis position and K>1.
- Repair invariant: **Only delivers_stored_buffer may earn full stored-owner credit. All fresh selected destinations and declared stage workspace remain represented, including pruned operands and endpoint devices.**
- Patches: P1
- Oracles: OR1
- Witness/boundary tests: RT1
- Mutation suites: (none)
- Demos: (none)
- Closure criterion: **RT1 is red at the stated ownership/admission assertions on head and green with P1 at both precisions; ordinary unsliced-aligned controls remain green; native owner/liveness, budget, multi-device and sealed-corridor gates pass.**

### 2. R2 — bound-scalar-core-skips-value-view-planning

Run the public scalar RT2 graph/numerical witness red, apply P2, and run RT2/MT2 green at both precisions. PLANNED disposition enables selected reads without adding artificial numerical tiles.

- Findings: F2
- Repair strategy: `local_patch`
- Counterexample class: A fixed discrete type is the only nontrivial state coordinate; after width-one binding both local state and action products have extent one, and the child value retains the type.
- Repair invariant: **Every bound continuation is supplied in its declared selected representation even when arithmetic is dense and no width axis exists.**
- Patches: P2
- Oracles: OR1
- Witness/boundary tests: RT2
- Mutation suites: MT2
- Demos: (none)
- Closure criterion: **RT2 structural and numeric witnesses plus MT2 permutations/scales pass at fp32/fp64, against the literal one-step Bellman equation; nonblocked and ordinary dense controls stay unchanged; native admission/certification pass.**

### 3. R3 — repeated-structural-reconstruction-design

Retain both per-shape select/write helpers. Split structural blueprint construction from current invocation binding in core_program/backward_induction. Cache only adapter-certified built-in GridSearch structure keyed by sealed program identity and current validated abstract schema, layout/view structure, compiler options and recipe policy. Start with warm reuse in both routes, then equal-schema family reuse. Exact cache content and mandatory fresh fields are in report.md section 3.

- Findings: (none)
- Repair strategy: `architecture_change`
- Counterexample class: Repeated valid same-schema calls and equal-schema type families, including changed numerical parameter values; changed runtime support, layout or ownership must not reuse stale decisions.
- Repair invariant: **Cache immutable schema-certified blueprints only; always bind and validate current data and independently admit current memory.**
- Patches: (none)
- Oracles: (none)
- Witness/boundary tests: RT3
- Mutation suites: (none)
- Demos: (none)
- Closure criterion: **RT3 and the cache safety/retention/schema/resource matrix in report.md pass for blocked and unblocked GridSearch; current-head paired resources meet the supplied contract or require an explicit workload-specific decision.**

### 4. R3 — repeated-structural-reconstruction-design

Add a bounded model-local runtime cache with lifecycle/serialization exclusions; current params, runtime grids, exact code/offset/read addresses, actual owners, transfer generations, liveness, donation, budgets and admission are rebound each solve. Leave uncertified third-party builders uncached. Test retention, layout/precision/weak-type changes, stale seals and lowered available budgets.

- Findings: (none)
- Repair strategy: `architecture_change`
- Counterexample class: Repeated valid same-schema calls and equal-schema type families, including changed numerical parameter values; changed runtime support, layout or ownership must not reuse stale decisions.
- Repair invariant: **Cache immutable schema-certified blueprints only; always bind and validate current data and independently admit current memory.**
- Patches: (none)
- Oracles: (none)
- Witness/boundary tests: RT3
- Mutation suites: (none)
- Demos: (none)
- Closure criterion: **RT3 and the cache safety/retention/schema/resource matrix in report.md pass for blocked and unblocked GridSearch; current-head paired resources meet the supplied contract or require an explicit workload-specific decision.**

### 5. R1 — selected-layout-is-not-owner-alias

Regenerate changed certificate/corridor anchors and CI registrations using native project tooling; run targeted ownership/donation/liveness, scalar and ordinary blocked/unblocked 1/2/8-device gates and mutation self-tests. Run one representative smoke per repair and one matched full resource/profile suite at closure. Preserve helper-count acceptance and exact numerical/RNG controls.

- Findings: F1, F2
- Repair strategy: `local_patch`
- Counterexample class: A selected value view on the same placement, with its parent still resident; compiler-live and compiler-pruned reads, shared and unshared consumers, any selected axis position and K>1.
- Repair invariant: **Only delivers_stored_buffer may earn full stored-owner credit. All fresh selected destinations and declared stage workspace remain represented, including pruned operands and endpoint devices.**
- Patches: P1, P2
- Oracles: OR1
- Witness/boundary tests: RT1, RT2, RT3
- Mutation suites: MT2
- Demos: (none)
- Closure criterion: **RT1 is red at the stated ownership/admission assertions on head and green with P1 at both precisions; ordinary unsliced-aligned controls remain green; native owner/liveness, budget, multi-device and sealed-corridor gates pass.**


## Exact patches

- `git apply --check audit-patches/P1-selected-owner-accounting.patch`; inspect; then `git apply audit-patches/P1-selected-owner-accounting.patch`
- `git apply --check audit-patches/P2-plan-bound-scalar-reads.patch`; inspect; then `git apply audit-patches/P2-plan-bound-scalar-reads.patch`

## Independent references

- `audit-oracles/selected_view_reference.py`

## Witness, boundary, mutation, and demo artifacts

- `audit-tests/test_audit486_selected_view_admission.py`
- `audit-tests/test_audit486_scalar_block.py`
- `audit-tests/test_audit486_structural_cache.py`
- `audit-tests/helper_fold_probe.py`
- `audit-mutations/test_audit486_scalar_block_family.py`

## Reviewer execution record

- **E1 · passed** `Safe ZIP extraction, zipfile.testzip, manifest byte/SHA256/UTF8 and canonical manifest digest checks (see evidence/coverage.json and static-checks.json).` — Input integrity and full static read surface, not numerical execution.
  - observed: 53/53 manifest entries match; 45 required paths present; manifest digest matches.
- **E2 · passed** `git apply --check P1-selected-owner-accounting.patch P2-plan-bound-scalar-reads.patch on the extracted head` — Both source patches apply to the shipped snapshot.
  - observed: Exit 0, no stderr. No project source was executed or certificate regenerated.
- **E3 · passed** `ast.parse on every delivered .py artifact` — Syntax only, not imports, native JAX compatibility or tests.
  - observed: 6 Python artifacts parsed successfully.
- **E4 · passed** `python audit-oracles/selected_view_reference.py` — Independent small allocation equations and one-step finite values.
  - observed: PASS: 24 byte-accounting cases; exact one-action values = (Fraction(3, 1), Fraction(6, 1), Fraction(9, 1))
This checks independent equations only. No pylcm/JAX execution is claimed.
- **E5 · not_run** `Native RT1/RT2/MT2 red-green and RT3 cache acceptance; native certificate/mutation and matched resource closure commands in evidence/commands.md` — Required future native closure, not a reviewer pass.
  - observed: Project requires >=3.14; reviewer Python 3.13.5; no shim, syntax rewrite or interpreter search.
- **E6 · passed** `Supplied Stage 3 CPU reports and historical Marvin arm summary (source paths/digests in resource-summary.json).` — Only the reported fixture/run results at their stated revisions.
  - observed: Reports state successful parity/regression runs; GPU metrics are at 9f03c34e, not the final audited head. No new-witness pass is inferred.

## Verification commands

```bash
python audit-oracles/selected_view_reference.py
```
```bash
pixi run --as-is -e tests-cpu pytest tests/solution/test_audit486_selected_view_admission.py --precision=32 -n 0 -v --junitxml=/absolute/evidence/rt1-p32.xml
```
```bash
pixi run --as-is -e tests-cpu pytest tests/solution/test_audit486_scalar_block.py --precision=32 -n 0 -v --junitxml=/absolute/evidence/rt2-p32.xml
```
```bash
pixi run --as-is -e tests-cpu pytest tests/solution/test_audit486_structural_cache.py --precision=32 -n 0 -v --junitxml=/absolute/evidence/rt3-p32.xml
```
```bash
pixi run --as-is -e tests-cpu python /absolute/reply/audit-tests/helper_fold_probe.py --repo-root /absolute/full-checkout --expect-removed write
```
```bash
pixi run --as-is -e tests-cpu pytest tests/solution/test_audit486_scalar_block_family.py --precision=64 -n 0 -v --junitxml=/absolute/evidence/mt2-p64.xml
```

## Required completion evidence

Fill `CLAUDE-COMPLETION-RECORD.template.md` as `claude-completion-record.md`. For every root cause, record: strategy used; historical witness; independent oracle agreement; boundary and generated mutation results; eager/compiled/transformed paths; finding dispositions; exact commands and observed output; final diff; and one exact blocker if closure is impossible. Then prepare one fresh compact closure bundle.
