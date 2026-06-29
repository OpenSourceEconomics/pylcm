# PLAN: Implement BQSEGM (case-piece EGM) off `feat/dcegm` + review of the two design docs

## Context

Two design docs in `pylcm/` propose **BQSEGM** — a *Branchwise/Casewise Query-Side
Endogenous Grid Method* — as the next solver after the DC-EGM / query-side envelope work
already on `feat/dcegm` (172 commits ahead of the current `feat/type-local-continuation-v`
checkout; EGM machinery in `src/_lcm/egm/`):

- `solver_choice_theory_bqsegm_egm_bruteforce.md` — theory: when brute/EGM/DC-EGM/BQSEGM/
  NEGM/G2EGM/RFC are correct, chunkable, GPU-friendly; a formal BQSEGM spec (§5),
  correctness theorems (§6), implementation-vs-paper ledger.
- `bqsegm_case_piece_api_design.md` — a `@lcm.case_boundary` / `@lcm.piece` decorator API
  exposing institutional piecewise structure (Medicaid eligibility, ACA subsidy brackets,
  premium default, consumption floor) to the solver, plus AST + JAXPR smoothness validators.

BQSEGM's economic point: expose finitely many institutional rules as smooth *pieces* so
that within each case the Euler RHS loses current-state dependence (`u'(c)=Φ_σ(a')` rather
than `Φ(a';x)`), restoring shared-curve EGM amortization and avoiding asset-row replication.

User asked for: (1) a careful implementation plan off `feat/dcegm`; (2) a pass-review of the
two docs for computational issues / math errors / performance traps; (3) a ChatGPT-Pro
review bundle of paper + code plan + my review+plan.

**Key code finding that shapes everything:** the existing query-side envelope
(`src/_lcm/egm/upper_envelope/query.py`, `envelope_at_query`) **already carries a
`segment_id` field, currently all-zero, "for future notch models," with a settled tie
convention** (`_VALUE_TIE_ATOL=1e-12`, right-continuous, larger value-slope wins). BQSEGM is
the consumer that lights up `segment_id`. So the solver is mostly: build per-case smooth DAG
variants, run the *existing* EGM step per case, assign per-case `segment_id`, mask + add
boundary candidates, and reduce with the *existing* multi-segment envelope. Most machinery
already exists.

---

## Part A — Pass review of the two design documents

Severity: **[serious]** breaks a real model or a stated theorem; **[perf]** scaling/compile/
memory trap; **[math]** overclaim/formal gap; **[minor]** notation/clarity.

### A1. [serious] The AST/JAXPR primitive-ban validator rejects every real model — and is inconsistent with pylcm's existing smoothness check
`api §13–14` validates *every function reachable from solver-required outputs* and forbids
`{lt,le,gt,ge,eq,ne,select_n,cond,switch,max,min,clamp,abs,sign,sort,searchsorted}`. But the
solver-required outputs include the candidate value `V_{σ,j}=u+β·C[V_{t+1}](a'_j)` and the
Euler RHS `Φ_σ=β·E[V'_{t+1}(a')]`, both of which **interpolate next period's value/marginal
on a grid** — inherently lowering to `searchsorted` + comparisons + `clamp`/`select_n`. That
machinery is smooth and mandatory, yet trips the forbidden list. Applied to the full reachable
graph, the validator flags every model.
**Corroborated by code:** pylcm already validates DC-EGM smoothness *numerically*
(`src/_lcm/egm/validation.py` — grid + two midpoint-refinement levels; a cliff's quarter-cell
increment fails to shrink like a derivative bound; `_CONTINUITY_SHRINK_FACTOR=0.4`), **not** by
banning primitives. The doc's approach contradicts the working in-repo approach.
**Decision (user):** keep **both** the numeric check and the AST/JAXPR ban as **hard
gates**. The two enabling constraints that make this workable — and which the plan treats
as mandatory, not optional — are: (1) **scope** every gate to the user-authored economic
sub-DAG of each variant (resources/utility/oop/premium/income/tax feeding the Euler),
**excluding** the continuation operator, grid interpolation, and the EGM kernel — without
this scoping the AST/JAXPR ban rejects every model; and (2) a per-function **allowlist**
escape hatch (`@lcm.smooth_helper`, api §18.4 — promoted into v1) so that legitimate
numerical `max/clip/abs` in a user node can be explicitly attested rather than silently
banned. Reuse `src/_lcm/utils/ast_inspection.py` for the AST pass; build the JAXPR pass
greenfield; both raise `BQSEGMCaseError` on violation.

### A2. [perf] Case-combination explosion under JAX static shapes
v1 is "enumerate and mask" (theory §5.4; api §6 forbids satisfiability pruning). Under static
shapes each of `N_C=∏_ℓ K_ℓ` combinations is a separately compiled specialized DAG + EGM solve,
materialized and masked → compile time and device memory scale with the case *product*, per
period per regime. The ACA target lists ~8 institutional features (theory §9.1). The plan
honors the api's own §19 sequence (**v1 = one output, one binary predicate**), caps K hard,
keeps the toy at K=1–2; multi-predicate is explicitly out of v1.

### A3. [math/perf] Theorem 7 (BQSEGM dominates asset-row when N_C<N_X) understates the envelope cost
`N_C·C_E + N_Q·N_seg·C_I < N_X·C_E^row` treats the query-side envelope as lower order. But
(i) `N_seg ≈ N_C × (monotone subsegments/case)`, so the term is really `N_Q·N_C·(segs/case)·C_I`;
with `N_Q ~ N_X` this is `O(N_C·N_X)`, **comparable to** candidate generation; and (ii)
asset-row mode also does a per-node query/interp, so the RHS needs its own interp term for a
fair comparison. The dominance threshold as stated is too optimistic for BQSEGM. The benchmark
must measure envelope cost empirically.

### A4. [math] Theorem 1 (exact case enumeration) is equality with brute only up to the interpolation convention
The conclusion `V_h(x)=max_{y∈F_h}Q_h(x,y)` and the proof step "the generated candidate set for
σ is exactly `F_{σ,h}`" conflate the EGM-generated candidate set (savings grid → endogenous `m`,
published as a PWL interpolant) with the brute discretized feasible set (action grid, exhaustive).
Different discrete sets, different interpolation. True **relative to the declared interpolation/
envelope convention** (the §1.2 definition), but as written it reads as bit-equality with brute,
which it is not. Restate relative to the convention.

### A5. [minor] Symbol/notation collisions in a "publication-quality" doc
- `N_C` = *BQSEGM case count* (§1.2, Thm 7) **and** *brute consumption-grid size* (§7.5 NEGM).
- §7.3 writes `η_B^F, η_B^M, η_E^F, η_E^M`, inconsistent with §2.1's `η_F, η_B`.
- Acronym BQSEGM still expands to "Branchwise…"; keep internal, but every *user-facing* surface
  + public docstrings must use case/piece/segment only (the docs already mandate this).

### A6. [math, coupled pair] Masking (§5.7) and boundary candidates (§5.8) must ship together
A candidate masked out in case σ (recovered `m` violates `p_k=σ_k`) leaves a gap exactly at the
threshold unless the other case's one-sided boundary candidate fills it. With masking alone the
union envelope is wrong precisely at the economically-important notch. Implement boundary
candidates in the same v1 increment as masking; the toy test must assert **threshold-local**
value/policy accuracy (theory §12.4 "switch-local accuracy"), not just global agreement.

### A7. [minor/impl] Parameter-dependent boundary locations are fine under static shapes — but note it
Thresholds like `assets == medicaid_asset_limit` move with params across MSM iterations. A
boundary candidate is one fixed-shape extra row whose `m`-coordinate is param-dependent and
re-sorts each iteration; JAX-compatible, but masking/envelope must not bake in a static sort
order. (The existing `envelope_at_query` already brackets order-independently within a segment —
good fit.)

### A8. [good, reuse] Tie convention must be the existing query-side one
Theory §5.10 demands one tie convention across full-row/query-side/simulation/oracle (DC-EGM's
classic footgun; pylcm hit it — FUES "exact-node crossing repair"). BQSEGM **reuses**
`envelope_at_query`'s convention (`_VALUE_TIE_ATOL=1e-12`, right-continuous, larger value-slope
wins), not a new one.

### A9. [good] Theorems 3/4/5 are correct and well-aimed
Thm 3 (finite samples cannot localize a threshold ⇒ metadata required, not inferable) correctly
motivates the decorators. Thm 4 (chunkable max) and Thm 5 (no continuous interpolant across a
jump) are trivially correct and fine.

### A10. [minor] `inspect.getsource` AST check is best-effort; fail loudly
api §12.3 uses `inspect.getsource`, which fails on lambdas/closures/exec'd functions (api §6
already excludes those). The existing `src/_lcm/utils/ast_inspection.py` already handles this
gracefully (rejects lambdas, catches syntax/inspection errors) — reuse it, and make
un-checkable functions **fail loudly** ("source unavailable") rather than silently pass.

> Net: docs are careful and largely self-aware. The one **blocking** issue is A1 (validator
> scope/approach) — and the in-repo numeric validator is the fix. Real **math** issues: A3, A4
> (both overclaims, both repaired by restating relative to candidate set + interpolation
> convention). Real **perf** trap: A2 (compile/memory ∝ case product). A6 is a correctness
> coupling the implementation must respect.

---

## Part B — Implementation plan (off `feat/dcegm`)

All anchors on `origin/feat/dcegm` (tip `c385d25`). TDD throughout (RED→GREEN→refactor),
prek + ty per commit, rely on CI for the full suite (machine slow). pylcm hands-off rule:
this is a deliberate, large pylcm feature — land it as its own stacked branch with PRs.

### B0. Base & branch
- Branch `feat/bqsegm` off `origin/feat/dcegm` tip. pylcm uses stacked PRs
  (`pylcm-pr-stack-workflow`): put successive increments as commits on the one branch, open one
  PR, cascade if it later stacks. _(confirm with user — open Q1.)_

### B1. Decorators (metadata-only, claw-safe) — `@lcm.case_boundary`, `@lcm.piece`
- New `src/lcm/case_piece.py` (or fold into `src/lcm/transition.py`'s neighborhood). Frozen
  `CaseBoundaryMeta(boundaries: tuple[tuple[str,str],...])` and
  `PieceMeta(output, predicate_name, side: Literal["when","otherwise"])`.
- Pattern: attach `fn.__lcm_case_boundary__` / `fn.__lcm_piece__` and **`return fn`**
  (confirmed claw-safe — beartype respects `__wrapped__`/identity; unlike `@categorical` which
  returns a new class, metadata-only must return the same object). Re-export from `src/lcm/__init__.py`.
- `piece(...)` enforces exactly one of `when=`/`otherwise=` at decoration (api §10).
- TDD: metadata attached; exactly-one-side enforced; predicate `__name__` captured.

### B2. Metadata collection + per-case smooth DAG variants
- Collect case/piece metadata alongside `constraints` during phase normalization
  (`src/_lcm/regime_building/phases.py` → `RegimePhaseSpec`), carried through canonicalize into
  the core processing.
- Build specialized DAGs at the **`_process_regime_core` seam**
  (`src/_lcm/regime_building/processing.py`) using the existing `dags.concatenate_functions`
  producer-swap (remove the combined `oop` producer; substitute `oop_medicaid`/`oop_private`) —
  **no new dags machinery** (the NEGM keeper-identity swap in
  `egm/regime_introspection.py::_concatenate_child_resources` is the existing precedent).
- Reachability per variant via existing `dags.get_ancestors` (already used in
  `regime_building/broadcast.py` for broadcast pruning) — drives which functions are active and
  which to validate (A1 scope).
- Coverage check (exactly one `when` + one `otherwise` per (output,predicate)); `BQSEGMCaseError`.
- TDD: a two-piece model lowers to two DAG variants with correct active-node sets; missing
  `otherwise` raises.

### B3. Smoothness validation (per A1 — BOTH gates hard, both scoped to user economic nodes)
- **Gate 1 (numeric):** extend the existing numeric smoothness checker
  (`src/_lcm/egm/validation.py`, grid+midpoint refinement) to run **per piece** on the user
  economic sub-DAG of each variant.
- **Gate 2 (AST + JAXPR):** AST pass reusing `src/_lcm/utils/ast_inspection.py` (reject Python
  `if`/`ifexp`/`match`; in smooth mode flag undeclared compares/boolops); greenfield JAXPR pass
  (`jax.make_jaxpr` → iterate `eqns` → `eqn.primitive.name`). **Both must pass.**
- **Scoping is mandatory (A1):** compute each variant's user economic sub-DAG via
  `dags.get_ancestors` from the resources/utility/Euler-RHS user targets and **exclude** the
  continuation operator, grid-interpolation, and EGM-kernel nodes before validating. Validate
  only user-authored nodes — never the continuation/interp subgraph (which legitimately uses
  `searchsorted`/compares).
- **Allowlist:** `@lcm.smooth_helper` (promoted into v1) attests a user node whose `max/clip/abs`
  is numerical-not-economic, exempting it from Gate 2.
- New exception `class BQSEGMCaseError(PyLCMError)` in `src/lcm/exceptions.py`.
- TDD: numeric gate catches a hidden cliff inside a piece; AST rejects Python `if`; JAXPR catches
  a hidden-helper `jnp.where` in a user node; an interpolated continuation does **not**
  false-positive (proves scoping works); an allowlisted `jnp.clip` passes; un-inspectable source
  fails loudly.

### B4. BQSEGM solver behind the Solver ABC
- New `BQSEGM` solver subclassing `Solver` (`src/_lcm/solution/contract.py`), registered like
  `DCEGM` (`src/_lcm/solution/solvers.py`), `requires_continuation_carries=True`,
  implementing `build_period_kernels(context) -> SolutionKernels`.
- Per case σ: build the variant pool (B2), reuse the existing one-asset EGM step
  (`egm/step_core.py`, `invert_euler`, `bind_continuation`) to emit candidate rows
  `(m,c,V,μ)`; **assign a distinct `segment_id` per case**; apply consistency masks (set masked
  candidates to the envelope's absent form: value `-inf`, marginal `0`); add one-sided
  **boundary candidates** at each declared `case_boundary` surface (A6 — same increment).
- Concatenate all cases' candidate rows and reduce with the existing **`envelope_at_query`**
  (multi-`segment_id` path already implemented; blocked scan for memory) → publish `EGMCarry`
  + V on the state grid. Tie convention inherited (A8).
- Reuse `kernel_scope.py` to reject out-of-scope configs with precise messages.
- TDD: single-predicate case reduces to standard EGM (one segment_id) bit-for-bit; two-case
  envelope selects the right side across the threshold.

### B5. Toy model + brute oracle tests (TDD, the deliverable)
- `tests/test_models/bqsegm_medicaid_toy.py`: `build_model(variant="bqsegm"|"brute", ...)` +
  `build_params(...)`, mirroring `tests/test_models/ds_app3_discrete_housing.py`. One continuous
  asset (Euler state), one binary Medicaid `case_boundary` on assets that shifts oop/premium.
- `tests/solution/test_bqsegm_medicaid_brute_agreement.py`: solve both, `np.testing.assert_allclose`
  on the interior slice (mirror `test_egm_app3_discrete_housing.py`, `atol≈2e-2, rtol≈5e-3`),
  **plus a threshold-local assertion** (A6) tightly around `medicaid_asset_limit`.
- Memory-vs-asset-row check (BQSEGM peak transient < asset-row for the same toy).
- Negative tests: hidden `where`/`searchsorted` in a user piece rejected; incomplete coverage;
  case boundary returning non-Boolean. Host oracle via `tests/solution/_envelope_oracle.py`.

### B6. Docs + AGENTS.md
- Explanation notebook `docs/explanations/bqsegm.ipynb` (mirror an existing one): the Medicaid
  problem, the case/piece decorators, brute comparison. AGENTS.md solver + decorator entries.

### Verification
- Per increment: `prek run --all-files`, `pixi run ty`. Full pylcm suite serially on CI.
- End-to-end: brute-vs-BQSEGM toy agreement (global + threshold-local, concrete-value asserts);
  validator negative tests; memory check.

---

## Part C — Pro (ChatGPT 5.5 Pro) review bundle
Run the `pro-math-code-review` skill (Mode 1) to bundle: (1) both design docs (paper), (2) this
plan incl. Part B (code plan), (3) Part A (my review). Add `@pro:` markers at: Thm 1 (A4), Thm 7
(A3), the validator scope/approach (A1), the masking↔boundary coupling (A6), and whether the
B4 solver design is faithful to spec §5 and to `envelope_at_query` semantics. Build the upload
package + prompt, hand to the user; ingest the reply into a checklist (Mode 2).
**Sequencing:** pro review of the design+plan **before** writing solver code, so the design is
validated first (the doc is a design, not yet code).

---

## Decisions (resolved with user)
1. **Validator** — BOTH numeric and AST/JAXPR as **hard gates**, both scoped to the user
   economic sub-DAG, with the `@lcm.smooth_helper` allowlist (B3).
2. **v1 scope** — single binary predicate (K=1) Medicaid toy; multi-predicate, lookup-table /
   piecewise-affine decorators, and binding-KKT cases deferred (matches api §19).
3. **Branch & sequencing** — `feat/bqsegm` off `origin/feat/dcegm` tip (`c385d25`); build & hand
   off the **Pro review of this plan first**, ingest the report, then implement B1–B5.
