# PLAN: Implement BQSEGM (case-piece EGM) off `feat/dcegm`

## Status / provenance

- Design docs (v2, in this dir) — `solver_choice_theory_bqsegm_egm_bruteforce.md` and
  `bqsegm_case_piece_api_design.md` — **rewritten** after a ChatGPT-5.5-Pro adversarial
  audit. The Pro re-review verdict is **`serious_gap`**, but the remaining gaps were in
  **this plan's Part B4**, not the docs. The docs now fix every theory/API issue from the
  first pass (Thm 1 convention-relative; Thm 7 full cost inequality; NaN-dead masking;
  endpoint ownership; validator scope; topology-preserving publication).
- This revision of the plan resolves the five Pro findings F1–F5 (traceability table at the
  end) and corrects the earlier over-optimistic "machinery already exists" framing.

## Context

**BQSEGM** = *Branchwise/Casewise Query-Side EGM*: the next solver after the DC-EGM /
query-side envelope work on `feat/dcegm` (172 commits ahead of the current checkout; EGM
machinery in `src/_lcm/egm/`). Economic point: expose finitely many institutional rules as
smooth **pieces** so that within each case the Euler RHS loses current-state dependence
(`u'(c)=Φ_σ(a')` not `Φ(a';x)`), restoring shared-curve EGM amortization and avoiding
asset-row replication.

**Corrected "what already exists" assessment (Pro F1–F3).** The query-side envelope
(`upper_envelope/query.py:envelope_at_query`) carries a `segment_id` field, but it is a
**closed-segment** primitive: it links adjacent live candidates with matching ids using
**closed brackets on both endpoints** and has **no endpoint-ownership flags**, and it treats
only NaN rows as dead (a finite abscissa with `value=-inf` is still a live link). `EGMCarry`
(`egm/carry.py`) holds only `endog_grid/value/marginal_utility/taste_shock_scale` — **no
segment ids, endpoint flags, or switch metadata**; `ContinuationPayload = EGMCarry`
(`solution/contract.py`). So BQSEGM needs **three real extensions**, not just "light up
`segment_id`":
1. per-monotone-subsegment segmentation with open/closed endpoint flags (not one id per case);
2. an **endpoint-aware** query envelope (extend `envelope_at_query` or pre-process boundary
   queries so an open endpoint cannot win at an excluded equality point);
3. a **topology-preserving** continuation payload (or a switch-refined aggregate grid proven
   convention-exact) — plain `EGMCarry` bridges switches (Thm 6).

Reusable as-is: the Solver ABC seam (`solution/contract.py`, `build_period_kernels`); the
per-case EGM step (`egm/step_core.py`, `invert_euler`, `bind_continuation`); the DAG
producer-swap via `dags.concatenate_functions` (precedent:
`regime_introspection._concatenate_child_resources`); `dags.get_ancestors` for per-variant
reachability; the brute-vs-EGM toy/oracle test harness
(`tests/solution/test_egm_app3_discrete_housing.py`, `tests/solution/_envelope_oracle.py`).

---

## Part A — Original pass-review of the docs (now folded into docs v2)

The first-pass review findings have been **incorporated into the rewritten docs**; kept here
only for traceability:
- **A1** (validator bans mandatory continuation-interp primitives) → docs v2 scope the
  forbidden-primitive check to *user-authored economic nodes only* (theory §5.5, api §11).
- **A3** (Thm 7 understates envelope cost) → Thm 7 is now a full cost inequality
  `T_BQ < T_row`; `N_C<N_X` is only a heuristic corollary.
- **A4** (Thm 1 overclaims bit-equality with brute) → Thm 1 is now relative to a declared
  convention `𝒦_h` over candidate records `𝒮_h`.
- **A6** (masking ⇄ boundary candidates are a coupled pair) → theory §5.7–5.8 now couple
  NaN-dead masking with side-aware boundary candidates + endpoint ownership.
- **A2** (case-combination explosion ∝ ∏Kᵢ under static shapes), **A5** (symbol collisions),
  **A7** (param-dependent boundary loc), **A8** (reuse the one tie convention), **A10**
  (fail loudly on un-inspectable source) — still hold and are reflected below.

---

## Part B — Implementation plan (off `feat/dcegm`)

All anchors on `origin/feat/dcegm` (tip `c385d25`). TDD throughout; prek + ty per commit;
rely on CI for the full suite (machine slow). Land as its own stacked branch with PRs.

### B0. Base & branch
Branch `feat/bqsegm` off `origin/feat/dcegm` tip. Stacked-PR workflow: increments as commits
on the one branch; cascade if it later stacks.

### B1. Decorators + boundary metadata (metadata-only, claw-safe)
- New `src/lcm/case_piece.py`. Frozen dataclasses **matching docs v2 §10**:
  - `BoundarySurface(variable, threshold, equality_owner: Literal["when","otherwise"],
    kind: Literal["continuous_kink","jump","hard_constraint"])`
  - `CaseBoundaryMeta(boundaries: tuple[BoundarySurface, ...])`
  - `PieceMeta(output, predicate_name, side)`
- `boundary(variable, threshold, *, equality, kind) -> BoundarySurface` helper;
  `case_boundary(*boundaries)` coerces each arg (a two-string tuple only when equality
  ownership can be inferred; otherwise require the explicit `boundary(...)` form);
  `piece(output, *, when=None, otherwise=None)` enforces exactly one side.
- Claw-safe: attach `fn.__lcm_case_boundary__` / `fn.__lcm_piece__` and **`return fn`**
  (same object — confirmed safe vs `@categorical` which returns a new class). Re-export
  `case_boundary`, `piece`, `boundary` from `src/lcm/__init__.py`.
- TDD: metadata attached; `equality`/`kind` captured; exactly-one-side enforced; a bare
  two-string tuple with non-inferable ownership raises asking for `boundary(...)`.

### B2. Metadata collection + per-case smooth DAG variants
- Collect case/piece/boundary metadata alongside `constraints` during phase normalization
  (`regime_building/phases.py` → `RegimePhaseSpec`), carried through to core processing.
- Build specialized DAGs at the **`_process_regime_core` seam** (`regime_building/processing.py`)
  via existing `dags.concatenate_functions` producer-swap (swap the combined `oop` producer
  for `oop_medicaid`/`oop_private`) — **no new dags machinery**.
- Per-variant active-node set via `dags.get_ancestors` (drives B3 validation scope).
- Coverage check (exactly one `when` + one `otherwise` per (output,predicate)) →
  `BQSEGMCaseError`.
- TDD: a two-piece model lowers to two variants with correct active-node sets; missing
  `otherwise` raises.

### B3. Smoothness validation — user-economic-node scope (Pro F4-corrected)
Per docs v2 §11 three checks; **both AST and JAXPR are the hard gate; numeric is a
complementary check, not a complete detector**:
- **Gate (hard): AST + JAXPR on user economic nodes.** AST pass reusing
  `src/_lcm/utils/ast_inspection.py` (reject Python `if`/`ifexp`/`match`; flag undeclared
  compares/boolops in smooth mode); greenfield JAXPR pass (`jax.make_jaxpr` → walk `eqns`,
  **including nested jaxprs** for `cond`/`scan`/`pjit` → `eqn.primitive.name`).
- **Mandatory scoping (Pro A1):** validate only the **user-authored economic sub-DAG** of
  each variant — computed via `dags.get_ancestors` from the resources/utility/Euler-RHS user
  targets, **excluding** the continuation operator, grid interpolation, and EGM-kernel nodes
  (those legitimately use `searchsorted`/compares). Without this scoping the ban rejects
  every model.
- **Numeric check (diagnostic, Pro F4):** the existing `egm/validation.py` grid+midpoint
  check is a **narrow node-resolution diagnostic** — it only fires on savings-stage functions
  that read the Euler state at node resolution and **skips functions whose needed args are
  free build-time parameters**. Run it as a complementary hard check **where a node is
  build-time-evaluable**; it does **not** by itself establish BQSEGM piece smoothness. The
  plan must not imply it is already a complete cliff detector.
- **Allowlist:** `@lcm.smooth_helper` (promoted into v1) attests a user node whose
  `max/clip/abs` is numerical-not-economic, exempting it from the AST/JAXPR gate.
- New `class BQSEGMCaseError(PyLCMError)` in `src/lcm/exceptions.py`.
- TDD: AST rejects Python `if` in a piece; JAXPR catches a hidden-helper `jnp.where` (incl.
  one nested in a `cond`); an interpolated continuation does **not** false-positive (proves
  scoping); an allowlisted `jnp.clip` passes; un-inspectable source fails loudly; the numeric
  diagnostic catches a build-time-evaluable cliff but is documented as not exhaustive.

### B4. BQSEGM solver — candidate generation + NaN-dead masking + boundary candidates (Pro F1, A6)
- New `BQSEGM` solver subclassing `Solver` (`solution/contract.py`), registered like `DCEGM`
  (`solution/solvers.py`), `requires_continuation_carries=True`, implementing
  `build_period_kernels(context) -> SolutionKernels`.
- Per case σ: build the variant pool (B2); reuse the existing EGM step (`egm/step_core.py`,
  `invert_euler`, `bind_continuation`) to emit candidate rows `(m,c,V,μ)`.
- **Consistency masks are NaN-dead (Pro F1, theory §5.7):** an invalid candidate sets **all
  of `endog_grid, value, policy, marginal` to NaN** before segment formation. Do **not** use
  `value=-inf, marginal=0` pre-envelope — `query.py` treats a finite abscissa with `-inf`
  value as a live link and can emit NaN via `0*-inf`. Reserve `-inf`/`0` for the *published*
  infeasible-choice rows only (post-envelope publication boundary).
- **Boundary candidates with side labels + endpoint ownership (theory §5.8, A6):** at each
  declared `BoundarySurface`, generate one-sided candidate(s) carrying the `equality_owner`;
  only the equality-owning side is eligible at the exact boundary query, the other side is
  **open** there. Same v1 increment as masking (they are coupled).
- Reuse `kernel_scope.py` to reject out-of-scope configs.

### B4a. Segment construction — per monotone feasible subsegment (Pro F2, theory §5.9)
- Split each case's candidate path into **monotone, hole-free feasible subsegments**: split
  on endogenous-grid folds and on interior NaN-dead holes from masking; assign a `segment_id`
  **per subsegment** (not per case), and carry `e_L, e_R ∈ {open, closed}` endpoint flags
  derived from boundary ownership.
- One id per case is allowed only after proving the case yields exactly one monotone hole-free
  segment (assert it for the K=1 toy; do not assume it in general).
- Static-shape JAX: fixed max-subsegment count with NaN padding; provenance recorded in B4.
- TDD: a folded case splits into ≥2 ids; a masked interior hole splits the path; endpoint
  flags propagate from `equality_owner`.

### B4b. Endpoint-aware query envelope (Pro F2, theory §5.10)
- Extend `envelope_at_query` with **endpoint-closure flags** (`e_L,e_R`) so eligibility is
  `l_s < q < r_s`, with equality at an endpoint allowed **only if that endpoint is closed** —
  OR pre-process boundary queries so an open endpoint cannot win at an excluded equality
  point. Preserve the existing tie convention (`_VALUE_TIE_ATOL=1e-12`, right-continuous,
  larger value-slope) consistently across query-side/full-row/sim/oracle (A8).
- **RED test = the Pro-supplied RT1** (`test_bqsegm_envelope_endpoint_contract.py`):
  `test_plan_closed_endpoint_can_select_excluded_side` (an excluded one-sided segment must
  NOT win at the boundary) and `test_plan_minus_inf_dead_candidate_is_not_absent` (proves the
  `-inf` hazard → motivates NaN-dead). Both currently pass against the *closed* primitive,
  demonstrating the gap; re-express them as the spec the endpoint-aware envelope must satisfy
  (excluded side must lose at equality).

### B4c. Topology-preserving publication (Pro F3, theory §5.11, Thm 6)
- Plain `EGMCarry` is insufficient (bridges switches). Choose **one** (recommend the payload
  for exactness):
  - **Topology-preserving payload:** widen `ContinuationPayload` beyond the `= EGMCarry`
    alias (`solution/contract.py`) to carry segment ids, endpoint flags, switch/boundary
    flags, and any top-two records for one-sided Euler reads; parent continuation evaluates
    with the same endpoint-aware segment convention (B4b).
  - **Switch-refined aggregate grid:** publish an ordinary grid only after inserting every
    switch/cliff/boundary node, **with a test proving aggregate interpolation == segment-aware
    interpolation** under the convention.
- TDD: a two-case continuation read near the switch matches the segment-aware value (no
  bridging); if the payload route, the parent Euler read recovers the correct one-sided
  marginal.

### B5. Toy model + brute oracle tests (TDD, the deliverable)
- `tests/test_models/bqsegm_medicaid_toy.py`: `build_model(variant="bqsegm"|"brute", ...)` +
  `build_params(...)`, mirroring `ds_app3_discrete_housing.py`. One continuous asset (Euler
  state), one binary Medicaid `case_boundary` on assets shifting oop/premium.
- `tests/solution/test_bqsegm_medicaid_brute_agreement.py`: solve both,
  `np.testing.assert_allclose` on the interior (`atol≈2e-2, rtol≈5e-3`), **plus a
  threshold-local assertion** tightly around `medicaid_asset_limit` (A6, §12.4 switch-local).
- Include **RT1** (B4b) and the publication exactness test (B4c) as standing regressions.
- Memory-vs-asset-row check (BQSEGM peak transient < asset-row for the same toy).
- Negative tests: hidden `where`/`searchsorted` in a user piece rejected; incomplete
  coverage; case boundary returning non-Boolean. Host oracle via `_envelope_oracle.py`.

### B6. Docs + AGENTS.md
- `docs/explanations/bqsegm.ipynb` (mirror an existing notebook); AGENTS.md solver + decorator
  + `boundary(...)` entries.

### Acceptance criteria / proof obligations (Pro F5)
Implementation is "done" only when these convention-exactness obligations are tests, green:
1. **Endpoint ownership:** excluded one-sided segment loses at the exact boundary (RT1).
2. **NaN-dead masking:** no `-inf`-induced NaN; masked candidates are absent pre-envelope.
3. **Segment topology:** folds/holes produce >1 `segment_id`; one-id-per-case only where
   proven single-monotone.
4. **Publication convention-exactness:** chosen payload/grid reproduces segment-aware
   continuation (no switch bridging) — Thm 6.
5. **Validator scope:** trusted continuation/interp never false-positives; user-node cliffs
   always caught (AST+JAXPR).

### Verification
Per increment: `prek run --all-files`, `pixi run ty`. Full suite serially on CI. End-to-end:
brute-vs-BQSEGM toy agreement (global + threshold-local); the five acceptance tests; memory
check.

---

## Part C — Pro review (DONE)
Bundle built + audited (verdict `serious_gap`, gaps in this plan now fixed in Part B above).
Pro re-review of *this* updated plan is optional before coding; otherwise proceed to B1.

## Decisions (resolved with user)
1. **Validator** — AST + JAXPR on user economic nodes are the hard gate (scoped, allowlisted);
   numeric grid+midpoint is a complementary diagnostic where build-time-evaluable (Pro F4).
2. **v1 scope** — single binary predicate (K=1) Medicaid toy; multi-predicate, lookup-table /
   piecewise-affine decorators, binding-KKT cases deferred (api §19).
3. **Branch & sequencing** — `feat/bqsegm` off `origin/feat/dcegm` tip (`c385d25`).

## Pro-finding → plan traceability
- **F1** (NaN-dead vs `-inf` pre-envelope) → **B4** masking rule rewritten.
- **F2** (segment_id per subsegment + endpoint ownership) → **B4a** + **B4b** (new steps;
  envelope extended).
- **F3** (topology-preserving publication, not plain EGMCarry) → **B4c** (new step; widen
  `ContinuationPayload`).
- **F4** (numeric validator overstated) → **B3** reframed: AST/JAXPR is the gate, numeric is a
  narrow diagnostic.
- **F5** (proof obligations) → **Acceptance criteria** section (5 standing tests).
