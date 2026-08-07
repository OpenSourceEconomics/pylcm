# ADR: inference-grade continuous outer choice for `NNBEGM`

**Status:** accepted (design freeze) **Base commit:** `a1d9ca7` on
`feat/nested-nbegm-ez`; work proceeds on `feat/continuous-outer`. **Scope:** this
document freezes the decisions, scope boundaries, and release gates for the
continuous-outer extension *before* any final estimate is observed, so the gates cannot
drift toward whatever numbers later appear.

## Problem

`NNBEGM` selects its outer post-decision margin by a finite grid search, so every
downstream object — simulated moments, their parameter Jacobian, and any standard error
built on it — is piecewise constant in the parameters between grid reassignments. A
finite-difference Jacobian of such a map measures grid artifacts, not derivatives, and
the Mahler–Yum replication's inference layer correctly *refuses* to report on it. The
extension makes the outer choice continuous (solve **and** simulation), then builds the
numerical error budget that decides when a Jacobian, and only then an inference table,
may be released.

## Governing decisions (non-negotiable)

1. **`paper` and `legacy_fortran` are separate configurations.** The canonical
   implementation follows the paper, appendix, equation map, and replication
   specification — not historical bugs. Historical-reproduction switches live in one
   manifest-visible object and can never leak into canonical estimation or standard
   errors.
1. **The first inference implementation uses controlled central finite differences.**
   Automatic/implicit differentiation arrives later as a cross-check and performance
   layer, never as the sole evidence the Jacobian is correct. On AD/FD disagreement
   without a diagnosed central-difference failure, AD is rejected — not the finite
   difference.
1. **The outer optimizer is globally safeguarded.** No global unimodality assumption:
   golden-section refinement runs only inside brackets identified on a global candidate
   mesh; the exact keeper is always evaluated separately; the best and second-best
   candidates are retained so tie margins are observable.
1. **Solve and simulation use the same continuous policy class.** A continuous solve
   followed by grid-restricted simulation is not acceptable; simulation re-runs the same
   interpolant + safeguarded search off-grid.

## Scope boundaries

*First supported solver:* `NNBEGM(inner=NBEGM(...))` only. `NEGM` generalization is out
of scope (candidate-specific endogenous grids need a separate design). *First supported
structure:* one continuous liquid state, one inner consumption/saving action, one scalar
continuous outer action, any already-supported discrete axes, an exact keeper branch,
deterministic or analytically integrated fixed adjustment cost, no EV1 shock on the
outer choice. *Non-goals for the first release:* multidimensional continuous
optimization; globally differentiable policies; valid derivatives at active bounds,
branch ties, or non-unique maxima; AD through simulation indicators; historical Fortran
identity in canonical mode; standard errors merely because a numerical Jacobian has full
rank.

## Array-axis conventions

The candidate bank stacks exact conditional inner solves along a **leading candidate
axis** `C`:

```text
outer_nodes          (C,)
V_arr                (C, *V_shape)          # V state order
carry rows           (C, *row_shape, P)     # regime discrete states (V order),
                                            # passive continuous states,
                                            # discrete actions, then the
                                            # shared trailing grid axis P
```

Carry rows follow the existing `EGMCarry` convention (leading regime discrete states in
V state order, then passive continuous states, then discrete actions, shared trailing
grid axis of static length). The keeper is **not** an entry in the bank: its outer
action is state-dependent, so it stays a separate `KernelResult` and enters only at
collapse.

## Collapse semantics (frozen, load-bearing)

The finite collapse must reproduce the incremental fold it replaces exactly, including
tie-breaking:

- `V = max(V_keeper, max_j W_j)` via `jnp.fmax` — NaN-dead cells never poison a cell
  another candidate solves; a cell stays NaN only when every candidate is infeasible
  there.
- Carry: pointwise winner per row entry with **strict** `>` against the running
  envelope, keeper first, then nodes in grid order. Hence the keeper wins exact ties,
  and an earlier node beats a later one. Any reimplementation that breaks this order
  breaks bit-identity on ties.

## Release gates (frozen before any estimate is observed)

No single threshold is proof of correctness; these are **joint necessary conditions**,
and the reference numbers below may be tightened but never loosened after estimates
exist.

*Solver gates:* zero unresolved outer intervals; zero simulation policy fallbacks; zero
nonfinite reachable values; one further mesh/tolerance refinement moves every moment by
`< 0.01 ×` its empirical SD and the effort policy by `< 1e-4`.

*Jacobian gates:* across accepted step / simulation-size / scramble / solver-tolerance
designs — stable full column rank; every singular value moves `< 10 %`; no column moves
`> 10 %` in moment-SD-normalized norm; non-negligible cells carry finite-difference
signal `≥ 10 ×` their estimated simulation-noise SD; failing columns are marked
unresolved, never zeroed.

*Inference-output gates:* recomputed per accepted design — CP-M SEs, one-step estimates,
CI endpoints, AGS sensitivity; require SE relative range `< 5 %`, one-step and both
CI-endpoint ranges `< 0.05 ×` the reference SE, AGS one-SD row relative-norm range
`< 5 %`. AGS comparisons use `Λ · diag(σ̂_m)` (one-SD moment units), judged **rowwise
per parameter**.

*Local-regularity gates:* refuse ordinary local-normal inference when the estimate is
within five final step widths of a bound; the minimum singular value is numerically
unresolved; material population mass lies within the numerical value-error bound of a
keeper/adjuster tie or non-unique outer maximum; a targeted median has an unresolved
atom; or the one-step estimate leaves the admissible region.

The full error-budget decomposition and its measurement plan live in
[the outer-search error budget](outer_search_error_budget.md).

## Frozen finite baseline

The pre-refactor finite-`NNBEGM` behavior is frozen in
`tests/data/n_nbegm_finite_baseline.npz`, captured at base commit `a1d9ca7` (x64, smooth
two-asset toy, 3 periods, `outer_batch_size ∈ {0, 1, 4}`): every alive period's `V_arr`
and all `EGMCarry` leaves, plus the public `Model.solve` output. The behavior-preserving
candidate-bank refactor must reproduce it to `V` within `1e-12` and carries within
`1e-11` (`tests/solution/test_n_nbegm_finite_baseline.py`).

## Implementation sequence

PR-numbered sequence (contract → candidate bank → interpolant/adaptive mesh → continuous
carry → nested simulation policy → analytic fixed cost → Mahler wiring → random
design/moment engine → parameterization/Jacobian → inference gate → custom JVP → legacy
backend), with the candidate-bank refactor deliberately numerically neutral and gated on
the frozen baseline above before any numerical-method change lands.
