---
title: The NB-EGM solver
---

# The NB-EGM solver

`NBEGM` is an endogenous-grid solver for a one-dimensional consumption–saving regime
whose budget is split by **declared institutional breakpoints** — asset tests, subsidy
brackets, benefit notches, consumption floors. The model author exposes each boundary as
metadata (a *case piece*); within each piece the budget is smooth, so `NBEGM` runs EGM
per piece and merges the pieces on the liquid grid with a branch-aware upper envelope,
resolving the kinks and jumps exactly at their declared locations.

Reach for `NBEGM` when a regime carries institutional discontinuities. A dense brute
grid (`GridSearch`) can only *approximate* the optimum near a cliff — unless you can
specify your grid statically as described in
[`PiecewiseLinSpacedGrid`](grids.md#piecewiselinspacedgrid), it places no candidate
exactly at the threshold and averages across it — so brute force is a diagnostic here,
not the correctness reference (see
[Validating an NB-EGM regime](#validating-an-nb-egm-regime)). For the full theory,
correctness results, and the conditions under which NB-EGM beats brute force, see the
NB-EGM methods paper; this page is the how-to.

## When it applies

`NBEGM` solves the sub-class where:

- one continuous state — the **liquid** (Euler) state — carries the Euler equation, with
  post-decision savings `savings = resources − consumption ≥ 0`;
- one continuous action (consumption) solves it;
- at most one discrete action enters the period problem;
- every other state (a continuous co-state, discrete type, or stochastic process) *rides
  along* — it enters the budget, utility, and transitions but carries no first-order
  condition of its own.

## Declaring breakpoints

Institutional boundaries are **declared, not discovered** — no solver can recover a
threshold's exact location from finitely many black-box evaluations, so the model author
exposes each boundary as metadata. The decorators only attach metadata and return the
function unchanged, so the same model still solves identically under `GridSearch`.

- `lcm.boundary(variable, threshold, *, equality, kind)` declares one equality surface:
  - `equality` — `"when"` or `"otherwise"`: the predicate side that owns the exact
    boundary point. This is part of the feasible-set definition, not a tie-break.
  - `kind` — `"continuous_kink"`, `"jump"`, or `"hard_constraint"`.
  - A bare `(variable, threshold)` tuple is rejected — `equality` and `kind` are
    required.
- `lcm.case_boundary(*boundaries)` marks a Boolean DAG predicate.
- `lcm.piece(output, when=…)` / `lcm.piece(output, otherwise=…)` marks the smooth
  formula for one side of a split output. Every split output must be covered by exactly
  one `when` and one `otherwise` piece; v1 requires exactly one split output per regime.

```python
import jax.numpy as jnp

import lcm
from lcm.typing import BoolND, ContinuousState, FloatND


@lcm.case_boundary(
    lcm.boundary("liquid", "medicaid_asset_limit", equality="otherwise", kind="jump")
)
def medicaid_eligible(liquid: ContinuousState, medicaid_asset_limit: float) -> BoolND:
    """Medicaid asset test: eligible while liquid wealth is below the limit."""
    return liquid < medicaid_asset_limit


@lcm.piece("subsidy", when=medicaid_eligible)
def subsidy_medicaid(subsidy_high: float) -> FloatND:
    """Subsidy into market resources for the Medicaid-eligible (low-asset) case."""
    return jnp.asarray(subsidy_high)


@lcm.piece("subsidy", otherwise=medicaid_eligible)
def subsidy_private(subsidy_low: float) -> FloatND:
    """Subsidy into market resources for the private (high-asset) case."""
    return jnp.asarray(subsidy_low)


def resources(liquid: ContinuousState, subsidy: FloatND) -> FloatND:
    """Market resources: liquid wealth plus the Medicaid-contingent subsidy."""
    return liquid + subsidy
```

The Medicaid-eligible subsidy exceeds the private one, so market resources — and hence
the value function — jump down as liquid wealth crosses the limit upward.

## Selecting the solver

The solver is a per-regime slot. Pass an `NBEGM` instance where you would otherwise
leave the default `GridSearch`:

```python
from lcm import LinSpacedGrid, Regime
from lcm.solvers import NBEGM

alive_regime = Regime(
    transition=next_regime,
    states={"liquid": LinSpacedGrid(start=0.0, stop=20.0, n_points=80)},
    actions={},  # consumption is the solver's continuous action
    state_transitions={"liquid": next_liquid},
    functions={
        "utility": utility,
        "medicaid_eligible": medicaid_eligible,
        "subsidy_medicaid": subsidy_medicaid,
        "subsidy_private": subsidy_private,
        "resources": resources,
    },
    constraints={"feasible": feasible},
    solver=NBEGM(savings_grid=LinSpacedGrid(start=0.0, stop=20.0, n_points=100)),
)
```

`NBEGM` requires a `savings_grid` (the post-decision savings nodes). Key optional
arguments:

- `budget_target` (default `"resources"`) — the DAG output the solver inverts against
  (the consumption budget), the same node `DCEGM` names via `resources=`.
- `continuous_state` / `post_decision_function` — name the ride-along co-state and its
  off-budget liquid law when the regime carries one.
- `jump_read` — the cliff-read mode (below).
- `probe_failure` — `"reject"` (default) or `"assume_declared"` (below).
- The batch-size knobs (below).

## The two cliff-read modes

A child regime's value cliffs cannot be represented by a single continuous interpolant.
`jump_read` selects how the continuation carry is published to parents:

- `"one_sided"` (default, exact). Each carry row holds every jump preimage as a
  duplicated abscissa carrying the exact one-sided value and marginal limits, plus the
  jump locations. Queries strictly below a jump interpolate toward the left value;
  queries at or above use the right value. This is the exact-convention mode.
- `"bridged"` (fast, approximate). Plain liquid-grid rows; the parent's interpolation
  may average across a cliff, exactly like any finite-grid solver. Cheaper, and the
  intended mode for consumers that tolerate finite-grid cliff error, such as estimation
  inner loops.

```{warning}
The bridged and one-sided solves define **different objective surfaces** near
institutional cliffs. Use `"bridged"` as a warm-start / screening mode only: final
estimates should re-optimize under `"one_sided"` from the bridged optimum (or an
explicit objective-surface comparison showing the two minimizers coincide within
the reported precision). Evaluating the one-sided objective once at the bridged
optimum does not detect the difference.
```

## The smoothness gate

Declared breakpoints are only as good as the smoothness of what lies between them. At
model build, `NBEGM.validate` runs two validators over the user economic functions
reachable in each case:

- an **AST gate** rejecting Python branching and hidden comparisons in smooth pieces
  (boundary predicates may compare — that is their job);
- a **JAXPR gate** tracing each smooth piece and rejecting piecewise primitives
  (`select_n`, comparisons, …) hidden inside called helpers the AST cannot see.

Mark a reviewed numerical `clip`/`max`/`abs` guard with `@lcm.smooth_helper` to exempt
it, stating the domain on which it is smooth.

## The piecewise-constancy probe

When the continuation reads the *current* liquid state (a co-state's next-state law or a
regime-transition probability switched at a declared threshold), `NBEGM` solves one
continuation row per declared interval. This is exact only if that liquid dependence is
piecewise-constant on the declared partition. A build-time probe screens for it and
**refuses by default** (`probe_failure="reject"`) when it detects smooth dependence or
cannot evaluate the model's DAG. Passing `probe_failure="assume_declared"` asserts the
precondition explicitly (emitting a warning); every exactness claim is then conditional
on that assertion, which must be discharged by independent validation.

The probe is a finite diagnostic, not a certificate — a dependence whose derivative
vanishes at every probed point passes undetected.

## Performance knobs

Every batching knob streams one axis and changes peak memory and schedule only, never
the result (up to floating-point reassociation):

| Knob                          | Axis streamed                                            |
| ----------------------------- | -------------------------------------------------------- |
| `cell_block_size`             | ride-along cells                                         |
| `branch_batch_size`           | discrete-action branches (`lax.map`, body compiled once) |
| `interval_batch_size`         | per-interval continuation reads                          |
| `stochastic_node_batch_size`  | child stochastic-node mesh                               |
| `envelope_segment_block_size` | envelope segment blocks (two-pass scan)                  |

All default to `0` (the whole axis in one vectorized pass). Raise a knob when the
corresponding buffer is the memory wall.

## Relation to other EGM methods

NB-EGM sits alongside pylcm's other endogenous-grid solvers rather than replacing them
(see [Choosing a solver](choosing_a_solver.md) for the full map):

- **`DCEGM`** is the discrete–continuous EGM of Iskhakov, Jørgensen, Rust & Schjerning
  (2017): a discrete choice (work vs. retire) makes the value function non-concave, and
  DC-EGM resolves the resulting *secondary kinks* with an upper-envelope scan. It is
  pylcm's direct reproduction of that method (see the
  [IJRS example](../examples/iskhakov_et_al_2017.md)) and the right tool for a
  discrete–continuous problem with no declared institutional breakpoints.
- **`NEGM`** nests a 1-D `DCEGM` consumption solve inside an outer deterministic search
  over a durable/illiquid margin (Druedahl 2021).
- **`NBEGM`** keeps that discrete–continuous capability without nesting `DCEGM`: it
  solves one continuous subproblem per discrete-action value and merges them with its
  own discrete upper envelope, and it splits the Euler path at folds to resolve the same
  secondary kinks a kinked continuation induces. So an IJRS-style retirement kink is
  within its reach. What NB-EGM *adds* is the exact treatment of **declared
  institutional breakpoints** — which DC-EGM's black-box envelope cannot locate exactly
  — and a query-side (map-reduce) upper envelope that parallelises better on a GPU than
  a topology-discovering scan.

## Validating an NB-EGM regime

`GridSearch` is a **diagnostic, not the correctness oracle**. Brute force evaluates the
combined (`jnp.where`) budget on a finite action grid, so it *smooths across every
breakpoint*. The optimal policy is often to save to just *below* an eligibility cliff
(to keep the benefit), so the optimum sits one step inside the eligible side — a point
no finite grid holds *exactly*, though a
[`PiecewiseLinSpacedGrid`](grids.md#piecewiselinspacedgrid) aligned to the breakpoint
may come close enough for practical purposes. Asserting *exact* agreement with brute is
therefore the wrong acceptance test.

- **Correctness oracle (selection).** A host-side reimplementation of NB-EGM's own
  convention — the same per-case EGM, candidate set, masking, endpoint ownership, and
  upper envelope, evaluated on the same grids — is the exact reference for the envelope
  and selection logic.
- **Brute agreement (diagnostic).** Solve the `GridSearch` variant of the same regime on
  a dense grid and compare, scoring two regions separately: *outside* the cliff band the
  two should agree to interpolation-error tolerance; *inside* the cliff band (cells
  adjacent to a jump preimage) the disagreement is expected — the brute reference has
  its own finite-grid error there, and under `"one_sided"` the gap collapses to that
  error.

Euler residuals are a useful report but not the acceptance criterion — they are blind to
the corner and boundary candidates that carry the economics of interest.
