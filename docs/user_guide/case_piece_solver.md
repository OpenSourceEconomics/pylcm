---
title: The case-piece solver (NB-EGM)
---

# The case-piece solver (NB-EGM)

`NBEGM` is an endogenous-grid solver for a one-dimensional consumption–saving regime
whose budget is split by **declared institutional breakpoints** — asset tests, subsidy
brackets, benefit notches, consumption floors. Within each piece of the budget the
problem is smooth, so `NBEGM` runs EGM per piece and merges the pieces on the liquid
grid with a branch-aware upper envelope, resolving the kinks and jumps exactly at their
declared locations instead of blurring them across a finite action grid.

The default solver, `GridSearch`, evaluates the continuous action on a dense grid. It is
exact to that grid, trivially parallel, and remains the right default for most regimes.
Reach for `NBEGM` when a regime carries institutional discontinuities that a brute grid
can only approximate near the cliff, and keep `GridSearch` as the agreement oracle you
validate against.

The method, its correctness results, and the conditions under which it beats brute force
are described in the NB-EGM methods paper; this page is the how-to.

## When it applies

`NBEGM` solves the sub-class where:

- one continuous state — the **liquid** (Euler) state — carries the Euler equation, with
  post-decision savings `savings = coh − consumption ≥ 0`;
- one continuous action (consumption) solves it;
- at most one discrete action enters the period problem;
- every other state (a continuous co-state, discrete type, or stochastic process) *rides
  along* — it enters the budget, utility, and transitions but carries no first-order
  condition of its own.

## Declaring breakpoints

Institutional boundaries are **declared, not discovered** — the solver cannot recover a
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
    """Subsidy into cash-on-hand for the Medicaid-eligible (low-asset) case."""
    return jnp.asarray(subsidy_high)


@lcm.piece("subsidy", otherwise=medicaid_eligible)
def subsidy_private(subsidy_low: float) -> FloatND:
    """Subsidy into cash-on-hand for the private (high-asset) case."""
    return jnp.asarray(subsidy_low)


def coh(liquid: ContinuousState, subsidy: FloatND) -> FloatND:
    """Cash-on-hand: liquid wealth plus the Medicaid-contingent subsidy."""
    return liquid + subsidy
```

The Medicaid-eligible subsidy exceeds the private one, so cash-on-hand — and hence the
value function — jumps down as liquid wealth crosses the limit upward.

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
        "coh": coh,
    },
    constraints={"feasible": feasible},
    solver=NBEGM(savings_grid=LinSpacedGrid(start=0.0, stop=20.0, n_points=100)),
)
```

`NBEGM` requires a `savings_grid` (the post-decision savings nodes). Key optional
arguments:

- `budget_target` (default `"coh"`) — the DAG output the solver treats as cash-on-hand.
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
estimates require re-optimizing under `"one_sided"` from the bridged optimum (or an
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
on that assertion, which must be discharged by independent validation (e.g. brute-force
agreement).

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

## Validating a case-piece regime

Keep the `GridSearch` variant of the same regime as the agreement oracle: solve both on
the same grids and assert concrete-value agreement, with adversarial queries at each
threshold and one ulp on either side. Outside the cliff band the two should agree to
interpolation-error tolerance; inside it, under `"one_sided"`, disagreement collapses to
the brute reference's own finite-grid error. Euler residuals are a useful report but not
the acceptance criterion — they are blind to the corner and boundary candidates that
carry the economics of interest.
