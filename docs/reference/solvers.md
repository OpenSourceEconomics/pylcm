---
title: Solvers and capabilities
---

# Solvers and capabilities

All solver-related names live under `lcm.solvers`:

```python
from lcm.solvers import DCEGM, EGM, GridSearch, NBEGM, NEGM, NNBEGM
```

A solver object contains numerical configuration. Specialized economic roles are
declared on the regime.

## Regime pairing

| Regime declaration               | Accepted solver family                                     |
| -------------------------------- | ---------------------------------------------------------- |
| `Regime`                         | `GridSearch` or another non-margin `Solver`                |
| `ConsumptionSavingsRegime`       | `GridSearch` or `OneMarginSolver`: `EGM`, `DCEGM`, `NBEGM` |
| `NestedConsumptionSavingsRegime` | `GridSearch` or `TwoMarginSolver`: `NEGM`, `NNBEGM`        |

Collective regimes and transition-local `JointTransition` lotteries currently require
`GridSearch`. EV1 taste shocks are supported by `GridSearch` and `DCEGM`; `NEGM`,
`NBEGM`, and `NNBEGM` reject them. Model construction validates the concrete solver's
remaining prerequisites.

## Capability table

| Solver       | Required declaration                                           | Problem shape                                                                      | Hard prerequisites and supported constraints                                                                                                                                                                                                                                                          | Main tradeoff                                                                                    |
| ------------ | -------------------------------------------------------------- | ---------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------ |
| `GridSearch` | `Regime` or a specialized regime                               | General discrete-continuous action product                                         | Ordinary callable constraints; no EGM structure required; supports EV1 taste shocks and transition-local joint lotteries                                                                                                                                                                                                                                              | Broadest representation; work covers the full action product; eligible singleton/collective hard-max and singleton EV1 JIT solve-value routes evaluate bounded action blocks |
| `EGM`        | `ConsumptionSavingsRegime` with one `LiquidMargin`             | Smooth, concave one-state/one-action cash-on-hand problem                          | Exactly one continuous state and action; no discrete/process states or actions; resources equal the liquid state; post-decision state equals state minus action; utility does not read the liquid state; default Koopmans aggregator; only a provable post-decision lower bound as a solve constraint | Narrowest contract and no upper envelope                                                         |
| `DCEGM`      | `ConsumptionSavingsRegime` with one `LiquidMargin`             | One liquid Euler margin with a genuine resources node and optional discrete choice | Valid liquid resources and post-decision roles; declared lower bound; solver-supported discrete/passive dimensions and continuation layout; supports EV1 taste shocks                                                                                                                                                            | Adds constrained candidates and an upper envelope; simulation may re-optimize on the action grid |
| `NBEGM`      | `ConsumptionSavingsRegime` with one `LiquidMargin`             | Supported declared kinks, jumps, hard boundaries, or smooth discrete branches      | Supported case-piece or piecewise-affine declaration; solver-proven constraint routes; no EV1 taste shocks                                                                                                                                                                                            | Preserves declared topology; structural probes and candidate geometry add cost                   |
| `NEGM`       | `NestedConsumptionSavingsRegime` with liquid and outer margins | A `DCEGM` inner solve conditional on a finite outer grid                           | Full inner `DCEGM` contract plus outer state, action, post-decision, no-adjustment, and cost roles; no EV1 taste shocks                                                                                                                                                                                                    | Exact relative to the outer candidate set; candidate retention can dominate memory               |
| `NNBEGM`     | `NestedConsumptionSavingsRegime` with liquid and outer margins | An `NBEGM` inner solve inside a finite or adaptive outer search                    | Full inner `NBEGM` contract plus a compatible outer search and branch aggregator; no EV1 taste shocks                                                                                                                                                                                                                      | Most expressive EGM route and the highest structural/computational burden                        |

## Nonlinear certainty equivalents

| Solver                 | Nonlinear certainty-equivalent support                                                                                                                                                           |
| ---------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `GridSearch`           | Any supported certainty-equivalent callable; values are aggregated directly on the action grid                                                                                                   |
| `EGM`, `DCEGM`, `NEGM` | None; their Euler inversions assume `LinearExpectation()`                                                                                                                                        |
| `NBEGM`, `NNBEGM`      | `PowerMean()` paired with `CESAggregator()`, only on NBEGM ride-along routes that pass the remaining structural gates; current-period jumps and liquid-dependent continuation reads are rejected |

`NBEGM` and `NNBEGM` implement the recursive Euler equation through the NBEGM inner
kernel. Here, a ride-along route means that the continuation varies over at least one
supported non-liquid state; the single-liquid NBEGM route remains additive and rejects a
nonlinear certainty equivalent. This capability does not waive their budget,
state-layout, taste-shock, or smoothness restrictions. See
[Preference aggregation and certainty equivalents](../methods/preferences.md) before
choosing the solver.

A valid declaration is part of the model, not a hint. Start with
[Authoring for EGM-family solvers](../user_guide/authoring_specialized_solvers.md).

## Constructors

(api-grid-search)=
### `GridSearch`

```python
GridSearch(action_block_width=None)
```

Covers the complete represented state-action product and applies constraints directly.
It is the broadest route and the default solver on `Regime`. Eligible JIT solve-value
routes evaluate bounded C-order action blocks while preserving the complete support.

`action_block_width` fixes the flattened action-product width of every eligible streamed
solve core. It must be an exact positive integer no larger than that product. A fixed
width is rejected when the route is deliberately dense, unsupported, or has no
nontrivial action product; it is an execution request, not a hint that can be ignored.
With no fixed width and no device-memory budget, pylcm streams every eligible core at
its bootstrap width: the largest power of two below the action product, capped at 64.
With an [`ExecutionConfig`](runtime_and_results.md#compiler-workspace-budgets), the
planner instead walks a deterministic width frontier widest-first and dispatches the
first candidate whose compiler-reported peak fits. Supplying both makes the fixed width
the only candidate, which must fit the budget.

This matrix uses exactly three disposition labels:

(gridsearch-jit-route-matrix)=
#### JIT solve route matrix

| Route shape | Disposition | Meaning |
| --- | --- | --- |
| Singleton hard max | streamed | Action blocks feed the hard-max reduction. |
| Collective hard max | streamed | Action blocks feed the collective scalarization, hard max, stakeholder readout, and dissolution-flag reduction. |
| Singleton EV1 with at least one discrete action | streamed | Each block reduces its continuous-action axes; the discrete expected maximum combines the resulting complete discrete support. |
| Same-period references, edge references, or gated targets without a co-mapped state | streamed | Each target artifact and its exact source argument path is declared; the resolved transfer supplies it as a dynamic input to the streamed solve core. |
| Ordinary co-mapped state route without a separate reference channel | streamed | Continuation leaves co-map with the state cell while actions stream. |
| Singleton hard max with folded states, including an ordinary co-map | streamed | Actions stream at each fold node; the unchanged quadrature still evaluates and reduces the full fold-node axis. |
| Co-mapped state plus a separate same-period or edge-reference channel | deliberately dense | The two data transports are not yet represented together by the streamed program. |
| No action, or an action product containing at most one candidate | deliberately dense | Blocking a trivial product adds no useful execution choice. |
| JIT disabled, including the raw execution path | deliberately dense | The action-streamed program is a JIT lowering route. |
| Collective EV1 | unsupported | The action-streamed solve program has no collective EV1 reduction. |
| EV1 plus a folded state | unsupported | The action-streamed solve program does not compose EV1 and fold reductions. |
| Collective hard max plus a folded state | unsupported | The action-streamed solve program does not compose collective and fold reductions. |
| EV1 without a discrete action | unsupported | The GridSearch EV1 reduction requires a discrete-choice axis. |
| Any simulation-policy construction | deliberately dense | Simulation lies outside this solve-route classifier and continues to recompute policies on the dense action product. |

The streamed rows assume JIT execution, a nontrivial action product, and no applicable
unsupported composition. When several rows describe one route, an unsupported
composition takes precedence, followed by a deliberately dense condition; only the
remaining eligible route is streamed.

Here, **unsupported** is a disposition of the action-streamed solve program, not a new
public solver capability verdict. Model validation may reject the corresponding
declaration; where the declaration is otherwise valid, the existing dense GridSearch
core remains the compatibility path. Streamed programs publish only solve-time `VALUE`,
or `VALUE` plus `DISSOLUTION_FLAG` for a collective route; replay and policy artifacts
are not integrated. Runtime and peak-memory effects require measurement; bounded action
evaluation alone is not a performance claim.

For planned GridSearch solve cores, a continuation or reference declaration identifies
both the stored target artifact and the exact channel and tree path at which the source
core consumes it. Planning resolves each read either as a local pass-through that keeps
the value's stored partitioning on the shared source mesh, or as an explicit copy into a
supported layout on the source core's mesh. The resolved plan is applied identically to
lowering and runtime arguments, and unsupported layout conversions or mismatched array
metadata are errors. Each declaration's `(source_regime, source_period, core_key)` must
also match the actual compiled core before its channel and argument-tree path are
resolved. Remaining-consumer counts are committed only after successful dispatch. A
zero count records eligibility for future memory
planning; it does not release, donate, or offload an array. Dense programs without
declared value reads and other unplanned consumers remain pinned.

With EV1 taste shocks, GridSearch first maximizes over the continuous-action axes within
each discrete-action combination and then applies the discrete log-sum. Simulation uses
the corresponding Gumbel-max choice rule. See
[`ExtremeValueTasteShocks`](model_and_regime.md#api-extreme-value-taste-shocks) for the complete
feature boundary.

(api-egm)=
### `EGM`

```python
EGM(savings_grid=...)
```

Plain one-margin EGM. It validates the exact cash-on-hand identity summarized in the
capability table and takes no upper envelope.

(api-dcegm)=
### `DCEGM`

```python
from lcm.solvers import DCEGM, LTMEnvelope

solver = DCEGM(
    savings_grid=...,
    envelope=LTMEnvelope(),
    refined_grid_factor=2.0,
    n_constrained_points=20,
    stochastic_node_batch_size=0,
)
```

`DCEGM` does not require a nontrivial discrete choice: it is also the supported route
for a smooth liquid problem whose genuine resources node, passive states, or stochastic
processes make plain `EGM` ineligible.

The supported typed envelope configurations are `ExactEnvelope`, `FUESEnvelope`,
`RFCEnvelope`, `LTMEnvelope`, and `MSSEnvelope`. String selectors are not part of the
public API. See [Upper envelopes](envelopes.md) for their distinct contracts.

`refined_grid_factor` provides NaN-padded storage headroom for ownership changes in each
envelope row. A row that needs more slots is reported as overflow and NaN-poisoned; this
field does not change the density of the policy read-out grid. `n_constrained_points`
controls the borrowing-corner segment, and `stochastic_node_batch_size` the
stochastic-node workspace.

:::{important} Solved and simulated continuous actions
A solve can expose an off-grid
DCEGM policy as an addressed replay artifact in `SolutionResult`. No envelope shipped
with pylcm currently passes the conservative off-grid policy-read gate, so ordinary
simulation recomputes the action argmax on the regime's declared action grid. Simulation
uses that gridded argmax whenever the model declares no off-grid policy-read route.

The simulated continuous action can therefore differ from the off-grid solve policy.
With taste shocks, simulated choice frequencies follow the grid-restricted
choice-specific values rather than necessarily matching the solve's off-grid choice
probabilities. The intrinsic budget is still applied as a simulation feasibility mask.
:::

(api-negm)=
### `NEGM`

```python
NEGM(inner=..., outer_grid=..., outer_batch_size=0)
```

Runs the bound `DCEGM` inner solve for every finite outer-grid node and includes the
keeper. `outer_batch_size` limits how many candidate values are evaluated at once. It
can reduce temporary evaluation memory, but it does not in general cap the size of the
candidate bank retained for later envelope or ordered-fold operations. Peak memory can
therefore continue to grow with the full candidate set. Measure both temporary and
retained arrays for the exact model and solver profile.

`NEGM` rejects EV1 taste shocks. Its outer durable-margin maximum currently wraps the
inner DCEGM solve, but a taste-shocked discrete choice must be the outermost aggregation:
`max_outer logsumexp_discrete` is not `logsumexp_discrete max_outer`. Use `GridSearch`,
or remove the taste shocks when the NEGM structure is required.

(api-nbegm)=
### `NBEGM`

```python
NBEGM(
    savings_grid=...,
    jump_read="one_sided",
    stochastic_node_batch_size=0,
    envelope_segment_block_size=0,
    envelope_arithmetic="certified",
    interval_batch_size=0,
    cell_block_size=0,
    branch_batch_size=0,
    probe_failure="reject",
)
```

`jump_read` selects topology-preserving one-sided continuation reads or a faster bridged
finite-grid read. `probe_failure="assume_declared"` turns an unexecutable structural
probe into an author assertion and warning; it does not relax the mathematical
prerequisite.

With `envelope_arithmetic="certified"`, candidate ownership is ordered from the stored
floating-point operands using fixed-width integer arithmetic. Exact affine value comes
first, followed by deterministic geometric and stable-index tie-breaks. NaNs and other
non-finite or invalid geometry are not ordinary ordered values: the query is rejected
and remains uncovered/NaN so runtime validation can surface it. `"ordinary"` compares in
the working floating format and requires model-specific validation near crossings.

**Native capability.** `"certified"` uses pylcm's installed exact-affine CPU/CUDA
payload; it is not a pure-JAX numerical option. A compatible payload must exist for the
active JAX backend. NBEGM never silently falls back: if the payload is absent or
unloadable, certified mode raises `ExactAffineKernelUnavailableError` before returning a
certified result. Select `envelope_arithmetic="ordinary"` only when working-format
ownership is acceptable under model-specific crossing checks. The same requirement
applies when this NBEGM is the inner solver of `NNBEGM`.

The memory controls do not all mean “compiled batch width”:

- `stochastic_node_batch_size` and `envelope_segment_block_size` stream their named
  intermediate axes;
- `interval_batch_size` streams the continuation read and the candidate-envelope fold
  together. A positive width reads only that many interval rows, folds their candidates
  into one standing winner per query, then requests the next block. The standing winner
  retains its global stored-link index, so every partition resolves ownership exactly as
  the one-shot layout does under both envelope arithmetics — the same query nodes are
  feasible and the same candidate owns each of them. The published levels agree to
  within a few units in the last place rather than bit for bit, because a width is also
  a compiled vmap width and the backend vectorizes each one differently.
  `interval_batch_size=0` keeps the one-shot continuation matrix and envelope reduction;
- `cell_block_size` and `branch_batch_size` are compiled `lax.map` batch widths for the
  ride-cell and discrete-branch axes. The continuation read behind the branch axis runs
  once per class of branches that agree on every discrete action reaching the
  continuation (the regime transition, a law of motion, stochastic-state transition
  weights, a child's resources, the discount factor, or a schedule variable), so a
  budget-only action costs one read per cell however many branches it declares;
- lower positive values bound the named streamed or mapped width. For the map-width
  controls, `0` or a value covering the axis selects one vectorized pass.

These fields bound only their named mapped work, not surrounding arrays or total memory.

(api-nnbegm)=
### `NNBEGM`

```python
NNBEGM(inner=..., outer_search=...)
```

Nests an `NBEGM` liquid solve inside a configurable outer search. The inner solver must
use a bridged carry compatible with the outer fold. See
[Outer search and branch aggregation](outer_search.md).

The nested period kernel publishes no traced body of its own: its core-program graph
republishes the inner NB-EGM programs as `keeper:main`, `keeper:replay`,
`adjuster:main`, and `adjuster:replay`, each with the inner program's output roles,
scope, and planned disposition. The keeper programs are built from the period's own
inputs; the adjuster programs bind the outer post-decision at the first outer node, the
same shape every per-node call rebinds. A values-only solve dispatches the inner `main`
programs and the nested collapse publishes the value and the carry alone; a
replay-retaining solve dispatches the inner `replay` programs and assembles the nested
policy from their banks.

How the keeper and adjuster branches combine is an economic declaration, not a solver
setting: it lives on [`OuterContinuousMargin.adjustment_cost`](consumption_savings.md).

With `FiniteOuterGrid`, NNBEGM replays the keeper-plus-outer-grid candidates ranked
during the solve; with `AdaptiveOuterMesh` it republishes the mesh policies and the
search settings, and re-refines per subject at that subject's own resources. The
adaptive replay reads the exact generated mesh, which the solving model instance holds
beside the result, so that policy is retained under `ResultRetention.VALUES_AND_REPLAY`
and omitted as `NOT_PERSISTED` under `ALL_PERSISTABLE_ARTIFACTS`; the finite candidate
bank is self-contained and persists under both. Every
declaration that can affect that replay must therefore be phase-invariant by object
identity: a bare declaration and `Phased(solve=f, simulate=f)` are accepted, while
distinct solve/simulate functions, state or regime transitions, Koopmans aggregators,
and carried-only states are rejected during `Model(...)` construction. Use identical
declaration objects, remove the carried-only state, or use `GridSearch` until
phase-specific NNBEGM replay is implemented. In the `Phased` spelling, `f` must be the
exact same callable object in both fields; two distinct functions that compute the same
formula still count as phase variation.

NNBEGM searches the outer margin over post-decision **targets** — the outer stock a
candidate reaches — so simulation recovers the outer **action** that reached a stored
target by inverting the declared post-decision map. That inversion is exact or refused,
never approximate, so the map must be affine in the outer action with a constant
coefficient that is exactly a power of two, positive or negative:

- **accepted:** `new = old + action`, `new = old + 2 * action`,
  `new = old + 0.5 * action`, and `new = offset(states, params) + action` for an
  arbitrarily nonlinear `offset`. Anything the action does not enter is unrestricted, so
  depreciation, returns, and fixed transfers are all free.
- **refused:** a non-affine dependence — `action ** 2`, `exp(action)`,
  `clip(action, ...)`, `jnp.where(action > 0, ...)`, or division by the action.
- **refused:** a coefficient of zero, i.e. a map the action does not enter. Such a map
  retains no information about the action that reached the target, so none can be
  recovered.
- **refused:** a constant but non-dyadic slope. `3`, `1.5`, and `0.9` all fail.
- **refused:** a state-dependent slope, such as `old + (1 + 0.1 * old) * action`.

Recovery divides the coefficient out of the retained target, and binary division is
exact only by a power of two. Any other factor rounds, and a rounded action reassembles
a stock away from the node the solve ranked — at the edge of the outer state's declared
grid, or off the declared domain entirely, where there is no value function to read.
Each refusal is a `RegimeInitializationError` raised where the map is declared, rather
than a candidate dropped silently during the solve.

The restriction excludes any **state-dependent conversion technology**: a scale economy
in durable investment, a portfolio-size-dependent transaction cost, increasing-returns
installation — any adjustment cost that does not separate into a state-only offset plus
a dyadic multiple of the action. That is a modelling restriction, not a formatting one.
A constant non-dyadic price is usually recoverable by choosing units so the action is
the stock increment itself and moving the conversion factor into the budget's cost term,
which the inversion never reads. A state-dependent slope has no single coefficient to
divide out, so it is not recoverable that way. Both solve under `GridSearch`, which
searches the outer action directly and so never inverts it.

## Capability is validated, not inferred from class names

The selected solver inspects finalized declarations before numerical lowering. A model
that violates its state/action count, budget form, constraint route, continuation
layout, shock, or boundary assumptions fails during `Model(...)` or the first
parameter-dependent validation. There is no supported “try EGM and see whether it runs”
workflow.

Start with [Choosing a solver](../user_guide/choosing_a_solver.md). The mathematical map
is [Solver families](../methods/solver_families.md).
