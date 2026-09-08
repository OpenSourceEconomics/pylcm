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

<!-- capability tables: rendered, do not edit by hand -->

| Solver       | Required declaration                                         | Problem shape                                                                | Hard prerequisites and supported constraints                                                                                                                                                                                                 | Main tradeoff                                                                                                                  |
| ------------ | ------------------------------------------------------------ | ---------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| `GridSearch` | Regime or a specialized regime                               | General discrete-continuous action product                                   | Ordinary callable constraints; EV1 taste shocks; transition-local joint lotteries                                                                                                                                                            | Broad representation; eligible singleton hard-max routes stream actions, while EV1 and collective reductions use dense actions |
| `EGM`        | ConsumptionSavingsRegime with one LiquidMargin               | Smooth concave one-state/one-action cash-on-hand problem                     | One continuous state and action; no discrete/process axes; identity resources; additive continuation; provable post-decision lower bound                                                                                                     | Narrow structural contract with no upper envelope                                                                              |
| `DCEGM`      | ConsumptionSavingsRegime with one LiquidMargin               | One liquid Euler margin and optional discrete choice                         | Valid resources and post-decision roles; lower bound; supported passive states and continuation layout; EV1 taste shocks                                                                                                                     | Constrained candidates and upper envelope; simulation may re-optimize on the action grid                                       |
| `NBEGM`      | ConsumptionSavingsRegime with one LiquidMargin               | Supported declared kinks, jumps, hard boundaries or smooth discrete branches | Supported case-piece or piecewise-affine declaration; proven constraint routes; nonlinear CE only on eligible ride-along routes; no EV1 taste shocks; marginal donation only on an unsharded self-carry main program with eligible ownership | Preserves declared topology; structural probes and candidate geometry add cost                                                 |
| `NEGM`       | NestedConsumptionSavingsRegime with liquid and outer margins | DCEGM inner solve conditional on a finite outer grid                         | Inner DCEGM contract plus outer state/action, post-decision, no-adjustment and cost roles; no EV1 taste shocks                                                                                                                               | Exact relative to the outer candidate set; retained candidates can dominate memory                                             |
| `NNBEGM`     | NestedConsumptionSavingsRegime with liquid and outer margins | NBEGM inner solve inside a finite or adaptive outer search                   | Inner NBEGM contract plus compatible outer search and branch aggregation; no EV1 taste shocks                                                                                                                                                | Declared budget topology inside each outer candidate adds structural and computational cost                                    |

## Execution axes

| Solver       | Reduced axes                            | Tiled axes                                              | Host axes         | Host-repeated programs             | Donation candidates | EV1 taste shocks | Nonlinear CE |
| ------------ | --------------------------------------- | ------------------------------------------------------- | ----------------- | ---------------------------------- | ------------------- | ---------------- | ------------ |
| `GridSearch` | `action_product`                        | `cell`                                                  | —                 | —                                  | —                   | Yes              | Yes          |
| `EGM`        | —                                       | —                                                       | —                 | —                                  | —                   | No               | No           |
| `DCEGM`      | `stochastic_node`                       | `cell`, `savings_point`, `euler_point`, `envelope_cell` | —                 | —                                  | —                   | Yes              | No           |
| `NBEGM`      | `stochastic_node`, `interval`, `branch` | `cell`                                                  | —                 | —                                  | `main`              | No               | Yes          |
| `NEGM`       | `stochastic_node`, `outer_candidate`    | `cell`, `savings_point`, `euler_point`, `envelope_cell` | —                 | —                                  | —                   | No               | No           |
| `NNBEGM`     | `stochastic_node`, `interval`, `branch` | `cell`                                                  | `outer_candidate` | `adjuster:main`, `adjuster:replay` | —                   | No               | Yes          |

<!-- end capability tables -->

Potential axes depend on the selected solver configuration. Model construction accepts
only axes declared by its actual programs; these tables do not waive structural
validation. Host-repeated programs may contain planned compiled cores. Regime submesh
placement and simulation subject tiling are separate execution policies.

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
GridSearch()
```

Covers the complete represented state-action product and applies constraints directly.
It is the broadest route and the default solver on `Regime`. Eligible JIT solve-value
routes evaluate bounded C-order action blocks while preserving the complete support.

An eligible streamed core declares the reduced axis `action_product`, the flattened
Cartesian product of the regime's actions. `ExecutionConfig(axis_widths=
{"action_product": n})` fixes the width every such core is compiled at; a width above
the product's extent selects the whole product. With no fixed width and no
device-memory budget, pylcm streams every eligible core at its bootstrap width: the
largest power of two below the action product, capped at 64. With an
[`ExecutionConfig`](runtime_and_results.md#compiler-workspace-budgets) budget, the
planner instead walks a deterministic width frontier widest-first and dispatches the
first candidate whose compiler-reported peak fits. Supplying both makes the fixed width
the only candidate, which must fit the budget. A route whose action reduction is
deliberately dense, unsupported, or trivial declares no `action_product` axis.

The separate `cell` axis tiles the flattened Cartesian product of inner states.
`ExecutionConfig(axis_widths={"cell": n})` bounds the number of state cells evaluated
together, including any folded quadrature nodes. Each output leaf recovers its state
axes before folding; collective values also retain their trailing stakeholder axis.
Sharded states stay outside this loop; fixed states also retain their co-mapping with
continuation values, preserving device-local reads. Empty and singleton state products
declare no `cell` axis. A core
with this axis is planned even when its action reduction remains deliberately dense.

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
resolved. Remaining-consumer counts are committed only after successful dispatch.
The scheduler releases an eligible temporary after its final consumer, while retained
outputs, aliases, and undeclared host reads keep their owners live. Donation additionally
requires a declared program candidate, sole eligible ownership, and a compiler-accepted
alias; a zero consumer count alone does not authorize it. The current built-in donation
route is the unsharded NBEGM self-carry main program described above. Dense programs
without declared value reads and other unplanned consumers remain pinned.

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
)
```

`DCEGM` supports a genuine resources node and optional discrete choice.
It does not require a nontrivial discrete choice: it is also the supported route
for a smooth liquid problem whose genuine resources node, passive states, or stochastic
processes make plain `EGM` ineligible.

The supported typed envelope configurations are `ExactEnvelope`, `FUESEnvelope`,
`RFCEnvelope`, `LTMEnvelope`, and `MSSEnvelope`. String selectors are not part of the
public API. Exact-envelope node cells are tiled through the model's
`ExecutionConfig(axis_widths={"envelope_cell": width})`; the axis is exported as
`ENVELOPE_CELL_AXIS`. FUES uses a fixed scan unroll factor of one, recorded in the
compiled program identity. See [Upper envelopes](envelopes.md) for their distinct
contracts.

`refined_grid_factor` provides NaN-padded storage headroom for ownership changes in each
envelope row. A row that needs more slots is reported as overflow and NaN-poisoned; this
field does not change the density of the policy read-out grid. `n_constrained_points`
controls the borrowing-corner segment.

DC-EGM owns no block-size field. Each loop it could stream is an execution axis its
value and replay programs declare, and the width is fixed with
`ExecutionConfig(axis_widths=...)`:

- `stochastic_node` folds the child stochastic-node expectation. It is a weighted sum,
  so two widths reorder floating-point adds and the values they publish agree to the
  working format's rounding rather than bit for bit.
- `cell` tiles the per-combo solve over the regime's output state cells — its discrete
  and passive states. Discrete actions stay outside it: the action aggregation needs
  every action's value at once.
- `savings_point` tiles the per-savings-node continuation, the dominant working buffer.
- `euler_point` tiles the per-node solve of the asset-row kernel, which runs when a
  savings-stage function reads the current Euler state.
- `envelope_cell` tiles adjacent-candidate resource cells inside the exact envelope;
  other envelope backends declare no such loop.

The last four concatenate their tiles rather than folding them, so every width names
the same result. A loop of a single cell has nothing to tile and carries no
declaration, so its name is refused for such a model. Grids declare economic support;
execution widths belong in `ExecutionConfig.axis_widths`.

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
NEGM(inner=..., outer_grid=...)
```

Runs the bound `DCEGM` inner solve for every finite outer-grid node and includes the
keeper. The sweep declares those nodes as the `outer_candidate` execution axis, so how
many candidate values are evaluated at once is a planner width
(`ExecutionConfig(axis_widths={"outer_candidate": k})`) rather than a solver field. A
narrow width reduces temporary evaluation memory, but it does not in general cap the
size of the candidate bank retained for later envelope or ordered-fold operations. Peak
memory can therefore continue to grow with the full candidate set. Measure both
temporary and retained arrays for the exact model and solver profile.

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
    envelope_arithmetic="certified",
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

Compiled widths belong to the model's `ExecutionConfig`, with constants imported from
`lcm.solvers`. Only axes a program actually uses are declared:

- `stochastic_node` (`STOCHASTIC_NODE_AXIS`) folds the child stochastic-node expectation
  where the child mesh is also present in the program's state grids. Absent, singleton,
  and cross-grid meshes keep a complete expectation and declare no such axis.
- `interval` (`INTERVAL_AXIS`) streams the continuation read and the candidate-envelope fold
  together. A positive width reads only that many interval rows, folds their candidates
  into one standing winner per query, then requests the next block. The standing winner
  retains its global stored-link index, so given the candidate records every partition
  decides ownership by the same total order over the same identities as the one-shot
  layout, under both envelope arithmetics: no width can hand a query to another
  candidate by where a record happens to be stored, and the step reports the owner of
  every node on request (`return_owner=True`) so a partition test asserts it exactly.
  What a width does change is the compiled vmap width the records are produced at, and
  the backend vectorizes each one differently: the published levels agree to within a
  few units in the last place rather than bit for bit, and two candidates whose reads
  tie to within that spacing are ordered by the records each width produced. The one
  place this is visible on a regular grid is a node where a savings-node point candidate
  coincides with an interior candidate: the two are the same point, and which of them
  is named the owner can differ between widths while the published level does not.
  This axis exists only when continuation reads the liquid state across multiple
  declared intervals. Omitting a width lets the planner choose the streamed width.
- `cell` (`CELL_AXIS`) tiles independent ride cells inside each co-mapped carry slice;
  it excludes the distributed states the carry is co-mapped over.
- `branch` (`BRANCH_AXIS`) batches discrete branches before their maximum, retaining
  the conditional banks needed for replay. The continuation read behind the branch axis runs
  once per class of branches that agree on every discrete action reaching the
  continuation (the regime transition, a law of motion, stochastic-state transition
  weights, a child's resources, the discount factor, or a schedule variable), so a
  budget-only action costs one read per cell however many branches it declares;
- Singleton meshes and routes without the corresponding computation omit its axis.

For a model that declares continuation intervals, the execution setting is:

```python
from lcm import ExecutionConfig
from lcm.solvers import INTERVAL_AXIS

execution_config = ExecutionConfig(axis_widths={INTERVAL_AXIS: 2})
```

Widths must be positive; values above an axis's extent clamp to that extent. The
canonical interval stream has no separate segment-width loop, and certified envelope
queries own their internal partition, so NBEGM exposes no envelope-segment axis.
These widths bound their named work, not surrounding arrays, retained branch banks,
compilation memory, or total device memory. The interval reduction remains a declaration
of the existing stable-identity fold (`INTERVAL_ENVELOPE_REDUCTION`).

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
`adjuster:main`, and `adjuster:replay`, each with the inner program's output roles and
scope. Both roles retain the inner planned disposition, including under
`AdaptiveOuterMesh`: the host decides which outer nodes to request, and each request
dispatches a compiled inner program with its planner axes and continuation transfers.
Both roles declare the leaves they read,
so the nested dispatch node has no undeclared reader. The keeper programs are built from
the period's own inputs; the adjuster programs bind the outer post-decision at the first
outer node, the same shape every per-node call rebinds. A values-only solve dispatches
the inner `main` programs and the nested collapse publishes the value and the carry
alone; a replay-retaining solve dispatches the inner `replay` programs and assembles the
nested policy from their banks.

Both outer-search routes declare `outer_candidate` as a host-dispatch axis. Set its
width when building the model with
`execution_config=ExecutionConfig(axis_widths={"outer_candidate": k})`. The host loop
dispatches at most `k` pending nodes per step, preserving their order; without a fixed
width it dispatches all pending nodes in one step. A finite values-only solve
releases each completed chunk's per-node
value temporaries before dispatching the next chunk. It still retains every
continuation carry, and a replay-retaining solve also keeps all node results.
The adaptive search retains its exact-node bank. The width therefore limits
dispatch chunks, not the memory occupied by these complete banks.

How the keeper and adjuster branches combine is an economic declaration, not a solver
setting: it lives on [`OuterContinuousMargin.adjustment_cost`](consumption_savings.md).

With `FiniteOuterGrid`, NNBEGM replays the keeper-plus-outer-grid candidates ranked
during the solve; with `AdaptiveOuterMesh` it republishes the mesh policies and the
search settings, and re-refines per subject at that subject's own resources. The
adaptive replay reads the exact generated mesh, whose nodes the result carries as the
candidate axis of the policy's descriptor, so both the adaptive policy and the finite
candidate bank are self-contained, retained under every replay-retaining
`ResultRetention`, and persist in the complete archive. Every
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
