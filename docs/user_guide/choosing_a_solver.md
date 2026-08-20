---
title: Choosing a solver
---

# Choosing a solver

pylcm ships several solvers for the continuous part of a regime's period problem. They
are not interchangeable: each is fastest — or only correct — for a particular problem
structure. Pick one in two passes. The **feasibility** tree narrows to the solvers that
are *correct* for your problem's structure; the **hardware / speed** tree picks the
fastest among those on your machine.

The guiding principle is conservative. **`GridSearch` (brute force) is the default, and
it is often the right answer.** On a GPU it is a dense map-reduce with static shapes and
perfect chunkability, and it is exact to its action grid. An endogenous-grid method has
lower arithmetic complexity, but that only pays off if it does not materialize large
transients, carry long sequential scans, or compile many shape variants. So the rule is:
**adopt a structure-specific solver only after benchmarking it against `GridSearch` on
your target hardware** — on peak memory, compile time, and wall-time. (Near
institutional cliffs, `GridSearch` smooths across the discontinuity, so there it is a
speed baseline and a diagnostic, not the accuracy reference — see
[the NB-EGM solver](nbegm.md).) See [Performance and Memory Tuning](tuning.md) and
[Benchmarking](benchmarking.md).

## Decision tree by feasibility

Which solvers are *correct* for your problem's structure.

```{mermaid}
flowchart TD
    q0(["How many continuous states carry an Euler equation?"])
    q0 -->|"None"| gs0["GridSearch"]
    q0 -->|"1"| qbp{"Declared institutional breakpoints? (asset tests, brackets, notches, floors)"}
    q0 -->|"2"| q2d{"Genuinely coupled 2-D first-order-condition system?"}

    q2d -->|"Yes"| twodim["GridSearch"]
    q2d -->|"No — clean inner nest (liquid + durable/illiquid)"| qnest{"Declared breakpoints on the liquid margin?"}

    qnest -->|"Yes"| nnbegm["NNBEGM"]
    qnest -->|"No"| negm["NEGM"]

    qbp -->|"Yes"| nbegm["NBEGM"]
    qbp -->|"No"| qdc{"Discrete choice induces non-concavity (secondary kinks)?"}

    qdc -->|"Yes"| dcegm["DCEGM (or NBEGM — see hardware tree)"]
    qdc -->|"No — smooth and concave"| egm["EGM or GridSearch"]
```

A genuinely coupled 2-D first-order-condition system has no EGM route in pylcm — the
specialised method for it is published with its own paper — but `GridSearch` solves it,
because brute force never forms a first-order condition at all. The cost is what argues
against it: a 2-D action space means the product of two action grids at every state
node, so the candidate count grows multiplicatively in the two grid sizes rather than
additively. That is the reason to look for a structure-specific solver, and it is a
reason about run time rather than about correctness.

Under the 2-continuous-state nest, `NEGM` and `NNBEGM` differ only in the inner solver:
`NEGM` nests a `DCEGM` solve, `NNBEGM` an `NBEGM` one, so declared liquid kinks, jumps,
and floors keep their exact treatment inside every outer candidate. Both re-solve the
whole inner problem per outer-grid node, so neither restricts how the outer margin
enters the model — only the finite outer grid limits them.

At the secondary-kink leaf, both `DCEGM` and `NBEGM` are correct. `DCEGM` is the natural
choice for a plain discrete–continuous problem with no institutional breakpoints;
`NBEGM` handles the same secondary kinks (via its discrete-branch envelope and
Euler-path fold-splitting) and is the choice once the model *also* carries declared
cliffs. Which is faster is a hardware question — the next tree.

## Decision tree by hardware and speed

Among the feasible solvers, which is fastest.

```{mermaid}
flowchart TD
    h0(["Target hardware?"])
    h0 -->|"GPU"| g1{"Action grid modest for the required accuracy?"}
    h0 -->|"CPU"| c1{"Branchy discrete–continuous envelope?"}

    g1 -->|"Yes"| gs["GridSearch — dense map-reduce usually wins"]
    g1 -->|"No — fine grid, or cliffs"| g2{"Full-row envelope is the memory wall?"}
    g2 -->|"Yes"| gq["Query-side segmented envelope: NBEGM, or DCEGM(envelope='ltm')"]
    g2 -->|"No"| ge["EGM-family: EGM / NEGM / NBEGM"]

    c1 -->|"Yes"| cd["DCEGM — FUES / RFC / LTM / MSS all viable on CPU"]
    c1 -->|"No — smooth"| ce["EGM, or GridSearch"]
```

Two cross-cutting factors:

- **GPU parallelism.** A GPU favours dense, static-shape map-reduces — `GridSearch`, and
  the query-side upper envelope used by `NBEGM` (and available to `DCEGM` via
  `envelope="ltm"`). A CPU tolerates the sequential, topology-discovering envelope scans
  (`DCEGM`'s FUES backend) that a GPU runs poorly. So at the secondary-kink leaf, prefer
  `NBEGM`'s query-side envelope on a GPU and `DCEGM`'s FUES on a CPU.
- **Compile-shape explosion.** Many static shapes — long age grids, per-period target
  splits, branch axes — multiply compiled programs. When that dominates, fall back to
  `GridSearch` or a simple EGM.

## Solvers at a glance

| Solver       | Use when                                                                                                                                              | Key constructor arguments   |
| ------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------- |
| `GridSearch` | The default. Any regime, especially with a modest continuous-action grid on a GPU.                                                                    | *(none)*                    |
| `EGM`        | Smooth, concave one-asset consumption–saving problem where a fine action grid would otherwise be needed.                                              | `savings_grid`              |
| `DCEGM`      | One liquid asset with a discrete choice that makes the value function non-concave (secondary kinks).                                                  | `savings_grid`, `envelope`  |
| `NEGM`       | Two continuous choices with a clean nest: an inner 1-D EGM consumption solve inside an outer deterministic search over a durable/illiquid post-state. | `inner`, `outer_grid`       |
| `NNBEGM`     | The `NEGM` nest with **declared** breakpoints on the inner liquid margin.                                                                             | `inner`, `outer_grid`       |
| `NBEGM`      | One liquid asset with **declared** institutional kinks and cliffs. See [the NB-EGM solver](nbegm.md).                                                 | `savings_grid`, `jump_read` |

## Writing a constraint: callable or `Condition`?

Start from the shape of the economic restriction, not from the solver. An ordinary
callable is enough when the restriction is simply executable Boolean logic:

```python
import jax.numpy as jnp

from lcm.typing import BoolND, ContinuousAction, FloatND


def consumption_is_feasible(
    consumption: ContinuousAction,
    resources: FloatND,
) -> BoolND:
    """Consumption is positive and does not exceed available resources."""
    return (consumption > 0.0) & jnp.isfinite(consumption) & (consumption <= resources)
```

Use a `Condition` when pylcm must retain the restriction as named comparisons, or when
that declarative form is clearer for the model. Build conditions with `lcm.ref`:

```python
import lcm

# One named value compared with a literal.
nonnegative_savings = lcm.ref("savings") >= 0.0

# One named value compared with another named value. The second name may be a
# computed function or a parameter.
within_borrowing_limit = lcm.ref("assets") < lcm.ref("borrowing_limit")

# Intersection and union.
max_hours = 40.0
liquid_and_available = (lcm.ref("cash") >= 0.0) & (lcm.ref("hours") <= max_hours)
insured_or_eligible = (lcm.ref("insured") == 1) | (
    lcm.ref("income") <= lcm.ref("eligibility_limit")
)

# Complement.
not_retired = ~(lcm.ref("retired") == 1)

# Conditional requirement.
working_respects_hours = lcm.implies(
    premise=lcm.ref("working") == 1,
    consequent=lcm.ref("hours") <= max_hours,
)
```

An ordinary callable can compute the same Boolean result in general. `Condition` is
**needed** only when pylcm must retain the named comparison structure — for example so a
solver can prove that its construction already enforces `savings >= 0`, compile a
declared boundary, or give a precise early refusal. A `Condition` does not make an
otherwise unsupported restriction supported: the selected solver still needs a route
where all required names are available, or a proof/compiler that understands its exact
shape.

Prefer a specialized declaration when pylcm provides one. A post-decision lower bound
can be written directly as `lcm.ref("savings") >= 0.0`, but the specialized form ties
the name to the regime's declared liquid margin and therefore cannot drift away from it:

```python
from lcm.consumption_savings_regime import (
    LiquidMargin,
    post_decision_lower_bound,
)

liquid = LiquidMargin(
    state="wealth",
    action="consumption",
    resources="wealth",
    post_decision_state="savings",
)
borrowing_limit = post_decision_lower_bound(margin=liquid, lower=0.0)
```

Both spell the same comparison. The specialized form is preferred when the declaration
belongs to that margin.

### When do I need retained structure?

| Constraint shape                                       | Declaration                                            | `GridSearch`                                                             | EGM-family / `NBEGM` routes                                                                        |
| ------------------------------------------------------ | ------------------------------------------------------ | ------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------- |
| Arbitrary executable logic                             | Ordinary callable                                      | Evaluates it on complete state-action candidates.                        | Evaluates it only where the route has every required name; otherwise refuses it as opaque.         |
| Named comparisons whose structure matters              | `Condition` built with `lcm.ref` and Boolean operators | Evaluates the condition like any other constraint.                       | May evaluate, prove, compile, or precisely refuse it, depending on the route and available names.  |
| Exact lower bound on a declared post-decision state    | `post_decision_lower_bound`                            | Evaluates the resulting comparison.                                      | Can prove it from the matching savings grid; a role or numeric mismatch is refused at model build. |
| Restriction requiring names no candidate stage exposes | Callable or `Condition`                                | Usually the broad fallback when its full candidate contains those names. | Refuses it unless an exact structural proof or boundary compiler covers it.                        |

A solver takes numerical configuration only. Which state, action, and function play the
liquid and outer roles is declared on the regime — `LiquidMargin` on a
`ConsumptionSavingsRegime`, plus `OuterContinuousMargin` on a
`NestedConsumptionSavingsRegime` for the nested solvers — so the same solver object can
be reused across regimes that name their margins differently.

`DCEGM`'s upper-envelope backend is selectable via `envelope=` (`"exact"`, `"fues"`,
`"rfc"`, `"ltm"`, `"mss"`). `"exact"` is the default and pylcm's own construction, which
resolves ownership by certified comparison rather than by scanning; the other four are
ports of the method columns of Dobrescu & Shanker 2024. `"fues"` is a
topology-discovering scan (CPU-friendly); `"ltm"` is a query-side segment evaluator
(GPU-friendly). Switch only under a benchmark.

## `EGM` has no resources function; `DCEGM` does

The two one-asset solvers differ in a way that decides how the budget is written, not
merely in how the envelope is taken.

`EGM`'s kernel forms the endogenous grid as `consumption + savings_grid` and publishes
the result on the liquid state's own grid. **For `EGM` the liquid state is cash-on-hand,
by enforced identity — there is no separate resources quantity.** Model build checks
this rather than assuming it: the post-decision function's leaf arguments must be
exactly the liquid state and the consumption action, and the composed function is then
sampled on both grids and required to equal `liquid_state - consumption` at
precision-aware tolerances. A post-decision function that fails either check is refused,
naming the liquid state and the action it must be written in.

The same identity is why `EGM` admits exactly one continuous state. The liquid role is
filled positionally and `liquid = consumption + savings` leaves no axis for a second
one.

`DCEGM` binds a genuine resources function through its regime's `LiquidMargin`, so there
`resources = wealth + labour_income` is a legitimate node and the liquid state is a
stock the budget is built from. If your budget has anything in it beyond the asset
itself — income, a transfer, a tax — that is a `DCEGM` (or `NBEGM`) model, not an `EGM`
one, and the check above is what tells you so at build rather than at the wrong answer.

## A note on current-state dependence

Standard EGM's speed comes from *amortization*: invert the Euler equation once per
post-decision savings node and read the resulting policy at every current state. That
only works when the Euler right-hand side depends on savings alone after conditioning on
discrete states and smooth branches. If an institutional rule leaves the right-hand side
depending on the *current* liquid state even after conditioning, the amortization is
lost. The exact fallback is one EGM problem per current-state node (`DCEGM`'s asset-row
mode), which forfeits EGM's advantage — at which point `GridSearch` is usually the
better choice. `NBEGM`'s per-interval continuation path is the structured escape hatch:
when the dependence is piecewise-constant on declared breakpoints, one curve per
interval restores exactness without per-node replication.
