---
title: Choosing a solver
---

# Choosing a solver

Choose a solver before you write the detailed model. Solver selection determines which
economic roles and structural boundaries the model must declare; it is not a performance
switch to flip after an arbitrary `Regime` has been assembled.

::::\{important} Choose the declaration before the specialized solver. EGM-family
solvers are not drop-in replacements for `GridSearch`: a model intended for one should
use `ConsumptionSavingsRegime` or `NestedConsumptionSavingsRegime` and name its margins
from the outset. Start from
[Authoring for EGM-family solvers](authoring_specialized_solvers.md). ::::

Make the decision in two passes:

1. keep only solvers whose assumptions represent the economic problem;
1. benchmark those candidates for accuracy, compilation, memory, and wall time on the
   target hardware.

## Before choosing an EGM-family solver

An EGM-family route is available only when the model exposes the structure the numerical
method consumes. Check these gates before selecting a solver:

- the declaration is `ConsumptionSavingsRegime` for one liquid margin or
  `NestedConsumptionSavingsRegime` for a liquid margin inside one outer continuous
  choice;
- each margin names its state, action, post-decision state, and resources or
  no-adjustment role;
- the savings grid begins at the declared `post_decision_lower_bound`;
- every additional constraint, discrete choice, stochastic object, boundary, and
  preference recursion is supported by the particular solver;
- kinks, jumps, and hard boundaries are declared with case pieces or a piecewise-affine
  schedule rather than hidden inside an opaque function.

Plain `EGM` has the narrowest gate: exactly one continuous state and action, no discrete
or process states/actions, liquid resources equal to the liquid state, savings equal to
state minus action, utility independent of the liquid state, the default Koopmans
aggregator, and no solve-time constraint except a provable savings lower bound. `DCEGM`,
`NBEGM`, and the nested solvers each add specific structure; they do not relax every
gate at once.

## Pass 1: represent the problem

```{mermaid}
flowchart TD
    start(["Continuous choice structure?"])
    start -->|"No special Euler margin"| gs["GridSearch + Regime"]
    start -->|"One liquid Euler margin"| one{"Declared kinks, cliffs, or hard boundaries?"}
    start -->|"Liquid margin nested inside an outer durable/illiquid choice"| two{"Declared structure on the liquid margin?"}
    one -->|"No"| disc{"Discrete choice creates competing branches?"}
    one -->|"Yes"| nb["NBEGM + ConsumptionSavingsRegime"]
    disc -->|"No; smooth and concave"| egm["EGM + ConsumptionSavingsRegime"]
    disc -->|"Yes"| dc["DCEGM + ConsumptionSavingsRegime"]
    two -->|"No"| ne["NEGM + NestedConsumptionSavingsRegime"]
    two -->|"Yes"| nn["NNBEGM + NestedConsumptionSavingsRegime"]
```

| Solver path                               | Problem shape and declaration                                                                        | Constraint shape                                                | Main tradeoff                                                                                         |
| ----------------------------------------- | ---------------------------------------------------------------------------------------------------- | --------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------- |
| `GridSearch + Regime`                     | General discrete-continuous action product                                                           | Ordinary callable constraints                                   | Broadest representation; cost grows with the full action product                                      |
| `EGM + ConsumptionSavingsRegime`          | Smooth, concave, one-state/one-action cash-on-hand problem                                           | Declared savings lower bound only                               | Fastest and simplest EGM route; very narrow contract                                                  |
| `DCEGM + ConsumptionSavingsRegime`        | One liquid margin with competing discrete-continuous branches                                        | Intrinsic budget and declared savings lower bound               | Off-grid Euler inversion plus an upper envelope; envelope work and simulation re-decision matter      |
| `NBEGM + ConsumptionSavingsRegime`        | One liquid margin with supported declared kinks, jumps, hard boundaries, or smooth discrete branches | Supported structured boundary declarations; no EV1 taste shocks | Preserves topology that ordinary DCEGM cannot; more validation and candidate geometry                 |
| `NEGM + NestedConsumptionSavingsRegime`   | A DCEGM liquid solve conditional on a finite outer candidate grid                                    | Inner DCEGM contract plus declared outer roles                  | Exact relative to the outer candidate set; work scales with its size                                  |
| `NNBEGM + NestedConsumptionSavingsRegime` | An NBEGM liquid solve inside a finite or adaptive outer search                                       | Inner NBEGM contract plus supported outer search/aggregation    | Handles both declared inner boundaries and an outer margin; highest structural and computational cost |

Use `GridSearch` for genuinely coupled multi-dimensional choices, unsupported
constraints, or any problem whose required structure cannot be declared honestly. It is
also the baseline against which specialized solutions should be checked.

The EGM routes require named liquid roles. Nested routes add named outer roles.
Institutional kinks and cliffs must be declared as case pieces or a piecewise-affine
schedule from the outset. Read
[Consumption-saving regimes and margins](../reference/consumption_savings.md) and
[Declared non-convex budgets](../methods/nonconvex_budgets.md) before authoring those
models.

## Pass 2: decide whether specialization pays

Among correct representations, compare:

- **candidate growth:** action products for grid search; savings nodes, branches,
  stochastic nodes, and outer candidates for EGM-family methods;
- **peak memory:** dense candidates versus streamed envelope/outer batches;
- **compilation:** number and size of distinct JAX programs;
- **hardware:** dense static work often favors GPUs; sequential topology scans often
  favor CPUs;
- **accuracy:** grid spacing, interpolation, boundary ownership, and approximation
  profiles;
- **workflow:** one solve, repeated estimation, or repeated simulation can amortize
  different fixed costs.

There is no universal break-even point. Benchmark the actual model and device. See
[Scaling, memory, and hardware](../methods/performance_scaling.md) for the reasoning and
[Performance and memory tuning](tuning.md) for the workflow.

## Solver summary

| Solver       | Represents                                                          | Main numerical configuration                              |
| ------------ | ------------------------------------------------------------------- | --------------------------------------------------------- |
| `GridSearch` | Broad discrete-continuous problems                                  | No solver-specific fields                                 |
| `EGM`        | Smooth one-margin cash-on-hand problem                              | Savings grid                                              |
| `DCEGM`      | One liquid margin with discrete-choice non-concavity                | Savings grid, envelope, refinement/batching               |
| `NEGM`       | `DCEGM` inner solve conditional on a finite outer grid              | Inner solver, outer grid, batch size                      |
| `NBEGM`      | Declared liquid kinks, jumps, hard boundaries, or discrete branches | Savings grid, jump read, comparison and batching controls |
| `NNBEGM`     | Nested outer choice with inner `NBEGM`                              | Inner solver, outer search, branch aggregation            |

Exact constructors and limitations are in
[Solvers and capabilities](../reference/solvers.md).

## Constraints do not all need `Condition`

Use an ordinary callable when pylcm only needs the Boolean result. Use a structured
`Condition` when a solver must retain named comparisons to prove, compile, or precisely
refuse a constraint—or when that declaration is clearer. A `Condition` does not grant a
solver support it otherwise lacks.

The syntax and solver interaction live on the separate
[Constraints and structured Conditions](../reference/conditions.md) page. Case pieces
and piecewise-affine schedules are separate declarations for budget structure, not
alternative constraint syntax.

## Verify the choice

Before trusting a specialized route:

1. solve a reduced version with `GridSearch`;
1. compare values and discrete decisions, not only absence of NaNs;
1. inspect borrowing corners and every declared boundary from both sides;
1. repeat at both supported numerical precisions where relevant;
1. measure cold compile, warm execution, and peak memory;
1. document which approximation profile and hardware produced the result.

Use `log_level="debug"` while developing the model so structural and numerical
validation fails loudly.
