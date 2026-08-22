---
title: The endogenous-grid method
---

# The endogenous-grid method

The endogenous-grid method (EGM) removes a continuous root search from a smooth
consumption-saving problem by choosing tomorrow's assets first and recovering today's
resources from the Euler equation {cite}`carroll2006`.

Write post-decision savings as

$$
s = m - c,
$$

where $m$ is liquid resources and $c$ is consumption. For an exogenous savings node
$s_i$, the Euler condition has the schematic form

$$
u_c(c_i) =
\beta R\,\mathbb{E}\left[V_{m,t+1}(m'\mid s_i)\right].
$$

When marginal utility is invertible,

$$
c_i = u_c^{-1}\!\left(
\beta R\,\mathbb{E}[V_{m,t+1}(m'\mid s_i)]
\right),
\qquad
m_i = c_i + s_i.
$$

The pairs $(m_i,c_i)$ form an **endogenous** policy grid. Interpolation evaluates that
policy on the regime's exogenous liquid-state grid.

## Why it can be faster

A brute-force solve compares many consumption candidates for every current resource
node. EGM evaluates the Euler right-hand side once per savings node and obtains the
consumption choice by inversion. The benefit relies on three properties:

1. the relevant continuous choice is a one-dimensional liquid margin;
1. marginal utility can be inverted;
1. conditional on solver rows and smooth branches, the Euler right-hand side is a
   function of the post-decision state rather than every current liquid node.

Violating the third property can require one EGM construction per current state, losing
the amortization that motivated the method.

## `EGM` and `DCEGM` encode different budgets

pylcm's plain `EGM` uses the liquid state itself as cash-on-hand. Its kernel constructs
the endogenous location as `consumption + savings`, and model validation checks that the
declared post-decision function implements the corresponding identity.

`DCEGM` binds a genuine resources node. Wealth, labor income, taxes, and transfers may
feed that node before consumption is paid. It also handles a discrete choice and takes
an upper envelope across the resulting branches.

This distinction affects model authoring. Both use a `ConsumptionSavingsRegime` and
`LiquidMargin`, but the `resources` role is the liquid state for the plain identity case
and a named function for the richer budget. See
[Consumption-saving regimes and margins](../reference/consumption_savings.md).

## Borrowing constraints

An EGM solver enforces the lower edge of its savings grid by construction. Declaring
that economic restriction with `post_decision_lower_bound` lets pylcm verify that the
constraint and grid agree. An arbitrary callable that happens to compute the same
Boolean does not retain enough structure for that proof.

This does not mean every condition must be declarative. Read
[Constraints and structured Conditions](../reference/conditions.md) for the precise
boundary between ordinary callables and retained structure.

## Corners, discrete choices, and non-convex budgets

Plain EGM is for a smooth concave margin. A borrowing corner, discrete branch, or
institutional cliff changes how candidates are generated and compared:

- `DCEGM` handles discrete-continuous non-concavity with an upper envelope.
- `NBEGM` handles declared budget runs, kinks, jumps, and supported hard constraints.
- `NEGM` and `NNBEGM` place one of those liquid solves inside an outer search.

Use [Choosing a solver](../user_guide/choosing_a_solver.md) rather than treating these
classes as interchangeable performance switches.
