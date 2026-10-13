---
title: Preference aggregation and certainty equivalents
---

# Preference aggregation and certainty equivalents

pylcm separates two operations that are often collapsed in textbook notation:

1. a **certainty equivalent** reduces the next-period value lottery to one number;
1. a **Koopmans aggregator** combines current utility with that continuation number.

For a non-terminal regime,

$$
Q_t = H\left(u_t,\operatorname{CE}_t(V_{t+1});\theta_H\right).
$$

The callable signatures declare their parameters. There is no separate list of parameter
names to synchronize.

## Linear recursion

`LinearExpectation()` computes the ordinary expectation of continuation values.
`LinearAggregator()` then combines current utility and the certainty equivalent in the
linear form with `discount_factor`.

The combination is the familiar time-additive expected-utility recursion. Either class
alone does not name a complete preference model: the result depends on both operations.

## Power and CES forms

`PowerMean()` applies a power-mean certainty equivalent to continuation risk.
`CESAggregator()` applies a CES form across current utility and the continuation
equivalent. Pairing them with the appropriate exponents gives the full Epstein–Zin
recursion; using only the CES aggregator does not by itself make preferences
Epstein–Zin.

`QuasiArithmeticMean` is the common certainty-equivalent contract for an invertible
transform and its inverse. Parameters beyond the required value/probability inputs
appear under the `certainty_equivalent` pseudo-function in the parameter template.

The numerical implementations share a stable weighted-power-mean kernel. This matters
near unit exponents and at small values, where a naive direct power expression can lose
the quantity or reverse action rankings.

## Model-level and regime-level declarations

`Model(koopmans_aggregator=..., certainty_equivalent=...)` broadcasts each object to all
non-terminal regimes. Alternatively, declare an object on every non-terminal regime.
Mixing model-level and selective regime-level ownership is rejected.

A terminal regime declares neither: there is no continuation to aggregate.

`Phased(solve=..., simulate=...)` is accepted for the Koopmans aggregator. It can
represent perceived versus realized intertemporal behavior, including naive
quasi-hyperbolic discounting.

`Decoupled(objective=..., value=...)` chooses actions with one aggregator and stores the
value of another, both with the Koopmans aggregator's signature:

- `objective` ranks the actions: the action taken maximizes it, or, with taste shocks,
  it sets the choice probabilities. It may be `Phased`.
- `value` is stored at the chosen action and passed back as the continuation. With taste
  shocks the stored value is its expectation at the noisy choice.

With `value = u + δ·CE`, an objective `u + β̃δ·CE` in the solve and `u + βδ·CE` in
simulation, this covers the naive (`β̃ = 1`), sophisticated (`β̃ = β`) and partially
naive (`β < β̃ < 1`) quasi-hyperbolic agents. For naive and partially naive agents the
stored value is the agent's assessment of the policy it expects to follow, which differs
from the long-run value of the policy it follows. The stored value is read off away from
its own maximum, so grid error in the chosen action enters it at first order. Only
`GridSearch` solves a `Decoupled` regime. See the executable
[beta-delta notebook](../explanations/beta_delta.ipynb).

Solver support for nonlinear certainty equivalents is narrower than support for linear
expectations:

- `GridSearch` aggregates any supported certainty equivalent directly on the action
  grid;
- `EGM`, `DCEGM`, and `NEGM` reject nonlinear certainty equivalents because their Euler
  inversions assume `LinearExpectation()`;
- `NBEGM` and `NNBEGM` implement `PowerMean()` paired with `CESAggregator()` through the
  NBEGM inner kernel, but only on qualified ride-along routes. A current-period jump,
  liquid-dependent continuation read, taste shock, or another incompatible declaration
  is still rejected.

These restrictions are validated when the model is built. The canonical matrix is in
[Solvers and capabilities](../reference/solvers.md#nonlinear-certainty-equivalents);
choose the solver before authoring the preference specification.

A runnable nonlinear example is [Epstein–Zin lifecycle](../examples/epstein_zin.ipynb).
Exact public names are listed in the
[API index](../reference/public_api.md#preferences-results-and-persistence).
