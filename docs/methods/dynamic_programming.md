---
title: Dynamic programming and pylcm
---

# Dynamic programming and pylcm

pylcm solves finite-horizon dynamic programs. In a single-regime notation, its general
period-$t$ recursion is

$$
V_t(x) = \max_{a \in A_t(x)}
H_t\left(u_t(x, a), \operatorname{CE}_t[V_{t+1}(x')]\right).
$$

With the default `LinearAggregator()` and `LinearExpectation()`, this reduces to

$$
V_t(x) = \max_{a \in A_t(x)}
\left\{u_t(x, a) + \beta_t\,
\mathcal{E}_t[V_{t+1}(x')]\right\}.
$$

The state $x$ is declared through `states`, the choice $a$ through `actions`, the flow
payoff through `functions["utility"]`, and the feasible set through `constraints`.
`state_transitions` and stochastic-process declarations determine $x'$. pylcm composes
those named pieces, solves backward over the `AgeGrid`, and then reuses the solution for
forward simulation.

This page maps the dynamic-programming objects to pylcm. It is not an introduction to
dynamic programming. For that, use the
[QuantEcon Dynamic Programming book](https://dp.quantecon.org/).

## Regimes add a discrete state with changing structure

Many lifecycle models change their equations and available choices across employment,
retirement, marriage, or death. pylcm represents each qualitatively distinct problem as
a `Regime`. A regime transition determines the next regime, and each regime supplies its
own states, actions, functions, constraints, and transition laws.

The continuation is therefore a weighted or deterministic read from the value functions
of reachable target regimes. Per-target transition dictionaries declare structural
reachability; omitted targets are not merely assigned zero probability, they are absent
from the problem.

See [Regimes](../user_guide/regimes.ipynb) for the workflow and
[Model and Regime](../reference/model_and_regime.md) for exact declaration forms.

## Solve and simulation are related phases, not identical programs

Backward induction evaluates values over numerical grids. Simulation evaluates policies
for a cohort at realized, potentially off-grid states. Most declarations broadcast to
both phases, but `Phased(solve=..., simulate=...)` permits different implementations
where their data topology genuinely differs. A carried state, for example, can be
derived during solution and remain a seeded state during simulation.

The exact grammar is in
[Transitions and phase specialization](../reference/transitions.md); the numerical
reason is developed in
[Phase-dependent model structure](../explanations/phase_grammar.ipynb).

## The solver changes the representation of the maximization

`GridSearch` constructs the full state-action product and evaluates the objective and
constraints on it. EGM-family solvers replace one continuous maximization with an
Euler-equation inversion and therefore need stronger assumptions and named economic
roles.

This is why solver selection belongs before detailed model authoring. The economic
problem and the numerical representation have to agree; changing only `solver=...`
cannot turn an arbitrary model into an endogenous-grid problem. Continue with
[Solver families](solver_families.md) and
[Choosing a solver](../user_guide/choosing_a_solver.md).
