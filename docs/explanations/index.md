---
title: Explanations
---

# Explanations

These are developer-focused explanations of internal pylcm mechanisms. They are designed
for contributors and advanced users who want to understand how pylcm works under the
hood.

- [Function Representation](function_representation.ipynb) — What the function
  representation does and how it works.
- [Interpolation and Extrapolation](interpolation.ipynb) — How pylcm's coordinate
  finders and `map_coordinates` handle interpolation and extrapolation across grid
  types.
- [Approximating Continuous Shocks](approximating_continuous_shocks.ipynb) — Quadrature
  rules and Markov chain approximations for IID and AR(1) shock processes.
- [Stochastic Transitions](stochastic_transitions.ipynb) — Why regime, discrete state,
  and continuous shock transitions use different runtime representations.
- [Dispatchers](dispatchers.ipynb) — How `productmap`, `vmap_1d`, and
  `simulation_spacemap` evaluate scalar functions on structured spaces.
- [The Phase Grammar](phase_grammar.ipynb) — How `Phased(solve=..., simulate=...)` gives
  regime slots per-phase variants, with carried states as the flagship use.
- [Beta-Delta (Quasi-Hyperbolic) Discounting](beta_delta.ipynb) — How to use the plugin
  system to model consumers with this particular form of time-inconsistent preferences.
- [Discrete-Continuous Choice: Iskhakov et al. (2017)](iskhakov_et_al_2017.ipynb) — Why
  discrete choices produce saw-tooth consumption functions, illustrated on the paper's
  retirement model, with a brute-force vs DC-EGM accuracy comparison.
- [Internal Architecture](architecture.md) — Map of the source tree, the user/engine
  boundary, and the role each package plays.
- [Adding a Solver](adding_a_solver.md) — The solver contract (`Solver`, `PeriodKernel`,
  `KernelResult`), the lifecycle the engine drives it through, and the invariants that
  keep the backward-induction loop solver-agnostic.
