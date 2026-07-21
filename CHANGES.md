# Changes


This is a record of all past PyLCM releases and what went into them in reverse
chronological order. We follow [semantic versioning](https://semver.org/).


## Unreleased

- Fixes the simulate-phase Q so a simulated agent prices its continuation under
  its *perceived* law. `Phased` state transitions are now supported for
  `MarkovTransition` laws, and the simulate-phase Q is assembled from two
  phase-closed halves: the current flow (utility, feasibility, `H`) from the
  simulate transitions and simulate function pool, the continuation (state laws,
  stochastic weights, and every helper they read) from the solve ones. The
  realized draw is unchanged and still follows the simulate laws.

  **Behaviour change.** A model is numerically unchanged unless a *phase-varying*
  (`Phased`) function lies in the dependency ancestry of a continuation
  transition or of a `next_<state>` read by within-period utility or
  feasibility — the solve phase is untouched, and for every non-`Phased` name
  both pools hold the same object. Models that do have such a dependency change,
  by design: they previously resolved that helper from the simulate pool in the
  continuation (and the solve law with simulate helpers in the flow), which is
  the bug being fixed. This pattern was reachable before this release via a
  `Phased` helper under a bare law, so the correction can move results for
  existing models.

  Two variants of a `Phased` stochastic law are validated separately; previously
  only one of them was checked numerically. A per-target dict inside `Phased`
  must be per-target in both phases and cover the same targets.

- Adds the DC-EGM solver (Iskhakov, Jørgensen, Rust & Schjerning 2017) as a
  per-regime alternative to grid search: `Regime(solver=lcm.DCEGM(...))`.
  Euler-equation inversion on an exogenous savings grid with a fast
  upper-envelope scan (Dobrescu & Shanker 2022) — no consumption grid enters
  the solve, and the credit-constrained segment is exact. Requires declared
  `resources`, post-decision, and `inverse_marginal_utility` regime functions;
  the model contract is validated at `Model` construction. Supports discrete
  states and actions, EV1 taste shocks, stochastic processes, and passive
  continuous states. Forward simulation works with grid-restricted consumption
  (the intrinsic budget constraint is applied as a feasibility mask).

- Adds regime-level EV1 taste shocks as a model property:
  `Regime(taste_shocks=lcm.ExtremeValueTasteShocks())` with the scale as the
  runtime param `{"taste_shocks": {"scale": ...}}`. The solve aggregates
  discrete actions by the smoothed expected maximum and simulation draws the
  discrete action by Gumbel-max — identical solutions under either solver.

- Promotes the Iskhakov et al. (2017) retirement model to
  `lcm_examples.iskhakov_et_al_2017` (brute-force and DC-EGM variants) with an
  explanation notebook comparing the two solvers.

## 0.0.1

### Initial Release

- First public release of PyLCM.

- Includes core functionality:

    - Specification of finite-horizon discrete-continuous choice models with an
       arbitrary number of discrete and continuous states and actions.

    - Linearly and Log-linearly spaced grids that approximate continuous states and
      actions.

    - Linear interpolation and extrapolation of the value function for continuous
       states.

    - Grid search (brute-force) for finding the optimal continuous policy.

    - Stochastic state transitions for discrete states which may depend on other
      discrete states and actions.

- Built with contributions from the PyLCM team.


### Contributions

Thanks to everyone who contributed to this release:

- {ghuser}`hmgaudecker`

  Initiated and drove the development agenda for PyLCM, ensuring strategic direction
  and alignment. He actively steered the project, facilitated collaboration, and secured
  funding to support core development. Additionally, he reviewed pull requests and
  provided feedback on the internal and external code structure and design.

- {ghuser}`janosg`

  Designed and implemented the initial prototype of PyLCM, laying the foundation for its
  development. He onboarded {ghuser}`timmens` and played a key role in shaping the
  project's direction. After stepping back from active development, he contributed to
  implementation discussions and later provided guidance on architectural decisions.

- {ghuser}`timmens`

  Took over development of PyLCM, expanding its functionality with key features like
  the simulation function, extrapolation capabilities, and special arguments. He led
  extensive refactoring to improve code clarity, maintainability, and testability,
  making the package easier to develop and extend. His contributions also include
  improved documentation, type annotations, static type checking, and the introduction
  of example and explanation notebooks.

- {ghuser}`mj023`

  Analyzed and optimized PyLCM's performance on the GPU, profiling execution and
  examining the computational graph of JAX-compiled functions. He fine-tuned the `solve`
  function's just-in-time compilation to reduce runtime and improve efficiency.
  Additionally, he compared PyLCM's performance against similar libraries, providing
  insights into its computational efficiency.

- {ghuser}`mo2561057`

  Added tests for the model processing and fully discrete models.

- {ghuser}`MImmesberger`

  Added checks to test PyLCM's results against analytical solutions.

#### Early contributors

- {ghuser}`segsell`

- {ghuser}`ChristianZimpelmann`

- {ghuser}`tobiasraabe`
