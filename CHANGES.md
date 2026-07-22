# Changes


This is a record of all past PyLCM releases and what went into them in reverse
chronological order. We follow [semantic versioning](https://semver.org/).


## Unreleased

- A continuous stochastic process may condition its `sigma` on a discrete regime
  state via `sigma=StateConditioned(on="<discrete state>", by={<category>: sigma})`.
  The scalar `sigma` field then defines a FIXED common node grid; each regime's
  transition row is evaluated directly at the from-value with that regime's `sigma`
  (no precomputed-row interpolation). This expresses regime-switching income risk /
  stochastic volatility. Supported for CDF-binned `NormalIIDProcess`
  (`gauss_hermite=False`) and `TauchenAR1Process`; Gauss-Hermite node placement and
  Rouwenhorst are rejected at construction (their fixed-node kernels cannot carry a
  state-conditioned `sigma`). Solving and simulating use the same conditioned law.
  Every grid parameter must be fixed at construction, and the conditioning state must
  map its categories to the same integer codes in every regime that carries it.
  Current-regime conditioning only. See `lcm_examples/stochastic_volatility.py`.

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
