# Changes


This is a record of all past PyLCM releases and what went into them in reverse
chronological order. We follow [semantic versioning](https://semver.org/).


## Unreleased

### Phase grammar, cross-regime transitions, and model-level regime slots

- `Phased(solve=..., simulate=...)` gives any regime-slot value a per-phase
  variant; a bare value broadcasts to both phases. Carried states —
  `Phased(solve=callable, simulate=Grid)` in `states` — are derived functions
  during backward induction and genuine seeded-and-evolved states in
  simulation. See the [phase grammar](docs/explanations/phase_grammar.ipynb)
  explanation.

- `fixed_transition(state_name)` marks a fixed state (identity law) in
  `state_transitions`. The `None` spelling for fixed states is removed; a
  regime-level `None` now masks a model-level entry instead.

- Regime transitions take a third form: a per-target dict
  `{target_regime: MarkovTransition(prob_func)}` whose key set declares the regime's
  reachable targets — omitted regimes are structurally unreachable. Per-target
  dicts in `state_transitions` hand state values across regime boundaries,
  including into states the source regime does not carry and across grids that
  differ between regimes.

- Model-level regime slots: `Model(functions=..., constraints=..., states=...,
  state_transitions=..., actions=...)` declares shared structure once and
  merges it into every regime under the exactly-one-level rule. Broadcast
  states and actions are pruned per regime by DAG reachability;
  `model.pruned_variables` records the result.

- `model.user_regimes` holds plain `lcm.regime.Regime` instances, finalized at
  model build (model-level slots merged, default `H` injected, completeness
  validated).

### Per-target parameters

- Per-target transition parameters nest under the target regime's name in the
  params template — `template[regime][target][func][param]` — replacing the
  `to_<target>_…` spelling. Param qnames parallel engine function qnames.

- Parameters resolve at four levels, most to least specific: target / function
  (one value broadcasts over the law's targets) / regime / model. Exactly one
  level per parameter; multi-level specifications are ambiguity errors.

- Canonical flat params always key transition-law params per target, every
  target of a broadcast value sharing one leaf object. A coarse regime
  transition is evaluated once and shared, so it takes no per-target
  parameters.

- Model-level `derived_categoricals` follow the exactly-one-level rule of the
  other model-level slots: a name declared at model level and regime level is
  an ambiguity error, also when the grids match.


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
