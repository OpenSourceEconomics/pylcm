# Changes


This is a record of all past PyLCM releases and what went into them in reverse
chronological order. We follow [semantic versioning](https://semver.org/).


## Unreleased

### PR #390 maintainer-review follow-up

- `ConsumptionSavingsRegime` and `NestedConsumptionSavingsRegime` declare the
  DAG roles the endogenous-grid solvers need — the liquid state, consumption
  action, resources node, and post-decision node, plus an outer continuous
  margin for the nested form — while retaining an arbitrary regime function
  DAG. `EGM`, `DCEGM`, and `NEGM` carry numerical configuration only and read
  their role names from the regime that binds them, so an endogenous-grid
  solver on a plain `Regime` is rejected at construction. `Regime` with
  `GridSearch` is unchanged.
- Every solver now participates in model-stage and build-stage validation
  through the common solver contract. Plain `EGM` rejects constraints,
  discrete/process axes, incompatible terminal targets, and a post-decision
  function that is not resources minus the continuous action before solving.
- DC-EGM semantic validation is independent of native exact-kernel presence.
  Exact-backend capability is checked only after the regime satisfies the model
  contract; exact-only tests declare that requirement explicitly, and CI records
  their node IDs and skip reasons on kernel-less platforms.
- Cross-regime endogenous-grid continuation calls preserve target-regime
  parameter identity for both runtime and fixed parameters.
- Upper-envelope selection uses typed backend configurations
  (`ExactEnvelope`, `FUESEnvelope`, `RFCEnvelope`, `LTMEnvelope`, and
  `MSSEnvelope`). Selecting the exact backend requires a loadable native kernel
  during model construction.
- EGM continuation templates and their static layouts are bundled in one
  `EGMContinuationSpec`, and GridSearch/terminal carries are published only for
  reachable targets that have an incoming endogenous-grid consumer.
- Simulation-policy host copies and retention are demand-driven. When an
  off-grid policy replacement is accepted, the reported value is the canonical
  value attained by that same emitted action.
- NEGM locates its durable carry axis by name rather than declaration order and
  checks utility separability through the complete composed utility DAG.
- Negative Euler targets fail loudly, numerical marginal-utility inversion
  expands its initial bracket, and ordinary one-dimensional continuation reads
  use nearest-segment extrapolation outside support.

### Fixes

- The marginal a brute (`GridSearch`) child publishes to an endogenous-grid
  parent is one-sided next to a feasibility boundary. A central difference
  straddling an infeasible state said nothing, so the first feasible state above
  a borrowing constraint carried a zero marginal and biased the parent's Euler
  inversion toward over-consumption there.

- A constraint reading an auto-named `next_<state>` (the NEGM budget cut on the
  next durable stock) no longer breaks `simulate()`. The initial-conditions
  feasibility check, its per-constraint diagnostic, and the additional-target
  pool now resolve that name the way the within-period decision does.

- `envelope="mss"`: a value decrease no larger than rounding noise is no
  longer read as a branch boundary. Along a near-linear tail the sign of the
  difference between consecutive candidate values is set by rounding, and
  splitting there silently dropped the top of the published row. A candidate
  whose value is not finite now costs only its own nodes instead of poisoning
  every node it covers with NaN.

- `envelope="exact"`: a handover between two links the pair's arithmetic
  cannot separate is refused rather than placed at a fabricated abscissa.

- An endogenous-grid regime reads its continuation on the grid the *target*
  regime tabulates at period `t+1`, not on its own period-`t` grid. The two
  differ whenever the target is a different regime whose grid differs, or
  whenever an `AgeSpecializedGrid` moves the nodes with age; before, such a
  model either raised on the length mismatch or published values inverted
  against the wrong abscissae. A target that does not carry the state now
  raises instead of falling back.

- `EGM` and `TwoAssetEGM` validate their regime when the model is built:
  the number of continuous states, the declared roles, and — for the
  two-asset solver — the retirement boundary target. The errors name the
  regime's own state names and the field that fixes them.

### `EGM` solves the law the regime declares

- `EGM` reads the two quantities the Euler inversion needs — where a level of
  savings lands next period, and how that landing point moves when savings move
  — off the regime's own transition, composed through the post-decision node
  and differentiated there. It previously rebuilt `(1 + r) * savings + income`
  from two parameters resolved by name, so every term the modeller declared
  outside that form was silently discarded and the solver published a policy
  for a model its user had not written. A per-period fixed cost, a means test,
  or a balance-dependent return now reaches the inversion like any other term.

- **Breaking:** `return_param` and `income_param` are gone. `EGM` takes
  `post_decision_function=` in their place, naming the function in
  `Regime.functions` that computes the end-of-period balance the liquid state's
  transition is written through. A model that renames every parameter in its
  laws now solves with no solver-side declaration at all.

- **Breaking:** the borrowing corner is the savings grid's lower bound rather
  than zero, so a household allowed to borrow is no longer solved as one that
  is not. A grid starting at zero reproduces the previous arithmetic exactly.

- Two laws are now refused instead of solved wrongly. A law reaching a state or
  action other than through the post-decision node is not a function of savings,
  so neither reading exists; it is named at model build rather than failing deep
  inside `dags`. A law whose landing points do not ascend strictly with savings
  breaks the interpolation back onto the regular grid, which returns quietly
  wrong numbers rather than raising — a falling law is now told it falls, a flat
  one that it is flat.

### Solver naming

- The endogenous-grid solvers are named by the problem they solve rather than
  by the dimension count they were built around: `OneAssetEGM` is now `EGM`
  and `TwoDimEGM` is now `TwoAssetEGM`. There are no aliases.

- `TwoAssetEGM` takes the regime's two continuous states by name —
  `liquid_state=` and `pension_state=` — instead of requiring them to be
  spelled `liquid` and `pension`.

- The refinement argument is `envelope=` on both `TwoAssetEGM` and `DCEGM`
  (was `upper_envelope=`). The `_lcm/egm/upper_envelope/` package keeps its
  name: it is the FUES backend, not the field.

### Platform support

- There is no `metal` / `tests-metal` pixi environment: macOS runs on CPU, and
  Apple-Silicon GPU acceleration is not installable from this project.

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

- A bare callable or bare `MarkovTransition` on `Regime.transition` declares
  conservative support over every regime active in the next period, so every
  temporally compatible candidate must have a valid state handoff (a carried
  state, a deterministic/stochastic law, or an explicit target-local/entry
  law). Use a per-target mapping to declare narrower support instead. Runtime
  transition probabilities of zero do not narrow this topology — only the
  declared form does.

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

### Discrete-continuous choice: DC-EGM, NEGM, and taste shocks

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
