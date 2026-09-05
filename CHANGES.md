# Changes

This is a record of all past PyLCM releases and what went into them in reverse
chronological order. We follow [semantic versioning](https://semver.org/).

## Unreleased

### Memory-aware action-width policy and solve/simulate GPU-memory attribution

- `ExecutionConfig(device_memory_bytes=...)`, passed to `model.solve(...)` or to an
  auto-solving `model.simulate(...)`, declares a per-device ceiling for the
  compiler-reported peak workspace of every compiled solve core. The planner walks a
  deterministic width frontier for each streamed action product widest-first, reads
  each executable's compiler memory report without running it, and dispatches the
  first candidate that fits, so a core whose full extent fits costs no extra
  compilation; when no candidate fits, `ExecutionPlanningError` is raised before
  backward induction. Without a budget the full action product is used.
- `GridSearch(action_block_width=...)` fixes the streamed action-block width of a
  regime's eligible solve cores. It is an execution request — rejected at model build on
  deliberately dense, unsupported, or trivial-product routes — and enters neither the
  model nor the parameter fingerprint.
- Period captures record the selected tile widths, so `replay_period` and the
  compiler-memory analyzer lower exactly the executable the solve dispatched.
- The ASV GPU-memory series for the ACA baseline and Mahler–Yum rows are split into
  three independently measured phases — automatic solve+simulate,
  `ALL_PERSISTABLE_ARTIFACTS` solve+save, and load+supplied-solution simulate — each in
  a fresh, phase-isolated child process with exact provenance. The combined
  timing/CPU subprocess no longer reports a GPU peak.

### Complete solution persistence and executable external replay

- `save_solution(solution=..., path=...)` and `SolutionResult.save(path=...)` atomically
  persist a complete labelled solution. `load_solution(path=...)` restores independently
  lazy values and artifacts; each numerical leaf is checksummed together with its
  logical address, shape, and dtype. The versioned archive contains JSON metadata and
  numerical datasets only — no model, plugin class, callable, pickle, or executable
  code. The old bare-mapping writer signature is removed;
  `load_legacy_solution(path=...)` remains the explicit reader for old value-only HDF5
  files.
- `SolutionMetadata` now carries a durable mathematical model fingerprint plus exact
  solver, replay-route, artifact, solution-schema, and archive-format identities. A
  restored result can replay in a separately constructed compatible model process; an
  in-memory result retains its additional same-instance guard. Compatibility is exact,
  and incompatible versions fail clearly rather than migrating implicitly.
- Persistence is selected per model-built artifact authority. Static or otherwise
  independently model-verifiable artifacts are saved, while a present artifact whose
  authority declares `NOT_PERSISTED` becomes an explicit omission in the restored
  result. Adaptive NNBEGM policy meshes remain in that category until the model can
  independently rederive their solve-generated coordinates.
- Retention now selects computation per exact artifact address. Replay alternatives and
  additive artifact programs declare `CoreProgram.retained_artifact_keys`; DCEGM,
  NB-EGM, NNBEGM, and external solvers avoid assembling outputs that the selected
  retention will discard.
- `ValueStore` and `ArtifactStore` expose per-entry `LoadState`. Metadata, omissions,
  coordinates, and whole-archive checksum verification do not materialize payloads;
  value-only inspection therefore does not load replay banks.
- External solvers can declare an `ExecutableReplayRoute` with durable plugin and route
  identities, period-specific artifact requirements, model-built artifact authorities,
  a mathematical preflight, and a JAX-transformable reader returning named
  `ActionOutput`. The engine canonicalizes required restored or caller-supplied entries
  once and passes the same immutable snapshot through validation, reader construction,
  and forward execution.
- An in-repository out-of-tree reference solver imports only public modules. It
  exercises a planner-owned program, a non-EGM continuation, a persistable
  plugin-defined replay PyTree, explicit non-persisted artifacts, lazy restoration into
  a freshly constructed model, custom replay, and fail-closed rejection of invalid
  artifacts.

### Every built-in kernel on the public execution contract

- Every shipped period kernel — plain EGM, DC-EGM, NEGM, NB-EGM, NNBEGM (finite and
  adaptive), grid search, and the engine's terminal-carry wrapper — publishes a native
  core-program graph and returns a `KernelOutput`. The centralized legacy adapter is
  gone, and with it `require_legacy_kernel_result`, `normalize_kernel_output`,
  `KernelResult`, the `UNPLANNED` layout, and the post-hoc repair that moved a kernel's
  published value onto its template's placement. A value now lands in its declared
  placement or the solve fails.
- NEGM's outer sweep over the durable margin is one compiled program driven by
  `jax.lax.map` rather than a Python chunk loop, with `outer_batch_size` selecting the
  block width. Solved values are invariant to that width to within a few units in the
  last place; the support of the compiled DC-EGM adjuster at float32 is not, because
  each width is a separate XLA kernel, so support identity is asserted under float64
  only.
- NB-EGM reads its continuation once per branch equivalence class rather than once per
  discrete action, and a solve retaining only values never compiles or lowers the replay
  program.

### A public contract for out-of-tree solvers

- A solver can be written against `lcm.solvers`, `lcm.solver_api`, `lcm.typing` and
  `lcm.grids` alone. Those modules now export the execution-contract types a solver
  constructs (`CoreProgram`, `CoreBuildContext`, `CoreExecutionRequirements`,
  `CoreExecutionDisposition`, `ProgramScope`, `StreamableProductAxis`,
  `ReductionSemantics`, `OutputRole`, `StateAxesLeading`, `PeriodKernel`,
  `StateActionSpace`), the continuation types and helpers (`ContinuationSpec`,
  `EGMContinuationSpec`, `EGMContinuationLayout`, `ContinuationArtifact`,
  `period_to_continuation_target`, `target_period_grid`, `union_free_params`,
  `union_fixed_params`), the parameter aliases (`FlatParams`, `FlatRegimeParams`,
  `EconFunction`, `EconFunctionsMapping`), and `ContinuousGrid`. The exact-version
  persistence, replay, and conformance contract is documented in
  [Custom solvers](reference/custom_solvers.md).
- `Solver.requires_continuation` is replaced by `Solver.required_continuation_keys`, a
  frozenset of `ArtifactKey`. Model building matches every declared key against what
  each reachable target publishes and refuses the model, naming both regimes and the
  demanded version, before anything compiles.
- The rolling continuation channel is keyed rather than concrete. A period kernel may
  publish any payload satisfying the `ContinuationArtifact` protocol under its own
  versioned key, and the engine stores and rolls it without reading its fields.
- Every regime declares one replay route, `SimulationPhase.replay_route`, carrying a
  `ReplayMode` of `EXACT_REPLAY`, `VALID_RECOMPUTATION`, or `UNSUPPORTED`, the exact
  payload class it retains, and the reader that consumes it. External routes now supply
  their own model-verifiable authorities, preflight validator, and JAX-transformable
  reader. Forward simulation dispatches on that declaration instead of on the class of
  whatever payload a solve happened to keep.

### Tile-local NB-EGM ride-along execution

- A regime carrying ride-along co-states solves each period in one tile-local `NBEGM`
  core: every cell block's transition-aware continuation read (the complete expectation
  over reachable targets and stochastic nodes, on the savings grid) is consumed by that
  block's envelope solve inside the same compiled body, so the expected-continuation
  stacks over every cell are never a complete array and never a core argument. The
  period kernel publishes a native two-program graph with planned outputs: `main` for a
  values-only solve (the value array and the carry) and `replay` for a solve retaining
  replay artifacts (adding the consumption policy and the conditional branch banks). A
  direct scalar oracle in the test suite, independent of the production expectation and
  envelope code, replaces the split continuation and envelope cores; the compile-only
  fused replay experiment is replaced by a per-period core memory analyzer that lowers
  the production programs.

### Gated edges into targets with disjoint activity windows

- Simulation reads a gated edge's gate references and leg fallbacks only in the periods
  where that edge's target is active. A regime declaring two edges whose targets are
  active over disjoint age windows no longer fails with a `KeyError` for a fallback
  regime the landing period never solved (#434). The per-period reference set is
  recorded on the canonical regime's simulation phase, and both the ahead-of-time
  lowering and the runtime call consult it.

### PR #433 execution planning and result convergence

- `Model.solve()` returns a `SolutionResult` and is the only public solve entry point.
  `solve_result()`, mapping and tuple returns, `return_simulation_policy`, and
  `return_dissolution_flags` are removed; `simulate(solution=...)` consumes the complete
  result, and omitting `solution` solves automatically. The result carries values,
  replay artifacts, metadata, and explicit omission reasons, and the model that produced
  it authenticates every value and replay cell on simulation.
- One native `CoreProgram` graph per `GridSearch` period kernel is the sole authority
  for each core's function, argument builder, execution requirements, value reads,
  output roles, execution disposition, and reason. Eager, JIT, ahead-of-time, liveness,
  output-layout, and period-replay paths resolve the same graph through one seam;
  unmigrated endogenous-grid kernels cross one fail-closed legacy adapter. A kernel
  publishing both a native graph and a legacy declaration is rejected at build.
- Streamed action reduction is planned per program with an explicit disposition and
  reason. Singleton expected-value (EV1) reductions and collective hard-max reductions
  stay dense by disposition: blockwise grouping changes the canonical floating-point
  reduction order, and the collective streamed row regressed every measured resource
  surface. The transition ledger in `docs/development/architecture_transition_ledger.md`
  records each remaining bridge with its retirement condition.
- The fp32/fp64 candidate certificate binds the native graph: seals, AST hashes, and 354
  synchronized mutations cover duplicate authority, wrapped or rebound functions and
  builders, erased requirements, roles, and reasons, forced dispositions, and
  eager/AOT/replay bypasses.
- The backward-induction diagnostics fold combines per-period scalars across value
  arrays with different device placements: a planned single-device layout is committed,
  and a mesh-sharded neighbour is moved onto the running flag's placement before it is
  combined.
- A gated edge into a stateless target is gated in solve: the source's continuation
  applies the gate and the leg fallbacks to the target's folded channel stack, so a
  closed gate pays the projected fallback. Previously the dense action reduction
  silently reduced over the channel axis and the source always paid the target's own
  value.

### PR #390 maintainer-review follow-up

- `ConsumptionSavingsRegime` and `NestedConsumptionSavingsRegime` declare the DAG roles
  the endogenous-grid solvers need — the liquid state, consumption action, resources
  node, and post-decision node, plus an outer continuous margin for the nested form —
  while retaining an arbitrary regime function DAG. `EGM`, `DCEGM`, and `NEGM` carry
  numerical configuration only and read their role names from the regime that binds
  them, so an endogenous-grid solver on a plain `Regime` is rejected at construction.
  `Regime` with `GridSearch` is unchanged.
- Every solver now participates in model-stage and build-stage validation through the
  common solver contract. Plain `EGM` rejects constraints, discrete/process axes,
  incompatible terminal targets, and a post-decision function that is not resources
  minus the continuous action before solving.
- DC-EGM semantic validation is independent of native exact-kernel presence.
  Exact-backend capability is checked only after the regime satisfies the model
  contract; exact-only tests declare that requirement explicitly, and CI records their
  node IDs and skip reasons on kernel-less platforms.
- Cross-regime endogenous-grid continuation calls preserve target-regime parameter
  identity for both runtime and fixed parameters.
- Upper-envelope selection uses typed backend configurations (`ExactEnvelope`,
  `FUESEnvelope`, `RFCEnvelope`, `LTMEnvelope`, and `MSSEnvelope`). Selecting the exact
  backend requires a loadable native kernel during model construction.
- EGM continuation templates and their static layouts are bundled in one
  `EGMContinuationSpec`, and GridSearch/terminal carries are published only for
  reachable targets that have an incoming endogenous-grid consumer.
- Simulation-policy host copies and retention are demand-driven. When an off-grid policy
  replacement is accepted, the reported value is the canonical value attained by that
  same emitted action.
- NEGM locates its durable carry axis by name rather than declaration order and checks
  utility separability through the complete composed utility DAG.
- Negative Euler targets fail loudly, numerical marginal-utility inversion expands its
  initial bracket, and ordinary one-dimensional continuation reads use nearest-segment
  extrapolation outside support.

### Fixes

- The marginal a brute (`GridSearch`) child publishes to an endogenous-grid parent is
  one-sided next to a feasibility boundary. A central difference straddling an
  infeasible state said nothing, so the first feasible state above a borrowing
  constraint carried a zero marginal and biased the parent's Euler inversion toward
  over-consumption there.

- A constraint reading an auto-named `next_<state>` (the NEGM budget cut on the next
  durable stock) no longer breaks `simulate()`. The initial-conditions feasibility
  check, its per-constraint diagnostic, and the additional-target pool now resolve that
  name the way the within-period decision does.

- `envelope="mss"`: a value decrease no larger than rounding noise is no longer read as
  a branch boundary. Along a near-linear tail the sign of the difference between
  consecutive candidate values is set by rounding, and splitting there silently dropped
  the top of the published row. A candidate whose value is not finite now costs only its
  own nodes instead of poisoning every node it covers with NaN.

- `envelope="exact"`: a handover between two links the pair's arithmetic cannot separate
  is refused rather than placed at a fabricated abscissa.

- An endogenous-grid regime reads its continuation on the grid the *target* regime
  tabulates at period `t+1`, not on its own period-`t` grid. The two differ whenever the
  target is a different regime whose grid differs, or whenever an `AgeSpecializedGrid`
  moves the nodes with age; before, such a model either raised on the length mismatch or
  published values inverted against the wrong abscissae. A target that does not carry
  the state now raises instead of falling back.

- `EGM` and `TwoAssetEGM` validate their regime when the model is built: the number of
  continuous states, the declared roles, and — for the two-asset solver — the retirement
  boundary target. The errors name the regime's own state names and the field that fixes
  them.

### `EGM` solves the law the regime declares

- `EGM` reads the two quantities the Euler inversion needs — where a level of savings
  lands next period, and how that landing point moves when savings move — off the
  regime's own transition, composed through the post-decision node and differentiated
  there. It previously rebuilt `(1 + r) * savings + income` from two parameters resolved
  by name, so every term the modeller declared outside that form was silently discarded
  and the solver published a policy for a model its user had not written. A per-period
  fixed cost, a means test, or a balance-dependent return now reaches the inversion like
  any other term.

- **Breaking:** `return_param` and `income_param` are gone. `EGM` takes
  `post_decision_function=` in their place, naming the function in `Regime.functions`
  that computes the end-of-period balance the liquid state's transition is written
  through. A model that renames every parameter in its laws now solves with no
  solver-side declaration at all.

- **Breaking:** the borrowing corner is the savings grid's lower bound rather than zero,
  so a household allowed to borrow is no longer solved as one that is not. A grid
  starting at zero reproduces the previous arithmetic exactly.

- Two laws are now refused instead of solved wrongly. A law reaching a state or action
  other than through the post-decision node is not a function of savings, so neither
  reading exists; it is named at model build rather than failing deep inside `dags`. A
  law whose landing points do not ascend strictly with savings breaks the interpolation
  back onto the regular grid, which returns quietly wrong numbers rather than raising —
  a falling law is now told it falls, a flat one that it is flat.

### Solver naming

- The endogenous-grid solvers are named by the problem they solve rather than by the
  dimension count they were built around: `OneAssetEGM` is now `EGM` and `TwoDimEGM` is
  now `TwoAssetEGM`. There are no aliases.

- `TwoAssetEGM` takes the regime's two continuous states by name — `liquid_state=` and
  `pension_state=` — instead of requiring them to be spelled `liquid` and `pension`.

- The refinement argument is `envelope=` on both `TwoAssetEGM` and `DCEGM` (was
  `upper_envelope=`). The `_lcm/egm/upper_envelope/` package keeps its name: it is the
  FUES backend, not the field.

### Platform support

- There is no `metal` / `tests-metal` pixi environment: macOS runs on CPU, and
  Apple-Silicon GPU acceleration is not installable from this project.

### Phase grammar, cross-regime transitions, and model-level regime slots

- `Phased(solve=..., simulate=...)` gives any regime-slot value a per-phase variant; a
  bare value broadcasts to both phases. Carried states —
  `Phased(solve=callable, simulate=Grid)` in `states` — are derived functions during
  backward induction and genuine seeded-and-evolved states in simulation. See the
  [phase grammar](docs/explanations/phase_grammar.ipynb) explanation.

- `fixed_transition(state_name)` marks a fixed state (identity law) in
  `state_transitions`. The `None` spelling for fixed states is removed; a regime-level
  `None` now masks a model-level entry instead.

- Regime transitions take a third form: a per-target dict
  `{target_regime: MarkovTransition(prob_func)}` whose key set declares the regime's
  reachable targets — omitted regimes are structurally unreachable. Per-target dicts in
  `state_transitions` hand state values across regime boundaries, including into states
  the source regime does not carry and across grids that differ between regimes.

- A bare callable or bare `MarkovTransition` on `Regime.transition` declares
  conservative support over every regime active in the next period, so every temporally
  compatible candidate must have a valid state handoff (a carried state, a
  deterministic/stochastic law, or an explicit target-local/entry law). Use a per-target
  mapping to declare narrower support instead. Runtime transition probabilities of zero
  do not narrow this topology — only the declared form does.

- Model-level regime slots:
  `Model(functions=..., constraints=..., states=..., state_transitions=..., actions=...)`
  declares shared structure once and merges it into every regime under the
  exactly-one-level rule. Broadcast states and actions are pruned per regime by DAG
  reachability; `model.pruned_variables` records the result.

- `model.user_regimes` holds plain `lcm.regime.Regime` instances, finalized at model
  build (model-level slots merged, the model-level Koopmans aggregator and certainty
  equivalent injected into non-terminal regimes, completeness validated).

### Per-target parameters

- Per-target transition parameters nest under the target regime's name in the params
  template — `template[regime][target][func][param]` — replacing the `to_<target>_…`
  spelling. Param qnames parallel engine function qnames.

- Parameters resolve at four levels, most to least specific: target / function (one
  value broadcasts over the law's targets) / regime / model. Exactly one level per
  parameter; multi-level specifications are ambiguity errors.

- Canonical flat params always key transition-law params per target, every target of a
  broadcast value sharing one leaf object. A coarse regime transition is evaluated once
  and shared, so it takes no per-target parameters.

- Model-level `derived_categoricals` follow the exactly-one-level rule of the other
  model-level slots: a name declared at model level and regime level is an ambiguity
  error, also when the grids match.

### State-conditioned stochastic processes

- A continuous stochastic process may condition its `sigma` on a discrete regime state
  via `sigma=StateConditioned(on="<discrete state>", by={<category>: sigma})`. The
  declaration stands where the scalar would, so which parameter is conditioned is
  explicit and there is no way to give that parameter twice.

  Every category shares one set of nodes, placed from the widest value in `by` — the
  narrowest axis that still covers all of them. The per-category values move no nodes;
  each row is evaluated directly at the from-value with the value for the time-$t$
  category, with no precomputed-row interpolation. This expresses regime-switching
  income risk and stochastic volatility.

  Supported for the CDF-binned `NormalIIDProcess` (`gauss_hermite=False`) and
  `TauchenAR1Process`, whose transition probabilities carry `sigma`. Gauss-Hermite node
  placement and Rouwenhorst are refused when the model is built, their fixed-node
  kernels having no channel to carry it. A `by` whose values are not all finite and
  positive is refused at construction.

  Solving and simulating use the same conditioned law. Every grid parameter must be
  fixed at construction, and the conditioning state must map its categories to the same
  integer codes in every regime that carries it. Current-regime conditioning only. See
  `lcm_examples/stochastic_volatility.py`.

### Perceived versus realized transitions in simulation

- A simulated agent prices its continuation under the law it *believes*, while the world
  it moves through follows the law that is *true*. `Phased` state transitions accept
  `MarkovTransition` laws, so perceived mortality, perceived health or income risk, and
  misread policy rules are expressible: give the `solve` variant the agent's beliefs and
  the `simulate` variant the data-generating process. See the phase-grammar explanation
  in the docs.

  The simulated state-action value is assembled from two halves. Today's payoff and
  feasible set — period utility, constraints, the Koopmans aggregator — come from the
  simulate phase, because they are known when the action is chosen. The continuation —
  next-period kernels, regime-transition probabilities, and every helper they read —
  comes from the solve phase, because the future is only perceived and the value
  function was solved under those beliefs. The realized draw is unchanged and still
  follows the simulate laws.

  The two phases need not agree on whether a law is stochastic: a deterministic law is a
  degenerate kernel, so an agent may perceive risk where there is none, or treat as
  certain a transition that is not.

  Constraints must be phase-invariant through their whole dependency chain. A constraint
  that reaches a `Phased` helper or law of motion is rejected when the model is built,
  because a phase-specific feasible set would let the simulated agent choose actions its
  value function was never computed for.

  **Behaviour change.** A model's numbers are unchanged unless a `Phased` function lies
  in the dependency ancestry of a continuation transition, or of a `next_<state>` read
  by period utility or feasibility. The solve phase is untouched, and for every
  phase-invariant name both phases hold the same function. Models that do have such a
  dependency change by design: the helper was resolved from the wrong phase. That
  pattern was reachable before this release through a `Phased` helper under a
  phase-invariant law, so the correction can move published results.

  Both variants of a `Phased` stochastic law are validated numerically; before, only one
  of them was. A per-target dict inside `Phased` must be per-target in both phases and
  cover the same targets.

### Discrete-continuous choice: DC-EGM, NEGM, and taste shocks

- Adds the DC-EGM solver (Iskhakov, Jørgensen, Rust & Schjerning 2017) as a per-regime
  alternative to grid search: `Regime(solver=lcm.DCEGM(...))`. Euler-equation inversion
  on an exogenous savings grid with a fast upper-envelope scan (Dobrescu & Shanker 2022)
  — no consumption grid enters the solve, and the credit-constrained segment is exact.
  Requires declared `resources`, post-decision, and `inverse_marginal_utility` regime
  functions; the model contract is validated at `Model` construction. Supports discrete
  states and actions, EV1 taste shocks, stochastic processes, and passive continuous
  states. Forward simulation works with grid-restricted consumption (the intrinsic
  budget constraint is applied as a feasibility mask).

- Adds regime-level EV1 taste shocks as a model property:
  `Regime(taste_shocks=lcm.ExtremeValueTasteShocks())` with the scale as the runtime
  param `{"taste_shocks": {"scale": ...}}`. The solve aggregates discrete actions by the
  smoothed expected maximum and simulation draws the discrete action by Gumbel-max —
  identical solutions under either solver.

- Promotes the Iskhakov et al. (2017) retirement model to
  `lcm_examples.iskhakov_et_al_2017` (brute-force and DC-EGM variants) with an
  explanation notebook comparing the two solvers.

## 0.0.1

### Initial Release

- First public release of PyLCM.

- Includes core functionality:

  - Specification of finite-horizon discrete-continuous choice models with an arbitrary
    number of discrete and continuous states and actions.

  - Linearly and Log-linearly spaced grids that approximate continuous states and
    actions.

  - Linear interpolation and extrapolation of the value function for continuous states.

  - Grid search (brute-force) for finding the optimal continuous policy.

  - Stochastic state transitions for discrete states which may depend on other discrete
    states and actions.

- Built with contributions from the PyLCM team.

### Contributions

Thanks to everyone who contributed to this release:

- {ghuser}`hmgaudecker`

  Initiated and drove the development agenda for PyLCM, ensuring strategic direction and
  alignment. He actively steered the project, facilitated collaboration, and secured
  funding to support core development. Additionally, he reviewed pull requests and
  provided feedback on the internal and external code structure and design.

- {ghuser}`janosg`

  Designed and implemented the initial prototype of PyLCM, laying the foundation for its
  development. He onboarded {ghuser}`timmens` and played a key role in shaping the
  project's direction. After stepping back from active development, he contributed to
  implementation discussions and later provided guidance on architectural decisions.

- {ghuser}`timmens`

  Took over development of PyLCM, expanding its functionality with key features like the
  simulation function, extrapolation capabilities, and special arguments. He led
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
