# Changes

This is a record of all past PyLCM releases and what went into them in reverse
chronological order. We follow [semantic versioning](https://semver.org/).

## Unreleased

### Explicit initial nodes and owned declarations

- Declare admissible starts with `InitialNodes(by_age={25: "working"})` or
  `InitialNodes(by_period={0: "working"})`, matching the model's clock. Exactly one
  nonempty mapping is required. This replaces the singular `InitialNode` interface.
  The model returns a normalized immutable `InitialNodes` mapping exact coordinates
  to sorted, unique regime tuples. Code unpacking `model.initial_nodes` must use
  `model.graph.initial_nodes` for expanded pairs. Legacy pair collections and bare
  selector mappings remain accepted in age models; period models require `by_period`.
- Edge mappings and nested transition-law mappings are copied and frozen at
  construction, including mappings in `ByAge`, `ByPeriod`, and `Phased`. Reusing caller dictionaries
  cannot change a model's published configuration; callable identity is preserved.

### Gates are declared beside the transition law; derived `Transition.targets`

- Breaking API: `ValueDependentTransition` is removed. A value-dependent destination is
  declared as `Transition(law=..., gates={target: Gate(...)})`. The law supplies the
  gated target's probability like any other per-target cell, and `Gate` carries the
  rest: `predicate` (formerly `gate`), `routes`, `references` (formerly
  `gate_references`) and `off_grid`. A law cell
  `tgt: ValueDependentTransition(probability=p, gate=g, routes=r, gate_references=refs)`
  becomes the law cell `tgt: p` plus `gates={tgt: Gate(predicate=g, routes=r,
  references=refs)}`.
- One `Gate` per target holds at every age the target is reached and in both phases; a
  gate is never wrapped in `ByAge` or `Phased`, and only a route's `fallback` may be
  `Phased`. With `Phased` edges, a target reached in both phases carries the equal
  `Gate` in both or none. A gate on a target its `Transition` never reaches is refused.
- Breaking API: the parameter paths of a gated target follow the declaration.
  `params["edges"][source][target]["probability"][arg]` becomes
  `params["edges"][source][target][arg]`, `["gate"][arg]` becomes
  `["predicate"][arg]`, and `["gate_references"][reference][state][arg]` becomes
  `["references"][reference][state][arg]`; route fallback paths are unchanged.
- `Transition.targets` is optional when the law names its targets — a per-target
  mapping, a regime name, or a `ByAge` / `Phased` of those. The destinations are
  derived from the law: each key at the non-final source ages its case covers, and each
  gate's route fallback regimes at the ages of the gated target. Supplied anyway,
  `targets` must equal the derived mapping exactly, or `Model(...)` raises a
  `ModelInitializationError` listing both. A law over all targets — a function, a
  `DeterministicTransition` or a full-vector `StochasticTransition` — still requires
  `targets`. See
  [the migration guide](docs/user_guide/migrating_dated_regimes.md#migrating-gates).
- Breaking API: `lcm.collective` does not re-export `StochasticTransition`; import it
  from `lcm`.

### Public production period capture

- `Model.solve(period_capture=PeriodCapture(...))` atomically records selected
  ordinary GridSearch entry inputs and appends completed references. A fresh
  `Model.replay_period` binds model/grid/parameter/source identities and validates
  recorded layout, widths, optimized HLO and admission before dispatch. Entry-only
  captures remain inspectable and cannot claim reference parity. See
  [period capture](docs/reference/runtime_and_results.md#api-period-capture).

### Opt-in action-partitioned GridSearch

- `ExecutionConfig(action_partitions={"<regime>": n})` shares a `GridSearch` regime's
  action product over `n` devices. Each device reduces its own contiguous run of action
  blocks with the exact hard maximum, and the devices exchange and merge one compact
  accumulator per state cell instead of any value over actions. Ties, infinities, NaNs
  and signed zeros keep the ordinary route's conventions; at the same action width the
  tested workloads publish values bitwise equal to the ordinary route, alone, with a
  sharded continuous state and with type-local blocks. The default `{}` and a count of
  one leave the solve unchanged. Unsupported requests are refused at model
  construction. See [Share a large action product over
  devices](docs/user_guide/tuning.md).
### A shock read only through its next-period draw stays in the regime

- A law may read the draw `next_<process>` of a process state, for instance costs
  realized after the period's choices and paid out of next period's wealth. Reading the
  draw is now a read of the state: the regime that reads it keeps the state, whether the
  state is declared at model level (broadcast pruning) or at regime level (the
  unused-variable check). A persistent process's draw is conditional on the state's
  current node. An IID process's draw is not, but it is still taken from the carried
  state, so the state keeps its axis and the value is constant along it.
- Toward a target that does not carry the process, the draw exists only inside the
  transition: it is taken from the source's process at the source's current node, read
  by the target's laws and discarded. A terminal regime valuing only wealth therefore
  carries only wealth. `GridSearch`, `DCEGM` and `NBEGM` solve and simulate such
  edges.
- The same holds for a Markov state (a `DiscreteGrid` with a `StochasticTransition`
  law): reading its draw `next_<state>` keeps the state in the reading regime, and
  toward a target that does not carry it the draw is taken from the source's Markov
  law, with that law's parameters. This requires the law to be declared once for
  every target; a per-target law names no law for a target that lacks the state.
- When a regime law's probabilities do not sum to one because the law has a cell for
  a target that the graph gives no edge at that age, the error names each such
  `(age, source -> target)` cell and says the graph declares no edge for it.
- A state law that reads `next_<state>` toward a target that neither carries the state
  nor receives a draw of it on that edge is refused when the model is built, naming
  the source, the target, the law and the `next_` argument.
- `DCEGM` and `NBEGM` accept a liquid law that reads a draw, persisted or local to the
  edge: the Euler state and its savings derivative are evaluated at every node of the
  draws the law reads. `NBEGM`'s save-to-cliff targets are inverted per node. Under the
  EGM solvers only the liquid law may read a draw; any other law reading one is refused
  at construction.

### Opt-in type-local GridSearch

- `ExecutionConfig(invariant_block_widths={"<state>": 1})` solves every non-terminal
  `GridSearch` regime carrying an invariant discrete state one code at a time. Each code
  runs one shared executable with the code as a runtime operand, and reads every
  continuation carrying the state through that code's block. The tested workloads
  publish values, policies and simulated panels bitwise equal to the unblocked solve's;
  the default `{}` leaves the solve unchanged. Unsupported or unsafe requests are refused at model construction. See
  [Solve one invariant code at a time](docs/user_guide/tuning.md).
- When the simulate phase also keeps the state fixed, `simulate` groups subjects by their
  starting code and reads each typed value through one code's block. Every subject keeps
  its original random draws and output row, so the panel is unchanged; otherwise
  simulation stays ungrouped.
- `ExecutionConfig(invariant_block_schedule=InvariantBlockSchedule.BLOCK_MAJOR)` solves
  each code through all of its periods before the next code, simulates its subjects
  while its values are still on the device, then copies them to the host and deletes the
  device buffers, so the device holds one code's values at a time. Every code runs the
  programs the first compiled. The result is complete: each value is assembled from the
  retained codes when read, on the layout the period-major schedule publishes, saves
  from the host, and simulates one code at a time; `values.materialize()` is refused
  when every value cannot fit the device budget. Values and panels equal the default
  period-major schedule's byte for byte. Unsupported requests (no blocked state, a
  regime without it, budgeted or ungrouped simulation, `log_path`) are refused. See
  [Solve, simulate and release one code at a time](docs/user_guide/tuning.md).

### Phase-specific model graphs and transition declarations

- Ordinary per-target `Phased` laws may declare different solve and simulation
  destinations, with state handoffs local to each phase. Every physically visited
  node is valued, together with its recursively required perceived continuations.
- Construction-fixed, exactly zero scalar probability edges are omitted from
  reachability. State, action, age and free-parameter dependencies retain their
  declared edges; invalid all-zero lotteries still raise at execution.
- Breaking API: `Choose` becomes `DeterministicTransition`, `MarkovTransition`
  becomes `StochasticTransition`, and `initial_regimes` becomes `initial_nodes`.
  Both transition wrappers support decorator factories and carry numerical kernels
  only. Plain functions remain deterministic. Required `Model.edges` maps each
  source to destinations and their source-age selectors, optionally through `Phased`.
  `model.graph` provides immutable declared edges, effective reachability and
  fixed-zero pruning reasons.
- `Model(edges=...)` is the only place regime transitions are declared, structure
  and law alike. Breaking API: `Regime` has no `regime_transitions` field.
  - A source with exactly one outgoing edge at every source age can be declared as
    a plain `{target: source_ages}` mapping, with no law: the graph is the law.
    Deterministic schedules such as `"dead"` or
    `ByAge.until(law="working", then="retired")` are expressed by the edges alone.
  - A source with a law is declared as
    `Transition(targets={target: source_ages, ...}, law=...)`; one where some
    source age has several outgoing edges needs one. The law is any form
    `regime_transitions` took: a per-target probability mapping, a selector
    function returning a regime code (a discrete choice), a full-vector
    `StochasticTransition`, a regime name, `ByAge` or `Phased`.
  - A supplied `Transition` law is evaluated at every source age with outgoing
    edges, also where one edge leaves the source; there it must put unit mass on
    that edge. Only an age a `ByAge` law leaves unselected uses its one edge
    instead; a `ByAge` law must select every age with several edges.
  - A regime with no outgoing edges is terminal.
  - A law that differs between the phases is phased inside one `Transition`:
    `Transition(targets=..., law=Phased(solve=..., simulate=...))`. Destinations that
    differ between the phases go in `Model(edges=Phased(solve={...}, simulate={...}))`,
    each phase's mapping with its own `Transition`.
  - `DeterministicTransition` and `StochasticTransition` have no `targets` field.
  - `Regime` carries no law, so `Regime.terminal`, `Regime.gated_edges` and
    `Regime.decomposed_transition` are gone. `model.graph.laws[name]` holds each
    regime's law as the model binds it, with `terminal` (no outgoing edges),
    `gated_edges` and `decomposed_transition`.
  - Parameters of edge-declared callables (the regime-transition law, gates, gate
    references, route fallbacks) live at their declaration path under
    `params["edges"][source]`, and resolve from that path,
    `params["edges"][source][arg]` or the model level. A regime-level value never
    reaches them; `next_regime` and `gate` keys under a regime raise an error naming the
    new path. Per-target state laws keep their paths under the source regime. See
    [the migration guide](docs/user_guide/migrating_dated_regimes.md#migrating-edge-parameters).
  - Names that become parameter-path segments contain no `__` and do not start or end
    with `_`; `edges` is reserved as a regime, function and argument name.
- Probability mass validation is shared by the solver consumers. Compiled validation
  now reliably rejects negative subnormal probabilities at both precisions.

### Required starting problems and keyword-only age-indexed declarations

- `Model(..., initial_nodes=...)` is required and has no default. It accepts exact
  `(age, regime)` pairs or maps age selectors (exact age, tuple, `range`,
  `AgeRange(start=..., exclusive_stop=...)`) to one regime name or a nonempty
  sequence of names; the resulting pairs are the admissible roots,
  published as `model.graph.initial_nodes`. `model.initial_nodes` is the normalized
  `InitialNodes` declaration. `None`, a bare name, an empty mapping, unknown
  names and off-grid ages raise.
- `AgeGrid.inclusive_stop` includes the final grid age; `AgeRange.exclusive_stop`
  excludes its upper selector bound.
- Solved problems are derived from the roots — physical successors plus declared value
  reads — rather than from transition schedules. `ByAge` selects laws only;
  `ByAge(cases=..., default=law)` fills every unmatched age, including the last.
- A regime is terminal exactly when it has no outgoing edges in `Model(edges=...)`.
- `Model.declared_transitions[phase][source]` is the `Transition` a source declared,
  exactly as declared, with `phase` one of `lcm.typing.Phase` (`"solve"`,
  `"simulate"`). Its `law` is typed `lcm.transition.TransitionLaw`, its `gates` map each
  gated target to its `Gate`, and its `targets` are the declared or derived
  destinations; `model.graph.laws` holds the law as bound to the graph.
- `ByAge`, `ByAge.until`, `AgeRange`, `DeterministicTransition` and `StochasticTransition` take keyword
  arguments only. `ByAge.until(*, stop_age_exclusive, law, then, start_age_inclusive)`
  uses `then` at the last source age below `stop_age_exclusive`. See
  [Migrating to age-indexed regimes](docs/user_guide/migrating_dated_regimes.md).

### Device-memory budget on by default

- `ExecutionConfig.device_memory_bytes` defaults to `"device"`: the budget is the
  smallest selected device's allocator pool limit less `device_memory_headroom_fraction`.
  On GPUs and TPUs, which report a pool limit, the planner now picks the widest widths
  that fit instead of compiling every omitted axis at its bootstrap width. CPUs report no
  limit, so the default stays unbudgeted there. `device_memory_bytes=None` restores the
  unbudgeted planning explicitly; an integer is unchanged.
- Importing pylcm no longer sets `XLA_PYTHON_CLIENT_PREALLOCATE=false`: JAX preallocates
  its default 75% of each GPU again, and the default budget is derived from that pool.
  To allocate on demand, set `XLA_PYTHON_CLIENT_PREALLOCATE=false` yourself and pass an
  explicit `device_memory_bytes` or `None`. With a reported pool limit the default
  refuses an on-demand pool, because the BFC allocator then grows it in separate regions
  and the limit does not promise one contiguous block; the error names the fixes
  (preallocation, `TF_GPU_ALLOCATOR=cuda_malloc_async`, an integer budget or `None`).
- The GPU test tasks and workflows preallocate a per-worker share instead of turning
  preallocation off: `XLA_PYTHON_CLIENT_MEM_FRACTION` is 0.75 for the serial policy
  launcher (`pixi run test`, `gpu32`, `gpu64`) and 0.1875 for the four-worker `tests`
  tasks, so each worker's default budget comes from its own slice.
- Routes that cannot be budgeted --- `enable_jit=False`, forward simulation through host
  gated or replay adapters, and a supplied foreign result with replay payloads --- refuse
  the default budget as they refuse an explicit one; pass `device_memory_bytes=None` to
  run them unbudgeted.
- Budgeted simulation of finite NB-EGM replay on GPU no longer charges the policy copy's
  host-side scratch to the device ceiling, which refused every such simulation with
  "names an unbudgeted device", and a warm budgeted call no longer retraces its process
  grids.
- Every admission refusal names the budget and its source, then lists the remedies:
  raise `device_memory_bytes` or lower `device_memory_headroom_fraction`, cap or fix
  widths with `axis_width_ceilings` / `axis_widths`, shard over more devices with
  `sharded_states` / `devices`, or disable admission with `device_memory_bytes=None`.
- The benchmarks, which run with preallocation off, pass the default budget as an
  explicit one wherever they built a model without a budget (`bench_aca_baseline`,
  `bench_collective_household`, `bench_iskhakov_et_al_2017`, `bench_mahler_yum`,
  `bench_precautionary_savings`, `bench_simulation_dispatch` and the paired GridSearch
  scenarios), so they plan under it and their widths and timings change.
  `lcm_examples.collective_household.get_model` takes an `execution_config`, and
  `lcm_examples.mahler_yum_2024.MAHLER_YUM_MODEL` is built on first access rather than at
  import.

### Initial-condition validation without simulating

- `Model.validate_initial_conditions(initial_conditions=..., params=...)` and
  `Model.initial_conditions_feasibility(initial_conditions=..., params=...)` check a
  population without solving or simulating. Both accept the `simulate` input forms (a
  mapping of arrays or a DataFrame with regime names, ages and categorical labels) and
  ordinary user parameters. The first raises `InvalidInitialConditionsError` with the
  simulation diagnostics; the second returns a one-dimensional boolean mask in caller
  order, `True` exactly for the subjects admitting a jointly feasible action combination
  on the declared action grids. Malformed inputs raise from both, and so does a
  collective start without the `own_stakeholder` declaration `simulate` demands, so a
  population the methods accept is one `simulate` accepts. The checks take no
  `log_level`; at present they run in one eager pass on one device without device-memory
  admission or subject padding, which does not affect the verdict.
- Subjects of a regime without actions are now checked against that regime's state-only
  constraints, in `simulate` as well. Populations that previously simulated from such
  rows raise `InvalidInitialConditionsError` when a state-only constraint fails.
- A constraint depending on an age-specialized function while subjects start away from
  the regime's representative age now raises `UnsupportedOperationError` instead of
  `InvalidInitialConditionsError`, and `simulate` never downgrades it to a warning.
- Coverage now measures the child Python processes the test suite spawns
  (`[tool.coverage.run] patch = ["subprocess"]`), so the execution and sharding code
  those tests exercise is reported as covered instead of missing from every diff.

### Documentation for the execution and certification layers

- `docs/explanations/architecture.md` now describes the execution layer as built:
  ownership and lifetime of device buffers, admission and workspace planning, value
  transfers between regime meshes, and the resolved simulation plan. The development
  pages gain `certification.md`, which explains the candidate certificate, its source
  corridors, the seal check and the re-pin tool, and `continuous_integration.md` now
  documents the sharded CPU workflow, the workload manifest and its generator. The user
  guide's debugging, tuning and subject-parallel pages cover the log tiers, the budget
  knobs and the sharding routes that landed in this cycle, and every engine module
  carries a module docstring naming what it owns.
- The claim that a `None` device-memory budget queries no device was wrong and is
  corrected everywhere it appeared: an unbudgeted request applies no pool-derived cap,
  but public model construction still reads each visible device's pool statistics once.
- `tests/ci/generate_ci_workloads.py` regenerates the CI workload manifest and checks
  it accounts for every test file; `tests/candidate_certificate/test_repin_corridors.py`
  pins the re-pin tool's contract.

### Every route charges a transfer operator's own storage on every device it touches

- Budgeted admission now reserves a planned transfer's declared temporary bytes — what
  the operator holds beyond the value it delivers, which for a copy onto another
  regime's mesh is a second whole value — whenever a solve plans a copy, rather than
  only on the route that shards a continuous state. The charge lands on each endpoint
  of the copy, the devices the stored value sits on as well as the reader's, and a
  device that only sources a copy therefore enters the admission maximum on the stored
  shards plus that storage instead of escaping the comparison because no kernel of the
  cell runs there. An endpoint outside the plan's devices has no ceiling to be compared
  against and is refused by name. Both admission comparisons see the complete figure:
  the position gate that decides whether a core is compiled at all, and the
  reservation-plus-residency test that selects its workspace width. A refusal names the
  transfer charge alongside the regime, period, core and resident bytes, so a cell
  refused over a copy it only reads is distinguishable from one refused over its own
  values. Plans whose reads all arrive in their stored layout are unaffected; a plan
  that copies is admitted at a strictly larger, and now complete, footprint, so a
  budget that previously admitted it may refuse it.

### A gated edge may project a sharded state onto the referenced regime's grid

- A gated edge's `ProjectedRegimeValue` — a gate reference or a leg fallback — may now
  have its projection read a discrete state the source declares in
  `ExecutionConfig.sharded_states`. Such a state's device axis is sliced off every
  value array before the continuation is read, so the interpolation places no
  coordinate on it; the projection, however, is evaluated at the point the source lands
  on and consumes the state as a value rather than as an index. The landing coordinate
  was dropped together with the axis, so the solve refused with
  `Expected arguments: [..., 'next_<state>'], missing: {'next_<state>'}`. A coordinate
  the continuation still names is now supplied, which reproduced on a single device as
  well, since the declaration alone co-maps the state. Values and simulated frames are
  unchanged wherever the solve already ran.

### Broadcast pruning closes both phase slices jointly

- Which model-level states and actions a regime keeps is now the least common fixed
  point of the solution-slice and simulation-slice reachability operators, instead of
  one application of each in a fixed order. The two slices feed each other: a target
  regime that keeps a state only because its simulation-side payoff reads it turns the
  *solution* side of the entry law toward that target into a live computation, and
  whatever that law reads has to survive in the source regime. Closing the slices one
  at a time stopped before that hand-over was visible and pruned a retained law's own
  input, which surfaced at solve time as a spurious required parameter named after the
  entry law (`<source>__<target>__next_<state>__<input>`). It also made promoting a
  state from regime level to model level change the retained-state set, contrary to
  the declaration-move guarantee. Models with more than two alternating hops close to
  the end of the chain, a source feeding several targets keeps what every retained
  entry law reads, and the closed set does not depend on which slice is closed first.
  The additional work is build-time dependency traversal only; no warm-call path
  changes.

### A value read across regime meshes is planned from where it is stored

- A regime reading another regime's stored value — a gated edge's
  `ProjectedRegimeValue` fallback is the case that reaches this most often — now has
  that read planned from the layout the value is actually stored on. Two regimes that
  retain differently named sharded states run on meshes with different axis names, and
  the reading core's own partition spec describes its own axes and its own rank, not
  the other regime's; applying it to the stored value refused a solve that is well
  defined, with a bare "source sharding is incompatible with value shape" at solve
  time. Such a value is delivered to the reading core as a replica on that core's mesh
  and classified by the transfer catalogue from the two concrete layouts, so a model
  sharding one discrete state in one group of regimes and another in a second group
  solves. Reads between regimes on one mesh are unchanged, and a pair of meshes that
  overlap without either containing the other — which no single operator serves — is
  refused while planning, naming both regimes and both device axes.

- What such a read costs is stated where the layout is chosen. The replica is charged
  at its own size — the whole value on every device of the reading mesh, not a shard —
  with the stored value still charged on its own devices, the operator's scratch
  charged at the same size again, and every device either end touches named on the
  planned transfer. A device whose budget the reservation exhausts hosts no workspace
  and is refused by regime and period. A partitioned delivery is refused rather than
  chosen even when the reading mesh carries an axis of the stored value's own name and
  extent, because a value read consumes the complete value on every device: an entry
  law puts probability on every category of a discrete axis and interpolation spans the
  whole continuous line, so splitting the value would move the missing pieces into
  compiled work as a collective the plan does not name.

### The resolved simulation execution plan is logged

- `Model.simulate` now reports, once per call, the resolved execution plan: the
  forward route (`legacy`/`subjects`), the ordered subject devices and their
  backend, the resolved planner axis widths by regime, the outer chunk count and
  admitted chunk widths, and the budget mode with its effective device-memory
  bytes. A one-line summary logs at `log_level="progress"` and `"debug"`; the
  complete record logs at `"debug"` only. Neither line appears at `"warning"` or
  `"off"`. The record is also exposed as `SimulationResult.plan_summary`
  (`None` after `save`/`load`, diagnostic only). This closes the gap where a
  clean production run carried no evidence of which route actually engaged
  (issue #450). Purely diagnostic: no numerical, RNG, ownership or admission
  behavior changes.

### Gated edges simulate across subject devices

- A model whose regime declares a gated edge — a `Gate` on its transition, such as the
  dissolution edge of a collective regime — may be simulated with
  `ExecutionConfig(simulation_sharding="subjects")` on more than one device. The gate
  fold reads and writes regime-level grids and declares no subject axis, so it is
  replicated and the continuations it publishes stay shared operands; the gate route
  recomputes each row's gate at that row's own realized candidate state, follows the leg
  its role selects, and writes its fallback coordinates under an elementwise mask, so
  its population operands are partitioned over the configured devices. The rows
  published are the rows one device publishes.

### Sharding follows pruning, regime by regime

- A state named in `ExecutionConfig.sharded_states` may be dropped from any regime whose
  DAG never reads it, not only from a terminal one. The regimes that read the state
  carry its grid axis and keep the submesh that axis defines; the regimes that prune it
  publish a value without the axis, are placed as single-device regimes, and exchange
  values with the sharded regimes through the existing transfer catalogue. A wage shock
  read through working life and dropped entirely in retirement therefore shards the
  expensive working-life regimes without forcing the retirement regimes to carry it.
  Only a state that every regime prunes is refused, because no grid axis is then left to
  spread over devices; continuous sharding keeps its stricter requirement that the state
  be retained in every regime.

### Entry laws survive broadcast pruning

- A regime that does not read a model-level state keeps the `state_transitions` entry
  through which it hands the state to a regime that does. Pruning drops a keyed entry
  law only for the targets that also prune the state, and an unkeyed law only when no
  reachable target retains it, so promoting a state from regime level to model level no
  longer turns a declared entry law into a "the source does not carry '<state>' and
  defines no entry law" build error.

### Device-memory headroom below the allocator pool

- `ExecutionConfig(device_memory_headroom_fraction=0.15)` keeps a share of each
  selected device's allocator pool out of the admission ceiling, so a caller may pass
  the device's whole pool limit as `device_memory_bytes` without planning against
  memory the pool cannot actually hand out. The effective budget is the smaller of the
  request and every selected device's pool limit less its headroom; an
  already-conservative request is never reduced again, a device that reports no pool
  limit places no cap, and `device_memory_bytes=None` stays unbudgeted and applies no
  pool-derived cap — model construction still reads each visible device's pool
  statistics once. The margin is an operational policy for pressure outside the represented
  accounting — collective buffers, library workspaces, driver context, fragmentation —
  and `0.0` restores planning against the whole pool. A capped budget is logged as a
  warning naming request, per-device limits and effective ceiling, and admission
  refusals name both budgets.

### Execution widths per regime

- `ExecutionConfig(axis_widths=...)` accepts a width per regime. A bare integer under an
  axis name fixes that axis in every regime declaring it; a mapping from
  regime name to width — `axis_widths={"cell": {"cheap": 512, "heavy": 8}}` — fixes it
  only in the regimes it names and leaves the planner free in every other one, so a
  width one regime's shape demands no longer chunks an unrelated regime. The two forms
  may be mixed across axes. A regime name the model does not declare, and a per-regime
  width for an axis only simulation programs declare, are refused at model build with an
  `ExecutionPlanningError` naming the declaration.

### Common taste shocks across counterfactuals

- `Model.simulate(taste_shock_seed=...)` selects an independent Threefry stream for
  EV1 taste shocks. Matching exact ages, initial subject rows and ordered discrete
  action domains share standardized draws across policies and regimes, independently
  of the ordinary seed, horizon endpoints, chunking and padding. Omitting the argument
  preserves the ordinary seeded behavior. Reordering or resizing a discrete action
  domain changes its stream.
- Cached simulation key generation now follows changes to JAX's Threefry partition
  setting between calls, including when an executable has already been compiled.

### Solver API version 3

- `SOLVER_API_VERSION` is 3. Every solver implements `capabilities`, returning the
  frozen `SolverExecutionCapabilities` exported by `lcm.solver_api` and `lcm.solvers`.
  Its structural requirements, execution axes and supported preference features drive
  the solver reference tables. Concrete model programs remain the authority for
  accepted execution widths. Custom plugins must implement this property and target
  API 3. Solver identities and model fingerprints change with this compatibility
  version; API 2 solution archives are not automatically migrated.
- A core program declares what it publishes to another
  program of the same kernel graph with `InternalOutputSpec`, and a consumer names what
  it reads with `InternalInputRef`; both are published through `lcm.solvers`. The engine
  lowers producers before consumers against the exact shapes, dtypes and weak typing the
  references select, so a consumer no longer runs against a stand-in filled in later.
  NEGM's keeper-to-sweep dependency uses the typed edge.
- A program graph's typed internal edges lower for every admitted shape, not only for
  a dense one-hop edge. A producer's abstract output is computed from the complete
  invocation the engine lowers — its arguments, the internal inputs it reads itself,
  and the widths the execution planner owns — so a chain of producers and a `PLANNED`
  producer both reach their consumers. Weak typing is part of the template a consumer
  is lowered against, since equal shapes and dtypes can still promote differently in a
  consumer: a handed-over leaf whose weak typing departs from what the consumer was
  traced with is refused at dispatch, naming the program and the argument. A producer
  whose published output would change with the selected width — its weak typing
  included — is refused with an `ExecutionPlanningError`.
- `CoreExecutionDisposition.HOST_DRIVEN` declares a program whose host loop dispatches
  it a data-dependent number of times. Like `DENSE` it owns its own width and must carry
  a `disposition_reason`; the engine plans nothing for it.
- A compiled solve executable is keyed by program identity — the durable model
  fingerprint, the regime name, the core name, the engine's per-period signature
  (`SolutionPhase.period_signatures`), and the solver's own period group key
  (`SolutionKernels.period_group_keys`) — instead of the identity of a Python callable.
  Equivalent programs built separately within one solve share one executable, and a
  solver whose published group key is coarser than what it actually specialized is
  refused at build time with an `ExecutionPlanningError` naming both colliding
  programs.
- The value-transfer catalogue names one operator for every stored/required layout pair
  the planner produces: `ALIGNED_LOCAL`, `COPY_TO_SOURCE_LAYOUT`, `ALL_GATHER`,
  `LOCAL_SLICE`, `RESHARD`, and `CROSS_MESH_COPY`. Two device meshes that share devices
  while neither contains the other are the one pair no single collective serves, and are
  refused while the period is planned with an `ExecutionPlanningError` naming both
  device sets.
- A published continuation answers questions about itself. `ContinuationReader` is the
  protocol a parent queries — `value_at`, `marginal_at` in a named state, and `leaves()`
  for the payload's addressable arrays — and `ContinuationCapabilities` is the payload's
  own statement of what it can answer. A parent declares what it needs in
  `Solver.required_continuation_capabilities`; every endogenous-grid solver demands the
  value and the marginal in `EGM_ENDOGENOUS_COORDINATE`, and the shipped `EGMCarry`
  publishes both. A reachable target whose payload is not a reader, or whose reader
  answers less than the parent asks, is refused while the model builds.
- A dense EGM-family core declares the continuation leaves it reads instead of being
  pinned conservatively against every reachable value. Which leaves exist is a fact
  about the target's published template, so a solver names them in the new
  `Solver.declare_continuation_reads`, which the engine calls once per
  continuation-reading regime after every regime is built, with
  `SolverBuildContext.continuation_specs` filled. The default declares nothing; the hook
  may only attach `value_reads` to the programs the kernels already publish, so no
  kernel is built twice. Only the returned container's `period_kernels` are honoured,
  and a hook that moves any other field, or that declares reads for a regime whose
  continuation the engine publishes on the solver's behalf, is refused at model build.
- The engine's gated-edge fold declares the same-period values it reads. Folding one
  edge at one period is a dispatch in its own right, so the target's value and each
  reference regime's value are counted consumers rather than values pinned wholesale
  because nobody had named them.
- Solution archives written under solver API version 1 are rejected with
  `IncompatibleSolutionError`. Compatibility remains exact; pylcm does not migrate an
  archive across a solver API version.
- Backward induction releases every cross-period input after its final consumer and
  donates sole-consumer inputs a program names in `donation_candidates`; the donation
  set is part of the compilation key. Regime values are never released. Under
  `log_level="debug"` every release and donation is logged with the artifact key and
  the closing dispatch.
- On several devices every regime is placed on a submesh before compilation: a
  distributed state of extent three solves on three of four devices, single-device
  regimes fill idle devices, and independent regimes of a period dispatch concurrently.
  Two placements of one model publish values that name the same real number — each
  partition is vectorized at its own width, so they agree to within a few units in
  the last place rather than bit for bit. One device or one regime per period is
  placed as before.
- A streaming width fits when its compiler-reported peak plus accounted external
  residency fits `ExecutionConfig.device_memory_bytes`. Retained owners and
  compiler-eliminated operands remain in residency; only the overlapping spans of
  actual compiler-kept inputs are excluded to avoid counting them twice. Fixed
  model inputs and scheduled transfer copies are included conservatively.

### Execution configuration and solve/simulate memory attribution

- Pass `ExecutionConfig` to `Model(...)` to select devices, sharded states, execution
  widths and a per-device memory budget for both phases. Grids describe economic
  support; their `batch_size` and `distributed` constructor arguments and the
  corresponding solver execution knobs are removed. Explicit `axis_widths` names the
  actual compiled program axes; a flattened state-cell width is not a per-state grid
  width. Execution policy does not enter the durable model fingerprint.
- With a budget, solve planning compiles candidates along a deterministic widest-first
  frontier and selects the first whose compiler peak plus accounted residency fits.
  Without a budget, streamed solve axes use their bootstrap widths unless an explicit
  width is supplied: a reduced axis, whose block is purely temporary, is capped at 64;
  a tiled output axis, whose tiles concatenate into a full-size resident result, is
  capped at up to 1024, the cap shrinking so the product of an unbudgeted candidate's
  widths stays bounded regardless of model size. An omitted width requests planning,
  and zero is not a full-width sentinel in `axis_widths`.
- Budgeted solves wait for earlier compiled work before dispatching another core
  whose execution or transfer devices overlap. Completion retains auxiliary outputs
  and copy witnesses through validation and error cleanup, and runs before eligible
  buffers are deleted or donated. Work with disjoint complete device footprints may
  remain asynchronous. This does not bound compiler autotuning or unprofiled host
  allocations.
- Configure subject chunks with `ExecutionConfig(axis_widths={"subject": width})`;
  `Model.simulate(subject_batch_size=...)` is removed. Device alignment can increase
  the outer chunk extent while preserving the requested inner width. With a budget
  and no fixed subject width, complete chunk profiles include retained results,
  numerical programs, RNG, diagnostics, padding and assembly. Chunk boundaries
  preserve each original subject's random stream. The constructor `n_subjects` hint
  and its duplicate concrete-template prewarm route are removed; call-shape runtime
  executors are cached instead, keyed on subject shape, so the first `simulate()` call
  at a given shape compiles it and later calls at that shape reuse it.
- Solve candidate preparation uses shape and layout descriptors instead of allocating
  transfer copies for each width. Budgeted foreign eager solution values use compiled
  copy admission and retain intermediate copies through validation. Native archive
  materialization, artifact copies and mixed-backend foreign copies remain unsupported
  under a budget.
- Budgeted simulation accepts built-in native GridSearch value archives, admitting
  verified uploads and both private copies while retaining the archive cache. Actual
  source devices on the execution backend participate in residency and transfer-scratch
  accounting even when they are outside the selected execution subset.
- Simulation programs whose inputs are all removed by the compiler still run on
  their selected devices, preserving inferred scalar and vector output layouts.
- Eager solve inputs now bind weakly typed values to matching declared strong types,
  preserving compiled type promotion and their actual device layout. Shape and dtype
  mismatches remain errors; normalization copies are shared within each call and
  released with that call.
- Eager solve cores place ordinary inputs on their planned submesh before executing
  the numerical body, including bodies that create constant outputs. Repeated inputs
  share a placement when their complete descriptors match. Equivalent runtime layouts
  are accepted without changing declared transfer or compilation identities. Declared
  producer outputs reach subsequent eager cores with their actual validated layouts.
- Solve operand profiling uses a shared descriptor callable, allowing model disposal
  to release every per-program nested function.
- Finite NNBEGM simulation admits its declared candidate preparation, dropped-candidate
  diagnostics and canonical ranking under the chunk budget. The full candidate bank
  remains owned and counted through ranking, with addressed policy transfers shared
  across both programs.
- Supplied solutions on regime submeshes are accepted by simulation. Forward programs
  and profiled allocation operations recheck current retained inputs and growing
  results at dispatch. Unprofiled eager or host-driven programs fail visibly under a
  budget. These checks do not yet bound every allocation across the whole call or
  memory used during compilation.
- Period captures record the selected tile widths, so `replay_period` and the
  compiler-memory analyzer lower exactly the executable the solve dispatched.
- The ASV GPU-memory series for the ACA baseline and Mahler–Yum rows are split into
  three independently measured phases — automatic solve+simulate,
  `ALL_PERSISTABLE_ARTIFACTS` solve+save, and load+supplied-solution simulate — each in
  a fresh, phase-isolated child process with exact provenance. The combined
  timing/CPU subprocess no longer reports a GPU peak.

### The MSS upper envelope decides its orderings from the stored operands

- The envelope's comparison arithmetic is selectable: `MSSEnvelope(arithmetic=...)` takes
  `"certified"`, the default, or `"ordinary"`. The geometry is the same either way —
  which stored piece covers an interval, which node owns a query, and where two branches
  hand over — so both settle a knot at the root of the pieces covering it, and only the
  comparison changes. The certified arithmetic decides on the stored operands, so an
  ordering the working format cannot separate is still settled and a comparison it cannot
  decide publishes `NaN`; it needs the installed exact-affine payload for the active
  backend, and a regime selecting it is refused at model construction when that payload
  is absent. The ordinary one compares two readings formed in the working format, each a
  slope and then an affine step rather than one correctly rounded value, so a reading
  carries no bound in representable steps: cancellation or a large common level can
  reverse an ordering the correctly rounded values would separate, and candidates whose
  readings coincide are separated by the declared tie order rather than by value. It
  reaches no native kernel, so it is the route available where that payload is absent;
  it trades certified decisions for warm cost, and its values, owners and crossings are
  the caller's to validate for the intended model.
- Which link owns a query is certified rather than read off a rounded comparison:
  every link bracketing that query enters one exact reduction through the integer
  comparator the other envelope paths already use. Links certified level with one
  another are separated right-continuously — the link reaching strictly
  right of the query, then the steeper one, then the earlier stored link — so a
  node where two branches meet is owned by the branch that owns the interval above
  it, and a switch decided by a single representable step is published at the node
  the geometry puts it at rather than lost.
- A query whose owner the exact comparator leaves undecided publishes no value: the
  envelope reads `NaN` there instead of falling back to a rounded comparison, so an
  ordering the arithmetic cannot settle is visible rather than silently chosen.
- Which links compete for a query is decided by their stored spans alone, never by
  whether reading one of them succeeded numerically. A model whose grids and values
  sit near the top or bottom of the working format keeps the owner it should have:
  the reading of the selected owner is range-safe, so an intermediate product that
  leaves the representable range no longer removes a finite winner from the contest.
- A link's value at a query is its own chord's value: the certified reader forms the
  exact rational through the two stored endpoints and rounds it once to the working
  format, without weighted floating products. The reading is the stored value at
  either endpoint exactly and elsewhere does not carry the cancellation of a line
  extrapolated from one far anchor. The published value and policy at a node always
  come from one owner.
- Ownership, support, orientation and node identity are decided from each link's original
  stored coordinates. A link stored as a single point keeps its own abscissa as its
  support instead of acquiring a readable width to the right, so a point and a segment
  that read the same value at a query are separated by the declared right-continuous
  order rather than by whichever carries the wider line. A readable surrogate is still
  used to read a channel, but never to decide who owns the query.
- Two abscissae name one published node exactly when they are the same geometric
  location, decided on the stored encodings: the two spellings of zero are one location,
  while distinct values closer together than the smallest normal remain distinct. The
  same rule orders, orients, admits and coalesces, so no two of those can disagree about
  whether two coordinates coincide.
- A crossing is constructed from the pieces that cover the interval the switch was
  observed in, not from whichever piece owns each of its two nodes. A branch entered at
  the right node is often represented there by the piece continuing above that node,
  whose line says nothing about the interval below it; the piece covering both of the
  interval's abscissae is the one the crossing is solved from, so a kink sits where the
  branches actually meet rather than where a continuation extrapolated backwards would
  meet them. Exact equality at the shared endpoint connects the selected piece to the
  node it hands over at, and a trace that is missing, ambiguous, or disconnected from
  that node publishes `NaN` rather than a plausible abscissa.
- The gap between those two pieces is signed only where both of them are supported, and
  its root is solved from the stored operands themselves, so a switch whose two rounded
  chord readings coincide is still placed at the abscissa the geometry puts it at
  instead of being collapsed onto an interval endpoint, and the policy read on each side
  of it belongs to the branch that owns that side. Its published value is the higher of
  the two chords there, so an emitted kink can never sit below both branches.
- A crossing landing exactly on one of the two query nodes is published rather than
  discarded. That node's own row is one of the two records the switch needs and the
  emission contributes the other: the incoming owner after a crossing at the left
  node, the outgoing owner before a crossing at the right node. Either way the kink
  abscissa carries exactly two rows, outgoing owner first.
- Whether a crossing lies on the envelope is settled by naming the owner at the crossing
  abscissa and requiring it to carry the same value there as the piece the crossing was
  built from, rather than by comparing two readings within a tolerance band. A shared
  branch label is not provenance on its own, so the emission depends on neither a
  declared band nor the working precision, and a crossing whose provenance the exact
  comparator cannot certify is published as `NaN` rather than admitted.

### Engine functions are defined once, not per call

- Every function the engine defines is a module-level function or a frozen dataclass
  with a `__call__`, apart from the five sources named below; no other function
  definition runs inside another function per model build, per solve, per simulate, or
  per trace. pylcm's beartype claw decorates each function definition it sees and
  beartype memoizes every decorated function object for the life of the process, so a
  per-call definition pinned everything it closed over: grids, arrays, tracers,
  compiled kernels, whole models. Building and solving, then dropping, a model of any
  shipped solver family leaves no engine function behind outside those five, and a long
  session or test worker no longer grows with every model it builds.
- Five sources still define a function per call, and the nested-function probe names
  each of them as an exemption: `regime_building/max_Q_over_a.py`,
  `regime_building/collective.py`, `regime_building/processing.py`,
  `solution/negm.py`, and `solution/nnbegm.py`. The candidate certificate pins the
  reducer builders' bodies verbatim, nested definitions included, so their shape
  changes with that certificate or not at all; the remaining three are converted
  together with the execution work that rewrites the same call sites.

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
  result. The adaptive NNBEGM policy carries its solve-generated outer nodes as the
  candidate axis of its descriptor, so it is model-verifiable and persists: a consuming
  model admits the nodes after checking that they are exact finite floats, strictly
  increasing, within the search's node budget and inside the outer state's domain for
  that period.
- Solver diagnostics are persisted. Each retained `SolverDiagnostics` payload is
  described by a model-verifiable descriptor the solve generates from the payload
  itself; a consuming model admits the descriptor only when it names exactly the
  published fields with the dtypes they carry, and a restored archive reads its
  diagnostics without a model. A diagnostic omission without a descriptor is refused
  on both save and load. The solution-format version is 2.
- A result restored by `load_solution` can be saved again without a model. Its payloads
  are re-read from the archive they came from and verified against their checksums and
  descriptors before they are written to the new archive.
- A `SolutionResult` is consumed according to its provenance. A result the same model
  instance solved for the same canonical parameters is read by reference, so a simulate
  following a solve copies and re-validates nothing; every other result — restored,
  unpickled, or from another instance — is validated in full and materialized once per
  consuming model and parameter vector, and later simulations from it reuse that view.
  The consumed result stays reachable as `SimulationResult.solution` until the
  simulation result is saved.
- A model is sealed when it is built. `Model(...)` records every global and closure
  binding its declared callables read; rebinding one afterwards makes `solve()` and
  `simulate()` refuse with `ModelSealError` naming the binding, and a model whose
  callables cannot be fingerprinted is refused at build with
  `ModelInitializationError`.
- Every external solver declares how its decision is replayed. `SolutionKernels`
  carries `replay_route`, which is an `ExecutableReplayRoute`,
  `DeclaredReplay.GRID_RECOMPUTATION` (the shipped argmax over the declared action
  grids), or `DeclaredReplay.UNSUPPORTED` (the model solves but refuses to simulate,
  naming the regime and solver). A solver outside the shipped set that leaves the route
  unset is refused at model build. `DeclaredReplay` is exported from `lcm.solvers`.
- `ValueRead`, `ValueArtifactAddress`, `ValueArtifactKind`,
  `ValueConsumerAddress`, and `ValueInputChannel` are public through `lcm.solvers`, so a
  core program that reads next-period stored values can declare each access;
  `SolverBuildContext.solution_reachability.targets(period=..., source=...)` names the
  targets to declare.
- Retention now selects computation per exact artifact address. Replay alternatives and
  additive artifact programs declare `CoreProgram.retained_artifact_keys`; DCEGM,
  NB-EGM, NNBEGM, and external solvers avoid assembling outputs that the selected
  retention will discard.
- `ValueStore` and `ArtifactStore` expose per-entry `LoadState`. Metadata, omissions,
  coordinates, and whole-archive checksum verification do not materialize payloads;
  value-only inspection therefore does not load replay banks.
- `ValueStore`, `ArtifactStore`, and the `omissions` of `SolutionResult` admit a public
  mapping through one item traversal and check every raw address before inserting it:
  a Boolean period alias (`True` for `1`) is refused rather than merged into the exact
  integer coordinate, a repeated logical address is refused rather than contracted, and
  a mapping's key view is never consulted, so flat and nested forms are recognized from
  the items alone.
- The durable model fingerprint and the canonical-parameter digest hash every referenced
  array with its original rank and shape; a scalar array and a length-one vector with
  the same bytes are distinct identities, while memory order does not enter the digest.
  The model-fingerprint record version is 6.
- The model fingerprint is split at the boundary no parameter vector crosses.
  `fingerprint_model_structure` digests what a model fixes at build — topology, names,
  identities, artifact descriptors, per-period state axes, and the declared callables'
  semantics — and `fingerprint_model` combines that digest with the concrete grid
  support and canonical solution parameters read from the parameter vector. A `Model`
  hashes its structure once and reuses it, so an estimation loop no longer walks every
  declared user callable per solve; the walk had been the largest single cost of a warm
  solve, ahead of the solve itself.
- A model function may read an array constant's `shape`, `size`, `ndim` or `dtype` and
  still be fingerprintable. Those four are functions of the array the digest already
  covers, so binding them adds no state the fingerprint misses. The array's own
  descriptor is what earns the exemption rather than the attribute name, and the bound
  value must be the metadata it claims to be, so a `size` property on a type that is
  not an array still fails closed.
- The durable model fingerprint sees through beartype guards that a downstream
  package's own claw wraps around its model functions: such a guard is accepted only
  when beartype regenerates its code from the bound callee with the guard's own
  configuration, and the callee is what enters the identity.
- `SolutionResult.save()` flushes the archive through a writable handle before the
  atomic rename, so publication also succeeds on platforms that refuse to flush a
  read-only one.
- A result nobody references any more releases its arrays. The store constructors and
  the artifact plan walkers keep their working state in explicit arguments rather than
  in per-call closures: pylcm's beartype claw decorates every function definition,
  including one executed inside a call, and beartype memoizes each decorated function
  object for the rest of the process, so a per-call closure would pin the value
  entries or artifact leaves it closed over long after the result was dropped.
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
- An omission names the artifact its solver declared. `SolutionResult.omissions` is
  enumerated from the model's own artifact authority, so a solver that publishes its
  own continuation key sees that key at its own cell under every retention, and no
  key the model never declared appears.
- A continuation published under a key its payload does not claim is refused where the
  producing regime and period are still known, naming both the publication key and the
  payload's own.

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
  `{target_regime: StochasticTransition(func=prob_func)}` whose key set declares the regime's
  reachable targets — omitted regimes are structurally unreachable. Per-target dicts in
  `state_transitions` hand state values across regime boundaries, including into states
  the source regime does not carry and across grids that differ between regimes.

- A bare callable or bare `StochasticTransition` on `Regime.transition` declares
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
  `StochasticTransition` laws, so perceived mortality, perceived health or income risk, and
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
