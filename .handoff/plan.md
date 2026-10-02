# Invariant-state-aware execution: implementation plan from pylcm #482

## Objective and starting point

Exploit a constant state such as ACA's preference type as an independent continuation problem, regardless of which dimensions distribute work across devices. Deliver type-local GridSearch first; extend the same execution machinery to hybrid meshes, simulation, and then multiple nodes. Shared arithmetic and action partitioning are separate, measured optimizations, not prerequisites for the first useful release.

This is an implementation specification, not a report of executed experiments. No GPU or cluster tests were run in preparing it.

Freeze these starting revisions:

| Repository | Revision | Role |
|---|---|---|
| pylcm | `2e2392eb1eeca117cb09f808d6ff575f5db77a88` | #482 head inspected for this plan |
| pylcm | `52407b79f5c25f93f6b80c33f60f7c0e2dd45bc8` | #482 base; secondary historical control |
| aca-model | `ad38653696ec366e318ac61b9a81b597a4ecb700` | ACA revision pinned by #482's benchmark environment |

#482 moved since the preceding discussion, which inspected `d2620826...`. Branch from the newer pinned head. If #482 changes or is merged before implementation, record the successor, review the intervening changes, and regenerate affected baseline evidence rather than silently substituting a moving branch. [S1, S2]

Preserve #482's dense whole-product path, streaming partial-width path, allocator policy, admission checks, lazy width-frontier work, and numerical reduction semantics. The PR's reported Mahler improvement is not a forecast for ACA. [S1]

## 1. Non-negotiable design decisions

**One economic model, one execution framework.** Do not create an ACA-specific solver, call `Model.solve` once per reconstructed one-type model, or add a competing scheduler or value cache. Extend canonical construction, `CoreProgram`, the existing scheduler, value transfers, and stores. Grids, admissible initial conditions, transitions, economic fingerprints, and public coordinate labels remain unchanged.

**GridSearch first.** Initially support the ordinary singleton hard-max GridSearch route needed by ACA. Keep existing routes working when the optimization is off. NB-EGM consumes the common interfaces later; DC-EGM, NEGM, collective models, and newly optimized gated/reference routes do not gate this work.

**Structural facts are not execution choices.** Separate invariant coordinates, function dependencies, physical partitions, and requested value views. A fixed coordinate can be unsharded, sequentially blocked, or distributed across device groups without changing its economic meaning.

**No new approximation.** Do not change grids, shorten stochastic reductions, alter probability-slot ordering, change argmax conventions, or substitute approximate values. Preserve existing phase and boundary semantics. Use keyword-only new public/internal APIs and the repository's development/test conventions. [S3]

**Opt in before changing defaults.** Invalid explicit requests raise `ExecutionPlanningError` with the failed condition and available remedies. Automatic selection considers only certified routes; an inapplicable optimization does not make an otherwise valid model invalid. Diagnostics remain under the existing `log_level` policy, but structural safety and budget checks must not depend on logging.

## 2. Extend the existing architecture, rather than recreating it

The starting tree already has exact value-read addresses and solver-declared programs, a transfer catalogue, memory/liveness machinery, fixed-state co-mapping, lazy value stores, subject-parallel simulation, and an exact mergeable hard-max reducer. Reuse these facilities. [S4–S10]

| Responsibility | Existing implementation surface | Required extension |
|---|---|---|
| Structural independence | `_lcm/regime_building/processing.py`, `fixed_components.py`, transition plans | Phase- and dependency-aware invariant-component analysis |
| Kernel inputs and value reads | `_lcm/execution/core_program.py`, `_lcm/solution/contract.py` | Explicit coordinate bindings and selected artifact views |
| Transfers | `_lcm/execution/value_transfer.py` | Select-before-transfer with distinct stored/view shapes |
| Placement | `_lcm/execution/placement.py`, `execution_plan.py`, `_lcm/engine.py` | Physical partition counts independent of grid extents |
| Execution and memory | `_lcm/solution/backward_induction.py`, execution scheduler, footprint, liveness, donation, workspace planning | Schedule component blocks and account for their real owners and buffers |
| Numerical kernels | `_lcm/solution/grid_search.py`, regime-building `max_Q_over_a.py`, `Q_and_F.py`, `V.py` | Fixed bindings without requiring the fixed coordinate to be a distributed axis |
| Results | `lcm/_solver_api/stores.py`, entries/result machinery, persistence | Complete logical results backed by selected physical blocks |
| Simulation | `_lcm/simulation/simulate.py`, programs, chunk admission/profiles | Type-aware subject grouping and selective value residency |

The transfer layer currently validates a single expected shape against stored and consumer shardings. Therefore, shape-changing selection is a real contract extension, not simply an extra `device_put`. Existing `LOCAL_SLICE` is a layout-transfer classification; do not quietly give it the different meaning of selecting economic coordinates. [S5]

### Minimal new execution controls

Add fields incrementally, only with their implementing stage. The following are proposed APIs, not existing options:

```python
# Stage 3: at most one preference type per active device group;
# all selected GPUs cooperate on that type's asset work.
ExecutionConfig(
    devices=tuple(range(8)),
    sharded_states=("assets",),
    invariant_block_widths={"pref_type": 1},
)
```

`invariant_block_widths` limits simultaneously evaluated invariant-coordinate values within each assigned group; it does not alter economic support. Continue using `axis_widths` for `cell`, `action_product`, and `subject`.

Empty new mappings preserve the current route and placement defaults. Validate positive exact integer counts, unknown/pruned-everywhere names, partition compatibility, and conflicting requests before lowering. Explicit physical counts must not be silently reduced. Initially permit one explicitly blocked invariant axis; design the internal selector as a named tuple of coordinate selections so later multiple invariants do not require a new representation.

## 3. Stage 0 — Freeze the oracle and expose the limiting cost

**Deliverable:** reproducible baselines plus an execution/memory breakdown; no algorithm changes.

Use the pinned benchmark environment and record Python, JAX/jaxlib, native build identity, CUDA/driver, allocator settings, GPU model/count, interconnect topology, precision, grids, parameters, seeds, and output-retention policy. #482 requires Python >=3.14 and JAX >=0.11.1; use its resolved environment rather than independently upgrading dependencies. [S2]

Create three workload levels: a tiny independent-type model with an independently checked solution; reduced ACA retaining the relevant regimes/transitions and two/three types; and the unchanged production ACA workload. Use existing benchmark/capture facilities and extend them, rather than building another harness.

Measure construction, validation/planning, tracing/lowering, backend compilation, warm solve, warm simulation, and output/I/O separately. Within solve, select expensive period/regime kernels for detailed profiling.

Add or expose compact plan records for invariant extent, selected block, logical and physical shapes, device groups, selected widths, dense/streamed dispatch, stored-owner bytes, active replica bytes, transfer workspace, compiler reservation, and last-use/release events. Separate logical gathered bytes from observed communication. An HLO interpolation `gather` is not itself an inter-device `all-gather`.

Capture A/A noise before A/B comparisons. Use synchronized boundaries, repeated paired runs, and unchanged clocks/allocator settings. Keep raw logs, compiler output, profiler traces, and exact array/panel comparators. Avoid diagnostic synchronizations inside the hot path solely to produce attractive timing attribution.

**Exit gate:** the team can explain where time and per-device memory go. Unsupported historical configurations are recorded as such; they are not made into misleading comparisons by changing grids or the model.

## 4. Stage 1 — Establish invariant components independently of placement

**Deliverable:** canonical structural metadata and rejection tests; execution unchanged.

A coordinate is eligible only when its preservation is established from accepted declarations across the relevant reachable dependency graph. Start with `fixed_transition` and reuse the existing `fixed_component` normalization. Do not infer invariance from names, observed simulation paths, estimated zero probabilities, or a few evaluations of a user function.

ACA's pinned model declares `pref_type` with `fixed_transition`; it also retains it in the terminal bequest problem. This is the first production target, not a hard-coded special case. [S11]

For each candidate record its canonical name/code mapping, carrying regimes, active periods, solve/simulate status, preserving edges, and shared type-free dependencies. Check every value-input channel: ordinary continuation, same-period references, edge references, gated artifacts, and replay reads as applicable. An unsupported channel is not evidence of independence.

Support the straightforward cases first: a type-preserving chain and a chain that reads a genuinely type-free terminal child. Preserve one shared node for the latter; do not duplicate its economic identity per type. A type-dependent child must retain its type-specific view. A dropped coordinate followed by re-entry requires an established binding; otherwise reject optimized execution of that component. A reset or cross-type value reference prevents independent scheduling unless its coupling is explicitly represented by a later supported route.

Analyze solve and simulate separately. A solve-invariant coordinate is not automatically a safe lifetime simulation grouping key. Preserve required value dependencies and admissible initial-condition roots; do not add a death root or discard types merely because a current population or mixture weight omits them.

**Tests:** identity and generated fixed components; type-dependent preferences and transition probabilities for other states; a shared type-free terminal; reset/re-entry and cross-type-reference negatives; phase mismatches; labels/slices not starting at type zero; disabled-optimization parity.

**Exit gate:** eligibility is independent of `sharded_states`, and explicit unsafe blocking fails before numerical dispatch.

## 5. Stage 2 — Add selected value views, transfers, and ownership

**Deliverable:** the common machinery needed to consume `V[..., type selection, ...]` without replicating the full type axis.

Extend the read/transfer contract with a private named value-view descriptor. It identifies the original logical artifact, selected canonical coordinate interval/code mapping, axes selected or removed, expected consumer shape/dtype/weak typing, and required layout. Distinguish an unsliced shared leaf from a selected leaf explicitly. Never infer selection from coincidentally matching array lengths.

Keep the original artifact address and economic identity. A view is a representation of that artifact, not a different value function. Extend versioned execution/specialization records where required; do not automatically change an economic fingerprint because a tile or device count changed.

The planned operation must be:

```text
stored artifact -> select invariant coordinates -> copy/gather within consumer group
```

not full replication followed by indexing. Represent selection and communication as inspectable stages with their actual shapes, costs, and dependencies. Permit an optimized fused lowering only when the planner still accounts for all physical allocations and observed behavior satisfies the same contract.

Transfer-result cache keys include solution generation/parameter identity, artifact address, selected coordinates, dtype, and destination layout. Executable keys include code-affecting shape/layout/selector structure, but ordinary type codes and estimated numerical parameters should remain runtime operands where legal. Equal-shaped type 0 and type 1 views must never alias numerically; equivalent same-group executable shapes should not compile separately simply because the type code changed.

Update liveness, resident inventory, pending work, and donation together. A selected view can retain a full parent allocation. A compact copy consumes memory until completed/released. Overlapping live views prevent unsafe donation of their owner. One task finishing does not permit release while another consumer or transfer is pending. Cache lifetime must match selected-component scheduling, not just the old period-only assumption.

**Tests:** selected versus original reads; same shape/different type collision; parameter change; shared leaf; axis-order permutation; source/consumer shape mismatch; alias/copy accounting; cross-group transfer; release after the last asynchronous consumer; tight-budget refusal before dispatch.

**Exit gate:** a standalone selected transfer has the expected numerical payload, reduced consumer footprint, and truthful owner/temporary accounting. Lowering or profiling must show it does not secretly replicate the full type axis first.

## 6. Stage 3 — First production slice: one type block, all GPUs

**Deliverable:** opt-in type-local GridSearch on the existing common execution path.

Initially keep backward induction period-major:

```text
for period in reverse order:
    for invariant block:
        execute that block's required program graph on the assigned GPUs
        release block-local replicas and temporaries after their last consumers
```

These are scheduler units in the existing graph, not a Python wrapper that constructs independent Models. Preserve same-period ordering and dependencies on shared nodes. Require every continuation read in the enabled route to resolve to the correct selected or shared view.

Extend fixed-state co-mapping so a coordinate can be bound without being a distributed grid axis. Preserve its original code for utility, parameters, transitions, and diagnostics, even when it is removed from the interpolated value array. A one-type slice containing global code 2 must not be renumbered to type 0.

Use the existing continuous-sharding numerical restrictions. Change only what is necessary to combine a proven invariant binding with that route. Do not simultaneously admit arbitrary grids, folds, gated/reference channels, taste shocks, or mixed solvers.

The inner state product excludes bound invariant coordinates; physical memory admission still includes the number of simultaneously active bindings. Preserve canonical action order and the existing hard-max implementation. Select widths from the selected inputs and actual compiled reservation—not by dividing a previous full-model reservation by the number of types.

Retain #482's dense fast path whenever the local named product is covered. A full action width or full local state width must not gain an unnecessary loop. Preserve canonical state dtypes and weak typing when binding type codes; do not introduce weakly typed Python scalars that change arithmetic promotion. Block type values dynamically where possible; do not generate a new Python closure and backend compile for every `(period, type)` when the kernels otherwise agree.

For equal-sized type slices, selecting B of K types reduces that particular active replica from C bytes to approximately `(B/K) C`. This says nothing by itself about retained full-solution owners, type-free leaves, or total peak memory. Keep complete canonical output assembly under the existing result contract and account for both old/new allocations where assembly requires them.

**Acceptance experiments:** first hold effective local cell/action widths fixed, then allow each route to reselect widths under the same budget. This separates the direct locality gain from the indirect gain of admitting larger dense kernels. Test one GPU and multiple GPUs with legal asset partitions, including three types using an eight-GPU group.

**Exit gate:** full reduced-model value/policy parity and unchanged public result schema; verified type-local replicas; no type-count multiplier in same-shape/same-placement warm compilation; measured ACA evidence. This stage can land without full lifetime block-major storage, hybrid meshes, action partitioning, or MPI.

## 7. Stage 4 — Skipped

Decided 2026-10-02: skipped. Production runs on Marvin nodes with 4×A100. ACA has three preference types, which do not divide four GPUs evenly, so a type × assets mesh would leave a GPU idle or need the deferred uneven groups. Stage 3's sequential type blocks on all four GPUs already cover three types on one node, and Stage 8 covers more than one node. Revisit only for a workload whose type count divides the per-node GPU count.

## 8. Stage 5 — Type-aware simulation and lifetime retention

### 5A. Group subjects without changing their random streams

The existing subject-parallel route partitions per-person argument trees but replicates shared continuation/policy arguments. Extend its existing transfer owner to request type-specific views. Do not add a second residency manager. [S9]

Build a stable internal mapping from original subject identity to component group, local row, and original output row. Carry already-constructed random keys through this mapping. Do not regenerate seeds from GPU rank, local row, or type-group order. Preserve existing draw-rounding barriers and padded-tail behavior.

Partition subjects independently within each type group, and restore original public ordering at the result boundary. Test unbalanced groups, empty groups, missing/irrelevant state cells, deaths, remainder tiles, and changed seeds/parameters. Grouping is permitted only when simulate-phase invariance was certified.

For latent-type likelihoods, preserve the requested subject/type contributions and original common-random-number relationship. Grouping is not permission to assign one type instead of integrating over all required types.

**Gate:** bitwise identical panels, including storage bytes/signed zeros, with the same input solution. Then repeat end-to-end with the new solve route. Measure both kernel and public-call time.

### 5B. Introduce block-major lifetime execution with explicit retention

Only after selected views/stores work, schedule a component block through its whole backward induction and subsequent simulation before moving to the next block. This is the stronger `solve block -> simulate block -> release working data` optimization.

Preserve the existing result contract. A normal solve requesting complete values must still produce a complete logical `ValueStore`; use existing lazy stores and persistence infrastructure for block-backed entries. Metadata inspection must not materialize all values. Engine consumers need selected reads rather than materializing a complete array to take a slice. [S10]

Keep requested outputs and their omission records truthful. Releasing a component's active GPU data is legal only after remaining consumers are finished and requested values are retained in an allowed representation. Do not make complete-result calls silently return summary-only or partial solutions.

Define archive ownership, completion manifests, cleanup behavior, and interrupted-run validity. Required tests include ordinary `solve` then `simulate`, combined execution, save/load, selective materialization, object lifetime after the producing scope exits, and refusal/explicit accounting when full materialization exceeds the available budget.

Benchmark I/O and retained host memory as well as GPU memory. A reduction in live GPU storage is not an end-to-end speedup when archive traffic dominates. Keep this optional until the complete workflow wins on its intended use case.

## 9. Stage 6 — Share only genuinely type-independent calculations

**Deliverable:** bounded shared evaluation where compiler/profiler evidence identifies repeated work.

Build conservative transitive dependency summaries from the accepted canonical DAG: state/action coordinates, period/age, runtime parameters/grids, and permitted captured dependencies. Use existing validation/fingerprinting trust boundaries. Unknown/opaque dependencies mean “not shareable”; do not add an unrelated permissive purity whitelist.

Distinguish three facts: a coordinate never changes; a function does not depend on it; and two stored results happen to be numerically equal. Only the first two justify these optimizations. Type-dependent transition probabilities or next-state coordinates do not prevent independent type solving, but do prevent sharing the corresponding preparation across types.

Start with the pinned ACA's separated preference and non-preference DAG functions as profiling candidates, not an assumed complete shareable set. [S12]

Split interpolation conceptually into coordinate generation, index/weight preparation, and weighted value reads. Share preparation only when coordinates, grids, boundary/extrapolation rules, and interpolation semantics agree. Values remain type-specific. Preserve corner order, weights, exceptional-value handling, and reduction order.

Prefer existing vectorization/broadcasting within a bounded state/action tile and a small local type batch. Inspect optimized HLO before adding explicit producers: the compiler may already avoid duplication. If materialization is useful, publish only contract-compatible results through the existing program graph's internal outputs/inputs and include residency/liveness in admission. Keep width-dependent scratch inside the core: the existing internal-output contract requires published shapes, dtypes, and weak typing to be invariant across a producer's width candidates. Do not disable that check to export a tile-sized table. Any genuine tile-scoped producer/consumer extension requires its own bounded contract change, not a hidden relaxation. [S4, S16]

Do not demand mutually incompatible savings: sequential width-one type execution cannot share transient work with a simultaneously absent type unless that work is retained. Start shared evaluation with local type batches of size two/three; compare their increased continuation footprint with their arithmetic saving. Do not precompute a full state × action × shock × type × period table.

Keep caches within the current solve first. Cross-call numerical reuse requires complete dependency-based invalidation and belongs to a later estimation optimization. Runtime preference parameters remain dynamic; they must not become static compilation constants.

**Gate:** unchanged numerical results and parameter-change behavior, bounded intermediates, and measured reduction in work/runtime. Retain a plain path when sharing increases traffic or worsens fusion.

## 10. Stage 7 — Action partitions using the existing exact reducer

**Deliverable:** an optional exact action-parallel GridSearch execution route.

Distribute canonical action-ID intervals within one component/state group. Each participant evaluates the unchanged Q/feasibility semantics and reduces locally. Exchange and merge compact accumulators rather than the full state–action tensor.

Use `HARD_MAX_REDUCTION` and its declared merge/finalize semantics, not separate ad hoc reductions of values and action IDs. Its current behavior includes lowest-global-ID ties, feasible `-inf` versus all-infeasible cells, feasible-NaN identity conventions, and signed-zero handling. [S8]

A first implementation may gather a small number of local accumulators and apply the existing merge in a fixed bounded order. Account for that accumulator workspace. A later custom collective requires the same numerical proof. Padded action slots are infeasible and never enter public action identities.

Do not extend EV1 or collective reductions merely because a max reduction distributes. Their canonical reduction contracts require separate work. Test different partition counts/orders, equal values, opposite signed zeros, infinities, feasible/infeasible NaNs, empty partitions, and nondivisible action counts.

Benchmark regimes with small state products and large action products. Compare action-only and state × action layouts using the same type-local continuation. This optimization can proceed after Stage 3 without waiting for Stage 6.

## 11. Stage 8 — Multi-node execution through independent node-local jobs

### 8A. Independent node-local component jobs

Use the validated block-major engine to assign independent type blocks to node-local jobs. A job can use all GPUs on its node without exchanging continuation values with other type jobs. The launcher supplies a task manifest to the same engine; it does not reconstruct different economic models or add another Bellman implementation.

Fragments retain the original model/parameter identity plus explicit coverage metadata. The collector verifies disjointness, completeness, schema, and checksums before exposing a complete result. Missing, duplicated, or failed components are errors, not completed solutions. Final likelihood/moment aggregation preserves its numerical contract and ordering.

This first distributed deliverable does not require MPI inside JAX. It scales independent components and independent parameter/scenario evaluations; measure throughput separately from the latency of one solve.

A single component spanning several nodes (native multi-process JAX, explicit MPI) was removed from the plan on 2026-10-02. One preference type fits on one 4×A100 node with room to spare, so independent node-local jobs are the only multi-node route.

## 12. Cross-stage correctness and performance gates

### Required correctness matrix

| Case | Required result |
|---|---|
| One type, optimization off/on | Same logical output; no unnecessary work introduced |
| Two/three types, different preferences | Every type gets its own values and policy |
| Selected block not starting at type zero | Original codes/parameter rows preserved |
| Type-dependent transitions of other states | Independent solving allowed; unsafe sharing disallowed |
| Reset/re-entry or cross-type value read | Correct explicit dependency or pre-dispatch refusal |
| Type-free and type-dependent terminal children | Correct shared versus selected reads |
| Permuted state axes and labels | Same labelled canonical output |
| Dense, streamed, and remainder blocks | Existing numerical/reduction contract preserved |
| NaNs, infinities, signed zeros, ties, all-infeasible cells | Existing hard-max behavior preserved |
| Same-shape types and changed parameters | No result-cache collision or stale reuse |
| Shared owners and concurrent transfers | No premature deletion/donation; truthful accounting |
| Subject regrouping, empty groups, tails | Original identities, random keys, and panel bytes |
| Split/combined solve-simulate and save/load | Complete and equivalent requested results |
| Distributed disagreement or missing output | Coordinated failure; no partial success publication |

Use independently constructed tiny-model oracles plus the pinned #482 path. Require existing exact/bitwise gates on their supported stack, including transformed execution and negative one-ULP/signed-zero comparator controls. Mathematical equivalence is not a license to introduce a blanket tolerance. Where existing state-sharding behavior already has documented floating-point limits, record them and compare like-for-like; investigate any newly introduced change before widening an acceptance contract. [S8, S9]

The user-approved exception is general blocked versus independently compiled
unblocked **published values**, bounded by eight ULP at each of fp32/fp64. It does
not relax policies, states, actions, regime/subject identity, schema, shape, dtype,
or sharding. Same compiled-program and same-input-solution byte gates, including
signed zeros, remain binding. The existing accepted Stage 7 one-ULP exception and
all certificate mutation controls are unchanged. Comparator controls must reject
nine value ULP and a one-ULP or signed-zero structural change.

Test fp32/fp64, default and explicit budgets, logging modes, relevant JIT configurations, and changed numerical parameters. Preserve the currently supported JAX transformations and autodiff behavior on differentiable probes; do not introduce host numerical work or stop-gradient boundaries as an implementation shortcut. Newly unsupported optimized combinations must fail clearly while unchanged unoptimized routes retain support.

### Required performance comparisons

Use the same model, parameters, retained outputs, precision, environment, and hardware for each matched pair. Keep base #482 and candidate optimization-off controls.

Compare: flat asset sharding; sequential type-local execution at fixed widths; sequential type-local execution with independently admitted widths; regular type × state meshes; type-aware simulation; complete block-major workflow including I/O; and, later, shared preparation, action partitions, and multi-node execution.

Report repeated paired medians and dispersion, compilation counts, raw/represented peaks, owned versus replicated bytes, communication/copy evidence, and dense/streamed selection. Use Nsight Systems for timelines and selected Nsight Compute profiles for dominant kernels, outside the timed acceptance runs.

A stage may demonstrate a useful capacity increase without a runtime improvement; label those separately. Do not claim a speedup from narrower scope, excluded I/O, changed grids, or another GPU count. Proposed review triggers are more than 5% warm-time, 10% cold-time, or 10% peak-memory regression on protected controls after accounting for noise; they are investigation thresholds, not promises or automatic waivers. Unexpected compile-count growth or a full-type gather in a certified selected route is a structural failure.

Default promotion requires numerical acceptance, target-hardware evidence, and an explicit selection rule. Do not silently turn bounded experiments into online exhaustive autotuning. Keep width/placement exploration bounded and report candidate counts.

## 13. Delivery order and conditional backlog

| Delivery | Dependencies | Must establish |
|---|---|---|
| 0. Baseline/attribution | #482 | Reproducible numerical and resource oracle |
| 1. Invariant analysis | 0 | Structural independence without placement coupling |
| 2. Selected views/transfers | 1 | Select-before-replicate plus truthful lifetime/admission |
| 3. Sequential type-local GridSearch | 2 | First complete production-useful route |
| 4. Hybrid partitions | — | Skipped: 3 types do not divide a 4×A100 node |
| 5A. Type-aware simulation | 2, 3 | Selective residency and unchanged RNG/panels |
| 5B. Block-major retention | 3, 5A | Complete results with bounded active lifetime storage |
| 6. Shared preparation | 3 | Measured repeated-work elimination |
| 7. Action partitions | 3 | Exact distributed action reduction |
| 8A. Node-local jobs | 5B | Complete validated component aggregation |

Deliver Stages 0–3 first. Later stages must not expand the first implementation's merge criteria. Use one owner for shared execution/transfer contracts; parallelize independent fixtures and benchmark preparation, not competing redesigns of those contracts. Each stage supplies exact source/diff identity, changed interfaces, commands and environments, test counts, raw evidence paths, and implemented/validated/deferred status. Do not reuse receipts across changes that invalidate them. [S3]

Keep these ideas conditional: uneven 3+3+2 groups or work stealing; remote interpolation routing; full custom fusion/Pallas; conditional invariance after irreversible transitions; and cross-call type-result reuse for estimation. Each needs a measured bottleneck and a bounded extension of the same contracts. An asset-neighbor halo is not valid without a proven bound on all required next-state interpolation indices. Mixture-weight-only reuse is legal only when those weights do not enter the Bellman problem. None of these is a prerequisite for the first useful type-local GridSearch release.

**First acceptance target:** on the pinned ACA workload, selecting one preference type before continuation replication permits the assigned asset GPUs to process that type without holding other types' active continuation replicas, while preserving values, policies, RNG, results, and admission. Establish that before adding communication sophistication.

## Sources

[S1] pylcm PR #482 metadata, inspected October 1, 2026: https://github.com/OpenSourceEconomics/pylcm/pull/482

[S2] Pinned environment and ACA dependency: https://github.com/OpenSourceEconomics/pylcm/blob/2e2392eb1eeca117cb09f808d6ff575f5db77a88/pyproject.toml

[S3] Repository operating/development contract: https://github.com/OpenSourceEconomics/pylcm/blob/2e2392eb1eeca117cb09f808d6ff575f5db77a88/AGENTS.md

[S4] Core programs, value reads, and internal producer/consumer contracts: https://github.com/OpenSourceEconomics/pylcm/blob/2e2392eb1eeca117cb09f808d6ff575f5db77a88/src/_lcm/execution/core_program.py

[S5] Value transfer addresses, layouts, costs, and cache protocol: https://github.com/OpenSourceEconomics/pylcm/blob/2e2392eb1eeca117cb09f808d6ff575f5db77a88/src/_lcm/execution/value_transfer.py

[S6] Existing backward induction, scheduler and memory integration: https://github.com/OpenSourceEconomics/pylcm/blob/2e2392eb1eeca117cb09f808d6ff575f5db77a88/src/_lcm/solution/backward_induction.py

[S7] Placement and mesh-size rules: https://github.com/OpenSourceEconomics/pylcm/blob/2e2392eb1eeca117cb09f808d6ff575f5db77a88/src/_lcm/execution/placement.py

[S8] Exact hard-max accumulator and merge semantics: https://github.com/OpenSourceEconomics/pylcm/blob/2e2392eb1eeca117cb09f808d6ff575f5db77a88/src/_lcm/solution/action_reduction.py

[S9] Subject sharding, random-stream contract, residency, and current limitations: https://github.com/OpenSourceEconomics/pylcm/blob/2e2392eb1eeca117cb09f808d6ff575f5db77a88/docs/user_guide/subject_parallel_simulation.md

[S10] Lazy canonical value and artifact stores: https://github.com/OpenSourceEconomics/pylcm/blob/2e2392eb1eeca117cb09f808d6ff575f5db77a88/src/lcm/_solver_api/stores.py

[S11] ACA pinned model states and fixed preference-type transition: https://github.com/OpenSourceEconomics/aca-model/blob/ad38653696ec366e318ac61b9a81b597a4ecb700/src/aca_model/baseline/regimes/_common.py#L790-L828

[S12] ACA pinned DAG functions: https://github.com/OpenSourceEconomics/aca-model/blob/ad38653696ec366e318ac61b9a81b597a4ecb700/src/aca_model/baseline/regimes/_common.py#L690-L747




[S16] Width-invariant internal-output contract: https://github.com/OpenSourceEconomics/pylcm/blob/2e2392eb1eeca117cb09f808d6ff575f5db77a88/src/_lcm/execution/internal_outputs.py
