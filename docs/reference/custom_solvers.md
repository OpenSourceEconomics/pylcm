---
title: Custom solvers
---

# Custom solvers

A solver can be written outside pylcm against public names only. The surface is
`lcm.solvers`, `lcm.solver_api`, `lcm.typing`, and `lcm.grids`; nothing in a custom
solver needs to import `_lcm`. The contract covers solve programs, keyed continuations,
model-authoritative replay, durable result persistence, and exact version identities. It
is an exact-version extension contract: a matching version is supported, while an
adapter or automatic migration across versions is not implied.

## What a solver owes the engine

A solver is a class deriving from `Solver` with one abstract method to implement,
`build_period_kernels`. The shipped solvers are frozen dataclasses because they carry
numerical configuration; a solver with no configuration needs no fields. Between them, a
solver and its kernels answer three questions.

- **Which continuations do I read?** `required_continuation_keys` returns a frozenset of
  `ArtifactKey`. Model building checks every key against what each reachable target
  regime publishes and refuses the model, naming both regimes and the demanded version,
  before anything compiles. Grid search returns the empty set; every endogenous-grid
  solver returns `{EGM_CONTINUATION}`.
- **What does a period compute?** `build_period_kernels(context=...)` returns
  `SolutionKernels` holding one period kernel per active period, and optionally a
  `ContinuationSpec` naming the artifact those kernels publish.
- **Which periods did I build alike?** Publish each period's group key in
  `SolutionKernels.period_group_keys`. The engine folds that key into the compiled
  program's identity beside its own per-period signature, so two periods share one
  executable only when both groupings agree. Whenever you build per period, publish a
  per-period key — `(your_solver_name, period)` will do; the engine's signature is free
  to merge periods your builds separate (a terminal regime's signature is the same at
  every period), and handing it an identity coarser than your own specialization is what
  the refusal below is about. Reserve the empty mapping for the two cases where you have
  nothing to add: one build serves every period, or the engine's own signature already
  separates your builds. The key has to be **durable**: build it from declared
  signatures, names, and literals, never from `id()`, because it is compared across
  model constructions. Publishing a key too coarse for what the solver actually
  specialized is refused at build time with an `ExecutionPlanningError`, naming both
  colliding programs, rather than running one period's closure in another period's
  place. The periods you group under one key must also reach **one callable object**:
  building a fresh, equivalent closure per period — or a `functools.partial` over
  equal-but-distinct bound values — is refused by that same `ExecutionPlanningError`, so
  a grouping solver builds once and hands every period in the group the object it built.
- **What does one period publish?** Each kernel declares a native core-program graph
  through `core_programs()` and returns a `KernelOutput` from its call.
- **How is my decision replayed?** `SolutionKernels.replay_route` names how simulation
  obtains the solved decision: an `ExecutableReplayRoute`, or one of the two
  `DeclaredReplay` values. Leaving it unset is a build error for any solver outside the
  shipped set; see [Declared replay routes](#declared-replay-routes).

## A minimal solver

The solver below publishes one dense program whose value is the regime's own wealth
grid. It is the shape every custom solver starts from: declare the program, build its
arguments from the build context, return a `KernelOutput`.

```python
import dataclasses
from collections.abc import Mapping
from types import MappingProxyType

from lcm.solvers import (
    CoreBuildContext,
    CoreExecutionDisposition,
    CoreExecutionRequirements,
    CoreProgram,
    DeclaredReplay,
    KernelOutput,
    OutputRole,
    SolutionKernels,
    Solver,
    SolverBuildContext,
    SolverIdentity,
)
from lcm.typing import Float1D


def wealth_value(*, wealth: Float1D) -> Float1D:
    """One value per state node: the wealth itself."""
    return wealth


@dataclasses.dataclass(frozen=True, kw_only=True)
class WealthKernel:
    """A period kernel that dispatches its single declared program."""

    programs: Mapping[str, CoreProgram]

    def core_programs(self) -> Mapping[str, CoreProgram]:
        return self.programs

    def with_fixed_params(self, *, fixed_flat_params: object) -> "WealthKernel":
        return self

    def __call__(
        self,
        *,
        compiled_cores: Mapping[str, object],
        state_action_space: object,
        next_regime_to_V_arr: Mapping[str, object],
        next_regime_to_continuation: Mapping[str, object],
        flat_params: Mapping[str, object],
        period: int,
        ages: object,
        **_unused: object,
    ) -> KernelOutput:
        context = CoreBuildContext(
            state_action_space=state_action_space,
            next_regime_to_V_arr=next_regime_to_V_arr,
            next_regime_to_continuation=next_regime_to_continuation,
            flat_params=flat_params,
            period=period,
            ages=ages,
        )
        arguments = self.programs["main"].argument_builder(context)
        return KernelOutput(value=compiled_cores["main"](**arguments))


class WealthSolver(Solver):
    """Publishes the wealth grid as the value in every active period."""

    @property
    def identity(self) -> SolverIdentity:
        """Return the package-owned compatibility identity."""
        return SolverIdentity(
            plugin_id="example.wealth_solver",
            plugin_version="1.0.0",
        )

    def build_period_kernels(self, *, context: SolverBuildContext) -> SolutionKernels:
        program = CoreProgram(
            name="main",
            function=wealth_value,
            argument_builder=lambda build: {
                "wealth": build.state_action_space.states["wealth"]
            },
            requirements=CoreExecutionRequirements(),
            output_roles=OutputRole.VALUE,
            disposition=CoreExecutionDisposition.DENSE,
            disposition_reason="one_row_per_state_node",
        )
        return SolutionKernels(
            period_kernels=MappingProxyType(
                {
                    period: WealthKernel(programs=MappingProxyType({"main": program}))
                    for period in context.regimes_to_active_periods[context.regime_name]
                }
            ),
            replay_route=DeclaredReplay.GRID_RECOMPUTATION,
        )
```

A program declares its disposition explicitly, and the cases are mutually exclusive
rather than a default plus an override. `DENSE` means the solver, not the planner, owns
the width its body runs at, and it must carry a non-blank `disposition_reason` saying
why. `HOST_DRIVEN` means the same, and adds that a host loop dispatches the compiled
program a data-dependent number of times — the driver that owns the loop also owns the
results it caches between dispatches — so it too must carry a reason. `PLANNED` hands
the width choice to the engine and must *not* carry a reason; declaring one is refused.
A planned program declares whichever axes the engine may stream — `reduced_axes` for an
axis folded by a reduction, `tiled_axes` for one whose tiles are concatenated — and a
solver whose body streams nothing declares an empty set; the shipped NB-EGM graph does
exactly that. Only a planned program may declare an execution axis.

A `ReducedAxis` is the Cartesian product of the grids in `coordinate_names`, counted in
`canonical_order`, and folded to a single result by its `reduction`. A `TiledOutputAxis`
is an output axis over the states in `state_names`: its tiles are concatenated, never
folded, so the published array is the same whatever width the tiles run at. Both name
the planner-visible axis in `name`, take the compiled width through `width_keyword`, and
constrain the widths the planner may pick with `minimum_width` and `alignment`: the
widths an axis admits are its full extent, whatever the alignment divides, plus every
multiple of `alignment` between `minimum_width` and that extent. A proposal is rounded
down onto that set, and one that falls through it — below the floor, or below the
alignment and so at zero — is lifted to the narrowest width the set holds, which is the
extent when no multiple of the alignment reaches the floor without passing it.
`ACTION_PRODUCT_AXIS` is the name the shipped `GridSearch` gives its action product, and
`OUTER_CANDIDATE_AXIS` the name a nested outer search gives its exogenous post-decision
candidates; each is the name to pass when fixing that solver's width.

`ExecutionConfig(axis_widths=...)` fixes the compiled width of a declared axis by its
name. It is hardware-local: it changes what is compiled, never what is published, and
never enters the durable model fingerprint. A width above an axis's extent is taken as
the extent, and a name no program of the model declares is refused with the declared
names listed, so a typo cannot pass as a tuning choice.

A reduced axis names its reduction at one of two levels. `ReductionDeclaration` is the
contract: a stable `semantic_key`, which enters static program identity so two programs
folding the same axis differently never share a compiled executable, and an `exactness`
— `"exact"` when block order cannot move the published value, `"tolerance_equivalent"`
when results agree to the working format's rounding. That pair is what the axis
references and what the engine validates, and it is all a solver owes when the fold
kernel belongs to the solver's own body: the planner's width reaches such a kernel
through the axis's `width_keyword`, as it does for any streamed core.

`ReductionSemantics` is a declaration that also publishes the fold itself, so the
planner may drive it block by block: `initialize` builds one accumulator from a value
template, `add` folds one block of candidates into it, `merge` combines two accumulators
covering disjoint blocks, and `finalize` turns an accumulator into the published result.
A reduction at this level publishes the same value whichever partition the planner
picks, and carries the dense argmax identity — for the hard maxes, the winner is the
candidate at the first canonical position attaining the maximum, whatever the block
boundaries are.

`WeightedExpectationReduction` (a probability-weighted sum over stochastic nodes) and
`IntervalEnvelopeReduction` (an upper envelope over candidate intervals) are
declarations: each names a contract that a solver's own kernel fulfils.
`HardMaxWithCarryReduction` publishes the full fold over an `OuterCandidateAccumulator`,
whose state is the running winner's value, its global candidate id and the payload that
winner carries; `add` folds a block of values, and a driver that owns a payload per
candidate merges that block's state in so the carry travels with the winner it belongs
to. The reductions the shipped `GridSearch` body owns — the hard max over the action
product, its collective counterpart, and the logsumexp under taste shocks — publish
their fold too, so the planner can drive them directly. `EXACTNESS_VALUES` holds the two
spellings an `exactness` may take, so a custom reduction can be checked against the
published set rather than against a literal.

`donation_candidates` names arguments the engine may donate to the compiled program. An
argument is donated when every artifact it carries by a declared `ValueRead` addressed
to it by name:

- is not the solve-lifetime template, the one input whose declared period lies beyond
  the model's last period;
- has this dispatch as its sole remaining reader;
- is not retained by the result;
- is not pinned by an undeclared reader;
- shares its buffer with no other key;
- reaches the program on its stored layout rather than as a transferred copy.

A donated input is unreadable after the call, so a builder must not keep a reference to
it.

(internal-outputs)=

## Internal outputs

A kernel that publishes more than one program can hand one program's output to another
as an argument, instead of lowering the consumer against a stand-in it fills in later.
The producer declares what it publishes and the consumer declares what it reads:

```{code-block} python
producer = CoreProgram(
    name="keeper",
    function=keeper_body,
    argument_builder=build_keeper_arguments,
    requirements=CoreExecutionRequirements(),
    output_roles=(VALUE, {"carry": VALUE}),
    disposition=CoreExecutionDisposition.DENSE,
    disposition_reason="one_row_per_state_node",
    internal_outputs=(
        InternalOutputSpec(label="value", path=(0,)),
        InternalOutputSpec(label="carry", path=(1,)),
    ),
)
consumer = CoreProgram(
    name="outer_sweep",
    function=sweep_body,
    argument_builder=build_sweep_arguments,
    requirements=CoreExecutionRequirements(
        internal_inputs={
            "keeper_value": InternalInputRef(producer="keeper", label="value"),
            "keeper_carry": InternalInputRef(producer="keeper", label="carry"),
        }
    ),
    output_roles=VALUE,
    disposition=CoreExecutionDisposition.DENSE,
    disposition_reason="one_row_per_outer_node",
)
```

`InternalOutputSpec.path` is a pytree path into the producer's raw output, so a label
may name a whole subtree rather than a single leaf. `InternalInputRef` is keyed by the
consumer's own argument name, which may not collide with a name its argument builder
already supplies.

The engine reads these declarations at three moments:

- **When the graph is built.** Every reference must name a program of the same graph and
  a label that program declares, labels within one producer must be unique, and the
  references must not form a cycle.
- **When a retention selects the graph's programs.** A retention that keeps a consumer
  must also keep every producer it reads, so a producer and its consumers belong in
  scopes that are selected together.
- **When the period is lowered.** The engine visits producers before consumers and
  traces each producer once with everything it is lowered with — the arguments its
  builder returned, the templates of the internal inputs it reads itself, and the widths
  the execution planner owns — then lowers the consumer against the exact shapes, dtypes
  and weak typing of the subtrees its references select. Weak typing belongs in that
  template because equal shapes and dtypes can still promote differently in a consumer:
  a weakly typed leaf takes the other operand's dtype, a strongly typed one forces its
  own. Those templates are part of the program's compilation identity, so two cells that
  differ only in an internal input's shape do not share an executable. A producer whose
  published subtree would change with the width the planner selects is refused while the
  period is planned: its consumers are lowered before that selection is made.

A typed internal edge is the route between two programs the engine lowers together. A
`HOST_DRIVEN` program is dispatched by the solver's own host loop, so a driver that
feeds one program's result into the next dispatch holds that result on the host and
passes it through its argument builder; the engine plans nothing for it and there is no
internal reference to declare.

At dispatch the compiled core refuses an internal input that is missing, or whose shape,
dtype, or weak typing departs from the template it was lowered against, naming the
program and the argument.

(reading-a-stored-value)=

## Reading a stored value

Every array a program reads across a regime-period boundary is declared rather than
discovered. A `ValueRead` names both ends independently: the stored artifact it reads
through a `ValueArtifactAddress`, and the argument leaf that receives it through a
`ValueConsumerAddress`. A program declares its reads on its requirements.

```{code-block} python
from lcm.solvers import (
    CoreExecutionRequirements,
    ValueArtifactAddress,
    ValueArtifactKind,
    ValueConsumerAddress,
    ValueInputChannel,
    ValueRead,
)


def wealth_requirements(
    *, regime_name: str, target: str, period: int
) -> CoreExecutionRequirements:
    """Declare that this core reads one target regime's next-period value."""
    return CoreExecutionRequirements(
        value_reads=(
            ValueRead(
                target=ValueArtifactAddress(
                    kind=ValueArtifactKind.REGIME_VALUE,
                    period=period + 1,
                    regime=target,
                ),
                source=ValueConsumerAddress(
                    source_period=period,
                    source_regime=regime_name,
                    core_key="main",
                    channel=ValueInputChannel.NEXT_REGIME_VALUE,
                    path=(target,),
                ),
            ),
        )
    )
```

An economic dependency points from a source regime to a target regime, while the stored
value moves the other way during backward induction, so the two addresses carry
different coordinates: the artifact's `period` is the target's solved period, and the
consumer's `source_period` is the period of the core that reads it. The artifact's
`kind` is `ValueArtifactKind.REGIME_VALUE` for a regime's solved value and
`ValueArtifactKind.GATED_CONTINUATION` for a gated edge's folded continuation.

The engine then names one operator per declared read, from the layout the value is
stored in to the layout the consuming program requires:

| stored layout              | required layout                             | operator                           |
| -------------------------- | ------------------------------------------- | ---------------------------------- |
| equal                      | equal                                       | `ALIGNED_LOCAL`                    |
| either side not mesh-named | any different placement                     | `COPY_TO_SOURCE_LAYOUT`            |
| sharded on named axis      | replicated                                  | `ALL_GATHER`                       |
| replicated                 | sharded on named axis                       | `LOCAL_SLICE`                      |
| sharded on axis `a`        | sharded on axis `b`                         | `RESHARD`                          |
| one mesh, given axes       | same mesh and axes, different `memory_kind` | `RESHARD`                          |
| any                        | different mesh (disjoint or nested submesh) | `CROSS_MESH_COPY`                  |
| any                        | overlapping but unequal meshes              | refused (`ExecutionPlanningError`) |

"Mesh-named" is a layout that names a device mesh and a partition spec over it; a value
pinned to one device is not, so any read whose two ends are not both mesh-named is a
copy onto the required placement, whether the stored value was sharded or not. A pair
that agrees on mesh and axes while differing in some other attribute takes the
conservative reading — a recorded representation change, not a silent no-op.

The table is total over the layout pairs the planner produces, so a declared read always
has exactly one operator. The last row is the one pair no single collective serves: two
meshes sharing devices while neither contains the other are refused while the period is
planned, naming both device sets, rather than moved through a placement the plan does
not record.

Lowering and dispatch apply the same immutable plan. An `ALIGNED_LOCAL` read hands the
stored array to the program unchanged; every other operator is one recorded copy onto
the required layout, and the compiled program refuses a value whose shape, dtype, or
layout departs from what was planned.

The engine declares the reads of its own gated-edge fold the same way a solver declares
a core's. Folding one declared edge onto its target's grid reads that period's target
value and every reference regime's value, so each is a `ValueRead` whose consumer names
the fold — the source regime, the folded period, and the edge's target — and the fold is
one dispatch of the solve alongside the period's cores. What a fold reads is therefore
counted, not merely retained.

A program whose builder reads a stored value it does not declare, or declares one its
builder does not read, is refused when the program is materialized.
`SolverBuildContext.solution_reachability.targets(period=..., source=...)` returns the
target regimes to declare for one period; the last period of the horizon has none. The
conformance fixture's `TargetValueSolver` is the reference shape.

(publishing-a-continuation)=

## Publishing a continuation

A solver whose parents invert an Euler equation publishes a continuation artifact. The
artifact is any type satisfying the `ContinuationArtifact` protocol, which asks for one
property: `artifact_key`, the versioned identity under which the payload is published.
The engine stores and rolls the artifact without reading its fields, so a solver family
can carry whatever its own parents need.

Three declarations must agree, and each is checked at a different moment, so a mistake
surfaces as early as it can be seen.

- The child solver's `SolutionKernels` carries
  `ContinuationSpec(template=..., artifact_key=key)`, whose template is an all-finite
  payload of the exact shapes the loop rolls and lowers. A template whose own
  `artifact_key` differs from the declared one is refused when the spec is constructed.
- Every parent that reads the continuation declares that same `key` in
  `required_continuation_keys`. A reachable target that publishes another key, or none,
  is refused while the model builds, with both regimes and the demanded version named.
- The child kernel returns `KernelOutput(value=..., continuations={key: payload})`. A
  missing required continuation, a payload under a key other than the child's declared
  `ContinuationSpec.artifact_key`, or a payload that is not a `ContinuationArtifact` is
  refused during the solve, with the regime and period named.

`EGMContinuationSpec` is the shipped specialization: its template is an `EGMCarry`, its
key is `EGM_CONTINUATION`, and it adds the layout properties a reading EGM parent needs.
The engine synthesizes a closed-form carry for a grid-search target only under
`EGM_CONTINUATION`; a solver family that invents its own key publishes it from its own
kernels in every regime it reads.

### What a reader answers

A parent asks its target's payload what the continuation is worth at a query rather than
interpolating the target's storage itself, so the two are coupled by a question, not by
a row layout. A payload that answers such questions satisfies `ContinuationReader`:

```{code-block} python
@runtime_checkable
class ContinuationReader(Protocol):
    @property
    def capabilities(self) -> ContinuationCapabilities: ...
    def value_at(self, *, query: FloatND) -> FloatND: ...
    def marginal_at(self, *, query: FloatND, state: StateName) -> FloatND: ...
    def leaves(self) -> Mapping[tuple[str, ...], FloatND]: ...
```

`capabilities` is the payload's own statement of what it can answer:

- `value` — whether `value_at` returns a continuation value.
- `marginal_states` — the states `marginal_at` accepts; any other state is refused.
- `exact_candidate_identity` — whether the payload names which candidate owns a query
  point.
- `discontinuities` — whether the payload locates its own one-sided boundaries.

A parent states what it needs of its targets in `required_continuation_capabilities`,
which defaults to `ContinuationCapabilities()` — asking nothing, so the payload is
rolled opaquely and the parent reads its fields itself. Every endogenous-grid solver in
pylcm demands the value and the marginal in `EGM_ENDOGENOUS_COORDINATE`, the coordinate
an EGM carry's rows are tabulated on. Model building compares each demand against every
reachable target's published payload, so a target of a querying parent that publishes no
reader at all, or one answering less than that parent asks, is named while the model
builds rather than during the solve.

`leaves()` is the addressable content of the payload: every published array under its
own pytree path.

### Declaring the leaves you read

A solver that reads its targets' carries cannot name the rows it reads while its own
kernels are being built: no regime has published a template yet, and the templates are
what say which rows exist. `Solver.declare_continuation_reads` is the second call for
exactly that:

```{code-block} python
class MySolver(Solver):
    def declare_continuation_reads(
        self, *, kernels: SolutionKernels, context: SolverBuildContext
    ) -> SolutionKernels:
        """Attach the reads each period's programs make on its targets' rows."""
        return attach_my_reads(kernels=kernels, context=context)
```

The engine calls it once per regime whose `required_continuation_keys` is non-empty,
after every regime in the model is built, with `context.continuation_specs` mapping each
regime name to the continuation it publishes. A regime absent from that mapping
publishes none. The default implementation returns `kernels` unchanged, so a solver that
reads no continuation — or one content to be pinned conservatively — implements nothing.

The contract is deliberately narrow. The hook may only attach `value_reads` to the core
programs its kernels already publish: the same programs under the same names, with the
same argument builders and the same compiled functions. Building kernels a second time
here would re-run every build-time consumer the model author declared — a compiled
constraint boundary, a boundary plan — and consume each of them twice. The result must
depend only on the arguments, so two builds of one model declare the same reads.

Only the `period_kernels` of the returned container are honoured. A container whose
`period_group_keys`, `continuation_spec`, `artifact_authorities`, `replay_route` or
`param_checks` differ from the ones the engine handed over is refused while the model
builds, naming the regime, the solver and the field — as is a declaration by a solver
whose regime publishes no continuation of its own, because the engine wraps such a
regime's kernels in the adapter that publishes one and the declarations would name
programs that adapter does not publish.

The shipped endogenous-grid solvers build one `ValueRead` per published leaf, addressed
either inside the rolling `next_regime_to_continuation` mapping or, where the argument
builder flattens the rows into named arguments, by the argument holding each row. A
period-`t` read names the target's carry at `t + 1` and the consumer at `t`.

## Declared replay routes

Every regime declares exactly one replay route, reachable as
`regime.simulation.replay_route`, and simulation dispatches on it rather than on the
class of whatever payload a solve happened to retain. A route names its `replay_mode`:

- `EXACT_REPLAY` — the decision comes from a retained payload, read at the subject's
  realized state. The route's `payload_type` is the exact class the solve must retain,
  and the pre-simulation check refuses any other.
- `VALID_RECOMPUTATION` — nothing is retained and the decision is recomputed on the
  regime's own action grids. `payload_type` is `None`. This is what a grid-search
  regime, and any regime whose configured search publishes no payload, declares.
- `UNSUPPORTED` — the solve's decision can be reproduced neither way, so simulating the
  regime is refused with a message naming the reason.

A shipped solver leaves `SolutionKernels.replay_route` unset and the engine reads its
decision through its own adapters. Every other solver must declare the route itself, and
a model whose external solver leaves it unset is refused when the model is built. Two
declarations need no code of their own:

- `DeclaredReplay.GRID_RECOMPUTATION` — the solver's decision is exactly the argmax over
  the regime's declared action grids at the subject's realized state, so simulation
  recomputes it there. This is what the example above declares: it retains no payload,
  and the regime's own utility and continuation define its decision.
- `DeclaredReplay.UNSUPPORTED` — the decision cannot be reproduced. The solve stays
  available, and `simulate()` raises `UnsupportedOperationError` naming the regime and
  the solver before any forward step.

An external solver that needs its own payload implements `ExecutableReplayRoute` and
returns it as `SolutionKernels(replay_route=...)`. The route supplies:

- a package-owned `SolverIdentity` and `ReplayRouteIdentity`;
- `requirements(context=...)`, declaring the exact artifact keys consumed for the
  period-specific model view;
- an `ArtifactAuthority` for each required key and solution cell;
- `validate(snapshot=..., context=...)` for solver-specific mathematical invariants; and
- `build_reader(snapshot=..., context=...)`, returning a JAX-transformable
  `ReplayReader` whose result is an `ActionOutput` mapping named actions to arrays.

`ArtifactAuthority` is constructed from the current model and route. It owns the exact
payload and container runtime types, `TreePath`-addressed numerical leaves, named-axis
roles and coordinates, state and action roles, categorical domains, required consumer,
and applicability. Its separate `ArtifactDescriptor` carries the transport-safe copy of
those facts together with the key, channel, payload identity, requiredness, and
persistence policy. A `MODEL_VERIFIABLE` artifact may be saved because another process
can reconstruct and check its authority independently. A dynamic artifact whose exact
axes exist only as a solve-side fact must declare `NOT_PERSISTED` unless its descriptor
carries those axes as data a consumer can check on its own, which is how pylcm's
adaptive NNBEGM policy carries its outer nodes and how solver diagnostics carry their
layout.

Before forward execution, pylcm checks the archive and solver-interface versions, model
and parameter fingerprints, plugin and route identities, key versions, coordinates,
channels, requiredness, shapes, dtypes, and the model-built authorities. It materializes
the required lazy entries once. At authority declaration it invokes a plugin PyTree's
flatten callback exactly once, then invokes its unflatten callback once with opaque leaf
tokens to compile a sealed construction plan. Later materialization copies numerical
leaves into private buffers and reconstructs fresh exact tuples or structurally closed
dataclass records from that plan without calling either plugin callback.
PyTree-represented static metadata is validated; callback-injected instance state is
canonicalized to the declared plan. The resulting owned snapshot is supplied to the
route's `validate` and `build_reader` methods. This ownership boundary does not sandbox
installed plugin validation or reader code, and a route cannot authorize itself from a
descriptor copied out of the result.

`ReplayModelContext` and `SimulationBuildContext` expose the same period-specific
solve-grid view: `state_names` and `action_names` are the canonical solution axes, and
their node mappings contain exactly those named grids. A state declared with
`Phased(solve=callable, simulate=Grid)` is carried per subject only during simulation;
it is therefore not an artifact axis and does not appear in either build context. The
reader still receives that carried state in its per-subject `states` mapping at runtime.

The reader receives only this public `SimulationBuildContext` and the validated
`ReplayRouteSnapshot`. Its call has this shape:

```text
reader(states={...}, fallback_actions={...})
    -> ActionOutput(actions={"consumption": ...})
```

It must be pure and JAX-transformable. Every declared action is returned by name as a
scalar or an array broadcastable to one entry per subject; it must not invoke Python I/O
or inspect an engine-private object.

## Persistence

`save_solution(solution=..., path=...)` stores public metadata, omissions, values, and
every present artifact whose descriptor declares `MODEL_VERIFIABLE`. Each numerical
entry is independently addressed and checksummed; the archive contains no plugin class,
callable, pickle, or executable code. An emitted artifact declared `NOT_PERSISTED` is
replaced in the restored result by an explicit omission with that reason.

Loading does not import a plugin named by archive metadata. Without the plugin, pylcm
can inspect metadata and omissions, verify checksums, lazily read ordinary array
entries, and read solver diagnostics, whose layout their descriptor fixes completely. A
plugin-defined PyTree stays uninterpreted until a model with the matching installed
route supplies its trusted template during replay. A restored result saves again without
a model: each payload is re-read from its archive and verified before it is written.

Compatibility is exact for `SOLVER_API_VERSION`, the archive and solution schema
versions, `SolverIdentity`, `ReplayRouteIdentity`, and every
`ArtifactKey.schema_version`. Changing a payload's meaning requires a new artifact
schema version. Changing route semantics requires a new route version. pylcm rejects
incompatible persisted results clearly; plugins own any migration they choose to provide
outside the replay path.

Custom artifact authorities must use plugin-owned type IDs. The built-in
`SIMULATION_POLICY`, `DISSOLUTION_FLAG`, `EGM_CONTINUATION`, and `SOLVER_DIAGNOSTICS`
type-ID namespaces (including other schema versions) are reserved for the engine's own
channel readers.

## Conformance contract

The repository carries an executable out-of-tree reference fixture, exercised by pylcm's
focused tests, that imports only `lcm.solvers`, `lcm.solver_api`, and `lcm.typing`. It
is a deliberately small two-state solver and establishes this minimum acceptance
contract:

1. declare a package identity and build all kernels through `SolverBuildContext`;
1. declare how every regime's decision is replayed — an executable route, grid
   recomputation, or an explicit refusal — and read stored next-period values only
   through declared target-value accesses;
1. publish retention-specialized `PLANNED` programs with a named `candidate`
   `ReducedAxis`, a custom reduction semantic key, exact `retained_artifact_keys`, an
   exact `retained_artifact_payload_types` entry for every retained key, an explicit
   `replaces_program` link from replay to values, and `StateAxesLeading` output roles,
   plus an additive artifact-only scratch program;
1. return `KernelOutput` with a non-EGM `Counter` continuation declared `NOT_PERSISTED`
   and a scratch auxiliary declared `MODEL_VERIFIABLE`;
1. publish a registered plugin-defined PyTree as a `MODEL_VERIFIABLE` replay artifact
   through an `ExecutableReplayRoute` with durable plugin and route identities;
1. exercise solve/result retention, omission records, custom tied-action replay, and a
   JAX-transformed reader;
1. save, load independently lazy entries, construct a fresh compatible model, validate
   the route, build its reader, and simulate from the restored result; and
1. reject structurally or mathematically invalid replay artifacts during preflight.

The fixture proves that the common planner and replay boundary need no engine-side
branch for this solver. It is reference source inside pylcm's test suite, not a packaged
or supported user-runnable conformance command. External plugin authors can copy its
contract shape and should reproduce the same matrix with a representative model.

The payload-type declaration names the final artifact published by the period kernel,
after any adapter or composite transformation. Every program retaining the same key must
name the same exact type, and that type must agree with the solver-built artifact
authority and consuming replay route. Conditional publication affects applicability and
requiredness, not the declared type of a payload when it is present.

(status)=

## Status

The contract above is exercised end to end by the in-repository reference solver. Its
source imports nothing from `_lcm`, and the focused tests cover persistence and restored
replay. The contract is supported only for the exact declared versions. pylcm is
pre-1.0, so a future release may deliberately increment `SOLVER_API_VERSION`; a plugin
must then update and re-run its own contract checks rather than assume source or archive
compatibility.

Use a shipped solver from `lcm.solvers` wherever one represents the economic problem,
and `GridSearch` where none does.
