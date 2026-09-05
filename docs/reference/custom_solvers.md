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
- **What does one period publish?** Each kernel declares a native core-program graph
  through `core_programs()` and returns a `KernelOutput` from its call.

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
            )
        )
```

A program declares its disposition explicitly, and the cases are mutually exclusive
rather than a default plus an override. `DENSE` means the solver, not the planner, owns
the width its body runs at, and it must carry a non-blank `disposition_reason` saying
why. `HOST_DRIVEN` means the same, and adds that a host loop dispatches the compiled
program a data-dependent number of times — the driver that owns the loop also owns the
results it caches between dispatches — so it too must carry a reason. `PLANNED` hands
the width choice to the engine and must *not* carry a reason; declaring one is refused.
A planned program declares whichever action axes the engine may stream, together with
the reduction each performs, and a solver whose body streams nothing declares an empty
set — the shipped NB-EGM graph does exactly that. Only a planned program may declare a
streamable axis.

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
- **When the period is lowered.** The engine visits producers before consumers, runs
  each producer's function abstractly once, and lowers the consumer against the exact
  shapes and dtypes of the subtrees its references select. Those templates are part of
  the program's compilation identity, so two cells that differ only in an internal
  input's shape do not share an executable.

At dispatch the compiled core refuses an internal input that is missing, or whose shape
or dtype departs from the template it was lowered against, naming the program and the
argument.

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
axes exist only as a solve-side fact must declare `NOT_PERSISTED` until those axes can
be rederived from the model.

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
can inspect metadata and omissions, verify checksums, and lazily read ordinary array
entries. A plugin-defined PyTree stays uninterpreted until a model with the matching
installed route supplies its trusted template during replay.

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
1. publish retention-specialized `PLANNED` programs with a named `candidate`
   `StreamableProductAxis`, a custom reduction semantic key, exact
   `retained_artifact_keys`, an exact `retained_artifact_payload_types` entry for every
   retained key, an explicit `replaces_program` link from replay to values, and
   `StateAxesLeading` output roles, plus an additive artifact-only scratch program;
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
