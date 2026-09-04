---
title: Custom solvers
---

# Custom solvers

A solver can be written outside pylcm against public names only. The surface is
`lcm.solvers`, `lcm.solver_api`, `lcm.typing`, and `lcm.grids`; nothing in a custom
solver needs to import `_lcm`. What is still missing before this counts as a *supported*
plugin API is listed under [Status](#status) at the end of this page, and the interface
remains experimental until those exist.

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
    OutputRole,
    SolutionKernels,
    Solver,
    SolverBuildContext,
)
from lcm.solver_api import KernelOutput
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

A program declares its disposition explicitly, and the two cases are mutually exclusive
rather than a default plus an override. `DENSE` means the solver, not the planner, owns
the width its body runs at, and it must carry a non-blank `disposition_reason` saying
why. `PLANNED` hands that choice to the engine and must *not* carry a reason; declaring
one is refused. A planned program declares whichever action axes the engine may stream,
together with the reduction each performs, and a solver whose body streams nothing
declares an empty set — the shipped NB-EGM graph does exactly that.

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
  missing required continuation, a payload under a key no parent reads, or a payload
  that is not a `ContinuationArtifact` is refused during the solve, with the regime and
  period named.

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

A custom solver's regimes declare `VALID_RECOMPUTATION`: publishing a replay payload
requires a reader in the simulation loop, which is engine-side and not part of this
surface.

(status)=

## Status

The names above are importable and tested end to end, including an assertion that the
exercising test module imports nothing from `_lcm`. They are not yet a supported plugin
API, and the following are the concrete gaps:

- `SolutionResult` has no persistence boundary and no durable cross-process identity, so
  a custom solver's artifacts cannot be saved and reloaded;
- there is no external conformance suite a solver can run to establish that it satisfies
  the contract;
- there is no published compatibility or versioning policy for these names.

Until those exist and the maintainers make an explicit stability decision, treat this
page as a contributor-facing description of a moving seam. Progress is tracked in
[issue #422](https://github.com/OpenSourceEconomics/pylcm/issues/422).

Use a shipped solver from `lcm.solvers` wherever one represents the economic problem,
and `GridSearch` where none does.
