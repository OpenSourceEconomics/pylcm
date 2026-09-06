"""A published internal edge carries its weak typing through a public solve.

A budgeted solve resolves a planned producer at every width of its frontier and
lowers the consumer against one of them, before the planner selects which width
runs. A scalar whose weak typing follows the width would therefore hand the
consumer arithmetic it was not traced for, so the solve refuses it while the
period is planned. A producer holding one convention at every width — weakly
typed throughout or strongly typed throughout — publishes one subtree and is
admitted.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, cast

import jax.numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal as aaae

from _lcm.execution.core_program import (
    CoreBuildContext,
    CoreExecutionDisposition,
    CoreExecutionRequirements,
    CoreProgram,
    InternalInputRef,
    InternalOutputSpec,
)
from _lcm.execution.output_layout import VALUE, StateAxesLeading
from _lcm.solution.contract import SolutionKernels, SolverBuildContext
from _lcm.typing import FlatParams, FloatND
from lcm import AgeGrid, DiscreteGrid, LinSpacedGrid, Model
from lcm.exceptions import ExecutionPlanningError
from lcm.execution import ExecutionConfig
from lcm.solver_api import KernelOutput, ResultRetention, SolverIdentity
from lcm.solvers import GridSearch, StreamableProductAxis
from tests.conftest import DECIMAL_PRECISION
from tests.test_models.deterministic.regression import (
    START_AGE,
    LaborSupply,
    RegimeId,
    dead,
    get_params,
    working_life,
)

_N_PERIODS = 3
_N_WEALTH = 3
_CANDIDATES = 5

# Frontier width at which the width-dependent producer publishes a weak scalar.
_WEAK_WIDTH = 1

# Budget generous enough to offer the candidate axis its whole width frontier.
_DEVICE_MEMORY_BYTES = 2**32


def _reduced_row(*, x: FloatND, candidate: FloatND, width: int) -> FloatND:
    """Return the state row shifted by the candidate maximum, at any legal width.

    The block count follows the planner's width and the padded tail is `-inf`,
    so the row itself is the same subtree at every width of the frontier.
    """
    pad = (-candidate.shape[0]) % width
    blocks = jnp.pad(candidate, (0, pad), constant_values=-jnp.inf).reshape((-1, width))
    best = jnp.asarray(-jnp.inf, dtype=x.dtype)
    for index in range(blocks.shape[0]):
        best = jnp.maximum(best, jnp.max(blocks[index]))
    return x + best


def _weak_scalar_body(
    *, x: FloatND, candidate: FloatND, width: int
) -> tuple[FloatND, FloatND]:
    """Publish the reduced row and a weakly typed scalar at every width."""
    return _reduced_row(x=x, candidate=candidate, width=width), jnp.asarray(1.0)


def _strong_scalar_body(
    *, x: FloatND, candidate: FloatND, width: int
) -> tuple[FloatND, FloatND]:
    """Publish the reduced row and a strongly typed scalar at every width."""
    return (
        _reduced_row(x=x, candidate=candidate, width=width),
        jnp.asarray(1.0, dtype=x.dtype),
    )


def _width_dependent_scalar_body(
    *, x: FloatND, candidate: FloatND, width: int
) -> tuple[FloatND, FloatND]:
    """Publish a scalar that is weakly typed only at the narrowest width."""
    scalar = (
        jnp.asarray(1.0) if width == _WEAK_WIDTH else jnp.asarray(1.0, dtype=x.dtype)
    )
    return _reduced_row(x=x, candidate=candidate, width=width), scalar


_PRODUCER_BODIES = MappingProxyType(
    {
        "weak": _weak_scalar_body,
        "strong": _strong_scalar_body,
        "width_dependent": _width_dependent_scalar_body,
    }
)


def _reader_body(*, x: FloatND, scalar: FloatND) -> FloatND:
    """Publish the state row shifted by the scalar its producer handed over."""
    return x + scalar


@dataclass(frozen=True, kw_only=True)
class _MaxReduction:
    """Semantics of the streamed candidate axis's reduction."""

    @property
    def semantic_key(self) -> tuple[str, int]:
        """Return the durable identity of this reduction."""
        return ("tests.public_weak_type_contract.max", 1)


@dataclass(frozen=True, kw_only=True)
class _StateAndCandidates:
    """Build the wealth row and the streamed candidate coordinate."""

    def __call__(self, context: CoreBuildContext) -> Mapping[str, object]:
        """Return the wealth row and the candidate coordinate it reduces over."""
        x = cast("Any", context.state_action_space).states["wealth"]
        return {"x": x, "candidate": jnp.arange(_CANDIDATES, dtype=x.dtype)}


@dataclass(frozen=True, kw_only=True)
class _StateOnly:
    """Build the wealth row the consumer shifts by its internal input."""

    def __call__(self, context: CoreBuildContext) -> Mapping[str, object]:
        """Return the wealth row alone."""
        return {"x": cast("Any", context.state_action_space).states["wealth"]}


def _programs(*, convention: str) -> Mapping[str, CoreProgram]:
    """Declare a planned producer publishing a scalar and the consumer reading it."""
    producer = CoreProgram(
        name="source",
        function=_PRODUCER_BODIES[convention],
        argument_builder=_StateAndCandidates(),
        requirements=CoreExecutionRequirements(
            streamable_axes=(
                StreamableProductAxis(
                    name="candidate",
                    coordinate_names=("candidate",),
                    coordinate_extents=(_CANDIDATES,),
                    canonical_order="c",
                    reduction=_MaxReduction(),
                    width_keyword="width",
                ),
            )
        ),
        output_roles=(VALUE, StateAxesLeading(state_names=())),
        disposition=CoreExecutionDisposition.PLANNED,
        internal_outputs=(InternalOutputSpec(label="scalar", path=(1,)),),
    )
    reader = CoreProgram(
        name="reader",
        function=_reader_body,
        argument_builder=_StateOnly(),
        requirements=CoreExecutionRequirements(
            internal_inputs={
                "scalar": InternalInputRef(producer="source", label="scalar")
            }
        ),
        output_roles=VALUE,
        disposition=CoreExecutionDisposition.DENSE,
        disposition_reason="deliberately_dense:one_row_per_wealth_node",
    )
    return MappingProxyType({"source": producer, "reader": reader})


@dataclass(frozen=True, kw_only=True)
class _ScalarEdgeKernel:
    """Dispatch the producer, then the consumer reading its published scalar."""

    programs: Mapping[str, CoreProgram]
    """The two-program graph this kernel publishes."""

    def core_programs(self) -> Mapping[str, CoreProgram]:
        """Return the graph as the sole execution authority."""
        return self.programs

    def with_fixed_params(self, *, fixed_flat_params: FlatParams) -> _ScalarEdgeKernel:
        """Return this kernel; its programs read no fixed regime params."""
        del fixed_flat_params
        return self

    def __call__(
        self,
        *,
        compiled_cores: Mapping[str, Any],
        state_action_space: object,
        next_regime_to_V_arr: Mapping[str, object],
        next_regime_to_continuation: Mapping[str, object],
        flat_params: FlatParams,
        period: int,
        ages: object,
        **unused: object,
    ) -> KernelOutput:
        """Hand the producer's published scalar to the consumer and publish its row."""
        del unused
        context = CoreBuildContext(
            state_action_space=state_action_space,
            next_regime_to_V_arr=next_regime_to_V_arr,
            next_regime_to_continuation=next_regime_to_continuation,
            flat_params=flat_params,
            period=period,
            ages=ages,
        )
        produced = compiled_cores["source"](
            **self.programs["source"].argument_builder(context)
        )
        value = compiled_cores["reader"](
            **self.programs["reader"].argument_builder(context),
            scalar=produced[1],
        )
        return KernelOutput(value=value)


@dataclass(frozen=True, kw_only=True)
class _ScalarEdgeSolver(GridSearch):
    """Publish the scalar-edge graph at every active period of its regime.

    The producer's typing convention is named rather than held: a solver's
    declared fields enter the model's durable structure fingerprint, which
    admits plain declarative values and refuses a program body carried on one.
    """

    convention: str
    """Which of this module's producer bodies every period dispatches."""

    @property
    def identity(self) -> SolverIdentity:
        """Return this test solver's durable plugin identity."""
        return SolverIdentity(
            plugin_id="tests.public_weak_type_contract", plugin_version="1.0.0"
        )

    def build_period_kernels(self, *, context: SolverBuildContext) -> SolutionKernels:
        """Share one kernel object across every active period of the regime."""
        kernel = _ScalarEdgeKernel(programs=_programs(convention=self.convention))
        return SolutionKernels(
            period_kernels=MappingProxyType(
                dict.fromkeys(
                    context.regimes_to_active_periods[context.regime_name], kernel
                )
            )
        )


def _model(*, convention: str) -> Model:
    """Build the regression regime with its solver replaced by the scalar edge."""
    last_age = START_AGE + _N_PERIODS - 2
    return Model(
        regimes={
            "working_life": working_life.replace(
                active=lambda age: age <= last_age,
                states={
                    "wealth": LinSpacedGrid(
                        start=1, stop=float(_N_WEALTH), n_points=_N_WEALTH
                    )
                },
                actions={
                    "labor_supply": DiscreteGrid(category_class=LaborSupply),
                    "consumption": LinSpacedGrid(start=1, stop=3, n_points=3),
                },
                solver=_ScalarEdgeSolver(convention=convention),
            ),
            "dead": dead,
        },
        ages=AgeGrid(start=START_AGE, stop=last_age + 1, step="Y"),
        regime_id_class=RegimeId,
        enable_jit=True,
    )


def _solve(*, convention: str) -> object:
    """Solve the scalar-edge model under a budget offering the whole frontier."""
    return _model(convention=convention).solve(
        params=get_params(n_periods=_N_PERIODS),
        log_level="off",
        retention=ResultRetention.VALUES,
        execution_config=ExecutionConfig(device_memory_bytes=_DEVICE_MEMORY_BYTES),
    )


def test_a_width_dependent_published_weak_typing_is_refused_by_a_solve() -> None:
    """A solve refuses a published scalar whose weak typing follows the width."""
    with pytest.raises(ExecutionPlanningError, match="scalar"):
        _solve(convention="width_dependent")


@pytest.mark.parametrize("convention", ["weak", "strong"])
def test_a_uniformly_typed_published_scalar_solves_to_its_consumers_row(
    *, convention: str
) -> None:
    """Either convention, held at every width, publishes the consumer's own row."""
    result = _solve(convention=convention)

    aaae(
        np.asarray(cast("Any", result).values[0]["working_life"]),
        np.linspace(1.0, float(_N_WEALTH), _N_WEALTH) + 1.0,
        decimal=DECIMAL_PRECISION,
    )
