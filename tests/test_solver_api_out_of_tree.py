"""A solver written against the public contract alone runs through the engine.

The public surface is `lcm.solvers`, `lcm.solver_api`, and `lcm.typing`: a period
kernel declares its native core-program graph, returns a `KernelOutput`, and may
publish a continuation artifact of its own type under its own versioned key. The
engine rolls that artifact opaquely, checks every parent's continuation demand
against what its targets publish at build, and simulates through the route each
regime declares.
"""

import ast
import dataclasses
import functools
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from lcm import AgeGrid, LinSpacedGrid, MarkovTransition, Model, Regime, categorical
from lcm.exceptions import RegimeInitializationError
from lcm.solver_api import (
    EGM_CONTINUATION,
    SIMULATION_POLICY,
    ArtifactKey,
    ArtifactStore,
    ContinuationArtifact,
    KernelOutput,
    ReplayMode,
)
from lcm.solvers import (
    NBEGM,
    ContinuationSpec,
    CoreBuildContext,
    CoreExecutionDisposition,
    CoreExecutionRequirements,
    CoreProgram,
    FiniteOuterGrid,
    OutputRole,
    SolutionKernels,
    Solver,
    SolverBuildContext,
    StateAxesLeading,
)
from lcm.typing import ContinuousState, Float1D, FloatND, ScalarFloat, ScalarInt
from tests.test_models import n_nbegm_toy


@categorical(ordered=False)
class RegimeId:
    alive: ScalarInt
    dead: ScalarInt


_WEALTH = LinSpacedGrid(start=1.0, stop=5.0, n_points=5)
_N_PERIODS = 3


def utility(wealth: ContinuousState) -> FloatND:
    return wealth


def next_wealth(wealth: ContinuousState) -> ContinuousState:
    return wealth


def next_regime_dead(age: ScalarFloat) -> ScalarInt:  # noqa: ARG001
    return RegimeId.dead


def stay_alive(age: ScalarFloat) -> ScalarFloat:  # noqa: ARG001
    return jnp.asarray(1.0)


def _wealth_value(*, wealth: Float1D) -> Float1D:
    """One value per state node: the wealth itself."""
    return wealth


@dataclasses.dataclass(frozen=True, kw_only=True)
class _GraphKernel:
    """A period kernel that dispatches its single declared program."""

    programs: Mapping[str, CoreProgram]
    continuation_key: ArtifactKey | None = None

    def core_programs(self) -> Mapping[str, CoreProgram]:
        return self.programs

    def with_fixed_params(
        self,
        *,
        fixed_flat_params: object,  # noqa: ARG002
    ) -> _GraphKernel:
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
        logger: object,  # noqa: ARG002
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
        out = compiled_cores["main"](**arguments)  # ty: ignore[call-non-callable]
        if self.continuation_key is None:
            return KernelOutput(value=out)
        value, artifact = out
        return KernelOutput(
            value=value, continuations={self.continuation_key: artifact}
        )


class WealthSolver(Solver):
    """Publishes the wealth grid as the value in every active period."""

    def build_period_kernels(self, *, context: SolverBuildContext) -> SolutionKernels:
        program = CoreProgram(
            name="main",
            function=_wealth_value,
            argument_builder=lambda build: {
                "wealth": build.state_action_space.states["wealth"]
            },
            requirements=CoreExecutionRequirements(),
            output_roles=OutputRole.VALUE,
            disposition=CoreExecutionDisposition.DENSE,
            disposition_reason="one_row_per_state_node",
        )
        kernels = {
            period: _GraphKernel(programs=MappingProxyType({"main": program}))
            for period in context.regimes_to_active_periods[context.regime_name]
        }
        return SolutionKernels(period_kernels=MappingProxyType(kernels))


def _two_regime_model(*, solver: Solver, self_looping: bool = False) -> Model:
    transition = (
        {"alive": MarkovTransition(stay_alive)} if self_looping else next_regime_dead
    )
    # A regime that dies into the terminal one leaves the last period to it, so
    # a simulated subject always has somewhere to go. A self-looping regime is
    # its own target and stays active throughout.
    alive_active = (
        (lambda _age: True) if self_looping else (lambda age: age < _N_PERIODS - 1)
    )
    return Model(
        regimes={
            "alive": Regime(
                transition=transition,
                active=alive_active,
                states={"wealth": _WEALTH},
                state_transitions={"wealth": next_wealth},
                functions={"utility": utility},
                solver=solver,
            ),
            "dead": Regime(
                transition=None,
                states={"wealth": _WEALTH},
                functions={"utility": lambda wealth: 0.0 * wealth},
            ),
        },
        ages=AgeGrid(start=0, stop=_N_PERIODS - 1, step="Y"),
        regime_id_class=RegimeId,
    )


def test_a_solver_written_against_lcm_solvers_alone_solves_a_two_period_model():
    """The engine runs an out-of-tree kernel's declared program in every period."""
    solution = _two_regime_model(solver=WealthSolver()).solve(
        params={"discount_factor": 1.0}, log_level="off"
    )
    for period in range(_N_PERIODS - 1):
        np.testing.assert_array_equal(
            np.asarray(solution.values[period]["alive"]), np.asarray(_WEALTH.to_jax())
        )


def test_this_module_imports_nothing_from_the_engine_package():
    """The solver above is written against the public surface only."""
    tree = ast.parse(Path(__file__).read_text())
    engine_imports = [
        node
        for node in ast.walk(tree)
        if (isinstance(node, ast.ImportFrom) and (node.module or "").startswith("_lcm"))
        or (
            isinstance(node, ast.Import)
            and any(alias.name.startswith("_lcm") for alias in node.names)
        )
    ]
    assert engine_imports == []


_VERSION_TWO = dataclasses.replace(EGM_CONTINUATION, schema_version=2)


class _VersionTwoSolver(WealthSolver):
    @property
    def required_continuation_keys(self) -> frozenset[ArtifactKey]:
        return frozenset({_VERSION_TWO})


def test_a_parent_requiring_a_version_a_child_does_not_publish_is_refused():
    """A continuation demand is matched against the target's published key at build."""
    with pytest.raises(RegimeInitializationError) as excinfo:
        _two_regime_model(solver=_VersionTwoSolver())
    message = str(excinfo.value)
    assert "'alive'" in message
    assert "'dead'" in message
    assert "version 2" in message


_COUNTER = ArtifactKey(type_id="tests.counter", schema_version=1)


@functools.partial(
    jax.tree_util.register_dataclass, data_fields=["count"], meta_fields=[]
)
@dataclasses.dataclass(frozen=True, kw_only=True)
class _Counter:
    """A continuation artifact the engine has never seen: one running count."""

    count: FloatND

    @property
    def artifact_key(self) -> ArtifactKey:
        return _COUNTER


def _counting_value(*, wealth: Float1D, count: FloatND) -> tuple[Float1D, _Counter]:
    return wealth + count, _Counter(count=count + 1.0)


class _CountingSolver(Solver):
    """Reads its own next-period artifact and republishes it incremented."""

    @property
    def required_continuation_keys(self) -> frozenset[ArtifactKey]:
        return frozenset({_COUNTER})

    def build_period_kernels(self, *, context: SolverBuildContext) -> SolutionKernels:
        program = CoreProgram(
            name="main",
            function=_counting_value,
            argument_builder=lambda build: {
                "wealth": build.state_action_space.states["wealth"],
                "count": build.next_regime_to_continuation["alive"].count,
            },
            requirements=CoreExecutionRequirements(),
            output_roles=(
                OutputRole.VALUE,
                # A role tree reuses the payload class with roles in its
                # leaves, the way the shipped carry role trees do.
                _Counter(
                    count=StateAxesLeading(state_names=(), shape=())  # ty: ignore[invalid-argument-type]
                ),
            ),
            disposition=CoreExecutionDisposition.DENSE,
            disposition_reason="one_row_per_state_node",
        )
        kernels = {
            period: _GraphKernel(
                programs=MappingProxyType({"main": program}), continuation_key=_COUNTER
            )
            for period in context.regimes_to_active_periods[context.regime_name]
        }
        return SolutionKernels(
            period_kernels=MappingProxyType(kernels),
            continuation_spec=ContinuationSpec(
                template=_Counter(count=jnp.zeros(())), artifact_key=_COUNTER
            ),
        )


def test_engine_rolls_a_non_egm_continuation_artifact_opaquely():
    """A self-declared artifact reaches the previous period unchanged in type."""
    assert isinstance(_Counter(count=jnp.zeros(())), ContinuationArtifact)
    model = _two_regime_model(solver=_CountingSolver(), self_looping=True)
    solution = model.solve(params={"discount_factor": 1.0}, log_level="off")
    wealth = np.asarray(_WEALTH.to_jax())
    for period in range(_N_PERIODS):
        periods_ahead = _N_PERIODS - 1 - period
        np.testing.assert_array_equal(
            np.asarray(solution.values[period]["alive"]), wealth + periods_ahead
        )


def _nnbegm_model() -> Model:
    return n_nbegm_toy.build_model(
        variant="n_nbegm",
        outer_search=FiniteOuterGrid(grid=n_nbegm_toy.OUTER_GRID),
    )


def _nnbegm_inner() -> NBEGM:
    return NBEGM(savings_grid=n_nbegm_toy.SAVINGS_GRID, envelope_arithmetic="ordinary")


@pytest.mark.parametrize(
    ("model", "expected_modes"),
    [
        (
            _two_regime_model(solver=WealthSolver()),
            {
                "alive": ReplayMode.VALID_RECOMPUTATION,
                "dead": ReplayMode.VALID_RECOMPUTATION,
            },
        ),
        (
            _nnbegm_model(),
            {"alive": ReplayMode.EXACT_REPLAY, "dead": ReplayMode.VALID_RECOMPUTATION},
        ),
    ],
    ids=["out_of_tree", "nnbegm_finite"],
)
def test_every_regime_declares_exactly_one_replay_mode(*, model: Model, expected_modes):
    """Every regime's declared route names one replay mode and its payload type."""
    assert {mode.name for mode in ReplayMode} == {
        "EXACT_REPLAY",
        "VALID_RECOMPUTATION",
        "UNSUPPORTED",
    }
    modes = {
        name: regime.simulation.replay_route.replay_mode
        for name, regime in model._regimes.items()
    }
    assert modes == expected_modes
    for name, regime in model._regimes.items():
        route = regime.simulation.replay_route
        assert (route.payload_type is None) == (
            route.replay_mode is ReplayMode.VALID_RECOMPUTATION
        ), name


def test_a_route_without_a_payload_simulates_on_the_grid():
    """With no payload the decision is recomputed on the regime's action grid."""
    model = _two_regime_model(solver=WealthSolver())
    result = model.simulate(
        params={"discount_factor": 1.0},
        initial_conditions={
            "wealth": jnp.asarray([1.0, 3.0]),
            "age": jnp.zeros(2),
            "regime_id": jnp.asarray([RegimeId.alive, RegimeId.alive]),
        },
        log_level="off",
    )
    panel = result.to_dataframe()
    np.testing.assert_array_equal(
        panel.query("period == 0")["wealth"].to_numpy(), np.asarray([1.0, 3.0])
    )


def test_a_payload_of_another_type_than_the_route_declares_fails_the_preflight():
    """Simulation refuses a replay artifact whose type is not the route's."""
    model = _nnbegm_model()
    solution = model.solve(params={"discount_factor": 0.95}, log_level="off")
    ref = next(ref for ref in solution.replay_artifacts if ref.key == SIMULATION_POLICY)
    route = model._regimes[ref.regime].simulation.replay_route
    declared = route.payload_type
    assert declared is not None
    assert isinstance(solution.replay_artifacts[ref], declared)
    mutated = dataclasses.replace(
        solution,
        replay_artifacts=ArtifactStore(
            dict(solution.replay_artifacts) | {ref: _Counter(count=jnp.zeros(()))}
        ),
    )
    with pytest.raises(Exception, match=declared.__name__) as excinfo:
        model.simulate(
            params={"discount_factor": 0.95},
            initial_conditions={
                "wealth": jnp.asarray([5.0, 10.0]),
                "illiquid": jnp.asarray([1.0, 2.0]),
                "age": jnp.full(2, 20.0),
                "regime_id": jnp.full(2, n_nbegm_toy.RegimeId.alive, dtype=jnp.int32),
            },
            solution=mutated,
            log_level="off",
        )
    assert "_Counter" in str(excinfo.value)
