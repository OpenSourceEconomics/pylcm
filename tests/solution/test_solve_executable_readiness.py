"""Producer outputs are completed before conflicting budgeted cores."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Self, cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.execution.compiler_inputs import compiler_input_paths
from _lcm.solution import backward_induction
from lcm import (
    AgeGrid,
    AgeRange,
    ByAge,
    ExecutionConfig,
    LinSpacedGrid,
    MarkovTransition,
    Model,
    Regime,
    categorical,
)
from lcm.solver_api import KernelOutput, ResultRetention, SolverExecutionCapabilities
from lcm.solvers import (
    CoreBuildContext,
    CoreExecutionDisposition,
    CoreExecutionRequirements,
    CoreProgram,
    DeclaredReplay,
    InternalInputRef,
    InternalOutputSpec,
    OutputRole,
    SolutionKernels,
    Solver,
    SolverBuildContext,
    StateAxesLeading,
)
from lcm.typing import ScalarInt
from tests.conftest import assert_agrees_to_ulp
from tests.solution import test_donation_solve as counter_fixture
from tests.test_models.initial_regimes import initial_regimes_of
from tests.test_solver_api_out_of_tree import _WEALTH
from tests.test_solver_api_out_of_tree import RegimeId as CounterRegimeId

_SIZE = 1024
_BUDGET = 128 * 1024 * 1024


@categorical(ordered=False)
class _RegimeId:
    working: ScalarInt
    terminal: ScalarInt


def _utility(*, wealth: jax.Array, work: jax.Array) -> jax.Array:
    return wealth + jnp.sum(work @ work) / work.size


def _produce(*, wealth: jax.Array, work: jax.Array) -> tuple[jax.Array, jax.Array]:
    return wealth, work @ work


def _consume(*, previous_value: jax.Array, previous_matrix: jax.Array) -> jax.Array:
    return previous_value + jnp.sum(previous_matrix) / previous_matrix.size


def _producer_arguments(context: CoreBuildContext) -> Mapping[str, object]:
    space = cast("Any", context.state_action_space)
    params = cast("Mapping[str, Mapping[str, object]]", context.flat_params)
    return {
        "wealth": space.states["wealth"],
        "work": params["working"]["utility__work"],
    }


def _consumer_arguments(_context: CoreBuildContext) -> Mapping[str, object]:
    return {}


@dataclass(frozen=True, kw_only=True)
class _TwoProgramKernel:
    programs: Mapping[str, CoreProgram]

    def core_programs(self) -> Mapping[str, CoreProgram]:
        return self.programs

    def with_fixed_params(self, *, fixed_flat_params: object) -> Self:
        del fixed_flat_params
        return self

    def __call__(
        self,
        *,
        compiled_cores: Mapping[str, Callable[..., object]],
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
        value, matrix = cast(
            "tuple[jax.Array, jax.Array]",
            compiled_cores["producer"](
                **self.programs["producer"].argument_builder(context)
            ),
        )
        output = compiled_cores["consumer"](
            previous_value=value, previous_matrix=matrix
        )
        return KernelOutput(value=cast("jax.Array", output))


@dataclass(frozen=True)
class _TwoProgramSolver(Solver):
    @property
    def capabilities(self) -> SolverExecutionCapabilities:
        return SolverExecutionCapabilities(
            required_declaration="Regime",
            problem_shape="Two-program matrix fixture",
            prerequisites="Finite dyadic rank-one matrix",
            main_tradeoff="Scheduling witness",
        )

    def build_period_kernels(self, *, context: SolverBuildContext) -> SolutionKernels:
        producer = CoreProgram(
            name="producer",
            function=_produce,
            argument_builder=_producer_arguments,
            requirements=CoreExecutionRequirements(),
            output_roles=(
                OutputRole.VALUE,
                StateAxesLeading(
                    state_names=(),
                    n_free_leading_axes=2,
                    dtype=context.state_action_space.states["wealth"].dtype,
                    shape=(_SIZE, _SIZE),
                ),
            ),
            disposition=CoreExecutionDisposition.DENSE,
            disposition_reason="fixed_matrix_fixture",
            internal_outputs=(
                InternalOutputSpec(label="value", path=(0,)),
                InternalOutputSpec(label="matrix", path=(1,)),
            ),
        )
        consumer = CoreProgram(
            name="consumer",
            function=_consume,
            argument_builder=_consumer_arguments,
            requirements=CoreExecutionRequirements(
                internal_inputs={
                    "previous_value": InternalInputRef(
                        producer="producer", label="value"
                    ),
                    "previous_matrix": InternalInputRef(
                        producer="producer", label="matrix"
                    ),
                }
            ),
            output_roles=OutputRole.VALUE,
            disposition=CoreExecutionDisposition.DENSE,
            disposition_reason="fixed_matrix_fixture",
        )
        kernel = _TwoProgramKernel(
            programs=MappingProxyType({"producer": producer, "consumer": consumer})
        )
        return SolutionKernels(
            period_kernels=dict.fromkeys(
                context.regimes_to_active_periods[context.regime_name], kernel
            ),
            replay_route=DeclaredReplay.UNSUPPORTED,
        )


def _model(*, budget: int | None) -> Model:
    grid = LinSpacedGrid(start=1, stop=2, n_points=2)
    return Model(
        regimes={
            "working": Regime(
                regime_transitions=ByAge(
                    cases={
                        AgeRange(start=0, stop=1): {
                            "terminal": MarkovTransition(func=lambda: jnp.asarray(1.0))
                        }
                    }
                ),
                states={"wealth": grid},
                state_transitions={"wealth": lambda wealth: wealth},
                functions={"utility": _utility},
                solver=_TwoProgramSolver(),
            ),
            "terminal": Regime(
                regime_transitions=None,
                states={"wealth": grid},
                functions={"utility": lambda wealth: wealth},
            ),
        },
        ages=AgeGrid(start=0, stop=1, step="Y"),
        regime_id_class=_RegimeId,
        execution_config=ExecutionConfig(devices=(0,), device_memory_bytes=budget),
        initial_regimes={0: "working"},
    )


@pytest.mark.parametrize("budget", [_BUDGET, None])
def test_public_multicore_waits_for_pending_auxiliary_before_next_core(
    *,
    budget: int | None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _model(budget=budget)
    left = (np.arange(_SIZE) % 8 + 1) / 8
    right = (np.arange(_SIZE) % 16 + 1) / 16
    host_work = np.outer(left, right)
    work = jnp.asarray(host_work).block_until_ready()
    original_call = jax.stages.Compiled.__call__
    original_bind = backward_induction._cores_with_transfer_cache
    array_type = type(work)
    original_complete = array_type.block_until_ready
    original_complete_tree = jax.block_until_ready
    names: dict[int, str] = {}
    returned: list[jax.Array] = []
    events: list[str] = []

    def record_completion(leaves: list[object]) -> None:
        events.extend(
            f"complete:{index}"
            for leaf in leaves
            for index, array in enumerate(returned)
            if leaf is array
        )

    def complete(array: jax.Array) -> jax.Array:
        record_completion([array])
        return original_complete(array)

    def complete_tree(tree: object) -> object:
        record_completion(jax.tree.leaves(tree))
        return original_complete_tree(tree)

    def bind(**kwargs: Any) -> Any:
        cores = original_bind(**kwargs)
        names.update({id(core.compiled): name for name, core in cores.items()})
        return cores

    def observe(
        compiled: jax.stages.Compiled, *args: object, **kwargs: object
    ) -> object:
        name = names.get(id(compiled))
        if name == "consumer":
            events.append("consumer_entry")
        output = original_call(compiled, *args, **kwargs)
        if name == "producer":
            returned.extend(jax.tree.leaves(output))
            events.append("producer_return")
        return output

    monkeypatch.setattr(backward_induction, "_cores_with_transfer_cache", bind)
    monkeypatch.setattr(jax.stages.Compiled, "__call__", observe)
    monkeypatch.setattr(array_type, "block_until_ready", complete)
    monkeypatch.setattr(jax, "block_until_ready", complete_tree)
    try:
        solution = model.solve(
            params={"discount_factor": 0.0, "work": work},
            retention=ResultRetention.ALL_PERSISTABLE_ARTIFACTS,
            log_level="off",
        )
    finally:
        original_complete_tree(returned)
    assert len(returned) == 2
    assert all(array.devices() == {jax.devices()[0]} for array in returned)
    # Completion is observed as a request, never as a readiness reading taken
    # while the producer may or may not have finished on its own.
    assert events.index("producer_return") < events.index("consumer_entry"), events
    between = events[
        events.index("producer_return") + 1 : events.index("consumer_entry")
    ]
    expected_completed = {"complete:0", "complete:1"} if budget is not None else set()
    assert set(between) == expected_completed, events
    expected = np.array([1.0, 2.0]) + np.sum(left) * np.sum(right) * np.dot(
        right, left
    ) / (_SIZE * _SIZE)
    assert_agrees_to_ulp(
        got=solution.values[0]["working"], expected=expected.astype(work.dtype), n_ulp=4
    )
    np.testing.assert_array_equal(work, host_work.astype(work.dtype))


def test_budgeted_real_donor_and_template_fallback_leave_no_stale_witness(
    *,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    counter_fixture._TEMPLATES.clear()
    counter_fixture._DISPATCHED.clear()
    base = counter_fixture._two_regime_model(
        solver=counter_fixture._DonatingCounterSolver(),
        self_looping=True,
    )
    model = Model(
        regimes=base.user_regimes,
        ages=base.ages,
        regime_id_class=CounterRegimeId,
        execution_config=ExecutionConfig(devices=(0,), device_memory_bytes=_BUDGET),
        initial_regimes=initial_regimes_of(model=base),
    )
    call = jax.stages.Compiled.__call__
    bind = backward_induction._cores_with_transfer_cache
    donation_by_executable: dict[int, tuple[str, ...]] = {}
    dispatches: list[tuple[int, tuple[str, ...], bool]] = []

    def observe_bind(**kwargs: Any) -> Any:
        # Cached declarations never receive this solve's concrete owner.
        assert all(core.pending_work is None for core in kwargs["cores"].values())
        cores = bind(**kwargs)
        assert all(core.pending_work is not None for core in cores.values())
        donation_by_executable.update(
            {id(core.compiled): core.donated_arguments for core in cores.values()}
        )
        return cores

    def observe_call(
        executable: jax.stages.Compiled, *args: object, **kwargs: object
    ) -> object:
        count = kwargs.get("count")
        if isinstance(count, jax.Array):
            kept = compiler_input_paths(compiled=executable, arguments=kwargs)
            assert (jax.tree_util.DictKey("count"),) in kept
            assert not count.is_deleted()
        output = call(executable, *args, **kwargs)
        if isinstance(count, jax.Array):
            dispatches.append(
                (
                    id(executable),
                    donation_by_executable[id(executable)],
                    count.is_deleted(),
                )
            )
        return output

    monkeypatch.setattr(backward_induction, "_cores_with_transfer_cache", observe_bind)
    monkeypatch.setattr(jax.stages.Compiled, "__call__", observe_call)
    try:
        result = model.solve(params={"discount_factor": 1.0}, log_level="off")
        assert [(donation, deleted) for _, donation, deleted in dispatches] == [
            ((), False),
            (("count",), True),
            (("count",), True),
        ]
        assert dispatches[0][0] != dispatches[1][0]
        assert dispatches[1][0] == dispatches[2][0]
        assert not counter_fixture._TEMPLATES[-1].count.is_deleted()
        np.testing.assert_array_equal(counter_fixture._TEMPLATES[-1].count, 0.0)
        wealth = np.asarray(_WEALTH.to_jax())
        for period in range(3):
            np.testing.assert_array_equal(
                result.values[period]["alive"], wealth + 2 - period
            )
    finally:
        counter_fixture._DISPATCHED.clear()
        counter_fixture._TEMPLATES.clear()
