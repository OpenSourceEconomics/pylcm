"""Literal row support follows its canonical producer, independently of leaf values."""

import functools
from collections.abc import Mapping
from types import MappingProxyType

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.egm.carry import EGMCarry
from _lcm.egm.continuation import bind_continuation
from _lcm.egm.step import _EGMStep
from _lcm.execution.core_program import CoreBuildContext, materialize_core_program
from _lcm.solution.dcegm import _DCEGMPeriodKernel
from _lcm.solution.feasibility import BoundedResult, ContinuationReadCounts
from _lcm.typing import EconFunctionKwargs
from lcm import AgeGrid, IrregSpacedGrid, Model, Regime, categorical
from lcm.solvers import DCEGM
from lcm.typing import BoolND, ContinuousState, FloatND, ScalarInt
from tests.egm.test_below_first_live_counts import _model, _params
from tests.egm.test_dynamic_dcegm_activation import RegimeId
from tests.solution._nbegm_direct_oracle import ride_along_kernel


@categorical(ordered=False)
class _TerminalRegimeId:
    parent: ScalarInt
    child: ScalarInt


def _never() -> BoolND:
    return jnp.asarray(a=False)


def _economic(*, liquid: ContinuousState) -> FloatND:
    return jnp.full_like(liquid, -jnp.inf)


def _literal_model(*, terminal_capacity: int | None) -> Model:
    """Keep one actual DC source beside independently declared target producers."""
    seed = _model(literal=True, parent_nodes=(-0.125, 0.5, 1.0))
    parent = seed.user_regimes["parent"].replace(
        solver=DCEGM(savings_grid=IrregSpacedGrid(points=np.array([-0.125, 0.5, 1.0])))
    )
    if terminal_capacity is None:
        child = seed.user_regimes["child"]
        regimes = {
            **seed.user_regimes,
            "parent": parent,
            "child": child.replace(constraints={**child.constraints, "never": _never}),
        }
        edges = seed.edges
        ages = seed.ages
        regime_id_class = RegimeId
    else:
        regimes = {
            "parent": parent,
            "child": Regime(
                states={
                    "liquid": IrregSpacedGrid(
                        points=np.array(
                            tuple(float(i) for i in range(terminal_capacity))
                        )
                    )
                },
                functions={"utility": _economic},
            ),
        }
        edges = {"parent": {"child": 0}}
        ages = AgeGrid(start=0, inclusive_stop=1, step="Y")
        regime_id_class = _TerminalRegimeId
    return Model(
        regimes=regimes,
        ages=ages,
        edges=edges,
        initial_nodes={0: "parent"},
        regime_id_class=regime_id_class,
        enable_jit=False,
    )


def _producer_and_read(
    *, model: Model, terminal: bool, query: float
) -> tuple[EGMCarry, ContinuationReadCounts]:
    """Read the actual captured child through its parent's canonical continuation."""
    params = _params(literal=True, parent=True)
    if terminal:
        params = {"parent": params["parent"], "child": {}}
    kernel, context = ride_along_kernel(
        model=model, params=params, regime_name="parent", period=0
    )
    assert isinstance(kernel, _DCEGMPeriodKernel)
    program = kernel.core_programs()["main"]
    assert isinstance(program.function, functools.partial)
    step = program.function.keywords["core"]
    assert isinstance(step, _EGMStep)
    arguments = materialize_core_program(
        program=program, context=CoreBuildContext(**context)
    ).arguments
    carries = arguments["next_regime_to_continuation"]
    assert isinstance(carries, Mapping)
    child = carries["child"]
    assert isinstance(child, EGMCarry)
    native_carries: dict[str, EGMCarry] = {}
    for name, payload in carries.items():
        assert isinstance(payload, EGMCarry)
        native_carries[name] = payload
    pool: EconFunctionKwargs = {}
    for name, value in arguments.items():
        if name in {"next_regime_to_continuation", "liquid"}:
            continue
        assert isinstance(value, jax.Array | float | int)
        pool[name] = value
    continuation = bind_continuation(
        plan=step.pieces.continuation_plan,
        combo_pool=pool,
        next_regime_to_continuation=MappingProxyType(native_carries),
        dtype=jnp.asarray(0.0).dtype,
        collect_continuation_read_counts=True,
    )
    result = continuation(jnp.asarray(query))
    assert isinstance(result, BoundedResult)
    counts = result.continuation_read_counts
    assert isinstance(counts, ContinuationReadCounts)
    np.testing.assert_array_equal(result.output, [-np.inf, 0.0])
    return child, counts


@pytest.mark.parametrize(
    ("query", "expected"),
    [(-0.125, (1, 1)), (0.0, (0, 1))],
    ids=["below-first", "equality"],
)
def test_literal_support_distinguishes_producer_on_identical_leaves(
    *, query: float, expected: tuple[int, int]
) -> None:
    """An infeasible DC placeholder has no support; an economic terminal row does."""
    dummy, dummy_counts = _producer_and_read(
        model=_literal_model(terminal_capacity=None),
        terminal=False,
        query=query,
    )
    capacity = dummy.leaves()[("endog_grid",)].size
    economic, economic_counts = _producer_and_read(
        model=_literal_model(terminal_capacity=capacity),
        terminal=True,
        query=query,
    )
    assert dummy.query_data_for_transport() is None
    assert economic.query_data_for_transport() is None
    assert len(dummy.leaves()) == len(economic.leaves()) == 4
    for label, leaf in dummy.leaves().items():
        np.testing.assert_array_equal(leaf, economic.leaves()[label])
    np.testing.assert_array_equal(dummy.leaves()[("endog_grid",)], np.arange(capacity))
    np.testing.assert_array_equal(dummy.leaves()[("value",)], -np.inf)
    np.testing.assert_array_equal(dummy.leaves()[("marginal_utility",)], 0.0)
    np.testing.assert_array_equal(dummy_counts, [0, 0])
    np.testing.assert_array_equal(economic_counts, expected)
