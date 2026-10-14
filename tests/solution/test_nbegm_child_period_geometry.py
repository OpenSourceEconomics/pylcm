"""NB-EGM save-to-cliff targets read the child period's own grids and functions.

The source decides at age 0; its child is next period at age 1. Two public
declarations make the child's cliff geometry differ from the source's:

- an age-specialized `wage` grid whose nodes at the child's age differ from the
  source's, with the same number of points, so the child's blended wage rows
  are not the rows the source grid would bracket;
- an age-specialized `gross_income` function closing over its age, so the
  child's cliff sits where the child's own income crosses the threshold.

Both the production cliff targets and the scalar oracle's must bracket the
savings preimages of the child's cliffs, which `_cliff_pullback_reference`
computes in exact arithmetic without reading either. The full period kernel must
also agree with the scalar oracle.

A unit wage persistence lands the source's wage 4 exactly on the child's middle
node 4; persistences one representable float either side of one land it on the
adjacent floats. The child's rows read are both nodes of the child segment the
query falls in — the segment above a node the query lands on exactly — and those
segments differ from the ones the source's grid would bracket.
"""

import bisect
import dataclasses
from collections.abc import Mapping, Sequence
from fractions import Fraction
from typing import NotRequired, Protocol, TypedDict

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpy.typing import ArrayLike

import lcm
from _lcm.egm.carry import EGMCarry
from _lcm.execution.core_program import (
    CoreBuildContext,
    core_program_graph,
    materialize_core_program,
)
from _lcm.solution.nbegm import _cliff_savings_targets as production_targets
from _lcm.solution.nbegm import _RideAlongNBEGMPeriodKernel
from _lcm.typing import ArgumentTree, EconFunctionArg
from lcm import AgeSpecializedFunction, AgeSpecializedGrid, DiscreteGrid, LinSpacedGrid
from lcm.typing import ContinuousState, FloatND, UserParams
from tests.solution._cliff_pullback_reference import (
    age_closure_preimage,
    blended_row_preimages,
)
from tests.solution._nbegm_direct_oracle import (
    ChildPeriodContext,
    OracleContext,
    WorkingDType,
    _economic_argument,
    child_period_context,
    ride_along_kernel,
)
from tests.solution._nbegm_direct_oracle import (
    _cliff_savings_targets as oracle_targets,
)
from tests.solution.test_nbegm_direct_oracle import (
    _assert_kernel_agrees_with_oracle as assert_kernel_agrees_with_oracle,
)
from tests.test_models import nbegm_continuous_ride_along_toy
from tests.test_models.nbegm_common import (
    make_alive_dead_model,
    resolve_solver,
    savings,
    utility,
)
from tests.test_models.nbegm_indexed_threshold_toy import (
    ConsumerKind,
    resources,
    subsidy,
)


class _GrossIncome(Protocol):
    def __call__(self, *, liquid: ContinuousState, base_income: float) -> FloatND: ...


class _Seam(TypedDict):
    kernel: _RideAlongNBEGMPeriodKernel
    context: OracleContext
    child: ChildPeriodContext
    kwargs: Mapping[str, ArgumentTree]
    cell: dict[str, jax.Array]
    dtype: WorkingDType
    centres: tuple[float, ...]
    source_centres: NotRequired[tuple[float, ...]]


def identity_liquid(*, savings: FloatND) -> ContinuousState:
    """Liquid wealth next period equals savings."""
    return savings


def zero_bequest(*, liquid: ContinuousState) -> FloatND:
    """The terminal regime values remaining wealth at zero."""
    return jnp.zeros_like(liquid)


def _wage_grid_model(
    *, child_top: float, wage_persistence: float = 0.9
) -> tuple[lcm.Model, UserParams]:
    """Wage nodes `(0, 2, 4)` at age 0 and `(0, top / 2, top)` from age 1 on."""

    def build_grid(age: float) -> LinSpacedGrid:
        return LinSpacedGrid(
            start=0.0, stop=4.0 if age < 1.0 else child_top, n_points=3
        )

    model = nbegm_continuous_ride_along_toy.build_model(
        variant="nbegm",
        n_periods=3,
        n_liquid=17,
        n_consumption=17,
        n_savings=8,
        wage_grid=AgeSpecializedGrid(
            build=build_grid, signature=lambda age: 4.0 if age < 1.0 else child_top
        ),
    )
    params = nbegm_continuous_ride_along_toy.build_params(
        crra=1.0,
        return_liquid=0.0,
        income=1.0,
        subsidy_high=0.5,
        wage_persistence=wage_persistence,
    )
    return model, params


def _income_closure_model(*, increment: float) -> tuple[lcm.Model, UserParams]:
    """Income `liquid + base_income + increment * age`, with age closed over."""

    def build_income(age: float) -> _GrossIncome:
        def gross_income(*, liquid: ContinuousState, base_income: float) -> FloatND:
            return liquid + base_income + increment * age

        return gross_income

    model = make_alive_dead_model(
        n_periods=3,
        n_liquid=17,
        liquid_max=30.0,
        n_consumption=17,
        liquid_grid=LinSpacedGrid(start=0.0, stop=30.0, n_points=17),
        alive_functions={
            "utility": utility,
            "gross_income": AgeSpecializedFunction(
                build=build_income, signature=lambda age: increment * age
            ),
            "subsidy": subsidy,
            "resources": resources,
            "savings": savings,
        },
        liquid_law=identity_liquid,
        alive_solver=resolve_solver(
            variant="nbegm",
            savings_grid=LinSpacedGrid(start=0.0, stop=28.0, n_points=8),
        ),
        constraints={},
        extra_states={"kind": DiscreteGrid(category_class=ConsumerKind)},
        extra_state_transitions={"kind": {"alive": lcm.fixed_transition("kind")}},
        dead_functions={"utility": zero_bequest},
    )
    params = {
        "alive": {
            "utility": {"crra": 1.0},
            "koopmans_aggregator": {"discount_factor": 0.95},
            "gross_income": {"base_income": 2.0},
            "subsidy": {
                "subsidy_low": 0.0,
                "subsidy_high": 5.0,
                "fpl_cliff": jnp.asarray([11.0, 11.0]),
            },
        }
    }
    return model, params


def _grid_centres(*, child_top: float) -> tuple[float, ...]:
    """Source wage 4 moves to `0.9 * 4`; each blended row's cliff is income 15."""
    top = Fraction(child_top)
    return tuple(
        float(centre)
        for centre in blended_row_preimages(
            child_nodes=(Fraction(0), top / 2, top),
            query=Fraction(9, 10) * 4,
            threshold=Fraction(15),
            slope=Fraction(1),
            offset=Fraction(1),
        )
    )


def _income_centres(*, increment: float) -> tuple[float, ...]:
    """The child, at age 1, loses the subsidy at income 11; `liquid' = s`."""
    return (
        float(
            age_closure_preimage(
                threshold=Fraction(11),
                base_income=Fraction(2),
                increment=Fraction(increment),
                child_age=Fraction(1),
            )
        ),
    )


# Each case: the model kind and its parameter. The wage-grid cases keep the
# source grid `(0, 2, 4)`; the income cases keep the source age 0.
_CASES = {
    "grid-unchanged": ("grid", 4.0),
    "grid-top-6": ("grid", 6.0),
    "grid-top-8": ("grid", 8.0),
    "income-age-invariant": ("income", 0.0),
    "income-rising-with-age": ("income", 3.0),
}


@pytest.fixture(scope="module", params=tuple(_CASES), ids=tuple(_CASES))
def seam(request: pytest.FixtureRequest) -> _Seam:
    kind, parameter = _CASES[request.param]
    centres = (
        _grid_centres(child_top=parameter)
        if kind == "grid"
        else _income_centres(increment=parameter)
    )
    model, params = (
        _wage_grid_model(child_top=parameter)
        if kind == "grid"
        else _income_closure_model(increment=parameter)
    )
    return _build_seam(model=model, params=params, kind=kind, centres=centres)


def _build_seam(
    *, model: lcm.Model, params: UserParams, kind: str, centres: tuple[float, ...]
) -> _Seam:
    """The source's period-0 kernel, its solved inputs and the cell it reads."""
    kernel, context = ride_along_kernel(model=model, params=params, period=0)
    assert isinstance(kernel, _RideAlongNBEGMPeriodKernel)
    replay = core_program_graph(kernel=kernel)["replay"]
    materialized = materialize_core_program(
        program=replay, context=CoreBuildContext(**context)
    )
    kwargs = dict(materialized.arguments)
    kwargs.update(getattr(replay.function, "keywords", None) or {})
    liquid = kwargs[kernel.statics.liquid_name]
    assert isinstance(liquid, jax.Array)
    dtype = liquid.dtype
    if dtype == np.dtype(np.float32):
        dtype = np.dtype(np.float32)
    else:
        assert dtype == np.dtype(np.float64)
        dtype = np.dtype(np.float64)
    cell = (
        {"wage": jnp.asarray(4.0, dtype=dtype)}
        if kind == "grid"
        else {"kind": jnp.asarray(0, dtype=jnp.int32)}
    )
    child = child_period_context(model=model, context=context)
    assert child is not None
    return {
        "kernel": kernel,
        "context": context,
        "child": child,
        "kwargs": kwargs,
        "cell": cell,
        "dtype": dtype,
        "centres": centres,
    }


_NODE_STEPS = {"below-node": -1, "on-node": 0, "above-node": 1}
_SOURCE_WAGE_NODES = (Fraction(0), Fraction(2), Fraction(4))
_CHILD_WAGE_NODES = (Fraction(0), Fraction(4), Fraction(8))


def _node_persistence(step: int) -> float:
    """One, or the representable float adjacent to one, at the canonical dtype.

    The source's wage 4 times this persistence is then 4 or the float adjacent to
    4 on the same side, exactly.
    """
    dtype = np.dtype(jnp.asarray(1.0).dtype)
    one = np.asarray(1.0, dtype=dtype)
    if step == 0:
        return float(one)
    return float(np.nextafter(one, np.asarray(1.0 + step, dtype=dtype)))


def _segment_centres(
    *, nodes: Sequence[Fraction], query: Fraction
) -> tuple[float, ...]:
    """Savings preimages of the income-15 cliffs of the rows a wage query reads.

    The rows are both nodes of the segment containing `query`, closed on the
    left, clipped to the grid's first and last segments; row `w` loses its subsidy
    at liquid `15 - w`, and `liquid' = s + 1`.
    """
    upper = min(max(bisect.bisect_right(nodes, query), 1), len(nodes) - 1)
    return tuple(
        sorted(float(Fraction(14) - node) for node in nodes[upper - 1 : upper + 1])
    )


@pytest.fixture(scope="module", params=tuple(_NODE_STEPS), ids=tuple(_NODE_STEPS))
def node_seam(request: pytest.FixtureRequest) -> _Seam:
    persistence = _node_persistence(_NODE_STEPS[request.param])
    query = Fraction(persistence) * 4
    model, params = _wage_grid_model(child_top=8.0, wage_persistence=persistence)
    seam = _build_seam(
        model=model,
        params=params,
        kind="grid",
        centres=_segment_centres(nodes=_CHILD_WAGE_NODES, query=query),
    )
    seam["source_centres"] = _segment_centres(nodes=_SOURCE_WAGE_NODES, query=query)
    return seam


def _combo_pool(*, seam: _Seam) -> dict[str, EconFunctionArg]:
    statics = seam["kernel"].statics
    params = {
        key: _economic_argument(value)
        for key, value in seam["kwargs"].items()
        if key not in statics.state_names and key != "next_regime_to_continuation"
    }
    return {**params, **seam["cell"]}


def _live_rows(raw: ArrayLike) -> np.ndarray:
    rows = np.asarray(raw).reshape(-1, 2)
    return rows[np.isfinite(rows).all(axis=1)]


def _brackets_exactly(
    *, rows: np.ndarray, centres: tuple[float, ...], dtype: WorkingDType
) -> bool:
    """Whether every centre is tightly bracketed and every pair brackets a centre.

    The centres are separated interior cliffs, so the width bound measures only
    the one-sided displacement of the targets, a few dozen units of roundoff.
    """
    eps = float(np.finfo(np.dtype(dtype)).eps)
    for centre in centres:
        widths = [upper - lower for lower, upper in rows if lower < centre < upper]
        if not widths or min(widths) > 64 * eps * max(1.0, abs(centre)):
            return False
    return all(
        any(lower < centre < upper for centre in centres) for lower, upper in rows
    )


def _production_rows(*, seam: _Seam, jit: bool) -> np.ndarray:
    kernel = seam["kernel"]

    carry = seam["context"]["next_regime_to_continuation"]["alive"]
    assert isinstance(carry, EGMCarry)

    def evaluate() -> FloatND:
        return production_targets(
            continuation_plan=kernel.continuation_plan,
            regime_name="alive",
            statics=kernel.statics,
            child_carry=carry,
            combo_pool=_combo_pool(seam=seam),
            savings_grid=jnp.asarray(kernel.savings_grid),
            dtype=seam["dtype"],
        )

    return _live_rows(jax.jit(evaluate)() if jit else evaluate())


def _oracle_rows(*, seam: _Seam, child: ChildPeriodContext) -> np.ndarray:
    kernel = seam["kernel"]
    return _live_rows(
        oracle_targets(
            plan=kernel.continuation_plan,
            regime_name="alive",
            combo_pool=_combo_pool(seam=seam),
            kwargs={
                key: _economic_argument(value)
                for key, value in seam["kwargs"].items()
                if key != "next_regime_to_continuation"
            },
            child=child,
            savings_grid=np.asarray(kernel.savings_grid, dtype=np.float64),
            dtype=seam["dtype"],
        )
    )


def _source_as_child(*, seam: _Seam) -> ChildPeriodContext:
    """The oracle's child context with the source period's grids and functions."""
    kernel, context = seam["kernel"], seam["context"]
    return dataclasses.replace(
        seam["child"],
        statics=kernel.statics,
        grids={
            name: np.asarray(nodes)
            for name, nodes in context["state_action_space"].states.items()
        },
        period=int(context["period"]),
        age=context["ages"].values[int(context["period"])],
    )


@pytest.mark.parametrize("jit", [False, True], ids=["eager", "jit"])
def test_production_cliff_targets_bracket_the_child_periods_cliffs(
    *, seam: _Seam, jit: bool
) -> None:
    """The kernel's save-to-cliff targets straddle exactly the child's centres."""
    rows = _production_rows(seam=seam, jit=jit)
    assert _brackets_exactly(rows=rows, centres=seam["centres"], dtype=seam["dtype"]), (
        rows,
        seam["centres"],
    )


def test_oracle_cliff_targets_bracket_the_child_periods_cliffs(*, seam: _Seam) -> None:
    """The scalar oracle's cliff targets straddle exactly the child's centres."""
    rows = _oracle_rows(seam=seam, child=seam["child"])
    assert _brackets_exactly(rows=rows, centres=seam["centres"], dtype=seam["dtype"]), (
        rows,
        seam["centres"],
    )


def test_period_kernel_agrees_with_the_oracle(*, seam: _Seam) -> None:
    """Value, carry and consumption match the oracle reading the child period."""
    assert_kernel_agrees_with_oracle(
        kernel=seam["kernel"], context=seam["context"], child=seam["child"]
    )


def test_reference_centres_are_the_child_periods() -> None:
    """The exact reference reproduces the hand-derived child cliff centres."""
    assert (
        _grid_centres(child_top=4.0),
        _grid_centres(child_top=6.0),
        _grid_centres(child_top=8.0),
        _income_centres(increment=0.0),
        _income_centres(increment=3.0),
    ) == ((10.0, 12.0), (8.0, 11.0), (10.0, 14.0), (9.0,), (6.0,))


def test_oracle_reading_the_source_period_misses_moved_cliffs(
    *, seam: _Seam, request: pytest.FixtureRequest
) -> None:
    """Handing the oracle the source period as the child moves the cliffs.

    The positive control of the oracle's child context: on the source's grids
    and functions the oracle brackets the source's centres, which differ from
    the child's exactly when the child's grid or age-closed function moved.
    """
    rows = _oracle_rows(seam=seam, child=_source_as_child(seam=seam))
    unchanged = request.node.callspec.id in {"grid-unchanged", "income-age-invariant"}
    assert (
        _brackets_exactly(rows=rows, centres=seam["centres"], dtype=seam["dtype"])
        == unchanged
    ), rows


@pytest.mark.parametrize("jit", [False, True], ids=["eager", "jit"])
def test_production_targets_at_a_child_node_bracket_the_child_segments_cliffs(
    *, node_seam: _Seam, jit: bool
) -> None:
    """On and beside child node 4 the targets straddle the child segment's cliffs.

    Just below the node the rows are wages 0 and 4 (centres 10 and 14); on and
    just above it they are wages 4 and 8 (centres 6 and 10).
    """
    rows = _production_rows(seam=node_seam, jit=jit)
    assert _brackets_exactly(
        rows=rows, centres=node_seam["centres"], dtype=node_seam["dtype"]
    ), (rows, node_seam["centres"])


def test_oracle_targets_at_a_child_node_bracket_the_child_segments_cliffs(
    *, node_seam: _Seam
) -> None:
    """The scalar oracle reads the same child segment on and beside node 4."""
    rows = _oracle_rows(seam=node_seam, child=node_seam["child"])
    assert _brackets_exactly(
        rows=rows, centres=node_seam["centres"], dtype=node_seam["dtype"]
    ), (rows, node_seam["centres"])


def test_oracle_on_the_source_grid_brackets_the_source_segments_cliffs(
    *, node_seam: _Seam
) -> None:
    """Handing the oracle the source period as the child moves every target.

    The source grid `(0, 2, 4)` puts each of the three queries in its last
    segment, wages 2 and 4 (centres 10 and 12), which no child segment shares, so
    using the source's geometry is detected on and beside the node.
    """
    rows = _oracle_rows(seam=node_seam, child=_source_as_child(seam=node_seam))
    assert _brackets_exactly(
        rows=rows, centres=node_seam["source_centres"], dtype=node_seam["dtype"]
    ), (rows, node_seam["source_centres"])


def test_period_kernel_at_a_child_node_agrees_with_the_oracle(
    *, node_seam: _Seam
) -> None:
    """Value, carry and consumption match the oracle on and beside node 4."""
    assert_kernel_agrees_with_oracle(
        kernel=node_seam["kernel"],
        context=node_seam["context"],
        child=node_seam["child"],
    )


@pytest.mark.parametrize(
    ("step", "child_centres", "source_centres"),
    [
        (-1, (10.0, 14.0), (10.0, 12.0)),
        (0, (6.0, 10.0), (10.0, 12.0)),
        (1, (6.0, 10.0), (10.0, 12.0)),
    ],
    ids=tuple(_NODE_STEPS),
)
def test_node_reference_centres(
    *,
    step: int,
    child_centres: tuple[float, ...],
    source_centres: tuple[float, ...],
) -> None:
    """The exact segment reference reproduces the hand-derived centres."""
    query = Fraction(_node_persistence(step)) * 4
    assert (
        _segment_centres(nodes=_CHILD_WAGE_NODES, query=query),
        _segment_centres(nodes=_SOURCE_WAGE_NODES, query=query),
    ) == (child_centres, source_centres)
