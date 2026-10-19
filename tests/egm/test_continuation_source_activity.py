"""Retained conditional rows keep attempted reads independently of publication."""

from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.egm.carry import EGMCarry
from _lcm.egm.query_data import QueryNodeKind
from _lcm.execution.core_program import (
    CoreBuildContext,
    core_program_graph,
    materialize_core_program,
    select_programs,
)
from _lcm.solution.dcegm import _DCEGMPeriodKernel
from _lcm.solution.feasibility import BoundedResult, ContinuationReadCounts
from _lcm.typing import PytreeValue
from _lcm.utils.dispatchers import map_over_leading_axis
from lcm import IrregSpacedGrid, LiquidMargin, Model, post_decision_lower_bound
from lcm.solvers import DCEGM
from lcm.typing import BoolND, ScalarInt
from tests.egm.test_below_first_live_counts import _model, _params
from tests.egm.test_dynamic_dcegm_activation import RegimeId, _lower
from tests.solution._nbegm_direct_oracle import ride_along_kernel


def _never() -> BoolND:
    return jnp.asarray(a=False)


@pytest.mark.parametrize(
    "source_active", [True, False], ids=["retained-bank", "padding"]
)
def test_retained_conditional_reads_survive_publication_rejection(
    *, source_active: bool
) -> None:
    """A rejected public state keeps its real bank reads; padded rows count none."""
    seed = _model(literal=False, parent_nodes=(-0.125, 0.5, 1.0))
    margin = LiquidMargin(
        state="liquid",
        action="consumption",
        resources="resources",
        post_decision_state="savings",
    )
    parent = seed.user_regimes["parent"].replace(
        solver=DCEGM(
            savings_grid=IrregSpacedGrid(points=np.array([0.0, 0.625, 1.125]))
        ),
        constraints={
            "borrowing": post_decision_lower_bound(margin=margin, lower=_lower),
            "never": _never,
        },
    )
    model = Model(
        regimes={**seed.user_regimes, "parent": parent},
        ages=seed.ages,
        edges=seed.edges,
        initial_nodes={0: "parent"},
        regime_id_class=RegimeId,
        enable_jit=False,
    )
    params = _params(literal=False, parent=True)
    params = {
        **params,
        "parent": {
            "koopmans_aggregator": {"discount_factor": 1.0},
            "borrowing_lower": {"floor": -0.125},
        },
    }
    kernel, context = ride_along_kernel(
        model=model,
        params=params,
        regime_name="parent",
        period=0,
    )
    assert isinstance(kernel, _DCEGMPeriodKernel)
    selected = select_programs(
        graph=core_program_graph(kernel=kernel),
        retain_replay=False,
        collect_continuation_read_counts=True,
    )
    program = materialize_core_program(
        program=selected["main"],
        context=CoreBuildContext(**context),
    )
    function: Callable[..., PytreeValue] = program.function
    produced = function(**program.arguments)
    assert isinstance(produced, BoundedResult)
    assert isinstance(produced.continuation_read_counts, ContinuationReadCounts)
    for flag in jax.tree.leaves((produced.bound_status, produced.query_status)):
        np.testing.assert_array_equal(actual=flag, desired=False)

    def retain_source(_row: ScalarInt) -> BoundedResult[PytreeValue]:
        return produced

    mapped = map_over_leading_axis(
        func=retain_source,
        xs=jnp.arange(1, dtype=jnp.int32),
        batch_size=0,
        bound_summary=True,
        active=jnp.asarray([source_active]),
    )
    assert isinstance(mapped.output, tuple)
    assert len(mapped.output) == 3
    value, carry, _feasibility = mapped.output
    assert isinstance(value, jax.Array)
    assert isinstance(carry, EGMCarry)
    np.testing.assert_array_equal(value, -np.inf)
    data = carry.query_data_for_transport()
    assert data is not None
    supported = data.node_kind == QueryNodeKind.FINITE
    assert bool(jnp.any(supported))
    assert bool(jnp.all(jnp.isfinite(carry.leaves()[("value",)])[supported]))
    # Three distinct Euler source candidates query the actual child at -.125,.5,1.
    # Its constrained prefix is positive and below .5, so precisely one is below.
    child = context["next_regime_to_continuation"]["child"]
    assert isinstance(child, EGMCarry)
    first = child.leaves()[("endog_grid",)][0]
    assert 0.0 < first < 0.5
    np.testing.assert_array_equal(
        mapped.continuation_read_counts, [1, 3] if source_active else [0, 0]
    )
