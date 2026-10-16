"""Public scheduled NB solves share source clocks and fixed borrowing floors."""

import math

import jax.numpy as jnp
import numpy as np
import pytest

from lcm import AgeGrid, IrregSpacedGrid, LiquidMargin, Model
from lcm.consumption_savings_regime import post_decision_lower_bound
from lcm.exceptions import ModelInitializationError
from lcm.regime import Regime
from lcm.solvers import NBEGM
from lcm.typing import ContinuousState, DiscreteAction, FloatND
from tests.conftest import DECIMAL_PRECISION, EXACT_KERNEL_SKIP_REASON
from tests.solution import test_dynamic_nbegm_floor as original

pytestmark = pytest.mark.requires_exact_affine_kernel(reason=EXACT_KERNEL_SKIP_REASON)

_MARGIN = LiquidMargin(
    state="liquid",
    action="consumption",
    resources="resources",
    post_decision_state="savings",
)


@pytest.mark.parametrize(
    "with_discrete", [False, True], ids=["scalar", "scalar_discrete"]
)
def test_scheduled_nbegm_uses_each_source_period_floor(*, with_discrete: bool) -> None:
    """Both source periods use their own floor above the same declared tax cliff."""
    params = {
        "discount_factor": 1.0,
        "saving": {"tax": {"tax_exemption": 1.0, "tax_lump": 0.5}},
    }
    control = original._build_model(
        lower=0.0, with_discrete=with_discrete, with_schedule=True
    ).solve(params=params, log_level="off")
    assert -1.0 / (1.5 - 0.5) + 3.0 / 4.0 < 0.0
    np.testing.assert_array_almost_equal(
        [control.values[period]["saving"][1] for period in (0, 1)],
        [math.log(4.0) + int(with_discrete)] * 2,
        decimal=DECIMAL_PRECISION,
    )

    model = original._build_model(
        lower=original.floor, with_discrete=with_discrete, with_schedule=True
    )
    solution = model.solve(params=params, log_level="off")
    expected = []
    for bound in (0.25, 0.5):
        consumption = (1.5 - 0.5) - bound
        assert -1.0 / consumption + 3.0 / (3.0 * bound + 4.0) < 0.0
        expected.append(
            math.log(consumption) + math.log(3.0 * bound + 4.0) + int(with_discrete)
        )
    np.testing.assert_array_almost_equal(
        [solution.values[period]["saving"][1] for period in (0, 1)],
        expected,
        decimal=DECIMAL_PRECISION,
    )


def _rebuild_source(*, baseline: Model, source: Regime) -> Model:
    """Keep the scalar fixture's terminal grids, source clocks and public graph."""
    return Model(
        regimes={"saving": source, "done": baseline.user_regimes["done"]},
        ages=AgeGrid(exact_values=(10, 20, 30)),
        edges={"saving": {"done": (10, 20)}},
        initial_nodes={10: "saving", 20: "saving"},
        regime_id_class=original.RegimeId,
    )


def test_callable_nbegm_refuses_runtime_only_savings_grid() -> None:
    """A callable NB floor requires actual savings-grid points at model build."""
    baseline = original._build_model(lower=0.0, with_discrete=False)
    source = baseline.user_regimes["saving"].replace(
        solver=NBEGM(savings_grid=IrregSpacedGrid(n_points=3)),
        constraints={
            "borrowing": post_decision_lower_bound(margin=_MARGIN, lower=original.floor)
        },
    )
    with pytest.raises(
        ModelInitializationError,
        match=r"Regime 'saving'.*callable borrowing bound.*savings grid.*runtime",
    ):
        _rebuild_source(baseline=baseline, source=source)


def _floor_from_helper(*, floor_helper: FloatND) -> FloatND:
    return floor_helper


def _floor_reading_liquid(*, liquid: ContinuousState) -> FloatND:
    return liquid


def _floor_reading_choice(*, choice: DiscreteAction) -> FloatND:
    return 0.0 + choice


@pytest.mark.parametrize("leaf", ["liquid", "choice"])
def test_nbegm_floor_refuses_liquid_and_discrete_action_helpers(*, leaf: str) -> None:
    """The bound's transitive RHS cannot read the liquid state or chosen action."""
    baseline = original._build_model(lower=0.0, with_discrete=leaf == "choice")
    original_source = baseline.user_regimes["saving"]
    source = original_source.replace(
        functions={
            **dict(original_source.functions),
            "floor_helper": (
                _floor_reading_liquid if leaf == "liquid" else _floor_reading_choice
            ),
        },
        constraints={
            "borrowing": post_decision_lower_bound(
                margin=_MARGIN, lower=_floor_from_helper
            )
        },
    )
    with pytest.raises(
        ModelInitializationError, match=rf"borrowing.*'{leaf}'.*floor_helper"
    ):
        _rebuild_source(baseline=baseline, source=source)


def _other_floor() -> FloatND:
    return jnp.asarray(0.125)


def test_nbegm_refuses_distinct_callable_floors_together() -> None:
    """One NB source cannot silently choose between distinct declared floors."""
    baseline = original._build_model(lower=0.0, with_discrete=False)
    source = baseline.user_regimes["saving"].replace(
        constraints={
            "first": post_decision_lower_bound(margin=_MARGIN, lower=original.floor),
            "second": post_decision_lower_bound(margin=_MARGIN, lower=_other_floor),
        },
    )
    with pytest.raises(
        ModelInitializationError,
        match=r"Regime 'saving'.*distinct lower bounds.*first.*second",
    ):
        _rebuild_source(baseline=baseline, source=source)
