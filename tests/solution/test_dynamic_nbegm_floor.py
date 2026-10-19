"""Current periods determine the floor on each supported scalar NB-EGM branch."""

import math

import jax.numpy as jnp
import numpy as np
import pytest

from lcm import (
    AgeGrid,
    ConsumptionSavingsRegime,
    DiscreteGrid,
    IrregSpacedGrid,
    LiquidMargin,
    Model,
    Regime,
    categorical,
    post_decision_lower_bound,
)
from lcm.solvers import NBEGM
from lcm.typing import (
    ContinuousAction,
    ContinuousState,
    DiscreteAction,
    FloatND,
    ScalarInt,
    UserFunction,
)
from tests.conftest import DECIMAL_PRECISION, EXACT_KERNEL_SKIP_REASON
from tests.test_models.nbegm_common import savings as savings_from_resources
from tests.test_models.nbegm_discrete_cliff_toy import tax_cliff


@categorical(ordered=False)
class RegimeId:
    saving: ScalarInt
    done: ScalarInt


@categorical(ordered=False)
class Choice:
    first: ScalarInt
    second: ScalarInt


def utility(*, consumption: ContinuousAction) -> FloatND:
    return jnp.log(consumption)


def utility_with_choice(
    *, consumption: ContinuousAction, choice: DiscreteAction
) -> FloatND:
    return jnp.log(consumption) + choice


def savings(*, liquid: ContinuousState, consumption: ContinuousAction) -> FloatND:
    return liquid - consumption


def resources(*, liquid: ContinuousState) -> FloatND:
    return liquid


def resources_net_of_tax(*, liquid: ContinuousState, tax: FloatND) -> FloatND:
    return liquid - tax


def next_liquid(*, savings: FloatND) -> ContinuousState:
    return 3.0 * savings + 4.0


def terminal(*, liquid: ContinuousState) -> FloatND:
    return jnp.log(liquid)


def floor(*, period: ScalarInt) -> FloatND:
    return jnp.where(period == 0, 0.25, 0.5)


def _build_model(
    *,
    lower: float | UserFunction,
    with_discrete: bool,
    with_schedule: bool = False,
) -> Model:
    """Build the same scalar budget under a literal or callable borrowing floor."""
    margin = LiquidMargin(
        state="liquid",
        action="consumption",
        resources="resources",
        post_decision_state="savings",
    )
    return Model(
        regimes={
            "saving": ConsumptionSavingsRegime(
                liquid=margin,
                states={"liquid": IrregSpacedGrid(points=(0.75, 1.5, 3.0))},
                actions={
                    "consumption": IrregSpacedGrid(points=(0.125, 0.25, 0.5, 1.0)),
                    **(
                        {"choice": DiscreteGrid(category_class=Choice)}
                        if with_discrete
                        else {}
                    ),
                },
                functions={
                    "utility": utility_with_choice if with_discrete else utility,
                    "savings": savings_from_resources if with_schedule else savings,
                    "resources": resources_net_of_tax if with_schedule else resources,
                    **({"tax": tax_cliff} if with_schedule else {}),
                },
                state_transitions={"liquid": next_liquid},
                constraints={
                    "borrowing": post_decision_lower_bound(margin=margin, lower=lower)
                },
                solver=NBEGM(savings_grid=IrregSpacedGrid(points=(0.0, 0.5, 1.0))),
            ),
            "done": Regime(
                states={
                    "liquid": IrregSpacedGrid(points=(4.0, 4.75, 5.5, 6.25, 7.0, 8.5))
                },
                functions={"utility": terminal},
            ),
        },
        ages=AgeGrid(exact_values=(10, 20, 30)),
        edges={"saving": {"done": (10, 20)}},
        initial_nodes={10: "saving", 20: "saving"},
        regime_id_class=RegimeId,
    )


@pytest.mark.requires_exact_affine_kernel(reason=EXACT_KERNEL_SKIP_REASON)
def test_discrete_nbegm_resource_map_excludes_utility_only_choice() -> None:
    """A utility-only choice preserves the exact literal-floor budget value."""
    solution = _build_model(lower=0.0, with_discrete=True).solve(
        params={"discount_factor": 1.0}, log_level="off"
    )
    assert -1.0 / 0.75 + 3.0 / 4.0 < 0.0
    np.testing.assert_array_almost_equal(
        [solution.values[period]["saving"][0] for period in (0, 1)],
        [math.log(3.0) + 1.0] * 2,
        decimal=DECIMAL_PRECISION,
    )


@pytest.mark.requires_exact_affine_kernel(reason=EXACT_KERNEL_SKIP_REASON)
def test_schedule_discrete_nbegm_resource_map_excludes_utility_only_choice() -> None:
    """A utility-only choice preserves the exact budget above a nonzero tax cliff."""
    solution = _build_model(lower=0.0, with_discrete=True, with_schedule=True).solve(
        params={
            "discount_factor": 1.0,
            "saving": {"tax": {"tax_exemption": 1.0, "tax_lump": 0.5}},
        },
        log_level="off",
    )
    assert -1.0 / (1.5 - 0.5) + 3.0 / 4.0 < 0.0
    np.testing.assert_array_almost_equal(
        [solution.values[period]["saving"][1] for period in (0, 1)],
        [math.log(4.0) + 1.0] * 2,
        decimal=DECIMAL_PRECISION,
    )


@pytest.mark.requires_exact_affine_kernel(reason=EXACT_KERNEL_SKIP_REASON)
@pytest.mark.parametrize(
    "with_discrete", [False, True], ids=["scalar", "scalar_discrete"]
)
def test_scalar_nbegm_uses_each_source_period_floor(*, with_discrete: bool) -> None:
    """Two admitted source periods publish their distinct exact constrained values."""
    control = _build_model(lower=0.0, with_discrete=with_discrete).solve(
        params={"discount_factor": 1.0}, log_level="off"
    )
    assert -1.0 / 0.75 + 3.0 / 4.0 < 0.0
    np.testing.assert_array_almost_equal(
        [control.values[period]["saving"][0] for period in (0, 1)],
        [math.log(3.0) + int(with_discrete)] * 2,
        decimal=DECIMAL_PRECISION,
    )
    model = _build_model(lower=floor, with_discrete=with_discrete)
    solution = model.solve(params={"discount_factor": 1.0}, log_level="off")
    expected = []
    for bound in (0.25, 0.5):
        consumption = 0.75 - bound
        assert -1.0 / consumption + 3.0 / (3.0 * bound + 4.0) < 0.0
        expected.append(
            math.log(consumption) + math.log(3.0 * bound + 4.0) + int(with_discrete)
        )
    np.testing.assert_array_almost_equal(
        [solution.values[period]["saving"][0] for period in (0, 1)],
        expected,
        decimal=DECIMAL_PRECISION,
    )
