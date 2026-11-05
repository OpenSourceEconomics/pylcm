"""NBEGM integrates a post-decision shock out of the regime's state space.

An IID income process that `alive` reads only as its draw `next_income` is a
post-decision shock: nothing chosen in the period depends on it, so the regime
stores no axis for it and the solve integrates the draw over the process's
nodes with their weights.

The reference is the same model whose cash-on-hand reads the current income at
zero weight, `resources + 0 * income`. Reading the current value keeps the income as
an ordinary state of `alive`, so that route solves every lagged node separately
and the draw toward `alive` persists as the child's node. Its value is
constant along the income axis, and the integrated route's value must equal the
reference's value integrated over the nodes with the node weights.
"""

from collections.abc import Callable, Mapping
from typing import cast

import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array

from lcm import LinSpacedGrid, Model, NormalIIDProcess
from lcm.exceptions import RegimeInitializationError
from lcm.typing import ContinuousState, FloatND, UserParams, UserParamsNode
from tests.conftest import assert_agrees_to_ulp
from tests.solution import test_transition_local_process_draw as alive_dead
from tests.test_models import nbegm_common
from tests.test_models import nbegm_medicaid_toy as medicaid
from tests.test_models import nbegm_stochastic_node_toy as toy
from tests.test_models.nbegm_common import make_alive_dead_model

# Both routes form the same continuation from the same carries; they differ in
# the compiled kernel (a vmapped income axis or none) and in the reference's own
# weighted node sum of identical values. Observed up to 3 ULP at the value
# array's scale at either precision.
_N_ULP = 8


def _draw_toy(*, tax_kind: str) -> Model:
    """The schedule toy on small grids, its liquid law reading the income draw."""
    return toy.build_model(
        variant="nbegm",
        n_periods=3,
        income_timing="draw",
        tax_kind=tax_kind,
        n_liquid=12,
        n_savings=16,
        n_consumption=24,
    )


def _affine_resources_reading_income(
    *, liquid: ContinuousState, base_income: float, income: ContinuousState
) -> FloatND:
    """The affine-law fixture's cash-on-hand, reading the income at zero weight."""
    return alive_dead._resources(liquid=liquid, base_income=base_income) + 0.0 * income


def _taxed_resources_reading_income(
    *,
    liquid: ContinuousState,
    tax: FloatND,
    base_income: float,
    income: ContinuousState,
) -> FloatND:
    """The schedule fixtures' cash-on-hand, reading the income at zero weight."""
    return toy.resources(liquid=liquid, tax=tax, base_income=base_income) + 0.0 * income


type _Fixture = tuple[Callable[[], Model], Callable[[], UserParams], NormalIIDProcess]


def _fixtures() -> dict[str, _Fixture]:
    draw_income = NormalIIDProcess(
        n_points=toy.N_INCOME_NODES,
        gauss_hermite=True,
        mu=toy.DRAW_INCOME_MU,
        sigma=toy.DRAW_INCOME_SIGMA,
    )
    return {
        "affine_law": (
            lambda: alive_dead._model(solver="nbegm", process="iid"),
            lambda: alive_dead._PARAMS,
            NormalIIDProcess(n_points=5, gauss_hermite=True, mu=0.0, sigma=0.3),
        ),
        "kink_schedule": (
            lambda: _draw_toy(tax_kind="kink"),
            lambda: toy.build_params(income_timing="draw"),
            draw_income,
        ),
        "jump_schedule": (
            lambda: _draw_toy(tax_kind="jump"),
            lambda: toy.build_params(tax_lump=1.0, income_timing="draw"),
            draw_income,
        ),
    }


_NAMES = tuple(_fixtures())


def _models(
    *, name: str, monkeypatch: pytest.MonkeyPatch
) -> tuple[Model, Model, UserParams, NormalIIDProcess]:
    """The integrated model, the axis-keeping reference, params and the process."""
    build_model, build_params, income = _fixtures()[name]
    integrated = build_model()
    with monkeypatch.context() as patch:
        patch.setattr(alive_dead, "_resources", _affine_resources_reading_income)
        patch.setattr(toy, "resources", _taxed_resources_reading_income)
        reference = build_model()
    return integrated, reference, build_params(), income


@pytest.mark.parametrize("name", _NAMES)
def test_the_integrated_regime_stores_no_shock_axis(
    *, name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    integrated, reference, _, _ = _models(name=name, monkeypatch=monkeypatch)

    assert integrated.state_names(regime_name="alive") == ("liquid",)
    assert "alive: integrated=[income]" in repr(integrated)
    assert set(reference.state_names(regime_name="alive")) == {"income", "liquid"}


@pytest.mark.parametrize("name", _NAMES)
def test_value_is_the_node_weighted_reference_value(
    *, name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    integrated, reference, params, income = _models(name=name, monkeypatch=monkeypatch)
    weights = np.asarray(income.get_transition_probs(), dtype=np.float64)[0]
    order = reference.state_names(regime_name="alive")
    income_axis = order.index("income")
    got = integrated.solve(params=params, log_level="off").values
    expected = reference.solve(params=params, log_level="off").values

    assert sorted(got) == sorted(expected)
    for period, regimes in expected.items():
        assert sorted(got[period]) == sorted(regimes)
        for regime, value in regimes.items():
            value_arr = np.asarray(value)
            if regime == "alive":
                # The reference solves every lagged node alike.
                assert_agrees_to_ulp(
                    got=value_arr,
                    expected=np.broadcast_to(
                        np.take(value_arr, [0], axis=income_axis), value_arr.shape
                    ),
                    n_ulp=_N_ULP,
                    err_msg=f"{name} reference period={period}",
                )
                value_arr = np.tensordot(
                    weights.astype(value_arr.dtype),
                    value_arr,
                    axes=([0], [income_axis]),
                )
            assert_agrees_to_ulp(
                got=np.asarray(got[period][regime]),
                expected=value_arr,
                n_ulp=_N_ULP,
                operand_magnitude=float(np.max(np.abs(value_arr))),
                err_msg=f"{name} period={period} regime={regime}",
            )


@pytest.mark.parametrize("name", _NAMES)
def test_first_period_consumption_is_the_reference_policy(
    *, name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Consumption chosen before any draw is the reference's at every lagged node."""
    integrated, reference, params, income = _models(name=name, monkeypatch=monkeypatch)
    n_nodes = income.n_points
    liquid = jnp.linspace(3.0, 9.0, 4)
    n = liquid.size * n_nodes
    common = {
        "liquid": jnp.repeat(liquid, n_nodes),
        "age": jnp.zeros(n),
        "regime_id": jnp.zeros(n, dtype=jnp.int32),
    }
    nodes = jnp.tile(jnp.asarray(income.to_jax()), liquid.size)

    def first_consumption(
        *, model: Model, conditions: Mapping[str, Array]
    ) -> np.ndarray:
        df = model.simulate(
            params=params, initial_conditions=conditions, log_level="off", seed=0
        ).to_dataframe()
        first = df.query("period == 0").sort_values("subject_id")
        return first["consumption"].to_numpy()

    got = first_consumption(model=integrated, conditions=common)
    expected = first_consumption(
        model=reference, conditions={**common, "income": nodes}
    )

    assert_agrees_to_ulp(
        got=got,
        expected=expected,
        n_ulp=_N_ULP,
        operand_magnitude=float(np.max(np.abs(expected))),
        err_msg=name,
    )


def _next_liquid_reading_the_shock_draw(
    *, savings: FloatND, next_shock: ContinuousState, return_liquid: float
) -> ContinuousState:
    return (1.0 + return_liquid) * savings + 0.5 * jnp.exp(next_shock)


def _case_piece_params() -> UserParams:
    """The Medicaid toy's params, with a liquid law that reads no flat income."""
    params = dict(medicaid.build_params())
    alive = dict(cast("Mapping[str, UserParamsNode]", params["alive"]))
    law = {"next_liquid": {"return_liquid": 0.03}}
    return {**params, "alive": {**alive, "alive": law, "dead": law}}


def test_a_case_piece_budget_with_a_post_decision_shock_is_refused() -> None:
    """The case-piece kernel reads no draw, so the regime is refused at build."""
    with pytest.raises(RegimeInitializationError, match="post-decision shock"):
        make_alive_dead_model(
            n_periods=3,
            n_liquid=12,
            liquid_max=20.0,
            n_consumption=24,
            alive_functions={
                "utility": nbegm_common.utility,
                "medicaid_eligible": medicaid.medicaid_eligible,
                "subsidy_medicaid": medicaid.subsidy_medicaid,
                "subsidy_private": medicaid.subsidy_private,
                "subsidy": medicaid.subsidy,
                "resources": medicaid.resources,
                "savings": nbegm_common.savings,
            },
            liquid_law=_next_liquid_reading_the_shock_draw,
            alive_solver=nbegm_common.resolve_solver(
                variant="nbegm",
                savings_grid=LinSpacedGrid(start=0.0, stop=20.0, n_points=16),
            ),
            constraints={},
            extra_states={
                "shock": NormalIIDProcess(
                    n_points=5, gauss_hermite=True, mu=0.0, sigma=0.2
                )
            },
        ).solve(params=_case_piece_params(), log_level="off")
