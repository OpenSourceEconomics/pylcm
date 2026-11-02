"""NNBEGM integrates a post-decision shock out of the regime's state space.

An IID income process that `alive` reads only as its draw `next_income`, in the
liquid law toward itself, is a post-decision shock: nothing chosen in the period
depends on it, so the regime stores no axis for it and the inner NB-EGM solve
integrates the draw over the process's nodes with their weights.

The reference is the same model whose cash-on-hand reads the current income at
zero weight. Reading the current value keeps the income as an ordinary state of
`alive`, so that route solves every lagged node separately and the draw toward
`alive` persists as the child's node. Its value is constant along the income
axis, and the integrated route's value must equal the reference's value
integrated over the nodes with the node weights.
"""

import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array

from lcm import Model, NormalIIDProcess
from lcm.typing import ContinuousState, FloatND
from tests.conftest import assert_agrees_to_ulp
from tests.test_models import n_nbegm_toy as toy

# Both routes form the same continuation from the same carries; they differ in
# the compiled kernel (a vmapped income axis or none) and in the reference's own
# weighted node sum of identical values.
_N_ULP = 8

_INCOME = NormalIIDProcess(n_points=5, gauss_hermite=True, mu=0.0, sigma=0.3)
_PARAMS = {"discount_factor": 0.95}


def _next_wealth_reading_the_draw(
    *, liquid_savings: FloatND, next_income: ContinuousState
) -> ContinuousState:
    return (1.0 + toy.LIQUID_RATE) * liquid_savings + 0.5 * jnp.exp(next_income)


def _resources_reading_income(
    *,
    wealth: ContinuousState,
    illiquid: ContinuousState,
    new_illiquid: ContinuousState,
    income: ContinuousState,
) -> FloatND:
    """The toy's cash-on-hand, reading the current income at zero weight."""
    return (
        toy.resources(wealth=wealth, illiquid=illiquid, new_illiquid=new_illiquid)
        + 0.0 * income
    )


def _model() -> Model:
    return toy.build_model(
        variant="n_nbegm",
        extra_states={"income": _INCOME},
        wealth_law={"alive": _next_wealth_reading_the_draw, "dead": toy.next_wealth},
    )


def _models(*, monkeypatch: pytest.MonkeyPatch) -> tuple[Model, Model]:
    """The integrated model and the axis-keeping reference."""
    integrated = _model()
    with monkeypatch.context() as patch:
        patch.setattr(toy, "resources", _resources_reading_income)
        reference = _model()
    return integrated, reference


def test_the_integrated_regime_stores_no_shock_axis(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    integrated, reference = _models(monkeypatch=monkeypatch)

    assert integrated.state_names(regime_name="alive") == ("wealth", "illiquid")
    assert "alive: integrated=[income]" in repr(integrated)
    assert set(reference.state_names(regime_name="alive")) == {
        "income",
        "wealth",
        "illiquid",
    }
    assert "alive: axis=[income]" in repr(reference)


def test_value_is_the_node_weighted_reference_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    integrated, reference = _models(monkeypatch=monkeypatch)
    weights = np.asarray(_INCOME.get_transition_probs(), dtype=np.float64)[0]
    income_axis = reference.state_names(regime_name="alive").index("income")
    got = integrated.solve(params=_PARAMS, log_level="off").values
    expected = reference.solve(params=_PARAMS, log_level="off").values

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
                    operand_magnitude=float(np.max(np.abs(value_arr))),
                    err_msg=f"reference period={period}",
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
                err_msg=f"period={period} regime={regime}",
            )


def test_first_period_policy_is_the_reference_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Both choices, made before any draw, are the reference's at every lagged node."""
    integrated, reference = _models(monkeypatch=monkeypatch)
    n_nodes = _INCOME.n_points
    wealth = jnp.linspace(3.0, 20.0, 4)
    illiquid = jnp.linspace(2.0, 15.0, 4)
    n = wealth.size * n_nodes
    common = {
        "wealth": jnp.repeat(wealth, n_nodes),
        "illiquid": jnp.repeat(illiquid, n_nodes),
        "age": jnp.full(n, 20.0),
        "regime_id": jnp.full(n, toy.RegimeId.alive, dtype=jnp.int32),
    }
    nodes = jnp.tile(jnp.asarray(_INCOME.to_jax()), wealth.size)

    def first_choices(*, model: Model, conditions: dict[str, Array]) -> np.ndarray:
        df = model.simulate(
            params=_PARAMS, initial_conditions=conditions, log_level="off", seed=0
        ).to_dataframe()
        first = df.query("period == 0").sort_values("subject_id")
        return first[["consumption", "illiquid_investment"]].to_numpy()

    got = first_choices(model=integrated, conditions=common)
    expected = first_choices(model=reference, conditions={**common, "income": nodes})

    assert_agrees_to_ulp(
        got=got,
        expected=expected,
        n_ulp=_N_ULP,
        operand_magnitude=float(np.max(np.abs(expected))),
    )


def test_simulation_draws_the_income_continuously() -> None:
    """The default `simulated_draws` takes next wealth off the continuous law."""
    n = 200
    df = (
        _model()
        .simulate(
            params=_PARAMS,
            initial_conditions={
                "wealth": jnp.full(n, 10.0),
                "illiquid": jnp.full(n, 5.0),
                "age": jnp.full(n, 20.0),
                "regime_id": jnp.full(n, toy.RegimeId.alive, dtype=jnp.int32),
            },
            log_level="off",
            seed=0,
        )
        .to_dataframe()
    )
    wealth = df.query("period == 1").sort_values("subject_id")["wealth"].to_numpy()

    # Identical subjects choose alike, so next wealth varies by the draw alone,
    # and a continuous draw takes as many values as there are subjects.
    assert len(np.unique(wealth)) > 10 * _INCOME.n_points
