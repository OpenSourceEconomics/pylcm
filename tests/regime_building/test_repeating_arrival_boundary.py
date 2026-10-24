"""A repeating arrival stops where the original source's supported edges stop."""

import jax.numpy as jnp
import numpy as np

from lcm import AgeGrid, Model, Transition, categorical
from lcm.solver_api import ResultRetention
from lcm.typing import ScalarInt
from tests.regime_building.test_collective_regime_simulate import (
    _BETA,
    _make_repeating_self_loop_regimes,
    _repeating_self_loop_laws,
)


@categorical(ordered=False)
class _RegimeId:
    src: ScalarInt
    src_exit: ScalarInt
    src_fallback: ScalarInt


def test_public_repeating_arrival_stops_at_the_sources_last_supported_date() -> None:
    """A kept row exits ordinarily after the actual final source date."""
    declaration = _repeating_self_loop_laws()["src"]
    assert isinstance(declaration, Transition)
    model = Model(
        regimes=_make_repeating_self_loop_regimes(),
        ages=AgeGrid(start=0, inclusive_stop=3, step="Y"),
        regime_id_class=_RegimeId,
        initial_nodes={0: "src"},
        edges={
            "src": Transition(
                targets={"src": 0, "src_exit": (0, 1), "src_fallback": 0},
                law=declaration.law,
                arrivals=declaration.arrivals,
            )
        },
    )
    params = {"discount_factor": _BETA}
    solution = model.solve(
        params=params, retention=ResultRetention.VALUES, log_level="off"
    )
    assert "src" in solution.values[0]
    assert "src" in solution.values[1]
    assert "src" not in solution.values[2]
    np.testing.assert_allclose(solution.values[0]["src"], [1.095, 4.8025], rtol=1e-6)
    np.testing.assert_allclose(solution.values[1]["src"], [1.475, 2.95], rtol=1e-6)
    result = model.simulate(
        params=params,
        initial_conditions={
            "wage": jnp.asarray([1.0, 2.0]),
            "age": jnp.asarray([0.0, 0.0]),
            "regime_id": jnp.full(2, model.regime_names_to_ids["src"], dtype=jnp.int32),
        },
        solution=solution,
        log_level="off",
        seed=0,
    ).to_dataframe()
    first = result.loc[result["period"] == 0].sort_values("subject_id")
    np.testing.assert_array_equal(first["subject_id"], [0, 1])
    np.testing.assert_allclose(first["value"], [1.095, 4.8025], rtol=1e-6)
    landing = result.loc[result["period"] == 1].sort_values("subject_id")
    np.testing.assert_array_equal(landing["subject_id"], [0, 1])
    np.testing.assert_array_equal(
        landing["regime_name"].to_numpy(), ["src_fallback", "src"]
    )
    np.testing.assert_array_equal(landing["wage"], [1.0, 2.0])
    np.testing.assert_allclose(landing["value"], [0.1, 2.95], rtol=1e-6)
    exited = result.loc[(result["period"] == 2) & (result["regime_name"] == "src_exit")]
    np.testing.assert_array_equal(exited["subject_id"], [1])
    np.testing.assert_array_equal(exited["wage"], [2.0])
    np.testing.assert_allclose(exited["value"], [1.0], rtol=1e-6)
