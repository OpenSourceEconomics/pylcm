from __future__ import annotations

import jax.numpy as jnp
import pandas as pd
from pybaum import tree_equal

from lcm.interfaces import InternalSimulationPeriodResults
from lcm.simulation.processing import (
    _compute_targets,
    as_panel,
    process_simulated_data,
)


def test_compute_targets():
    processed_results = {
        "a": jnp.arange(3),
        "b": 1 + jnp.arange(3),
        "c": 2 + jnp.arange(3),
    }

    def f_a(a, params):
        return a + params["disutility_of_work"]

    def f_b(b, params):  # noqa: ARG001
        return b

    def f_c(params):  # noqa: ARG001
        return None

    functions = {"fa": f_a, "fb": f_b, "fc": f_c}

    got = _compute_targets(
        processed_results=processed_results,
        targets=["fa", "fb"],
        functions=functions,  # type: ignore[arg-type]
        params={"disutility_of_work": -1.0},
    )
    expected = {
        "fa": jnp.arange(3) - 1.0,
        "fb": 1 + jnp.arange(3),
    }
    assert tree_equal(expected, got)


def test_as_panel():
    processed = {
        "value": -6 + jnp.arange(6),
        "a": jnp.arange(6),
        "b": 6 + jnp.arange(6),
    }
    got = as_panel(processed, n_periods=2)
    expected = pd.DataFrame(
        {
            "period": [0, 0, 0, 1, 1, 1],
            "initial_state_id": [0, 1, 2, 0, 1, 2],
            **processed,
        },
    ).set_index(["period", "initial_state_id"])
    pd.testing.assert_frame_equal(got, expected)


def test_process_simulated_data():
    simulated = {
        0: InternalSimulationPeriodResults(
            value=jnp.array([0.1, 0.2]),
            states={"a": jnp.array([1, 2]), "b": jnp.array([-1, -2])},
            actions={"c": jnp.array([5, 6]), "d": jnp.array([-5, -6])},
        ),
        1: InternalSimulationPeriodResults(
            value=jnp.array([0.3, 0.4]),
            states={
                "b": jnp.array([-3, -4]),
                "a": jnp.array([3, 4]),
            },
            actions={
                "d": jnp.array([-7, -8]),
                "c": jnp.array([7, 8]),
            },
        ),
    }
    expected = {
        "value": jnp.array([0.1, 0.2, 0.3, 0.4]),
        "c": jnp.array([5, 6, 7, 8]),
        "d": jnp.array([-5, -6, -7, -8]),
        "a": jnp.array([1, 2, 3, 4]),
        "b": jnp.array([-1, -2, -3, -4]),
    }

    got = process_simulated_data(
        simulated,
        # Rest is none, since we are not computing any additional targets
        internal_regime=None,  # type: ignore[arg-type]
        params=None,  # type: ignore[arg-type]
        additional_targets=None,
    )
    assert tree_equal(expected, got)
