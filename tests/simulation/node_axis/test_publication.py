"""Publish finite-support expectations without contributions from null events."""

import jax.numpy as jnp
import numpy as np
import pytest

from lcm.shocks import PhysicalValues
from tests.simulation._node_axis_models import _probe_model, _two_axis_model


@pytest.mark.coverage(backends=("cpu",), precisions="both")
def test_zero_mass_node_cannot_poison_a_mixed_solver_reader() -> None:
    """Adding an unrelated EGM reader cannot change the probe's expectation."""
    values = []
    for mixed in (False, True):
        model = _probe_model(
            mixed=mixed, nodes=(0.0, 1.0), weights=(1.0, 0.0), bad_node=1.0, payoff=4.0
        )
        result = model.solve(
            params={"discount_factor": 1.0} if mixed else {},
            log_level="off",
            runtime_checks=False,
            max_compilation_workers=1,
        )
        values.append(float(np.asarray(result.values[0]["probe"])))
    np.testing.assert_array_equal(values, [4.0, 4.0])


@pytest.mark.coverage(backends=("cpu",), precisions="both")
@pytest.mark.parametrize(
    ("nodes", "weights", "bad_node", "payoff"),
    [
        ((1.0, 0.0), (0.0, 1.0), 1.0, 4.0),  # declaration permutation
        ((0.0, 1.0), (0.0, 1.0), 0.0, -3.0),  # null low node, finite translation
        ((0.0, 0.5, 1.0), (0.5, 0.0, 0.5), 0.5, 8.0),  # null interior node
        ((0.0, 1.0), (0.75, 0.25), 1.0, 4.0),  # positive mass must retain -inf
        ((0.0, 1.0), (0.75, 0.25), 2.0, 4.0),  # all finite, positive fast path
    ],
)
def test_zero_mass_publication_class(
    *,
    nodes: tuple[float, ...],
    weights: tuple[float, ...],
    bad_node: float,
    payoff: float,
) -> None:
    model = _probe_model(
        mixed=True, nodes=nodes, weights=weights, bad_node=bad_node, payoff=payoff
    )
    result = model.solve(
        params={"discount_factor": 1.0},
        log_level="off",
        runtime_checks=False,
        max_compilation_workers=1,
    )
    actual = float(np.asarray(result.values[0]["probe"]))
    # An independent finite-support oracle, with null events omitted by definition.
    positive_bad = any(
        w > 0 and z == bad_node for z, w in zip(nodes, weights, strict=True)
    )
    if positive_bad:
        assert np.isneginf(actual)
    else:
        np.testing.assert_array_equal(actual, payoff)


@pytest.mark.coverage(backends=("cpu",), precisions="both")
@pytest.mark.parametrize("reverse", [False, True])
def test_two_null_axes_keep_ordinary_state_and_simulation_publication(
    *, reverse: bool
) -> None:
    """A product of null-event rules reduces two axes and keeps the offset axis.

    Positive support is the single joint point (eps, eta)=(0,0), so the literal
    expectation is exactly 4+offset, independent of axis declaration order.
    Replaying the same positive-support point also traverses present_simulation.
    """
    model = _two_axis_model(reverse=reverse)
    params = {"discount_factor": 1.0}
    solution = model.solve(
        params=params,
        log_level="off",
        runtime_checks=False,
        max_compilation_workers=1,
    )
    expected = np.asarray([5.0, 6.0])
    np.testing.assert_array_equal(solution.values[0]["probe"], expected)
    dtype = jnp.asarray(0.0).dtype
    frame = model.simulate(
        params=params,
        solution=solution,
        initial_conditions={
            "offset": np.asarray([1.0, 2.0], dtype=dtype),
            "wealth": np.ones(2, dtype=dtype),
            "age": np.zeros(2, dtype=np.int32),
            "regime_id": np.full(2, model.regime_names_to_ids["probe"], dtype=np.int32),
            "eps": PhysicalValues(values=np.zeros(2, dtype=dtype)),
            "eta": PhysicalValues(values=np.zeros(2, dtype=dtype)),
        },
        seed=17,
        log_level="off",
        runtime_checks=False,
    )
    np.testing.assert_array_equal(frame.period_to_regime_to_V_arr[0]["probe"], expected)
    realized = frame.raw_results["probe"][0]
    np.testing.assert_array_equal(realized.V_arr, expected)
    np.testing.assert_array_equal(realized.shocks["eps"], np.zeros(2))
    np.testing.assert_array_equal(realized.shocks["eta"], np.zeros(2))
