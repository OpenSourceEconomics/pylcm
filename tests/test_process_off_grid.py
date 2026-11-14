"""A process value beyond its outer nodes follows the process's `off_grid` rule."""

from collections.abc import Callable
from dataclasses import replace

import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.processes.base import _ContinuousStochasticProcess
from _lcm.regime_building.processing import _get_weights_func_for_process
from _lcm.simulation.simulate import _row_selections
from _lcm.solution import fingerprint as fingerprints
from lcm import TauchenAR1Process
from lcm.exceptions import OffGridStateError
from lcm.typing import OffGrid
from tests.test_models.processes import RegimeId
from tests.test_process_simulation_laws import (
    _model,
    _reference_row,
    _rouwenhorst,
    _tauchen,
)

_RULES: tuple[OffGrid, ...] = ("extrapolate", "clip_lookup", "clip_state", "raise")


def _on_nodes(*, off_grid: OffGrid = "extrapolate") -> TauchenAR1Process:
    return TauchenAR1Process.on_nodes(
        nodes=jnp.array([-1.0, 0.0, 1.5]),
        rho=0.5,
        sigma=0.3,
        mu=0.0,
        off_grid=off_grid,
    )


def _simulate(
    *, process: _ContinuousStochasticProcess, starts: np.ndarray, seed: int = 11
) -> tuple[np.ndarray, np.ndarray]:
    """Return each subject's period-0 value and period-1 state."""
    model = _model(process)
    solution = model.solve(params={}, log_level="off")
    frame = model.simulate(
        params={},
        solution=solution,
        initial_conditions={
            "income": jnp.asarray(starts),
            "age": jnp.zeros(len(starts)),
            "regime_id": jnp.full(len(starts), RegimeId.alive, dtype=jnp.int32),
        },
        seed=seed,
        log_level="off",
    ).to_dataframe(use_categorical_codes=True)
    first = frame.loc[frame["period"] == 0].sort_values("subject_id")
    second = frame.loc[frame["period"] == 1].sort_values("subject_id")
    return first["value"].to_numpy(), second["income"].to_numpy()


@pytest.mark.parametrize("rule", _RULES)
def test_on_nodes_keeps_the_declared_off_grid_rule(*, rule: OffGrid) -> None:
    assert _on_nodes(off_grid=rule).off_grid == rule


def test_process_off_grid_is_not_a_distribution_parameter() -> None:
    process = TauchenAR1Process(
        n_points=3, gauss_hermite=False, n_std=2.0, off_grid="clip_state"
    )
    assert set(process.params_to_pass_at_runtime) == {"rho", "sigma", "mu"}


@pytest.mark.parametrize("rule", _RULES[1:])
def test_process_off_grid_rule_enters_the_fingerprint(*, rule: OffGrid) -> None:
    process = _tauchen()
    assert fingerprints._semantic_fingerprint(process) != (
        fingerprints._semantic_fingerprint(replace(process, off_grid=rule))
    )


@pytest.mark.parametrize("side", ["above", "below"])
@pytest.mark.parametrize("rule", ["clip_lookup", "clip_state"])
@pytest.mark.parametrize(
    "build",
    [_tauchen, _on_nodes, _rouwenhorst],
    ids=["tauchen", "on-nodes", "rouwenhorst"],
)
def test_transition_row_beyond_the_outer_nodes_is_the_edge_node_row(
    *,
    build: Callable[..., _ContinuousStochasticProcess],
    rule: OffGrid,
    side: str,
) -> None:
    """A clipped rule reads the nearest edge node's row beyond the nodes."""
    process = build(off_grid=rule)
    nodes = np.asarray(process.get_gridpoints())
    value = float(nodes[-1] + 0.7) if side == "above" else float(nodes[0] - 0.7)
    weights = _get_weights_func_for_process(name="income", grid=process)
    rows = np.asarray(process.get_transition_probs())
    np.testing.assert_allclose(
        np.asarray(weights(income=jnp.asarray(value))),
        rows[-1] if side == "above" else rows[0],
        rtol=1e-6,
    )


@pytest.mark.parametrize(
    ("rule", "expected"),
    [
        ("extrapolate", [0.0, -0.5, 1.5]),
        ("clip_lookup", [0.0, 0.0, 1.0]),
        ("clip_state", [0.0, 0.0, 1.0]),
    ],
)
def test_policy_read_beyond_the_top_node_blends_rows_by_the_rule(
    *, rule: OffGrid, expected: list[float]
) -> None:
    """Half a segment past the top node: extrapolated or clipped row coefficients."""
    regime = _model(_on_nodes(off_grid=rule))._regimes["alive"]
    selections = _row_selections(
        regime=regime,
        names=("income",),
        states={"income": jnp.array([2.25])},
    )
    coefficients = [float(np.asarray(c)[0]) for _, c in selections]
    np.testing.assert_allclose(coefficients, expected, rtol=0, atol=1e-6)


@pytest.mark.parametrize("rule", _RULES[:3])
def test_value_beyond_the_top_node_reads_the_continuation_by_the_rule(
    *, rule: OffGrid
) -> None:
    """Clipped rules read the top node's row; extrapolate reads the law's row there.

    The next period is the last alive one, so its value at a node is its utility.
    """
    process = _tauchen(rho=0.9, off_grid=rule)
    nodes = np.asarray(process.get_gridpoints(), dtype=np.float64)
    beyond = nodes[-1] + 2.0
    values, _ = _simulate(process=process, starts=np.array([beyond]))
    row = (
        _reference_row(process=process, value=beyond)
        if rule == "extrapolate"
        else np.asarray(process.get_transition_probs(), dtype=np.float64)[-1]
    )
    np.testing.assert_allclose(values[0] - beyond**2, 0.9 * row @ nodes**2, rtol=1e-5)


@pytest.mark.parametrize("rule", ["clip_lookup", "clip_state"])
def test_continuous_draw_beyond_the_nodes_is_clipped_only_under_clip_state(
    *, rule: OffGrid
) -> None:
    """`clip_state` clips the realized value to the outer nodes; `clip_lookup` not."""
    starts = np.linspace(-6.0, 6.0, 32)
    process = _tauchen(rho=0.9, off_grid=rule)
    nodes = np.asarray(process.get_gridpoints())
    _, raw = _simulate(process=replace(process, off_grid="extrapolate"), starts=starts)
    _, got = _simulate(process=process, starts=starts)
    expected = np.clip(raw, nodes[0], nodes[-1]) if rule == "clip_state" else raw
    np.testing.assert_array_equal(got, expected)


def test_continuous_draw_beyond_the_nodes_raises_under_raise() -> None:
    with pytest.raises(OffGridStateError):
        _simulate(
            process=_tauchen(rho=0.9, off_grid="raise"),
            starts=np.linspace(-6.0, 6.0, 32),
        )


@pytest.mark.parametrize("rule", _RULES)
def test_node_draw_beyond_the_top_node_follows_the_top_row_under_every_rule(
    *, rule: OffGrid
) -> None:
    """A node draw never samples an extrapolated row, whatever the lookup rule."""
    process = _tauchen(rho=0.9, simulated_draws="nodes", off_grid=rule)
    top = float(process.get_gridpoints()[-1])
    _, beyond = _simulate(process=process, starts=np.full(256, top + 5.0), seed=537)
    _, at_top = _simulate(process=process, starts=np.full(256, top), seed=537)
    np.testing.assert_array_equal(beyond, at_top)
