"""Named normal shocks share one covariance law in quadrature and simulation."""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_almost_equal as aaae

import lcm
import lcm.typing
from lcm.exceptions import (
    GridInitializationError,
    InvalidInitialConditionsError,
    InvalidParamsError,
)
from lcm.processes import JointNormalIIDProcess
from tests.conftest import DECIMAL_PRECISION


@lcm.categorical(ordered=False)
class RegimeId:
    working: lcm.typing.ScalarInt
    done: lcm.typing.ScalarInt


def _utility(
    *, x: lcm.typing.ContinuousState, y: lcm.typing.ContinuousState
) -> lcm.typing.FloatND:
    return x * y + x * x


def _manual_utility(
    *, z_1: lcm.typing.ContinuousState, z_2: lcm.typing.ContinuousState
) -> lcm.typing.FloatND:
    return z_1 * (0.4 * z_1 + jnp.sqrt(1.84) * z_2) + z_1 * z_1


# keyword-only-exempt: primary-argument=regime
def _model(
    regime: lcm.Regime, *, execution_config: lcm.ExecutionConfig | None = None
) -> lcm.Model:
    return lcm.Model(
        regimes={"working": regime, "done": regime},
        n_periods=2,
        regime_id_class=RegimeId,
        initial_nodes=lcm.InitialNodes(by_period={0: "working"}),
        edges={"working": {"done": lcm.Periods(values=(0,))}},
        fixed_params={"working": {"koopmans_aggregator": {"discount_factor": 0.9}}},
        execution_config=execution_config
        or lcm.ExecutionConfig(device_memory_bytes=None),
    )


def test_joint_normal_process_is_public() -> None:
    """A regime can declare a named joint normal IID process."""
    assert hasattr(lcm, "JointNormalIIDProcess")


@pytest.mark.parametrize("runtime", [False, True])
@pytest.mark.parametrize("fold", [False, True])
@pytest.mark.parametrize(
    "cov",
    [((1.0, -0.4), (-0.4, 2.0)), ((0.0, 0.0), (0.0, 2.0)), ((1.0, 2.0), (2.0, 4.0))],
)
def test_physical_initial_state_names_and_bounds(
    *, runtime: bool, fold: bool, cov: tuple[tuple[float, float], tuple[float, float]]
) -> None:
    """Recipe introspection names physical shocks and their marginal extents."""
    process = JointNormalIIDProcess(
        states=("x", "y"),
        n_points=(3, 5),
        gauss_hermite=True,
        mean=None if runtime else (1.0, -2.0),
        cov=None if runtime else cov,
        fold=fold,
    )
    model = _model(
        lcm.Regime(states={"joint": process}, functions={"utility": _utility})
    )
    params = (
        {
            name: {"joint": {"mean": jnp.array([1.0, -2.0]), "cov": jnp.array(cov)}}
            for name in ("working", "done")
        }
        if runtime
        else {}
    )
    assert model.initial_state_names(regime_name="working") == ("x", "y")
    assert model.state_names(regime_name="working") == (
        "joint_normal_0",
        "joint_normal_1",
    )
    physical = np.asarray(
        process.compute_gridpoints(mean=jnp.array([1.0, -2.0]), cov=jnp.array(cov))
    )
    for dimension, name in enumerate(process.states):
        bounds = model.initial_state_bounds(
            params=params, regime_name="working", state_name=name, period=0
        )
        np.testing.assert_allclose(
            bounds,
            (physical[:, dimension].min(), physical[:, dimension].max()),
            rtol=10**-DECIMAL_PRECISION,
        )


@pytest.fixture
def joint_process_class() -> type[JointNormalIIDProcess]:
    assert hasattr(lcm, "JointNormalIIDProcess"), "Joint normal process API missing"
    return lcm.JointNormalIIDProcess


@pytest.mark.parametrize("gauss_hermite", [True, False])
def test_joint_normal_tensor_matches_independent_rotation(
    *, joint_process_class: type[JointNormalIIDProcess], gauss_hermite: bool
) -> None:
    """The tensor support is the literal Cholesky rotation of scalar nodes."""
    process = joint_process_class(
        states=("husband", "wife"),
        n_points=3,
        gauss_hermite=gauss_hermite,
        mean=(1.0, -2.0),
        cov=((4.0, 1.0), (1.0, 9.0)),
        n_std=None if gauss_hermite else 3.0,
    )
    normal = lcm.NormalIIDProcess(
        n_points=3,
        gauss_hermite=gauss_hermite,
        mu=0.0,
        sigma=1.0,
        n_std=None if gauss_hermite else 3.0,
    )
    nodes = np.asarray(normal.get_gridpoints())
    weights = np.asarray(normal.get_transition_probs())[0]
    expected = []
    expected_weights = []
    for i, z_h in enumerate(nodes):
        for j, z_w in enumerate(nodes):
            expected.append((1.0 + 2.0 * z_h, -2.0 + 0.5 * z_h + np.sqrt(8.75) * z_w))
            expected_weights.append(weights[i] * weights[j])
    aaae(process.get_gridpoints(), expected, decimal=DECIMAL_PRECISION)
    aaae(process.get_weights(), expected_weights, decimal=DECIMAL_PRECISION)


@pytest.mark.parametrize("cov", [((1.0, 1.0), (1.0, 1.0)), ((0.0, 0.0), (0.0, 4.0))])
def test_joint_normal_singular_covariance(
    *,
    joint_process_class: type[JointNormalIIDProcess],
    cov: tuple[tuple[float, ...], ...],
) -> None:
    """Zero variance and perfect correlation retain their joint moments."""
    process = joint_process_class(
        states=("x", "y"),
        n_points=3,
        gauss_hermite=True,
        mean=(0.0, 0.0),
        cov=cov,
    )
    nodes, weights = process.get_gridpoints(), process.get_weights()
    aaae(
        jnp.sum(weights[:, None] * nodes, axis=0), (0.0, 0.0), decimal=DECIMAL_PRECISION
    )
    aaae(nodes.T @ (weights[:, None] * nodes), cov, decimal=DECIMAL_PRECISION)


@pytest.mark.parametrize(
    "cov",
    [((1.0, 2.0), (2.0, 1.0)), ((1.0, 0.5), (0.0, 1.0)), ((1.0, 0.0), (0.0, np.nan))],
)
def test_joint_normal_rejects_invalid_covariance(
    *,
    joint_process_class: type[JointNormalIIDProcess],
    cov: tuple[tuple[float, ...], ...],
) -> None:
    """An invalid covariance fails instead of changing the Gaussian law."""
    with pytest.raises(GridInitializationError, match="cov"):
        joint_process_class(
            states=("x", "y"),
            n_points=3,
            gauss_hermite=True,
            mean=(0.0, 0.0),
            cov=cov,
        )


def test_joint_normal_runtime_quadrature_and_draws(
    *, joint_process_class: type[JointNormalIIDProcess]
) -> None:
    """Runtime covariance binds the same nodes and continuous joint law."""
    process = joint_process_class(
        states=("x", "y"),
        n_points=3,
        gauss_hermite=True,
        mean=(0.0, 0.0),
        cov=None,
    )
    cov = jnp.array(((1.0, 0.4), (0.4, 2.0)))
    nodes = jax.jit(process.compute_gridpoints)(cov=cov)
    weights = process.get_weights()
    aaae(nodes.T @ (weights[:, None] * nodes), cov, decimal=DECIMAL_PRECISION)
    keys = jax.random.split(jax.random.key(581), 20000)
    draws = jax.jit(jax.vmap(lambda key: process.draw_shock(cov=cov, key=key)))(keys)
    np.testing.assert_allclose(np.cov(np.asarray(draws).T), cov, atol=0.04, rtol=0.04)


@pytest.mark.parametrize("fold", [False, True])
@pytest.mark.parametrize("runtime", [False, True])
def test_joint_normal_native_values_and_named_simulation(
    *,
    joint_process_class: type[JointNormalIIDProcess],
    fold: bool,
    runtime: bool,
    tmp_path: Path,
) -> None:
    """Native values agree with scalar rotation; output and inputs name shocks."""
    cov = ((1.0, 0.4), (0.4, 2.0))
    process = joint_process_class(
        states=("x", "y"),
        n_points=(3, 5),
        gauss_hermite=True,
        mean=None if runtime else (0.0, 0.0),
        cov=None if runtime else cov,
        fold=fold,
    )
    model = _model(
        lcm.Regime(states={"wages": process}, functions={"utility": _utility})
    )
    manual = _model(
        lcm.Regime(
            states={
                "z_1": lcm.NormalIIDProcess(
                    n_points=3, gauss_hermite=True, mu=0.0, sigma=1.0, fold=fold
                ),
                "z_2": lcm.NormalIIDProcess(
                    n_points=5, gauss_hermite=True, mu=0.0, sigma=1.0, fold=fold
                ),
            },
            functions={"utility": _manual_utility},
        )
    )
    params = (
        {
            name: {"wages": {"mean": jnp.zeros(2), "cov": jnp.asarray(cov)}}
            for name in ("working", "done")
        }
        if runtime
        else {}
    )
    template = model.get_params_template()
    if runtime:
        assert set(template["working"]["wages"]) == {"mean", "cov"}
    solution = model.solve(params=params, log_level="off")
    reference = manual.solve(params={}, log_level="off")
    physical = {"x": jnp.array((0.0, 0.5)), "y": jnp.array((0.0, -0.2))}
    lookup = model.lookup_policy(
        params=params,
        solution=solution,
        period=0,
        regime_name="working",
        states=physical,
    )
    manual_lookup = manual.lookup_policy(
        params={},
        solution=reference,
        period=0,
        regime_name="working",
        states={
            "z_1": physical["x"],
            "z_2": (physical["y"] - 0.4 * physical["x"]) / jnp.sqrt(1.84),
        },
    )
    aaae(lookup.value, manual_lookup.value, decimal=DECIMAL_PRECISION)
    for period, regime_name in enumerate(("working", "done")):
        aaae(
            solution.value(period=period, regime=regime_name),
            reference.value(period=period, regime=regime_name),
            decimal=DECIMAL_PRECISION,
        )
    size = 10000
    initial = {
        "regime_id": jnp.zeros(size, dtype=jnp.int32),
        "period": jnp.zeros(size, dtype=jnp.int32),
    }
    initial.update(x=jnp.zeros(size), y=jnp.zeros(size))
    result = model.simulate(
        params=params,
        initial_conditions=initial,
        solution=solution,
        seed=521,
        log_level="off",
    )
    frame = result.to_dataframe()
    all_targets = result.to_dataframe(additional_targets="all")
    assert "wages" not in all_targets.columns
    pd.testing.assert_frame_equal(frame, all_targets[list(frame.columns)])
    assert {"x", "y"} <= set(frame.columns)
    assert {"x", "y"} <= set(result.state_names)
    assert not any("normal_" in name for name in frame.columns)
    aaae(
        frame.loc[frame["period"] == 0, ["x", "y"]],
        np.zeros((size, 2)),
        decimal=DECIMAL_PRECISION,
    )
    next_draws = frame.loc[frame["period"] == 1, ["x", "y"]].to_numpy()
    np.testing.assert_allclose(np.cov(next_draws.T), cov, atol=0.06, rtol=0.06)
    if not fold and not runtime:
        result.save(directory=tmp_path / "simulation")
        reloaded = lcm.SimulationResult.load(directory=tmp_path / "simulation")
        pd.testing.assert_frame_equal(frame, reloaded.to_dataframe())


def test_joint_normal_runtime_validation_names_process(
    *, joint_process_class: type[JointNormalIIDProcess]
) -> None:
    """An invalid estimated covariance reports the declared process name."""
    process = joint_process_class(
        states=("x", "y"), n_points=3, gauss_hermite=True, mean=(0.0, 0.0), cov=None
    )
    model = _model(
        lcm.Regime(states={"wages": process}, functions={"utility": _utility})
    )
    with pytest.raises(InvalidParamsError, match=r"wages.*cov"):
        model.solve(
            params={
                name: {"wages": {"cov": jnp.asarray(((1.0, 2.0), (2.0, 1.0)))}}
                for name in ("working", "done")
            },
            log_level="off",
            runtime_checks=False,
        )


def test_joint_normal_singular_native_roundtrip() -> None:
    """Rounded draws on a singular Gaussian support remain valid public inputs."""
    process = JointNormalIIDProcess(
        states=("x", "y"),
        n_points=3,
        gauss_hermite=True,
        mean=(0.7, -0.3),
        cov=((2.0, 2.0), (2.0, 2.0)),
    )
    model = _model(
        lcm.Regime(states={"wages": process}, functions={"utility": _utility})
    )
    solution = model.solve(params={}, log_level="off")
    shocks = jax.vmap(lambda z: process.transform(standard=z))(
        jnp.array(((0.25, -0.7), (-0.4, 0.6)))
    )
    states = {"x": shocks[:, 0], "y": shocks[:, 1]}
    lookup = model.lookup_policy(
        params={},
        solution=solution,
        period=0,
        regime_name="working",
        states=states,
    )
    aaae(
        lookup.value,
        shocks[:, 0] * shocks[:, 1] + shocks[:, 0] ** 2 + 0.9 * 4.28,
        decimal=DECIMAL_PRECISION,
    )
    initial = {
        **states,
        "period": jnp.zeros(2, dtype=jnp.int32),
        "regime_id": jnp.zeros(2, dtype=jnp.int32),
    }
    result = model.simulate(
        params={},
        solution=solution,
        initial_conditions=initial,
        seed=521,
        log_level="off",
    )
    frame = result.to_dataframe()
    aaae(frame.loc[frame["period"] == 0, ["x", "y"]], shocks, decimal=DECIMAL_PRECISION)
    with pytest.raises(InvalidInitialConditionsError, match="singular"):
        model.simulate(
            params={},
            solution=solution,
            initial_conditions={**initial, "y": shocks[:, 1] + 0.01},
            seed=521,
            log_level="off",
        )


def test_joint_normal_budgeted_runtime_lookup_and_simulation() -> None:
    """Runtime covariance and physical inputs use finite entry admission once."""
    process = JointNormalIIDProcess(
        states=("x", "y"),
        n_points=3,
        gauss_hermite=True,
        mean=(0.0, 0.0),
        cov=None,
    )
    model = _model(
        lcm.Regime(states={"wages": process}, functions={"utility": _utility}),
        execution_config=lcm.ExecutionConfig(
            device_memory_bytes=2**30, axis_widths={"subject": 1}
        ),
    )
    params = {
        name: {"wages": {"cov": np.array(((1.0, 0.4), (0.4, 2.0)))}}
        for name in ("working", "done")
    }
    solution = model.solve(params=params, log_level="off")
    states = {"x": jnp.array((0.1, 0.5)), "y": jnp.array((0.2, -0.4))}
    expected = states["x"] * states["y"] + states["x"] ** 2 + 0.9 * 1.4
    lookup = model.lookup_policy(
        params=params, solution=solution, period=0, regime_name="working", states=states
    )
    aaae(lookup.value, expected, decimal=DECIMAL_PRECISION)
    initial = {
        **states,
        "period": jnp.zeros(2, dtype=jnp.int32),
        "regime_id": jnp.zeros(2, dtype=jnp.int32),
    }
    result = model.simulate(
        params=params,
        initial_conditions=initial,
        solution=solution,
        seed=521,
        log_level="off",
    )
    frame = result.to_dataframe()
    aaae(frame.loc[frame["period"] == 0, "value"], expected, decimal=DECIMAL_PRECISION)
    aaae(
        frame.loc[frame["period"] == 0, ["x", "y"]],
        np.column_stack((states["x"], states["y"])),
        decimal=DECIMAL_PRECISION,
    )


@pytest.mark.parametrize("malformed", ["scalar", "short", "matrix", "regime_matrix"])
def test_joint_normal_rejects_malformed_physical_inputs(malformed: str) -> None:
    process = JointNormalIIDProcess(
        states=("x", "y"),
        n_points=3,
        gauss_hermite=True,
        mean=(0.0, 0.0),
        cov=((1.0, 0.0), (0.0, 1.0)),
    )
    model = _model(
        lcm.Regime(states={"wages": process}, functions={"utility": _utility})
    )
    solution = model.solve(params={}, log_level="off")
    states = {"x": np.zeros(2), "y": np.zeros(2)}
    if malformed != "regime_matrix":
        states["x"] = {
            "scalar": np.array(0.0),
            "short": np.zeros(1),
            "matrix": np.zeros((2, 1)),
        }[malformed]
        with pytest.raises(
            InvalidInitialConditionsError, match=r"one-dimensional|matching"
        ):
            model.lookup_policy(
                params={},
                solution=solution,
                period=0,
                regime_name="working",
                states={name: jnp.asarray(value) for name, value in states.items()},
            )
    initial = {
        **states,
        "period": np.zeros(2, dtype=np.int32),
        "regime_id": np.zeros(
            (2, 1) if malformed == "regime_matrix" else 2, dtype=np.int32
        ),
    }
    with pytest.raises(
        InvalidInitialConditionsError, match=r"one-dimensional|matching"
    ):
        model.simulate(
            params={},
            solution=solution,
            initial_conditions=initial,
            seed=521,
            log_level="off",
        )
