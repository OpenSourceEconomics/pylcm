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


_FULL_RANK_COV = ((1.0, -0.4), (-0.4, 2.0))


def test_joint_normal_process_is_public() -> None:
    """A regime can declare a named joint normal IID process."""
    assert hasattr(lcm, "JointNormalIIDProcess")


@pytest.mark.parametrize("runtime", [False, True])
@pytest.mark.parametrize("fold", [False, True])
@pytest.mark.parametrize(
    "cov",
    [_FULL_RANK_COV, ((0.0, 0.0), (0.0, 2.0)), ((1.0, 2.0), (2.0, 4.0))],
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
    if runtime and cov != _FULL_RANK_COV:
        # Only a fixed covariance may be singular.
        with pytest.raises(GridInitializationError, match="fixed cov"):
            model.initial_state_bounds(
                params=params, regime_name="working", state_name="x", period=0
            )
        return
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


@pytest.mark.parametrize("runtime", [False, True])
@pytest.mark.parametrize("fold", [False, True])
@pytest.mark.parametrize(
    "cov",
    [_FULL_RANK_COV, ((0.0, 0.0), (0.0, 2.0)), ((1.0, 2.0), (2.0, 4.0))],
)
def test_joint_normal_validate_transitions_preserves_named_runtime_law(
    *, runtime: bool, fold: bool, cov: tuple[tuple[float, float], tuple[float, float]]
) -> None:
    """Public preflight shares fixed/runtime joint-law validation with native solve."""
    process = JointNormalIIDProcess(
        states=("x", "y"),
        n_points=(3, 5),
        gauss_hermite=True,
        mean=None if runtime else (1.0, -2.0),
        cov=None if runtime else cov,
        fold=fold,
    )
    model = _model(
        lcm.Regime(states={"wages": process}, functions={"utility": _utility})
    )
    params = (
        {
            name: {"wages": {"mean": jnp.array([1.0, -2.0]), "cov": jnp.array(cov)}}
            for name in ("working", "done")
        }
        if runtime
        else {}
    )
    if runtime and cov != _FULL_RANK_COV:
        # Only a fixed covariance may be singular.
        with pytest.raises(GridInitializationError, match="fixed cov"):
            model.validate_transitions(params=params)
        return
    assert model.validate_transitions(params=params) is None
    solution = model.solve(params=params, log_level="off", runtime_checks=True)
    for regime_to_values in solution.values.values():
        for values in regime_to_values.values():
            assert np.isfinite(np.asarray(values)).all()

    if runtime:
        template = model.get_params_template()
        for name in ("working", "done"):
            assert set(template[name]["wages"]) == {"mean", "cov"}
        # Leave the first regime valid; fail the later regime's named runtime law.
        # Fresh mapping leaves the valid input untouched for the subsequent call.
        invalid = {
            **params,
            "done": {
                "wages": {
                    "mean": jnp.array([1.0, -2.0]),
                    "cov": jnp.array(((1.0, 2.0), (2.0, 1.0))),
                }
            },
        }
        with pytest.raises(InvalidParamsError, match=r"wages.*cov") as preflight:
            model.validate_transitions(params=invalid)
        with pytest.raises(InvalidParamsError, match=r"wages.*cov") as solve:
            model.solve(params=invalid, log_level="off", runtime_checks=True)
        assert str(preflight.value) == str(solve.value)
        assert model.validate_transitions(params=params) is None


@lcm.time_varying_params("wage")
def _age_profile_joint_utility(
    *,
    x: lcm.typing.ContinuousState,
    y: lcm.typing.ContinuousState,
    wage: lcm.typing.FloatND,
) -> lcm.typing.FloatND:
    return x * y + x * x + wage


@pytest.mark.parametrize("fold", [False, True])
def test_joint_normal_age_profile_alignment_matches_manual(*, fold: bool) -> None:
    process = JointNormalIIDProcess(
        states=("x", "y"),
        n_points=3,
        gauss_hermite=True,
        mean=(0.0, 0.0),
        cov=None,
        fold=fold,
    )
    regime = lcm.Regime(
        states={"joint": process}, functions={"utility": _age_profile_joint_utility}
    )
    model = lcm.Model(
        ages=lcm.AgeGrid(start=40, inclusive_stop=41, step="Y"),
        regimes={"working": regime, "done": regime},
        regime_id_class=RegimeId,
        initial_nodes=((40, "working"),),
        edges={"working": {"done": (40,)}},
        fixed_params={"discount_factor": 0.9},
        execution_config=lcm.ExecutionConfig(device_memory_bytes=None),
    )
    mapped = lcm.TimeVarying.from_profile(
        values=jnp.array([99.0, 5.0, 3.0]),
        labels=(39, 42, 41),
        age_to_label={40: 41, 41: 42},
    )
    manual = lcm.TimeVarying(values=jnp.array([3.0, 5.0]), ages=(40, 41))
    solutions = {}
    frames = {}
    for name, profile in (("mapped", mapped), ("manual", manual)):
        params = {
            "wage": profile,
            **{
                regime_name: {"joint": {"cov": np.array(((1.0, 0.4), (0.4, 2.0)))}}
                for regime_name in ("working", "done")
            },
        }
        model.validate_transitions(params=params)
        solution = model.solve(params=params, runtime_checks=True, log_level="off")
        solutions[name] = solution
        states = {"x": jnp.array((0.1, 0.5)), "y": jnp.array((0.2, -0.4))}
        expected = states["x"] * states["y"] + states["x"] ** 2 + 3.0 + 0.9 * 6.4
        lookup = model.lookup_policy(
            params=params,
            solution=solution,
            regime_name="working",
            period=0,
            states=states,
        )
        aaae(lookup.value, expected, decimal=DECIMAL_PRECISION)
        physical = np.asarray(
            process.compute_gridpoints(cov=jnp.array(((1.0, 0.4), (0.4, 2.0))))
        )
        for dimension, state_name in enumerate(("x", "y")):
            bounds = model.initial_state_bounds(
                params=params,
                regime_name="working",
                state_name=state_name,
                period=0,
            )
            np.testing.assert_allclose(
                bounds,
                (physical[:, dimension].min(), physical[:, dimension].max()),
                rtol=10**-DECIMAL_PRECISION,
            )
        frames[name] = model.simulate(
            params=params,
            solution=solution,
            initial_conditions={
                **states,
                "age": jnp.full(2, 40, dtype=jnp.int32),
                "regime_id": jnp.zeros(2, dtype=jnp.int32),
            },
            seed=521,
            runtime_checks=True,
            log_level="off",
        ).to_dataframe()
    for period, regime_name in ((0, "working"), (1, "done")):
        np.testing.assert_array_equal(
            solutions["mapped"].values[period][regime_name],
            solutions["manual"].values[period][regime_name],
        )
    pd.testing.assert_frame_equal(frames["mapped"], frames["manual"], check_exact=True)


def test_joint_normal_budgeted_runtime_typed_seed_preserves_physical_frame() -> None:
    """Typed roots use the native budgeted joint law without exposing latent states."""
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
    solution = model.solve(params=params, log_level="off", runtime_checks=True)
    states = {"x": jnp.array((0.1, 0.5)), "y": jnp.array((0.2, -0.4))}
    expected = states["x"] * states["y"] + states["x"] ** 2 + 0.9 * 1.4
    lookup = model.lookup_policy(
        params=params, solution=solution, period=0, regime_name="working", states=states
    )
    aaae(lookup.value, expected, decimal=DECIMAL_PRECISION)
    parts = model.lookup_policy(
        params=params,
        solution=solution,
        period=0,
        regime_name="working",
        states=states,
        keep_intermediates=("Q", "F", "U", "CE"),
    )
    assert parts.U is not None
    assert parts.CE is not None
    assert parts.Q is not None
    assert parts.F is not None
    aaae(
        parts.U, states["x"] * states["y"] + states["x"] ** 2, decimal=DECIMAL_PRECISION
    )
    aaae(parts.CE, jnp.full_like(expected, 1.4), decimal=DECIMAL_PRECISION)
    aaae(parts.Q, expected, decimal=DECIMAL_PRECISION)
    aaae(parts.value, lookup.value, decimal=DECIMAL_PRECISION)
    np.testing.assert_array_equal(parts.F, jnp.ones_like(expected, dtype=bool))
    initial = {
        **states,
        "period": jnp.zeros(2, dtype=jnp.int32),
        "regime_id": jnp.zeros(2, dtype=jnp.int32),
    }
    # Keys are created in the test body, after the launcher/worker pre-JAX cache.
    # Control integer implementation explicitly; don't inherit an unrelated RBG default.
    with jax.default_prng_impl("threefry2x32"):
        threefry = jax.random.key(521, impl="threefry2x32")
        integer_result = model.simulate(
            params=params,
            initial_conditions=initial,
            solution=solution,
            seed=521,
            log_level="off",
            runtime_checks=True,
        )
        threefry_result = model.simulate(
            params=params,
            initial_conditions=initial,
            solution=solution,
            seed=threefry,
            log_level="off",
            runtime_checks=True,
        )
    integer_frame = integer_result.to_dataframe()
    threefry_frame = threefry_result.to_dataframe()
    pd.testing.assert_frame_equal(integer_frame, threefry_frame, check_exact=True)
    for result in (integer_result, threefry_result):
        frame = result.to_dataframe()
        assert {"x", "y"} <= set(frame.columns)
        assert not any("normal_" in name for name in frame.columns)
        assert np.isfinite(frame[["x", "y", "value"]].to_numpy()).all()
        aaae(
            frame.loc[frame["period"] == 0, "value"],
            expected,
            decimal=DECIMAL_PRECISION,
        )
        aaae(
            frame.loc[frame["period"] == 0, ["x", "y"]],
            np.column_stack((states["x"], states["y"])),
            decimal=DECIMAL_PRECISION,
        )
        assert (frame["period"] == 1).any()


def _joint_carried_terminal_utility(
    *,
    x: lcm.typing.ContinuousState,
    y: lcm.typing.ContinuousState,
    lag: lcm.typing.ContinuousState,
) -> lcm.typing.FloatND:
    return x * y + x * x + lag


@pytest.mark.parametrize("fold", [False, True])
def test_joint_normal_keeps_an_ordinary_state_carried_to_reader(*, fold: bool) -> None:
    process = JointNormalIIDProcess(
        states=("x", "y"),
        n_points=3,
        gauss_hermite=True,
        mean=(0.0, 0.0),
        cov=None,
        fold=fold,
    )
    lag_grid = lcm.LinSpacedGrid(start=0.0, stop=2.0, n_points=3)
    model = lcm.Model(
        regimes={
            "working": lcm.Regime(
                states={"joint": process, "lag": lag_grid},
                state_transitions={"lag": lcm.fixed_transition("lag")},
                functions={"utility": _utility},
            ),
            "done": lcm.Regime(
                states={"joint": process, "lag": lag_grid},
                functions={"utility": _joint_carried_terminal_utility},
            ),
        },
        n_periods=2,
        regime_id_class=RegimeId,
        initial_nodes=lcm.InitialNodes(by_period={0: "working"}),
        edges={"working": {"done": lcm.Periods(values=(0,))}},
        fixed_params={"discount_factor": 0.9},
        execution_config=lcm.ExecutionConfig(device_memory_bytes=None),
    )
    params = {
        name: {"joint": {"cov": np.array(((1.0, 0.4), (0.4, 2.0)))}}
        for name in ("working", "done")
    }
    assert set(model.initial_state_names(regime_name="working")) == {"x", "y", "lag"}
    assert model.initial_state_bounds(
        params=params, regime_name="working", state_name="lag", period=0
    ) == (0.0, 2.0)
    model.validate_transitions(params=params)
    solution = model.solve(params=params, runtime_checks=True, log_level="off")
    states = {
        "x": jnp.array((0.1, 0.5)),
        "y": jnp.array((0.2, -0.4)),
        "lag": jnp.array((0.5, 1.5)),
    }
    expected = (
        states["x"] * states["y"] + states["x"] ** 2 + 0.9 * (1.4 + states["lag"])
    )
    lookup = model.lookup_policy(
        params=params,
        solution=solution,
        regime_name="working",
        period=0,
        states=states,
    )
    aaae(lookup.value, expected, decimal=DECIMAL_PRECISION)
    frame = (
        model.simulate(
            params=params,
            solution=solution,
            initial_conditions={
                **states,
                "period": jnp.zeros(2, dtype=jnp.int32),
                "regime_id": jnp.zeros(2, dtype=jnp.int32),
            },
            seed=521,
            runtime_checks=True,
            log_level="off",
        )
        .to_dataframe(use_labels=False)
        .sort_values(["subject_id", "period"])
    )
    aaae(frame.loc[frame["period"] == 0, "value"], expected, decimal=DECIMAL_PRECISION)
    for period in (0, 1):
        aaae(
            frame.loc[frame["period"] == period, "lag"],
            states["lag"],
            decimal=DECIMAL_PRECISION,
        )
    assert {"x", "y", "lag"} <= set(frame.columns)
    assert np.isfinite(frame[["x", "y", "lag", "value"]].to_numpy()).all()


def _declared_regime() -> lcm.Regime:
    return lcm.Regime(
        states={
            "wages": JointNormalIIDProcess(
                states=("x", "y"),
                n_points=3,
                gauss_hermite=True,
                mean=(0.0, 0.0),
                cov=_FULL_RANK_COV,
            )
        },
        functions={"utility": _utility},
    )


def test_regime_keeps_the_joint_declaration_as_written() -> None:
    """`states` and `functions` hold what the author declared, not the lowering."""
    regime = _declared_regime()
    assert (set(regime.states), set(regime.functions)) == ({"wages"}, {"utility"})


def test_regime_replace_swaps_the_joint_declaration() -> None:
    """Replacing the joint process changes the law the model builds from."""
    replacement = JointNormalIIDProcess(
        states=("x", "y"),
        n_points=3,
        gauss_hermite=True,
        mean=(0.0, 0.0),
        cov=((2.0, 0.5), (0.5, 1.0)),
    )
    regime = _declared_regime().replace(states={"wages": replacement})
    assert regime.states["wages"] is replacement


def test_regime_replace_of_functions_keeps_the_joint_outputs() -> None:
    """Replacing `utility` leaves the physical shocks available to the model."""
    regime = _declared_regime().replace(functions={"utility": _utility})
    model = _model(regime)
    assert model.initial_state_names(regime_name="working") == ("x", "y")
