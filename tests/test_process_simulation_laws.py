"""Process simulation uses either solve nodes or its declared phase law."""

from dataclasses import replace
from types import MappingProxyType

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.processes.base import _ContinuousStochasticProcess
from _lcm.regime_building.next_state import _create_continuous_stochastic_next_func
from _lcm.typing import StochasticNextFunction
from _lcm.utils.logging import LogLevel
from lcm import (
    AgeGrid,
    DiscreteGrid,
    Model,
    NormalIIDProcess,
    Phased,
    Regime,
    StateConditioned,
    TauchenAR1Process,
    fixed_transition,
)
from lcm.exceptions import (
    GridInitializationError,
    PyLCMError,
    RegimeInitializationError,
)
from lcm.typing import ContinuousState, FloatND
from tests.test_models.processes import Health, RegimeId


def _utility(income: ContinuousState) -> FloatND:
    return income**2


def _model(process: _ContinuousStochasticProcess | Phased) -> Model:
    return Model(
        regimes={
            "alive": Regime(
                states={"income": process}, functions={"utility": _utility}
            ),
            "dead": Regime(functions={"utility": lambda: jnp.asarray(0.0)}),
        },
        ages=AgeGrid(start=0, inclusive_stop=2, step="Y"),
        initial_nodes={0: "alive"},
        edges={"alive": {"alive": 0, "dead": 1}},
        regime_id_class=RegimeId,
        fixed_params={"discount_factor": 0.9},
    )


def _draw(process: _ContinuousStochasticProcess) -> StochasticNextFunction:
    return _create_continuous_stochastic_next_func(
        target_regime_name="alive",
        next_state_name="next_income",
        all_grids=MappingProxyType({"alive": MappingProxyType({"income": process})}),
    )


def test_iid_nodes_draw_exact_weighted_support() -> None:
    """The finite lottery uses the quadrature weights, including rare tail nodes."""
    process = NormalIIDProcess(
        n_points=3, gauss_hermite=True, mu=0.0, sigma=1.0, simulate_draws="nodes"
    )
    draw = _draw(process)
    keys = jax.random.split(jax.random.key(193), 20000)
    nodes = process.get_gridpoints()
    weights = process.get_transition_probs()[0]
    got = jax.jit(
        jax.vmap(
            lambda key: draw(
                key_alive__next_income=key,
                weight_alive__next_income=weights,
                nodes_alive__next_income=nodes,
            )
        )
    )(keys)
    assert np.isin(np.asarray(got), np.asarray(nodes)).all()
    frequencies = jnp.mean(got[:, None] == nodes[None, :], axis=0)
    np.testing.assert_allclose(frequencies, weights, atol=0.015, rtol=0.0)
    assert not np.isin(
        np.asarray(
            _draw(replace(process, simulate_draws="continuous"))(
                key_alive__next_income=keys[0]
            )
        ),
        np.asarray(nodes),
    ).all()


def test_nodes_is_not_a_distribution_parameter() -> None:
    process = NormalIIDProcess(n_points=3, gauss_hermite=True, simulate_draws="nodes")
    assert set(process.params_to_pass_at_runtime) == {"mu", "sigma"}


def test_invalid_simulation_draw_mode_is_rejected() -> None:
    with pytest.raises(GridInitializationError):
        NormalIIDProcess(n_points=3, gauss_hermite=True, simulate_draws="nearest")  # ty: ignore[invalid-argument-type]


def test_phased_ar1_keeps_solve_law_and_uses_distinct_forward_law() -> None:
    """A different forward persistence and variance leave perceived values intact."""
    solve = TauchenAR1Process(
        n_points=3, gauss_hermite=False, n_std=2.0, mu=0.0, rho=0.4, sigma=0.3
    )
    simulate = replace(solve, rho=0.8, sigma=0.7)
    ordinary = _model(solve)
    phased = _model(Phased(solve=solve, simulate=simulate))
    ordinary_solution = ordinary.solve(params={}, log_level="off")
    phased_solution = phased.solve(params={}, log_level="off")
    for period, regimes in ordinary_solution.values.items():
        for name, values in regimes.items():
            np.testing.assert_array_equal(values, phased_solution.values[period][name])
    initial = {
        "income": jnp.array([0.2, -0.3]),
        "age": jnp.zeros(2),
        "regime_id": jnp.full(2, RegimeId.alive, dtype=jnp.int32),
    }
    got = phased.simulate(
        params={},
        solution=phased_solution,
        initial_conditions=initial,
        seed=537,
        log_level="off",
    ).to_dataframe(use_labels=False)
    reference = (
        _model(simulate)
        .simulate(
            params={},
            initial_conditions=initial,
            seed=537,
            log_level="off",
        )
        .to_dataframe(use_labels=False)
    )
    np.testing.assert_array_equal(got["income"], reference["income"])
    assert not np.array_equal(
        got.loc[got["period"] == 1, "income"].to_numpy(),
        ordinary.simulate(
            params={}, initial_conditions=initial, seed=537, log_level="off"
        )
        .to_dataframe(use_labels=False)
        .loc[lambda frame: frame["period"] == 1, "income"]
        .to_numpy(),
    )


@pytest.mark.parametrize("fold", [False, True])
def test_native_iid_node_simulation_and_repeat_seed(*, fold: bool) -> None:
    process = NormalIIDProcess(
        n_points=3,
        gauss_hermite=True,
        mu=0.0,
        sigma=0.7,
        simulate_draws="nodes",
        fold=fold,
    )
    model = _model(process)
    initial = {
        "income": jnp.zeros(64),
        "age": jnp.zeros(64),
        "regime_id": jnp.full(64, RegimeId.alive, dtype=jnp.int32),
    }
    solution = model.solve(params={}, log_level="off")
    frames = [
        model.simulate(
            params={},
            solution=solution,
            initial_conditions=initial,
            seed=537,
            log_level="off",
        ).to_dataframe(use_labels=False)
        for _ in range(2)
    ]
    np.testing.assert_array_equal(frames[0]["income"], frames[1]["income"])
    realized = frames[0].loc[frames[0]["period"] == 1, "income"].to_numpy()
    assert np.isin(realized, np.asarray(process.get_gridpoints())).all()


def test_runtime_supplied_node_markov_rows_and_support() -> None:
    nodes = (-1.3, 0.1, 2.4)
    process = TauchenAR1Process.on_nodes(nodes=nodes, simulate_draws="nodes")
    model = _model(process)
    params = {"alive": {"income": {"rho": 0.6, "sigma": 0.4}}}
    initial = {
        "income": jnp.full(2000, nodes[1]),
        "age": jnp.zeros(2000),
        "regime_id": jnp.full(2000, RegimeId.alive, dtype=jnp.int32),
    }
    frame = model.simulate(
        params=params, initial_conditions=initial, seed=537, log_level="off"
    ).to_dataframe(use_labels=False)
    realized = frame.loc[frame["period"] == 1, "income"].to_numpy()
    support = np.asarray(process.get_gridpoints())
    assert np.isin(realized, support).all()
    expected = np.asarray(
        process.compute_transition_probs(
            rho=jnp.asarray(0.6), sigma=jnp.asarray(0.4), mu=jnp.asarray(0.0)
        )
    )[1]
    observed = (realized[:, None] == support).mean(axis=0)
    np.testing.assert_allclose(observed, expected, rtol=0.0, atol=0.04)


@pytest.mark.parametrize("invalid", ["fold", "nodes", "nonprocess"])
def test_unsupported_phased_process_forms_fail_at_declaration(invalid: str) -> None:
    process = NormalIIDProcess(n_points=3, gauss_hermite=True, mu=0.0, sigma=1.0)
    other = (
        replace(process, fold=True)
        if invalid == "fold"
        else replace(process, mu=1.0, simulate_draws="nodes")
        if invalid == "nodes"
        else lambda: 0.0
    )
    with pytest.raises(RegimeInitializationError):
        Regime(
            states={"income": Phased(solve=process, simulate=other)},
            functions={"utility": _utility},
        )


def test_zero_probability_nodes_are_never_drawn() -> None:
    process = NormalIIDProcess(
        n_points=3, gauss_hermite=True, mu=0.0, sigma=1.0, simulate_draws="nodes"
    )
    nodes = process.get_gridpoints()
    draw = _draw(process)
    keys = jax.random.split(jax.random.key(537), 64)
    realized = jax.jit(
        jax.vmap(
            lambda key: draw(
                key_alive__next_income=key,
                weight_alive__next_income=jnp.array([0.0, 1.0, 0.0]),
                nodes_alive__next_income=nodes,
            )
        )
    )(keys)
    np.testing.assert_array_equal(realized, jnp.full(64, nodes[1]))


def test_phased_process_conditioning_is_an_intrinsic_state_dependency() -> None:
    def process(scale: float) -> NormalIIDProcess:
        return NormalIIDProcess(
            n_points=3,
            gauss_hermite=False,
            n_std=2.0,
            mu=0.0,
            sigma=StateConditioned(on="health", by={"bad": scale, "good": 2 * scale}),
        )

    model = Model(
        regimes={
            "alive": Regime(
                states={
                    "income": Phased(solve=process(0.1), simulate=process(0.3)),
                    "health": DiscreteGrid(Health),
                },
                state_transitions={"health": fixed_transition("health")},
                functions={"utility": _utility},
            ),
            "dead": Regime(functions={"utility": lambda: jnp.asarray(0.0)}),
        },
        ages=AgeGrid(start=0, inclusive_stop=2, step="Y"),
        initial_nodes={0: "alive"},
        edges={"alive": {"alive": 0, "dead": 1}},
        regime_id_class=RegimeId,
        fixed_params={"discount_factor": 0.9},
    )
    frame = model.simulate(
        params={},
        initial_conditions={
            "income": jnp.zeros(2),
            "health": jnp.array([Health.bad, Health.good]),
            "age": jnp.zeros(2),
            "regime_id": jnp.full(2, RegimeId.alive, dtype=jnp.int32),
        },
        seed=537,
        log_level="off",
    ).to_dataframe(use_labels=False)
    assert np.isfinite(frame.loc[frame["period"] == 1, "income"]).all()


def test_node_operand_names_do_not_unmap_similarly_named_states() -> None:
    process = TauchenAR1Process(
        n_points=3, gauss_hermite=False, n_std=2.0, mu=0.0, rho=0.4, sigma=0.3
    )
    model = Model(
        regimes={
            "alive": Regime(
                states={"nodes_income": process},
                functions={"utility": lambda nodes_income: nodes_income**2},
            ),
            "dead": Regime(functions={"utility": lambda: jnp.asarray(0.0)}),
        },
        ages=AgeGrid(start=0, inclusive_stop=2, step="Y"),
        initial_nodes={0: "alive"},
        edges={"alive": {"alive": 0, "dead": 1}},
        regime_id_class=RegimeId,
        fixed_params={"discount_factor": 0.9},
    )
    frame = model.simulate(
        params={},
        initial_conditions={
            "nodes_income": jnp.array([0.2, -0.3]),
            "age": jnp.zeros(2),
            "regime_id": jnp.full(2, RegimeId.alive, dtype=jnp.int32),
        },
        seed=537,
        log_level="off",
    ).to_dataframe(use_labels=False)
    assert frame.loc[frame["period"] == 1, "nodes_income"].shape == (2,)


@pytest.mark.parametrize("runtime_checks", [True, False])
@pytest.mark.parametrize("log_level", ["off", "debug"])
def test_node_draws_do_not_depend_on_runtime_checks_or_logging(
    *, runtime_checks: bool, log_level: LogLevel
) -> None:
    process = NormalIIDProcess(
        n_points=3, gauss_hermite=True, mu=0.0, sigma=1.0, simulate_draws="nodes"
    )
    model = _model(process)
    initial = {
        "income": jnp.zeros(3),
        "age": jnp.zeros(3),
        "regime_id": jnp.full(3, RegimeId.alive, dtype=jnp.int32),
    }
    solution = model.solve(params={}, log_level="off")
    reference = model.simulate(
        params={},
        solution=solution,
        initial_conditions=initial,
        seed=537,
        log_level="off",
    ).to_dataframe(use_labels=False)
    got = model.simulate(
        params={},
        solution=solution,
        initial_conditions=initial,
        seed=537,
        log_level=log_level,
        runtime_checks=runtime_checks,
    ).to_dataframe(use_labels=False)
    np.testing.assert_array_equal(got["income"], reference["income"])


def test_node_kernel_accepts_empty_subject_keys() -> None:
    process = NormalIIDProcess(
        n_points=3, gauss_hermite=True, mu=0.0, sigma=1.0, simulate_draws="nodes"
    )
    draw = _draw(process)
    got = jax.jit(
        jax.vmap(
            lambda key: draw(
                key_alive__next_income=key,
                weight_alive__next_income=process.get_transition_probs()[0],
                nodes_alive__next_income=process.get_gridpoints(),
            )
        )
    )(jax.random.split(jax.random.key(537), 0))
    assert got.shape == (0,)


def test_nonfinite_simulation_law_fails_with_output_disabled() -> None:
    solve = NormalIIDProcess(n_points=3, gauss_hermite=True, mu=0.0, sigma=1.0)
    model = _model(Phased(solve=solve, simulate=replace(solve, sigma=float("nan"))))
    initial = {
        "income": jnp.zeros(2),
        "age": jnp.zeros(2),
        "regime_id": jnp.full(2, RegimeId.alive, dtype=jnp.int32),
    }
    with pytest.raises(PyLCMError):
        model.simulate(
            params={},
            initial_conditions=initial,
            seed=537,
            log_level="off",
            runtime_checks=True,
        )
