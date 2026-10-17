"""Process simulation uses either solve nodes or its declared phase law."""

from dataclasses import replace
from types import MappingProxyType

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.grids import Grid
from _lcm.processes.base import _ContinuousStochasticProcess
from _lcm.regime_building.next_state import _create_continuous_stochastic_next_func
from _lcm.typing import StochasticNextFunction
from _lcm.utils.logging import LogLevel
from lcm import (
    AgeGrid,
    DiscreteGrid,
    LinSpacedGrid,
    Model,
    NormalIIDProcess,
    Phased,
    Regime,
    StateConditioned,
    TauchenAR1Process,
    UniformIIDProcess,
    categorical,
    fixed_transition,
)
from lcm.exceptions import (
    GridInitializationError,
    ModelInitializationError,
    PyLCMError,
    RegimeInitializationError,
)
from lcm.typing import ContinuousState, FloatND, ScalarInt, UserParams
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


def _local_draw_utility(z: ContinuousState) -> FloatND:
    return z**2


def _terminal_utility(wealth: ContinuousState) -> FloatND:
    return wealth


def _wealth_from_draw(next_z: ContinuousState) -> FloatND:
    return next_z


def _drawn_z(next_z: ContinuousState) -> FloatND:
    return next_z


def _wealth_from_helper(drawn_z: FloatND) -> FloatND:
    return drawn_z


def _local_draw_model(
    *, process: UniformIIDProcess | Phased, via_helper: bool = False
) -> Model:
    """Build a model whose terminal edge consumes `next_z` but drops `z`."""
    functions = {"utility": _local_draw_utility}
    if via_helper:
        functions["drawn_z"] = _drawn_z
    return Model(
        regimes={
            "alive": Regime(
                states={"z": process},
                functions=functions,
                state_transitions={
                    "wealth": _wealth_from_helper if via_helper else _wealth_from_draw
                },
            ),
            "dead": Regime(
                states={"wealth": LinSpacedGrid(start=0.0, stop=12.0, n_points=13)},
                functions={"utility": _terminal_utility},
            ),
        },
        ages=AgeGrid(start=0, inclusive_stop=1, step="Y"),
        initial_nodes={0: "alive"},
        edges={"alive": {"dead": 0}},
        regime_id_class=RegimeId,
        fixed_params={"discount_factor": 0.9},
    )


def _local_draw_terminal_wealth(*, model: Model, params: UserParams) -> np.ndarray:
    frame = model.simulate(
        params=params,
        initial_conditions={
            "z": jnp.array([0.25, 0.5, 1.0, 1.75]),
            "age": jnp.zeros(4),
            "regime_id": jnp.full(4, RegimeId.alive, dtype=jnp.int32),
        },
        seed=537,
        log_level="off",
    ).to_dataframe(use_labels=False)
    wealth = frame.loc[frame["period"] == 1, "wealth"].to_numpy()
    assert wealth.shape == (4,)
    return wealth


def test_ordinary_local_draw_stays_on_the_process_support() -> None:
    """A draw consumed on an edge dropping the process lands on its nodes."""
    process = UniformIIDProcess(n_points=3, start=0.0, stop=2.0)
    got = _local_draw_terminal_wealth(
        model=_local_draw_model(process=process), params={}
    )
    assert np.isin(got, np.asarray(process.get_gridpoints())).all()


def test_phased_local_draw_solves_with_the_solve_law() -> None:
    """Value functions of a phased model equal those of its solve law alone."""
    solve = UniformIIDProcess(n_points=3, start=0.0, stop=2.0)
    simulate = UniformIIDProcess(n_points=3, start=10.0, stop=12.0)
    ordinary = _local_draw_model(process=solve).solve(params={}, log_level="off")
    phased = _local_draw_model(process=Phased(solve=solve, simulate=simulate)).solve(
        params={}, log_level="off"
    )
    assert all(
        np.array_equal(values, phased.values[period][name])
        for period, regimes in ordinary.values.items()
        for name, values in regimes.items()
    )


@pytest.mark.parametrize("via_helper", [False, True])
@pytest.mark.parametrize(
    ("solve_size", "simulate_size", "start", "stop"),
    [
        (3, 3, 10.0, 12.0),
        (3, 2, 10.0, 12.0),
        (3, 4, 10.0, 12.0),
        (2, 2, 4.0, 6.0),
        (4, 2, 4.0, 12.0),
    ],
)
def test_phased_local_draw_lands_on_the_simulation_support(
    *, solve_size: int, simulate_size: int, start: float, stop: float, via_helper: bool
) -> None:
    """A transition-local draw follows `Phased.simulate`, whatever its size."""
    solve = UniformIIDProcess(n_points=solve_size, start=0.0, stop=2.0)
    simulate = UniformIIDProcess(n_points=simulate_size, start=start, stop=stop)
    model = _local_draw_model(
        process=Phased(solve=solve, simulate=simulate), via_helper=via_helper
    )
    got = _local_draw_terminal_wealth(model=model, params={})
    assert np.isin(got, np.asarray(simulate.get_gridpoints())).all()


@pytest.mark.parametrize(("start", "stop"), [(10.0, 12.0), (4.0, 6.0)])
def test_local_draw_uses_simulation_only_runtime_support(
    *, start: float, stop: float
) -> None:
    """Runtime parameters of the simulation law set the local draw's support."""
    solve = UniformIIDProcess(n_points=2, start=0.0, stop=2.0)
    model = _local_draw_model(
        process=Phased(solve=solve, simulate=UniformIIDProcess(n_points=2))
    )
    got = _local_draw_terminal_wealth(
        model=model, params={"alive": {"z": {"start": start, "stop": stop}}}
    )
    assert np.isin(got, np.array([start, stop])).all()


def test_retained_model_local_draw_follows_new_runtime_parameters() -> None:
    """Re-simulating one model with shifted runtime bounds shifts every draw."""
    solve = UniformIIDProcess(n_points=2, start=0.0, stop=2.0)
    model = _local_draw_model(
        process=Phased(solve=solve, simulate=UniformIIDProcess(n_points=2))
    )
    first = _local_draw_terminal_wealth(
        model=model, params={"alive": {"z": {"start": 10.0, "stop": 12.0}}}
    )
    second = _local_draw_terminal_wealth(
        model=model, params={"alive": {"z": {"start": 4.0, "stop": 6.0}}}
    )
    np.testing.assert_array_equal(second, first - 6.0)


def test_simulation_only_runtime_parameters_leave_solve_values_unchanged() -> None:
    """Parameters bound only by `Phased.simulate` do not move the solution."""
    solve = UniformIIDProcess(n_points=2, start=0.0, stop=2.0)
    model = _local_draw_model(
        process=Phased(solve=solve, simulate=UniformIIDProcess(n_points=2))
    )
    first = model.solve(
        params={"alive": {"z": {"start": 10.0, "stop": 12.0}}}, log_level="off"
    )
    second = model.solve(
        params={"alive": {"z": {"start": 4.0, "stop": 6.0}}}, log_level="off"
    )
    assert all(
        np.array_equal(values, second.values[period][name])
        for period, regimes in first.values.items()
        for name, values in regimes.items()
    )


@categorical(ordered=False)
class _WorkRetiredId:
    work: ScalarInt
    retired: ScalarInt
    dead: ScalarInt


@categorical(ordered=False)
class _CodedHealth:
    bad: ScalarInt
    good: ScalarInt


@categorical(ordered=False)
class _RecodedHealth:
    good: ScalarInt
    bad: ScalarInt


def _tauchen(**kwargs: float | str | None) -> TauchenAR1Process:
    law = {"n_points": 3, "gauss_hermite": False, "n_std": 2.0, "mu": 0.0}
    return TauchenAR1Process(**(law | {"rho": 0.5, "sigma": 0.3} | kwargs))  # ty: ignore[invalid-argument-type]


def _work_retired_model(
    *,
    work: _ContinuousStochasticProcess | Phased,
    retired: _ContinuousStochasticProcess | Phased,
    extra_work_states: dict[str, DiscreteGrid] | None = None,
    extra_retired_states: dict[str, DiscreteGrid] | None = None,
) -> Model:
    def regime(states: dict[str, Grid | Phased]) -> Regime:
        return Regime(
            states=states,
            state_transitions={
                name: fixed_transition(name) for name in states if name != "income"
            },
            functions={"utility": _utility},
        )

    return Model(
        regimes={
            "work": regime({"income": work} | (extra_work_states or {})),
            "retired": regime({"income": retired} | (extra_retired_states or {})),
            "dead": Regime(functions={"utility": lambda: jnp.asarray(0.0)}),
        },
        ages=AgeGrid(start=0, inclusive_stop=3, step="Y"),
        initial_nodes={0: "work"},
        edges={"work": {"work": 0, "retired": 1}, "retired": {"dead": 2}},
        regime_id_class=_WorkRetiredId,
        fixed_params={"discount_factor": 0.9},
    )


@pytest.mark.parametrize(
    ("work", "retired"),
    [
        (
            Phased(solve=_tauchen(rho=None), simulate=_tauchen(rho=0.9)),
            _tauchen(rho=None),
        ),
        (_tauchen(rho=None, simulate_draws="nodes"), _tauchen(rho=None)),
    ],
    ids=["phased-source", "draw-mode-only"],
)
def test_process_carried_under_different_declarations_solves_like_fixed_law(
    *,
    work: _ContinuousStochasticProcess | Phased,
    retired: _ContinuousStochasticProcess,
) -> None:
    """A process both regimes declare is carried, whatever the declarations differ in.

    With its runtime parameter supplied, the value functions equal those of the
    model whose declarations fix the same parameter at construction.
    """
    runtime = _work_retired_model(work=work, retired=retired).solve(
        params={"work": {"income": {"rho": 0.5}}, "retired": {"income": {"rho": 0.5}}},
        log_level="off",
    )
    fixed = _work_retired_model(work=_tauchen(), retired=_tauchen()).solve(
        params={}, log_level="off"
    )
    assert all(
        np.array_equal(values, runtime.values[period][name])
        for period, regimes in fixed.values.items()
        for name, values in regimes.items()
    )


def test_simulation_conditioning_codes_must_agree_across_regimes() -> None:
    """A simulation law conditioned on a state coded per regime is refused."""
    conditioned = NormalIIDProcess(
        n_points=3,
        gauss_hermite=False,
        n_std=2.0,
        mu=0.0,
        sigma=StateConditioned(on="health", by={"bad": 1.0, "good": 0.1}),
    )
    income = Phased(
        solve=NormalIIDProcess(
            n_points=3, gauss_hermite=False, n_std=2.0, mu=0.0, sigma=0.5
        ),
        simulate=conditioned,
    )
    with pytest.raises(ModelInitializationError, match="same integer codes"):
        _work_retired_model(
            work=income,
            retired=income,
            extra_work_states={"health": DiscreteGrid(_CodedHealth)},
            extra_retired_states={"health": DiscreteGrid(_RecodedHealth)},
        )


def test_node_draw_from_beyond_the_top_node_follows_the_top_node_row() -> None:
    """An off-grid current value draws as from the nearest edge node."""
    process = _tauchen(rho=0.9, simulate_draws="nodes")
    model = _model(process)
    solution = model.solve(params={}, log_level="off")
    top = float(process.get_gridpoints()[-1])

    def realized(start: float) -> np.ndarray:
        frame = model.simulate(
            params={},
            solution=solution,
            initial_conditions={
                "income": jnp.full(256, start),
                "age": jnp.zeros(256),
                "regime_id": jnp.full(256, RegimeId.alive, dtype=jnp.int32),
            },
            seed=537,
            log_level="off",
        ).to_dataframe(use_labels=False)
        return frame.loc[frame["period"] == 1, "income"].to_numpy()

    np.testing.assert_array_equal(realized(top + 5.0), realized(top))


@pytest.mark.parametrize(
    "simulate_change",
    [{"simulate_draws": "nodes"}, {"fold": True}],
    ids=["draw-mode", "fold"],
)
def test_phased_process_with_one_law_is_accepted(
    simulate_change: dict[str, str | bool],
) -> None:
    """Phase declarations sharing one law are not distinct laws."""
    base = NormalIIDProcess(
        n_points=3,
        gauss_hermite=True,
        mu=0.0,
        sigma=1.0,
        fold="fold" in simulate_change,
    )
    process = Phased(solve=base, simulate=replace(base, **simulate_change))
    assert isinstance(
        Regime(states={"income": process}, functions={"utility": _utility}), Regime
    )
