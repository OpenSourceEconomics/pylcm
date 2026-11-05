"""Process simulation uses either solve nodes or its declared phase law."""

import functools
import itertools
import math
from collections.abc import Callable
from dataclasses import replace
from types import MappingProxyType
from typing import Literal, cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpy.typing import NDArray

from _lcm.grids import Grid
from _lcm.processes.ar1 import _TauchenOnNodesProcess
from _lcm.processes.base import _ContinuousStochasticProcess
from _lcm.regime_building import Q_and_F as q_and_f
from _lcm.regime_building.next_state import _create_continuous_stochastic_next_func
from _lcm.regime_building.processing import _get_weights_func_for_process
from _lcm.typing import (
    ArrayTree,
    EconFunctionsMapping,
    RegimeName,
    StochasticNextFunction,
)
from _lcm.utils.logging import LogLevel
from lcm import (
    AgeGrid,
    CertaintyEquivalent,
    DiscreteGrid,
    LinearExpectation,
    LinSpacedGrid,
    Model,
    NormalIIDProcess,
    Phased,
    PowerMean,
    Regime,
    RouwenhorstAR1Process,
    StateConditioned,
    StochasticTransition,
    TauchenAR1Process,
    TauchenNormalMixtureAR1Process,
    Transition,
    UniformIIDProcess,
    categorical,
    fixed_transition,
)
from lcm.exceptions import (
    GridInitializationError,
    ModelInitializationError,
    PyLCMError,
    RegimeInitializationError,
    ScaledLotteryDifferentiationError,
)
from lcm.typing import ContinuousState, FloatND, ScalarInt, UserParams
from tests.conftest import DECIMAL_PRECISION
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
        n_points=3, gauss_hermite=True, mu=0.0, sigma=1.0, simulated_draws="nodes"
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
            _draw(replace(process, simulated_draws="continuous"))(
                key_alive__next_income=keys[0]
            )
        ),
        np.asarray(nodes),
    ).all()


def test_nodes_is_not_a_distribution_parameter() -> None:
    process = NormalIIDProcess(n_points=3, gauss_hermite=True, simulated_draws="nodes")
    assert set(process.params_to_pass_at_runtime) == {"mu", "sigma"}


def test_invalid_simulation_draw_mode_is_rejected() -> None:
    with pytest.raises(GridInitializationError):
        NormalIIDProcess(n_points=3, gauss_hermite=True, simulated_draws="nearest")  # ty: ignore[invalid-argument-type]


@pytest.mark.parametrize(
    "build",
    [
        lambda: NormalIIDProcess(
            n_points=3,
            gauss_hermite=True,
            simulate_draws="nodes",  # ty: ignore[unknown-argument]
        ),
        lambda: TauchenAR1Process(
            n_points=3,
            gauss_hermite=False,
            n_std=2.0,
            simulate_draws="nodes",  # ty: ignore[unknown-argument]
        ),
        lambda: TauchenAR1Process.on_nodes(
            nodes=jnp.array([-1.0, 0.0, 1.0]),
            simulate_draws="nodes",
        ),
    ],
    ids=["iid", "ar1", "ar1-on-nodes"],
)
def test_renamed_draw_mode_keyword_names_its_replacement(
    build: Callable[[], _ContinuousStochasticProcess],
) -> None:
    with pytest.raises(TypeError, match=r"simulate_draws=.*renamed.*simulated_draws="):
        build()


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
    ).to_dataframe(use_categorical_codes=True)
    reference = (
        _model(simulate)
        .simulate(
            params={},
            initial_conditions=initial,
            seed=537,
            log_level="off",
        )
        .to_dataframe(use_categorical_codes=True)
    )
    np.testing.assert_array_equal(got["income"], reference["income"])
    assert not np.array_equal(
        got.loc[got["period"] == 1, "income"].to_numpy(),
        ordinary.simulate(
            params={}, initial_conditions=initial, seed=537, log_level="off"
        )
        .to_dataframe(use_categorical_codes=True)
        .loc[lambda frame: frame["period"] == 1, "income"]
        .to_numpy(),
    )


def test_native_iid_node_simulation_and_repeat_seed() -> None:
    process = NormalIIDProcess(
        n_points=3,
        gauss_hermite=True,
        mu=0.0,
        sigma=0.7,
        simulated_draws="nodes",
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
        ).to_dataframe(use_categorical_codes=True)
        for _ in range(2)
    ]
    np.testing.assert_array_equal(frames[0]["income"], frames[1]["income"])
    realized = frames[0].loc[frames[0]["period"] == 1, "income"].to_numpy()
    assert np.isin(realized, np.asarray(process.get_gridpoints())).all()


def test_runtime_supplied_node_markov_rows_and_support() -> None:
    nodes = (-1.3, 0.1, 2.4)
    process = TauchenAR1Process.on_nodes(nodes=np.array(nodes), simulated_draws="nodes")
    model = _model(process)
    params = {"alive": {"income": {"rho": 0.6, "sigma": 0.4}}}
    initial = {
        "income": jnp.full(2000, nodes[1]),
        "age": jnp.zeros(2000),
        "regime_id": jnp.full(2000, RegimeId.alive, dtype=jnp.int32),
    }
    frame = model.simulate(
        params=params, initial_conditions=initial, seed=537, log_level="off"
    ).to_dataframe(use_categorical_codes=True)
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


@pytest.mark.parametrize("invalid", ["nodes", "nonprocess"])
def test_unsupported_phased_process_forms_fail_at_declaration(invalid: str) -> None:
    process = NormalIIDProcess(n_points=3, gauss_hermite=True, mu=0.0, sigma=1.0)
    other = (
        replace(process, mu=1.0, simulated_draws="nodes")
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
        n_points=3, gauss_hermite=True, mu=0.0, sigma=1.0, simulated_draws="nodes"
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
    ).to_dataframe(use_categorical_codes=True)
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
    ).to_dataframe(use_categorical_codes=True)
    assert frame.loc[frame["period"] == 1, "nodes_income"].shape == (2,)


@pytest.mark.parametrize("runtime_checks", [True, False])
@pytest.mark.parametrize("log_level", ["off", "debug"])
def test_node_draws_do_not_depend_on_runtime_checks_or_logging(
    *, runtime_checks: bool, log_level: LogLevel
) -> None:
    process = NormalIIDProcess(
        n_points=3, gauss_hermite=True, mu=0.0, sigma=1.0, simulated_draws="nodes"
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
    ).to_dataframe(use_categorical_codes=True)
    got = model.simulate(
        params={},
        solution=solution,
        initial_conditions=initial,
        seed=537,
        log_level=log_level,
        runtime_checks=runtime_checks,
    ).to_dataframe(use_categorical_codes=True)
    np.testing.assert_array_equal(got["income"], reference["income"])


def test_node_kernel_accepts_empty_subject_keys() -> None:
    process = NormalIIDProcess(
        n_points=3, gauss_hermite=True, mu=0.0, sigma=1.0, simulated_draws="nodes"
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
    ).to_dataframe(use_categorical_codes=True)
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
        (_tauchen(rho=None, simulated_draws="nodes"), _tauchen(rho=None)),
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
    process = _tauchen(rho=0.9, simulated_draws="nodes")
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
        ).to_dataframe(use_categorical_codes=True)
        return frame.loc[frame["period"] == 1, "income"].to_numpy()

    np.testing.assert_array_equal(realized(top + 5.0), realized(top))


def test_phased_process_with_one_law_is_accepted() -> None:
    """Phase declarations sharing one law are not distinct laws."""
    base = NormalIIDProcess(n_points=3, gauss_hermite=True, mu=0.0, sigma=1.0)
    process = Phased(solve=base, simulate=replace(base, simulated_draws="nodes"))
    assert isinstance(
        Regime(states={"income": process}, functions={"utility": _utility}), Regime
    )


def _positive_utility(income: ContinuousState) -> FloatND:
    return income**2 + 1.0


def _normal_cdf(x: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.vectorize(math.erf)(np.asarray(x) / math.sqrt(2.0)))


def _binned_row(cdf_at_midpoints: np.ndarray) -> np.ndarray:
    """First bin the lower tail, last the upper tail, the rest CDF differences."""
    return np.concatenate(
        [cdf_at_midpoints[:1], np.diff(cdf_at_midpoints), 1 - cdf_at_midpoints[-1:]]
    )


def _reference_row(*, process: _ContinuousStochasticProcess, value: float):
    """The process's transition row from `value`, written from the law itself."""
    params = {k: float(v) for k, v in process.params.items()}
    nodes = np.asarray(process.get_gridpoints(), dtype=np.float64)
    mean = params["mu"] + params["rho"] * value
    if isinstance(process, TauchenNormalMixtureAR1Process):
        half = 0.5 * (nodes[-1] - nodes[0]) / (len(nodes) - 1)

        def mixture(x: np.ndarray) -> np.ndarray:
            return params["p1"] * _normal_cdf(
                (x - params["mu1"]) / params["sigma1"]
            ) + (1 - params["p1"]) * _normal_cdf((x - params["mu2"]) / params["sigma2"])

        return _binned_row(mixture(nodes[:-1] + half - mean))
    rule = getattr(process, "transition_rule", "midpoint")
    sigma = params["sigma"]
    if rule == "density":
        density = np.exp(-0.5 * ((nodes - mean) / sigma) ** 2)
        return density / density.sum()
    if rule == "innovation_quadrature":
        process = cast("_TauchenOnNodesProcess", process)
        weights = np.asarray(process.innovation_weights)
        weights = weights / weights.sum()
        row = np.zeros_like(nodes)
        for shock, weight in zip(process.innovation_nodes, weights, strict=True):
            target = np.clip(mean + sigma * shock, nodes[0], nodes[-1])
            lower = min(
                int(np.searchsorted(nodes, target, side="right")) - 1, len(nodes) - 2
            )
            lower = max(lower, 0)
            share = (target - nodes[lower]) / (nodes[lower + 1] - nodes[lower])
            row[lower] += weight * (1 - share)
            row[lower + 1] += weight * share
        return row
    return _binned_row(_normal_cdf(((nodes[:-1] + nodes[1:]) / 2 - mean) / sigma))


_TAUCHEN_FAMILY = {
    "tauchen": lambda: _tauchen(rho=0.9, mu=0.2),
    "tauchen-gauss-hermite": lambda: TauchenAR1Process(
        n_points=3, gauss_hermite=True, rho=0.9, sigma=0.3, mu=0.2
    ),
    "on-nodes-midpoint": lambda: TauchenAR1Process.on_nodes(
        nodes=jnp.array([-1.0, 0.0, 1.5]), rho=0.9, sigma=0.3, mu=0.2
    ),
    "on-nodes-density": lambda: TauchenAR1Process.on_nodes(
        nodes=jnp.array([-1.0, 0.0, 1.5]),
        rho=0.9,
        sigma=0.3,
        mu=0.2,
        transition_rule="density",
    ),
    "on-nodes-innovation-quadrature": lambda: TauchenAR1Process.on_nodes(
        nodes=jnp.array([-1.0, 0.0, 1.5]),
        rho=0.9,
        sigma=0.3,
        mu=0.2,
        transition_rule="innovation_quadrature",
    ),
    "normal-mixture": lambda: TauchenNormalMixtureAR1Process(
        n_points=3,
        rho=0.9,
        mu=0.2,
        n_std=2.0,
        p1=0.7,
        mu1=-0.1,
        sigma1=0.2,
        mu2=0.2333333,
        sigma2=0.5,
    ),
}


@pytest.mark.parametrize("position", ["between", "above", "below"])
@pytest.mark.parametrize("family", list(_TAUCHEN_FAMILY))
def test_tauchen_family_row_off_the_nodes_is_the_law(
    *, family: str, position: str
) -> None:
    """Off its nodes, a Tauchen-family row bins the law from the actual value."""
    process = _TAUCHEN_FAMILY[family]()
    nodes = np.asarray(process.get_gridpoints(), dtype=np.float64)
    value = {
        "between": 0.3 * nodes[0] + 0.7 * nodes[1],
        "above": nodes[-1] + 0.8,
        "below": nodes[0] - 0.8,
    }[position]
    weights = _get_weights_func_for_process(name="income", grid=process)
    np.testing.assert_allclose(
        np.asarray(weights(income=jnp.asarray(value))),
        _reference_row(process=process, value=value),
        rtol=1e-5,
        atol=1e-6,
    )


@pytest.mark.parametrize("family", list(_TAUCHEN_FAMILY))
def test_tauchen_family_row_at_a_node_is_the_tabulated_row(*, family: str) -> None:
    process = _TAUCHEN_FAMILY[family]()
    weights = _get_weights_func_for_process(name="income", grid=process)
    nodes = process.get_gridpoints()
    np.testing.assert_array_equal(
        np.stack([np.asarray(weights(income=node)) for node in nodes]),
        np.asarray(process.get_transition_probs()),
    )


def _rouwenhorst(**kwargs: float | str) -> RouwenhorstAR1Process:
    law = {"n_points": 3, "rho": 0.9, "sigma": 0.3, "mu": 0.0}
    return RouwenhorstAR1Process(**(law | kwargs))  # ty: ignore[invalid-argument-type]


@pytest.mark.parametrize(
    ("position", "expected"),
    [("between", lambda rows: 0.25 * rows[0] + 0.75 * rows[1]), ("above", None)],
)
def test_rouwenhorst_row_blends_neighbouring_rows_and_reads_the_edge_beyond(
    *, position: str, expected: Callable[[np.ndarray], np.ndarray] | None
) -> None:
    """Rows exist only at nodes: a convex blend between them, the edge row beyond."""
    process = _rouwenhorst()
    nodes = np.asarray(process.get_gridpoints(), dtype=np.float64)
    rows = np.asarray(process.get_transition_probs(), dtype=np.float64)
    value = 0.25 * nodes[0] + 0.75 * nodes[1] if position == "between" else 9.0
    weights = _get_weights_func_for_process(name="income", grid=process)
    np.testing.assert_allclose(
        np.asarray(weights(income=jnp.asarray(value))),
        rows[-1] if expected is None else expected(rows),
        rtol=1e-6,
    )


def test_carried_tauchen_onto_narrower_nodes_solves_with_the_law_row() -> None:
    """A source node beyond the target's nodes reads the target law's row there."""
    work, retired = _tauchen(n_std=3.0), _tauchen(n_std=2.0)
    solution = _work_retired_model(work=work, retired=retired).solve(
        params={}, log_level="off"
    )
    work_nodes = np.asarray(work.get_gridpoints(), dtype=np.float64)
    retired_values = np.asarray(solution.values[2]["retired"], dtype=np.float64)
    expected = [
        z**2 + 0.9 * _reference_row(process=retired, value=z) @ retired_values
        for z in work_nodes
    ]
    np.testing.assert_allclose(
        np.asarray(solution.values[1]["work"]), expected, rtol=1e-5
    )


def test_carried_rouwenhorst_with_different_nodes_is_refused() -> None:
    with pytest.raises(ModelInitializationError, match="Rouwenhorst"):
        _work_retired_model(
            work=_rouwenhorst(sigma=0.3), retired=_rouwenhorst(sigma=0.2)
        )


def _period_zero_continuations(
    *,
    process: _ContinuousStochasticProcess,
    certainty_equivalent: CertaintyEquivalent | None,
    starts: np.ndarray,
) -> np.ndarray:
    """Return each start's period-0 simulated value minus its utility."""
    model = Model(
        regimes={
            "alive": Regime(
                states={"income": process},
                functions={"utility": _positive_utility},
                certainty_equivalent=certainty_equivalent,
            ),
            "dead": Regime(functions={"utility": lambda: jnp.asarray(0.0)}),
        },
        ages=AgeGrid(start=0, inclusive_stop=2, step="Y"),
        initial_nodes={0: "alive"},
        edges={"alive": {"alive": 0, "dead": 1}},
        regime_id_class=RegimeId,
        fixed_params={"discount_factor": 0.9},
    )
    params = (
        {}
        if certainty_equivalent is None
        else {"alive": {"certainty_equivalent": {"risk_aversion": 4.0}}}
    )
    frame = model.simulate(
        params=params,
        solution=model.solve(params=params, log_level="off"),
        initial_conditions={
            "income": jnp.asarray(starts),
            "age": jnp.zeros(len(starts)),
            "regime_id": jnp.full(len(starts), RegimeId.alive, dtype=jnp.int32),
        },
        seed=11,
        log_level="off",
    ).to_dataframe(use_categorical_codes=True)
    values = frame.loc[frame["period"] == 0].sort_values("subject_id")["value"]
    return values.to_numpy() - (starts**2 + 1.0)


@pytest.mark.parametrize("side", ["above", "below"])
@pytest.mark.parametrize("simulated_draws", ["continuous", "nodes"])
@pytest.mark.parametrize(
    "certainty_equivalent", [None, PowerMean()], ids=["expectation", "power-mean"]
)
def test_rouwenhorst_continuation_beyond_an_outer_node_extends_the_edge_segment(
    *,
    certainty_equivalent: CertaintyEquivalent | None,
    simulated_draws: str,
    side: str,
) -> None:
    """Beyond the nodes, the continuation is linear in the two edge nodes' values.

    The continuation at a node is the expectation (or certainty equivalent) under
    that node's transition row, so the extension is linear in those, not the
    certainty equivalent of an extrapolated, signed row.
    """
    process = _rouwenhorst(simulated_draws=simulated_draws)
    nodes = np.asarray(process.get_gridpoints(), dtype=np.float64)
    edge, inner = (nodes[-1], nodes[-2]) if side == "above" else (nodes[0], nodes[1])
    beyond = edge + 2.0 * (edge - inner)
    got, at_edge, at_inner = _period_zero_continuations(
        process=process,
        certainty_equivalent=certainty_equivalent,
        starts=np.array([beyond, edge, inner]),
    )
    np.testing.assert_allclose(got, at_edge + 2.0 * (at_edge - at_inner), rtol=1e-5)


@categorical(ordered=False)
class _ForkId:
    source: ScalarInt
    left: ScalarInt
    right: ScalarInt


class _LinearExpectationSubclass(LinearExpectation):
    """Use the declared aggregate rather than the exact-type engine reduction."""


def _fork_model(
    *,
    certainty_equivalent: CertaintyEquivalent | None,
    different_laws: bool = False,
    two_coordinates: bool = False,
    process: RouwenhorstAR1Process | None = None,
) -> Model:
    """Build two mutually exclusive targets with carried process coordinates."""
    if process is None:
        process = RouwenhorstAR1Process(
            n_points=2,
            rho=0.6,
            sigma=0.8,
            mu=0.0,
            simulated_draws="continuous",
            off_grid="extrapolate",
        )

    def left_utility(income: ContinuousState) -> FloatND:
        return 1.5 + 0.5 * income

    def right_utility(income: ContinuousState) -> FloatND:
        return 2.5 + 1.5 * income

    def source_utility(income: ContinuousState) -> FloatND:
        return 1.0 + income**2

    def source_product(*, income: ContinuousState, wage: ContinuousState) -> FloatND:
        return 1.0 + income**2 + wage**2

    def left_product(*, income: ContinuousState, wage: ContinuousState) -> FloatND:
        return 4.0 + income * wage

    def right_product(*, income: ContinuousState, wage: ContinuousState) -> FloatND:
        return 6.0 + 2.0 * income * wage

    states = {"income": process, **({"wage": process} if two_coordinates else {})}
    return Model(
        regimes={
            "source": Regime(
                states=states,
                functions={
                    "utility": source_product if two_coordinates else source_utility
                },
                certainty_equivalent=certainty_equivalent,
            ),
            "left": Regime(
                states=states,
                functions={
                    "utility": left_product if two_coordinates else left_utility
                },
            ),
            "right": Regime(
                states=(
                    {"income": replace(process, rho=-0.6)} if different_laws else states
                ),
                functions={
                    "utility": right_product if two_coordinates else right_utility
                },
            ),
        },
        ages=AgeGrid(start=0, inclusive_stop=1, step="Y"),
        initial_nodes={0: "source"},
        edges={
            "source": Transition(
                targets={"left": 0, "right": 0},
                law={
                    "left": StochasticTransition(func=lambda: jnp.asarray(0.5)),
                    "right": StochasticTransition(func=lambda: jnp.asarray(0.5)),
                },
            )
        },
        regime_id_class=_ForkId,
        fixed_params={"discount_factor": 0.9},
    )


type _EdgeBuildResult = tuple[
    Callable[..., q_and_f.ContinuationAggregate],
    tuple[Callable[..., ArrayTree], ...],
    frozenset[str],
]


def _build_literal_process_corners(
    *,
    functions: EconFunctionsMapping,
    period_targets: tuple[RegimeName, ...],
    build: Callable[..., _EdgeBuildResult],
) -> tuple[q_and_f._ComputeCE, ...]:
    """Enumerate original row substitutions independently of the index/loop program."""
    prefixes = tuple(f"weight_{target}__" for target in period_targets)
    edges = {
        name: rows
        for name, func in functions.items()
        if name.startswith(prefixes)
        and isinstance(rows := getattr(func, "edge_rows", None), q_and_f.EdgeRows)
    }
    corners = []
    for choices in itertools.product((False, True), repeat=len(edges)):
        corner = build(
            functions=MappingProxyType(
                dict(functions)
                | {
                    name: rows.neighbour
                    for (name, rows), use_neighbour in zip(
                        edges.items(), choices, strict=True
                    )
                    if use_neighbour
                }
            ),
            period_targets=period_targets,
        )[0]
        assert isinstance(corner, q_and_f._ComputeCE)
        assert corner.edge_extension is None
        corners.append(corner)
    return tuple(corners)


@pytest.mark.parametrize("certainty_equivalent", [None, PowerMean()])
def test_process_edges_build_one_reusable_corner_program(
    *,
    monkeypatch: pytest.MonkeyPatch,
    certainty_equivalent: CertaintyEquivalent | None,
) -> None:
    """Build the base and one corner program for every joint process extension."""
    original = q_and_f._get_compute_CE_with_process_edges
    observations: list[int] = []

    def observed(
        *,
        functions: EconFunctionsMapping,
        period_targets: tuple[RegimeName, ...],
        phase: Literal["solution", "simulation"],
        build: Callable[..., _EdgeBuildResult],
    ) -> _EdgeBuildResult:
        count = 0

        def counted(
            *, functions: EconFunctionsMapping, period_targets: tuple[RegimeName, ...]
        ) -> _EdgeBuildResult:
            nonlocal count
            count += 1
            return build(functions=functions, period_targets=period_targets)

        result = original(
            functions=functions,
            period_targets=period_targets,
            phase=phase,
            build=counted,
        )
        if phase == "simulation" and set(period_targets) == {"left", "right"}:
            observations.append(count)
        return result

    monkeypatch.setattr(q_and_f, "_get_compute_CE_with_process_edges", observed)
    _fork_model(certainty_equivalent=certainty_equivalent, two_coordinates=True)
    assert observations
    assert set(observations) == {2}


@pytest.mark.parametrize("two_coordinates", [False, True])
def test_process_edge_program_builds_each_target_twice(
    *, monkeypatch: pytest.MonkeyPatch, two_coordinates: bool
) -> None:
    """Build each target once for the base and once for the indexed corner."""
    target_builds = 0
    observations: list[int] = []
    original_target = q_and_f._build_target_continuation
    original_edges = q_and_f._get_compute_CE_with_process_edges

    def count_target[**P, R](func: Callable[P, R]) -> Callable[P, R]:
        @functools.wraps(func)
        def counted(*args: P.args, **kwargs: P.kwargs) -> R:
            nonlocal target_builds
            target_builds += 1
            return func(*args, **kwargs)

        return counted

    def observed(
        *,
        functions: EconFunctionsMapping,
        period_targets: tuple[RegimeName, ...],
        phase: Literal["solution", "simulation"],
        build: Callable[..., _EdgeBuildResult],
    ) -> _EdgeBuildResult:
        before = target_builds
        result = original_edges(
            functions=functions, period_targets=period_targets, phase=phase, build=build
        )
        if phase == "simulation" and set(period_targets) == {"left", "right"}:
            observations.append(target_builds - before)
        return result

    monkeypatch.setattr(
        q_and_f, "_build_target_continuation", count_target(original_target)
    )
    monkeypatch.setattr(q_and_f, "_get_compute_CE_with_process_edges", observed)
    _fork_model(certainty_equivalent=None, two_coordinates=two_coordinates)
    assert observations
    assert set(observations) == {4}


@pytest.mark.parametrize("two_coordinates", [False, True])
@pytest.mark.parametrize("certainty_equivalent", [None, LinearExpectation()])
def test_linear_process_edges_read_only_each_targets_local_corners(
    *,
    monkeypatch: pytest.MonkeyPatch,
    two_coordinates: bool,
    certainty_equivalent: CertaintyEquivalent | None,
) -> None:
    """Reuse a target's expectation when only another target's row changes."""
    reader_calls = 0
    captured: list[q_and_f._ComputeCE] = []
    original_target = q_and_f._build_target_continuation
    original_edges = q_and_f._get_compute_CE_with_process_edges

    def count_reader[**P, R](func: Callable[P, R]) -> Callable[P, R]:
        @functools.wraps(func)
        def counted(*args: P.args, **kwargs: P.kwargs) -> R:
            nonlocal reader_calls
            reader_calls += 1
            return func(*args, **kwargs)

        return counted

    def wrap_target[**P](
        func: Callable[P, q_and_f._TargetContinuation],
    ) -> Callable[P, q_and_f._TargetContinuation]:
        @functools.wraps(func)
        def wrapped(*args: P.args, **kwargs: P.kwargs) -> q_and_f._TargetContinuation:
            target = func(*args, **kwargs)
            assert target.next_V is not None
            return replace(target, next_V=count_reader(target.next_V))

        return wrapped

    def observed(
        *,
        functions: EconFunctionsMapping,
        period_targets: tuple[RegimeName, ...],
        phase: Literal["solution", "simulation"],
        build: Callable[..., _EdgeBuildResult],
    ) -> _EdgeBuildResult:
        result = original_edges(
            functions=functions, period_targets=period_targets, phase=phase, build=build
        )
        if phase == "simulation" and set(period_targets) == {"left", "right"}:
            assert isinstance(result[0], q_and_f._ComputeCE)
            captured.append(result[0])
        return result

    monkeypatch.setattr(
        q_and_f, "_build_target_continuation", wrap_target(original_target)
    )
    monkeypatch.setattr(q_and_f, "_get_compute_CE_with_process_edges", observed)
    _fork_model(
        certainty_equivalent=certainty_equivalent, two_coordinates=two_coordinates
    )
    values = MappingProxyType(
        {
            "left": jnp.asarray(
                [[5.0, 3.0], [3.0, 5.0]] if two_coordinates else [1.0, 2.0]
            ),
            "right": jnp.asarray(
                [[8.0, 4.0], [4.0, 8.0]] if two_coordinates else [1.0, 4.0]
            ),
        }
    )
    reader_calls = 0
    with jax.disable_jit():
        captured[0](
            next_regime_to_V_arr=values,
            zero=jnp.asarray(0.0),
            states_actions_params=MappingProxyType(
                {
                    "income": jnp.asarray(3.0),
                    **({"wage": jnp.asarray(3.0)} if two_coordinates else {}),
                }
            ),
        )
    assert reader_calls == (8 if two_coordinates else 4)


@pytest.mark.parametrize("two_coordinates", [False, True])
def test_process_edge_program_traces_each_target_twice(
    *, monkeypatch: pytest.MonkeyPatch, two_coordinates: bool
) -> None:
    """Trace the base and one corner body without leaking state between transforms."""
    reader_calls = 0
    captured: list[q_and_f._ComputeCE] = []
    original_target = q_and_f._build_target_continuation
    original_edges = q_and_f._get_compute_CE_with_process_edges

    def count_reader[**P, R](func: Callable[P, R]) -> Callable[P, R]:
        @functools.wraps(func)
        def counted(*args: P.args, **kwargs: P.kwargs) -> R:
            nonlocal reader_calls
            reader_calls += 1
            return func(*args, **kwargs)

        return counted

    def wrap_target[**P](
        func: Callable[P, q_and_f._TargetContinuation],
    ) -> Callable[P, q_and_f._TargetContinuation]:
        @functools.wraps(func)
        def wrapped(*args: P.args, **kwargs: P.kwargs) -> q_and_f._TargetContinuation:
            target = func(*args, **kwargs)
            assert target.next_V is not None
            return replace(target, next_V=count_reader(target.next_V))

        return wrapped

    def observed(
        *,
        functions: EconFunctionsMapping,
        period_targets: tuple[RegimeName, ...],
        phase: Literal["solution", "simulation"],
        build: Callable[..., _EdgeBuildResult],
    ) -> _EdgeBuildResult:
        result = original_edges(
            functions=functions, period_targets=period_targets, phase=phase, build=build
        )
        if phase == "simulation" and set(period_targets) == {"left", "right"}:
            assert isinstance(result[0], q_and_f._ComputeCE)
            captured.append(result[0])
        return result

    monkeypatch.setattr(
        q_and_f, "_build_target_continuation", wrap_target(original_target)
    )
    monkeypatch.setattr(q_and_f, "_get_compute_CE_with_process_edges", observed)
    _fork_model(certainty_equivalent=None, two_coordinates=two_coordinates)
    assert captured
    compute = captured[0]
    assert compute.edge_extension is not None
    assert len(compute.edge_extension.distances) == (4 if two_coordinates else 2)
    values = MappingProxyType(
        {
            "left": jnp.asarray(
                [[5.0, 3.0], [3.0, 5.0]] if two_coordinates else [1.0, 2.0]
            ),
            "right": jnp.asarray(
                [[8.0, 4.0], [4.0, 8.0]] if two_coordinates else [1.0, 4.0]
            ),
        }
    )

    def evaluate(next_values: MappingProxyType[RegimeName, FloatND]) -> FloatND:
        result = compute(
            next_regime_to_V_arr=next_values,
            zero=jnp.asarray(0.0),
            states_actions_params=MappingProxyType(
                {
                    "income": jnp.asarray(3.0),
                    **({"wage": jnp.asarray(3.0)} if two_coordinates else {}),
                }
            ),
        )
        assert isinstance(result, tuple)
        return result[0]

    reader_calls = 0
    with jax.checking_leaks():
        compiled = jax.jit(evaluate)
        actual = compiled(values)
    assert reader_calls == 4
    # E[income] = E[wage] = rho * 3 = 1.8; the two target weights are one half.
    expected = 9.86 if two_coordinates else 3.8
    np.testing.assert_allclose(actual, expected, rtol=10.0**-DECIMAL_PRECISION)
    np.testing.assert_allclose(
        compiled(jax.tree.map(lambda value: value + 10.0, values)),
        expected + 10.0,
        rtol=10.0**-DECIMAL_PRECISION,
    )
    with jax.checking_leaks():
        np.testing.assert_allclose(
            evaluate(values), expected, rtol=10.0**-DECIMAL_PRECISION
        )
        with jax.disable_jit():
            np.testing.assert_allclose(
                evaluate(values), expected, rtol=10.0**-DECIMAL_PRECISION
            )
        batched = jax.vmap(evaluate)(
            jax.tree.map(lambda value: jnp.stack((value, value + 10.0)), values)
        )
    np.testing.assert_allclose(
        batched, [expected, expected + 10.0], rtol=10.0**-DECIMAL_PRECISION
    )
    _, tangent = jax.jvp(evaluate, (values,), (jax.tree.map(jnp.ones_like, values),))
    np.testing.assert_allclose(tangent, 1.0, rtol=10.0**-DECIMAL_PRECISION)
    gradients = jax.grad(evaluate)(values)
    expected_gradient = (
        [[0.08, -0.28], [-0.28, 0.98]] if two_coordinates else [-0.2, 0.7]
    )
    for gradient in gradients.values():
        np.testing.assert_allclose(
            gradient, expected_gradient, rtol=10.0**-DECIMAL_PRECISION
        )


@pytest.mark.parametrize("compiled", [False, True], ids=["eager", "jit"])
@pytest.mark.parametrize("case", ["negative-inf", "signed-zero", "finite-overflow"])
def test_process_edge_corners_keep_nonfinite_and_zero_semantics(
    *, monkeypatch: pytest.MonkeyPatch, case: str, compiled: bool
) -> None:
    """Keep the tensor expansion's nonfinite outcomes and represented zero sign."""
    captured: list[tuple[q_and_f._ComputeCE, tuple[q_and_f._ComputeCE, ...]]] = []
    original = q_and_f._get_compute_CE_with_process_edges

    def observed(
        *,
        functions: EconFunctionsMapping,
        period_targets: tuple[RegimeName, ...],
        phase: Literal["solution", "simulation"],
        build: Callable[..., _EdgeBuildResult],
    ) -> _EdgeBuildResult:
        result = original(
            functions=functions, period_targets=period_targets, phase=phase, build=build
        )
        if phase == "simulation" and set(period_targets) == {"left", "right"}:
            assert isinstance(result[0], q_and_f._ComputeCE)
            captured.append(
                (
                    result[0],
                    _build_literal_process_corners(
                        functions=functions,
                        period_targets=period_targets,
                        build=build,
                    ),
                )
            )
        return result

    dtype = np.asarray(jnp.asarray(1.0)).dtype
    rho = np.nextafter(dtype.type(1.0), dtype.type(0.0))
    process = RouwenhorstAR1Process(
        n_points=2,
        rho=0.6 if case == "finite-overflow" else float(rho),
        sigma=0.8 if case == "finite-overflow" else float(np.sqrt(1.0 - rho**2)),
        mu=0.0,
        off_grid="extrapolate",
    )
    with monkeypatch.context() as context:
        context.setattr(q_and_f, "_get_compute_CE_with_process_edges", observed)
        _fork_model(certainty_equivalent=None, process=process)
    assert captured
    compute, corners = captured[0]
    extension = compute.edge_extension
    assert extension is not None
    assert len(corners) == 4
    nodes = process.get_gridpoints()
    assert np.isfinite(np.asarray(nodes)).all()
    rows = np.asarray(process.get_transition_probs())
    if case == "finite-overflow":
        assert np.all(rows > 0)
        left = right = [np.finfo(dtype).max, 1.0]
    else:
        np.testing.assert_array_equal(rows, np.eye(2, dtype=dtype))
        left = [-np.inf, 1.0] if case == "negative-inf" else [-0.0, 0.0]
        right = [2.0, 2.0] if case == "negative-inf" else [0.0, -0.0]
    values = MappingProxyType(
        {
            "left": jnp.asarray(left, dtype=dtype),
            "right": jnp.asarray(right, dtype=dtype),
        }
    )

    def evaluate(corner: q_and_f._ComputeCE) -> NDArray[np.float32 | np.float64]:
        def read(next_values: MappingProxyType[RegimeName, FloatND]) -> FloatND:
            aggregate = corner(
                next_regime_to_V_arr=next_values,
                zero=jnp.asarray(0.0, dtype=dtype),
                states_actions_params=MappingProxyType({"income": 3.0 * nodes[-1]}),
            )
            assert isinstance(aggregate, tuple)
            return aggregate[0]

        return np.asarray(jax.jit(read)(values) if compiled else read(values))

    actual = evaluate(compute)
    f00, f01, f10, f11 = (evaluate(corner) for corner in corners)
    with np.errstate(invalid="ignore", over="ignore"):
        literal = 4.0 * f00 - 2.0 * f01 - 2.0 * f10 + f11
    np.testing.assert_array_equal(actual, literal)
    if case == "negative-inf":
        assert np.isnan(actual)
    elif case == "finite-overflow":
        assert np.isneginf(actual)
    else:
        assert not np.signbit(actual)
        assert not np.signbit(literal)


@pytest.mark.parametrize(
    ("certainty_equivalent", "different_laws", "two_coordinates"),
    [
        (None, False, False),
        (LinearExpectation(), False, False),
        (PowerMean(), False, False),
        (_LinearExpectationSubclass(), False, False),
        (None, True, False),
        (None, False, True),
    ],
    ids=["default", "linear", "power-mean", "subclass", "distinct-laws", "two-states"],
)
def test_process_edges_retain_the_original_cartesian_corners(
    *,
    monkeypatch: pytest.MonkeyPatch,
    certainty_equivalent: CertaintyEquivalent | None,
    different_laws: bool,
    two_coordinates: bool,
) -> None:
    """Every process weight retains its corner in the original tensor expansion."""
    original = q_and_f._get_compute_CE_with_process_edges
    observations: list[int] = []
    captured: list[tuple[q_and_f._ComputeCE, tuple[q_and_f._ComputeCE, ...]]] = []

    def observed(
        *,
        functions: EconFunctionsMapping,
        period_targets: tuple[RegimeName, ...],
        phase: Literal["solution", "simulation"],
        build: Callable[..., _EdgeBuildResult],
    ) -> _EdgeBuildResult:
        count = 0

        def counted(
            *, functions: EconFunctionsMapping, period_targets: tuple[RegimeName, ...]
        ) -> _EdgeBuildResult:
            nonlocal count
            count += 1
            return build(functions=functions, period_targets=period_targets)

        result = original(
            functions=functions,
            period_targets=period_targets,
            phase=phase,
            build=counted,
        )
        if phase == "simulation" and set(period_targets) == {"left", "right"}:
            observations.append(count)
            assert isinstance(result[0], q_and_f._ComputeCE)
            captured.append(
                (
                    result[0],
                    _build_literal_process_corners(
                        functions=functions,
                        period_targets=period_targets,
                        build=build,
                    ),
                )
            )
        return result

    monkeypatch.setattr(q_and_f, "_get_compute_CE_with_process_edges", observed)
    _fork_model(
        certainty_equivalent=certainty_equivalent,
        different_laws=different_laws,
        two_coordinates=two_coordinates,
    )
    assert observations
    assert set(observations) == {2}
    compute, literal_corners = captured[0]
    extension = compute.edge_extension
    assert extension is not None
    assert len(literal_corners) == (16 if two_coordinates else 4)
    values = MappingProxyType(
        {
            "left": jnp.asarray(
                [[5.0, 3.0], [3.0, 5.0]] if two_coordinates else [1.0, 2.0]
            ),
            "right": jnp.asarray(
                [[8.0, 4.0], [4.0, 8.0]] if two_coordinates else [1.0, 4.0]
            ),
        }
    )
    states = MappingProxyType(
        {
            "income": jnp.asarray(3.0),
            **({"wage": jnp.asarray(3.0)} if two_coordinates else {}),
            **{
                name: jnp.asarray(-1.0) for name in compute.ce_flat_param_names.values()
            },
        }
    )
    for index, literal in enumerate(literal_corners):
        reference = literal(
            next_regime_to_V_arr=values,
            zero=jnp.asarray(0.0),
            states_actions_params=states,
        )
        actual = extension.read_corner(
            index=index,
            next_regime_to_V_arr=values,
            zero=jnp.asarray(0.0),
            states_actions_params=states,
        )
        assert isinstance(actual, tuple)
        assert isinstance(reference, tuple)
        np.testing.assert_allclose(
            actual[0], reference[0], rtol=10.0**-DECIMAL_PRECISION
        )
    if isinstance(certainty_equivalent, PowerMean):
        for read in (
            literal_corners[0],
            functools.partial(extension.read_corner, index=0),
        ):
            # keyword-only-exempt: library-callback=jax.jvp
            def evaluate(
                next_values: MappingProxyType[RegimeName, FloatND],
                *,
                read: Callable[..., q_and_f.ContinuationAggregate] = read,
            ) -> FloatND:
                aggregate = read(
                    next_regime_to_V_arr=next_values,
                    zero=jnp.asarray(0.0),
                    states_actions_params=states,
                )
                assert isinstance(aggregate, tuple)
                return aggregate[0]

            with pytest.raises(ScaledLotteryDifferentiationError):
                jax.jvp(evaluate, (values,), (jax.tree.map(jnp.ones_like, values),))
            with pytest.raises(ScaledLotteryDifferentiationError):
                jax.vjp(evaluate, values)


@pytest.mark.parametrize(
    "case", ["linear", "power-mean", "distinct-laws", "two-states"]
)
def test_process_edge_sharing_preserves_the_continuation(case: str) -> None:
    """Carried targets retain their analytic values on and beyond the process nodes."""
    model = _fork_model(
        certainty_equivalent=PowerMean() if case == "power-mean" else None,
        different_laws=case == "distinct-laws",
        two_coordinates=case == "two-states",
    )
    params = (
        {"source": {"certainty_equivalent": {"risk_aversion": -1.0}}}
        if case == "power-mean"
        else {}
    )
    starts = jnp.asarray([3.0, 1.0, -1.0])
    initial_conditions = {
        "income": starts,
        "age": jnp.zeros(3),
        "regime_id": jnp.full(3, _ForkId.source, dtype=jnp.int32),
        **({"wage": starts} if case == "two-states" else {}),
    }
    frame = model.simulate(
        params=params,
        solution=model.solve(params=params, log_level="off"),
        initial_conditions=initial_conditions,
        seed=11,
        log_level="off",
    ).to_dataframe(use_categorical_codes=True)
    got = frame.loc[frame["period"] == 0].sort_values("subject_id")["value"]
    expected = {
        "linear": [3.8, 2.6, 1.4],
        "distinct-laws": [1.1, 1.7, 2.3],
        "two-states": [9.86, 5.54, 5.54],
        "power-mean": [
            4.0 * math.sqrt(8.2)
            - 2.0 * math.sqrt(3.7)
            - 2.0 * math.sqrt(7.3)
            + math.sqrt(2.8),
            math.sqrt(8.2),
            math.sqrt(2.8),
        ],
    }[case]
    np.testing.assert_allclose(
        got.to_numpy()
        - (1.0 + np.asarray(starts) ** 2 * (2 if case == "two-states" else 1)),
        0.9 * np.asarray(expected),
        rtol=10.0**-DECIMAL_PRECISION,
    )


def test_process_edge_program_observes_another_targets_row(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Select another target's row before a dependent probability callback reads it."""
    captured: list[q_and_f._ComputeCE] = []

    def inject_cross_read[**P, R](func: Callable[P, R]) -> Callable[P, R]:
        @functools.wraps(func)
        def observed(*args: P.args, **kwargs: P.kwargs) -> R:
            if kwargs.get("extend_process_edges") != "simulation" or set(
                cast("tuple[RegimeName, ...]", kwargs["period_targets"])
            ) != {"left", "right"}:
                return func(*args, **kwargs)
            functions = cast("EconFunctionsMapping", kwargs["functions"])
            original_left = functions["weight_left__next_income"]
            rows = getattr(original_left, "edge_rows", None)
            assert isinstance(rows, q_and_f.EdgeRows)

            def left_row(
                *, income: ContinuousState, weight_right__next_income: FloatND
            ) -> FloatND:
                return (
                    0.5 * original_left(income=income) + 0.5 * weight_right__next_income
                )

            def left_neighbour(
                *, income: ContinuousState, weight_right__next_income: FloatND
            ) -> FloatND:
                return (
                    0.5 * rows.neighbour(income=income)
                    + 0.5 * weight_right__next_income
                )

            monkeypatch.setattr(
                left_row,
                "edge_rows",
                replace(rows, neighbour=left_neighbour),
                raising=False,
            )
            functions = MappingProxyType(
                dict(functions) | {"weight_left__next_income": left_row}
            )
            result = functools.partial(func, *args, **kwargs)(functions=functions)
            compute = cast("_EdgeBuildResult", result)[0]
            assert isinstance(compute, q_and_f._ComputeCE)
            captured.append(compute)
            return result

        return observed

    monkeypatch.setattr(
        q_and_f, "_get_compute_CE", inject_cross_read(q_and_f._get_compute_CE)
    )
    _fork_model(certainty_equivalent=None)
    assert captured
    compute = captured[0]
    extension = compute.edge_extension
    assert extension is not None
    assert len(extension.distances) == 2

    def evaluate(index: int | None) -> FloatND:
        read = (
            compute
            if index is None
            else functools.partial(extension.read_corner, index=index)
        )
        result = read(
            next_regime_to_V_arr=MappingProxyType(
                {"left": jnp.asarray([1.0, 2.0]), "right": jnp.asarray([1.0, 4.0])}
            ),
            zero=jnp.asarray(0.0),
            states_actions_params=MappingProxyType({"income": jnp.asarray(3.0)}),
        )
        assert isinstance(result, tuple)
        return result[0]

    # Literal two-node probabilities for the four joint edge/neighbour rows.
    np.testing.assert_allclose(
        [evaluate(index) for index in range(4)],
        [2.6, 1.55, 2.45, 1.4],
        rtol=10.0**-DECIMAL_PRECISION,
    )
    np.testing.assert_allclose(evaluate(None), 3.8, rtol=10.0**-DECIMAL_PRECISION)
