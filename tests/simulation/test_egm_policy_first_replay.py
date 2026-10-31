"""Plain DC-EGM simulation is policy-first.

The solve publishes each branch's conditional value and policy rows on the
endogenous resources grid. Simulation picks the discrete branch by the largest
interpolated conditional value and reads the winner's policy row at the
subject's resources, so the dense action-grid argmax (`decide()`, the
`"decision"` simulation program) is neither traced nor executed while every
subject's read is accepted. `DCEGM(replay="grid")` keeps the grid argmax and
serves as the reference route.
"""

import dataclasses
from collections.abc import Mapping

import jax.numpy as jnp
import numpy as np
import pytest

import _lcm.simulation.simulate as simulation
import lcm_examples.iskhakov_et_al_2017 as iskhakov
from lcm import (
    AgeGrid,
    ConsumptionSavingsRegime,
    DiscreteGrid,
    ExecutionConfig,
    InitialNodes,
    IrregSpacedGrid,
    LinSpacedGrid,
    LiquidMargin,
    Model,
    NormalIIDProcess,
    Periods,
    Regime,
    categorical,
    fixed_transition,
)
from lcm.solvers import DCEGM, RFCEnvelope
from lcm.typing import (
    ContinuousAction,
    ContinuousState,
    DiscreteAction,
    DiscreteState,
    FloatND,
    ScalarInt,
)
from lcm_examples.iskhakov_et_al_2017 import CONSUMPTION_GRID, dead, get_edges
from tests.conftest import DECIMAL_PRECISION, invariance_tolerances
from tests.test_models.deterministic import base
from tests.test_models.deterministic.dcegm_variants import (
    DCEGM_SOLVER,
    dcegm_retirement,
    dcegm_working_life,
    get_full_params,
)

pytestmark = pytest.mark.slow

_N_PERIODS = 4
_EGM_REGIMES = ("working_life", "retirement")


def _model(*, replay: str, execution_config: ExecutionConfig | None = None) -> Model:
    solver = dataclasses.replace(DCEGM_SOLVER, replay=replay)
    ages = AgeGrid(start=40, inclusive_stop=40 + (_N_PERIODS - 1) * 10, step="10Y")
    return Model(
        edges=get_edges(ages=ages),
        regimes={
            "working_life": dcegm_working_life.replace(solver=solver),
            "retirement": dcegm_retirement.replace(solver=solver),
            "dead": dead,
        },
        ages=ages,
        regime_id_class=base.RegimeId,
        initial_nodes=((40, "working_life"), (40, "retirement")),
        execution_config=ExecutionConfig()
        if execution_config is None
        else execution_config,
    )


def _initial_conditions(*, wealth: np.ndarray) -> dict[str, jnp.ndarray]:
    return {
        "wealth": jnp.asarray(wealth),
        "age": jnp.full(wealth.shape, 40.0),
        "regime_id": jnp.full(wealth.shape, base.RegimeId.working_life),
    }


def _decision_program_owners(*, monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Record the regime of every executed `"decision"` simulation program."""
    running: list[str] = []
    executed: list[str] = []
    run_regime = simulation._simulate_regime_in_period
    run_program = simulation.execute_simulation_program

    def record_regime(**kwargs):
        running.append(kwargs["regime_name"])
        try:
            return run_regime(**kwargs)
        finally:
            running.pop()

    def record_program(*, family: str, **kwargs):
        if family == "decision":
            executed.append(running[-1])
        return run_program(family=family, **kwargs)

    monkeypatch.setattr(simulation, "_simulate_regime_in_period", record_regime)
    monkeypatch.setattr(simulation, "execute_simulation_program", record_program)
    return executed


def _simulate(*, model: Model, wealth: np.ndarray):
    return model.simulate(
        params=get_full_params(n_periods=_N_PERIODS),
        initial_conditions=_initial_conditions(wealth=wealth),
        seed=0,
        log_level="off",
    ).to_dataframe(use_categorical_codes=True)


_IN_SUPPORT_WEALTH = np.linspace(20.5, 180.5, 9)


def test_default_dcegm_simulate_does_not_run_the_grid_argmax(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """In-support subjects never reach the dense decision program of an EGM regime."""
    model = _model(replay="candidate")
    executed = _decision_program_owners(monkeypatch=monkeypatch)

    _simulate(model=model, wealth=_IN_SUPPORT_WEALTH)

    assert not set(executed) & set(_EGM_REGIMES), executed


def test_grid_replay_runs_the_grid_argmax(monkeypatch: pytest.MonkeyPatch) -> None:
    """Positive control: the spy sees the decision program on the grid route."""
    model = _model(replay="grid")
    executed = _decision_program_owners(monkeypatch=monkeypatch)

    _simulate(model=model, wealth=_IN_SUPPORT_WEALTH)

    assert set(_EGM_REGIMES) <= set(executed), executed


def _consumption_step() -> float:
    nodes = np.asarray(CONSUMPTION_GRID.to_jax())
    return float(np.max(np.diff(nodes)))


def _per_period_means(frame) -> Mapping[int, tuple[float, float]]:
    alive = frame[frame["regime_name"].isin(_EGM_REGIMES)]
    grouped = alive.groupby("period")
    return {
        int(period): (float(group["consumption"].mean()), float(group["value"].mean()))
        for period, group in grouped
    }


def test_budgeted_simulate_reads_the_policy_first_and_matches_unbudgeted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A device-memory budget admits the policy-first route and changes no result.

    The last subject starts beyond the published rows, so its period-0 read is
    rejected and the budgeted grid-argmax fallback runs as well.
    """
    wealth = np.append(_IN_SUPPORT_WEALTH, 1000.0)
    unbudgeted = _simulate(
        model=_model(
            replay="candidate",
            execution_config=ExecutionConfig(device_memory_bytes=None),
        ),
        wealth=wealth,
    )
    budgeted_model = _model(
        replay="candidate",
        execution_config=ExecutionConfig(device_memory_bytes=2**30),
    )
    executed = _decision_program_owners(monkeypatch=monkeypatch)

    budgeted = _simulate(model=budgeted_model, wealth=wealth)

    assert "working_life" in executed
    assert list(budgeted.columns) == list(unbudgeted.columns)
    for column in budgeted.columns:
        got, expected = budgeted[column].to_numpy(), unbudgeted[column].to_numpy()
        if np.issubdtype(expected.dtype, np.floating):
            rtol, atol = invariance_tolerances(expected)
            np.testing.assert_allclose(got, expected, rtol=rtol, atol=atol)
        else:
            np.testing.assert_array_equal(got, expected)


def test_policy_first_tracks_the_grid_route_within_the_action_grid_step() -> None:
    """Mean consumption and value agree with the grid route up to grid error.

    The grid route picks consumption on `CONSUMPTION_GRID`, so each of its
    choices is within one grid step `h` of the policy read (half a step from
    the continuous optimum, plus the read's own interpolation error). Wealth
    carries those differences forward one-for-one at zero interest, so after
    `t` periods the mean consumption gap is bounded by `(t + 1) * h`. The
    value gap is first-order in the action gap only through `u'(c) * h`
    along the path, and log utility at the sampled consumption keeps
    `u'(c) = 1 / c <= 1`, so the same `(t + 1) * h` bound applies to the
    mean value per period.
    """
    rng = np.random.default_rng(0)
    wealth = rng.uniform(20.0, 180.0, size=200)
    policy = _per_period_means(
        _simulate(model=_model(replay="candidate"), wealth=wealth)
    )
    grid = _per_period_means(_simulate(model=_model(replay="grid"), wealth=wealth))

    step = _consumption_step()
    assert policy.keys() == grid.keys()
    for period in policy:
        band = (period + 1) * step
        policy_c, policy_v = policy[period]
        grid_c, grid_v = grid[period]
        assert abs(policy_c - grid_c) <= band, (period, policy_c, grid_c, band)
        assert abs(policy_v - grid_v) <= band, (period, policy_v, grid_v, band)


def test_lookup_policy_on_default_dcegm_returns_the_simulated_decision() -> None:
    """`lookup_policy` answers on a default DC-EGM model with `simulate`'s choice.

    At every simulated working-life state, the looked-up labor supply equals the
    simulated one, and the looked-up consumption and value agree with the
    simulated ones up to the reassociation a different row count allows. The
    last subject starts beyond the published rows, so its period-0 read is
    rejected and both take the grid argmax there.
    """
    n_periods = 4
    model = iskhakov.get_dcegm_model(n_periods)
    params = iskhakov.get_params(n_periods=n_periods)
    solution = model.solve(params=params, log_level="off")
    wealth = np.append(np.linspace(5.0, 180.0, 12), 1000.0)
    frame = model.simulate(
        params=params,
        initial_conditions={
            "wealth": jnp.asarray(wealth),
            "age": jnp.full(wealth.shape, 40.0),
            "regime_id": jnp.full(wealth.shape, iskhakov.RegimeId.working_life),
        },
        solution=solution,
        seed=0,
        log_level="off",
    ).to_dataframe(use_categorical_codes=True)

    for period in (0, 1):
        rows = frame[
            (frame["period"] == period) & (frame["regime_name"] == "working_life")
        ]
        lookup = model.lookup_policy(
            params=params,
            solution=solution,
            period=period,
            regime_name="working_life",
            states={"wealth": jnp.asarray(rows["wealth"].to_numpy())},
            use_categorical_codes=True,
        )
        assert lookup.actions is not None
        np.testing.assert_array_equal(
            np.asarray(lookup.actions["labor_supply"]),
            rows["labor_supply"].to_numpy(),
        )
        for got, expected in (
            (lookup.actions["consumption"], rows["consumption"]),
            (lookup.value, rows["value"]),
        ):
            reference = expected.to_numpy()
            rtol, atol = invariance_tolerances(reference)
            np.testing.assert_allclose(np.asarray(got), reference, rtol=rtol, atol=atol)


# Node-state DC-EGM: an IID process declared as a state is a node-valued row axis
# of the published policy. Simulated draws are continuous, so they fall between
# nodes; the read blends the two bracketing node rows linearly in the process
# coordinate, which is how the transition weights of an off-node value are formed.
# The shock enters resources only (a shock entering utility is refused at build),
# so every node row is the same policy of resources and the blend is exact.


@categorical(ordered=False)
class _NodeRegimeId:
    alive: ScalarInt
    done: ScalarInt


@categorical(ordered=True)
class _NodeHealth:
    bad: ScalarInt
    good: ScalarInt


_NODE_SAVINGS_GRID = IrregSpacedGrid(points=np.linspace(0.5, 6.0, 12))
_SHOCK = NormalIIDProcess(n_points=5, gauss_hermite=True, mu=0.0, sigma=1.0)


def _cost_resources(*, wealth: ContinuousState, shock: ContinuousState) -> FloatND:
    return wealth + 2.0 - 0.5 * shock


def _health_resources(*, wealth: ContinuousState, health: DiscreteState) -> FloatND:
    return wealth + 2.0 + 0.5 * health


def _health_cost_resources(
    *, wealth: ContinuousState, shock: ContinuousState, health: DiscreteState
) -> FloatND:
    return wealth + 2.0 - 0.5 * shock + 0.5 * health


def _log_utility(*, consumption: ContinuousAction) -> FloatND:
    return jnp.log(consumption)


def _inverse_log_marginal(*, marginal_continuation: FloatND) -> FloatND:
    return 1.0 / marginal_continuation


def _node_savings(*, resources: FloatND, consumption: ContinuousAction) -> FloatND:
    return resources - consumption


def _node_next_wealth(*, savings: FloatND) -> ContinuousState:
    return savings


def _node_bequest(*, wealth: ContinuousState) -> FloatND:
    return jnp.log(wealth)


def _health_bequest(*, wealth: ContinuousState, health: DiscreteState) -> FloatND:
    # `health` rides along into the terminal without changing the bequest.
    return jnp.log(wealth) + 0.0 * health


def _node_state_model(
    *,
    with_health: bool = False,
    execution_config: ExecutionConfig | None = None,
) -> Model:
    """Two-period DC-EGM source whose IID cost shock is a process state.

    The shock subtracts a cost from resources, so every node row is the same
    function of resources. `with_health` adds an ordinary discrete state raising
    resources by `0.5 * health`, a second, per-subject row axis. With log bequest
    utility and no discounting the optimum is `R / 2`.
    """
    functions = {
        "utility": _log_utility,
        "resources": _health_cost_resources if with_health else _cost_resources,
        "inverse_marginal_utility": _inverse_log_marginal,
    }
    states: dict[str, LinSpacedGrid | NormalIIDProcess | DiscreteGrid] = {
        "wealth": LinSpacedGrid(start=1.0, stop=5.0, n_points=5),
        "shock": _SHOCK,
    }
    transitions = {"wealth": {"done": _node_next_wealth}}
    done_states: dict[str, IrregSpacedGrid | DiscreteGrid] = {
        "wealth": _NODE_SAVINGS_GRID
    }
    if with_health:
        states["health"] = done_states["health"] = DiscreteGrid(
            category_class=_NodeHealth
        )
        transitions["health"] = {"done": fixed_transition("health")}
    alive = ConsumptionSavingsRegime(
        states=states,
        actions={"consumption": LinSpacedGrid(start=0.5, stop=6.0, n_points=12)},
        functions={**functions, "savings": _node_savings},
        state_transitions=transitions,
        liquid=LiquidMargin(
            state="wealth",
            action="consumption",
            resources="resources",
            post_decision_state="savings",
        ),
        solver=DCEGM(savings_grid=_NODE_SAVINGS_GRID, envelope=RFCEnvelope()),
    )
    return Model(
        regimes={
            "alive": alive,
            "done": Regime(
                states=done_states,
                functions={
                    "utility": _health_bequest if with_health else _node_bequest
                },
            ),
        },
        n_periods=2,
        initial_nodes=InitialNodes(by_period={0: "alive"}),
        edges={"alive": {"done": Periods(values=(0,))}},
        regime_id_class=_NodeRegimeId,
        fixed_params={"discount_factor": 1.0},
        execution_config=ExecutionConfig()
        if execution_config is None
        else execution_config,
    )


def _simulate_node_state(*, model: Model, n_subjects: int = 200):
    """Period-0 rows of a panel whose shock draws are continuous (off node)."""
    rng = np.random.default_rng(0)
    initial_conditions = {
        "wealth": jnp.asarray(rng.uniform(1.0, 5.0, size=n_subjects)),
        "shock": jnp.asarray(rng.standard_normal(n_subjects)),
        "period": jnp.zeros(n_subjects, dtype=jnp.int32),
        "regime_id": jnp.zeros(n_subjects, dtype=jnp.int32),
    }
    if "health" in model.state_names(regime_name="alive"):
        initial_conditions["health"] = jnp.asarray(
            rng.integers(0, 2, size=n_subjects), dtype=jnp.int32
        )
    frame = model.simulate(
        params={},
        initial_conditions=initial_conditions,
        seed=0,
        log_level="off",
    ).to_dataframe(use_categorical_codes=True)
    frame = frame.sort_values(["subject_id", "period"])
    return frame[frame["period"] == 0]


def test_node_state_dcegm_simulate_does_not_run_the_grid_argmax(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Off-node process draws are decided from the node rows, not the grid argmax."""
    model = _node_state_model()
    executed = _decision_program_owners(monkeypatch=monkeypatch)

    _simulate_node_state(model=model)

    assert "alive" not in executed, executed


@pytest.mark.parametrize("with_health", [False, True])
def test_node_state_dcegm_reads_the_euler_policy_at_off_node_draws(
    with_health: bool,  # noqa: FBT001
) -> None:
    """Consumption is the continuous optimum `R / 2`, not an action-grid point.

    The cost shock only shifts resources, so every node row is the same Euler
    policy of resources, and the blend of the two bracketing node rows reads it
    exactly. A further discrete state selects its own rows per subject alongside
    the blend.
    """
    first = _simulate_node_state(model=_node_state_model(with_health=with_health))

    shock = first["shock"].to_numpy(dtype=float)
    resources = first["wealth"].to_numpy(dtype=float) + 2.0 - 0.5 * shock
    if with_health:
        resources += 0.5 * first["health"].to_numpy(dtype=float)
    assert np.any(np.abs(np.remainder(resources, 1.0)) > 0.1)
    np.testing.assert_array_almost_equal(
        first["consumption"].to_numpy(dtype=float),
        resources / 2.0,
        decimal=DECIMAL_PRECISION,
    )


def test_budgeted_node_state_simulate_reads_the_policy_and_matches_unbudgeted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A device-memory budget admits the node-state policy read; results agree."""
    unbudgeted = _simulate_node_state(
        model=_node_state_model(
            execution_config=ExecutionConfig(device_memory_bytes=None),
        )
    )
    budgeted_model = _node_state_model(
        execution_config=ExecutionConfig(device_memory_bytes=2**30)
    )
    executed = _decision_program_owners(monkeypatch=monkeypatch)

    budgeted = _simulate_node_state(model=budgeted_model)

    assert "alive" not in executed, executed
    for column in ("consumption", "value", "wealth", "shock"):
        expected = unbudgeted[column].to_numpy(dtype=float)
        rtol, atol = invariance_tolerances(expected)
        np.testing.assert_allclose(
            budgeted[column].to_numpy(dtype=float), expected, rtol=rtol, atol=atol
        )


@categorical(ordered=False)
class _Work:
    no: ScalarInt
    yes: ScalarInt


def _work_resources(*, wealth: ContinuousState, work: DiscreteAction) -> FloatND:
    return wealth - 1.0 + 2.0 * work


def _work_utility(*, consumption: ContinuousAction, work: DiscreteAction) -> FloatND:
    return jnp.log(consumption) - 0.1 * work


def _work_branch_model() -> Model:
    """Two-period DC-EGM source whose `no`-work branch can be unaffordable.

    Not working leaves `wealth - 1` resources, which for `wealth < 1.5` lies
    below the savings floor `0.5`: no consumption is feasible on that branch.
    Working adds `2`, so with log bequest utility and no discounting the optimum
    of a subject left with only the work branch is `R / 2`.
    """
    alive = ConsumptionSavingsRegime(
        states={"wealth": LinSpacedGrid(start=1.0, stop=5.0, n_points=5)},
        actions={
            "consumption": LinSpacedGrid(start=0.5, stop=6.0, n_points=12),
            "work": DiscreteGrid(category_class=_Work),
        },
        functions={
            "utility": _work_utility,
            "resources": _work_resources,
            "inverse_marginal_utility": _inverse_log_marginal,
            "savings": _node_savings,
        },
        state_transitions={"wealth": {"done": _node_next_wealth}},
        liquid=LiquidMargin(
            state="wealth",
            action="consumption",
            resources="resources",
            post_decision_state="savings",
        ),
        solver=DCEGM(savings_grid=_NODE_SAVINGS_GRID),
    )
    return Model(
        regimes={
            "alive": alive,
            "done": Regime(
                states={"wealth": _NODE_SAVINGS_GRID},
                functions={"utility": _node_bequest},
            ),
        },
        n_periods=2,
        initial_nodes=InitialNodes(by_period={0: "alive"}),
        edges={"alive": {"done": Periods(values=(0,))}},
        regime_id_class=_NodeRegimeId,
        fixed_params={"discount_factor": 1.0},
    )


def test_unaffordable_branch_does_not_send_subjects_to_the_grid_argmax(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A branch whose resources lie below the savings floor is dead, not unread.

    Its row read extrapolates below the row's support; that must not reject the
    subject, whose other branch decides it from the policy.
    """
    model = _work_branch_model()
    executed = _decision_program_owners(monkeypatch=monkeypatch)
    wealth = np.linspace(1.0, 5.0, 41)
    assert np.any(wealth - 1.0 < 0.5)

    frame = model.simulate(
        params={},
        initial_conditions={
            "wealth": jnp.asarray(wealth),
            "period": jnp.zeros(wealth.shape, dtype=jnp.int32),
            "regime_id": jnp.zeros(wealth.shape, dtype=jnp.int32),
        },
        seed=0,
        log_level="off",
    ).to_dataframe(use_categorical_codes=True)

    assert "alive" not in executed, executed
    first = frame[frame["period"] == 0].sort_values("subject_id")
    poor = first["wealth"].to_numpy(dtype=float) - 1.0 < 0.5
    assert np.all(first["work"].to_numpy()[poor] == _Work.yes)
    np.testing.assert_array_almost_equal(
        first["consumption"].to_numpy(dtype=float)[poor],
        (first["wealth"].to_numpy(dtype=float)[poor] + 1.0) / 2.0,
        decimal=DECIMAL_PRECISION,
    )
