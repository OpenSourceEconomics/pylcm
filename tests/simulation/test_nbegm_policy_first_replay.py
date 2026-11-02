"""NB-EGM simulation is policy-first.

A regime solved by `NBEGM` with a state besides its liquid state publishes the
optimal consumption on its state grid every period. Simulation reads that policy
at each subject's realized state: linear in the liquid state, and, for a process
state between its nodes, linear in the node coordinate, the rule the process's
transition weights follow off node. The dense action-grid argmax (`decide()`,
the `"decision"` simulation program) then runs only for subjects whose read is
rejected, for example a liquid state outside the published grid.
"""

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest
from pandas.testing import assert_frame_equal

from _lcm.egm.published_policy import NBEGMGridPolicy
from lcm import (
    ConsumptionSavingsRegime,
    DiscreteGrid,
    ExecutionConfig,
    InitialNodes,
    LinSpacedGrid,
    LiquidMargin,
    Model,
    Periods,
    Regime,
    categorical,
    fixed_transition,
)
from lcm.persistence import load_solution, save_solution
from lcm.solver_api import SIMULATION_POLICY, PersistencePolicy, ResultRetention
from lcm.solvers import NBEGM
from lcm.typing import (
    ContinuousAction,
    ContinuousState,
    DiscreteAction,
    DiscreteState,
    FloatND,
    ScalarInt,
)
from tests.conftest import DECIMAL_PRECISION, invariance_tolerances
from tests.simulation.test_egm_policy_first_replay import (
    _NODE_SAVINGS_GRID,
    _SHOCK,
    _cost_resources,
    _decision_program_owners,
    _health_bequest,
    _health_resources,
    _inverse_log_marginal,
    _log_utility,
    _node_bequest,
    _node_next_wealth,
    _node_savings,
    _NodeHealth,
    _NodeRegimeId,
)

pytestmark = pytest.mark.slow

_WEALTH_GRID = LinSpacedGrid(start=1.0, stop=5.0, n_points=5)
_WORK_INCOME = 1.5
_WORK_DISUTILITY = 0.4


@categorical(ordered=False)
class _Work:
    no: ScalarInt
    yes: ScalarInt


def _work_resources(
    *, wealth: ContinuousState, health: DiscreteState, work: DiscreteAction
) -> FloatND:
    return wealth + 2.0 + 0.5 * health + _WORK_INCOME * work


def _work_utility(*, consumption: ContinuousAction, work: DiscreteAction) -> FloatND:
    return jnp.log(consumption) - _WORK_DISUTILITY * work


def _nbegm_model(
    *, ride: str, execution_config: ExecutionConfig | None = None
) -> Model:
    """Two-period NB-EGM source with one ride-along state besides wealth.

    `ride="health"` adds a discrete state raising resources by `0.5 * health`;
    `ride="work"` adds to that a work choice raising resources by `1.5` at a
    utility cost of `0.4`; `ride="shock"` adds an IID process state subtracting
    `0.5 * shock`. With log utility, log bequest utility and no discounting the
    optimum given the work choice is `R / 2`.
    """
    actions: dict[str, LinSpacedGrid | DiscreteGrid] = {
        "consumption": LinSpacedGrid(start=0.5, stop=6.0, n_points=12)
    }
    utility = _log_utility
    if ride in {"health", "work"}:
        states = {
            "wealth": _WEALTH_GRID,
            "health": DiscreteGrid(category_class=_NodeHealth),
        }
        resources = _health_resources
        done_states = {"wealth": _NODE_SAVINGS_GRID, "health": states["health"]}
        transitions = {
            "wealth": {"done": _node_next_wealth},
            "health": {"done": fixed_transition("health")},
        }
        bequest = _health_bequest
        if ride == "work":
            actions["work"] = DiscreteGrid(category_class=_Work)
            resources = _work_resources
            utility = _work_utility
    else:
        states = {"wealth": _WEALTH_GRID, "shock": _SHOCK}
        resources = _cost_resources
        done_states = {"wealth": _NODE_SAVINGS_GRID}
        transitions = {"wealth": {"done": _node_next_wealth}}
        bequest = _node_bequest
    alive = ConsumptionSavingsRegime(
        states=states,
        actions=actions,
        functions={
            "utility": utility,
            "resources": resources,
            "inverse_marginal_utility": _inverse_log_marginal,
            "savings": _node_savings,
        },
        state_transitions=transitions,
        liquid=LiquidMargin(
            state="wealth",
            action="consumption",
            resources="resources",
            post_decision_state="savings",
        ),
        solver=NBEGM(savings_grid=_NODE_SAVINGS_GRID),
    )
    return Model(
        regimes={
            "alive": alive,
            "done": Regime(states=done_states, functions={"utility": bequest}),
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


def _initial_conditions(
    *, ride: str, n_subjects: int, wealth: np.ndarray | None = None
) -> dict[str, jnp.ndarray]:
    rng = np.random.default_rng(0)
    if wealth is None:
        wealth = rng.uniform(1.0, 5.0, size=n_subjects)
    conditions = {
        "wealth": jnp.asarray(wealth),
        "period": jnp.zeros(wealth.shape, dtype=jnp.int32),
        "regime_id": jnp.zeros(wealth.shape, dtype=jnp.int32),
    }
    if ride in {"health", "work"}:
        conditions["health"] = jnp.asarray(
            rng.integers(0, 2, size=wealth.shape), dtype=jnp.int32
        )
    else:
        conditions["shock"] = jnp.asarray(rng.standard_normal(wealth.shape))
    return conditions


def _simulate(*, model: Model, ride: str, n_subjects: int = 200, **kwargs):
    """Panel of a simulate run, sorted by subject and period."""
    frame = model.simulate(
        params={},
        initial_conditions=_initial_conditions(ride=ride, n_subjects=n_subjects),
        seed=0,
        log_level="off",
        **kwargs,
    ).to_dataframe(use_categorical_codes=True)
    return frame.sort_values(["subject_id", "period"]).reset_index(drop=True)


def _resources(*, first, ride: str) -> np.ndarray:
    wealth = first["wealth"].to_numpy(dtype=float)
    if ride == "health":
        return wealth + 2.0 + 0.5 * first["health"].to_numpy(dtype=float)
    return wealth + 2.0 - 0.5 * first["shock"].to_numpy(dtype=float)


@pytest.mark.parametrize("ride", ["health", "work", "shock"])
def test_default_nbegm_simulate_does_not_run_the_grid_argmax(
    *, ride: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Subjects inside the published state grid never reach the decision program."""
    model = _nbegm_model(ride=ride)
    executed = _decision_program_owners(monkeypatch=monkeypatch)

    _simulate(model=model, ride=ride)

    assert "alive" not in executed, executed


def test_a_subject_off_the_published_grid_takes_the_grid_argmax(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Positive control: a liquid state beyond the state grid runs the fallback."""
    model = _nbegm_model(ride="health")
    executed = _decision_program_owners(monkeypatch=monkeypatch)
    wealth = np.array([1.5, 3.25, 9.0])

    model.simulate(
        params={},
        initial_conditions=_initial_conditions(
            ride="health", n_subjects=3, wealth=wealth
        ),
        seed=0,
        log_level="off",
    )

    assert "alive" in executed, executed


def test_nbegm_reads_the_euler_policy_between_wealth_nodes() -> None:
    """Consumption is the continuous optimum `R / 2`, and savings carry it forward.

    With a discrete ride-along state the solve publishes `R / 2` at every
    state-grid node, so the linear read in wealth is exact between nodes. Next
    period's wealth is the savings of the same consumption the subject reports.
    """
    frame = _simulate(model=_nbegm_model(ride="health"), ride="health")
    first = frame[frame["period"] == 0]
    second = frame[frame["period"] == 1]
    resources = _resources(first=first, ride="health")
    consumption = first["consumption"].to_numpy(dtype=float)

    assert np.any(
        np.abs(np.remainder(first["wealth"].to_numpy(dtype=float), 1.0)) > 0.1
    )
    np.testing.assert_array_almost_equal(
        consumption, resources / 2.0, decimal=DECIMAL_PRECISION
    )
    np.testing.assert_array_almost_equal(
        second["wealth"].to_numpy(dtype=float),
        resources - consumption,
        decimal=DECIMAL_PRECISION,
    )


def test_work_choice_follows_the_larger_published_branch_value() -> None:
    """The branch whose published value reads larger wins, with its own policy.

    The solve publishes each work choice's conditional value and consumption on
    the state grid. Each subject reads both values linearly in wealth at its
    health row, takes the larger, and consumes that branch's policy read the same
    way. Both choices occur in the panel, and the budget is the chosen branch's.
    """
    model = _nbegm_model(ride="work")
    solution = model.solve(params={}, log_level="off")
    (policy,) = solution.replay_artifacts.values()
    assert isinstance(policy, NBEGMGridPolicy)
    assert policy.state_names == ("health", "wealth")
    assert policy.branch_value is not None
    assert policy.branch_inner_action is not None
    frame = _simulate(model=model, ride="work", solution=solution)
    first = frame[frame["period"] == 0]
    second = frame[frame["period"] == 1]

    wealth_nodes = np.asarray(_WEALTH_GRID.to_jax(), dtype=float)
    wealth = first["wealth"].to_numpy(dtype=float)
    health = first["health"].to_numpy()

    def read(bank: FloatND) -> np.ndarray:
        return np.stack(
            [
                np.array(
                    [
                        np.interp(at, wealth_nodes, branch[row])
                        for at, row in zip(wealth, health, strict=True)
                    ]
                )
                for branch in np.asarray(bank, dtype=float)
            ]
        )

    work = first["work"].to_numpy()
    expected_work = np.argmax(read(policy.branch_value), axis=0)
    expected_consumption = read(policy.branch_inner_action)[
        expected_work, np.arange(work.size)
    ]
    consumption = first["consumption"].to_numpy(dtype=float)

    assert set(np.unique(work)) == {0, 1}
    np.testing.assert_array_equal(work, expected_work)
    np.testing.assert_array_almost_equal(
        consumption, expected_consumption, decimal=DECIMAL_PRECISION
    )
    np.testing.assert_array_almost_equal(
        second["wealth"].to_numpy(dtype=float),
        _resources(first=first, ride="health") + _WORK_INCOME * work - consumption,
        decimal=DECIMAL_PRECISION,
    )


def test_lookup_policy_on_nbegm_returns_the_simulated_decision() -> None:
    """`lookup_policy` answers at simulated states with `simulate`'s decision."""
    model = _nbegm_model(ride="work")
    solution = model.solve(params={}, log_level="off")
    frame = _simulate(model=model, ride="work", solution=solution)
    first = frame[frame["period"] == 0]

    lookup = model.lookup_policy(
        params={},
        solution=solution,
        period=0,
        regime_name="alive",
        states={
            "wealth": jnp.asarray(first["wealth"].to_numpy()),
            "health": jnp.asarray(first["health"].to_numpy(), dtype=jnp.int32),
        },
        use_categorical_codes=True,
    )

    assert lookup.actions is not None
    np.testing.assert_array_equal(
        np.asarray(lookup.actions["work"]), first["work"].to_numpy()
    )
    for got, expected in (
        (lookup.actions["consumption"], first["consumption"]),
        (lookup.value, first["value"]),
    ):
        reference = expected.to_numpy(dtype=float)
        rtol, atol = invariance_tolerances(reference)
        np.testing.assert_allclose(np.asarray(got), reference, rtol=rtol, atol=atol)


def test_off_node_draw_reads_the_bilinear_blend_of_the_published_policy() -> None:
    """An off-node draw blends its two bracketing node policies, each linear in wealth.

    The published policy holds one consumption per (shock node, wealth node).
    A subject at wealth coordinate `i + u` and shock coordinate `j + v` reads
    `sum (1 - u | u) (1 - v | v) c[j | j + 1, i | i + 1]`; the shock coordinate
    extrapolates along the edge segment beyond the outer nodes, as the transition
    weights of an off-node process value do. Next period's wealth is the savings
    of the same consumption the subject reports.
    """
    model = _nbegm_model(ride="shock")
    solution = model.solve(params={}, log_level="off")
    (policy,) = solution.replay_artifacts.values()
    assert isinstance(policy, NBEGMGridPolicy)
    assert policy.state_names == ("shock", "wealth")
    frame = _simulate(model=model, ride="shock", solution=solution)
    first = frame[frame["period"] == 0]
    second = frame[frame["period"] == 1]

    published = np.asarray(policy.action, dtype=float)
    nodes = np.asarray(_SHOCK.get_gridpoints(), dtype=float)
    wealth_nodes = np.asarray(_WEALTH_GRID.to_jax(), dtype=float)
    shock = first["shock"].to_numpy(dtype=float)
    wealth = first["wealth"].to_numpy(dtype=float)
    j = np.clip(np.searchsorted(nodes, shock, side="right") - 1, 0, nodes.size - 2)
    v = (shock - nodes[j]) / (nodes[j + 1] - nodes[j])
    i = np.clip(
        np.searchsorted(wealth_nodes, wealth, side="right") - 1,
        0,
        wealth_nodes.size - 2,
    )
    u = (wealth - wealth_nodes[i]) / (wealth_nodes[i + 1] - wealth_nodes[i])
    expected = (1.0 - v) * (
        (1.0 - u) * published[j, i] + u * published[j, i + 1]
    ) + v * ((1.0 - u) * published[j + 1, i] + u * published[j + 1, i + 1])
    consumption = first["consumption"].to_numpy(dtype=float)

    assert np.any(np.min(np.abs(shock[:, None] - nodes[None, :]), axis=1) > 0.1)
    assert np.any(np.abs(shock) > nodes[-1])
    np.testing.assert_array_almost_equal(
        consumption, expected, decimal=DECIMAL_PRECISION
    )
    np.testing.assert_array_almost_equal(
        second["wealth"].to_numpy(dtype=float),
        _resources(first=first, ride="shock") - consumption,
        decimal=DECIMAL_PRECISION,
    )


def test_budgeted_nbegm_simulate_reads_the_policy_and_matches_unbudgeted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A device-memory budget admits the NB-EGM policy read; results agree."""
    unbudgeted = _simulate(
        model=_nbegm_model(
            ride="shock", execution_config=ExecutionConfig(device_memory_bytes=None)
        ),
        ride="shock",
    )
    budgeted_model = _nbegm_model(
        ride="shock", execution_config=ExecutionConfig(device_memory_bytes=2**30)
    )
    executed = _decision_program_owners(monkeypatch=monkeypatch)

    budgeted = _simulate(model=budgeted_model, ride="shock")

    assert "alive" not in executed, executed
    for column in ("consumption", "value", "wealth", "shock"):
        expected = unbudgeted[column].to_numpy(dtype=float)
        rtol, atol = invariance_tolerances(expected)
        np.testing.assert_allclose(
            budgeted[column].to_numpy(dtype=float), expected, rtol=rtol, atol=atol
        )


def test_saved_nbegm_policy_simulates_in_a_fresh_model(
    *, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A saved NB-EGM policy is model-verifiable and replays after loading.

    The model that loads the archive has never solved, so its declarations alone
    reconstruct the policy's layout.
    """
    source_model = _nbegm_model(ride="shock")
    solution = source_model.solve(
        params={},
        log_level="off",
        retention=ResultRetention.ALL_PERSISTABLE_ARTIFACTS,
    )
    policy_refs = tuple(
        ref for ref in solution.replay_artifacts if ref.key == SIMULATION_POLICY
    )
    descriptors = solution.metadata.artifact_descriptors
    assert policy_refs
    assert all(
        type(solution.replay_artifacts[ref]) is NBEGMGridPolicy for ref in policy_refs
    )
    assert all(
        descriptors[ref].persistence is PersistencePolicy.MODEL_VERIFIABLE
        for ref in policy_refs
    )
    expected = _simulate(model=source_model, ride="shock", solution=solution)

    restored = load_solution(
        path=save_solution(solution=solution, path=tmp_path / "nbegm.lcm")
    )
    fresh_model = _nbegm_model(ride="shock")
    executed = _decision_program_owners(monkeypatch=monkeypatch)
    actual = _simulate(model=fresh_model, ride="shock", solution=restored)

    assert "alive" not in executed, executed
    assert_frame_equal(actual, expected)
