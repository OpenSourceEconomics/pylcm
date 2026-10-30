"""An EGM regime refuses a process entering utility unless it simulates on nodes.

Simulation reads an EGM policy at a process value between nodes by blending the two
bracketing node rows, which is exact only for a process that shifts resources. A
process declared with `simulate_draws="nodes"` is only ever drawn at its nodes, so the
read takes one node row with coefficient exactly one.
"""

from typing import Literal

import jax.numpy as jnp
import numpy as np
import pytest

from lcm import (
    ConsumptionSavingsRegime,
    InitialNodes,
    IrregSpacedGrid,
    LinSpacedGrid,
    LiquidMargin,
    Model,
    NormalIIDProcess,
    Periods,
    Regime,
    RouwenhorstAR1Process,
    TauchenAR1Process,
    categorical,
)
from lcm.exceptions import ModelInitializationError
from lcm.solvers import DCEGM, NBEGM
from lcm.typing import ContinuousAction, ContinuousState, FloatND, ScalarInt

_MARGIN = LiquidMargin(
    state="wealth",
    action="consumption",
    resources="resources",
    post_decision_state="savings",
)
_SAVINGS_GRID = IrregSpacedGrid(points=np.linspace(0.5, 6.0, 12))
type _Process = NormalIIDProcess | RouwenhorstAR1Process | TauchenAR1Process


@categorical(ordered=False)
class _RegimeId:
    alive: ScalarInt
    done: ScalarInt


def _shocked_utility(
    *, consumption: ContinuousAction, shock: ContinuousState
) -> FloatND:
    return (1.0 + 0.2 * shock) * jnp.log(consumption)


def _shocked_inverse_marginal_utility(
    *, marginal_continuation: FloatND, shock: ContinuousState
) -> FloatND:
    return (1.0 + 0.2 * shock) / marginal_continuation


def _log_utility(*, consumption: ContinuousAction) -> FloatND:
    return jnp.log(consumption)


def _inverse_marginal_utility(*, marginal_continuation: FloatND) -> FloatND:
    return 1.0 / marginal_continuation


def _resources(*, wealth: ContinuousState) -> FloatND:
    return wealth + 1.5


def _shocked_resources(*, wealth: ContinuousState, shock: ContinuousState) -> FloatND:
    return wealth + 1.5 + 0.2 * shock


def _savings(*, resources: FloatND, consumption: ContinuousAction) -> FloatND:
    return resources - consumption


def _next_wealth(*, savings: FloatND) -> ContinuousState:
    return savings


def _log_wealth(*, wealth: ContinuousState) -> FloatND:
    return jnp.log(wealth)


def _ar1(*, simulate_draws: Literal["continuous", "nodes"]) -> TauchenAR1Process:
    return TauchenAR1Process(
        n_points=3,
        gauss_hermite=False,
        rho=0.8,
        sigma=0.5,
        mu=0.0,
        n_std=2,
        simulate_draws=simulate_draws,
    )


def _model(*, solver: str, shock: _Process, enters: str) -> Model:
    """Three-period savings model whose process state enters utility or resources."""
    in_utility = enters == "utility"
    alive = ConsumptionSavingsRegime(
        states={
            "wealth": LinSpacedGrid(start=1.0, stop=5.0, n_points=5),
            "shock": shock,
        },
        actions={"consumption": LinSpacedGrid(start=0.5, stop=6.0, n_points=12)},
        functions={
            "utility": _shocked_utility if in_utility else _log_utility,
            "inverse_marginal_utility": (
                _shocked_inverse_marginal_utility
                if in_utility
                else _inverse_marginal_utility
            ),
            "resources": _resources if in_utility else _shocked_resources,
            "savings": _savings,
        },
        state_transitions={"wealth": dict.fromkeys(("alive", "done"), _next_wealth)},
        liquid=_MARGIN,
        solver=(DCEGM if solver == "dcegm" else NBEGM)(savings_grid=_SAVINGS_GRID),
    )
    done = Regime(states={"wealth": _SAVINGS_GRID}, functions={"utility": _log_wealth})
    return Model(
        regimes={"alive": alive, "done": done},
        n_periods=3,
        initial_nodes=InitialNodes(by_period={0: "alive"}),
        edges={"alive": {"alive": Periods(values=(0,)), "done": Periods(values=(1,))}},
        regime_id_class=_RegimeId,
        fixed_params={"discount_factor": 0.95},
    )


_REFUSAL = (
    r"(?s){solver} regime 'alive' reads the process state\(s\) \['shock'\] other "
    r"than through resources \('resources'\).*Move the process into resources "
    r"\('resources'\), declare it with `simulate_draws=\"nodes\"`, or use "
    r"GridSearch for this regime\."
)


@pytest.mark.parametrize(
    "shock",
    [
        _ar1(simulate_draws="continuous"),
        RouwenhorstAR1Process(n_points=3, rho=0.8, sigma=0.5, mu=0.0),
        TauchenAR1Process.on_nodes(
            nodes=np.asarray([-1.0, 0.0, 1.0]), rho=0.8, sigma=0.5
        ),
    ],
    ids=["tauchen", "rouwenhorst", "on_nodes"],
)
@pytest.mark.parametrize("solver", ["dcegm", "nbegm"])
def test_ar1_process_entering_utility_with_continuous_draws_is_refused(
    *, solver: str, shock: _Process
) -> None:
    """A Markov process drawn off its nodes may not enter EGM utility."""
    with pytest.raises(
        ModelInitializationError, match=_REFUSAL.format(solver=solver.upper())
    ):
        _model(solver=solver, shock=shock, enters="utility")


@pytest.mark.parametrize(
    "shock",
    [
        _ar1(simulate_draws="nodes"),
        NormalIIDProcess(
            n_points=3, gauss_hermite=True, mu=0.0, sigma=0.5, simulate_draws="nodes"
        ),
    ],
    ids=["ar1", "iid"],
)
@pytest.mark.parametrize("solver", ["dcegm", "nbegm"])
def test_process_entering_utility_on_node_draws_builds_and_simulates(
    *, solver: str, shock: _Process
) -> None:
    """A process simulated on its nodes is read at one node row, exactly.

    Every simulated value is a solve node, bit for bit, so the policy read's
    bracket gives that node's row coefficient exactly one and no blend.
    """
    model = _model(solver=solver, shock=shock, enters="utility")
    assert model._regimes["alive"].simulation.egm_policy_read is not None
    nodes = np.asarray(model._regimes["alive"].simulation.grids["shock"].to_jax())
    panel = model.simulate(
        params={},
        initial_conditions={
            "wealth": jnp.linspace(1.5, 4.5, 20),
            "shock": jnp.asarray(np.resize(nodes, 20)),
            "period": jnp.zeros(20, dtype=jnp.int32),
            "regime_id": jnp.zeros(20, dtype=jnp.int32),
        },
        seed=0,
        log_level="off",
    ).to_dataframe()
    draws = panel.loc[panel["regime_name"] == "alive", "shock"].to_numpy()
    assert len(draws) > 20
    assert np.isin(draws, nodes).all()
    lo = np.clip(np.searchsorted(nodes, draws, side="right") - 1, 0, len(nodes) - 2)
    weights = (draws - nodes[lo]) / (nodes[lo + 1] - nodes[lo])
    assert np.isin(weights, [0.0, 1.0]).all()
    assert np.isfinite(panel["consumption"].dropna().to_numpy()).all()


@pytest.mark.parametrize(
    "shock",
    [
        _ar1(simulate_draws="continuous"),
        NormalIIDProcess(n_points=3, gauss_hermite=True, mu=0.0, sigma=0.5),
    ],
    ids=["ar1", "iid"],
)
@pytest.mark.parametrize("solver", ["dcegm", "nbegm"])
def test_process_entering_resources_only_builds(
    *, solver: str, shock: _Process
) -> None:
    """A process that shifts resources only keeps the exact blended read."""
    model = _model(solver=solver, shock=shock, enters="resources")
    assert model._regimes["alive"].simulation.egm_policy_read is not None
