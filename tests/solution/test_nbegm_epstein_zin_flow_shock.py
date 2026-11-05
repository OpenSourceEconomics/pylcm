"""NBEGM matches the brute oracle when an IID income shock meets EZ preferences.

An IID node that shifts income gives every node its own carry row. Under a
nonlinear certainty equivalent the power mean must transform every node's value
before the lottery sum (`E[g(V)] != g(E[V])`), so the NBEGM read keeps the node
axis and transforms per node, whether the nodes are reduced in one fused pass or
in planner blocks.
"""

import jax.numpy as jnp
import numpy as np

from lcm import (
    AgeGrid,
    AgeRange,
    CESAggregator,
    DeterministicTransition,
    ExecutionConfig,
    IrregSpacedGrid,
    LinSpacedGrid,
    Model,
    NormalIIDProcess,
    PowerMean,
    Regime,
    Transition,
    categorical,
)
from lcm.consumption_savings_regime import ConsumptionSavingsRegime, LiquidMargin
from lcm.solvers import NBEGM, STOCHASTIC_NODE_AXIS, GridSearch, OneMarginSolver
from lcm.typing import ContinuousAction, ContinuousState, FloatND, ScalarInt

_N_PERIODS = 3
_RETURN = 0.03
_BASE_INCOME = 1.0

_LIQUID_GRID = LinSpacedGrid(start=0.5, stop=20.0, n_points=15)
# The oracle's action grid contains every liquid point's exact cash-on-hand at
# the zero shock node:
# at a binding borrowing constraint the optimum is the corner `c = coh`, where
# the value is first-order in the action-grid gap — a dense grid alone leaves
# the brute reference visibly below the true corner value.
_CONSUMPTION_GRID = IrregSpacedGrid(
    points=np.array(
        tuple(
            np.unique(
                np.concatenate(
                    [
                        np.linspace(0.1, 21.0, 160),
                        np.linspace(0.5, 20.0, 15) + _BASE_INCOME,
                    ]
                )
            ).tolist()
        )
    )
)
_SAVINGS_GRID = LinSpacedGrid(start=0.0, stop=18.0, n_points=80)
_HEALTH = NormalIIDProcess(n_points=5, gauss_hermite=True)
_INCOME_PER_SHOCK = 0.25


@categorical(ordered=False)
class _RegimeId:
    alive: ScalarInt
    dead: ScalarInt


def _utility(consumption: ContinuousAction) -> FloatND:
    """Positive flow, single-power in consumption."""
    return consumption


def _resources(*, liquid: ContinuousState, health: ContinuousState) -> FloatND:
    return liquid + _BASE_INCOME + _INCOME_PER_SHOCK * health


def _savings(*, resources: FloatND, consumption: ContinuousAction) -> FloatND:
    return resources - consumption


def _feasible(*, resources: FloatND, consumption: ContinuousAction) -> FloatND:
    return consumption <= resources


def _next_liquid(savings: FloatND) -> ContinuousState:
    return (1.0 + _RETURN) * savings


def _terminal_value(liquid: ContinuousState) -> FloatND:
    return jnp.sqrt(liquid + 1.0)


def _next_regime(*, age: int, final_age_alive: float) -> ScalarInt:
    return jnp.where(age >= final_age_alive, _RegimeId.dead, _RegimeId.alive)


def _build_model(
    *,
    solver: OneMarginSolver | GridSearch,
    execution_config: ExecutionConfig = ExecutionConfig(),  # noqa: B008
) -> Model:
    final_age_alive = float(20 + (_N_PERIODS - 2) * 5)
    alive_law = DeterministicTransition(func=_next_regime)
    alive = ConsumptionSavingsRegime(
        states={"liquid": _LIQUID_GRID, "health": _HEALTH},
        state_transitions={"liquid": _next_liquid},
        actions={"consumption": _CONSUMPTION_GRID},
        functions={
            "utility": _utility,
            "resources": _resources,
            "savings": _savings,
        },
        constraints=(
            {} if isinstance(solver, OneMarginSolver) else {"feasible": _feasible}
        ),
        koopmans_aggregator=CESAggregator(),
        certainty_equivalent=PowerMean(),
        solver=solver,
        liquid=LiquidMargin(
            state="liquid",
            action="consumption",
            resources="resources",
            post_decision_state="savings",
        ),
    )
    dead = Regime(
        # Starts at zero so the terminal carry covers the whole attainable
        # savings range: the corner `s = 0` reads the terminal value exactly
        # instead of the edge bracket's secant.
        states={"liquid": LinSpacedGrid(start=0.0, stop=20.0, n_points=15)},
        functions={"utility": _terminal_value},
    )
    return Model(
        execution_config=execution_config,
        regimes={"alive": alive, "dead": dead},
        regime_id_class=_RegimeId,
        ages=AgeGrid(start=20, inclusive_stop=20 + (_N_PERIODS - 1) * 5, step="5Y"),
        fixed_params={"final_age_alive": final_age_alive},
        initial_nodes={20: "alive"},
        edges={
            "alive": Transition(
                targets={
                    "alive": AgeRange(exclusive_stop=final_age_alive),
                    "dead": AgeRange(exclusive_stop=final_age_alive + 5),
                },
                law=alive_law,
            )
        },
    )


_PARAMS = {
    "alive": {
        "koopmans_aggregator": {
            "discount_factor": 0.5,
            "intertemporal_elasticity_of_substitution": 1.5,
        },
        "certainty_equivalent": {"risk_aversion": 8.0},
        "health": {"mu": 0.0, "sigma": 1.0},
    },
    "dead": {},
}


def test_nbegm_epstein_zin_flow_shock_matches_brute_force() -> None:
    """The joint CE over an income-shifting IID node matches the dense grid search."""
    nbegm = (
        _build_model(
            solver=NBEGM(
                savings_grid=_SAVINGS_GRID,
                envelope_arithmetic="ordinary",
            )
        )
        .solve(params=_PARAMS, log_level="debug")
        .values
    )
    brute = (
        _build_model(solver=GridSearch())
        .solve(params=_PARAMS, log_level="debug")
        .values
    )
    for period in (0, 1):
        nbegm_V = np.asarray(nbegm[period]["alive"])
        brute_V = np.asarray(brute[period]["alive"])
        assert not np.isnan(nbegm_V).any(), f"period {period}"
        np.testing.assert_allclose(nbegm_V, brute_V, rtol=2e-2, atol=1e-2)


def test_stochastic_node_batching_matches_the_fused_expectation() -> None:
    """A positive stochastic-node planner width reproduces the fused EZ solve.

    The anchored transform partials are additive across stochastic-node
    blocks: each block reduces to `(a, S~, b, T~)`, blocks combine by
    re-anchored summation, and one inversion follows the full lottery — so
    the block-scan accumulation equals the single fused reduction up to
    floating-point reassociation.
    """
    fused = (
        _build_model(
            solver=NBEGM(
                savings_grid=_SAVINGS_GRID,
                envelope_arithmetic="ordinary",
            )
        )
        .solve(params=_PARAMS, log_level="debug")
        .values
    )
    batched = (
        _build_model(
            solver=NBEGM(
                savings_grid=_SAVINGS_GRID,
                envelope_arithmetic="ordinary",
            ),
            execution_config=ExecutionConfig(axis_widths={STOCHASTIC_NODE_AXIS: 2}),
        )
        .solve(params=_PARAMS, log_level="debug")
        .values
    )
    for period in (0, 1):
        batched_arr = np.asarray(batched[period]["alive"])
        fused_arr = np.asarray(fused[period]["alive"])
        # Reassociation noise scales with the active float dtype's precision.
        rtol = 64.0 * float(np.finfo(batched_arr.dtype).eps)
        np.testing.assert_allclose(batched_arr, fused_arr, rtol=rtol)
