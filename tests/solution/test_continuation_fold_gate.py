"""Spec for when the stochastic pre-fold may engage in the continuation read.

Folding a stochastic dimension replaces its carry rows by their
probability-weighted average before interpolation. That is a pure scheduling
change for the linear expected-utility read — expectation and linear reads
commute — so the folded and unfolded solves must agree to floating-point
reassociation. Under a nonlinear certainty equivalent the fold is not
available at all (`E[g(V)] != g(E[V])`): the read must keep the node axis and
transform every row, so the fold must never engage there.
"""

from unittest.mock import patch

import jax.numpy as jnp
import numpy as np

from _lcm.egm import continuation
from lcm import (
    NBEGM,
    AgeGrid,
    H_epstein_zin,
    LinSpacedGrid,
    Model,
    NormalIIDProcess,
    PowerMean,
    Regime,
    categorical,
)
from lcm.solvers import Solver
from lcm.typing import ContinuousAction, ContinuousState, FloatND, ScalarInt

_N_PERIODS = 3

_LIQUID_GRID = LinSpacedGrid(start=0.5, stop=20.0, n_points=12)
_CONSUMPTION_GRID = LinSpacedGrid(start=0.1, stop=15.0, n_points=40)
_SAVINGS_GRID = LinSpacedGrid(start=0.0, stop=18.0, n_points=50)
_HEALTH = NormalIIDProcess(n_points=5, gauss_hermite=True)


@categorical(ordered=False)
class _RegimeId:
    alive: ScalarInt
    dead: ScalarInt


def _utility(consumption: ContinuousAction, health: ContinuousState) -> FloatND:
    return consumption * jnp.exp(health)


def _resources(liquid: ContinuousState) -> FloatND:
    return liquid + 1.0


def _savings(resources: FloatND, consumption: ContinuousAction) -> FloatND:
    return resources - consumption


def _feasible(resources: FloatND, consumption: ContinuousAction) -> FloatND:
    return consumption <= resources


def _next_liquid(savings: FloatND) -> ContinuousState:
    return 1.03 * savings


def _terminal_value(liquid: ContinuousState) -> FloatND:
    return jnp.sqrt(liquid + 1.0)


def _next_regime(age: int, final_age_alive: float) -> ScalarInt:
    return jnp.where(age >= final_age_alive, _RegimeId.dead, _RegimeId.alive)


def _build_model(*, solver: Solver, epstein_zin: bool) -> Model:
    final_age_alive = float(20 + (_N_PERIODS - 2) * 5)
    functions = {
        "utility": _utility,
        "resources": _resources,
        "savings": _savings,
    }
    if epstein_zin:
        functions["H"] = H_epstein_zin
    alive = Regime(
        active=lambda age, n=final_age_alive: age <= n,
        states={"liquid": _LIQUID_GRID, "health": _HEALTH},
        state_transitions={"liquid": _next_liquid},
        actions={"consumption": _CONSUMPTION_GRID},
        transition=_next_regime,
        functions=functions,
        constraints={"feasible": _feasible},
        certainty_equivalent=PowerMean() if epstein_zin else None,
        solver=solver,
    )
    dead = Regime(
        transition=None,
        active=lambda age, n=final_age_alive: age > n,
        states={"liquid": _LIQUID_GRID},
        functions={"utility": _terminal_value},
    )
    return Model(
        regimes={"alive": alive, "dead": dead},
        regime_id_class=_RegimeId,
        ages=AgeGrid(start=20, stop=20 + (_N_PERIODS - 1) * 5, step="5Y"),
        fixed_params={"final_age_alive": final_age_alive},
    )


def _nbegm() -> NBEGM:
    return NBEGM(
        post_decision_function="savings",
        budget_target="resources",
        savings_grid=_SAVINGS_GRID,
        continuous_state="liquid",
    )


def _linear_params() -> dict:
    return {
        "alive": {"discount_factor": 0.9, "health": {"mu": 0.0, "sigma": 0.6}},
        "dead": {},
    }


def _ez_params() -> dict:
    return {
        "alive": {
            "H": {
                "discount_factor": 0.9,
                "intertemporal_elasticity_of_substitution": 1.5,
            },
            "certainty_equivalent": {"risk_aversion": 4.0},
            "health": {"mu": 0.0, "sigma": 0.6},
        },
        "dead": {},
    }


def test_fold_commutes_with_the_linear_expected_utility_read() -> None:
    """Folded and unfolded linear-EU solves agree to floating-point noise.

    The fold is a pure scheduling lever for the linear read: disabling it (an
    identity pass-through, so every node is read and summed individually)
    changes only the order of the floating-point reduction.
    """
    folded = _build_model(solver=_nbegm(), epstein_zin=False).solve(
        params=_linear_params(), log_level="off"
    )

    def identity_fold(*, read, carry, stochastic_node_values, weight_vecs):
        return read, carry, stochastic_node_values, weight_vecs

    with patch.object(continuation, "_fold_stochastic_dims", identity_fold):
        unfolded = _build_model(solver=_nbegm(), epstein_zin=False).solve(
            params=_linear_params(), log_level="off"
        )
    for period in (0, 1):
        folded_arr = np.asarray(folded[period]["alive"])
        unfolded_arr = np.asarray(unfolded[period]["alive"])
        # Reassociation noise scales with the active float dtype's precision.
        rtol = 64.0 * float(np.finfo(folded_arr.dtype).eps)
        np.testing.assert_allclose(folded_arr, unfolded_arr, rtol=rtol)


def test_fold_engages_for_linear_but_never_for_a_certainty_equivalent() -> None:
    """The fold runs on the linear read and never under Epstein-Zin.

    A nonlinear certainty equivalent must transform every lottery node before
    the sum, so the pre-fold — a linear expectation over the node rows — is
    structurally unavailable there.
    """
    calls: list[bool] = []
    original = continuation._fold_stochastic_dims

    def spy(**kwargs):
        calls.append(True)
        return original(**kwargs)

    with patch.object(continuation, "_fold_stochastic_dims", spy):
        _build_model(solver=_nbegm(), epstein_zin=False).solve(
            params=_linear_params(), log_level="off"
        )
        linear_calls = len(calls)
        _build_model(solver=_nbegm(), epstein_zin=True).solve(
            params=_ez_params(), log_level="off"
        )
        ez_calls = len(calls) - linear_calls
    assert linear_calls > 0
    assert ez_calls == 0
