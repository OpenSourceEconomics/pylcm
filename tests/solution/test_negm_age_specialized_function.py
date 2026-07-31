"""NEGM regimes honour `AgeSpecializedFunction` in their outer helpers per period.

An age-specialized function is a *different function* at each age, so a period's
kernel has to be built from that period's concrete function. NEGM reads two helpers
from the regime's function pool before the inner DC-EGM builder ever runs — the
no-adjustment candidate (`NEGM.outer_no_adjustment_candidate`) and the outer cost
(`NEGM.outer_cost`) — and both must be that period's own.

The oracle is the last age at which the NEGM regime is active. Its value depends on
exactly two things: its own economics and a continuation into the terminal regime,
which no age specialization touches. So the age-specialized solve must reproduce,
*exactly*, a plain solve whose helper is the concrete function `build(age)` returns
at that age. Resolving the specialization at some other age moves the value by a
finite amount, which makes equality the right instrument rather than a tolerance.
"""

from dataclasses import replace

import jax.numpy as jnp
import numpy as np
import pytest

from lcm import AgeGrid, AgeSpecializedFunction, Model, Regime
from lcm.typing import ContinuousState, FloatND
from tests.conftest import DECIMAL_PRECISION
from tests.test_models import negm_kinked_toy
from tests.test_models.negm_kinked_toy import (
    N_PERIODS,
    NEGM_SOLVER,
    RegimeId,
    build_dead_regime,
    credited,
    inverse_marginal_utility,
    keep_illiquid,
    liquid_savings,
    next_regime,
    next_wealth,
    resources_before_outer_cost,
    utility,
)

_MIN_AGE = 20
_AGE_STEP = 5
_FINAL_AGE_ALIVE = _MIN_AGE + (N_PERIODS - 2) * _AGE_STEP
_PARAMS = {"discount_factor": 0.95, "alive": {}}


def _make_depreciating_keep(age: float):
    """`keep_illiquid` with the no-adjustment stock depreciating by age."""
    retained = 1.0 - 0.02 * (age - _MIN_AGE)

    def keep_illiquid_at_age(illiquid: ContinuousState) -> FloatND:
        return retained * illiquid

    return keep_illiquid_at_age


def _make_penalised_credited(age: float):
    """`credited` with the withdrawal penalty drifting by age."""
    penalty = 0.10 + 0.02 * (age - _MIN_AGE)

    def credited_at_age(
        illiquid: ContinuousState, next_illiquid: ContinuousState
    ) -> FloatND:
        investment = next_illiquid - illiquid
        return jnp.where(investment < 0.0, (1.0 - penalty) * investment, investment)

    return credited_at_age


_HELPERS = {
    "keep_illiquid": (_make_depreciating_keep, keep_illiquid),
    "credited": (_make_penalised_credited, credited),
}


def _build_model(*, helper_name: str, override) -> Model:
    """The kinked NEGM toy with one outer helper optionally replaced.

    `override` is the value to bind to `helper_name` — an `AgeSpecializedFunction`,
    a concrete function pinned to one age, or `None` to keep the model's own
    age-invariant helper.
    """
    functions = {
        "utility": utility,
        "resources_before_outer_cost": resources_before_outer_cost,
        "liquid_savings": liquid_savings,
        "keep_illiquid": keep_illiquid,
        "credited": credited,
        "inverse_marginal_utility": inverse_marginal_utility,
    }
    if override is not None:
        functions[helper_name] = override
    alive = Regime(
        active=lambda age: age <= _FINAL_AGE_ALIVE,
        states={
            "wealth": negm_kinked_toy.WEALTH_GRID,
            "illiquid": negm_kinked_toy.ILLIQUID_GRID,
        },
        state_transitions={
            "wealth": next_wealth,
            "illiquid": negm_kinked_toy.durable_transition,
        },
        actions={
            "consumption": negm_kinked_toy.CONSUMPTION_GRID,
            "illiquid_investment": negm_kinked_toy.ILLIQUID_INVESTMENT_GRID,
        },
        transition=next_regime,
        functions=functions,
        solver=replace(NEGM_SOLVER),
    )
    return Model(
        regimes={"alive": alive, "dead": build_dead_regime()},
        regime_id_class=RegimeId,
        ages=AgeGrid(
            start=_MIN_AGE,
            stop=_MIN_AGE + (N_PERIODS - 1) * _AGE_STEP,
            step=f"{_AGE_STEP}Y",
        ),
        fixed_params={"final_age_alive": _FINAL_AGE_ALIVE},
    )


def _specialized(helper_name: str) -> AgeSpecializedFunction:
    build, _ = _HELPERS[helper_name]
    return AgeSpecializedFunction(build=build, signature=lambda age: age)


@pytest.mark.parametrize("helper_name", ["keep_illiquid", "credited"])
def test_the_last_active_age_uses_that_ages_own_outer_helper(helper_name):
    """The last active age's value equals a plain solve pinned to that age's helper.

    The last active period's value depends on its own economics and on a
    continuation into the terminal regime, which no age specialization touches. So
    the age-specialized solve must reproduce, *exactly*, a plain solve whose helper
    is the concrete function `build(age)` returns at that age.
    """
    build, _ = _HELPERS[helper_name]
    specialized = _build_model(
        helper_name=helper_name, override=_specialized(helper_name)
    ).solve(params=_PARAMS, log_level="debug")
    last_active = max(
        period for period, regimes in specialized.items() if "alive" in regimes
    )
    pinned = _build_model(
        helper_name=helper_name,
        override=build(_MIN_AGE + last_active * _AGE_STEP),
    ).solve(params=_PARAMS, log_level="debug")

    expected = np.asarray(pinned[last_active]["alive"])
    got = np.asarray(specialized[last_active]["alive"])
    np.testing.assert_array_equal(np.isneginf(got), np.isneginf(expected))
    finite = np.isfinite(expected)
    np.testing.assert_array_almost_equal(
        got[finite], expected[finite], decimal=DECIMAL_PRECISION
    )


@pytest.mark.parametrize("helper_name", ["keep_illiquid", "credited"])
def test_an_age_specialized_outer_helper_moves_the_negm_solution(helper_name):
    """The drifting helper changes the NEGM value function.

    Without this, the agreement test above would still pass if the specialization
    were ignored in both solves in the same way.
    """
    drifting = _build_model(
        helper_name=helper_name, override=_specialized(helper_name)
    ).solve(params=_PARAMS, log_level="debug")
    flat = _build_model(helper_name=helper_name, override=None).solve(
        params=_PARAMS, log_level="debug"
    )

    moved = [
        period
        for period in drifting
        if "alive" in drifting[period]
        and not np.allclose(
            np.asarray(drifting[period]["alive"]),
            np.asarray(flat[period]["alive"]),
            equal_nan=True,
        )
    ]
    assert moved, "the age-specialized outer helper left every period unchanged"


@pytest.mark.parametrize("helper_name", ["keep_illiquid", "credited"])
def test_ages_sharing_one_signature_resolve_to_one_concrete_helper(helper_name):
    """A signature constant across ages makes every age use one concrete helper.

    `signature(age)` is the sharing key and a correctness precondition: ages with
    equal signatures share one compiled program, so an equal signature must imply an
    identical resolved closure. A specialization that declares one signature for
    every age therefore has to behave exactly like a plain solve pinned to the single
    helper that group resolves to — the first active age's.

    This is the companion to the per-age agreement test above. That one pins that
    distinct signatures are *not* collapsed; this one pins that equal signatures *are*
    shared rather than silently re-resolved per period.
    """
    build, _ = _HELPERS[helper_name]
    constant_signature = _build_model(
        helper_name=helper_name,
        override=AgeSpecializedFunction(build=build, signature=lambda _age: 0),
    ).solve(params=_PARAMS, log_level="debug")
    pinned = _build_model(helper_name=helper_name, override=build(_MIN_AGE)).solve(
        params=_PARAMS, log_level="debug"
    )

    for period, regimes in pinned.items():
        if "alive" not in regimes:
            continue
        expected = np.asarray(regimes["alive"])
        got = np.asarray(constant_signature[period]["alive"])
        finite = np.isfinite(expected)
        np.testing.assert_array_almost_equal(
            got[finite], expected[finite], decimal=DECIMAL_PRECISION
        )
