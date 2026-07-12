"""Integration tests for the NON-terminal collective-regime (E1) solve (slice 2).

Extends the slice-1b terminal machinery with per-stakeholder continuation
values: a non-terminal `Regime(stakeholders=("f", "m"))` computes
`Q^s = u^s + beta * E[V'^s]` per stakeholder, argmaxes the household
scalarization `O = sum_s lambda_s Q^s` over the feasible actions, and reads off
each stakeholder's OWN `Q^s` at that shared argmax (design doc
`pylcm-extension-collective-regimes.md` §2, E1 — no gates, which are slice 4).

The economics mirror the 1b work/leisure setup (same utility functions) so the
hand computations carry over; only the wage grid's low point moves from 10 to 8
so every argmax is strict (at wage 10 the myopic household objective ties).

Hand computation, wage grid {8, 40}, beta = 0.95, next_wage = 40*work +
8*(1-work) (working today yields the high wage tomorrow):

Terminal period (t=1), myopic household argmax of O = (u_f + u_m)/2:
  wage=8 : leisure (30, 0) O=15  vs work (8, 16)  O=12  -> leisure, V=(30, 0)
  wage=40: leisure (30, 0) O=15  vs work (40, 80) O=60  -> work,    V=(40, 80)

Period 0, Q^s = u^s + 0.95 * V'^s(next_wage):
  wage=8 : leisure Q=(30+28.5, 0)      =(58.5, 0)  O=29.25
           work    Q=(8+0.95*40, 16+0.95*80)=(46, 92)  O=69  -> work, V=(46, 92)
  wage=40: leisure Q=(58.5, 0)  O=29.25
           work    Q=(40+38, 80+76)=(78, 156)      O=117    -> work, V=(78, 156)

At wage=8 the period-0 household argmax (work) DIFFERS from the myopic
terminal-period one (leisure): only the continuation through tomorrow's high
wage makes work jointly optimal.
"""

from types import MappingProxyType

import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.regime_building.finalize import finalize_regimes
from _lcm.regime_building.processing import process_regimes
from _lcm.solution.backward_induction import solve
from _lcm.utils.logging import get_logger
from lcm import DiscreteGrid, LinSpacedGrid, Model, categorical
from lcm.ages import AgeGrid
from lcm.certainty_equivalent import PowerMean
from lcm.regime import Regime
from lcm.taste_shocks import ExtremeValueTasteShocks
from lcm.transition import MarkovTransition
from lcm.typing import (
    ContinuousState,
    DiscreteAction,
    DiscreteState,
    FloatND,
    ScalarInt,
)


@categorical(ordered=True)
class Work:
    leisure: ScalarInt  # code 0
    work: ScalarInt  # code 1


@categorical(ordered=False)
class RegimeId:
    couple: ScalarInt  # code 0
    couple_terminal: ScalarInt  # code 1


def _utility_f(wage: ContinuousState, work: DiscreteAction) -> FloatND:
    """Wife: values her own leisure highly, also sees household consumption."""
    consumption = wage * work
    return consumption + 30.0 * (1.0 - work)


def _utility_m(wage: ContinuousState, work: DiscreteAction) -> FloatND:
    """Husband: values household consumption, indifferent to leisure."""
    consumption = wage * work
    return 2.0 * consumption


def _next_wage(work: DiscreteAction) -> ContinuousState:
    """Deterministic wage law: working today yields the high wage tomorrow."""
    return 40.0 * work + 8.0 * (1.0 - work)


def _next_regime() -> ScalarInt:
    return RegimeId.couple_terminal


_WAGE_GRID = LinSpacedGrid(start=8.0, stop=40.0, n_points=2)

_EXPECTED_V_TERMINAL = np.array([[30.0, 0.0], [40.0, 80.0]])
_EXPECTED_V_PERIOD_0 = np.array([[46.0, 92.0], [78.0, 156.0]])


def _make_couple_regimes() -> dict[str, Regime]:
    couple = Regime(
        transition=_next_regime,
        active=lambda age: age < 1,
        stakeholders=("f", "m"),
        states={"wage": _WAGE_GRID},
        state_transitions={"wage": _next_wage},
        actions={"work": DiscreteGrid(Work)},
        functions={"utility_f": _utility_f, "utility_m": _utility_m},
    )
    couple_terminal = Regime(
        transition=None,
        active=lambda age: age >= 1,
        stakeholders=("f", "m"),
        states={"wage": _WAGE_GRID},
        actions={"work": DiscreteGrid(Work)},
        functions={"utility_f": _utility_f, "utility_m": _utility_m},
    )
    return {"couple": couple, "couple_terminal": couple_terminal}


def test_nonterminal_collective_regime_solves_with_continuation():
    """Kernel-level: finalize -> process -> backward induction, two periods."""
    ages = AgeGrid(start=0, stop=2, step="Y")
    regimes = process_regimes(
        user_regimes=finalize_regimes(
            user_regimes=_make_couple_regimes(), derived_categoricals={}
        ),
        ages=ages,
        regime_names_to_ids=MappingProxyType(
            {"couple": jnp.int32(0), "couple_terminal": jnp.int32(1)}
        ),
        enable_jit=False,
    )

    solution, _sim_policies, _dissolution_flags = solve(
        flat_params=MappingProxyType(
            {
                "couple": MappingProxyType({"H__discount_factor": jnp.asarray(0.95)}),
                "couple_terminal": MappingProxyType({}),
            }
        ),
        ages=ages,
        regimes=regimes,
        logger=get_logger(log_level="off"),
        enable_jit=False,
    )

    V_terminal = solution[1]["couple_terminal"]
    V_0 = solution[0]["couple"]

    # Both V arrays carry the trailing stakeholder axis: (wage grid = 2) x (S = 2).
    assert V_terminal.shape == (2, 2)
    assert V_0.shape == (2, 2)

    np.testing.assert_allclose(np.asarray(V_terminal), _EXPECTED_V_TERMINAL, rtol=1e-6)
    np.testing.assert_allclose(np.asarray(V_0), _EXPECTED_V_PERIOD_0, rtol=1e-6)

    # Per-stakeholder values genuinely differ (not the scalarization).
    assert not np.allclose(np.asarray(V_0[:, 0]), np.asarray(V_0[:, 1]))

    # The continuation flips the argmax at the low wage: the myopic (terminal)
    # household choice there is leisure, whose period-0 stakeholder values would
    # be (30 + 0.95*30, 0 + 0.95*0) = (58.5, 0). The solved V is the work
    # branch's (46, 92) instead.
    assert not np.allclose(np.asarray(V_0[0]), [58.5, 0.0])


def test_nonterminal_collective_full_model_solve_matches_kernel_level():
    """Model-level: the same two regimes through public Model(...) + solve()."""
    ages = AgeGrid(start=0, stop=2, step="Y")
    model = Model(
        regimes=_make_couple_regimes(),
        ages=ages,
        regime_id_class=RegimeId,
    )

    solution = model.solve(params={"discount_factor": 0.95}, log_level="off")

    V_terminal = solution[1]["couple_terminal"]
    V_0 = solution[0]["couple"]
    assert V_terminal.shape == (2, 2)
    assert V_0.shape == (2, 2)
    np.testing.assert_allclose(np.asarray(V_terminal), _EXPECTED_V_TERMINAL, rtol=1e-6)
    np.testing.assert_allclose(np.asarray(V_0), _EXPECTED_V_PERIOD_0, rtol=1e-6)


@categorical(ordered=True)
class Mood:
    grumpy: ScalarInt  # code 0
    cheerful: ScalarInt  # code 1


def _utility_f_mood(
    wage: ContinuousState, work: DiscreteAction, mood: DiscreteState
) -> FloatND:
    """As `_utility_f`, plus a mood bonus only the wife's felicity carries."""
    consumption = wage * work
    return consumption + 30.0 * (1.0 - work) + 5.0 * mood


def _next_mood(mood: DiscreteState) -> FloatND:  # noqa: ARG001
    """IID coin-flip mood: probabilities (0.5, 0.5) regardless of today's mood."""
    return jnp.array([0.5, 0.5])


def test_nonterminal_collective_stochastic_state_expectation_is_per_stakeholder():
    """A Markov state's expectation is taken per stakeholder slice.

    Adds an IID binary `mood` state (probabilities 1/2 each) whose realization
    enters ONLY the wife's felicity (`+ 5 * mood`). Hand computation on top of
    the module-level one:

    Terminal V (mood shifts u_f additively, so argmax is unchanged):
      V(m, wage=8)  = (30 + 5m, 0)      V(m, wage=40) = (40 + 5m, 80)

    Period 0: E over mood' hits the stakeholder slices differently —
    E[V'_f](wage') = base_f(wage') + 2.5 while E[V'_m](wage') is unchanged:
      (m, wage=8) : leisure Q=(30+5m+0.95*32.5, 0)=(60.875+5m, 0)   O=30.4375+2.5m
                    work    Q=(8+5m+0.95*42.5, 16+76)=(48.375+5m, 92) O=70.1875+2.5m
                    -> work, V=(48.375+5m, 92)
      (m, wage=40): work analogously -> V=(80.375+5m, 156)
    """
    couple = Regime(
        transition=_next_regime,
        active=lambda age: age < 1,
        stakeholders=("f", "m"),
        states={"mood": DiscreteGrid(Mood), "wage": _WAGE_GRID},
        state_transitions={
            "mood": MarkovTransition(_next_mood),
            "wage": _next_wage,
        },
        actions={"work": DiscreteGrid(Work)},
        functions={"utility_f": _utility_f_mood, "utility_m": _utility_m},
    )
    couple_terminal = Regime(
        transition=None,
        active=lambda age: age >= 1,
        stakeholders=("f", "m"),
        states={"mood": DiscreteGrid(Mood), "wage": _WAGE_GRID},
        actions={"work": DiscreteGrid(Work)},
        functions={"utility_f": _utility_f_mood, "utility_m": _utility_m},
    )
    ages = AgeGrid(start=0, stop=2, step="Y")
    regimes = process_regimes(
        user_regimes=finalize_regimes(
            user_regimes={"couple": couple, "couple_terminal": couple_terminal},
            derived_categoricals={},
        ),
        ages=ages,
        regime_names_to_ids=MappingProxyType(
            {"couple": jnp.int32(0), "couple_terminal": jnp.int32(1)}
        ),
        enable_jit=False,
    )

    solution, _sim_policies, _dissolution_flags = solve(
        flat_params=MappingProxyType(
            {
                "couple": MappingProxyType({"H__discount_factor": jnp.asarray(0.95)}),
                "couple_terminal": MappingProxyType({}),
            }
        ),
        ages=ages,
        regimes=regimes,
        logger=get_logger(log_level="off"),
        enable_jit=False,
    )

    V_terminal = solution[1]["couple_terminal"]
    V_0 = solution[0]["couple"]

    # (mood = 2) x (wage grid = 2) x (stakeholders = 2).
    assert V_terminal.shape == (2, 2, 2)
    assert V_0.shape == (2, 2, 2)

    expected_terminal = np.array(
        [
            [[30.0, 0.0], [40.0, 80.0]],  # mood = 0
            [[35.0, 0.0], [45.0, 80.0]],  # mood = 1
        ]
    )
    expected_period_0 = np.array(
        [
            [[48.375, 92.0], [80.375, 156.0]],  # mood = 0
            [[53.375, 92.0], [85.375, 156.0]],  # mood = 1
        ]
    )
    np.testing.assert_allclose(np.asarray(V_terminal), expected_terminal, rtol=1e-6)
    np.testing.assert_allclose(np.asarray(V_0), expected_period_0, rtol=1e-6)


def test_collective_model_simulates_end_to_end_via_public_model_api():
    """Simulation of a collective regime is no longer a raising stub (E4, slice 6).

    The full recomputed-argmax mechanics (both stakeholders tracked, values
    matching the hand computation above) are covered in depth by
    `tests/regime_building/test_collective_regime_simulate.py`; this pin just
    confirms the PUBLIC `Model.simulate()` path — auto-solving, then
    simulating — runs end to end instead of raising, for a collective regime
    with no gated edges (so `period_to_regime_to_V_arr`/dissolution flags need no
    special threading).
    """
    ages = AgeGrid(start=0, stop=2, step="Y")
    model = Model(
        regimes=_make_couple_regimes(),
        ages=ages,
        regime_id_class=RegimeId,
    )
    result = model.simulate(
        params={"discount_factor": 0.95},
        initial_conditions={
            "wage": jnp.array([8.0, 40.0]),
            "age": jnp.array([0.0, 0.0]),
            "regime_id": jnp.array([0, 0], dtype=jnp.int32),
        },
        period_to_regime_to_V_arr=None,
        log_level="off",
        seed=0,
    )
    period_0 = result.raw_results["couple"][0]
    np.testing.assert_allclose(
        np.asarray(period_0.V_arr), _EXPECTED_V_PERIOD_0, rtol=1e-6
    )


def test_nonterminal_collective_regime_with_singleton_target_is_rejected():
    """Routing a collective regime toward a singleton target is slice 4 (E3')."""

    def _utility_single(wage: ContinuousState) -> FloatND:
        return wage

    couple = Regime(
        transition=_next_regime,
        active=lambda age: age < 1,
        stakeholders=("f", "m"),
        states={"wage": _WAGE_GRID},
        state_transitions={"wage": _next_wage},
        actions={"work": DiscreteGrid(Work)},
        functions={"utility_f": _utility_f, "utility_m": _utility_m},
    )
    single_terminal = Regime(
        transition=None,
        active=lambda age: age >= 1,
        states={"wage": _WAGE_GRID},
        functions={"utility": _utility_single},
    )
    ages = AgeGrid(start=0, stop=2, step="Y")

    with pytest.raises(NotImplementedError, match="identical `stakeholders`"):
        process_regimes(
            user_regimes=finalize_regimes(
                user_regimes={"couple": couple, "single_terminal": single_terminal},
                derived_categoricals={},
            ),
            ages=ages,
            regime_names_to_ids=MappingProxyType(
                {"couple": jnp.int32(0), "single_terminal": jnp.int32(1)}
            ),
            enable_jit=False,
        )


def test_collective_regime_with_taste_shocks_is_rejected():
    """EV1 taste shocks on a collective regime are out of scope for E1."""
    with pytest.raises(NotImplementedError, match="taste shocks"):
        Regime(
            transition=_next_regime,
            stakeholders=("f", "m"),
            taste_shocks=ExtremeValueTasteShocks(),
            states={"wage": _WAGE_GRID},
            state_transitions={"wage": _next_wage},
            actions={"work": DiscreteGrid(Work)},
            functions={"utility_f": _utility_f, "utility_m": _utility_m},
        )


def test_collective_regime_with_certainty_equivalent_is_rejected():
    """A nonlinear certainty equivalent on a collective regime is out of scope."""
    with pytest.raises(NotImplementedError, match="certainty equivalent"):
        Regime(
            transition=_next_regime,
            stakeholders=("f", "m"),
            certainty_equivalent=PowerMean(),
            states={"wage": _WAGE_GRID},
            state_transitions={"wage": _next_wage},
            actions={"work": DiscreteGrid(Work)},
            functions={"utility_f": _utility_f, "utility_m": _utility_m},
        )
