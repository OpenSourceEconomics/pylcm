"""A `Phased` state law prices Q under the SOLVE law, realizes under the SIMULATE one.

"Agents act on beliefs and live in the truth." The simulate phase recomputes Q from the
supplied value arrays, so the law that weights that continuation is observable. These
tests pin the split in both directions -- a build that used one law for both would pass
a one-directional test.

The laws are disjoint POINT MASSES, so nothing here depends on sampling: under the
belief law the agent must choose `stay`, and the world must nevertheless realize the
state the TRUE law dictates.
"""

from typing import Any

import jax.numpy as jnp
import pandas as pd
import pytest

from _lcm.regime_building.stochastic_state_transitions import (
    collect_stochastic_state_transitions,
)
from lcm import (
    AgeGrid,
    DiscreteGrid,
    MarkovTransition,
    Model,
    NormalIIDProcess,
    Phased,
    Regime,
    categorical,
)
from lcm.exceptions import InvalidStateTransitionProbabilitiesError
from lcm.typing import DiscreteAction, FloatND, ScalarInt


@categorical(ordered=True)
class Move:
    stay: ScalarInt
    switch: ScalarInt


@categorical(ordered=True)
class Good:
    bad: ScalarInt
    good: ScalarInt


@categorical(ordered=False)
class RegimeId:
    live: ScalarInt
    last: ScalarInt


def utility(good: DiscreteAction, move: DiscreteAction) -> FloatND:
    # Flow utility is FLAT in the action (`move` enters with a zero weight, only so that
    # every action is "used"), so the argmax is decided purely by the law that weights
    # next period's V. That is what makes the belief/truth wedge observable in `move`.
    return 1.0 * good + 0.0 * move


def _point_mass(to_good: FloatND) -> FloatND:
    """[P(bad), P(good)] as a degenerate distribution."""
    return jnp.stack([1.0 - to_good, to_good], axis=-1)


def next_good_belief(move: DiscreteAction) -> FloatND:
    """BELIEF: `stay` leads to good, `switch` leads to bad."""
    return _point_mass(jnp.where(move == Move.stay, 1.0, 0.0))


def next_good_actual(move: DiscreteAction) -> FloatND:
    """TRUTH: exactly the opposite -- `stay` leads to bad, `switch` leads to good."""
    return _point_mass(jnp.where(move == Move.stay, 0.0, 1.0))


def _next_regime(period):
    return jnp.where(period >= 1, RegimeId.last, RegimeId.live)


def _model(law: Any) -> Model:
    common: dict[str, Any] = {
        "states": {"good": DiscreteGrid(Good)},
        "actions": {"move": DiscreteGrid(Move)},
        "functions": {"utility": utility},
    }
    live = Regime(
        transition=_next_regime, state_transitions={"good": law}, **common
    ).replace(active=lambda age: age < 2)
    last = Regime(transition=None, state_transitions={}, **common).replace(
        active=lambda age: age >= 2
    )
    return Model(
        regimes={"live": live, "last": last},
        ages=AgeGrid(exact_values=(0, 1, 2)),
        regime_id_class=RegimeId,
        description="perceived vs true stochastic law",
    )


PARAMS = {"discount_factor": 0.95, "live": {}, "last": {}}
IC = pd.DataFrame({"regime_name": "live", "age": 0, "good": ["bad"] * 8})


def _simulate(law: Any) -> pd.DataFrame:
    model = _model(law)
    V = model.solve(params=PARAMS, log_level="off")
    res = model.simulate(
        params=PARAMS,
        initial_conditions=IC,
        period_to_regime_to_V_arr=V,
        log_level="off",
        seed=1,
    )
    return res.to_dataframe().reset_index()


def test_phased_stochastic_law_is_accepted():
    """A `MarkovTransition` inside `Phased` is legal for a state transition."""
    _model(
        Phased(
            solve=MarkovTransition(next_good_belief),
            simulate=MarkovTransition(next_good_actual),
        )
    )


def test_q_uses_the_solve_law_and_the_draw_uses_the_simulate_law():
    """The agent acts on beliefs and lives in the truth.

    Belief: `stay` -> good. Truth: `stay` -> bad. So a correct build must
      (a) CHOOSE `stay` in period 0, because that is what the belief law rewards, and
      (b) REALIZE `bad` in period 1, because that is what the true law delivers.

    Before the fix, Q was weighted by the SIMULATE law, so the agent chose `switch` --
    i.e. it optimized against the truth it was not supposed to know.
    """
    df = _simulate(
        Phased(
            solve=MarkovTransition(next_good_belief),
            simulate=MarkovTransition(next_good_actual),
        )
    )
    period_0 = df[df["period"] == 0]
    period_1 = df[df["period"] == 1]

    assert (period_0["move"] == "stay").all(), "Q must be priced under the BELIEF law"
    assert (period_1["good"] == "bad").all(), "the draw must follow the TRUE law"


def test_outer_phased_per_target_dict_splits_by_phase():
    """`Phased(solve={t: ...}, simulate={t: ...})` carries the same split, per target.

    The outer-dict form is normalized into one entry per target, each holding a
    `Phased` of that target's two laws. This pins that the rewrite preserves the
    belief/truth split rather than merely building: the assertions are the same as for
    the bare `Phased` law above.
    """
    df = _simulate(
        Phased(
            solve={
                "live": MarkovTransition(next_good_belief),
                "last": MarkovTransition(next_good_belief),
            },
            simulate={
                "live": MarkovTransition(next_good_actual),
                "last": MarkovTransition(next_good_actual),
            },
        )
    )
    assert (df[df["period"] == 0]["move"] == "stay").all(), (
        "each target's SOLVE cell must price Q"
    )
    assert (df[df["period"] == 1]["good"] == "bad").all(), (
        "each target's SIMULATE cell must realize the draw"
    )


def test_bare_law_is_unchanged():
    """With one law, beliefs and truth coincide: the agent optimizes against it."""
    df = _simulate(MarkovTransition(next_good_actual))
    period_0 = df[df["period"] == 0]
    period_1 = df[df["period"] == 1]

    # Truth rewards `switch`, and with no belief/truth wedge the agent takes it.
    assert (period_0["move"] == "switch").all()
    assert (period_1["good"] == "good").all()


def test_markov_and_process_states_coexist():
    """A `Phased` Markov state and a stochastic PROCESS state in one regime.

    The two kinds of stochastic next-state weight are built by different machinery and
    multiplied into one joint weight over the target bundle. Routing the continuation
    through the solve pool must not disturb that: process grids cannot be `Phased`, so
    the process marginal is phase-invariant and only the Markov marginal carries the
    belief/truth wedge. Utility is flat in the shock, so the split stays the observable.
    """

    def utility_with_shock(
        good: DiscreteAction, move: DiscreteAction, shock: FloatND
    ) -> FloatND:
        return 1.0 * good + 0.0 * move + 0.0 * shock

    common: dict[str, Any] = {
        "states": {
            "good": DiscreteGrid(Good),
            "shock": NormalIIDProcess(n_points=3, gauss_hermite=True),
        },
        "actions": {"move": DiscreteGrid(Move)},
        "functions": {"utility": utility_with_shock},
    }
    live = Regime(
        transition=_next_regime,
        state_transitions={
            "good": Phased(
                solve=MarkovTransition(next_good_belief),
                simulate=MarkovTransition(next_good_actual),
            )
        },
        **common,
    ).replace(active=lambda age: age < 2)
    last = Regime(transition=None, state_transitions={}, **common).replace(
        active=lambda age: age >= 2
    )
    model = Model(
        regimes={"live": live, "last": last},
        ages=AgeGrid(exact_values=(0, 1, 2)),
        regime_id_class=RegimeId,
        description="Markov law and process state in one regime",
    )
    shock_params = {"shock": {"mu": 0.0, "sigma": 1.0}}
    params = {
        "discount_factor": 0.95,
        "live": dict(shock_params),
        "last": dict(shock_params),
    }
    initial_conditions = IC.assign(shock=0.0)
    V = model.solve(params=params, log_level="off")
    df = (
        model.simulate(
            params=params,
            initial_conditions=initial_conditions,
            period_to_regime_to_V_arr=V,
            log_level="off",
            seed=1,
        )
        .to_dataframe()
        .reset_index()
    )
    assert (df[df["period"] == 0]["move"] == "stay").all(), (
        "the Markov marginal must still be priced under the BELIEF law"
    )
    assert (df[df["period"] == 1]["good"] == "bad").all(), (
        "the Markov draw must still follow the TRUE law"
    )


def test_continuation_helper_resolves_from_the_solve_phase():
    """A `Phased` HELPER that a transition depends on must also resolve from solve.

    `dags` resolves a transition's arguments against whichever function pool it is
    handed, so selecting the solve transition node alone is not enough: its helper would
    still be read from the simulate pool. Here the law is a single (bare) function and
    the belief/truth wedge lives entirely in the helper it reads.
    """

    def target_belief() -> FloatND:
        return jnp.array(1.0)  # believes `stay` leads to good

    def target_actual() -> FloatND:
        return jnp.array(0.0)  # in truth it does not

    def next_good(move: DiscreteAction, stay_target: FloatND) -> FloatND:
        to_good = jnp.where(move == Move.stay, stay_target, 1.0 - stay_target)
        return _point_mass(to_good)

    common: dict[str, Any] = {
        "states": {"good": DiscreteGrid(Good)},
        "actions": {"move": DiscreteGrid(Move)},
        "functions": {
            "utility": utility,
            "stay_target": Phased(solve=target_belief, simulate=target_actual),
        },
    }
    live = Regime(
        transition=_next_regime,
        state_transitions={"good": MarkovTransition(next_good)},
        **common,
    ).replace(active=lambda age: age < 2)
    last = Regime(transition=None, state_transitions={}, **common).replace(
        active=lambda age: age >= 2
    )
    model = Model(
        regimes={"live": live, "last": last},
        ages=AgeGrid(exact_values=(0, 1, 2)),
        regime_id_class=RegimeId,
        description="phase-varying helper under a bare stochastic law",
    )
    V = model.solve(params=PARAMS, log_level="off")
    df = (
        model.simulate(
            params=PARAMS,
            initial_conditions=IC,
            period_to_regime_to_V_arr=V,
            log_level="off",
            seed=1,
        )
        .to_dataframe()
        .reset_index()
    )

    assert (df[df["period"] == 0]["move"] == "stay").all(), (
        "the continuation must read the SOLVE variant of a Phased helper"
    )
    assert (df[df["period"] == 1]["good"] == "bad").all(), (
        "the realized draw must read the SIMULATE variant of a Phased helper"
    )


def test_both_phase_variants_get_their_own_metadata_entry():
    """A `Phased` law contributes one entry PER PHASE, keyed apart.

    Both variants used to be inserted under the same `next_good` key, so the simulate
    record overwrote the solve one and the perceived law reached the solver with no
    runtime check at all. Asserting the two funcs are *distinct and both present* is
    what pins that -- a single shared key cannot satisfy it.
    """
    live = Regime(
        transition=_next_regime,
        state_transitions={
            "good": Phased(
                solve=MarkovTransition(next_good_belief),
                simulate=MarkovTransition(next_good_actual),
            )
        },
        states={"good": DiscreteGrid(Good)},
        actions={"move": DiscreteGrid(Move)},
        functions={"utility": utility},
    )
    entries = collect_stochastic_state_transitions(
        user_regime=live, user_regimes={"live": live}
    )
    by_phase = {entry.phase: entry for entry in entries.values()}
    assert set(by_phase) == {"solve", "simulate"}
    assert by_phase["solve"].func is next_good_belief
    assert by_phase["simulate"].func is next_good_actual
    assert all(entry.n_outcomes == 2 for entry in entries.values())


def test_bare_law_keeps_its_unqualified_key_and_no_phase():
    """A phase-invariant law is untouched by the phase tagging."""
    live = Regime(
        transition=_next_regime,
        state_transitions={"good": MarkovTransition(next_good_actual)},
        states={"good": DiscreteGrid(Good)},
        actions={"move": DiscreteGrid(Move)},
        functions={"utility": utility},
    )
    entries = collect_stochastic_state_transitions(
        user_regime=live, user_regimes={"live": live}
    )
    assert set(entries) == {"next_good"}
    assert entries["next_good"].phase is None


def _bad_probs(move: DiscreteAction) -> FloatND:
    """Malformed: rows do not sum to 1 (and one weight is negative)."""
    return _point_mass(jnp.where(move == Move.stay, 1.2, -0.2))


@pytest.mark.parametrize("bad_phase", ["solve", "simulate"])
def test_a_malformed_law_is_caught_in_either_phase(bad_phase):
    """Exactly one variant is malformed at a time, and the named phase must fail.

    The point of the parametrization: with both variants valid, an implementation that
    checks only one of them still passes. Making each one invalid IN TURN is what
    distinguishes "both are validated" from "one is validated". A malformed BELIEF law
    is not benign -- it prices every action in backward induction.
    """
    laws = {"solve": next_good_belief, "simulate": next_good_actual}
    laws[bad_phase] = _bad_probs
    model = _model(
        Phased(
            solve=MarkovTransition(laws["solve"]),
            simulate=MarkovTransition(laws["simulate"]),
        )
    )
    with pytest.raises(
        InvalidStateTransitionProbabilitiesError, match=f"{bad_phase} phase"
    ):
        model.solve(params=PARAMS, log_level="debug")
