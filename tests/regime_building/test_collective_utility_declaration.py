"""What `CollectiveUtility` promises about the bodies it names.

A household names its members once, in the `utilities` keys, and that
order is the order every published array carries. A body may be written there,
may vary by phase, or may be left to arrive from the model level — and the last
of those is what lets one model mix collective and singleton regimes without
either of them writing the other's utility.
"""

from collections.abc import Mapping

import jax.numpy as jnp
import pytest

from lcm import (
    AgeGrid,
    CollectiveUtility,
    DiscreteGrid,
    LinSpacedGrid,
    Model,
    Phased,
    Regime,
    categorical,
    fixed_transition,
)
from lcm.exceptions import ModelInitializationError, RegimeInitializationError
from lcm.typing import (
    ContinuousState,
    DiscreteAction,
    FloatND,
    ScalarInt,
    UserFunction,
)

_WEALTH = LinSpacedGrid(start=1.0, stop=3.0, n_points=3)


@categorical(ordered=True)
class Work:
    """The binary action of the miniature."""

    leisure: ScalarInt
    work: ScalarInt


@categorical(ordered=False)
class RegimeId:
    """Regime ids of the miniature."""

    couple: ScalarInt
    couple_terminal: ScalarInt


def _u_f(*, wealth: ContinuousState, work: DiscreteAction) -> FloatND:
    """The first member's flow utility, mildly averse to working."""
    return jnp.log(wealth) - 0.1 * work


def _u_m(*, wealth: ContinuousState, work: DiscreteAction) -> FloatND:
    """The second member's flow utility, mildly averse to working."""
    return 0.5 * jnp.log(wealth) - 0.1 * work


def _u_m_simulate(*, wealth: ContinuousState, work: DiscreteAction) -> FloatND:
    """The second member's flow utility as simulation realizes it."""
    return 0.25 * jnp.log(wealth) - 0.1 * work


def _u_f_terminal(wealth: ContinuousState) -> FloatND:
    """The first member's terminal payoff."""
    return jnp.log(wealth)


def _u_m_terminal(wealth: ContinuousState) -> FloatND:
    """The second member's terminal payoff."""
    return 0.5 * jnp.log(wealth)


def _couple(
    *, functions: Mapping[str, UserFunction | Phased | CollectiveUtility | None]
) -> Regime:
    """The collective regime of the miniature, with `functions` supplied."""
    return Regime(
        states={"wealth": _WEALTH},
        state_transitions={"wealth": fixed_transition("wealth")},
        actions={"work": DiscreteGrid(category_class=Work)},
        functions=functions,
    )


def test_a_none_body_takes_the_utility_already_present_under_its_name():
    """`utilities={"f": None}` means f's utility is the regime's `utility_f`."""
    regime = _couple(
        functions={
            "utility": CollectiveUtility(utilities={"f": None, "m": _u_m}),
            "utility_f": _u_f,
        }
    )

    assert regime.decomposed_functions["utility_f"] is _u_f


def test_a_none_body_may_still_arrive_from_the_model_level():
    """A delegated body supplied by the model reaches the household that named it.

    Completeness is a property of the merged regime, so a bare `Regime` whose
    household delegates a body carries no error yet — the model is where the
    body arrives and where its absence would be reported.
    """
    couple = _couple(
        functions={"utility": CollectiveUtility(utilities={"f": None, "m": _u_m})}
    )
    terminal = Regime(
        states={"wealth": _WEALTH},
        functions={
            "utility": CollectiveUtility(
                utilities={"f": _u_f_terminal, "m": _u_m_terminal}
            ),
        },
    )

    model = Model(
        regimes={"couple": couple, "couple_terminal": terminal},
        edges={"couple": {"couple_terminal": 0}},
        ages=AgeGrid(start=0, inclusive_stop=2, step="Y"),
        regime_id_class=RegimeId,
        functions={"utility_f": _u_f},
        initial_nodes={0: "couple"},
    )

    assert model.user_regimes["couple"].decomposed_functions["utility_f"] is _u_f


def test_a_none_body_with_nothing_to_delegate_to_is_refused_by_name():
    """A member left undeclared everywhere names the entry that is missing."""
    couple = _couple(
        functions={"utility": CollectiveUtility(utilities={"f": None, "m": _u_m})}
    )
    terminal = Regime(
        states={"wealth": _WEALTH},
        functions={
            "utility": CollectiveUtility(
                utilities={"f": _u_f_terminal, "m": _u_m_terminal}
            ),
        },
    )

    with pytest.raises(ModelInitializationError, match="utility_f"):
        Model(
            regimes={"couple": couple, "couple_terminal": terminal},
            edges={"couple": {"couple_terminal": 0}},
            ages=AgeGrid(start=0, inclusive_stop=2, step="Y"),
            regime_id_class=RegimeId,
            initial_nodes={0: "couple"},
        )


def test_the_utilities_keys_fix_the_member_order():
    """Member order follows the declaration, not the `functions` order."""
    regime = _couple(
        functions={
            "utility_m": _u_m,
            "utility": CollectiveUtility(utilities={"f": _u_f, "m": None}),
        }
    )

    assert regime.member_axis == ("f", "m")


def test_a_members_utility_may_differ_between_the_two_phases():
    """A `Phased` body declares what solve prices and what simulation realizes.

    Written as the FIRST entry: the declaration's own annotation is what has to
    admit it, and a checker that inspects one entry of a mapping inspects that
    one.
    """
    body = Phased(solve=_u_m, simulate=_u_m_simulate)
    regime = _couple(
        functions={"utility": CollectiveUtility(utilities={"m": body, "f": _u_f})}
    )

    assert regime.decomposed_functions["utility_m"] is body


def test_a_household_with_no_members_is_refused_where_it_is_written():
    """`utilities={}` names no household, and the declaration says so itself."""
    with pytest.raises(RegimeInitializationError, match=r"(?i)at least one member"):
        CollectiveUtility(utilities={})


def test_a_phased_member_utility_solves_and_simulates_its_own_variant():
    """The two phases of a `Phased` body reach the engine as declared."""
    couple = _couple(
        functions={
            "utility": CollectiveUtility(
                utilities={"f": _u_f, "m": Phased(solve=_u_m, simulate=_u_m_simulate)}
            )
        }
    )
    terminal = Regime(
        states={"wealth": _WEALTH},
        functions={
            "utility": CollectiveUtility(
                utilities={"f": _u_f_terminal, "m": _u_m_terminal}
            ),
        },
    )
    model = Model(
        regimes={"couple": couple, "couple_terminal": terminal},
        edges={"couple": {"couple_terminal": 0}},
        ages=AgeGrid(start=0, inclusive_stop=2, step="Y"),
        regime_id_class=RegimeId,
        initial_nodes={0: "couple"},
    )

    solved = model.solve(
        params={
            "couple": {"koopmans_aggregator": {"discount_factor": 0.9}},
            "couple_terminal": {},
        },
        log_level="off",
    ).values

    # Member m's period-0 value is her solve utility plus the discounted
    # terminal one, at wealth held fixed: 0.5*log(w) + 0.9*0.5*log(w).
    expected = 0.5 * jnp.log(jnp.asarray([1.0, 2.0, 3.0])) * 1.9
    assert jnp.allclose(solved[0]["couple"][..., 1], expected)


def _single(*, utility: UserFunction) -> Regime:
    """A regime of the miniature that declares a plain utility function."""
    return Regime(
        states={"wealth": _WEALTH},
        state_transitions={"wealth": fixed_transition("wealth")},
        actions={"work": DiscreteGrid(category_class=Work)},
        functions={"utility": utility},
    )


def test_a_plain_utility_declares_the_single_member_self():
    """A regime with a plain utility function has exactly one member, `"self"`."""
    assert _single(utility=_u_f).members == ("self",)


def test_a_plain_utility_declares_no_member_axis():
    """A regime with a plain utility function carries no member axis."""
    assert _single(utility=_u_f).member_axis is None


def test_a_household_declares_its_utilities_keys_as_members():
    """A household's members are its `utilities` keys, in the order written."""
    regime = _couple(
        functions={"utility": CollectiveUtility(utilities={"m": _u_m, "f": _u_f})}
    )

    assert regime.members == ("m", "f")


def test_a_households_member_axis_is_its_members():
    """A household's member axis names its members in the order written."""
    regime = _couple(
        functions={"utility": CollectiveUtility(utilities={"m": _u_m, "f": _u_f})}
    )

    assert regime.member_axis == ("m", "f")


def test_a_household_member_named_self_is_refused_where_it_is_written():
    """`"self"` is the single member of a plain utility, so no household names it."""
    with pytest.raises(RegimeInitializationError, match=r"'self'"):
        CollectiveUtility(utilities={"self": _u_f, "m": _u_m})
