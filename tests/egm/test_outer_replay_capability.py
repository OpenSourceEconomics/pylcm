"""The settled answer to what a simulation replay may assume about one period.

Both NNBEGM outer searches publish a policy some reader replays. The questions
that reader would otherwise ask itself — can the declared outer map be
inverted, can its arguments be supplied at a realized state, can the published
rows be addressed there — are answered once here, before either search runs.
"""

from fractions import Fraction

import pytest

from _lcm.egm.outer_inversion import DeclaredOuterInverse
from _lcm.egm.outer_replay_capability import (
    fail_if_continuous_outer_replay_is_unsupported,
    resolve_outer_replay_capability,
)
from lcm.exceptions import RegimeInitializationError
from lcm.typing import ContinuousAction, ContinuousState, FloatND

_UNIT_INVERSE = DeclaredOuterInverse(coefficient=Fraction(1), low=0.0, high=20.0)
_BINDABLE = frozenset({"illiquid", "wealth", "interest_rate", "period", "age"})


def _new_illiquid(
    *, illiquid: ContinuousState, illiquid_investment: ContinuousAction
) -> ContinuousState:
    """`s' = Z + Iz`: replayable from the durable stock and the outer action."""
    return illiquid + illiquid_investment


def _new_illiquid_reading_consumption(
    *,
    illiquid: ContinuousState,
    illiquid_investment: ContinuousAction,
    consumption: ContinuousAction,
) -> ContinuousState:
    """`s' = Z + Iz + c`: affine in the outer action, but reads the inner one."""
    return illiquid + illiquid_investment + consumption


def _keep_illiquid(illiquid: ContinuousState) -> FloatND:
    """The no-adjustment candidate: keeping holds the durable stock."""
    return illiquid


def _resolve(
    *,
    functions=None,
    outer_no_adjustment_name=None,
    row_passive_state_names=("illiquid",),
    row_discrete_action_names=(),
    inverse=_UNIT_INVERSE,
):
    """Resolve the capability of a replayable declaration, with one override."""
    return resolve_outer_replay_capability(
        inverse=inverse,
        functions={"new_illiquid": _new_illiquid, "keep_illiquid": _keep_illiquid}
        if functions is None
        else functions,
        bindable_names=_BINDABLE,
        outer_post_decision_name="new_illiquid",
        outer_action_name="illiquid_investment",
        outer_no_adjustment_name=outer_no_adjustment_name,
        outer_state_name="illiquid",
        state_names=frozenset({"illiquid", "wealth"}),
        row_passive_state_names=row_passive_state_names,
        row_discrete_action_names=row_discrete_action_names,
    )


def test_a_replayable_declaration_is_supported():
    """A one-for-one map of durable stock and outer action supports replay."""
    assert _resolve().continuous_replay_is_supported


def test_an_outer_map_reading_the_inner_action_names_the_argument_it_cannot_bind():
    """An outer map argument no realized state or parameter supplies is reported."""
    capability = _resolve(functions={"new_illiquid": _new_illiquid_reading_consumption})

    assert capability.unbindable_functions == (("new_illiquid", ("consumption",)),)


def test_an_undeclared_outer_map_is_reported_as_undeclared():
    """An outer post-decision the regime does not declare is reported by name."""
    capability = _resolve(functions={"keep_illiquid": _keep_illiquid})

    assert capability.undeclared_functions == ("new_illiquid",)


def test_an_undeclared_no_adjustment_candidate_is_reported_as_undeclared():
    """A named keeper candidate absent from the pool is reported by name."""
    capability = _resolve(
        functions={"new_illiquid": _new_illiquid},
        outer_no_adjustment_name="keep_illiquid",
    )

    assert capability.undeclared_functions == ("keep_illiquid",)


def test_a_keeper_holding_an_unreadable_outer_state_is_reported():
    """Holding a durable replay cannot read leaves the keeper with no stock."""
    capability = resolve_outer_replay_capability(
        inverse=_UNIT_INVERSE,
        functions={"new_illiquid": _new_illiquid},
        bindable_names=_BINDABLE,
        outer_post_decision_name="new_illiquid",
        outer_action_name="illiquid_investment",
        outer_no_adjustment_name=None,
        outer_state_name="illiquid",
        state_names=frozenset({"wealth"}),
        row_passive_state_names=("illiquid",),
        row_discrete_action_names=(),
    )

    assert capability.unavailable_keeper_states == ("illiquid",)


def test_a_second_passive_row_axis_is_reported():
    """Passive continuous row axes beyond the single bracketed one are reported."""
    capability = _resolve(row_passive_state_names=("illiquid", "reserve"))

    assert capability.unaddressable_passive_states == ("illiquid", "reserve")


def test_a_discrete_action_row_axis_is_reported():
    """A discrete-action row axis, which a nested read has no address for."""
    capability = _resolve(row_discrete_action_names=("buy_private",))

    assert capability.unaddressable_discrete_actions == ("buy_private",)


def test_a_non_unit_coefficient_is_unsupported_for_continuous_replay():
    """Continuous replay subtracts an offset, so only a unit coefficient inverts."""
    doubled = DeclaredOuterInverse(coefficient=Fraction(2), low=0.0, high=20.0)

    assert not _resolve(inverse=doubled).continuous_replay_is_supported


def test_the_refusal_names_the_argument_replay_cannot_bind():
    """The refusal reports the offending argument, not merely that one exists."""
    capability = _resolve(functions={"new_illiquid": _new_illiquid_reading_consumption})

    with pytest.raises(RegimeInitializationError, match="consumption"):
        fail_if_continuous_outer_replay_is_unsupported(
            capability=capability,
            regime_name="alive",
            outer_action_name="illiquid_investment",
        )


def test_the_refusal_names_the_passive_states_replay_cannot_address():
    """The refusal reports which passive row axes exceeded the single address."""
    capability = _resolve(row_passive_state_names=("illiquid", "reserve"))

    with pytest.raises(RegimeInitializationError, match="reserve"):
        fail_if_continuous_outer_replay_is_unsupported(
            capability=capability,
            regime_name="alive",
            outer_action_name="illiquid_investment",
        )


def test_the_refusal_points_at_the_finite_outer_grid_remedy():
    """The refusal names the outer search that does replay the declaration."""
    capability = _resolve(row_passive_state_names=("illiquid", "reserve"))

    with pytest.raises(RegimeInitializationError, match="FiniteOuterGrid"):
        fail_if_continuous_outer_replay_is_unsupported(
            capability=capability,
            regime_name="alive",
            outer_action_name="illiquid_investment",
        )


def test_a_supported_declaration_is_not_refused():
    """A replayable declaration publishes: the refusal is the control's negative."""
    fail_if_continuous_outer_replay_is_unsupported(
        capability=_resolve(),
        regime_name="alive",
        outer_action_name="illiquid_investment",
    )
