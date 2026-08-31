"""Writing engine-generated functions back onto a regime that declares a household.

Regime building composes extra functions — case-piece outputs, margin
resources, derived categoricals — and hands the result back to the regime. It
reads that mapping from `decomposed_functions`, so a plain write-back into
`functions` would replace the author's declaration with what the declaration
was decomposed into, and the household would be gone. The write-back seam
keeps the declaration and overlays only what the engine actually added.
"""

from collections.abc import Mapping

import jax.numpy as jnp
import pytest

from lcm import (
    CollectiveUtility,
    DiscreteGrid,
    LinSpacedGrid,
    Regime,
    categorical,
    fixed_transition,
)
from lcm.exceptions import RegimeInitializationError
from lcm.typing import ContinuousState, DiscreteAction, FloatND, ScalarInt

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
    """The first stakeholder's flow utility."""
    return jnp.log(wealth) - 0.1 * work


def _u_m(*, wealth: ContinuousState, work: DiscreteAction) -> FloatND:
    """The second stakeholder's flow utility."""
    return 0.5 * jnp.log(wealth) - 0.1 * work


def _other_u_m(*, wealth: ContinuousState, work: DiscreteAction) -> FloatND:
    """A different body for the second stakeholder."""
    return 0.9 * jnp.log(wealth) - 0.1 * work


def _resources(wealth: ContinuousState) -> FloatND:
    """A function regime building composes and hands back."""
    return 1.05 * wealth


def _couple(*, functions: Mapping[str, object]) -> Regime:
    """The collective regime of the miniature."""
    return Regime(
        transition=lambda: RegimeId.couple_terminal,
        active=lambda age: age < 1,
        states={"wealth": _WEALTH},
        state_transitions={"wealth": fixed_transition("wealth")},
        actions={"work": DiscreteGrid(category_class=Work)},
        functions=functions,  # ty: ignore[invalid-argument-type]
    )


def _declaring_regime() -> Regime:
    """A regime whose household is written as a `CollectiveUtility`."""
    return _couple(
        functions={"utility": CollectiveUtility(utilities={"f": _u_f, "m": _u_m})}
    )


def test_the_declaration_survives_the_write_back():
    """What the author wrote is still there after the engine adds to it."""
    regime = _declaring_regime()
    engine_functions = {**regime.decomposed_functions, "resources": _resources}

    written = regime.with_engine_functions(engine_functions=engine_functions)

    assert written.functions["utility"] is regime.functions["utility"]


def test_an_engine_addition_reaches_the_decomposed_view():
    """The composed function is what the next stage reads off the regime."""
    regime = _declaring_regime()
    engine_functions = {**regime.decomposed_functions, "resources": _resources}

    written = regime.with_engine_functions(engine_functions=engine_functions)

    assert written.decomposed_functions["resources"] is _resources


def test_a_stakeholders_own_body_is_not_written_into_the_raw_slot():
    """A synthesized entry belongs to the declaration, not beside it."""
    regime = _declaring_regime()
    engine_functions = {**regime.decomposed_functions, "resources": _resources}

    written = regime.with_engine_functions(engine_functions=engine_functions)

    assert "utility_f" not in written.functions


def test_rewriting_a_stakeholders_body_is_refused():
    """A stakeholder's utility is hers to declare; the engine may not swap it."""
    regime = _declaring_regime()
    engine_functions = {**regime.decomposed_functions, "utility_m": _other_u_m}

    with pytest.raises(RegimeInitializationError, match="utility_m"):
        regime.with_engine_functions(engine_functions=engine_functions)


def test_replacing_the_declaration_itself_is_refused():
    """Handing back a plain `utility` would silently dissolve the household."""
    regime = _declaring_regime()
    engine_functions = {**regime.decomposed_functions, "utility": _u_f}

    with pytest.raises(RegimeInitializationError, match="utility"):
        regime.with_engine_functions(engine_functions=engine_functions)


def test_dropping_a_stakeholder_is_refused():
    """The write-back must reproduce what the regime decomposes to, in full."""
    regime = _declaring_regime()
    engine_functions = {
        name: func
        for name, func in regime.decomposed_functions.items()
        if name != "utility_m"
    }

    with pytest.raises(RegimeInitializationError, match="utility_m"):
        regime.with_engine_functions(engine_functions=engine_functions)


def test_a_delegated_body_may_be_updated_because_it_is_the_regimes_own():
    """A `None` stakeholder's body is the regime's own, so the engine may compose it."""
    regime = _couple(
        functions={
            "utility": CollectiveUtility(utilities={"f": _u_f, "m": None}),
            "utility_m": _u_m,
        }
    )
    engine_functions = {**regime.decomposed_functions, "utility_m": _other_u_m}

    written = regime.with_engine_functions(engine_functions=engine_functions)

    assert written.decomposed_functions["utility_m"] is _other_u_m


def test_a_singleton_regime_is_written_back_wholesale():
    """With no declaration to protect, the engine mapping is the mapping."""
    regime = _couple(functions={"utility": _u_f, "helper": _resources})
    engine_functions = {"utility": _u_f, "resources": _resources}

    written = regime.with_engine_functions(engine_functions=engine_functions)

    assert dict(written.functions) == engine_functions
