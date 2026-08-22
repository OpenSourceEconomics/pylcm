"""Construction-time contracts for collective regimes."""

import jax.numpy as jnp
import pytest

from lcm import (
    AgeGrid,
    DiscreteGrid,
    LinSpacedGrid,
    Model,
    categorical,
    fixed_transition,
)
from lcm.exceptions import RegimeInitializationError
from lcm.regime import Regime
from lcm.typing import (
    ContinuousAction,
    DiscreteAction,
    FloatND,
    IntND,
    ScalarInt,
)

# Shared building blocks: a stripped-down couples problem, in which the two
# stakeholders differ only in their disutility of work.


@categorical(ordered=True)
class LaborSupply:
    do_not_work: ScalarInt
    work: ScalarInt


def _utility_f(
    consumption: ContinuousAction,
    labor_supply_f: DiscreteAction,
    match_quality: float,
) -> FloatND:
    """Wife's per-stakeholder felicity (illustrative)."""
    return (
        jnp.log(consumption)
        - 0.3 * (labor_supply_f == LaborSupply.work)
        + match_quality
    )


def _utility_m(
    consumption: ContinuousAction,
    labor_supply_m: DiscreteAction,
    match_quality: float,
) -> FloatND:
    """Husband's per-stakeholder felicity (illustrative)."""
    return (
        jnp.log(consumption)
        - 0.5 * (labor_supply_m == LaborSupply.work)
        + match_quality
    )


_WEALTH = LinSpacedGrid(start=1, stop=10, n_points=5)
_CONSUMPTION = LinSpacedGrid(start=1, stop=5, n_points=5)


@categorical(ordered=False)
class _CoupleRegimeId:
    """Regime ids of the minimal two-regime couples model."""

    married: ScalarInt
    widowed: ScalarInt


def _next_regime_widowed(age: FloatND) -> IntND:
    """The married household enters `widowed` next period."""
    return jnp.full_like(age, _CoupleRegimeId.widowed, dtype=jnp.int32)


def test_declaring_non_terminal_stakeholders_constructs():
    """A non-terminal collective regime constructs.

    Declaring `stakeholders` alongside a regime transition is accepted: the
    per-stakeholder contract is checked at construction, and the constructed
    regime keeps both its stakeholder tuple and its non-terminal status. What a
    collective regime is allowed to route to, and how it solves, is pinned in
    `tests/regime_building/test_nonterminal_collective_solve.py`.
    """

    def _some_transition(_age: float) -> int:
        return 0

    regime = Regime(
        transition=_some_transition,
        stakeholders=("f", "m"),
        states={"wealth": _WEALTH},
        actions={"labor_supply_f": DiscreteGrid(LaborSupply)},
        state_transitions={"wealth": lambda wealth: wealth},
        functions={"utility_f": _utility_f, "utility_m": _utility_m},
    )
    assert regime.stakeholders == ("f", "m")
    assert not regime.terminal


def test_terminal_stakeholders_without_per_stakeholder_utility_is_rejected():
    """A collective regime must carry a `utility_<s>` for every stakeholder.

    Supplying a single `utility` where `utility_f` and `utility_m` are required
    is a configuration error. Completeness is a property of the merged regime —
    a bare `Regime` may still receive functions from a model-level slot — so it
    is reported when the model finalizes its regimes.
    """
    married = Regime(
        transition=_next_regime_widowed,
        active=lambda age: age < 1,
        stakeholders=("f", "m"),
        states={"wealth": _WEALTH},
        state_transitions={"wealth": fixed_transition("wealth")},
        actions={
            "labor_supply_f": DiscreteGrid(LaborSupply),
            "labor_supply_m": DiscreteGrid(LaborSupply),
            "consumption": _CONSUMPTION,
        },
        functions={"utility_f": _utility_f, "utility_m": _utility_m},
    )
    widowed = Regime(
        transition=None,
        active=lambda age: age >= 1,
        stakeholders=("f", "m"),
        states={"wealth": _WEALTH},
        actions={
            "labor_supply_f": DiscreteGrid(LaborSupply),
            "consumption": _CONSUMPTION,
        },
        functions={"utility": _utility_f},
    )

    with pytest.raises(RegimeInitializationError, match="per-stakeholder utility"):
        Model(
            regimes={"married": married, "widowed": widowed},
            ages=AgeGrid(start=0, stop=2, step="Y"),
            regime_id_class=_CoupleRegimeId,
        )


def test_singleton_default_is_untouched():
    """A regime that declares no stakeholders is a plain singleton regime.

    Omitting `stakeholders` leaves the field `None` and construction takes the
    ordinary single-value path — the collective branch is reached only by an
    explicit declaration, never by default.
    """

    def utility(consumption: ContinuousAction) -> FloatND:
        return jnp.log(consumption)

    regime = Regime(
        transition=None,
        states={"wealth": _WEALTH},
        actions={"consumption": _CONSUMPTION},
        functions={"utility": utility},
    )
    assert regime.stakeholders is None


def test_stakeholders_field_default_is_none():
    """`stakeholders` defaults to `None` (the singleton) when omitted."""
    field = Regime.__dataclass_fields__["stakeholders"]
    assert field.default is None
