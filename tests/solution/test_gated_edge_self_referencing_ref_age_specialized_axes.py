"""A gate reference naming the gated target is read on that period's own grid.

A `SamePeriodRef` may name the very regime its edge gates. The fold then reads
that regime's value twice over — once as the target it folds, once as the
reference its gate consults — and the reference read goes through an
interpolator that closes over the grid's nodes. So the name the reference
carries must not change which age's nodes the fold uses.

An `AgeSpecializedGrid` is what makes a wrong choice visible: `n_points` stays
fixed while the bounds move with age, so a reference read against another
period's nodes lands on a different node of the same value array and no shape
check can object. The two regimes here are built to be mathematically
interchangeable — `mirror` carries an identical grid and an identical value — so
a difference between them is a difference in bookkeeping, not in the model.
"""

import jax.numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal as aaae

from lcm import (
    AgeGrid,
    AgeSpecializedGrid,
    EdgeLeg,
    GatedEdge,
    LinSpacedGrid,
    MarkovTransition,
    Model,
    Regime,
    SamePeriodRef,
    categorical,
)
from lcm.typing import BoolND, ContinuousState, FloatND, ScalarInt
from tests.conftest import DECIMAL_PRECISION

_DISCOUNT_FACTOR = 0.5

# The account's balance ceiling tightens once, so its three nodes are
# `[0, 4, 8]` at the first age it is active and `[0, 2, 4]` at the second.
_CAP_EARLY = 8.0
_CAP_LATE = 4.0

# The gate keeps the account while the referenced value clears this.
_HURDLE = 3.0

_ANNUITY_GRID = LinSpacedGrid(start=0.0, stop=8.0, n_points=3)
_ANNUITY_BASE = 100.0
_ENTRY_BALANCE = 4.0

# Read on age 2's own nodes `[0, 2, 4]`, the entry balance of `4` is the top
# node, worth `4`, which clears the hurdle: the gate is open and the saver keeps
# the account's own value of `4`.
_EXPECTED_SAVER_V = 2.0

# Read on age 1's nodes `[0, 4, 8]` instead, the same balance is the MIDDLE node
# of age 2's value array, worth `2`, which does not clear the hurdle: the gate
# shuts and the whole cell falls back to the annuity. Named so the test can say
# which wrong answer it is ruling out.
_V_FROM_THE_WRONG_AGE = 52.0


@categorical(ordered=False)
class _RegimeId:
    """Regime ids of the saver / account / mirror / annuity model."""

    saver: ScalarInt
    account: ScalarInt
    mirror: ScalarInt
    annuity: ScalarInt


@pytest.mark.parametrize("reference_regime", ["account", "mirror"])
def test_gate_ref_is_read_at_its_own_period_axes_whichever_regime_it_names(
    reference_regime: str,
) -> None:
    """The saver is worth the same whether its gate reference names target or twin.

    `mirror` duplicates the account's grid and value, so the two references
    answer with the same number at every age. Reading either against the wrong
    age's nodes prices the saver at `52` instead of `2`.
    """
    solution = _build_model(reference_regime=reference_regime).solve(
        params={"discount_factor": _DISCOUNT_FACTOR}, log_level="debug"
    )

    aaae(
        np.asarray(solution[1]["saver"]),
        _EXPECTED_SAVER_V,
        decimal=DECIMAL_PRECISION,
    )


@pytest.mark.parametrize("reference_regime", ["account", "mirror"])
def test_the_saver_is_priced_at_the_continuation_it_actually_receives(
    reference_regime: str,
) -> None:
    """The saver's value is the discounted value of the regime it lands in.

    Waiting pays nothing, so a saver at the age it opens the account is worth
    exactly `beta` times whatever it continues into. The gate decides which
    regime that is, and the router re-decides it from the same predicate, so the
    two answers have to describe one household: a fold whose gate shuts prices
    an annuitant, and a router whose gate opens delivers an accountholder.
    """
    model = _build_model(reference_regime=reference_regime)
    params = {"discount_factor": _DISCOUNT_FACTOR}

    result = model.simulate(
        params=params,
        initial_conditions=_initial_conditions(n_subjects=2),
        period_to_regime_to_V_arr=model.solve(params=params, log_level="off"),
        log_level="debug",
    )

    simulated = result.to_dataframe()
    priced = simulated.query("regime_name == 'saver' and period == 1")["value"]
    received = simulated.query("period == 2")["value"]
    aaae(
        priced.to_numpy(),
        _DISCOUNT_FACTOR * received.to_numpy(),
        decimal=DECIMAL_PRECISION,
    )


def test_a_self_referencing_gate_ref_splits_the_fold_by_target_grid() -> None:
    """A gate reference naming the target obliges the fold to compile per grid.

    The fold interpolates whatever regime its references name. When one of them
    is the target and the target's grid moves with age, the fold closes over
    those moving nodes and owes one compiled object per distinct set of them —
    the same obligation a reference to any other age-specialized regime creates.
    """
    edge = (
        _build_model(reference_regime="account")
        ._regimes["saver"]
        .gated_edges["account"]
    )

    assert len({id(fold) for fold in edge.folds_by_period.values()}) == 2


def _build_model(*, reference_regime: str) -> Model:
    """Build a saver whose gated edge consults `reference_regime` at its gate.

    Topology over ages 0-2: `saver` stays put at age 0 and opens the account at
    age 1. `account` is active at ages 1 and 2 and holds `balance` on a grid
    running to the age's ceiling. `mirror` duplicates that grid under a
    different state name and pays it out identically, so naming it and naming
    `account` are the same model. `annuity` is where a closed account's balance
    rolls.

    Args:
        reference_regime: Which regime the gate's reference names, `"account"`
            (the gated target itself) or `"mirror"` (its interchangeable twin).

    Returns:
        The model, which `{"discount_factor": 0.5}` solves.

    """
    projection = (
        {"balance": _same_balance}
        if reference_regime == "account"
        else {"m_balance": _same_balance}
    )
    saver = Regime(
        transition={
            "saver": MarkovTransition(_probability_of_staying_put),
            "account": MarkovTransition(_probability_of_opening),
        },
        active=lambda age: age < 2,
        state_transitions={"balance": {"account": _entry_balance}},
        functions={"utility": _saver_utility},
        gated_edges={
            "account": GatedEdge(
                gate=_clears_the_hurdle,
                gate_refs={
                    "ref_value": SamePeriodRef(
                        regime=reference_regime, projection=projection
                    )
                },
                legs={
                    "only": EdgeLeg(
                        fallback=SamePeriodRef(
                            regime="annuity",
                            projection={"principal": _same_balance},
                        )
                    )
                },
            )
        },
    )
    account = Regime(
        transition=None,
        active=lambda age: age >= 1,
        states={
            "balance": AgeSpecializedGrid(build=_balance_grid, signature=_cap),
        },
        functions={"utility": _account_utility},
    )
    mirror = Regime(
        transition=None,
        active=lambda age: age >= 1,
        states={
            "m_balance": AgeSpecializedGrid(build=_mirror_grid, signature=_cap),
        },
        functions={"utility": _mirror_utility},
    )
    annuity = Regime(
        transition=None,
        active=lambda age: age >= 1,
        states={"principal": _ANNUITY_GRID},
        functions={"utility": _annuity_utility},
    )
    return Model(
        regimes={
            "saver": saver,
            "account": account,
            "mirror": mirror,
            "annuity": annuity,
        },
        ages=AgeGrid(start=0, stop=2, step="Y"),
        regime_id_class=_RegimeId,
    )


def _initial_conditions(*, n_subjects: int) -> dict[str, FloatND]:
    """Seed every subject as a saver at age 0; `saver` carries no state."""
    return {
        "age": jnp.zeros(n_subjects),
        "regime_id": jnp.full(n_subjects, _RegimeId.saver, dtype=jnp.int32),
    }


def _cap(age: float) -> float:
    """The highest balance the account may hold at this age."""
    return _CAP_EARLY if age <= 1 else _CAP_LATE


def _balance_grid(age: float) -> LinSpacedGrid:
    """The account's balance grid: zero to the age's ceiling, three nodes."""
    return LinSpacedGrid(start=0.0, stop=_cap(age), n_points=3)


def _mirror_grid(age: float) -> LinSpacedGrid:
    """The mirror's grid, identical to the account's at every age."""
    return LinSpacedGrid(start=0.0, stop=_cap(age), n_points=3)


def _probability_of_staying_put(age: FloatND) -> FloatND:
    """The saver waits one period before opening the account."""
    return jnp.where(age < 1.0, 1.0, 0.0)


def _probability_of_opening(age: FloatND) -> FloatND:
    """The saver opens the account at the second age at which it is active."""
    return jnp.where(age < 1.0, 0.0, 1.0)


def _entry_balance() -> ContinuousState:
    """The balance the saver carries into the account."""
    return jnp.asarray(_ENTRY_BALANCE)


def _saver_utility() -> FloatND:
    """Waiting pays nothing, so the saver's value is its continuation alone."""
    return jnp.asarray(0.0)


def _account_utility(balance: ContinuousState) -> FloatND:
    """The account pays out its balance."""
    return balance


def _mirror_utility(m_balance: ContinuousState) -> FloatND:
    """The mirror pays out its balance, exactly as the account does."""
    return m_balance


def _annuity_utility(principal: ContinuousState) -> FloatND:
    """The annuity pays a base amount plus the principal rolled into it."""
    return _ANNUITY_BASE + principal


def _clears_the_hurdle(ref_value: FloatND) -> BoolND:
    """The account stays open while the referenced value clears the hurdle."""
    return ref_value > _HURDLE


def _same_balance(balance: ContinuousState) -> ContinuousState:
    """Both the gate reference and the fallback read the balance unchanged."""
    return balance
