"""Simulated routing reads the target's own value on that period's state grid.

Deciding which regime a subject occupies next is a separate computation from
the folded continuation `Wbar`: the fold reads the target's value at its grid
NODES, by exact indexing, while routing recomputes the gate at the realized
candidate state and so INTERPOLATES that same value. The interpolation measures
the realized state against the target regime's grid, which must be the grid the
target was solved on in the period whose value is being read.

An `AgeSpecializedGrid` on the target is what makes the two distinguishable. It
holds `n_points` fixed while the bounds move with age, so every array in the
gate has the same shape at every age and no shape check can separate the
periods — only the coordinate the realized state maps to, and through it the
regime the subject ends up in.

The gate here reads the target's VALUE rather than its state, because a state
operand is bound from the realized draw and never touches a grid. The account's
value is convex in its balance, so the same balance read one period's nodes off
returns a different number rather than the same one.
"""

import jax.numpy as jnp
import numpy as np
from numpy.testing import assert_array_almost_equal as aaae

from lcm import (
    AgeGrid,
    AgeSpecializedGrid,
    LinSpacedGrid,
    MarkovTransition,
    Model,
    ProjectedRegimeValue,
    Regime,
    StakeholderRoute,
    ValueDependentTransition,
    categorical,
)
from lcm.typing import BoolND, ContinuousState, FloatND, ScalarInt
from tests.conftest import DECIMAL_PRECISION

_DISCOUNT_FACTOR = 0.5

# The account's balance grid runs from zero to the age's contribution cap on
# two nodes. The cap tightens once, so the nodes differ between the two ages at
# which the account is active: `[0, 10]` early, `[0, 4]` late.
_CAP_EARLY = 10.0
_CAP_LATE = 4.0

# The account pays the square of its balance, so its age-2 value array is
# `[0, 16]`. Read at a balance of `4` on age 2's own `[0, 4]` that is the top
# node, worth `16`; measured against age 1's `[0, 10]` the same balance sits at
# coordinate `0.4` of the very same array and reads `6.4`.
_GATE_FLOOR = 10.0

# The balance the saver carries into the account.
_ENTRY_BALANCE = 4.0

# `saver`'s value at the age at which it opens the account: the fold reads the
# account's value at its NODES, so it is `_ENTRY_BALANCE**2` discounted once,
# whichever grid the ROUTING gate happens to measure against.
_EXPECTED_SAVER_V_AT_AGE_1 = 8.0

_ANNUITY_BASE = 100.0
_ANNUITY_GRID = LinSpacedGrid(start=0.0, stop=8.0, n_points=3)


@categorical(ordered=False)
class _RegimeId:
    """Regime ids of the saver / account / annuity model."""

    saver: ScalarInt  # code 0
    account: ScalarInt  # code 1
    annuity: ScalarInt  # code 2


def test_simulated_gate_reads_the_target_value_at_its_own_period_axes() -> None:
    """A saver leaving at age 1 continues in `account`, the open gate's target.

    At age 2 a balance of `4` is the top node of the account's `[0, 4]` grid, so
    the gate reads the account's own `16` and clears the floor of `10`.
    Measuring that balance against age 1's `[0, 10]` grid instead places it at
    coordinate `0.4` of the same value array, reads `6.4`, shuts the gate and
    sends the saver to the annuity.
    """
    model = _build_model()
    result = _simulate(model)
    routed = result.to_dataframe().query("period == 2")
    np.testing.assert_array_equal(
        routed["regime_name"].to_numpy(), np.full(2, "account")
    )


def test_gated_continuation_is_unchanged_by_which_grid_the_routing_gate_reads() -> None:
    """The age-1 saver is worth `8` whichever grid the routing gate measures on.

    The folded continuation reads the account's value at its grid NODES, by
    exact indexing, so no interpolation and no grid enters it. Pinning it here
    is what makes the routing assertion above a statement about routing alone
    rather than about a continuation that moved with it.
    """
    model = _build_model()
    result = _simulate(model)
    simulated = result.to_dataframe().query("regime_name == 'saver' and period == 1")
    aaae(
        simulated["value"].to_numpy(),
        np.full(2, _EXPECTED_SAVER_V_AT_AGE_1),
        decimal=DECIMAL_PRECISION,
    )


def _simulate(model: Model):
    """Solve, then simulate two savers from age 0."""
    solution = model.solve(
        params={"discount_factor": _DISCOUNT_FACTOR}, log_level="off"
    )
    return model.simulate(
        params={"discount_factor": _DISCOUNT_FACTOR},
        initial_conditions=_initial_conditions(n_subjects=2),
        solution=solution,
        log_level="debug",
    )


def _build_model() -> Model:
    """Build a saver whose gate reads an account's value on an age-varying grid.

    Topology over ages 0-3: `saver` is active at ages 0 and 1 and stays put at
    age 0, so it leaves for `account` at age 1 only. `account` is active at ages
    1 and 2 and holds `balance` on a grid running from zero to the age's
    contribution cap. `annuity` is active from age 1 on and is where the saver
    goes when the gate shuts.

    Hand computation, $\\beta = 0.5$:

    - `account` pays the square of its balance, so its value is `[0, 100]` at
      age 1 and `[0, 16]` at age 2.
    - `annuity` pays $100 + \\text{principal}$ on `[0, 4, 8]`, so its value is
      `[100, 104, 108]`.
    - The age-2 fold reads the account's value at its own nodes: `0` clears
      nothing and takes the annuity's `100`, `16` clears the floor of `10` and
      keeps the account, so `Wbar = [100, 16]`.
    - The saver enters at balance `4`, the top node of age 2's grid, so its
      age-1 value is $0 + 0.5 \\cdot 16 = 8$.

    Returns:
        The model, which `{"discount_factor": 0.5}` solves.

    """
    saver = Regime(
        transition={
            "saver": MarkovTransition(_probability_of_staying_put),
            "account": ValueDependentTransition(
                probability=MarkovTransition(_probability_of_opening_the_account),
                gate=_account_value_clears_the_floor,
                routes={
                    "only": StakeholderRoute(
                        fallback=ProjectedRegimeValue(
                            regime="annuity",
                            projection={"principal": _principal_from_balance},
                        )
                    )
                },
            ),
        },
        active=lambda age: age < 2,
        state_transitions={"balance": {"account": _entry_balance}},
        functions={"utility": _saver_utility},
    )
    account = Regime(
        transition=None,
        active=lambda age: (age >= 1) & (age < 3),
        states={
            "balance": AgeSpecializedGrid(
                build=_balance_grid, signature=_contribution_cap
            )
        },
        functions={"utility": _account_utility},
    )
    annuity = Regime(
        transition=None,
        active=lambda age: age >= 1,
        states={"principal": _ANNUITY_GRID},
        functions={"utility": _annuity_utility},
    )
    return Model(
        regimes={"saver": saver, "account": account, "annuity": annuity},
        ages=AgeGrid(start=0, stop=3, step="Y"),
        regime_id_class=_RegimeId,
    )


def _initial_conditions(*, n_subjects: int) -> dict[str, FloatND]:
    """Seed every subject as a saver at age 0; `saver` carries no state."""
    return {
        "age": jnp.zeros(n_subjects),
        "regime_id": jnp.full(n_subjects, _RegimeId.saver, dtype=jnp.int32),
    }


def _contribution_cap(age: float) -> float:
    """The highest balance the account may hold at this age."""
    return _CAP_EARLY if age <= 1 else _CAP_LATE


def _balance_grid(age: float) -> LinSpacedGrid:
    """The account's balance grid: zero to the age's contribution cap."""
    return LinSpacedGrid(start=0.0, stop=_contribution_cap(age), n_points=2)


def _probability_of_staying_put(age: FloatND) -> FloatND:
    """The saver waits one period before opening the account."""
    return jnp.where(age < 1.0, 1.0, 0.0)


def _probability_of_opening_the_account(age: FloatND) -> FloatND:
    """The saver opens the account at the second age at which it is active."""
    return jnp.where(age < 1.0, 0.0, 1.0)


def _entry_balance() -> ContinuousState:
    """The balance the saver carries into the account."""
    return jnp.asarray(_ENTRY_BALANCE)


def _saver_utility() -> FloatND:
    """Waiting pays nothing, so the saver's value is its continuation alone."""
    return jnp.asarray(0.0)


def _account_utility(balance: ContinuousState) -> FloatND:
    """The account pays the square of its balance."""
    return balance * balance


def _annuity_utility(principal: ContinuousState) -> FloatND:
    """The annuity pays a base amount plus the principal rolled into it."""
    return _ANNUITY_BASE + principal


def _account_value_clears_the_floor(V_target: FloatND) -> BoolND:
    """The account stays open only where its own value clears the floor."""
    return V_target > _GATE_FLOOR


def _principal_from_balance(balance: ContinuousState) -> ContinuousState:
    """A closed account rolls its whole balance into the annuity's principal."""
    return balance
