"""A gated edge's leg fallback may name a regime whose grid moves with age.

A closed gate projects the target's cell into the fallback regime and reads that
regime's value there. The fallback's own grid decides what the projected
coordinate means, so an `AgeSpecializedGrid` on it must be read at the age whose
value array is being folded — `n_points` is fixed while the bounds move, so a
read against another age's nodes lands on a different point of the same array
with no shape to object to.
"""

import jax.numpy as jnp
import numpy as np
import pandas as pd
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

# The annuity's principal grid runs from zero to the age's ceiling on three
# nodes. The ceiling drops once, so the nodes differ between the ages at which
# the annuity is active: `[0, 4, 8]` early, `[0, 3, 6]` late.
_PRINCIPAL_CEILING_EARLY = 8.0
_PRINCIPAL_CEILING_LATE = 6.0

# The account's own balance grid, `[0, 4]`, is the same at every age.
_BALANCE_GRID = LinSpacedGrid(start=0.0, stop=4.0, n_points=2)

# The gate opens only above this balance, which sits above the account's top
# node. So the gate is shut across the whole target grid and both of `Wbar`'s
# nodes are a fallback read, putting the age-specialized annuity under test at
# two different principals rather than one.
_GATE_HURDLE = 100.0

# The annuity pays this plus the SQUARE of the principal a closed account rolls
# into it. The square is what makes the two ages disagree: it is convex, so
# reading it on `[0, 3, 6]` by interpolation differs from reading it on the
# `[0, 4, 8]` nodes, where a principal of `4` happens to be exact.
_ANNUITY_BASE = 100.0

# The balance the saver carries into the account.
_ENTRY_BALANCE = 4.0

# `saver`'s solved value at the two ages at which it is active. See `_build_model`.
_EXPECTED_SAVER_V_AT_AGE_1 = 59.0
_EXPECTED_SAVER_V_AT_AGE_0 = 29.5

# The annuity's realized payoff at the principal a routed subject lands on:
# `100 + 4 ** 2`. It differs from the `118` the age-2 fold interpolates, because
# the fold reads a grid too coarse to represent a square exactly.
_EXPECTED_ROUTED_ANNUITY_VALUE = 116.0

# The principal a routed subject lands on, the whole balance it rolled over.
_EXPECTED_ROUTED_PRINCIPAL = 4.0


@categorical(ordered=False)
class _RegimeId:
    """Regime ids of the saver / account / annuity model."""

    saver: ScalarInt  # code 0
    account: ScalarInt  # code 1
    annuity: ScalarInt  # code 2


def test_leg_fallback_on_an_age_specialized_reference_is_read_at_its_own_age() -> None:
    """`saver`'s solved value at age 1 reads `annuity` on age 2's principal grid.

    At age 2 the annuity's grid is `[0, 3, 6]`, so a principal of `4` falls
    between two nodes and interpolates to `118`, making the saver worth `59`.
    Measuring that principal against the annuity's age-1 grid `[0, 4, 8]`
    instead makes it an exact node worth `116`, which would make the saver
    worth `58`.
    """
    solution = _build_model().solve(
        params={"discount_factor": _DISCOUNT_FACTOR}, log_level="debug"
    )

    aaae(
        np.asarray(solution[1]["saver"]),
        _EXPECTED_SAVER_V_AT_AGE_1,
        decimal=DECIMAL_PRECISION,
    )


def test_leg_fallback_on_an_age_specialized_reference_discounts_into_the_prior_age():
    """`saver` is worth `29.5` at age 0, half its age-1 value.

    The saver stays put at age 0 with probability one and pays nothing, so its
    value there is the discounted age-1 value and nothing else. This pins the
    fold's result one step further back than the age at which it is taken.
    """
    solution = _build_model().solve(
        params={"discount_factor": _DISCOUNT_FACTOR}, log_level="debug"
    )

    aaae(
        np.asarray(solution[0]["saver"]),
        _EXPECTED_SAVER_V_AT_AGE_0,
        decimal=DECIMAL_PRECISION,
    )


def test_simulated_saver_value_agrees_with_backward_induction() -> None:
    """A simulated saver at period 1 is worth the `59` the solver gives it.

    Simulation rebuilds the gated continuation from the stored value functions
    rather than reusing the solved one, so it owes the same folded value —
    including the same choice of which age's annuity grid to read.
    """
    model = _build_model()
    solution = model.solve(
        params={"discount_factor": _DISCOUNT_FACTOR}, log_level="off"
    )
    result = model.simulate(
        params={"discount_factor": _DISCOUNT_FACTOR},
        initial_conditions=_initial_conditions(n_subjects=2),
        period_to_regime_to_V_arr=solution,
        log_level="debug",
    )

    simulated = result.to_dataframe().query("regime_name == 'saver' and period == 1")
    aaae(
        simulated["value"].to_numpy(),
        np.full(2, _EXPECTED_SAVER_V_AT_AGE_1),
        decimal=DECIMAL_PRECISION,
    )


def test_shut_gate_routes_the_saver_into_the_age_specialized_fallback() -> None:
    """A saver leaving at age 1 continues in `annuity`, the shut gate's fallback.

    Which regime the row occupies is a discrete outcome, so it pins the routing
    without any tolerance.
    """
    routed = _routed_rows()

    np.testing.assert_array_equal(
        routed["regime_name"].to_numpy(), np.full(2, "annuity")
    )


def test_routed_saver_lands_on_the_principal_its_projection_supplies() -> None:
    """A routed subject enters `annuity` at a principal of `4`, its whole balance."""
    routed = _routed_rows()

    aaae(
        routed["principal"].to_numpy(),
        np.full(2, _EXPECTED_ROUTED_PRINCIPAL),
        decimal=DECIMAL_PRECISION,
    )


def test_routed_saver_is_paid_the_annuity_at_its_realized_principal() -> None:
    """The routed row is worth `116`, the annuity's payoff at a principal of `4`.

    A simulated subject sits at its realized principal rather than on a node, so
    it is paid the annuity's exact `100 + 4 ** 2` — not the `118` the fold
    interpolates from age 2's coarser `[0, 3, 6]`.
    """
    routed = _routed_rows()

    aaae(
        routed["value"].to_numpy(),
        np.full(2, _EXPECTED_ROUTED_ANNUITY_VALUE),
        decimal=DECIMAL_PRECISION,
    )


def _routed_rows() -> pd.DataFrame:
    """Simulate the model and return the period-2 rows every saver routes into."""
    model = _build_model()
    solution = model.solve(
        params={"discount_factor": _DISCOUNT_FACTOR}, log_level="off"
    )
    result = model.simulate(
        params={"discount_factor": _DISCOUNT_FACTOR},
        initial_conditions=_initial_conditions(n_subjects=2),
        period_to_regime_to_V_arr=solution,
        log_level="debug",
    )
    return result.to_dataframe().query("period == 2")


def _build_model() -> Model:
    r"""Build a saver whose leg fallback pays out on an age-varying grid.

    Topology over ages 0-3: `saver` is active at ages 0 and 1 and stays put at
    age 0, so it leaves for `account` at age 1 only. `account` is active at ages
    1 and 2 and holds `balance` on the age-invariant grid `[0, 4]`. `annuity` is
    active from age 1 on and holds `principal` on a grid running to the age's
    ceiling, so its nodes move with age.

    Hand computation, $\beta = 0.5$:

    - `annuity` pays $100 + \text{principal}^2$, so its value is
      `[100, 116, 164]` on age 1's `[0, 4, 8]` and `[100, 109, 136]` on age 2's
      `[0, 3, 6]`.
    - `account` pays out its balance, so its value is `[0, 4]`.
    - The gate is shut across the account's whole grid, so the age-2 fold is a
      fallback read at every node: it reads the annuity at a principal equal to
      the balance, giving `100` at balance `0` (an exact node) and
      $109 + \frac{1}{3}(136 - 109) = 118$ at balance `4`. So
      `Wbar = [100, 118]`.
    - The saver enters at balance `4`, the top node of the account's grid, so
      its age-1 value is $0 + 0.5 \cdot 118 = 59$.
    - At age 0 the saver stays put with probability one, so its value is
      $0.5 \cdot 59 = 29.5$.

    Reading the annuity against age 1's `[0, 4, 8]` instead makes a principal of
    `4` an exact node worth `116`, giving `Wbar = [100, 116]` and a saver worth
    `58` at age 1.

    Returns:
        The model, which `{"discount_factor": 0.5}` solves.

    """
    saver = Regime(
        transition={
            "saver": MarkovTransition(_probability_of_staying_put),
            "account": MarkovTransition(_probability_of_opening_the_account),
        },
        active=lambda age: age < 2,
        state_transitions={"balance": {"account": _entry_balance}},
        functions={"utility": _saver_utility},
        gated_edges={
            "account": GatedEdge(
                gate=_balance_clears_the_hurdle,
                legs={
                    "only": EdgeLeg(
                        fallback=SamePeriodRef(
                            regime="annuity",
                            projection={"principal": _principal_from_balance},
                        )
                    )
                },
            )
        },
    )
    account = Regime(
        transition=None,
        active=lambda age: (age >= 1) & (age < 3),
        states={"balance": _BALANCE_GRID},
        functions={"utility": _account_utility},
    )
    annuity = Regime(
        transition=None,
        active=lambda age: age >= 1,
        states={
            "principal": AgeSpecializedGrid(
                build=_principal_grid, signature=_principal_ceiling
            )
        },
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


def _principal_ceiling(age: float) -> float:
    """The highest principal the annuity's grid reaches at this age."""
    return _PRINCIPAL_CEILING_EARLY if age <= 1 else _PRINCIPAL_CEILING_LATE


def _principal_grid(age: float) -> LinSpacedGrid:
    """The annuity's principal grid: zero to the age's ceiling, on three nodes."""
    return LinSpacedGrid(start=0.0, stop=_principal_ceiling(age), n_points=3)


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
    """The account pays out its balance."""
    return balance


def _annuity_utility(principal: ContinuousState) -> FloatND:
    """The annuity pays a base amount plus the square of its principal."""
    return _ANNUITY_BASE + principal**2


def _balance_clears_the_hurdle(balance: ContinuousState) -> BoolND:
    """The account stays open only above a hurdle its grid never reaches."""
    return balance > _GATE_HURDLE


def _principal_from_balance(balance: ContinuousState) -> ContinuousState:
    """A closed account rolls its whole balance into the annuity's principal."""
    return balance
