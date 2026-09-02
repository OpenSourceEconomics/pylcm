"""A gated edge reads its gate reference on that period's own state grid.

A `GatedEdge`'s gate may read another regime's same-period value at a projected
coordinate (`gate_refs`). That value is interpolated on the REFERENCE regime's
grid, so the grid the interpolator measures coordinates against must be the one
the reference regime was solved on in the period whose value is being folded —
not the grid of some other age at which the reference happens to be active.

An `AgeSpecializedGrid` on the REFERENCE regime is what makes the two
distinguishable. It holds `n_points` fixed while the bounds move with age, so
every array in the fold has the same shape at every age and no shape check can
separate the periods. The target regime and the leg's fallback regime both carry
age-invariant grids here, so the reference regime's own axes are the only thing
under test — and what they decide is the gate, a discrete branch: reading them
at the wrong age hands the source the fallback's value instead of the target's.
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

# The rating index's `level` grid runs from zero to the age's ceiling on three
# nodes. The ceiling drops once, so the nodes differ between the ages at which
# the index is active: `[0, 4, 8]` early, `[0, 2, 4]` late.
_LEVEL_CEILING_EARLY = 8.0
_LEVEL_CEILING_LATE = 4.0

# The account's own balance grid, `[0, 4]`, is the same at every age.
_BALANCE_GRID = LinSpacedGrid(start=0.0, stop=4.0, n_points=2)

# The gate keeps the account open only where the index's value clears this
# hurdle. On age 2's `[0, 2, 4]` a balance of `4` reads an index value of `4`
# and clears it; on age 1's `[0, 4, 8]` the same balance reads `2` and does not.
_INDEX_HURDLE = 3.0

# The annuity pays this plus the principal a closed account rolls into it, so a
# closed gate is worth far more here than the account itself.
_ANNUITY_BASE = 100.0
_ANNUITY_GRID = LinSpacedGrid(start=0.0, stop=8.0, n_points=3)

# The balance the saver carries into the account.
_ENTRY_BALANCE = 4.0

# `saver`'s value at the age at which it opens the account. See `_build_model`.
_EXPECTED_SAVER_V_AT_AGE_1 = 2.0


@categorical(ordered=False)
class _RegimeId:
    """Regime ids of the saver / account / index / annuity model."""

    saver: ScalarInt  # code 0
    account: ScalarInt  # code 1
    index: ScalarInt  # code 2
    annuity: ScalarInt  # code 3


def test_gated_edge_gate_ref_is_read_at_its_own_period_axes() -> None:
    """`saver`'s solved value at age 1 reads `index` on age 2's level grid.

    At age 2 the index's grid is `[0, 2, 4]`, so a level of `4` is its top node
    and worth `4` — above the hurdle, so the gate stays open and the saver's
    continuation is the account's own value. Measuring that level against the
    index's age-1 grid `[0, 4, 8]` instead places it on the MIDDLE node, worth
    `2`, which shuts the gate and hands the saver the annuity's value.
    """
    model = _build_model()
    solution = model.solve(
        params={"discount_factor": _DISCOUNT_FACTOR}, log_level="debug"
    )
    aaae(
        np.asarray(solution.values[1]["saver"]),
        _EXPECTED_SAVER_V_AT_AGE_1,
        decimal=DECIMAL_PRECISION,
    )


def test_simulated_gated_edge_gate_ref_is_read_at_its_own_period_axes() -> None:
    """A simulated saver's value at age 1 reads `index` on age 2's grid.

    Simulation rebuilds the gated continuation from the stored value functions
    rather than reusing the solved one, so it owes the same folded value as
    backward induction: an age-1 saver is worth `2`.
    """
    model = _build_model()
    solution = model.solve(
        params={"discount_factor": _DISCOUNT_FACTOR}, log_level="off"
    )
    result = model.simulate(
        params={"discount_factor": _DISCOUNT_FACTOR},
        initial_conditions=_initial_conditions(n_subjects=2),
        solution=solution,
        log_level="debug",
    )
    simulated = result.to_dataframe().query("regime_name == 'saver' and period == 1")
    aaae(
        simulated["value"].to_numpy(),
        np.full(2, _EXPECTED_SAVER_V_AT_AGE_1),
        decimal=DECIMAL_PRECISION,
    )


def test_simulated_gate_routes_the_saver_by_its_own_period_gate() -> None:
    """A saver leaving at age 1 continues in `account`, the open gate's target.

    Routing recomputes the gate at the realized candidate state from the age-2
    arrays, so it reads the index at a level of `4` on age 2's `[0, 2, 4]` and
    clears the hurdle. Reading that level against the index's age-1 grid
    `[0, 4, 8]` instead lands on the middle node, worth `2`, and would send the
    saver to the annuity. Which regime the row occupies is a discrete outcome,
    so it separates the two reads without any tolerance.
    """
    model = _build_model()
    solution = model.solve(
        params={"discount_factor": _DISCOUNT_FACTOR}, log_level="off"
    )
    result = model.simulate(
        params={"discount_factor": _DISCOUNT_FACTOR},
        initial_conditions=_initial_conditions(n_subjects=2),
        solution=solution,
        log_level="debug",
    )
    routed = result.to_dataframe().query("period == 2")
    np.testing.assert_array_equal(
        routed["regime_name"].to_numpy(), np.full(2, "account")
    )


def _build_model() -> Model:
    """Build a saver whose gate reads an index on an age-varying grid.

    Topology over ages 0-3: `saver` is active at ages 0 and 1 and stays put at
    age 0, so it leaves for `account` at age 1 only. `account` is active at ages
    1 and 2 and holds `balance` on the age-invariant grid `[0, 4]`. `index` and
    `annuity` are active from age 1 on; the index's `level` grid runs to the
    age's ceiling, so its nodes move with age, while the annuity's do not.

    Hand computation, $\\beta = 0.5$:

    - `index` pays out its level, so its value is its own grid: `[0, 4, 8]` at
      age 1 and `[0, 2, 4]` at age 2.
    - `account` pays out its balance, so its value is `[0, 4]`.
    - `annuity` pays $100 + \\text{principal}$ on `[0, 4, 8]`, so its value is
      `[100, 104, 108]`.
    - The age-2 fold reads the index at a level equal to the balance: `0` at
      balance `0` and `4` at balance `4`. Only the latter clears the hurdle of
      `3`, so the gate is shut at balance `0` (the annuity's `100`) and open at
      balance `4` (the account's own `4`), giving `Wbar = [100, 4]`.
    - The saver enters at balance `4`, the top node of the account's grid, so
      its age-1 value is $0 + 0.5 \\cdot 4 = 2$.
    - At age 0 the saver stays put with probability one, so its value is
      $0.5 \\cdot 2 = 1$.

    Returns:
        The model, which `{"discount_factor": 0.5}` solves.

    """
    saver = Regime(
        transition={
            "saver": MarkovTransition(_probability_of_staying_put),
            "account": ValueDependentTransition(
                probability=MarkovTransition(_probability_of_opening_the_account),
                gate=_index_clears_the_hurdle,
                routes={
                    "only": StakeholderRoute(
                        fallback=ProjectedRegimeValue(
                            regime="annuity",
                            projection={"principal": _principal_from_balance},
                        )
                    )
                },
                gate_references={
                    "index_value": ProjectedRegimeValue(
                        regime="index",
                        projection={"level": _level_from_balance},
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
        states={"balance": _BALANCE_GRID},
        functions={"utility": _account_utility},
    )
    index = Regime(
        transition=None,
        active=lambda age: age >= 1,
        states={
            "level": AgeSpecializedGrid(build=_level_grid, signature=_level_ceiling)
        },
        functions={"utility": _index_utility},
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
            "index": index,
            "annuity": annuity,
        },
        ages=AgeGrid(start=0, stop=3, step="Y"),
        regime_id_class=_RegimeId,
    )


def _initial_conditions(*, n_subjects: int) -> dict[str, FloatND]:
    """Seed every subject as a saver at age 0; `saver` carries no state."""
    return {
        "age": jnp.zeros(n_subjects),
        "regime_id": jnp.full(n_subjects, _RegimeId.saver, dtype=jnp.int32),
    }


def _level_ceiling(age: float) -> float:
    """The highest level the index's grid reaches at this age."""
    return _LEVEL_CEILING_EARLY if age <= 1 else _LEVEL_CEILING_LATE


def _level_grid(age: float) -> LinSpacedGrid:
    """The index's level grid: zero to the age's ceiling, on three nodes."""
    return LinSpacedGrid(start=0.0, stop=_level_ceiling(age), n_points=3)


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


def _index_utility(level: ContinuousState) -> FloatND:
    """The index pays out its level."""
    return level


def _annuity_utility(principal: ContinuousState) -> FloatND:
    """The annuity pays a base amount plus the principal rolled into it."""
    return _ANNUITY_BASE + principal


def _index_clears_the_hurdle(index_value: FloatND) -> BoolND:
    """The account stays open only where the index's value clears the hurdle."""
    return index_value > _INDEX_HURDLE


def _level_from_balance(balance: ContinuousState) -> ContinuousState:
    """The index is read at a level equal to the account's balance."""
    return balance


def _principal_from_balance(balance: ContinuousState) -> ContinuousState:
    """A closed account rolls its whole balance into the annuity's principal."""
    return balance
