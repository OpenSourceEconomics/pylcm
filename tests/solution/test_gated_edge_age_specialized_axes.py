"""A gated edge folds the target's value on that period's own state grid.

A `GatedEdge` turns a target regime's value into the source's continuation
`Wbar`, choosing cell by cell between the target's own value (gate open) and a
fallback regime's value at a projected state (gate closed). Gate and projection
are both evaluated on the target's state grid, so the grid they see must be the
one the target was solved on in the period whose value is being folded — not the
grid of some other age at which the target happens to be active.

An `AgeSpecializedGrid` is what makes the two distinguishable. It holds
`n_points` fixed while the bounds move with age, so every array in the fold has
the same shape at every age and no shape check can separate the periods; only
the folded value can. Solve and simulate each build the continuation, so each
owes the same answer.
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

# The `account` balance grid runs from zero to the age's contribution cap. The
# cap tightens once, so the grid's two nodes differ between the two ages at
# which `account` is active.
_CAP_EARLY = 10.0
_CAP_LATE = 4.0

# The saver keeps the account only while the balance clears this floor.
_MIN_BALANCE = 5.0

# The balance the saver carries into the account.
_ENTRY_BALANCE = 4.0

# The annuity's principal grid, `[0, 4, 8]`, so a rolled-over balance of
# `_ENTRY_BALANCE` lands on a node rather than between two.
_ANNUITY_GRID = LinSpacedGrid(start=0.0, stop=8.0, n_points=3)
_ANNUITY_BASE = 100.0

# `saver`'s value at the age at which it opens the account. See `_build_model`.
_EXPECTED_SAVER_V_AT_AGE_1 = 52.0

# The account is active at two ages and the cap differs between them, so its
# balance grid resolves to two distinct sets of nodes.
_N_DISTINCT_ACCOUNT_GRIDS = 2


@categorical(ordered=False)
class _RegimeId:
    """Regime ids of the saver / account / annuity model."""

    saver: ScalarInt  # code 0
    account: ScalarInt  # code 1
    annuity: ScalarInt  # code 2


def test_gated_edge_fold_reads_the_target_at_its_own_period_axes() -> None:
    """`saver`'s solved value at age 1 prices `account` on age 2's balance grid.

    At age 2 the contribution cap has tightened to `4`, so no balance on the
    account's grid clears the `5` the gate demands and the whole continuation is
    the annuity's. Reading the account on age 1's wider grid instead would find a
    balance that clears the gate and price the top cell at the account's own
    value.
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


def test_simulated_gated_continuation_reads_the_target_at_its_own_period_axes() -> None:
    """A simulated saver's value at age 1 prices `account` on age 2's grid.

    Simulation rebuilds the gated continuation from the stored value functions
    rather than reusing the solved one, so it owes the same folded value as
    backward induction: an age-1 saver is worth `52`, the discounted annuity
    payout its `4` balance rolls into.
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


def test_fold_compiles_once_when_only_the_target_grid_is_age_specialized() -> None:
    """A moving TARGET grid does not recompile the fold.

    The fold takes the target's value array as a runtime argument and needs its
    state NAMES alone, which an `AgeSpecializedGrid` holds fixed. So it closes
    over nothing that moves with age here and compiles once however many ages
    the target is active at.
    """
    edge = _build_model()._regimes["saver"].gated_edges["account"]
    assert len({id(fold) for fold in edge.folds_by_period.values()}) == 1


def test_simulate_gate_evaluator_recompiles_per_distinct_target_grid() -> None:
    """A moving TARGET grid does recompile the simulate gate evaluator.

    Unlike the fold, the evaluator interpolates the target's value array at a
    realized point, so it closes over that grid's nodes and owes one compiled
    object per distinct set of them.
    """
    edge = _build_model()._regimes["saver"].gated_edges["account"]
    assert (
        len({id(ev) for ev in edge.simulate_gate_evaluators_by_period.values()})
        == _N_DISTINCT_ACCOUNT_GRIDS
    )


def _build_model() -> Model:
    """Build a saver whose gated edge reads an account on an age-varying grid.

    Topology over ages 0-3: `saver` is active at ages 0 and 1 and stays put at
    age 0, so it leaves for `account` at age 1 only. `account` is active at ages
    1 and 2 and holds `balance` on a grid running from zero to the age's
    contribution cap. `annuity` is active from age 1 on and is where a balance
    goes when the saver's gated edge closes.

    Hand computation, $\\beta = 0.5$:

    - `account` pays out its balance, so its value is its own grid: `[0, 10]` at
      age 1 and `[0, 4]` at age 2.
    - `annuity` pays $100 + \\text{principal}$ on `[0, 4, 8]`, so its value is
      `[100, 104, 108]`.
    - The age-2 fold: no node of `[0, 4]` clears the gate's `5`, so both cells
      take the annuity at the rolled-over principal, `Wbar = [100, 104]`.
    - The saver enters at balance `4`, the top node of age 2's grid, so its age-1
      value is $0 + 0.5 \\cdot 104 = 52$.
    - At age 0 the saver stays put with probability one, so its value is
      $0.5 \\cdot 52 = 26$.

    Returns:
        The model, which `{"discount_factor": 0.5}` solves.

    """
    saver = Regime(
        transition={
            "saver": MarkovTransition(_probability_of_staying_put),
            "account": ValueDependentTransition(
                probability=MarkovTransition(_probability_of_opening_the_account),
                gate=_balance_clears_the_floor,
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
    """The account pays out its balance."""
    return balance


def _annuity_utility(principal: ContinuousState) -> FloatND:
    """The annuity pays a base amount plus the principal rolled into it."""
    return _ANNUITY_BASE + principal


def _balance_clears_the_floor(balance: ContinuousState) -> BoolND:
    """The saver keeps the account only while the balance clears the floor."""
    return balance > _MIN_BALANCE


def _principal_from_balance(balance: ContinuousState) -> ContinuousState:
    """A closed account rolls its whole balance into the annuity's principal."""
    return balance
