"""A collective gated edge reads its period-`t` operands on period `t`'s axes.

A collective source declares one edge leg per stakeholder, and a collective
target publishes a dissolution flag `D` that no singleton regime has. Both
enlarge what one gated edge closes over: the fold interpolates every leg's own
fallback grid and every gate reference's, and the simulate-side router
interpolates the target's own value AND its `D` array at the realized candidate
state. Each of those grids must be the one its regime was solved on in the
period whose arrays are being read.

An `AgeSpecializedGrid` is what makes two periods distinguishable. It holds
`n_points` fixed while the bounds move with age, so every array keeps its shape
at every age and no shape check can separate the periods — only the coordinate a
value maps to, and through it the gate, which is a discrete branch.

Two models, one per collective-only object:

- `_build_gate_ref_model` puts the moving grid on a GATE REFERENCE and gives the
  two stakeholders different outside options, so a wrongly-aged read moves both
  components of the source's value.
- `_build_dissolution_model` puts it on the collective TARGET, whose `D` the gate
  reads, so a wrongly-aged read maps the realized state onto a different node of
  the same flag array and dissolves a household that consents.
"""

import jax.numpy as jnp
import numpy as np
from numpy.testing import assert_array_almost_equal as aaae

from lcm import (
    AgeGrid,
    AgeSpecializedGrid,
    DiscreteGrid,
    EdgeLeg,
    GatedEdge,
    LinSpacedGrid,
    MarkovTransition,
    Model,
    Regime,
    SamePeriodRef,
    categorical,
    fixed_transition,
)
from lcm.typing import (
    BoolND,
    ContinuousState,
    DiscreteAction,
    FloatND,
    ScalarInt,
)
from tests.conftest import DECIMAL_PRECISION

_DISCOUNT_FACTOR = 0.5

# Both models let one grid's ceiling drop once, so its three nodes are
# `[0, 4, 8]` at age 1 and `[0, 2, 4]` at age 2.
_CEILING_EARLY = 8.0
_CEILING_LATE = 4.0

# The balance the couple carries into the target regime: the top node of age 2's
# grid, and the MIDDLE node of age 1's.
_ENTRY = 4.0

_ANNUITY_GRID = LinSpacedGrid(start=0.0, stop=8.0, n_points=3)
_OUTSIDE_BASE_F = 100.0
_OUTSIDE_BASE_M = 200.0

# The husband's felicity premium in the target regime, so the two stakeholder
# components of every folded quantity differ.
_M_BONUS = 10.0

# The gate keeps the account open only where the rating index clears this.
_INDEX_HURDLE = 3.0

# The wife's participation floor in the dissolution model. Her felicity is `w`,
# so the household's feasible set empties exactly where `w < 3`:
# age 1's `[0, 4, 8]` gives `D = [True, False, False]`, age 2's `[0, 2, 4]`
# gives `D = [True, True, False]`.
_FLOOR_F = 3.0

# The couple's value at the age at which it leaves, per stakeholder. The gate is
# open at the entry point on the period's OWN axes, so each stakeholder discounts
# the target's own component: `0.5 * 4` and `0.5 * (10 + 4)`.
_EXPECTED_COUPLE_V_AT_AGE_1 = np.array([2.0, 7.0])


@categorical(ordered=True)
class _Effort:
    """A collective regime's discrete action; no felicity depends on it."""

    idle: ScalarInt
    busy: ScalarInt


@categorical(ordered=False)
class _GateRefRegimeId:
    """Regime ids of the couple / account / index / annuity model."""

    couple: ScalarInt
    account: ScalarInt
    index: ScalarInt
    annuity_f: ScalarInt
    annuity_m: ScalarInt


@categorical(ordered=False)
class _DissolutionRegimeId:
    """Regime ids of the couple / pair / singles model."""

    couple: ScalarInt
    pair: ScalarInt
    pair_terminal: ScalarInt
    single_f: ScalarInt
    single_m: ScalarInt


def test_collective_source_gate_ref_is_read_at_its_own_period_axes() -> None:
    """Both stakeholders of an age-1 couple are worth `(2, 7)`.

    At age 2 the index's grid is `[0, 2, 4]`, so a level of `4` is its top node
    and worth `4` — above the hurdle of `3`, so the gate stays open and each
    stakeholder's continuation is the account's own component. Measuring that
    level against the index's age-1 grid `[0, 4, 8]` instead places it on the
    MIDDLE node, worth `2`, which shuts the gate and hands the two stakeholders
    their separate annuities, `(52, 102)`.
    """
    model = _build_gate_ref_model()
    solution = model.solve(
        params={"discount_factor": _DISCOUNT_FACTOR}, log_level="debug"
    )
    aaae(
        np.asarray(solution[1]["couple"]),
        _EXPECTED_COUPLE_V_AT_AGE_1,
        decimal=DECIMAL_PRECISION,
    )


def test_simulated_collective_source_gate_ref_is_read_at_its_own_period_axes() -> None:
    """A simulated age-1 couple is worth `(2, 7)` per stakeholder.

    Simulation rebuilds the gated continuation from the stored value functions
    rather than reusing the solved one, so it owes backward induction's folded
    value on every leg of the collective source.
    """
    model = _build_gate_ref_model()
    result = _simulate_gate_ref(model)
    simulated = result.to_dataframe().query("regime_name == 'couple' and period == 1")
    aaae(
        simulated[["value_f", "value_m"]].to_numpy(),
        np.tile(_EXPECTED_COUPLE_V_AT_AGE_1, (2, 1)),
        decimal=DECIMAL_PRECISION,
    )


def test_simulated_collective_gate_routes_the_couple_by_its_own_period_gate() -> None:
    """A couple leaving at age 1 continues in `account`, the open gate's target.

    Routing recomputes the gate at the realized candidate state from the age-2
    arrays, so it reads the index at a level of `4` on age 2's `[0, 2, 4]` and
    clears the hurdle. Reading that level against the index's age-1 grid
    `[0, 4, 8]` instead lands on the middle node, worth `2`, and would send the
    row to its own leg's fallback annuity. Which regime the row occupies is a
    discrete outcome, so it separates the two reads without any tolerance.
    """
    model = _build_gate_ref_model()
    result = _simulate_gate_ref(model)
    routed = result.to_dataframe().query("period == 2")
    np.testing.assert_array_equal(
        routed["regime_name"].to_numpy(), np.full(2, "account")
    )


def test_collective_dissolution_gate_routes_by_its_own_period_flag() -> None:
    """A consenting couple stays in `pair` rather than dissolving.

    `pair`'s flag at age 2 is `[True, True, False]` on its own `[0, 2, 4]` grid,
    so a realized `w` of `4` is its top node and the household consents.
    Measuring that `w` against `pair`'s age-1 grid `[0, 4, 8]` reads the SAME
    flag array at its middle entry, `True`, and would dissolve the household
    into the two single regimes.
    """
    model = _build_dissolution_model()
    result = _simulate_dissolution(model)
    routed = result.to_dataframe().query("period == 2")
    np.testing.assert_array_equal(routed["regime_name"].to_numpy(), np.full(2, "pair"))


def test_collective_dissolution_fold_reads_the_flag_at_its_own_period() -> None:
    """Both stakeholders of an age-1 couple are worth `(2, 7)` under a `D` gate.

    The fold reads the target's flag at its grid NODES, by exact indexing, and
    pairs node `k` of the flag with node `k` of the target's value. Pinning the
    folded value here is what makes the routing assertion above a statement about
    routing alone rather than about a continuation that moved with it.
    """
    model = _build_dissolution_model()
    solution, _flags = model.solve(
        params={"discount_factor": _DISCOUNT_FACTOR, "floor_f": _FLOOR_F},
        log_level="debug",
        return_dissolution_flags=True,
    )
    aaae(
        np.asarray(solution[1]["couple"]),
        _EXPECTED_COUPLE_V_AT_AGE_1,
        decimal=DECIMAL_PRECISION,
    )


def _build_gate_ref_model() -> Model:
    r"""Build a couple whose gate reads an index on an age-varying grid.

    Topology over ages 0-3: `couple` is collective (`stakeholders=("f", "m")`),
    active at ages 0 and 1, and stays put at age 0, so it leaves for `account` at
    age 1 only. `account` is collective too, active at ages 1 and 2, and holds
    `balance` on the age-invariant grid `[0, 4]`. `index` and the two annuities
    are active from age 1 on; the index's `level` grid runs to the age's ceiling,
    so its nodes move with age, while the annuities' do not.

    Hand computation, $\beta = 0.5$:

    - `index` pays out its level, so its value is its own grid: `[0, 4, 8]` at
      age 1 and `[0, 2, 4]` at age 2.
    - `account` pays the balance to the wife and `10 +` the balance to the
      husband, so its value is `[[0, 10], [4, 14]]`.
    - `annuity_f` pays $100 + \text{principal}$ and `annuity_m` pays
      $200 + \text{principal}$ on `[0, 4, 8]`, so their values are
      `[100, 104, 108]` and `[200, 204, 208]`.
    - The age-2 fold reads the index at a level equal to the balance: `0` at
      balance `0` and `4` at balance `4`. Only the latter clears the hurdle of
      `3`, so the gate is shut at balance `0` and open at balance `4`, giving
      `Wbar = [[100, 200], [4, 14]]`.
    - The couple enters at balance `4`, the top node of the account's grid, so
      its age-1 value is $(0, 0) + 0.5 \cdot (4, 14) = (2, 7)$.

    Returns:
        The model, which `{"discount_factor": 0.5}` solves.

    """
    couple = Regime(
        transition={
            "couple": MarkovTransition(_probability_of_staying_put),
            "account": MarkovTransition(_probability_of_leaving),
        },
        active=lambda age: age < 2,
        stakeholders=("f", "m"),
        state_transitions={"balance": {"account": _entry_amount}},
        actions={"effort": DiscreteGrid(_Effort)},
        functions={"utility_f": _no_felicity, "utility_m": _no_felicity},
        gated_edges={
            "account": GatedEdge(
                gate=_index_clears_the_hurdle,
                gate_refs={
                    "index_value": SamePeriodRef(
                        regime="index",
                        projection={"level": _level_from_balance},
                    )
                },
                legs={
                    "f": EdgeLeg(
                        target_stakeholder="f",
                        fallback=SamePeriodRef(
                            regime="annuity_f",
                            projection={"principal": _principal_from_balance},
                        ),
                    ),
                    "m": EdgeLeg(
                        target_stakeholder="m",
                        fallback=SamePeriodRef(
                            regime="annuity_m",
                            projection={"principal": _principal_from_balance},
                        ),
                    ),
                },
            )
        },
    )
    account = Regime(
        transition=None,
        active=lambda age: (age >= 1) & (age < 3),
        stakeholders=("f", "m"),
        states={"balance": LinSpacedGrid(start=0.0, stop=4.0, n_points=2)},
        actions={"effort": DiscreteGrid(_Effort)},
        functions={"utility_f": _account_felicity_f, "utility_m": _account_felicity_m},
    )
    index = Regime(
        transition=None,
        active=lambda age: age >= 1,
        states={
            "level": AgeSpecializedGrid(build=_moving_grid, signature=_moving_ceiling)
        },
        functions={"utility": _index_felicity},
    )
    annuity_f = Regime(
        transition=None,
        active=lambda age: age >= 1,
        states={"principal": _ANNUITY_GRID},
        functions={"utility": _annuity_felicity_f},
    )
    annuity_m = Regime(
        transition=None,
        active=lambda age: age >= 1,
        states={"principal": _ANNUITY_GRID},
        functions={"utility": _annuity_felicity_m},
    )
    return Model(
        regimes={
            "couple": couple,
            "account": account,
            "index": index,
            "annuity_f": annuity_f,
            "annuity_m": annuity_m,
        },
        ages=AgeGrid(start=0, stop=3, step="Y"),
        regime_id_class=_GateRefRegimeId,
    )


def _build_dissolution_model() -> Model:
    r"""Build a couple whose gate reads a collective target's dissolution flag.

    Topology over ages 0-3: `couple` is collective, active at ages 0 and 1, and
    leaves for the collective `pair` at age 1 only. `pair` is active at ages 1
    and 2 and holds `w` on the age-varying grid, so its own nodes — and the nodes
    of the flag `D` tabulated on them — move with age. `pair_terminal` closes the
    lifecycle at zero felicity, and the two singles are the legs' fallbacks on the
    age-invariant grid `[0, 4, 8]`.

    Hand computation, $\beta = 0.5$:

    - `pair` pays `w` to the wife and `10 + w` to the husband, and its
      continuation is zero, so $Q_f = w$. Her participation constraint
      $Q_f \ge 3$ therefore empties the household's feasible set exactly where
      `w < 3`: `D = [True, False, False]` on age 1's `[0, 4, 8]` and
      `D = [True, True, False]` on age 2's `[0, 2, 4]`.
    - `pair`'s own value is $-\infty$ in each dissolved cell and $(w, 10 + w)$
      elsewhere, so at age 2 it is `[[-inf, -inf], [-inf, -inf], [4, 14]]`.
    - `single_f` pays $100 + s$ and `single_m` pays $200 + s$ on `[0, 4, 8]`.
    - The age-2 fold's gate is `~D = [False, False, True]`, so
      `Wbar = [[100, 200], [102, 202], [4, 14]]`.
    - The couple enters at `w = 4`, the top node of `pair`'s age-2 grid, so its
      age-1 value is $(0, 0) + 0.5 \cdot (4, 14) = (2, 7)$.

    Returns:
        The model, which `{"discount_factor": 0.5, "floor_f": 3.0}` solves.

    """
    couple = Regime(
        transition={
            "couple": MarkovTransition(_probability_of_staying_put),
            "pair": MarkovTransition(_probability_of_leaving),
        },
        active=lambda age: age < 2,
        stakeholders=("f", "m"),
        state_transitions={"w": {"pair": _entry_amount}},
        actions={"effort": DiscreteGrid(_Effort)},
        functions={"utility_f": _no_felicity, "utility_m": _no_felicity},
        gated_edges={
            "pair": GatedEdge(
                gate=_household_consents,
                legs={
                    "f": EdgeLeg(
                        target_stakeholder="f",
                        fallback=SamePeriodRef(
                            regime="single_f", projection={"s": _single_state_from_w}
                        ),
                    ),
                    "m": EdgeLeg(
                        target_stakeholder="m",
                        fallback=SamePeriodRef(
                            regime="single_m", projection={"s": _single_state_from_w}
                        ),
                    ),
                },
            )
        },
    )
    pair = Regime(
        transition={"pair_terminal": MarkovTransition(_certainty)},
        active=lambda age: (age >= 1) & (age < 3),
        stakeholders=("f", "m"),
        states={"w": AgeSpecializedGrid(build=_moving_grid, signature=_moving_ceiling)},
        state_transitions={"w": fixed_transition("w")},
        actions={"effort": DiscreteGrid(_Effort)},
        functions={"utility_f": _pair_felicity_f, "utility_m": _pair_felicity_m},
        value_constraints={"ir_f": _wife_participates},
    )
    pair_terminal = Regime(
        transition=None,
        active=lambda age: age >= 2,
        stakeholders=("f", "m"),
        states={"w": _ANNUITY_GRID},
        actions={"effort": DiscreteGrid(_Effort)},
        functions={"utility_f": _no_felicity_on_w, "utility_m": _no_felicity_on_w},
    )
    single_f = Regime(
        transition=None,
        active=lambda age: age >= 1,
        states={"s": _ANNUITY_GRID},
        functions={"utility": _single_felicity_f},
    )
    single_m = Regime(
        transition=None,
        active=lambda age: age >= 1,
        states={"s": _ANNUITY_GRID},
        functions={"utility": _single_felicity_m},
    )
    return Model(
        regimes={
            "couple": couple,
            "pair": pair,
            "pair_terminal": pair_terminal,
            "single_f": single_f,
            "single_m": single_m,
        },
        ages=AgeGrid(start=0, stop=3, step="Y"),
        regime_id_class=_DissolutionRegimeId,
    )


def _simulate_gate_ref(model: Model):
    """Solve, then simulate two couples from age 0 as the wife's row."""
    solution = model.solve(
        params={"discount_factor": _DISCOUNT_FACTOR}, log_level="off"
    )
    return model.simulate(
        params={"discount_factor": _DISCOUNT_FACTOR},
        initial_conditions=_initial_conditions(
            n_subjects=2, regime_id=_GateRefRegimeId.couple
        ),
        period_to_regime_to_V_arr=solution,
        log_level="debug",
        own_stakeholder="f",
    )


def _simulate_dissolution(model: Model):
    """Solve for values and flags, then simulate two couples from age 0."""
    params = {"discount_factor": _DISCOUNT_FACTOR, "floor_f": _FLOOR_F}
    solution, flags = model.solve(
        params=params, log_level="off", return_dissolution_flags=True
    )
    return model.simulate(
        params=params,
        initial_conditions=_initial_conditions(
            n_subjects=2, regime_id=_DissolutionRegimeId.couple
        ),
        period_to_regime_to_V_arr=solution,
        period_to_regime_to_dissolution_flags=flags,
        log_level="debug",
        own_stakeholder="f",
    )


def _initial_conditions(*, n_subjects: int, regime_id: ScalarInt) -> dict[str, FloatND]:
    """Seed every subject as a couple at age 0; the couple carries no state."""
    return {
        "age": jnp.zeros(n_subjects),
        "regime_id": jnp.full(n_subjects, regime_id, dtype=jnp.int32),
    }


def _moving_ceiling(age: float) -> float:
    """The highest value the age-varying grid reaches at this age."""
    return _CEILING_EARLY if age <= 1 else _CEILING_LATE


def _moving_grid(age: float) -> LinSpacedGrid:
    """The age-varying grid: zero to the age's ceiling, on three nodes."""
    return LinSpacedGrid(start=0.0, stop=_moving_ceiling(age), n_points=3)


def _probability_of_staying_put(age: FloatND) -> FloatND:
    """The couple waits one period before leaving."""
    return jnp.where(age < 1.0, 1.0, 0.0)


def _probability_of_leaving(age: FloatND) -> FloatND:
    """The couple leaves at the second age at which it is active."""
    return jnp.where(age < 1.0, 0.0, 1.0)


def _certainty(age: FloatND) -> FloatND:
    """A regime transition taken with probability one."""
    return jnp.ones_like(age, dtype=float)


def _entry_amount() -> ContinuousState:
    """The amount the couple carries into the regime it leaves for."""
    return jnp.asarray(_ENTRY)


def _no_felicity(effort: DiscreteAction) -> FloatND:
    """Waiting pays nothing, so the couple's value is its continuation alone."""
    return 0.0 * effort


def _no_felicity_on_w(w: ContinuousState, effort: DiscreteAction) -> FloatND:
    """A terminal couple's felicity, zero at every state and action."""
    return 0.0 * w + 0.0 * effort


def _account_felicity_f(balance: ContinuousState, effort: DiscreteAction) -> FloatND:
    """The account pays out its balance to the wife."""
    return balance + 0.0 * effort


def _account_felicity_m(balance: ContinuousState, effort: DiscreteAction) -> FloatND:
    """The account pays the husband a premium on top of the balance."""
    return _M_BONUS + balance + 0.0 * effort


def _pair_felicity_f(w: ContinuousState, effort: DiscreteAction) -> FloatND:
    """The pair pays out `w` to the wife."""
    return w + 0.0 * effort


def _pair_felicity_m(w: ContinuousState, effort: DiscreteAction) -> FloatND:
    """The pair pays the husband a premium on top of `w`."""
    return _M_BONUS + w + 0.0 * effort


def _index_felicity(level: ContinuousState) -> FloatND:
    """The index pays out its level."""
    return level


def _annuity_felicity_f(principal: ContinuousState) -> FloatND:
    """The wife's annuity pays a base amount plus the principal rolled in."""
    return _OUTSIDE_BASE_F + principal


def _annuity_felicity_m(principal: ContinuousState) -> FloatND:
    """The husband's annuity pays a higher base amount plus the principal."""
    return _OUTSIDE_BASE_M + principal


def _single_felicity_f(s: ContinuousState) -> FloatND:
    """The wife's single regime pays a base amount plus her carried state."""
    return _OUTSIDE_BASE_F + s


def _single_felicity_m(s: ContinuousState) -> FloatND:
    """The husband's single regime pays a higher base amount plus his state."""
    return _OUTSIDE_BASE_M + s


def _index_clears_the_hurdle(index_value: FloatND) -> BoolND:
    """The account stays open only where the index's value clears the hurdle."""
    return index_value > _INDEX_HURDLE


def _household_consents(D_target: BoolND) -> BoolND:
    """The couple continues wherever the target household did not dissolve."""
    return ~D_target


def _wife_participates(Q_f: FloatND, floor_f: FloatND) -> BoolND:
    """The pair is feasible only where the wife's value clears her floor."""
    return Q_f >= floor_f


def _level_from_balance(balance: ContinuousState) -> ContinuousState:
    """The index is read at a level equal to the account's balance."""
    return balance


def _principal_from_balance(balance: ContinuousState) -> ContinuousState:
    """A closed account rolls its whole balance into the annuity's principal."""
    return balance


def _single_state_from_w(w: ContinuousState) -> ContinuousState:
    """A dissolved household carries `w` into each stakeholder's single regime."""
    return w
