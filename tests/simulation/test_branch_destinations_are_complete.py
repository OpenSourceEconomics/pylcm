"""Each branch of a gated edge leaves a row in a complete, named place.

A leg says four things about where its rows go — the open branch's regime and
role, and the closed branch's — and simulation has to carry all four. Carrying
only the regime leaves a row in the right household under the wrong identity,
and the next edge it meets then routes it down the other partner's leg.

Roles are model-wide integer codes, so two collective regimes may name entirely
different roles: the miniature below sends the wife's closed branch into a
`care_pair` as its `carer` — a role neither the source nor the open target
declares — and the husband's into a singleton, where he occupies none.
"""

import jax.numpy as jnp
import numpy as np

from _lcm.regime_building.collective import NO_ROLE
from lcm import (
    AgeGrid,
    EdgeLeg,
    GatedEdge,
    IrregSpacedGrid,
    Model,
    Regime,
    SamePeriodRef,
    categorical,
    fixed_transition,
)
from lcm.transition import MarkovTransition
from lcm.typing import BoolND, ContinuousState, FloatND, ScalarInt

_WAGE = IrregSpacedGrid(points=(1.0, 2.0))

# The gate opens above this wage, so `wage = 1` closes it and `wage = 2` opens it.
_STAY_TOGETHER_ABOVE = 1.5


@categorical(ordered=False)
class RegimeId:
    household: ScalarInt
    household_next: ScalarInt
    care_pair: ScalarInt
    lodging: ScalarInt


def _certain(wage: ContinuousState) -> FloatND:
    return jnp.ones_like(wage)


def _zero(wage: ContinuousState) -> FloatND:
    return jnp.zeros_like(wage)


def _wage_payoff(wage: ContinuousState) -> FloatND:
    return wage


def _identity_wage(wage: ContinuousState) -> ContinuousState:
    return wage


def _prosperous_enough(wage: ContinuousState) -> BoolND:
    return wage > _STAY_TOGETHER_ABOVE


def _make_model() -> Model:
    household = Regime(
        transition={"household_next": MarkovTransition(_certain)},
        active=lambda age: age < 1,
        stakeholders=("f", "m"),
        states={"wage": _WAGE},
        state_transitions={"wage": fixed_transition("wage")},
        functions={"utility_f": _zero, "utility_m": _zero},
        gated_edges={
            "household_next": GatedEdge(
                gate=_prosperous_enough,
                legs={
                    "f": EdgeLeg(
                        target_stakeholder="f",
                        fallback=SamePeriodRef(
                            regime="care_pair",
                            stakeholder="carer",
                            projection={"wage": _identity_wage},
                        ),
                    ),
                    "m": EdgeLeg(
                        target_stakeholder="m",
                        fallback=SamePeriodRef(
                            regime="lodging",
                            projection={"wage": _identity_wage},
                        ),
                    ),
                },
            )
        },
    )
    household_next = Regime(
        transition=None,
        active=lambda age: age >= 1,
        stakeholders=("f", "m"),
        states={"wage": _WAGE},
        functions={"utility_f": _wage_payoff, "utility_m": _wage_payoff},
    )
    care_pair = Regime(
        transition=None,
        active=lambda age: age >= 1,
        stakeholders=("carer", "ward"),
        states={"wage": _WAGE},
        functions={"utility_carer": _wage_payoff, "utility_ward": _zero},
    )
    lodging = Regime(
        transition=None,
        active=lambda age: age >= 1,
        states={"wage": _WAGE},
        functions={"utility": _zero},
    )
    return Model(
        regimes={
            "household": household,
            "household_next": household_next,
            "care_pair": care_pair,
            "lodging": lodging,
        },
        ages=AgeGrid(start=0, stop=2, step="Y"),
        regime_id_class=RegimeId,
    )


_PARAMS = {
    "household": {"koopmans_aggregator": {"discount_factor": 1.0}},
    "household_next": {},
    "care_pair": {},
    "lodging": {},
}


def _simulate(*, wage: float):
    """Two rows — the wife and the husband of one household — at `wage`."""
    model = _make_model()
    solution = model.solve(params=_PARAMS, log_level="off")
    roles = model.stakeholder_names_to_ids
    result = model.simulate(
        params=_PARAMS,
        initial_conditions={
            "wage": jnp.full(2, wage),
            "age": jnp.zeros(2),
            "regime_id": jnp.full(2, model.regime_names_to_ids["household"]),
            "own_stakeholder": jnp.asarray([roles["f"], roles["m"]], dtype=jnp.int32),
        },
        period_to_regime_to_V_arr=solution,
        log_level="off",
        seed=0,
    )
    return result, roles


def test_the_role_vocabulary_spans_every_regimes_stakeholders() -> None:
    """One vocabulary covers `f`, `m`, `carer` and `ward` with distinct codes.

    Two collective regimes naming different roles have to remain
    distinguishable; per-regime numbering would give `carer` the same code as
    `f` and make a routed role mean two things.
    """
    model = _make_model()

    assert set(model.stakeholder_names_to_ids) == {"f", "m", "carer", "ward"}
    assert len(set(model.stakeholder_names_to_ids.values())) == 4


def test_a_closed_branch_leaves_each_row_in_its_own_legs_regime() -> None:
    """At `wage = 1` she enters `care_pair` and he enters `lodging`."""
    result, _roles = _simulate(wage=1.0)

    np.testing.assert_array_equal(
        np.asarray(result.raw_results["care_pair"][1].in_regime), [True, False]
    )
    np.testing.assert_array_equal(
        np.asarray(result.raw_results["lodging"][1].in_regime), [False, True]
    )


def test_a_closed_branch_gives_a_row_its_destinations_own_role() -> None:
    """She becomes the `carer`, and he becomes nobody's stakeholder.

    Her closed destination is collective under a vocabulary neither the source
    nor the open target uses, so the role she carries afterwards can come only
    from the leg's own `fallback.stakeholder`. His is a singleton, so the role
    he carries is the no-role sentinel rather than the `m` he arrived with.
    """
    result, roles = _simulate(wage=1.0)

    assert int(result.raw_results["care_pair"][1].own_stakeholder[0]) == roles["carer"]
    assert int(result.raw_results["lodging"][1].own_stakeholder[1]) == NO_ROLE


def test_an_open_branch_gives_a_row_the_targets_own_role() -> None:
    """At `wage = 2` both stay, she as `f` and he as `m`."""
    result, roles = _simulate(wage=2.0)

    np.testing.assert_array_equal(
        np.asarray(result.raw_results["household_next"][1].in_regime), [True, True]
    )
    np.testing.assert_array_equal(
        np.asarray(result.raw_results["household_next"][1].own_stakeholder),
        [roles["f"], roles["m"]],
    )
