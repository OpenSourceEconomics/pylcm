"""A gated edge's closed branch may value one thing and realize another.

What a household expects from leaving and what a settlement actually hands it
are two objects, not one. The value the source maximizes against has to be the
expectation — that is what the decision was taken under — while the row a
simulation writes has to be the realization, in the regime and at the state the
settlement really produces.

`EdgeLeg(fallback=Phased(solve=..., simulate=...))` declares them separately:
the solve leg is folded into `Wbar` and prices the decision, and the simulate
leg supplies the regime, the role and the state coordinates a routed row lands
on. A bare `SamePeriodRef` is both, as before.
"""

import jax.numpy as jnp
import numpy as np

from lcm import (
    AgeGrid,
    EdgeLeg,
    GatedEdge,
    LinSpacedGrid,
    Model,
    Phased,
    Regime,
    SamePeriodRef,
    categorical,
    fixed_transition,
)
from lcm.transition import MarkovTransition
from lcm.typing import BoolND, ContinuousState, FloatND, ScalarInt
from tests.conftest import DECIMAL_PRECISION

_WEALTH = LinSpacedGrid(start=0.0, stop=2.0, n_points=3)

# The edge stays open above this wealth, so only the top node keeps the target.
_KEEPS_THE_TARGET_ABOVE = 1.5

# What the settlement actually hands a routed row, as a share of its wealth.
_SETTLEMENT_SHARE = 0.5


@categorical(ordered=False)
class RegimeId:
    worker: ScalarInt
    retired: ScalarInt
    hardship: ScalarInt
    shelter: ScalarInt


def _certain(wealth: ContinuousState) -> FloatND:
    return jnp.ones_like(wealth)


def _zero(wealth: ContinuousState) -> FloatND:
    return jnp.zeros_like(wealth)


def _generous(wealth: ContinuousState) -> FloatND:
    return 10.0 * wealth


def _meagre(wealth: ContinuousState) -> FloatND:
    return wealth


def _lavish(wealth: ContinuousState) -> FloatND:
    return 100.0 * wealth


def _well_off(wealth: ContinuousState) -> BoolND:
    return wealth > _KEEPS_THE_TARGET_ABOVE


def _whole_wealth(wealth: ContinuousState) -> ContinuousState:
    """What she expects to keep: the projection the decision is priced under."""
    return wealth


def _settled_wealth(wealth: ContinuousState) -> ContinuousState:
    """What the settlement hands her: strictly less, and only in simulation."""
    return _SETTLEMENT_SHARE * wealth


def _make_model() -> Model:
    worker = Regime(
        transition={"retired": MarkovTransition(_certain)},
        active=lambda age: age < 1,
        states={"wealth": _WEALTH},
        state_transitions={"wealth": fixed_transition("wealth")},
        functions={"utility": _zero},
        gated_edges={
            "retired": GatedEdge(
                gate=_well_off,
                legs={
                    "only": EdgeLeg(
                        fallback=Phased(
                            solve=SamePeriodRef(
                                regime="hardship",
                                projection={"wealth": _whole_wealth},
                            ),
                            simulate=SamePeriodRef(
                                regime="shelter",
                                stakeholder="guest",
                                projection={"wealth": _settled_wealth},
                            ),
                        )
                    )
                },
            )
        },
    )
    retired = Regime(
        transition=None,
        active=lambda age: age >= 1,
        states={"wealth": _WEALTH},
        functions={"utility": _generous},
    )
    hardship = Regime(
        transition=None,
        active=lambda age: age >= 1,
        states={"wealth": _WEALTH},
        functions={"utility": _meagre},
    )
    shelter = Regime(
        transition=None,
        active=lambda age: age >= 1,
        stakeholders=("guest", "host"),
        states={"wealth": _WEALTH},
        functions={"utility_guest": _lavish, "utility_host": _zero},
    )
    return Model(
        regimes={
            "worker": worker,
            "retired": retired,
            "hardship": hardship,
            "shelter": shelter,
        },
        ages=AgeGrid(start=0, stop=2, step="Y"),
        regime_id_class=RegimeId,
    )


_PARAMS = {
    "worker": {"koopmans_aggregator": {"discount_factor": 1.0}},
    "retired": {},
    "hardship": {},
    "shelter": {},
}


def _simulate(model: Model, solution):
    """Two subjects, one on each side of the gate."""
    return model.simulate(
        params=_PARAMS,
        initial_conditions={
            "wealth": jnp.asarray([1.0, 2.0]),
            "age": jnp.zeros(2),
            "regime_id": jnp.full(2, model.regime_names_to_ids["worker"]),
        },
        period_to_regime_to_V_arr=solution,
        log_level="off",
        seed=0,
    )


def test_the_solved_value_is_priced_under_the_solve_leg() -> None:
    """`worker`'s value folds `hardship`'s value at the whole-wealth projection.

    On the wealth nodes `(0, 1, 2)` the gate closes at the first two, so the
    source's continuation is `V_hardship = (0, 1)` there and `V_retired = 20` at
    the top node. The realized settlement is worth far more (`shelter` pays
    `100 * wealth`), so a solve that read the simulate leg instead would report
    a strictly larger value at both closed nodes.
    """
    model = _make_model()

    solution = model.solve(params=_PARAMS, log_level="off")

    np.testing.assert_array_almost_equal(
        np.asarray(solution[0]["worker"]),
        np.array([0.0, 1.0, 20.0]),
        decimal=DECIMAL_PRECISION,
    )


def test_a_routed_row_lands_in_the_simulate_legs_regime() -> None:
    """The closed row enters `shelter`, the regime the settlement names.

    The solve leg names `hardship`, so a router reading the priced leg would
    put the row there — the regime she expected rather than the one she got.
    """
    model = _make_model()
    solution = model.solve(params=_PARAMS, log_level="off")

    result = _simulate(model, solution)

    np.testing.assert_array_equal(
        np.asarray(result.raw_results["shelter"][1].in_regime), [True, False]
    )
    np.testing.assert_array_equal(
        np.asarray(result.raw_results["hardship"][1].in_regime), [False, False]
    )


def test_a_routed_row_lands_at_the_simulate_legs_state() -> None:
    """The closed row carries `0.5 * wealth`, the settlement's own coordinate."""
    model = _make_model()
    solution = model.solve(params=_PARAMS, log_level="off")

    result = _simulate(model, solution)

    np.testing.assert_array_almost_equal(
        np.asarray(result.raw_results["shelter"][1].states["wealth"])[0],
        _SETTLEMENT_SHARE * 1.0,
        decimal=DECIMAL_PRECISION,
    )


def test_a_routed_row_carries_the_simulate_legs_role() -> None:
    """She becomes the `guest` of the household that took her in.

    The role travels with the realized destination, not with the priced one:
    the solve leg names a singleton regime and carries no role at all.
    """
    model = _make_model()
    solution = model.solve(params=_PARAMS, log_level="off")

    result = _simulate(model, solution)

    assert (
        int(result.raw_results["shelter"][1].own_stakeholder[0])
        == model.stakeholder_names_to_ids["guest"]
    )


def test_the_open_branch_is_untouched_by_the_phase_split() -> None:
    """The row above the gate still retires, at its own wealth."""
    model = _make_model()
    solution = model.solve(params=_PARAMS, log_level="off")

    result = _simulate(model, solution)

    np.testing.assert_array_equal(
        np.asarray(result.raw_results["retired"][1].in_regime), [False, True]
    )
    np.testing.assert_array_almost_equal(
        np.asarray(result.raw_results["retired"][1].states["wealth"])[1],
        2.0,
        decimal=DECIMAL_PRECISION,
    )
