"""A gated edge's closed branch may value one thing and realize another.

What a household expects from leaving and what a settlement actually hands it
are two objects, not one. The value the source maximizes against has to be the
expectation — that is what the decision was taken under — while the row a
simulation writes has to be the realization, in the regime and at the state the
settlement really produces.

`StakeholderRoute(fallback=Phased(solve=..., simulate=...))` declares them separately:
the solve leg is folded into `Wbar` and prices the decision, and the simulate
leg supplies the regime, the role and the state coordinates a routed row lands
on. A bare `ProjectedRegimeValue` is both, as before.
"""

import jax.numpy as jnp
import numpy as np

from lcm import (
    AgeGrid,
    ByAge,
    CollectiveUtility,
    Gate,
    LinSpacedGrid,
    Model,
    Phased,
    ProjectedRegimeValue,
    Regime,
    StakeholderRoute,
    Transition,
    categorical,
    fixed_transition,
)
from lcm.transition import StochasticTransition
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
    worker_law = ByAge(cases={0: {"retired": StochasticTransition(func=_certain)}})
    worker_gates = {
        "retired": Gate(
            predicate=_well_off,
            routes={
                "only": StakeholderRoute(
                    fallback=Phased(
                        solve=ProjectedRegimeValue(
                            regime="hardship",
                            projection={"wealth": _whole_wealth},
                        ),
                        simulate=ProjectedRegimeValue(
                            regime="shelter",
                            stakeholder="guest",
                            projection={"wealth": _settled_wealth},
                        ),
                    )
                )
            },
        )
    }
    worker = Regime(
        states={"wealth": _WEALTH},
        state_transitions={"wealth": fixed_transition("wealth")},
        functions={"utility": _zero},
    )
    retired = Regime(
        states={"wealth": _WEALTH},
        functions={"utility": _generous},
    )
    hardship = Regime(
        states={"wealth": _WEALTH},
        functions={"utility": _meagre},
    )
    shelter = Regime(
        states={"wealth": _WEALTH},
        functions={
            "utility": CollectiveUtility(utilities={"guest": _lavish, "host": _zero})
        },
    )
    return Model(
        regimes={
            "worker": worker,
            "retired": retired,
            "hardship": hardship,
            "shelter": shelter,
        },
        ages=AgeGrid(start=0, inclusive_stop=2, step="Y"),
        regime_id_class=RegimeId,
        initial_nodes={0: "worker"},
        edges=Phased(
            solve={
                "worker": Transition(
                    targets={"retired": 0, "hardship": 0},
                    law=worker_law,
                    gates=worker_gates,
                )
            },
            simulate={
                "worker": Transition(
                    targets={"retired": 0, "shelter": 0},
                    law=worker_law,
                    gates=worker_gates,
                )
            },
        ),
    )


_PARAMS = {
    "worker": {"koopmans_aggregator": {"discount_factor": 1.0}},
    "retired": {},
    "hardship": {},
    "shelter": {},
}


def _simulate(*, model: Model, solution):
    """Two subjects, one on each side of the gate."""
    return model.simulate(
        params=_PARAMS,
        initial_conditions={
            "wealth": jnp.asarray([1.0, 2.0]),
            "age": jnp.zeros(2),
            "regime_id": jnp.full(2, model.regime_names_to_ids["worker"]),
        },
        solution=solution,
        log_level="off",
        seed=0,
    )


def test_a_routed_row_lands_in_the_simulate_legs_regime() -> None:
    """The solve leg prices the decision; the simulate leg places the routed row.

    On the wealth nodes `(0, 1, 2)` the gate closes at the first two, so the
    source's continuation is `V_hardship = (0, 1)` there and `V_retired = 20` at
    the top node. The realized settlement is worth far more (`shelter` pays
    `100 * wealth`), so a solve that read the simulate leg would report a
    strictly larger value at both closed nodes.

    The closed row enters `shelter`, the regime the settlement names, at
    `0.5 * wealth` and as the `guest` of the household that took her in; a
    router reading the priced leg would put her in `hardship` without a role.
    The row above the gate still retires, at its own wealth.
    """
    model = _make_model()
    solution = model.solve(params=_PARAMS, log_level="off")

    np.testing.assert_array_almost_equal(
        np.asarray(solution.values[0]["worker"]),
        np.array([0.0, 1.0, 20.0]),
        decimal=DECIMAL_PRECISION,
    )

    result = _simulate(model=model, solution=solution)

    shelter = result.raw_results["shelter"][1]
    np.testing.assert_array_equal(np.asarray(shelter.in_regime), [True, False])
    assert 1 not in result.raw_results["hardship"]
    np.testing.assert_array_almost_equal(
        np.asarray(shelter.states["wealth"])[0],
        _SETTLEMENT_SHARE * 1.0,
        decimal=DECIMAL_PRECISION,
    )
    assert int(shelter.own_stakeholder[0]) == model.stakeholder_names_to_ids["guest"]

    retired = result.raw_results["retired"][1]
    np.testing.assert_array_equal(np.asarray(retired.in_regime), [False, True])
    np.testing.assert_array_almost_equal(
        np.asarray(retired.states["wealth"])[1],
        2.0,
        decimal=DECIMAL_PRECISION,
    )
