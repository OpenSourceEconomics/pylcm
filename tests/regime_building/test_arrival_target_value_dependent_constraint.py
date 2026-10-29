"""An arrival prices a target whose participation constraint reads another regime.

Married lets the wife work (her utility 1.25) or not (0.75); the husband gets 100
only if she does not work. Her participation constraint against single's same-
date value 1.0 leaves only work feasible, so her married value is 1.25. The
arrival keeps the marriage where her married value exceeds single's, so young's
value is 1.25. Ignoring the constraint makes her married value 0.75, the arrival
rejects, and young's value is single's 1.0.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from lcm import (
    AgeGrid,
    ArrivalDecision,
    ArrivalRead,
    ByAge,
    CollectiveUtility,
    DiscreteGrid,
    Model,
    ProjectedRegimeValue,
    Regime,
    StakeholderRoute,
    Transition,
    ValueDependentConstraint,
    categorical,
)
from lcm.arrivals import ArrivalValueKind
from lcm.transition import StochasticTransition
from lcm.typing import BoolND, DiscreteAction, FloatND, ScalarInt


@categorical(ordered=False)
class _Work:
    off: ScalarInt
    on: ScalarInt


@categorical(ordered=False)
class _RegimeId:
    young: ScalarInt
    married: ScalarInt
    single: ScalarInt


def _zero() -> FloatND:
    return jnp.asarray(0.0)


def _single() -> FloatND:
    return jnp.asarray(1.0)


def _wife(work: DiscreteAction) -> FloatND:
    return 0.75 + 0.5 * work


def _husband(work: DiscreteAction) -> FloatND:
    return 100.0 * (1.0 - work)


def _participates(*, Q_f: FloatND, V_ref: FloatND) -> BoolND:
    return Q_f >= V_ref


def _marries(*, V_target_f: FloatND, V_single: FloatND) -> BoolND:
    return V_target_f > V_single


def _one(age: FloatND) -> FloatND:
    return jnp.ones_like(age, dtype=float)


def _model(
    *,
    target_kind: ArrivalValueKind,
    reference_kind: ArrivalValueKind,
    enable_jit: bool,
) -> Model:
    single = ProjectedRegimeValue(regime="single", projection={})
    return Model(
        regimes={
            "young": Regime(functions={"utility": _zero}),
            "married": Regime(
                actions={"work": DiscreteGrid(category_class=_Work)},
                functions={
                    "utility": CollectiveUtility(utilities={"f": _wife, "m": _husband})
                },
                constraints={
                    "participation": ValueDependentConstraint(
                        predicate=_participates, references={"V_ref": single}
                    )
                },
            ),
            "single": Regime(functions={"utility": _single}),
        },
        edges={
            "young": Transition(
                targets={"married": 0, "single": 0},
                law=ByAge(cases={0: {"married": StochasticTransition(func=_one)}}),
                arrivals={
                    "married": ArrivalDecision(
                        predicate=_marries,
                        reads={
                            "V_target_f": ArrivalRead(
                                kind=target_kind,
                                regime="married",
                                stakeholder="f",
                                projection={},
                            ),
                            "V_single": ArrivalRead(
                                kind=reference_kind, regime="single", projection={}
                            ),
                        },
                        routes={
                            "only": StakeholderRoute(
                                target_stakeholder="f", fallback=single
                            )
                        },
                    )
                },
            )
        },
        ages=AgeGrid(start=0, inclusive_stop=1, step="Y"),
        initial_nodes={0: "young"},
        regime_id_class=_RegimeId,
        enable_jit=enable_jit,
    )


@pytest.mark.parametrize("enable_jit", [False, True], ids=["eager", "jit"])
@pytest.mark.parametrize(
    ("target_kind", "reference_kind"),
    [
        ("Participating", "Raw"),
        ("Raw", "Raw"),
        ("Participating", "IntegratedFull"),
        ("IntegratedFull", "IntegratedFull"),
    ],
)
def test_arrival_prices_a_target_constraint_on_another_regimes_value(
    *,
    target_kind: ArrivalValueKind,
    reference_kind: ArrivalValueKind,
    enable_jit: bool,
) -> None:
    model = _model(
        target_kind=target_kind, reference_kind=reference_kind, enable_jit=enable_jit
    )
    params = {
        "young": {"koopmans_aggregator": {"discount_factor": 1.0}},
        "married": {},
        "single": {},
    }
    values = model.solve(params=params, log_level="off").values
    np.testing.assert_array_equal(values[1]["married"], [1.25, 0.0])
    np.testing.assert_array_equal(values[0]["young"], 1.25)
