"""Global physical names cannot become source arrival parameters.

The landing target holds no current shock. An explicitly queried fallback can
produce its observed eps; an unrelated regime alone cannot. On nodes [-1, 1],
positive eps keeps target6 and negative eps takes fallback9, giving source15/4
at beta1/2. Explicit read eps=10 remains a different, opaque query operand.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from lcm import (
    AgeGrid,
    ArrivalDecision,
    ArrivalRead,
    ByAge,
    Model,
    ProjectedRegimeValue,
    Regime,
    StakeholderRoute,
    Transition,
    categorical,
)
from lcm.exceptions import RegimeInitializationError, UnsupportedOperationError
from lcm.shocks import Normal, PhysicalValues, ShockGroup, UserNodes
from lcm.typing import BoolND, FloatND, ScalarInt, UserFunction


@categorical(ordered=False)
class _RegimeId:
    source: ScalarInt
    target: ScalarInt
    fallback: ScalarInt
    foreign: ScalarInt


def _zero() -> float:
    return 0.0


def _six() -> float:
    return 6.0


def _ten() -> float:
    return 10.0


def _payoff(eps: FloatND) -> FloatND:
    return 10.0 + eps


def _physical(*, eps: FloatND, V_ref: FloatND) -> BoolND:
    return (eps > 0.0) & (V_ref == 10.0 + eps)


def _physical_code(eps_code: FloatND) -> BoolND:
    return eps_code > 0.0


def _read_only(eps: FloatND) -> BoolND:
    return eps > 5.0


def _model(
    *,
    observed: bool,
    producer: bool,
    predicate: UserFunction = _physical,
    read_name: str = "V_ref",
) -> Model:
    return Model(
        regimes={
            "source": Regime(functions={"utility": _zero}),
            "target": Regime(functions={"utility": _six}),
            "fallback": Regime(
                functions={"utility": _payoff if producer else _ten},
                shocks=("innovation",) if producer else (),
            ),
            "foreign": Regime(functions={"utility": _payoff}, shocks=("innovation",)),
        },
        shocks={
            "innovation": ShockGroup(
                shocks={"eps": Normal(mu=0.0, sigma=1.0)},
                nodes=UserNodes(
                    nodes=np.asarray([[-1.0], [1.0]]),
                    weights=np.asarray([0.5, 0.5]),
                ),
            )
        },
        edges={
            "source": Transition(
                targets={"target": 0, "fallback": 0},
                law=ByAge(cases={0: "target"}),
                arrivals={
                    "target": ArrivalDecision(
                        predicate=predicate,
                        reads={
                            read_name: ArrivalRead(
                                kind="Raw",
                                regime="fallback",
                                projection={},
                                observes=("eps",) if observed else (),
                            )
                        },
                        routes={
                            "only": StakeholderRoute(
                                fallback=ProjectedRegimeValue(
                                    regime="fallback", projection={}
                                )
                            )
                        },
                        observes=("eps",) if observed else (),
                    )
                },
            )
        },
        ages=AgeGrid(exact_values=(0, 1)),
        initial_nodes={0: "source"},
        regime_id_class=_RegimeId,
        enable_jit=False,
    )


def test_foreign_physical_input_cannot_become_an_arrival_edge_parameter() -> None:
    with pytest.raises(
        RegimeInitializationError, match=r"Arrival callback.*unobserved.*eps"
    ):
        _model(observed=False, producer=False)


def test_declared_foreign_observation_requires_an_actual_query_producer() -> None:
    with pytest.raises(
        ValueError, match=r"no producing context for observed physical inputs.*eps"
    ):
        _model(observed=True, producer=False)


def test_arrival_observes_physical_input_from_its_actual_read_producer() -> None:
    model = _model(observed=True, producer=True)
    assert "edges" not in model.get_params_template()
    solution = model.solve(params={"discount_factor": 0.5}, log_level="off")
    np.testing.assert_array_equal(solution.values[0]["source"], 3.75)
    frame = model.simulate(
        params={"discount_factor": 0.5},
        solution=solution,
        initial_conditions={
            "age": jnp.array([0.0, 0.0]),
            "regime_id": jnp.array(
                [_RegimeId.source, _RegimeId.source], dtype=jnp.int32
            ),
        },
        shock_paths={
            "eps": PhysicalValues(values=jnp.array([[2.0, 2.0], [-2.0, -2.0]]))
        },
        seed=0,
        log_level="off",
    ).to_dataframe()
    source = frame.loc[frame["period"] == 0].sort_values("subject_id")
    np.testing.assert_array_equal(source["subject_id"], [0, 1])
    np.testing.assert_array_equal(source["value"], [3.75, 3.75])
    landed = frame.loc[frame["period"] == 1].sort_values("subject_id")
    np.testing.assert_array_equal(landed["subject_id"], [0, 1])
    assert landed["regime_name"].tolist() == ["target", "fallback"]
    np.testing.assert_array_equal(landed["value"], [6.0, 8.0])
    fallback = landed.loc[landed["regime_name"] == "fallback"]
    np.testing.assert_array_equal(fallback["eps"], [-2.0])


def test_foreign_continuous_physical_code_keeps_its_domain_refusal() -> None:
    with pytest.raises(
        UnsupportedOperationError, match=r"eps_code.*continuous physical.*no quadrature"
    ):
        _model(observed=True, producer=True, predicate=_physical_code)


def test_explicit_arrival_read_shadows_a_global_physical_name() -> None:
    model = _model(
        observed=False, producer=False, predicate=_read_only, read_name="eps"
    )
    assert "edges" not in model.get_params_template()
    solution = model.solve(params={"discount_factor": 0.5}, log_level="off")
    np.testing.assert_array_equal(solution.values[0]["source"], 3.0)
