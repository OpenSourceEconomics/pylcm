"""Validate exit and gate fallback coordinates independently of the move taken."""

import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

from lcm import (
    ExecutionConfig,
    Gate,
    InitialNodes,
    IrregSpacedGrid,
    MemberRoute,
    Model,
    Periods,
    ProjectedRegimeValue,
    Regime,
    TauchenAR1Process,
    Transition,
    categorical,
)
from lcm.typing import BoolND, DiscreteState, FloatND, ScalarInt
from tests.simulation.test_realised_dissolution_route import (
    _PARAMS,
    _build_model,
)

pytestmark = pytest.mark.coverage(backends=("cpu",), precisions="both")


def _replay_wages(*, model: Model, wages: tuple[float, ...]) -> pd.DataFrame:
    solution = model.solve(params=_PARAMS, log_level="off")
    return (
        model.simulate(
            params=_PARAMS,
            solution=solution,
            initial_conditions={
                "wage": jnp.asarray(wages),
                "age": jnp.zeros(len(wages)),
                "regime_id": jnp.full(
                    len(wages), model.regime_names_to_ids["source"], jnp.int32
                ),
                "own_member": jnp.full(
                    len(wages), model.member_names_to_ids["m"], jnp.int32
                ),
            },
            seed=0,
            log_level="off",
        )
        .to_dataframe(use_categorical_codes=True)
        .query("period == 1")
        .sort_values("agent_id")
    )


@pytest.mark.parametrize(
    ("enable_jit", "budget", "width"),
    [(False, None, None), (True, None, 1), (True, 2**28, 2)],
    ids=["eager", "compiled-one-row", "budgeted-two-row"],
)
def test_staying_and_leaving_rows_validate_their_actual_composition_action_projection(
    *, enable_jit: bool, budget: int | None, width: int | None
) -> None:
    """Positive-domain valid exit projections cannot reject a staying row's buffer."""
    model = _build_model(
        payoff_name="ramp",
        fallback_grid=IrregSpacedGrid(
            points=np.array([1.0, 2.0, 3.0]), off_grid="raise"
        ),
        enable_jit=enable_jit,
        execution_config=ExecutionConfig(
            device_memory_bytes=budget,
            axis_widths={} if width is None else {"agent": width},
        ),
    )
    panel = _replay_wages(model=model, wages=(3.0, 1.0, 2.0))
    # The wife accepts exactly when wage-1 > 1/2. Every actual exit coordinate
    # lies in [1,3], including those priced for staying rows.
    assert panel["regime_name"].tolist() == ["pair", "alone_m", "pair"]
    np.testing.assert_array_equal(panel["wage"], [3.0, 1.0, 2.0])


@categorical(ordered=False)
class _ProcessFallbackRegimes:
    source: ScalarInt
    target: ScalarInt
    fallback: ScalarInt


def _zero() -> float:
    return 0.0


def _five() -> float:
    return 5.0


def _node_two() -> float:
    return 2.0


def _node_value(*, z_code: DiscreteState) -> FloatND:
    return 1.0 + z_code


def _accept() -> BoolND:
    return jnp.asarray(a=True)


def _reject() -> BoolND:
    return jnp.asarray(a=False)


@pytest.mark.parametrize("accept", [True, False])
def test_process_code_validation_uses_the_actual_fallback_node(*, accept: bool) -> None:
    """An inactive zero is off nodes [1,2,3], while the priced projection is node 2."""
    model = Model(
        regimes={
            "source": Regime(functions={"utility": _zero}),
            "target": Regime(functions={"utility": _five}),
            "fallback": Regime(
                states={
                    "z": TauchenAR1Process.on_nodes(
                        nodes=np.array([1.0, 2.0, 3.0]),
                        rho=0.0,
                        sigma=1.0,
                        simulated_draws="nodes",
                    )
                },
                functions={"utility": _node_value},
            ),
        },
        n_periods=2,
        regime_id_class=_ProcessFallbackRegimes,
        initial_nodes=InitialNodes(by_period={0: "source"}),
        edges={
            "source": Transition(
                targets={
                    "target": Periods(values=(0,)),
                    "fallback": Periods(values=(0,)),
                },
                law="target",
                gates={
                    "target": Gate(
                        predicate=_accept if accept else _reject,
                        routes={
                            "only": MemberRoute(
                                fallback=ProjectedRegimeValue(
                                    regime="fallback", projection={"z": _node_two}
                                )
                            )
                        },
                    )
                },
            )
        },
        execution_config=ExecutionConfig(device_memory_bytes=None),
    )
    params = {"discount_factor": 1.0}
    solution = model.solve(params=params, log_level="off")
    np.testing.assert_array_equal(solution.values[0]["source"], 5.0 if accept else 2.0)
    panel = model.simulate(
        params=params,
        solution=solution,
        initial_conditions={
            "period": jnp.asarray([0], dtype=jnp.int32),
            "regime_id": jnp.asarray([0], dtype=jnp.int32),
        },
        seed=0,
        log_level="off",
    ).to_dataframe(use_categorical_codes=True)
    landed = panel.query("period == 1")
    assert landed["regime_name"].tolist() == ["target" if accept else "fallback"]
    np.testing.assert_array_equal(landed["value"], [5.0 if accept else 2.0])
    if not accept:
        np.testing.assert_array_equal(landed["z"], [2.0])
