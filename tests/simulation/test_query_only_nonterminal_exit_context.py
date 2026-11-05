"""Query-only nonterminal policies keep genuine next-period continuation."""

import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.policy_queries import QueryKind
from lcm import (
    CollectiveUtility,
    Consent,
    Counterparty,
    DiscreteGrid,
    ExecutionConfig,
    Exit,
    ExitTarget,
    InitialNodes,
    Model,
    Periods,
    Regime,
    categorical,
)
from lcm.typing import FloatND, IntND, ScalarInt


@categorical(ordered=False)
class _QueryOwners:
    source: ScalarInt
    formation: ScalarInt
    incoming_f: ScalarInt
    incoming_m: ScalarInt
    single_done: ScalarInt
    couple_done: ScalarInt


def _source_flow() -> float:
    return 1.0


def _incoming_flow(*, work: IntND) -> FloatND:
    return 3.0 - work


def _female_flow() -> float:
    return 8.0


def _male_flow() -> float:
    return 7.0


def _single_terminal() -> float:
    return 5.0


def _female_terminal() -> float:
    return 10.0


def _male_terminal() -> float:
    return 20.0


@pytest.mark.coverage(backends=("cpu",), precisions="both")
def test_query_only_nonterminal_raw_and_nested_participating_keep_continuation() -> (
    None
):
    """Consent reads the incoming Raw and formation's nested divorce outside."""
    work = DiscreteGrid(values=np.array([0, 1], dtype=np.int32))
    model = Model(
        regimes={
            "source": Regime(
                functions={"utility": _source_flow},
                exits={
                    "marry": Exit(
                        targets={
                            "self": ExitTarget(
                                regime="formation", role="f", projection={}
                            )
                        },
                        rule=Consent(
                            counterparty=Counterparty(
                                regime="incoming_m", role="m", projection={}
                            )
                        ),
                    )
                },
            ),
            "formation": Regime(
                functions={
                    "utility": CollectiveUtility(
                        utilities={"f": _female_flow, "m": _male_flow}
                    )
                },
                exits={
                    "divorce": Exit(
                        targets={
                            "f": ExitTarget(regime="incoming_f", projection={}),
                            "m": ExitTarget(regime="incoming_m", projection={}),
                        },
                        rule="unilateral",
                    )
                },
            ),
            "incoming_f": Regime(
                actions={"work": work}, functions={"utility": _incoming_flow}
            ),
            "incoming_m": Regime(
                actions={"work": work}, functions={"utility": _incoming_flow}
            ),
            "single_done": Regime(functions={"utility": _single_terminal}),
            "couple_done": Regime(
                functions={
                    "utility": CollectiveUtility(
                        utilities={"f": _female_terminal, "m": _male_terminal}
                    )
                }
            ),
        },
        n_periods=2,
        regime_id_class=_QueryOwners,
        initial_nodes=InitialNodes(by_period={0: "source"}),
        edges={
            "source": {"single_done": Periods(values=(0,))},
            "incoming_f": {"single_done": Periods(values=(0,))},
            "incoming_m": {"single_done": Periods(values=(0,))},
            "formation": {"couple_done": Periods(values=(0,))},
        },
        execution_config=ExecutionConfig(device_memory_bytes=None),
    )
    for owner in ("incoming_f", "incoming_m"):
        regime = model._regimes[owner]
        assert (0, owner) not in model.graph.nodes
        assert 0 in model._query_periods_by_phase["simulation"][owner]
        assert regime.active_periods == ()
        assert 0 in regime.simulation.Q_and_F
        assert not regime.simulation.next_state
        assert not regime.simulation.programs.decision
        assert not regime.simulation.programs.transition
        assert not regime.simulation.programs.route
        assert not regime.simulation.programs.exit_full
        assert regime.simulation.current_iid_draw_plan is None
    full = model._regimes["source"].simulation.exit_queries[0].full
    assert full is not None
    requested = {
        (node.read.kind, node.read.target) for node in full.definitions.values()
    }
    assert (QueryKind.RAW, "incoming_m") in requested
    assert (QueryKind.PARTICIPATING, "incoming_m") in requested
    assert (QueryKind.PARTICIPATING, "incoming_f") in requested
    assert (QueryKind.FULL, "incoming_m") not in requested
    params = {"discount_factor": 1.0}
    solution = model.solve(params=params, log_level="off")
    assert set(solution.values[0]) == {"source"}
    # Incoming max_a(3-a)+5=8; formation's (8+10,7+20) strictly beats
    # both that outside and source Raw1+5=6. Consent therefore returns female18.
    np.testing.assert_allclose(solution.values[0]["source"], 18.0)
    np.testing.assert_allclose(solution.values[1]["single_done"], 5.0)
    np.testing.assert_allclose(solution.values[1]["couple_done"], [10.0, 20.0])
    frame = model.simulate(
        params=params,
        solution=solution,
        initial_conditions={
            "period": jnp.zeros(1, dtype=jnp.int32),
            "regime_id": jnp.zeros(1, dtype=jnp.int32),
        },
        seed=17,
        log_level="off",
    ).to_dataframe(use_categorical_codes=True)
    np.testing.assert_array_equal(frame["regime_name"], ["formation", "couple_done"])
    np.testing.assert_allclose(frame["value_f"], [18.0, 10.0])
    np.testing.assert_allclose(frame["value_m"], [27.0, 20.0])
