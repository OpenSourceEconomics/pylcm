"""The simulation contracts the draws the solve contracts, so both read one Q.

Its Q reads the solve's continuation, so a contracted draw is summed into the
next-period value there as well, inside the subject tiles as the solve builds it
inside the cell and action tiles. A process draw whose weights read the subject's
own value, which may lie between the process's nodes, keeps its axis. A source with
a gated edge contracts in neither phase.

With both contracting, the transposed cancellation of
`test_lottery_contraction_domain` makes the simulation choose what the solve chose.
"""

from typing import Literal

import jax.numpy as jnp
import numpy as np
import pytest

from lcm import (
    AgeGrid,
    DiscreteGrid,
    ExecutionConfig,
    Gate,
    LinSpacedGrid,
    Model,
    ProjectedRegimeValue,
    Regime,
    StakeholderRoute,
    StochasticTransition,
    Transition,
    categorical,
)
from lcm.solvers import GridSearch
from lcm.typing import BoolND, FloatND, ScalarInt
from tests.conftest import DECIMAL_PRECISION
from tests.regime_building.test_lottery_contraction import (
    _model,
    _params,
    _planned_contraction,
    _RegimeId,
)
from tests.regime_building.test_lottery_contraction_domain import (
    _assert_all_equal,
    _Binary,
    _equal_probabilities,
    _next_wealth_by_landing,
    _Regimes,
    _transposed_alive_utility,
    _transposed_model,
    _transposed_terminal_utility,
)


@categorical(ordered=False)
class _GatedRegimes:
    alive: ScalarInt
    final: ScalarInt
    gated: ScalarInt
    fallback: ScalarInt


# Two subjects per tile, so five subjects run in several subject tiles.
_NARROW_TILES = ExecutionConfig(axis_widths={"cell": 2, "subject": 2})
_TRANSPOSED_PARAMS = {
    "alive": {"koopmans_aggregator": {"discount_factor": 1.0}},
    "final": {},
}


def _initial_conditions(*, n_subjects: int) -> dict[str, jnp.ndarray]:
    codes = jnp.arange(n_subjects) % 2
    return {
        "wealth": jnp.linspace(0.5, 3.5, n_subjects),
        # Between the process's nodes, where a contracted row would be a node's.
        "income": jnp.linspace(-0.15, 0.2, n_subjects),
        "z": jnp.arange(n_subjects) % 3,
        "q": codes,
        "w": 1 - codes,
        "y": codes,
        "age": jnp.zeros(n_subjects),
        "regime_id": jnp.full(n_subjects, _RegimeId.alive, dtype=jnp.int32),
    }


@pytest.mark.parametrize(
    ("income_kind", "expected"),
    [
        pytest.param("markov", {"z"}, id="process-reading-its-value-keeps-its-axis"),
        pytest.param("iid", {"z", "income"}, id="iid-process-contracts"),
    ],
)
def test_simulation_keeps_a_process_draw_that_reads_an_off_node_value(
    *, income_kind: Literal["markov", "iid"], expected: set[str]
) -> None:
    """A subject's process value may lie between nodes, so its row is not a node's."""
    contraction = _planned_contraction(
        model=_model(final_z="same", solver=GridSearch(), income_kind=income_kind),
        source_name="alive",
        target_name="final",
        current_values_on_nodes=False,
    )
    assert contraction is not None
    assert {lottery.state_name for lottery in contraction.lotteries} == expected


@pytest.mark.parametrize("income_kind", ["markov", "iid"])
def test_simulation_with_contraction_chooses_as_without(
    *, income_kind: Literal["markov", "iid"]
) -> None:
    """Away from cancellation, both routes choose the same actions and values."""
    frames = {}
    for contract in (False, True):
        model = _model(
            final_z="same",
            solver=GridSearch(lottery_contraction=contract),
            income_kind=income_kind,
            execution_config=_NARROW_TILES,
        )
        frames[contract] = model.simulate(
            params=_params(final_z="same"),
            initial_conditions=_initial_conditions(n_subjects=5),
            seed=0,
            log_level="off",
        ).to_dataframe()

    assert (
        frames[True]["labor"].astype(str).tolist()
        == frames[False]["labor"].astype(str).tolist()
    )
    columns = ["consumption", "value"]
    np.testing.assert_allclose(
        frames[True][columns].to_numpy(dtype=float),
        frames[False][columns].to_numpy(dtype=float),
        rtol=10.0**-DECIMAL_PRECISION,
        atol=10.0**-DECIMAL_PRECISION,
    )


@pytest.mark.parametrize(
    ("contract", "landing", "value"),
    [
        pytest.param(False, "low", 0.5, id="enumerated-chooses-the-midpoint"),
        pytest.param(True, "high", 0.25, id="contracted-chooses-the-endpoint"),
    ],
)
def test_simulation_chooses_what_the_solve_chose_under_cancellation(
    *, contract: bool, landing: str, value: float
) -> None:
    """The simulation reads the solve's Q, so it picks the action the solve picked.

    Contracted, the solve values the endpoint above the midpoint; before the
    simulation contracted too, it chose the midpoint there.
    """
    model = _transposed_model(contract=contract)
    solution = model.solve(params=_TRANSPOSED_PARAMS, log_level="off")
    frame = model.simulate(
        params=_TRANSPOSED_PARAMS,
        solution=solution,
        initial_conditions={
            "wealth": jnp.array([0.0, 1.0, 2.0]),
            "z": jnp.array([0, 1, 0]),
            "age": jnp.zeros(3),
            "regime_id": jnp.full(3, _Regimes.alive, dtype=jnp.int32),
        },
        seed=0,
        log_level="off",
    ).to_dataframe()
    first = frame.loc[frame["age"] == 0]

    assert first["landing"].astype(str).tolist() == [landing] * 3
    _assert_all_equal(values=np.asarray(solution.values[0]["alive"]), expected=value)
    # The simulated value is the solve's maximum, so its choice is the solve's.
    _assert_all_equal(values=first["value"].to_numpy(dtype=float), expected=value)


def _half(age: FloatND) -> FloatND:
    return jnp.full_like(age, 0.5, dtype=float)


def _gated_utility() -> FloatND:
    return jnp.asarray(1.0)


def _fallback_utility() -> FloatND:
    return jnp.asarray(0.0)


def _gate_open(age: FloatND) -> BoolND:
    return jnp.ones_like(age, dtype=bool)


def _transposed_model_with_a_gated_edge(*, contract: bool) -> Model:
    """The transposed cancellation toward `final`, half the mass on a gated edge."""
    states = {
        "wealth": LinSpacedGrid(start=0.0, stop=2.0, n_points=3),
        "z": DiscreteGrid(category_class=_Binary),
    }
    alive = Regime(
        states=states,
        actions={"landing": DiscreteGrid(category_class=_Binary)},
        state_transitions={
            "wealth": _next_wealth_by_landing,
            "z": StochasticTransition(func=_equal_probabilities),
        },
        functions={"utility": _transposed_alive_utility},
        solver=GridSearch(lottery_contraction=contract),
    )
    return Model(
        regimes={
            "alive": alive,
            "final": Regime(
                states=states, functions={"utility": _transposed_terminal_utility}
            ),
            "gated": Regime(functions={"utility": _gated_utility}),
            "fallback": Regime(functions={"utility": _fallback_utility}),
        },
        regime_id_class=_GatedRegimes,
        ages=AgeGrid(start=0, inclusive_stop=1, step="Y"),
        initial_nodes=((0, "alive"),),
        edges={
            "alive": Transition(
                targets={"final": 0, "gated": 0, "fallback": 0},
                law={
                    "final": StochasticTransition(func=_half),
                    "gated": StochasticTransition(func=_half),
                },
                gates={
                    "gated": Gate(
                        predicate=_gate_open,
                        routes={
                            "only": StakeholderRoute(
                                fallback=ProjectedRegimeValue(
                                    regime="fallback", projection={}
                                )
                            )
                        },
                    )
                },
            )
        },
        durable_identity=False,
    )


@pytest.mark.parametrize("contract", [False, True])
def test_simulation_enumerates_where_a_gated_edge_keeps_the_solve_enumerating(
    *, contract: bool
) -> None:
    """A gated edge keeps every draw of its source enumerated, in both phases.

    Half the mass reaches `final` through the transposed cancellation and half the
    open gate, worth 1. Enumerated, the midpoint is worth `0.5 * 0.5 + 0.5 = 0.75`
    and the endpoint `0.5 * 0.25 + 0.5 = 0.625`, so both phases choose the midpoint
    whether contraction is requested or not.
    """
    model = _transposed_model_with_a_gated_edge(contract=contract)
    params = {
        "alive": {"koopmans_aggregator": {"discount_factor": 1.0}},
        "final": {},
        "gated": {},
        "fallback": {},
    }
    solution = model.solve(params=params, log_level="off")
    frame = model.simulate(
        params=params,
        solution=solution,
        initial_conditions={
            "wealth": jnp.array([0.0, 1.0, 2.0]),
            "z": jnp.array([0, 1, 0]),
            "age": jnp.zeros(3),
            "regime_id": jnp.full(3, _GatedRegimes.alive, dtype=jnp.int32),
        },
        seed=0,
        log_level="off",
    ).to_dataframe()
    first = frame.loc[frame["age"] == 0]

    assert first["landing"].astype(str).tolist() == ["low"] * 3
    _assert_all_equal(values=np.asarray(solution.values[0]["alive"]), expected=0.75)
    _assert_all_equal(values=first["value"].to_numpy(dtype=float), expected=0.75)
