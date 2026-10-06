"""A regime law naming a target that the graph gives no edge at that age.

`A`'s age-0 law puts probability 0.5 on `A` and 0.5 on `B`. With the edge
`(0, A -> B)` declared, the model solves: with a discount factor of one,
`V_A(1) = 0.5 * 10 + 0.5 * 1 = 5.5` and `V_A(0) = 0.5 * 5.5 + 0.5 * 10 = 7.75`.
Without that edge, the `B` cell has nowhere to go, the remaining mass is 0.5, and
the error names the missing edge rather than the law.
"""

import jax.numpy as jnp
import pytest

from lcm import AgeGrid, ByAge, Model, Regime, StochasticTransition, categorical
from lcm.exceptions import InvalidRegimeTransitionProbabilitiesError
from lcm.typing import FloatND, ScalarInt


@categorical(ordered=False)
class _RegimeId:
    A: ScalarInt
    B: ScalarInt
    C: ScalarInt


def _half() -> FloatND:
    return jnp.asarray(0.5)


def _quarter() -> FloatND:
    return jnp.asarray(0.25)


def _zero() -> FloatND:
    return jnp.asarray(0.0)


def _ten() -> FloatND:
    return jnp.asarray(10.0)


def _one() -> FloatND:
    return jnp.asarray(1.0)


def _model(
    *, a_to_b_ages: tuple[int, ...], b_at_age_0: StochasticTransition | None = None
) -> Model:
    half = StochasticTransition(func=_half)
    law = ByAge(
        cases={0: {"A": half, "B": b_at_age_0 or half}, 1: {"B": half, "C": half}}
    )
    return Model(
        regimes={
            "A": Regime(regime_transitions=law, functions={"utility": _zero}),
            "B": Regime(regime_transitions=None, functions={"utility": _ten}),
            "C": Regime(regime_transitions=None, functions={"utility": _one}),
        },
        regime_id_class=_RegimeId,
        ages=AgeGrid(start=0, inclusive_stop=2, step="Y"),
        edges={"A": {"A": 0, "B": a_to_b_ages, "C": 1}},
        initial_nodes=((0, "A"),),
        fixed_params={"discount_factor": 1.0},
    )


def test_declared_edge_solves_to_the_hand_computed_value() -> None:
    solution = _model(a_to_b_ages=(0, 1)).solve(params={}, log_level="off")

    assert float(solution.value(period=0, regime="A")) == pytest.approx(7.75, abs=1e-6)


def test_a_cell_without_an_edge_is_named_in_the_mass_error() -> None:
    model = _model(a_to_b_ages=(1,))

    with pytest.raises(
        InvalidRegimeTransitionProbabilitiesError,
        match=r"(?s)do not sum to 1\.0.*at age 0.*'A' -> 'B'.*declares no edge",
    ):
        model.solve(params={}, log_level="off")


def test_a_mass_error_without_dropped_cells_points_at_the_law() -> None:
    model = _model(a_to_b_ages=(0, 1), b_at_age_0=StochasticTransition(func=_quarter))

    with pytest.raises(InvalidRegimeTransitionProbabilitiesError) as error:
        model.solve(params={}, log_level="off")

    assert "Check the 'next_regime' function of the 'A' regime" in str(error.value)
    assert "edge" not in str(error.value)


def test_simulate_names_the_cell_without_an_edge() -> None:
    model = _model(a_to_b_ages=(1,))

    with pytest.raises(
        InvalidRegimeTransitionProbabilitiesError,
        match=r"(?s)at age 0.*'A' -> 'B'.*declares no edge",
    ):
        model.simulate(
            params={},
            initial_conditions={
                "age": jnp.zeros(2),
                "regime_id": jnp.zeros(2, dtype=jnp.int32),
            },
            log_level="off",
            seed=0,
        )
