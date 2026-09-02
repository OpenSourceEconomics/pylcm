"""Rank contract for a value function handed back to `simulate`.

A regime's stored value function carries one axis per solve state it keeps as a
grid axis, plus one trailing stakeholder axis when the regime is collective. A
folded state is integrated out by quadrature when the value is stored and
contributes no axis at all.

Solving and then simulating with the returned solution is the documented
workflow, so the rank a solve produces is by definition the rank `simulate`
accepts — at every log level, since `log_level` selects how loudly validation
speaks and never what counts as valid. The rank check still has work to do:
an array of the wrong rank broadcasts rather than raising, so nothing
downstream can recover it.
"""

from dataclasses import replace
from types import MappingProxyType

import jax.numpy as jnp
import pytest
from numpy.testing import assert_array_almost_equal as aaae

from lcm.exceptions import InvalidSimulationInputError
from tests.collective_fixtures import (
    DISCOUNT_FACTOR,
    FOLD_TERMINAL_PAYOFF,
    folded_shock_nodes,
    make_couple_initial_conditions,
    make_folding_singleton_initial_conditions,
    make_folding_singleton_model,
    make_two_stakeholder_model,
)
from tests.conftest import DECIMAL_PRECISION


def test_supplied_collective_V_passes_shape_validation_at_debug():
    """A collective regime's own solved value function is accepted by `simulate`.

    The solver writes one trailing stakeholder axis onto a collective regime's
    value function, so the rank check counts that axis alongside the state
    axes. Both subjects then simulate their household's per-stakeholder values:
    the wife's leisure taste loses to the continuation at period 0, so both
    couples work and land on the high wage.
    """
    model, params = make_two_stakeholder_model()
    solution = model.solve(params=params, log_level="off")

    result = model.simulate(
        params=params,
        initial_conditions=make_couple_initial_conditions(n_subjects=2),
        solution=solution,
        log_level="debug",
        seed=0,
    )

    values = (
        result.to_dataframe()
        .set_index(["subject_id", "period"])
        .sort_index()[["value_f", "value_m"]]
        .to_numpy()
    )
    aaae(
        values,
        [
            [46.0, 92.0],  # subject 0, period 0: wage 8, works
            [40.0, 80.0],  # subject 0, period 1: wage 40, works
            [78.0, 156.0],  # subject 1, period 0: wage 40, works
            [40.0, 80.0],  # subject 1, period 1: wage 40, works
        ],
        decimal=DECIMAL_PRECISION,
    )


def test_supplied_folded_V_passes_shape_validation_at_debug():
    """A folded state's regime supplies a value function the rank check accepts.

    The fold takes the quadrature when the value is stored, so the stored array
    has one axis fewer than the regime has states — here none at all. Each
    simulated subject still realizes its own node: working pays `10 + shock`
    and the terminal regime pays a constant on top.
    """
    model, params = make_folding_singleton_model()
    solution = model.solve(params=params, log_level="off")

    result = model.simulate(
        params=params,
        initial_conditions=make_folding_singleton_initial_conditions(n_subjects=2),
        solution=solution,
        log_level="debug",
        seed=0,
    )

    nodes = folded_shock_nodes()
    entry_values = 10.0 + nodes[:2] + DISCOUNT_FACTOR * FOLD_TERMINAL_PAYOFF
    values = (
        result.to_dataframe()
        .set_index(["subject_id", "period"])
        .sort_index()["value"]
        .to_numpy()
    )
    aaae(
        values,
        [
            entry_values[0],  # subject 0, period 0: seeded at the lowest node
            FOLD_TERMINAL_PAYOFF,  # subject 0, period 1
            entry_values[1],  # subject 1, period 0: seeded at the second node
            FOLD_TERMINAL_PAYOFF,  # subject 1, period 1
        ],
        decimal=DECIMAL_PRECISION,
    )


def test_supplying_a_value_function_of_the_wrong_rank_is_rejected():
    """A supplied value function of a rank no regime produces is rejected.

    The `couple` regime carries one wage axis and one stakeholder axis, so a
    rank-four array cannot be its value function under any reading of the rank
    rule. It is reported by period and regime, naming the rank that arrived.
    """
    model, params = make_two_stakeholder_model()
    solution = model.solve(params=params, log_level="off")

    per_period = {period: dict(regimes) for period, regimes in solution.values.items()}
    per_period[0]["couple"] = jnp.zeros((2, 2, 2, 2))
    corrupted_values = MappingProxyType(
        {period: MappingProxyType(regimes) for period, regimes in per_period.items()}
    )
    corrupted = replace(solution, values=corrupted_values)

    # The pattern pins the injected array's own line, so the test reports this
    # rank-four entry rather than any other mismatch in the same message.
    with pytest.raises(
        InvalidSimulationInputError,
        match=r"\(0, 'couple'\) shape=\(2, 2, 2, 2\)",
    ):
        model.simulate(
            params=params,
            initial_conditions=make_couple_initial_conditions(n_subjects=2),
            solution=corrupted,
            log_level="debug",
            seed=0,
        )
