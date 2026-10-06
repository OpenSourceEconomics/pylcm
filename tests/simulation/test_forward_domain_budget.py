"""Budgeted simulation plans forward work only for pairs a subject can occupy.

A pair solved only for its value (a perceived target of a phased transition that
no subject physically reaches) has no decision program. Budgeted simulation must
profile and dispatch exactly the forward pairs the unbudgeted path dispatches,
while the solved-only pair's value stays available to the decision reading it.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from lcm import (
    AgeGrid,
    ByAge,
    ExecutionConfig,
    LinSpacedGrid,
    Model,
    Phased,
    Regime,
    categorical,
    fixed_transition,
)
from lcm.typing import ContinuousState, FloatND, ScalarInt


@categorical(ordered=False)
class DomainId:
    source: ScalarInt
    perceived: ScalarInt
    realized: ScalarInt
    end: ScalarInt


def _utility(*, wealth: ContinuousState) -> FloatND:
    return wealth


def _perceived_utility(*, wealth: ContinuousState) -> FloatND:
    return wealth + 1.0


def _regime(*, law, perceived=False):
    return Regime(
        regime_transitions=law,
        states={"wealth": LinSpacedGrid(start=0.0, stop=1.0, n_points=2)},
        state_transitions={} if law is None else {"wealth": fixed_transition("wealth")},
        functions={"utility": _perceived_utility if perceived else _utility},
    )


def _model(*, budgeted, promote, reverse, width):
    regimes = {
        "source": _regime(
            law=ByAge(
                cases={
                    0: Phased(solve="perceived", simulate="realized"),
                }
            )
        ),
        "perceived": _regime(law=ByAge(cases={1: "end"}), perceived=True),
        "realized": _regime(law=ByAge(cases={1: "end"})),
        "end": _regime(law=None),
    }
    if reverse:
        regimes = dict(reversed(tuple(regimes.items())))
    roots: dict[object, str] = {0: "source"}
    if promote:
        roots[1] = "perceived"
    execution_config = (
        ExecutionConfig(
            devices=(0,),
            axis_widths={"subject": width},
            device_memory_bytes=2**30,
        )
        if budgeted
        else ExecutionConfig()
    )
    return Model(
        enable_jit=True,
        ages=AgeGrid(start=0, inclusive_stop=2, step="Y"),
        regime_id_class=DomainId,
        initial_nodes=roots,
        regimes=regimes,
        execution_config=execution_config,
        edges=Phased(
            solve={
                "source": {"perceived": 0},
                "perceived": {"end": 1},
                "realized": {"end": 1},
            },
            simulate={
                "source": {"realized": 0},
                "perceived": {"end": 1},
                "realized": {"end": 1},
            },
        ),
    )


def _simulate(*, model, solution, workers):
    return model.simulate(
        params={"discount_factor": 0.5},
        solution=solution,
        initial_conditions={
            "wealth": jnp.array([0.0, 0.5, 1.0]),
            "age": jnp.zeros(3),
            "regime_id": jnp.full(3, model.regime_names_to_ids["source"]),
        },
        seed=7,
        log_level="off",
        max_compilation_workers=workers,
    )


def _check_result(*, result):
    # Realized transitions, never the perceived/valuation branch.
    for period, name in enumerate(("source", "realized", "end")):
        record = result.raw_results[name][period]
        np.testing.assert_array_equal(np.asarray(record.in_regime), [True] * 3)
        np.testing.assert_array_equal(np.asarray(record.states["wealth"]), [0, 0.5, 1])
    if 1 in result.raw_results["perceived"]:
        assert not np.asarray(result.raw_results["perceived"][1].in_regime).any()


def _cache_size(*, model):
    return sum(
        len(executor.cache) + len(executor.operations.cache)
        for executor in {
            id(regime.simulation.programs.executor): regime.simulation.programs.executor
            for regimes in model._simulate_runtime_regimes.values()
            for regime in regimes.values()
        }.values()
    ) + len(model._simulate_entry_operations.cache)


@pytest.mark.parametrize("promote", [False, True])
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("width", [1, 2, 3])
@pytest.mark.parametrize("workers", [1, 2])
def test_budgeted_forward_domain(*, promote, reverse, width, workers):
    """Budgeted and unbudgeted simulation agree across roots, order, widths, workers."""
    baseline = _model(budgeted=False, promote=promote, reverse=reverse, width=width)
    budgeted = _model(budgeted=True, promote=promote, reverse=reverse, width=width)
    for model in (baseline, budgeted):
        assert (1, "perceived") in model.reachability.nodes
        assert ((1, "perceived") in model.reachability.visited_nodes) is promote
    base_solution = baseline.solve(
        params={"discount_factor": 0.5},
        log_level="off",
        max_compilation_workers=1,
    )
    solution = budgeted.solve(
        params={"discount_factor": 0.5},
        log_level="off",
        max_compilation_workers=1,
    )
    # Solved-only operands must remain available: filtering all of S is wrong.
    for sol in (base_solution, solution):
        np.testing.assert_array_equal(np.asarray(sol.values[0]["source"]), [0.5, 2.25])
    reference = _simulate(model=baseline, solution=base_solution, workers=workers)
    _check_result(result=reference)
    first = _simulate(model=budgeted, solution=solution, workers=workers)
    jax.effects_barrier()
    before = _cache_size(model=budgeted)
    second = _simulate(model=budgeted, solution=solution, workers=workers)
    jax.effects_barrier()
    assert _cache_size(model=budgeted) == before
    for result in (first, second):
        _check_result(result=result)
        for period, name in enumerate(("source", "realized", "end")):
            np.testing.assert_array_equal(
                np.asarray(result.raw_results[name][period].V_arr),
                np.asarray(reference.raw_results[name][period].V_arr),
            )
