"""Budgeted admission derives gate requirements from the forward inventory.

A registered gated regime that no subject can occupy owes no gate programs, so
its declaration must not veto a memory-budgeted simulation of the regimes that
subjects do occupy. A gated regime subjects can occupy still needs its compiled
gate stages.
"""

import dataclasses
from types import MappingProxyType

import jax.numpy as jnp
import numpy as np
import pytest

from lcm import (
    AgeGrid,
    ByAge,
    ExecutionConfig,
    Gate,
    LinSpacedGrid,
    Model,
    ProjectedRegimeValue,
    Regime,
    SimulationResult,
    StakeholderRoute,
    StochasticTransition,
    Transition,
    categorical,
    fixed_transition,
)
from lcm.exceptions import ExecutionPlanningError
from lcm.solvers import SolutionResult
from lcm.typing import BoolND, ContinuousState, FloatND, ScalarInt


@categorical(ordered=False)
class DormantId:
    main: ScalarInt
    latent: ScalarInt
    end: ScalarInt


def _utility(*, wealth: ContinuousState) -> FloatND:
    return wealth


def _probability() -> FloatND:
    return jnp.asarray(1.0)


def _gate(*, wealth: ContinuousState) -> BoolND:
    return wealth > 0.0


def _projection(*, wealth: ContinuousState) -> ContinuousState:
    return wealth


def _regime(*, terminal: bool) -> Regime:
    return Regime(
        states={"wealth": LinSpacedGrid(start=1.0, stop=2.0, n_points=3)},
        state_transitions={} if terminal else {"wealth": fixed_transition("wealth")},
        functions={"utility": _utility},
    )


def _model(
    *, budgeted: bool, gated: bool, promote: bool, reverse: bool, width: int
) -> Model:
    latent_edges = (
        Transition(
            targets={"end": 0},
            law=ByAge(cases={0: {"end": StochasticTransition(func=_probability)}}),
            gates={
                "end": Gate(
                    predicate=_gate,
                    routes={
                        "only": StakeholderRoute(
                            target_stakeholder=None,
                            fallback=ProjectedRegimeValue(
                                regime="end", projection={"wealth": _projection}
                            ),
                        )
                    },
                )
            },
        )
        if gated
        else {"end": 0}
    )
    regimes = {
        "main": _regime(terminal=False),
        "latent": _regime(terminal=False),
        "end": _regime(terminal=True),
    }
    if reverse:
        regimes = dict(reversed(tuple(regimes.items())))
    return Model(
        ages=AgeGrid(start=0, inclusive_stop=1, step="Y"),
        regimes=regimes,
        regime_id_class=DormantId,
        initial_nodes={0: ("main", "latent") if promote else "main"},
        enable_jit=True,
        execution_config=(
            ExecutionConfig(
                devices=(0,),
                axis_widths={"subject": width},
                device_memory_bytes=2**30,
            )
            if budgeted
            else ExecutionConfig()
        ),
        edges={"main": {"end": 0}, "latent": latent_edges},
    )


def _run(*, model: Model, solution: SolutionResult, workers: int) -> SimulationResult:
    return model.simulate(
        params={"discount_factor": 0.5},
        solution=solution,
        initial_conditions={
            "wealth": jnp.asarray([1.0, 1.5, 2.0]),
            "age": jnp.zeros(3),
            "regime_id": jnp.full(3, model.regime_names_to_ids["main"]),
        },
        seed=7,
        log_level="off",
        max_compilation_workers=workers,
    )


def _check_panel(*, result: SimulationResult) -> None:
    for period, name, value in (
        (0, "main", [1.5, 2.25, 3.0]),
        (1, "end", [1.0, 1.5, 2.0]),
    ):
        record = result.raw_results[name][period]
        np.testing.assert_array_equal(np.asarray(record.in_regime), [True] * 3)
        np.testing.assert_array_equal(
            np.asarray(record.states["wealth"]), [1.0, 1.5, 2.0]
        )
        np.testing.assert_array_equal(np.asarray(record.V_arr), value)
    for record in result.raw_results.get("latent", {}).values():
        assert not np.asarray(record.in_regime).any()


def exercise(
    *, gated: bool, promote: bool, reverse: bool, width: int, workers: int
) -> None:
    """Simulate main -> end with and without a budget and compare exact panels.

    `V_end(w) = w` and `V_main(w) = w + V_end(w) / 2 = 3w / 2` on the wealth grid
    `(1, 3/2, 2)`. Without `promote`, `latent` has no forward or solved node.
    """
    models = {
        budgeted: _model(
            budgeted=budgeted,
            gated=gated,
            promote=promote,
            reverse=reverse,
            width=width,
        )
        for budgeted in (False, True)
    }
    solved: dict[bool, SolutionResult] = {}
    for budgeted, model in models.items():
        solved[budgeted] = model.solve(params={"discount_factor": 0.5}, log_level="off")
        np.testing.assert_array_equal(
            np.asarray(solved[budgeted].values[0]["main"]), [1.5, 2.25, 3.0]
        )
        if not promote:
            latent = model._regimes["latent"]
            assert model.reachability.nodes == frozenset({(0, "main"), (1, "end")})
            assert model.reachability.visited_nodes == frozenset(
                {(0, "main"), (1, "end")}
            )
            assert tuple(latent.active_periods) == ()
            assert not latent.simulation.programs.decision
            assert bool(latent.gated_edges) is gated
            assert not latent.simulation.programs.gate_fold
            assert not latent.simulation.programs.gate_route
            assert set(solved[budgeted].values[0]) == {"main"}
    ordinary = _run(model=models[False], solution=solved[False], workers=workers)
    _check_panel(result=ordinary)
    budgeted = _run(model=models[True], solution=solved[True], workers=workers)
    _check_panel(result=budgeted)
    repeated = _run(model=models[True], solution=solved[True], workers=workers)
    _check_panel(result=repeated)
    for period, name in ((0, "main"), (1, "end")):
        for result in (budgeted, repeated):
            np.testing.assert_array_equal(
                np.asarray(result.raw_results[name][period].V_arr),
                np.asarray(ordinary.raw_results[name][period].V_arr),
            )


def test_dormant_gate_budget_admission_witness() -> None:
    """An unreachable gated declaration leaves the budgeted main -> end panel exact."""
    exercise(gated=True, promote=False, reverse=False, width=2, workers=1)


def test_budgeted_simulation_refuses_demanded_gate_without_compiled_stages() -> None:
    """A reachable gated regime whose gate programs are missing is refused."""
    model = _model(budgeted=True, gated=True, promote=True, reverse=False, width=2)
    solution = model.solve(params={"discount_factor": 0.5}, log_level="off")
    latent = model._regimes["latent"]
    assert tuple(latent.simulation.programs.gate_fold) == (0,)
    assert tuple(latent.simulation.programs.gate_route) == (0,)
    model._regimes = MappingProxyType(
        {
            **model._regimes,
            "latent": dataclasses.replace(
                latent,
                simulation=dataclasses.replace(
                    latent.simulation,
                    programs=dataclasses.replace(
                        latent.simulation.programs,
                        gate_fold=MappingProxyType({}),
                        gate_route=MappingProxyType({}),
                    ),
                ),
            ),
        }
    )
    with pytest.raises(ExecutionPlanningError, match="compiled decision programs"):
        _run(model=model, solution=solution, workers=1)
