"""Solver preconditions that need parameter values run on every solve.

A solver may declare a precondition it can only check against the model's actual
parameters — an affine-budget probe that differentiates the budget DAG, say, whose
tax-schedule arguments have no values until `solve` is called. Such a solver
publishes the check alongside its kernels; the engine runs it for every parameter
draw before dispatching the numerical kernels.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import cast

import pytest

from _lcm.solution.contract import (
    ParamCheck,
    SolutionKernels,
    SolverBuildContext,
)
from _lcm.solution.grid_search import GridSearch
from _lcm.typing import FlatParams
from lcm import Model
from lcm.typing import ContinuousState, FloatND
from tests.test_models.nbegm_common import (
    feasible,
    make_alive_dead_model,
    next_liquid_from_savings,
    savings,
    utility,
)

_PARAMS = {
    "alive": {
        "utility": {"crra": 2.0},
        "koopmans_aggregator": {"discount_factor": 0.95},
        "resources": {"base_income": 2.0},
        "alive": {
            "next_liquid": {"return_liquid": 0.03, "income": 1.0},
            "next_regime": {"final_age_alive": 3.0},
        },
        "dead": {
            "next_liquid": {"return_liquid": 0.03, "income": 1.0},
            "next_regime": {"final_age_alive": 3.0},
        },
    },
    "dead": {"utility": {"crra": 2.0}},
}


def _resources(*, liquid: ContinuousState, base_income: float) -> FloatND:
    """Cash-on-hand: liquid wealth plus base income."""
    return liquid + base_income


@dataclass(frozen=True, kw_only=True)
class _RecordingCheck:
    """A param check that records the flat params it was handed."""

    calls: list[FlatParams] = field(default_factory=list)

    def __call__(self, *, flat_params: FlatParams) -> None:
        self.calls.append(flat_params)


@dataclass(frozen=True, kw_only=True)
class _RejectingCheck:
    """A param check that refuses every parameter vector."""

    def __call__(self, *, flat_params: FlatParams) -> None:  # noqa: ARG002
        raise ValueError("this regime's precondition does not hold")


@dataclass(frozen=True, kw_only=True)
class _CheckingGridSearch(GridSearch):
    """Grid search that publishes one parameter-dependent precondition."""

    check: ParamCheck

    def build_period_kernels(self, *, context: SolverBuildContext) -> SolutionKernels:
        """Delegate to grid search and attach the declared check."""
        kernels = super().build_period_kernels(context=context)
        return SolutionKernels(
            period_kernels=kernels.period_kernels,
            continuation_spec=kernels.continuation_spec,
            param_checks=(self.check,),
        )


def _build_model(
    *, check: ParamCheck, fixed_params: Mapping[str, object] | None = None
) -> Model:
    return make_alive_dead_model(
        fixed_params=fixed_params,
        n_periods=3,
        n_liquid=10,
        liquid_max=30.0,
        n_consumption=15,
        alive_functions={
            "utility": utility,
            "resources": _resources,
            "savings": savings,
        },
        liquid_law=next_liquid_from_savings,
        alive_solver=_CheckingGridSearch(check=check),
        constraints={"feasible": feasible},
    )


def test_param_check_does_not_run_at_model_construction() -> None:
    """Building a model leaves a parameter-dependent precondition unchecked."""
    check = _RecordingCheck()

    _build_model(check=check)

    assert check.calls == []


def test_param_check_runs_on_solve() -> None:
    """A solve runs each published check exactly once for its parameter draw."""
    check = _RecordingCheck()
    model = _build_model(check=check)

    model.solve(params=_PARAMS, log_level="off")

    assert len(check.calls) == 1


def test_param_check_sees_the_models_actual_parameter_values() -> None:
    """The check reads real parameter values, not synthetic stand-ins."""
    check = _RecordingCheck()
    model = _build_model(check=check)

    model.solve(params=_PARAMS, log_level="off")

    crra = check.calls[0]["alive"]["utility__crra"]

    assert float(cast("FloatND", crra)) == 2.0


def test_param_check_sees_a_parameter_the_model_fixed_at_construction() -> None:
    """A fixed parameter reaches the check with its value, like a free one.

    Fixed params are bound into the kernels rather than supplied at solve, so
    they are absent from what `solve` threads through. A precondition on the
    model's functions has to see the whole parameter vector — a tax schedule
    declared fixed is still what the budget reads.
    """
    check = _RecordingCheck()
    free_params = {
        regime: {
            key: value
            for key, value in entries.items()
            if not (regime == "alive" and key == "utility")
        }
        for regime, entries in _PARAMS.items()
    }
    model = _build_model(
        check=check, fixed_params={"alive": {"utility": {"crra": 2.0}}}
    )

    model.solve(params=free_params, log_level="off")

    crra = check.calls[0]["alive"]["utility__crra"]

    assert float(cast("FloatND", crra)) == 2.0


def test_param_check_runs_for_each_parameter_draw() -> None:
    """Every solve re-checks the actual draw before numerical dispatch."""
    check = _RecordingCheck()
    model = _build_model(check=check)

    model.solve(params=_PARAMS, log_level="off")
    model.solve(params=_PARAMS, log_level="off")

    assert len(check.calls) == 2


def test_param_check_failure_surfaces_from_solve() -> None:
    """A refused precondition raises out of `solve`."""
    model = _build_model(check=_RejectingCheck())

    with pytest.raises(ValueError, match="precondition does not hold"):
        model.solve(params=_PARAMS, log_level="off")


def test_solver_publishing_no_check_leaves_the_seam_empty() -> None:
    """Kernels default to no parameter-dependent preconditions."""
    kernels = SolutionKernels(period_kernels=MappingProxyType({}))

    assert kernels.param_checks == ()


def test_param_checks_are_a_mapping_free_tuple() -> None:
    """The published checks are an ordered tuple, so the engine runs them in order."""
    check = _RecordingCheck()
    kernels = SolutionKernels(
        period_kernels=MappingProxyType({}), param_checks=(check,)
    )

    assert isinstance(kernels.param_checks, tuple)
    assert not isinstance(kernels.param_checks, Mapping)
