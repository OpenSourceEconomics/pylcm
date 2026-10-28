"""Scalar DC-R read census and single-flight call-owned generation."""

import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import numpy as np
import pytest

import _lcm.solution.backward_induction as backward_induction_module
from _lcm.solution.diagnostics import _DiagnosticRow
from _lcm.solution.feasibility import ContinuationReadCounts
from _lcm.solution.generation import CanonicalGenerationCache, CanonicalModelGeneration
from lcm import IrregSpacedGrid
from lcm.model import _ModelConstructorInputs
from lcm.shocks import Normal, ShockGroup, UserNodes
from lcm.solvers import DCEGM
from lcm.typing import RegimeName, UserParams
from tests.conftest import EXACT_KERNEL_SKIP_REASON
from tests.egm.test_current_iid_call_owned_generation import (
    _declarations,
    _params,
    _public_model_from_declarations,
)

pytestmark = pytest.mark.requires_exact_affine_kernel(reason=EXACT_KERNEL_SKIP_REASON)

_N_CALLERS = 4
_WAIT_SECONDS = 300.0

type _Census = dict[tuple[RegimeName, int], ContinuationReadCounts | tuple[int, int]]


def _regime_params(*, parent_discount: float) -> UserParams:
    """The P1 parameters with the parent's own discount factor set separately."""
    return {
        "parent": {
            "koopmans_aggregator": {"discount_factor": parent_discount},
            "child": {"next_wealth": {"slope": 2.0}},
        },
        "child": {
            "koopmans_aggregator": {"discount_factor": 1.0},
            "done": {"next_wealth": {"slope": 1.0}},
        },
    }


def _with_income_weights(
    *, declarations: _ModelConstructorInputs, weights: list[float]
) -> _ModelConstructorInputs:
    return replace(
        declarations,
        shocks={
            "income": ShockGroup(
                shocks={"eps": Normal(mu=0.0, sigma=1.0)},
                nodes=UserNodes(
                    nodes=np.array([[0.0], [1.0]]), weights=np.array(weights)
                ),
            )
        },
    )


def _with_parent_savings(
    *, declarations: _ModelConstructorInputs, points: list[float]
) -> _ModelConstructorInputs:
    parent = declarations.regimes["parent"]
    solver = parent.solver
    assert isinstance(solver, DCEGM)
    return replace(
        declarations,
        regimes={
            **declarations.regimes,
            "parent": replace(
                parent,
                solver=replace(
                    solver, savings_grid=IrregSpacedGrid(points=np.array(points))
                ),
            ),
        },
    )


def _n_savings_points(
    *, declarations: _ModelConstructorInputs, regime_name: RegimeName
) -> int:
    solver = declarations.regimes[regime_name].solver
    assert isinstance(solver, DCEGM)
    assert isinstance(solver.savings_grid, IrregSpacedGrid)
    points = solver.savings_grid.points
    assert points is not None
    return len(points)


def _variant(*, name: str) -> tuple[_ModelConstructorInputs, UserParams]:
    """One P1 model and its parameters; only `fewer_parent_savings` changes reads."""
    declarations = _declarations()
    if name == "declared":
        return declarations, _regime_params(parent_discount=1.0)
    if name == "parent_discount_zero":
        return declarations, _regime_params(parent_discount=0.0)
    if name == "zero_income_weight":
        return (
            _with_income_weights(declarations=declarations, weights=[1.0, 0.0]),
            _regime_params(parent_discount=1.0),
        )
    assert name == "fewer_parent_savings"
    return (
        _with_parent_savings(declarations=declarations, points=[0.5, 1.0, 4.0]),
        _regime_params(parent_discount=1.0),
    )


def _observe_census(*, monkeypatch: pytest.MonkeyPatch) -> _Census:
    """Record each row's host read census at its period barrier."""
    census: _Census = {}
    native = backward_induction_module._materialize_period_feasibility_counts

    def observe(
        *, diagnostic_rows: list[_DiagnosticRow], period: int, enabled: bool
    ) -> None:
        native(diagnostic_rows=diagnostic_rows, period=period, enabled=enabled)
        for row in diagnostic_rows:
            if row.period == period and row.continuation_read_counts is not None:
                census[(row.regime_name, period)] = row.continuation_read_counts

    monkeypatch.setattr(
        backward_induction_module, "_materialize_period_feasibility_counts", observe
    )
    return census


@pytest.mark.coverage(backends=("cpu",), precisions="both")
@pytest.mark.parametrize(
    "variant",
    [
        "declared",
        "parent_discount_zero",
        "zero_income_weight",
        "fewer_parent_savings",
    ],
)
def test_public_case_r_census_counts_every_required_read_before_masks(
    *, monkeypatch: pytest.MonkeyPatch, variant: str
) -> None:
    """Every required parent read of the child row is counted and delivered.

    The parent reads the child's projected continuation once per declared parent
    saving, and the child reads the terminal once per declared child saving. A
    zero parent discount or a zero-weight income node masks the contribution,
    not the read, so both counts follow the declared grids alone. All reads lie
    on the supported domain, and the solve completes, so each was delivered.
    """
    declarations, params = _variant(name=variant)
    census = _observe_census(monkeypatch=monkeypatch)
    _public_model_from_declarations(declarations=declarations).solve(
        params=params, log_level="warning", max_compilation_workers=1
    )
    assert census == {
        ("parent", 0): (
            0,
            _n_savings_points(declarations=declarations, regime_name="parent"),
        ),
        ("child", 1): (
            0,
            _n_savings_points(declarations=declarations, regime_name="child"),
        ),
    }


def _observe_generation_builds(
    *, monkeypatch: pytest.MonkeyPatch, cache: CanonicalGenerationCache
) -> tuple[list[str], list[str]]:
    """Count cache entries and factory runs; a build waits for every caller.

    The factory does not start building until all callers have entered the
    cache, so every caller overlaps the one cold build.
    """
    entered: list[str] = []
    builds: list[str] = []
    entered_lock = threading.Lock()
    all_entered = threading.Event()
    native = cache.acquire

    def acquire(
        *, key: str, factory: Callable[[], CanonicalModelGeneration]
    ) -> CanonicalModelGeneration:
        with entered_lock:
            entered.append(key)
            if len(entered) == _N_CALLERS:
                all_entered.set()

        def counted() -> CanonicalModelGeneration:
            builds.append(key)
            assert all_entered.wait(timeout=_WAIT_SECONDS), "Callers did not overlap."
            return factory()

        return native(key=key, factory=counted)

    monkeypatch.setattr(cache, "acquire", acquire)
    return entered, builds


@pytest.mark.coverage(backends=("cpu",), precisions="both")
@pytest.mark.parametrize(
    "call_params",
    [
        pytest.param([_params()] * _N_CALLERS, id="identical"),
        pytest.param(
            [
                _params(),
                _params(discount=0.95),
                _params(parent_slope=3.0),
                _params(discount=0.95, parent_slope=3.0),
            ],
            id="same_structure",
        ),
    ],
)
def test_public_concurrent_calls_build_one_call_owned_generation(
    *, monkeypatch: pytest.MonkeyPatch, call_params: list[UserParams]
) -> None:
    """Concurrent calls sharing a structural key build and share one generation."""
    model = _public_model_from_declarations(declarations=_declarations())
    entered, builds = _observe_generation_builds(
        monkeypatch=monkeypatch, cache=model._generation_cache
    )
    barrier = threading.Barrier(_N_CALLERS)

    def call(params: UserParams) -> CanonicalModelGeneration:
        barrier.wait(timeout=_WAIT_SECONDS)
        _, generation = model._prepare_call(params=params)
        return generation

    with ThreadPoolExecutor(max_workers=_N_CALLERS) as pool:
        generations = list(pool.map(call, call_params))

    first = generations[0]
    assert all(generation is first for generation in generations)
    assert entered == [first.key] * _N_CALLERS
    assert builds == [first.key]
    first.solver_preparation(regime_name="child", period=1).validate_owner()
