"""Envelope work widths belong to the execution plan."""

import functools
from collections.abc import Hashable, Mapping
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.egm.upper_envelope import segment_envelope
from _lcm.execution.core_program import core_program_graph
from _lcm.solution import backward_induction
from _lcm.solution import negm as nested_solver
from lcm import AgeGrid, ExecutionConfig, LinSpacedGrid, Model
from lcm.exceptions import ExecutionPlanningError
from lcm.solvers import DCEGM, NEGM, EnvelopeConfig, ExactEnvelope, FUESEnvelope
from lcm.typing import ContinuousState, FloatND, ScalarFloat
from tests.conftest import EXACT_KERNEL_SKIP_REASON, assert_agrees_to_ulp
from tests.test_models import n_nbegm_toy
from tests.test_models.deterministic.dcegm_variants import dcegm_retirement
from tests.test_models.deterministic.retirement_only import (
    RetirementOnlyRegimeId,
    dead,
    get_params,
)


def _law_reading_current_wealth(
    *,
    savings: FloatND,
    wealth: ContinuousState,
    interest_rate: float,
    labor_income: float,
) -> ContinuousState:
    return (1.0 + interest_rate) * savings + labor_income + 0.01 * wealth


def _model(
    *,
    widths: Mapping[str, int],
    envelope: EnvelopeConfig,
    asset_rows: bool = False,
    device_memory_bytes: int | None = None,
) -> Model:
    """Small real DC-EGM model exercising either numerical envelope consumer."""
    retirement = dcegm_retirement.replace(
        active=lambda age: age < 60,
        states={"wealth": LinSpacedGrid(start=1.0, stop=10.0, n_points=6)},
        solver=DCEGM(
            savings_grid=LinSpacedGrid(start=0.0, stop=12.0, n_points=8),
            n_constrained_points=4,
            envelope=envelope,
        ),
    )
    if asset_rows:
        retirement = retirement.replace(
            state_transitions={"wealth": _law_reading_current_wealth}
        )
    return Model(
        regimes={"retirement": retirement, "dead": dead},
        ages=AgeGrid(start=40, stop=60, step="10Y"),
        regime_id_class=RetirementOnlyRegimeId,
        execution_config=ExecutionConfig(
            axis_widths=widths, device_memory_bytes=device_memory_bytes
        ),
    )


def test_exact_envelope_refuses_a_model_owned_cell_width() -> None:
    """The execution plan owns the width of the exact envelope's cell loop."""
    with pytest.raises(TypeError, match="cell_batch_size"):
        ExactEnvelope(cell_batch_size=2)  # ty: ignore[unknown-argument]


def test_fues_envelope_refuses_a_model_owned_scan_unroll() -> None:
    """FUES uses one fixed compiler unroll setting for every model."""
    with pytest.raises(TypeError, match="scan_unroll"):
        FUESEnvelope(scan_unroll=2)  # ty: ignore[unknown-argument]


@pytest.mark.requires_exact_affine_kernel(reason=EXACT_KERNEL_SKIP_REASON)
@pytest.mark.parametrize("asset_rows", [False, True])
def test_the_envelope_cell_loop_accepts_a_planner_width(
    *, asset_rows: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Both full-row refinement and query refinement consume the planned axis."""
    observed: set[int] = set()
    original = segment_envelope._sub_cells_per_node_cell

    def observe(*, cell_width: int, **kwargs: Any) -> object:
        observed.add(cell_width)
        return original(cell_width=cell_width, **kwargs)

    monkeypatch.setattr(segment_envelope, "_sub_cells_per_node_cell", observe)
    model = _model(
        widths={"envelope_cell": 3},
        envelope=ExactEnvelope(max_runs=4),
        asset_rows=asset_rows,
    )
    model.solve(params=get_params(n_periods=3), log_level="off")
    assert observed == {3}


def test_fues_refuses_an_axis_its_programs_do_not_run() -> None:
    """The planner only accepts axes actually declared by this model's programs."""
    with pytest.raises(ExecutionPlanningError, match="envelope_cell"):
        _model(widths={"envelope_cell": 3}, envelope=FUESEnvelope())


@pytest.mark.requires_exact_affine_kernel(reason=EXACT_KERNEL_SKIP_REASON)
@pytest.mark.parametrize("asset_rows", [False, True])
@pytest.mark.parametrize("width", [1, 3, 11])
def test_planner_cell_width_preserves_the_solution(
    *, asset_rows: bool, width: int
) -> None:
    """Changing the scan partition preserves values in every solved period."""
    reference = _model(
        widths={"envelope_cell": 1},
        envelope=ExactEnvelope(max_runs=4),
        asset_rows=asset_rows,
    ).solve(params=get_params(n_periods=3), log_level="off")
    candidate = _model(
        widths={"envelope_cell": width},
        envelope=ExactEnvelope(max_runs=4),
        asset_rows=asset_rows,
    ).solve(params=get_params(n_periods=3), log_level="off")
    assert_agrees_to_ulp(
        got=np.concatenate(
            [np.asarray(candidate.values[p]["retirement"]) for p in (0, 1)]
        ),
        expected=np.concatenate(
            [np.asarray(reference.values[p]["retirement"]) for p in (0, 1)]
        ),
        n_ulp=4,
    )


@pytest.mark.requires_exact_affine_kernel(reason=EXACT_KERNEL_SKIP_REASON)
@pytest.mark.parametrize(("width", "n_chunks"), [(1, 11), (3, 4), (11, 1), (100, 1)])
def test_cell_width_sizes_the_real_scan(*, width: int, n_chunks: int) -> None:
    """The cell scan visits exactly the planned number of bounded chunks."""
    grid = jnp.linspace(1.0, 12.0, 12)
    trace = jax.make_jaxpr(
        functools.partial(
            segment_envelope.refine_envelope_exact,
            n_refined=24,
            max_runs=4,
            cell_width=width,
        )
    )(endog_grid=grid, policy=grid / 2.0, value=grid)
    scan_lengths = [
        equation.params["length"]
        for equation in trace.jaxpr.eqns
        if equation.primitive.name == "scan"
    ]
    assert scan_lengths == [n_chunks]


@pytest.mark.parametrize("device_memory_bytes", [None, 10**9], ids=["lazy", "aot"])
def test_fues_compiler_option_reaches_actual_lowering_keys(
    *, device_memory_bytes: int | None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Both solve compilation routes distinguish the fixed FUES compiler option."""
    original = backward_induction._lowering_key
    observed: list[bool] = []

    def observe(**kwargs: Any) -> Hashable:
        key = original(**kwargs)
        if kwargs.get("compiler_options") == (("scan_unroll", 1),):
            plain_kwargs: dict[str, Any] = dict(kwargs)
            plain_kwargs["compiler_options"] = ()
            plain = original(**plain_kwargs)
            observed.append(key != plain)
        return key

    monkeypatch.setattr(backward_induction, "_lowering_key", observe)
    model = _model(
        widths={"savings_point": 2},
        envelope=FUESEnvelope(),
        device_memory_bytes=device_memory_bytes,
    )
    model.solve(params=get_params(n_periods=3), log_level="off")
    assert set(observed) == {True}


def _nested_model(
    *,
    envelope: EnvelopeConfig,
    widths: Mapping[str, int],
    monkeypatch: pytest.MonkeyPatch,
) -> Model:
    """Build a small nested model with an explicitly selected inner envelope."""
    configured = NEGM(
        inner=DCEGM(
            savings_grid=LinSpacedGrid(start=0.0, stop=30.0, n_points=8),
            n_constrained_points=4,
            envelope=envelope,
        ),
        outer_grid=LinSpacedGrid(start=0.0, stop=15.0, n_points=3),
    )

    def build_solver(**_kwargs: object) -> NEGM:
        return configured

    monkeypatch.setattr(n_nbegm_toy, "build_solver", build_solver)
    return n_nbegm_toy.build_model(
        variant="negm",
        n_periods=2,
        illiquid_grid=LinSpacedGrid(start=0.0, stop=10.0, n_points=3),
        execution_config=ExecutionConfig(axis_widths=widths),
    )


def test_nested_fues_programs_preserve_the_inner_compiler_options(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every outer program containing a FUES solve carries its compiler identity."""
    model = _nested_model(envelope=FUESEnvelope(), widths={}, monkeypatch=monkeypatch)
    kernel = model._regimes["alive"].solution.period_kernels[0]
    programs = core_program_graph(kernel=kernel)
    assert {name: program.compiler_options for name, program in programs.items()} == {
        "keeper": (("scan_unroll", 1),),
        "outer_sweep": (("scan_unroll", 1),),
    }


@pytest.mark.requires_exact_affine_kernel(reason=EXACT_KERNEL_SKIP_REASON)
def test_nested_adjuster_dispatch_receives_the_planned_envelope_width(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The sweep dispatches its inner solve with the same planner-owned width."""
    original = nested_solver._NodeSolver.__call__
    observed: set[object] = set()

    # keyword-only-exempt: library-callback=jax.lax.map
    def observe(self: nested_solver._NodeSolver, node: ScalarFloat) -> object:
        observed.add(self.adjuster_arguments.get("_lcm_envelope_cell_width", 1))
        return original(self, node)

    monkeypatch.setattr(nested_solver._NodeSolver, "__call__", observe)
    model = _nested_model(
        envelope=ExactEnvelope(max_runs=4),
        widths={"envelope_cell": 3, "outer_candidate": 1},
        monkeypatch=monkeypatch,
    )
    model.solve(params={"discount_factor": 0.95}, log_level="off")
    assert observed == {3}


@pytest.mark.requires_exact_affine_kernel(reason=EXACT_KERNEL_SKIP_REASON)
def test_nested_envelope_width_preserves_the_solution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The inherited width preserves values across both nested solver branches."""
    reference = _nested_model(
        envelope=ExactEnvelope(max_runs=4),
        widths={"envelope_cell": 1, "outer_candidate": 1},
        monkeypatch=monkeypatch,
    ).solve(params={"discount_factor": 0.95}, log_level="off")
    candidate = _nested_model(
        envelope=ExactEnvelope(max_runs=4),
        widths={"envelope_cell": 3, "outer_candidate": 1},
        monkeypatch=monkeypatch,
    ).solve(params={"discount_factor": 0.95}, log_level="off")
    assert_agrees_to_ulp(
        got=np.asarray(candidate.values[0]["alive"]),
        expected=np.asarray(reference.values[0]["alive"]),
        n_ulp=4,
    )
