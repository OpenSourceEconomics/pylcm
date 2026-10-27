"""Dated preparation source contracts and genuine ordinary fixed-param controls.

These definitions are held for the isolated native-qualified CPU fixture gate.
The metadata observer returns every original result unchanged; it supplies no
solver output, parameter shape, economic continuation, or successful native row.
"""

from collections.abc import Mapping
from dataclasses import replace
from types import MappingProxyType
from typing import Never, NotRequired, TypedDict, Unpack

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from _lcm import model_processing
from _lcm.continuation import ProjectedContinuationSpec
from _lcm.egm.current_iid_balanced import BalancedProjectedEGMCarry
from _lcm.egm.projected_child import ProjectedChildReader
from _lcm.execution.execution_plan import ResolvedExecution
from _lcm.model_processing import prepare_dated_parameters
from _lcm.params.edges import regime_kernel_params
from _lcm.params.processing import create_params_template
from _lcm.regime_building import processing
from _lcm.regime_building.finalize import FinalizedUserRegime
from _lcm.regime_building.processing import PreparedModelStructure, _SolutionBuild
from _lcm.solution.current_iid_dated import prepare_dated_balanced_kernels
from _lcm.solution.current_iid_parent_demands import (
    _incoming_parent_dates,
    _prepared_parent_pieces,
)
from _lcm.solution.dcegm import _BoundDCEGM
from _lcm.time import TimeAxis
from _lcm.typing import RegimeNamesToIds
from lcm import (
    ConsumptionSavingsRegime,
    InitialNodes,
    IrregSpacedGrid,
    LiquidMargin,
    Model,
    Periods,
    Regime,
    categorical,
)
from lcm.exceptions import UnsupportedOperationError
from lcm.solvers import DCEGM, ExactEnvelope
from lcm.typing import (
    ContinuousAction,
    ContinuousState,
    FloatND,
    RegimeName,
    ScalarInt,
    UserParams,
)
from tests.conftest import DECIMAL_PRECISION, EXACT_KERNEL_SKIP_REASON
from tests.egm.test_current_iid_balanced import (
    _coverage_missing,
    _native_balanced,
    _smooth_query,
)

pytestmark = pytest.mark.requires_exact_affine_kernel(reason=EXACT_KERNEL_SKIP_REASON)


class _ProcessArguments(TypedDict):
    ages: TimeAxis
    user_regimes: Mapping[RegimeName, FinalizedUserRegime]
    regime_names_to_ids: RegimeNamesToIds
    enable_jit: bool
    prepared_structure: PreparedModelStructure
    execution: NotRequired[ResolvedExecution | None]


@categorical(ordered=False)
class _RegimeId:
    live: ScalarInt
    done: ScalarInt


def _utility(*, consumption: ContinuousAction, scale: float) -> FloatND:
    return scale * jnp.log(consumption)


def _inverse(*, marginal_continuation: FloatND, scale: float) -> FloatND:
    return scale / marginal_continuation


def _resources(*, wealth: ContinuousState) -> FloatND:
    return wealth


def _savings(*, resources: FloatND, consumption: ContinuousAction) -> FloatND:
    return resources - consumption


def _landing(*, savings: FloatND) -> ContinuousState:
    return savings + 1.0


def _terminal(*, wealth: ContinuousState, scale: float) -> FloatND:
    return scale * jnp.log(wealth)


def _ordinary_model(*, fixed_params: UserParams, terminal_start: bool = False) -> Model:
    grid = IrregSpacedGrid(points=np.array([1.0, 2.0, 4.0]))
    live = ConsumptionSavingsRegime(
        states={"wealth": grid},
        actions={"consumption": IrregSpacedGrid(points=np.array([0.05, 0.5, 2.0]))},
        functions={
            "utility": _utility,
            "inverse_marginal_utility": _inverse,
            "resources": _resources,
            "savings": _savings,
        },
        state_transitions={"wealth": {"live": _landing, "done": _landing}},
        liquid=LiquidMargin(
            state="wealth",
            action="consumption",
            resources="resources",
            post_decision_state="savings",
        ),
        solver=DCEGM(
            savings_grid=IrregSpacedGrid(points=np.array([0.25, 0.5, 1.5])),
            envelope=ExactEnvelope(),
        ),
    )
    return Model(
        regimes={
            "live": live,
            "done": Regime(states={"wealth": grid}, functions={"utility": _terminal}),
        },
        n_periods=3,
        initial_nodes=InitialNodes(
            by_period={2: "done"} if terminal_start else {0: "live"}
        ),
        edges={
            "live": {
                "live": Periods(values=(0,)),
                "done": Periods(values=(1,)),
            }
        },
        regime_id_class=_RegimeId,
        fixed_params=fixed_params,
    )


@pytest.mark.coverage(backends=("cpu",), precisions="both")
def test_original_self_transition_keeps_distinct_incoming_dates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Use actual canonical contexts; a regime cycle is an advancing date edge."""
    captured: dict[RegimeName, _SolutionBuild] = {}
    original = processing._with_declared_continuation_reads

    def observe(
        *,
        canonical_regimes: dict[RegimeName, processing.Regime],
        solution_builds: Mapping[RegimeName, _SolutionBuild],
    ) -> MappingProxyType[RegimeName, processing.Regime]:
        captured.update(solution_builds)
        return original(
            canonical_regimes=canonical_regimes, solution_builds=solution_builds
        )

    monkeypatch.setattr(processing, "_with_declared_continuation_reads", observe)
    _ordinary_model(fixed_params={})
    contexts = {name: build.context for name, build in captured.items()}
    assert _incoming_parent_dates(
        target="live", target_period=1, contexts=contexts
    ) == (("live", 0),)
    assert _incoming_parent_dates(
        target="done", target_period=2, contexts=contexts
    ) == (("live", 1),)
    assert (
        _incoming_parent_dates(target="live", target_period=0, contexts=contexts) == ()
    )
    solver = captured["live"].solver
    assert isinstance(solver, _BoundDCEGM)
    with pytest.raises(UnsupportedOperationError, match=r"missing source dates=.*live"):
        prepare_dated_balanced_kernels(
            contexts=contexts,
            solvers={"live": solver},
            declarations={},
            ordinary_templates={},
            ordinary_kernels={"done": captured["done"].kernels},
            call_contexts={},
            fixed_flat_params=MappingProxyType({}),
        )


@pytest.mark.coverage(backends=("cpu",), precisions="both")
def test_metadata_template_and_normalized_fixed_operands_precede_kernels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Early metadata agrees with real canonical namespaces and fixed operands."""
    calls: list[_ProcessArguments] = []
    original = model_processing.process_regimes

    def observe(**kwargs: Unpack[_ProcessArguments]):
        calls.append(kwargs)
        return original(**kwargs)

    monkeypatch.setattr(model_processing, "process_regimes", observe)
    model = _ordinary_model(fixed_params={"scale": 2.0})
    actual = calls[-1]

    def forbidden(
        *_args: object,  # noqa: PAN001 - refusal sentinel never reads operands
        **_kwargs: object,  # noqa: PAN001 - refusal sentinel never reads operands
    ) -> Never:
        pytest.fail("Metadata-only preparation attempted to construct solver kernels.")

    monkeypatch.setattr(processing._CanonicalRegimeBuilder, "__call__", forbidden)
    prepared = prepare_dated_parameters(
        **actual,
        fixed_params={"scale": 2.0},
        params_already_consumed=frozenset(),
        phase_transitions=model.declared_transitions,
    )
    metadata = prepared.builder.parameter_metadata()
    raw_template = create_params_template(
        regimes=model._regimes,
        query_periods_by_phase=actual["prepared_structure"].query_periods_by_phase,
    )
    early_template = create_params_template(
        regimes=metadata,
        query_periods_by_phase=actual["prepared_structure"].query_periods_by_phase,
    )
    assert early_template == raw_template
    assert prepared.fixed.remaining_template == model._params_template
    for name, regime in model._regimes.items():
        expected = regime_kernel_params(prepared.fixed.granular, regime_name=name)
        assert set(expected) == set(regime.solution.resolved_fixed_params)
        for key, value in expected.items():
            np.testing.assert_array_equal(
                value, regime.solution.resolved_fixed_params[key]
            )


@pytest.mark.coverage(backends=("cpu",), precisions="both")
def test_ordinary_fixed_and_free_utility_operands_keep_the_same_values() -> None:
    """Actual ordinary solves retain fixed/free algebra and terminal carry binding."""
    fixed = _ordinary_model(fixed_params={"scale": 2.0}).solve(
        params={"discount_factor": 0.95}, log_level="off"
    )
    free = _ordinary_model(fixed_params={}).solve(
        params={"scale": 2.0, "discount_factor": 0.95}, log_level="off"
    )
    assert fixed.values.keys() == free.values.keys()
    for period, regimes in fixed.values.items():
        assert regimes.keys() == free.values[period].keys()
        for name, value in regimes.items():
            np.testing.assert_array_almost_equal(
                value, free.values[period][name], decimal=DECIMAL_PRECISION
            )


@pytest.mark.coverage(backends=("cpu",), precisions="both")
def test_original_balanced_guard_refuses_an_equivalent_schema_at_another_date(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The original produced guard's clock cannot be replaced by equal leaf shapes."""
    fixture = _native_balanced(monkeypatch)
    row = fixture.produce(fixture.pool).projected
    assert isinstance(row, BalancedProjectedEGMCarry)
    assert isinstance(fixture.spec, ProjectedContinuationSpec)
    date = fixture.spec.date_at(period=1)
    changed = replace(date, authority=replace(date.authority, period=0))
    with pytest.raises(TypeError, match="aggregate identity"):
        replace(fixture.spec, dates_by_period={1: changed})
    _, query = _smooth_query(row)
    reader = ProjectedChildReader.bind_date(
        date=changed,
        target="child",
        carry=row,
        next_liquid=lambda savings: savings,
    )
    result = reader(query)
    assert bool(_coverage_missing(result))
    assert date is fixture.spec.date_at(period=1)
    assert date.authority is fixture.spec.authority_at(period=1)
    clock = row.guard.pool["period"]
    assert isinstance(clock, jax.Array)
    poisoned = replace(
        row,
        guard=replace(
            row.guard,
            pool_values=tuple(
                jnp.repeat(clock, 2) if name == "period" else value
                for name, value in zip(
                    row.guard.pool_names, row.guard.pool_values, strict=True
                )
            ),
        ),
    )
    with pytest.raises(TypeError, match="producer schema"):
        ProjectedChildReader.bind_date(
            date=date,
            target="child",
            carry=poisoned,
            next_liquid=lambda savings: savings,
        )


@pytest.mark.coverage(backends=("cpu",), precisions="both")
def test_real_zero_node_source_has_an_explicit_unrepresented_consumer_refusal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A real terminal-only start leaves the nonterminal source without a kernel."""
    captured: dict[RegimeName, _SolutionBuild] = {}
    original = processing._with_declared_continuation_reads

    def observe(
        *,
        canonical_regimes: dict[RegimeName, processing.Regime],
        solution_builds: Mapping[RegimeName, _SolutionBuild],
    ) -> MappingProxyType[RegimeName, processing.Regime]:
        captured.update(solution_builds)
        return original(
            canonical_regimes=canonical_regimes, solution_builds=solution_builds
        )

    monkeypatch.setattr(processing, "_with_declared_continuation_reads", observe)
    model = _ordinary_model(fixed_params={}, terminal_start=True)
    source = captured["live"]
    assert not model.graph.laws["live"].terminal
    assert source.context.regimes_to_active_periods["live"] == ()
    assert not source.kernels.period_kernels
    with pytest.raises(
        UnsupportedOperationError,
        match=r"consumer 'live' at period 0 lacks original scalar Euler pieces",
    ):
        _prepared_parent_pieces(
            context=source.context, period=0, steps=MappingProxyType({})
        )
