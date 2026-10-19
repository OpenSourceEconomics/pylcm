"""Unexecuted public NNBEGM literal-parent / callable NB-child witness.

Integration: copy this file and oracle8_child_status.py into tests/solution/.
Native command after integration (not run by the reviewer):
  pytest tests/solution/test_round8_literal_parent_child_status.py -q

Only the parent's bound representation changes from the shipped resource
fixture: callable zero becomes literal zero.  Economics and the actual child
reader remain the same.  The finite route isolates this transport obligation
without introducing adaptive mesh-convergence assumptions.  The corresponding
adaptive source path has the same child-only publication omission; acceptance
must cover that retained route too under its existing convergence contract.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.solution.feasibility import (
    ChildStatusKernelOutput,
    ContinuationReadCounts,
    FeasibilityKernelOutput,
    QueryInputStatus,
)
from _lcm.solution.nbegm import _RideAlongNBEGMPeriodKernel
from _lcm.solution.nnbegm import _InheritedChildSummary, _NNBEGMPeriodKernel
from lcm import (
    AgeGrid,
    ConsumptionSavingsRegime,
    IrregSpacedGrid,
    LiquidMargin,
    Model,
    NestedConsumptionSavingsRegime,
    OuterContinuousMargin,
    Regime,
    outer_unchanged,
    post_decision_lower_bound,
)
from lcm.exceptions import InvalidValueFunctionError
from lcm.solver_api import KernelOutput, ResultRetention
from lcm.solvers import NBEGM, NNBEGM, AdaptiveOuterMesh, FiniteOuterGrid
from tests.conftest import EXACT_KERNEL_SKIP_REASON
from tests.solution import test_nbegm_query_actual_resources as fixture
from tests.solution._oracle8_child_status import (
    ExecutedRow,
    fold_executed,
    require_publication,
)

pytestmark = pytest.mark.requires_exact_affine_kernel(reason=EXACT_KERNEL_SKIP_REASON)


def make_literal_parent_model(*, resources, outer_search=None) -> Model:
    """Use the shipped admitted model with a literal parent and DAG child floor."""
    margin = LiquidMargin(
        state="liquid",
        action="consumption",
        resources="resources",
        post_decision_state="savings",
    )
    common = {
        "utility": fixture.utility,
        "savings": fixture.savings,
        "inverse_marginal_utility": fixture.inverse_marginal_utility,
    }
    parent = NestedConsumptionSavingsRegime(
        states={"liquid": fixture._LIQUID, "stock": fixture._STOCK},
        actions={
            "consumption": fixture._CONSUMPTION,
            "investment": fixture._INVESTMENT,
        },
        state_transitions={
            "liquid": fixture.next_liquid,
            "stock": fixture.next_parent_stock,
        },
        functions={
            **common,
            "resources": fixture.parent_resources,
            "new_stock": fixture.new_stock,
        },
        constraints={"borrowing": post_decision_lower_bound(margin=margin, lower=0.0)},
        liquid=margin,
        outer_continuous=OuterContinuousMargin(
            state="stock",
            action="investment",
            post_decision_state="new_stock",
            no_adjustment=outer_unchanged,
        ),
        solver=NNBEGM(
            inner=NBEGM(savings_grid=fixture._SAVINGS),
            outer_search=outer_search
            or FiniteOuterGrid(grid=IrregSpacedGrid(points=np.array(fixture._OUTER))),
        ),
    )
    child = ConsumptionSavingsRegime(
        koopmans_aggregator=parent.koopmans_aggregator,
        certainty_equivalent=parent.certainty_equivalent,
        states={"liquid": fixture._LIQUID, "stock": fixture._STOCK},
        actions={"consumption": fixture._CONSUMPTION},
        state_transitions={
            "liquid": fixture.next_liquid,
            "stock": fixture.next_child_stock,
        },
        functions={**common, "resources": resources},
        constraints={
            "borrowing": post_decision_lower_bound(margin=margin, lower=fixture.floor)
        },
        liquid=margin,
        solver=NBEGM(savings_grid=fixture._SAVINGS),
    )
    model = Model(
        regimes={
            "parent": parent,
            "child": child,
            "done": Regime(
                states={"liquid": fixture._LIQUID, "stock": fixture._STOCK},
                functions={"utility": fixture.terminal},
            ),
        },
        ages=AgeGrid(exact_values=(0, 10, 20)),
        edges={"parent": {"child": (0,)}, "child": {"done": (10,)}},
        initial_nodes={0: "parent"},
        regime_id_class=fixture.RegimeId,
    )
    fixture.fail_if_not_actual_nb_query_path(model=model)
    kernel = model._regimes["parent"].solution.period_kernels[0]
    assert isinstance(kernel, _NNBEGMPeriodKernel)
    assert kernel.query_model is None
    authorities = (model._regimes["parent"].solution.query_input_authorities or {}).get(
        0, ()
    )
    assert authorities, "This witness requires the actual inherited child slots."
    return model


def _row(*, name, output) -> ExecutedRow:
    slots = tuple(
        tuple(bool(np.asarray(flag)) for flag in status)
        for status in (getattr(output, "query_status", None) or ())
    )
    native_counts = getattr(output, "continuation_read_counts", None)
    counts = (
        None
        if native_counts is None
        else tuple(int(np.asarray(x)) for x in native_counts)
    )
    return ExecutedRow(name=name, slots=slots, counts=counts)


@pytest.mark.parametrize("retention", list(ResultRetention))
@pytest.mark.parametrize("runtime_checks", [False, True])
@pytest.mark.parametrize("disable_jit", [False, True])
def test_public_literal_parent_preserves_executed_child_summaries(
    *, monkeypatch, runtime_checks, disable_jit, retention
) -> None:
    model = make_literal_parent_model(resources=fixture.child_positive_resources)
    authorities = (model._regimes["parent"].solution.query_input_authorities or {})[0]
    native_inner = _RideAlongNBEGMPeriodKernel.__call__
    native_outer = _NNBEGMPeriodKernel.__call__
    stack = []
    observed = []

    def observe_inner(self, **kwargs):
        output = native_inner(self, **kwargs)
        if stack and self.regime_name == "parent":
            rows = stack[-1]
            rows.append(
                _row(
                    name="keeper" if not rows else f"adjuster-{len(rows)}",
                    output=output,
                )
            )
        return output

    def observe_outer(self, **kwargs):
        if self.regime_name != "parent":
            return native_outer(self, **kwargs)
        rows = []
        stack.append(rows)
        try:
            output = native_outer(self, **kwargs)
        finally:
            stack.pop()
        assert len(rows) == 1 + len(fixture._OUTER)
        # Model.solve runtime_checks selects the normal checked variant even
        # with log_level="off". Require real count collection, not an all-None
        # inner/outer pair that would vacuously agree with the optional oracle.
        if runtime_checks:
            assert all(row.counts is not None for row in rows)
            assert any(row.counts[1] > 0 for row in rows)
        else:
            assert all(row.counts is None for row in rows)
        expected = fold_executed(n_slots=len(authorities), rows=tuple(rows))
        require_publication(
            expected=expected, actual=_row(name="published", output=output)
        )
        observed.append(expected)
        return output

    monkeypatch.setattr(_RideAlongNBEGMPeriodKernel, "__call__", observe_inner)
    monkeypatch.setattr(_NNBEGMPeriodKernel, "__call__", observe_outer)
    with jax.disable_jit(disable_jit):
        model.solve(
            params=fixture.PARAMS,
            log_level="off",
            runtime_checks=runtime_checks,
            retention=retention,
        )
    assert observed


@pytest.mark.parametrize("retention", list(ResultRetention))
@pytest.mark.parametrize("runtime_checks", [False, True])
def test_public_literal_parent_rejects_actual_nonfinite_child_budget(
    *, runtime_checks, retention
) -> None:
    """The host must report the actual child failure, with logging/checks irrelevant."""
    model = make_literal_parent_model(resources=fixture.child_nan_resources)
    with pytest.raises(InvalidValueFunctionError, match="query_coverage_missing"):
        model.solve(
            params=fixture.PARAMS,
            log_level="off",
            runtime_checks=runtime_checks,
            retention=retention,
        )


def _native_row(*, slots, counts):
    """Build one executed literal inner result with native scalar summaries."""
    return FeasibilityKernelOutput(
        value=jnp.zeros(2),
        feasibility_counts=(jnp.int32(0), jnp.int32(0)),
        query_status=tuple(
            QueryInputStatus(*(jnp.asarray(flag) for flag in slot)) for slot in slots
        ),
        continuation_read_counts=(
            None
            if counts is None
            else ContinuationReadCounts(*(jnp.int32(count) for count in counts))
        ),
    )


def test_inherited_child_fold_preserves_ordered_slots_and_sums_counts() -> None:
    """The production fold is the oracle fold on two ordered slots."""
    rows = (
        ExecutedRow("keeper", ((True, False, False), (False, False, False)), (1, 3)),
        ExecutedRow("winner", ((False, False, False), (False, True, False)), (2, 4)),
        ExecutedRow("excluded", ((False, False, False), (False, False, True)), (2, 4)),
    )
    native = tuple(_native_row(slots=row.slots, counts=row.counts) for row in rows)
    summary = _InheritedChildSummary.of(native[0])
    for row in native[1:]:
        summary = summary.add(row)
    published = summary.publish(KernelOutput(value=jnp.zeros(2)))
    assert type(published) is ChildStatusKernelOutput
    require_publication(
        expected=fold_executed(n_slots=2, rows=rows),
        actual=_row(name="published", output=published),
    )


def test_inherited_child_fold_rejects_mixed_count_schema() -> None:
    """A row without requested counts cannot be laundered into a partial sum."""
    slots = ((False, False, False),)
    summary = _InheritedChildSummary.of(_native_row(slots=slots, counts=(0, 1)))
    with pytest.raises(TypeError, match="read-count schema"):
        summary.add(_native_row(slots=slots, counts=None))


def test_inherited_child_fold_rejects_a_row_that_lost_its_slots() -> None:
    """Every executed solve carries the declared slots, even when all false."""
    summary = _InheritedChildSummary.of(
        _native_row(slots=((False, False, False),), counts=None)
    )
    with pytest.raises(TypeError, match="declaration slots"):
        summary.add(KernelOutput(value=jnp.zeros(2)))


def test_public_literal_adaptive_parent_preserves_every_executed_node() -> None:
    """A converged adaptive mesh publishes the child slots of every solved node."""
    model = make_literal_parent_model(
        resources=fixture.child_positive_resources,
        # Loose tolerances let this kinked surface converge under the mesh's
        # own fail-closed contract; the test checks status transport only.
        outer_search=AdaptiveOuterMesh(
            initial_grid=IrregSpacedGrid(points=np.array(fixture._OUTER)),
            value_atol=1.0,
            value_rtol=1.0,
        ),
    )
    authorities = (model._regimes["parent"].solution.query_input_authorities or {})[0]
    native_inner = _RideAlongNBEGMPeriodKernel.__call__
    native_outer = _NNBEGMPeriodKernel.__call__
    stack = []
    observed = []

    def observe_inner(self, **kwargs):
        output = native_inner(self, **kwargs)
        if stack and self.regime_name == "parent":
            stack[-1].append(_row(name=f"row-{len(stack[-1])}", output=output))
        return output

    def observe_outer(self, **kwargs):
        if self.regime_name != "parent":
            return native_outer(self, **kwargs)
        rows = []
        stack.append(rows)
        try:
            output = native_outer(self, **kwargs)
        finally:
            stack.pop()
        assert len(rows) >= 1 + len(fixture._OUTER)
        assert all(row.counts is not None for row in rows)
        expected = fold_executed(n_slots=len(authorities), rows=tuple(rows))
        require_publication(
            expected=expected, actual=_row(name="published", output=output)
        )
        observed.append(expected)
        return output

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(_RideAlongNBEGMPeriodKernel, "__call__", observe_inner)
        patch.setattr(_NNBEGMPeriodKernel, "__call__", observe_outer)
        model.solve(params=fixture.PARAMS, log_level="off", runtime_checks=True)
    assert observed
