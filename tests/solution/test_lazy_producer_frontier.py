"""A consumed producer's width frontier is resolved one bound candidate at a time.

A budget offers the planner a ranked frontier of widths per core. A consumer is
lowered against its producer's top-ranked record, so only that record is traced
up front; a narrower candidate is traced, and held against the top-ranked record,
when a refusal or a width search actually binds it.
"""

from typing import Any

import jax
import pytest

from _lcm.execution import internal_outputs
from _lcm.solution import backward_induction
from lcm import ExecutionConfig
from lcm.exceptions import ExecutionPlanningError
from tests.conftest import EXACT_KERNEL_SKIP_REASON
from tests.test_models import negm_kinked_toy

_GENEROUS_BUDGET = 2**40


def _solve_counting(
    *, monkeypatch: pytest.MonkeyPatch, device_memory_bytes: int | None
) -> tuple[int, list[int], list[Any]]:
    """Solve the kinked NEGM toy; count producer traces and frontier sizes."""
    traced: list[int] = []
    frontier_sizes: list[int] = []
    frontiers: list[Any] = []
    resolve = backward_induction.resolve_producer
    candidates = backward_induction.workspace_width_candidates
    structural = backward_induction._resolve_output_layouts_and_lowering_keys

    def counting_resolve(**kwargs: Any) -> Any:
        traced.append(1)
        return resolve(**kwargs)

    def sized_candidates(**kwargs: Any) -> Any:
        widths = candidates(**kwargs)
        frontier_sizes.append(len(widths))
        return widths

    def kept_structural(**kwargs: Any) -> Any:
        result = structural(**kwargs)
        frontiers.append(result[-1])
        return result

    monkeypatch.setattr(backward_induction, "resolve_producer", counting_resolve)
    monkeypatch.setattr(
        backward_induction, "workspace_width_candidates", sized_candidates
    )
    monkeypatch.setattr(
        backward_induction,
        "_resolve_output_layouts_and_lowering_keys",
        kept_structural,
    )
    model = negm_kinked_toy.build_model(
        execution_config=ExecutionConfig(device_memory_bytes=device_memory_bytes),
    )
    model.solve(params={"discount_factor": 0.95, "alive": {}}, log_level="off")
    return len(traced), frontier_sizes, frontiers


@pytest.mark.requires_exact_affine_kernel(reason=EXACT_KERNEL_SKIP_REASON)
def test_a_budget_traces_each_consumed_producer_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With every top-ranked width admitted, a budget traces what no budget does."""
    unbudgeted, _, _ = _solve_counting(
        monkeypatch=monkeypatch, device_memory_bytes=None
    )
    budgeted, frontier_sizes, _ = _solve_counting(
        monkeypatch=monkeypatch, device_memory_bytes=_GENEROUS_BUDGET
    )

    # Positive control: the budget does offer a frontier of several widths.
    assert max(frontier_sizes) > 1
    assert budgeted == unbudgeted > 0


def _keeper_frontier(frontier: Any) -> tuple[Any, tuple[str, int, str]]:
    """Return the lazy frontier and one consumed producer triple with a frontier."""
    lazy = frontier
    triples = [
        triple
        for triple, core in lazy.frontiers.items()
        if core.top_record is not None and len(core.widths) > 1
    ]
    assert triples
    return lazy, triples[0]


@pytest.mark.requires_exact_affine_kernel(reason=EXACT_KERNEL_SKIP_REASON)
def test_binding_a_later_producer_width_traces_and_checks_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A narrower candidate is traced once, when bound, against the top record."""
    _, _, frontiers = _solve_counting(
        monkeypatch=monkeypatch, device_memory_bytes=_GENEROUS_BUDGET
    )
    lazy, triple = _keeper_frontier(frontiers[-1])
    traced: list[int] = []
    resolve = internal_outputs.resolve_producer

    def counting_resolve(**kwargs: Any) -> Any:
        traced.append(1)
        return resolve(**kwargs)

    monkeypatch.setattr(backward_induction, "resolve_producer", counting_resolve)

    lazy.candidate(triple=triple, position=1)

    assert len(traced) == 1


@pytest.mark.requires_exact_affine_kernel(reason=EXACT_KERNEL_SKIP_REASON)
def test_a_later_width_publishing_a_different_subtree_is_refused_when_bound(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Width dependence of a published output is caught at the width that binds."""
    _, _, frontiers = _solve_counting(
        monkeypatch=monkeypatch, device_memory_bytes=_GENEROUS_BUDGET
    )
    lazy, triple = _keeper_frontier(frontiers[-1])
    resolve = internal_outputs.resolve_producer

    def width_dependent_resolve(**kwargs: Any) -> Any:
        record = resolve(**kwargs)
        widened = jax.tree.map(
            lambda leaf: jax.ShapeDtypeStruct((*leaf.shape, 2), leaf.dtype),
            record.abstract_output,
        )
        return internal_outputs.ResolvedProducer(
            name=record.name,
            function=record.function,
            internal_input_templates=record.internal_input_templates,
            static_kwargs=record.static_kwargs,
            internal_outputs=record.internal_outputs,
            abstract_output=widened,
        )

    monkeypatch.setattr(backward_induction, "resolve_producer", width_dependent_resolve)

    with pytest.raises(ExecutionPlanningError, match="may not depend on the"):
        lazy.candidate(triple=triple, position=1)
