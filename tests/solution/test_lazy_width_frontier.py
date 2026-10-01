"""A budgeted solve binds the width candidates admission actually asks for.

`plan_workspace` walks a core's ranked width frontier widest first and keeps the
first candidate admission admits, so every narrower candidate of a core whose
widest one fits is resolved, lowered and traced for nothing. The planner binds
candidate `k + 1` only once candidate `k` has been refused.

These witnesses hold the binding against the decision: the number of candidates
bound per core, and — against an arm that binds the whole frontier ahead of
admission, the way the planner used to — the selected widths, the solved values
and the refusal a core that fits at no width reports.
"""

import dataclasses
import math
from collections.abc import Mapping
from typing import Any, cast

import jax
import jax.numpy as jnp
import pytest

from _lcm.execution.workspace_planning import workspace_width_candidates
from _lcm.solution import backward_induction
from lcm import ExecutionConfig
from lcm.exceptions import ExecutionPlanningError
from tests.solution.test_footprint_width_selection import (
    _ACTION_EXTENT,
    _FULL_WIDTH_PRODUCT,
    _build_model,
    _fake_peak,
    _fixed_fixture_bytes,
    _value_bytes,
)


@dataclasses.dataclass(kw_only=True)
class _SolveObservation:
    """What one budgeted solve bound, selected and produced."""

    bound_per_triple: dict[tuple[str, int, str], int]
    frontier_lengths: Mapping[tuple[str, int, str], int]
    resolutions: int
    argument_keys: int
    selected_widths: dict[tuple[str, int], tuple[tuple[str, int], ...]]
    axes_by_triple: dict[tuple[str, int, str], tuple[Any, ...]]
    values: Any


def _bind_whole_frontier(
    frontier: backward_induction._LazyCandidateFrontier, /, **asked: Any
) -> tuple[tuple[str, int, str], tuple[tuple[str, int], ...]]:
    """Bind every candidate of a core before admission sees the first one.

    This is the binding the planner performed before the frontier became lazy,
    installed over `_LazyCandidateFrontier.candidate` as the arm the lazy one is
    held against. It takes that method's arguments as `**asked` because it is a
    method body, not a callable of this module.
    """
    triple = asked["triple"]
    bound = frontier.candidates_by_triple[triple]
    while len(bound) < frontier.frontier_lengths[triple]:
        frontier._bind_next(triple=triple)
    return bound[asked["position"]]


def _observe_solve(
    *,
    monkeypatch: pytest.MonkeyPatch,
    budget_bytes: int,
    bind_whole_frontier: bool = False,
) -> _SolveObservation:
    """Solve the footprint fixture under a budget and report what it bound."""
    frontiers: list[backward_induction._LazyCandidateFrontier] = []
    compiled: list[backward_induction._CompiledPrograms] = []
    resolutions = 0
    argument_keys = 0
    original_planning = backward_induction._resolve_output_layouts_and_lowering_keys
    original_candidates = backward_induction.resolve_core_program_candidates
    original_argument_key = backward_induction._abstract_arguments_key
    original_compile = backward_induction._compile_all_functions

    def capture_compiled(**kwargs: Any) -> backward_induction._CompiledPrograms:
        result = original_compile(**kwargs)
        compiled.append(result)
        return result

    def observe_planning(**kwargs: Any) -> tuple:
        result = original_planning(**kwargs)
        frontiers.append(result[7])
        return result

    def count_candidates(**kwargs: Any) -> tuple:
        nonlocal resolutions
        resolutions += len(kwargs["tile_widths"])
        return original_candidates(**kwargs)

    def count_argument_keys(**kwargs: Any) -> Any:
        nonlocal argument_keys
        argument_keys += 1
        return original_argument_key(**kwargs)

    monkeypatch.setattr(backward_induction, "compiler_memory_reservation", _fake_peak)
    monkeypatch.setattr(
        backward_induction,
        "_resolve_output_layouts_and_lowering_keys",
        observe_planning,
    )
    monkeypatch.setattr(
        backward_induction, "resolve_core_program_candidates", count_candidates
    )
    monkeypatch.setattr(
        backward_induction, "_abstract_arguments_key", count_argument_keys
    )
    monkeypatch.setattr(backward_induction, "_compile_all_functions", capture_compiled)
    if bind_whole_frontier:
        monkeypatch.setattr(
            backward_induction._LazyCandidateFrontier,
            "candidate",
            _bind_whole_frontier,
        )

    model = _build_model(
        execution_config=ExecutionConfig(device_memory_bytes=budget_bytes)
    )
    params = cast("dict[str, Any]", model.get_params_template())
    params["acting"]["koopmans_aggregator"]["discount_factor"] = 0.5
    solution = model.solve(params=params, log_level="off")
    (frontier,) = frontiers
    (programs,) = compiled
    return _SolveObservation(
        bound_per_triple={
            triple: len(bound)
            for triple, bound in frontier.candidates_by_triple.items()
        },
        frontier_lengths=frontier.frontier_lengths,
        resolutions=resolutions,
        argument_keys=argument_keys,
        selected_widths={
            cell: tuple(cores["main"].tile_widths.items())
            for cell, cores in programs.executables.items()
        },
        axes_by_triple={
            triple: frontier.resolved_programs[bound[0]].requirements.axes
            for triple, bound in frontier.candidates_by_triple.items()
        },
        values=solution.values,
    )


_GENEROUS_BUDGET = _fixed_fixture_bytes() + (_FULL_WIDTH_PRODUCT + 3) * _value_bytes()
_NARROWING_BUDGET = _fixed_fixture_bytes() + _ACTION_EXTENT * _value_bytes()


def test_a_core_admitted_at_its_widest_binds_that_candidate_alone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The frontier below an admitted widest candidate is never bound."""
    observed = _observe_solve(monkeypatch=monkeypatch, budget_bytes=_GENEROUS_BUDGET)

    assert (
        set(observed.bound_per_triple.values()),
        observed.resolutions,
        max(observed.frontier_lengths.values()),
    ) == ({1}, len(observed.bound_per_triple), max(observed.frontier_lengths.values()))
    assert max(observed.frontier_lengths.values()) > 1


def test_a_refused_candidate_binds_exactly_one_more(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A core binds as many candidates as admission refused, plus the winner."""
    observed = _observe_solve(monkeypatch=monkeypatch, budget_bytes=_NARROWING_BUDGET)

    assert observed.resolutions == sum(observed.bound_per_triple.values())
    assert any(bound > 1 for bound in observed.bound_per_triple.values())
    assert all(
        bound <= observed.frontier_lengths[triple]
        for triple, bound in observed.bound_per_triple.items()
    )


def test_the_bound_candidate_count_is_the_selected_rank(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A core stops at the rank it was admitted at, and binds nothing beyond it."""
    observed = _observe_solve(monkeypatch=monkeypatch, budget_bytes=_NARROWING_BUDGET)

    for (regime_name, period), widths in observed.selected_widths.items():
        triple = (regime_name, period, "main")
        if triple not in observed.bound_per_triple:
            continue
        frontier = workspace_width_candidates(
            axes=observed.axes_by_triple[triple],
            budget_bytes=_NARROWING_BUDGET,
        )
        rank = [tuple(candidate.items()) for candidate in frontier].index(widths)
        assert observed.bound_per_triple[triple] == rank + 1


def test_lazy_binding_selects_what_whole_frontier_binding_selects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Binding on refusal moves neither the selected widths nor the values."""
    lazy = _observe_solve(monkeypatch=monkeypatch, budget_bytes=_NARROWING_BUDGET)
    eager = _observe_solve(
        monkeypatch=monkeypatch,
        budget_bytes=_NARROWING_BUDGET,
        bind_whole_frontier=True,
    )

    lazy_leaves = jax.tree.leaves(lazy.values)
    eager_leaves = jax.tree.leaves(eager.values)

    assert lazy.selected_widths == eager.selected_widths
    assert len(lazy_leaves) == len(eager_leaves)
    assert all(
        bool(jnp.array_equal(got, expected))
        for got, expected in zip(lazy_leaves, eager_leaves, strict=True)
    )
    assert lazy.resolutions < eager.resolutions


def test_one_argument_description_per_core_survives_the_refusals(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A candidate bound after a refusal reuses its core's argument description."""
    observed = _observe_solve(monkeypatch=monkeypatch, budget_bytes=_NARROWING_BUDGET)

    assert observed.resolutions > len(observed.bound_per_triple)
    assert observed.argument_keys == len(observed.bound_per_triple)


def _refuse(*, monkeypatch: pytest.MonkeyPatch, bind_whole_frontier: bool) -> str:
    """Solve at a budget no width serves and return the refusal's message."""
    with pytest.raises(ExecutionPlanningError) as refusal:
        _observe_solve(
            monkeypatch=monkeypatch,
            budget_bytes=_fixed_fixture_bytes() + _value_bytes() // 2,
            bind_whole_frontier=bind_whole_frontier,
        )
    return str(refusal.value)


def test_a_core_that_fits_at_no_width_reports_the_same_refusal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Total refusal still walks the whole frontier, so its diagnosis is unchanged."""
    lazy = _refuse(monkeypatch=monkeypatch, bind_whole_frontier=False)
    eager = _refuse(monkeypatch=monkeypatch, bind_whole_frontier=True)

    assert lazy == eager
    assert "budget" in lazy


def test_the_width_product_of_the_selection_is_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The generous budget still spends itself on the full extent."""
    observed = _observe_solve(monkeypatch=monkeypatch, budget_bytes=_GENEROUS_BUDGET)

    assert (
        math.prod(dict(observed.selected_widths[("acting", 2)]).values())
        == _FULL_WIDTH_PRODUCT
    )
