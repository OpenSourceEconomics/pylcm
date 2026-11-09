"""A standalone driver around the bounded width selector.

The solve loop drives `BoundedWidthSelector` itself, one proposal and one
recorded decision at a time. This driver runs the same seed, shrink and refine
walk to completion for one core, so the selector's policy can be tested on
synthetic compilers and memory records.
"""

from collections.abc import Callable, Collection, Mapping
from dataclasses import dataclass
from types import MappingProxyType

from _lcm.execution.compiler_memory import CompilerMemoryReport
from _lcm.execution.core_program import ReducedAxis, TiledOutputAxis
from _lcm.execution.workspace_planning import (
    BoundedWidthSelector,
    CompilerMemoryReservation,
    WorkspaceBudgetError,
    WorkspacePlan,
    _memory_for_candidate,
    _resident_bytes_for_candidate,
    _resident_exhausts_budget_message,
    _validate_axes,
    _validate_budget,
    _validate_fixed_widths,
    _validate_resident_bytes,
    _validate_width_ceilings,
    _widths_key,
    bootstrap_widths,
    compiler_memory_reservation,
    plan_workspace,
)
from lcm.exceptions import ExecutionPlanningError
from lcm.execution import WidthSearch, WidthSearchPolicy

# The policy a caller who declares none plans under: the ranked walk.
_EXHAUSTIVE_POLICY = WidthSearchPolicy()


def plan_workspace_bounded[Compiled](
    *,
    axes: tuple[ReducedAxis | TiledOutputAxis, ...],
    fixed_widths: Mapping[str, int] = MappingProxyType({}),
    width_ceilings: Mapping[str, int] = MappingProxyType({}),
    compile_candidate: Callable[[Mapping[str, int]], Compiled],
    budget_bytes: int | None = None,
    memory_for: Callable[[Compiled], CompilerMemoryReservation] | None = None,
    resident_bytes: int = 0,
    resident_bytes_for: Callable[[Compiled], int] | None = None,
    policy: WidthSearchPolicy = _EXHAUSTIVE_POLICY,
    hint: Mapping[str, int] | None = None,
    cached_analysis_for: Callable[[Mapping[str, int]], CompilerMemoryReport | None]
    | None = None,
    covered_axes: Collection[str] = (),
) -> WorkspacePlan[Compiled]:
    """Search a bounded number of widths for one the budget admits.

    Admission is `plan_workspace`'s: a candidate fits when its represented
    compiler reservation plus the bytes the plan keeps resident at the node's
    position fits the budget, and the returned executable is the exact object
    compiled for the selected width. What changes is how many widths are
    compiled to find it. Under `WidthSearch.EXHAUSTIVE` this is `plan_workspace`
    itself. Under `WidthSearch.BOUNDED` the search runs:

    - **seed** — a compatible `hint`, else the widest candidate under
      `seed="widest"`, else the conservative bootstrap anchor (`bootstrap_widths`),
      which is narrower than the full extent on every unfixed axis.
    - **shrink** — on refusal, halve one unfixed axis and round the half down
      onto the widths that axis admits. The axis halved is the one declaring the
      largest extent, ties going to declaration order. An axis already at its
      narrowest admissible width is passed over, and the search is exhausted once
      every unfixed axis is.
    - **refine** — on the first admission, widen one axis at a time, most
      recently shrunk first, by bisecting between the admitted width and the
      last width refused on that axis. That refused ceiling was measured while
      every other axis was still at its wider, pre-shrink width, so it bounds
      the axis more tightly than the admitted configuration requires and the
      refinement is conservative. A proposal that rounds back onto the admitted
      width cannot widen, so that axis is done.
    - **stop** — at `max_evaluations` distinct width mappings, at
      `refinement_share` refinement evaluations, or when refinement cannot widen.

    Memory is treated as nonmonotone in the widths throughout: a refusal at one
    width says nothing about its neighbours, which is why refinement compiles
    every width it admits rather than inferring one.

    Args:
        axes: The reduced and tiled axes the core declares, in declaration order.
        fixed_widths: Widths the caller pinned; a pinned axis is never proposed
            at another width.
        compile_candidate: Compiles one width mapping into an executable.
        budget_bytes: Per-device ceiling, or `None` to compile the bootstrap
            mapping once without consulting any memory report.
        memory_for: Reads one executable's reservation, or `None` to read the
            compiler report directly.
        resident_bytes: Bytes the plan keeps resident at the node's position.
        resident_bytes_for: Refines that lower bound per executable.
        policy: Which search to run and the budget it runs under.
        hint: A width mapping to evaluate first — the one `policy.hints` holds
            for this core's regime, selected by the caller. A hint any axis
            declaration refuses, or one that moves a fixed axis, is logged and
            skipped.
        cached_analysis_for: Returns a compiler report the caller already holds
            for a width mapping, or `None`. A report that refuses the width
            spends an evaluation and no compilation; one that admits it is
            followed by the compilation that owns the returned executable.
        covered_axes: Axes whose conservative seed is their whole extent where
            `bootstrap_widths` covers them.

    Returns:
        The widest admitted candidate the search reached, with its already
        compiled executable.

    Raises:
        ExecutionPlanningError: The node's residency already exhausts the
            budget, or the search spent its evaluations without an admission.

    """
    if policy.kind is WidthSearch.EXHAUSTIVE:
        return plan_workspace(
            axes=axes,
            fixed_widths=fixed_widths,
            width_ceilings=width_ceilings,
            compile_candidate=compile_candidate,
            budget_bytes=budget_bytes,
            memory_for=memory_for,
            resident_bytes=resident_bytes,
            resident_bytes_for=resident_bytes_for,
            covered_axes=covered_axes,
        )

    declared_axes = _validate_axes(axes=axes)
    widths_by_axis = _validate_fixed_widths(fixed_widths=fixed_widths)
    ceilings = _validate_width_ceilings(width_ceilings=width_ceilings)
    budget = _validate_budget(budget_bytes=budget_bytes)
    resident = _validate_resident_bytes(resident_bytes=resident_bytes)
    if not callable(compile_candidate):
        raise TypeError("The workspace candidate compiler must be callable.")
    if memory_for is not None and not callable(memory_for):
        raise TypeError("The workspace memory lookup must be callable or None.")
    if resident_bytes_for is not None and not callable(resident_bytes_for):
        raise TypeError("The workspace residency lookup must be callable or None.")
    if cached_analysis_for is not None and not callable(cached_analysis_for):
        raise TypeError("The workspace profile lookup must be callable or None.")

    if budget is None:
        widths = bootstrap_widths(
            axes=declared_axes,
            fixed_widths=widths_by_axis,
            width_ceilings=ceilings,
            covered_axes=covered_axes,
        )
        return WorkspacePlan(
            widths=widths, peak_bytes=None, compiled=compile_candidate(widths)
        )
    if resident >= budget:
        raise WorkspaceBudgetError(
            _resident_exhausts_budget_message(resident=resident, budget=budget)
        )

    search = _BoundedWidthSearch(
        axes=declared_axes,
        fixed_widths=widths_by_axis,
        width_ceilings=ceilings,
        compile_candidate=compile_candidate,
        budget=budget,
        memory_for=memory_for,
        resident=resident,
        resident_bytes_for=resident_bytes_for,
        policy=policy,
        cached_analysis_for=cached_analysis_for,
        covered_axes=covered_axes,
    )
    return search.run(hint=hint)


@dataclass(frozen=True, slots=True, kw_only=True)
class _Evaluation[Compiled]:
    """One width mapping the bounded search compiled or read a profile for."""

    widths: MappingProxyType[str, int]
    """The mapping evaluated, in axis declaration order."""
    reservation_bytes: int
    """Represented compiler requirement the report carried for this mapping."""
    resident_bytes: int
    """Bytes the plan keeps resident at the node's position for this mapping."""
    peak_bytes: int
    """Raw compiler peak the report carried for this mapping."""
    compiled: Compiled | None
    """The executable, or `None` for a mapping a cached profile refused."""

    @property
    def admitted(self) -> bool:
        """Report whether the budget admitted this mapping."""
        return self.compiled is not None


class _CachedReport:
    """A compiler report the caller already holds, shaped like an executable."""

    def __init__(self, *, analysis: CompilerMemoryReport) -> None:
        self.analysis = analysis

    def memory_analysis(self) -> CompilerMemoryReport:
        """Return the report the caller handed the planner."""
        return self.analysis


@dataclass(frozen=True, slots=True, kw_only=True)
class _BoundedWidthSearch[Compiled]:
    """The seed, shrink and refine walk of one core's width declarations."""

    axes: tuple[ReducedAxis | TiledOutputAxis, ...]
    fixed_widths: Mapping[str, int]
    width_ceilings: Mapping[str, int]
    compile_candidate: Callable[[Mapping[str, int]], Compiled]
    budget: int
    memory_for: Callable[[Compiled], CompilerMemoryReservation] | None
    resident: int
    resident_bytes_for: Callable[[Compiled], int] | None
    policy: WidthSearchPolicy
    cached_analysis_for: (
        Callable[[Mapping[str, int]], CompilerMemoryReport | None] | None
    )
    covered_axes: Collection[str]

    def run(self, *, hint: Mapping[str, int] | None) -> WorkspacePlan[Compiled]:
        """Walk the policy and return its widest admitted candidate."""
        selector = BoundedWidthSelector(
            axes=self.axes,
            fixed_widths=self.fixed_widths,
            width_ceilings=self.width_ceilings,
            policy=self.policy,
            hint=hint,
            covered_axes=self.covered_axes,
        )
        compiled_by_widths: dict[tuple[tuple[str, int], ...], Compiled] = {}
        while (widths := selector.propose()) is not None:
            evaluation = self._evaluate(widths=widths)
            if evaluation.compiled is not None:
                compiled_by_widths[_widths_key(widths=widths)] = evaluation.compiled
            selector.record(
                widths=widths,
                reservation_bytes=evaluation.reservation_bytes,
                resident_bytes=evaluation.resident_bytes,
                peak_bytes=evaluation.peak_bytes,
                admitted=evaluation.admitted,
            )
        best = selector.selected
        if best is None:
            raise ExecutionPlanningError(
                selector.exhaustion_message(budget=self.budget)
            )
        return WorkspacePlan(
            widths=best.widths,
            peak_bytes=best.peak_bytes,
            reservation_bytes=best.reservation_bytes,
            compiled=compiled_by_widths[_widths_key(widths=best.widths)],
        )

    def _evaluate(self, *, widths: MappingProxyType[str, int]) -> _Evaluation[Compiled]:
        """Read one mapping's accounting, compiling it unless a profile refuses it."""
        cached = (
            None
            if self.cached_analysis_for is None
            else self.cached_analysis_for(widths)
        )
        if cached is not None:
            memory = compiler_memory_reservation(
                compiled=_CachedReport(analysis=cached), widths=widths
            )
            if memory.reservation_bytes + self.resident > self.budget:
                return _Evaluation(
                    widths=widths,
                    reservation_bytes=memory.reservation_bytes,
                    resident_bytes=self.resident,
                    peak_bytes=memory.peak_bytes,
                    compiled=None,
                )
        compiled = self.compile_candidate(widths)
        memory = _memory_for_candidate(
            compiled=compiled, widths=widths, memory_for=self.memory_for
        )
        candidate_resident = _resident_bytes_for_candidate(
            compiled=compiled,
            widths=widths,
            lower_bound=self.resident,
            resident_bytes_for=self.resident_bytes_for,
        )
        fits = memory.reservation_bytes + candidate_resident <= self.budget
        return _Evaluation(
            widths=widths,
            reservation_bytes=memory.reservation_bytes,
            resident_bytes=candidate_resident,
            peak_bytes=memory.peak_bytes,
            compiled=compiled if fits else None,
        )
