"""Compile-only selection of memory-feasible workspace widths.

The planner owns one narrow seam: callers describe reduced and tiled axes and
provide a compiler for a concrete width mapping.  This module enumerates the static
frontier in rank order, inspects compiler memory reports without executing a
candidate, and returns the first feasible candidate — the already-compiled winner —
for dispatch.
"""

import itertools
import logging
import math
import operator
from collections.abc import Callable, Collection, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Protocol, SupportsIndex, cast

from _lcm.execution.core_program import ReducedAxis, TiledOutputAxis
from lcm.exceptions import ExecutionPlanningError
from lcm.execution import WidthSearch, WidthSearchPolicy

_logger = logging.getLogger("lcm")

# The policy a caller who declares none plans under: today's ranked walk.
_EXHAUSTIVE_POLICY = WidthSearchPolicy()

_MISSING = object()

# The single candidate of a program that declares no width axis.
_NO_WIDTHS: MappingProxyType[str, int] = MappingProxyType({})

# Largest width an unbudgeted reduced axis is lowered at, and the floor no
# unbudgeted axis is lowered below.
BOOTSTRAP_WIDTH_CAP = 64

# Largest width an unbudgeted tiled output axis is lowered at.  A tiled axis
# concatenates its tiles into a result resident at the full extent whatever the
# width, so only its temporaries grow with it, while a reduced axis block is
# pure temporary.
BOOTSTRAP_TILE_WIDTH_CAP = 1024

# Largest product of unbudgeted widths a candidate aims for, so the live block
# stays bounded by a fixed number of cells on every backend.
BOOTSTRAP_BLOCK_CAP = BOOTSTRAP_WIDTH_CAP * BOOTSTRAP_TILE_WIDTH_CAP


class _MemoryAnalyzable(Protocol):
    """Compiler result exposing JAX-style memory analysis."""

    def memory_analysis(self) -> object:
        """Return compiler workspace statistics."""
        ...


@dataclass(frozen=True, slots=True, kw_only=True)
class CompilerMemoryRecord:
    """One device's raw peak and represented default-memory allocations."""

    peak_bytes: int
    """Unmodified compiler-reported peak for this device record."""
    argument_bytes: int
    """Storage assigned to the executable's input arguments."""
    output_bytes: int
    """Storage assigned to executable outputs, before subtracting aliases."""
    alias_bytes: int
    """Argument/output overlap to subtract once from represented storage."""
    temporary_bytes: int
    """Compiler-reported preallocated temporary storage."""

    def __post_init__(self) -> None:
        """Require complete counters and an argument/output overlap that can exist."""
        for value in (
            self.peak_bytes,
            self.argument_bytes,
            self.output_bytes,
            self.alias_bytes,
            self.temporary_bytes,
        ):
            _non_negative_bytes(value=value)
        if self.alias_bytes > min(self.argument_bytes, self.output_bytes):
            raise ValueError(
                "Compiler aliases exceed represented arguments or outputs."
            )

    @property
    def allocation_bytes(self) -> int:
        """Count represented storage, removing argument/output overlap once."""
        return (
            self.argument_bytes
            + self.output_bytes
            - self.alias_bytes
            + self.temporary_bytes
        )

    @property
    def reservation_bytes(self) -> int:
        """Enforce both reported requirements within the represented storage scope."""
        return max(self.peak_bytes, self.allocation_bytes)


@dataclass(frozen=True, slots=True, kw_only=True)
class CompilerMemoryReservation:
    """Complete per-device records, with raw peak kept separate from reservation.

    Allocation counters represent arguments, outputs, their aliases, and
    preallocated temporaries. This scope excludes generated code, allocator
    overhead, thread stacks, and other runtime storage omitted by the report.
    """

    records: tuple[CompilerMemoryRecord, ...]
    """Nonempty device records whose allocation fields remain paired."""

    def __post_init__(self) -> None:
        """Refuse an empty device report."""
        if not self.records:
            raise ValueError("Compiler memory reservation needs a device record.")

    @property
    def peak_bytes(self) -> int:
        """Return the largest raw compiler peak across devices."""
        return max(record.peak_bytes for record in self.records)

    @property
    def reservation_bytes(self) -> int:
        """Return the largest reservation after accounting within each device."""
        return max(record.reservation_bytes for record in self.records)


@dataclass(frozen=True, slots=True)
class WorkspacePlan[Compiled]:
    """One selected width mapping and its already-compiled executable."""

    widths: Mapping[str, int]
    peak_bytes: int | None
    compiled: Compiled
    reservation_bytes: int | None = None
    """Selected represented compiler requirement, or None without a budget."""

    def __post_init__(self) -> None:
        """Own an immutable snapshot of the planner-selected widths."""
        object.__setattr__(self, "widths", MappingProxyType(dict(self.widths)))


def workspace_width_candidates(
    *,
    axes: tuple[ReducedAxis | TiledOutputAxis, ...],
    fixed_widths: Mapping[str, int] = MappingProxyType({}),
    budget_bytes: int | None = None,
    width_ceilings: Mapping[str, int] = MappingProxyType({}),
    covered_axes: Collection[str] = (),
) -> tuple[Mapping[str, int], ...]:
    """Return the candidate sequence in planner rank order without compiling it.

    Without a budget the sequence holds one candidate: each axis at its fixed width
    when `fixed_widths` names it, else at its bootstrap width (see
    `bootstrap_widths`).  With a budget it holds the Cartesian product of the
    per-axis frontiers, widest first: descending width product, ties broken toward
    the lexicographically greatest width tuple in axis declaration order.  A fixed
    axis contributes one width.  Every width an axis contributes satisfies the
    width policy it declares: it is the full extent, or a multiple of its
    `alignment` at or above its `minimum_width`.  Names no axis declares are ignored
    here; a name no program of the solve declares is refused before planning starts.

    `width_ceilings` is an opt-in upper bound per axis name. It intersects the
    legal candidates with `width <= ceiling`; it shortens no axis extent,
    overrides no fixed width policy, and a ceiling below an axis's narrowest
    legal width is refused.

    `covered_axes` names the axes whose bootstrap width is their whole extent
    where `bootstrap_widths` covers them; a budgeted frontier already holds it.
    """
    declared_axes = _validate_axes(axes=axes)
    widths = _validate_fixed_widths(fixed_widths=fixed_widths)
    ceilings = _validate_width_ceilings(width_ceilings=width_ceilings)
    budget = _validate_budget(budget_bytes=budget_bytes)
    return _workspace_width_candidates(
        axes=declared_axes,
        fixed_widths=widths,
        budget_bytes=budget,
        width_ceilings=ceilings,
        covered_axes=covered_axes,
    )


def plan_workspace[Compiled](
    *,
    axes: tuple[ReducedAxis | TiledOutputAxis, ...],
    fixed_widths: Mapping[str, int] = MappingProxyType({}),
    width_ceilings: Mapping[str, int] = MappingProxyType({}),
    compile_candidate: Callable[[Mapping[str, int]], Compiled],
    budget_bytes: int | None = None,
    memory_for: Callable[[Compiled], CompilerMemoryReservation] | None = None,
    resident_bytes: int = 0,
    resident_bytes_for: Callable[[Compiled], int] | None = None,
    covered_axes: Collection[str] = (),
) -> WorkspacePlan[Compiled]:
    """Compile the width frontier widest-first and return the first candidate that fits.

    Without a budget, the bootstrap width (or a fixed axis width) is compiled
    exactly once and compiler memory analysis is deliberately not consulted, so
    `resident_bytes` is not consulted either.  With a budget, candidates are
    compiled and analyzed in rank order — descending width product, ties broken
    toward the lexicographically greatest width tuple in axis declaration order —
    and the first whose compiler reservation fits is returned.
    A candidate is feasible when its represented allocation reservation plus the
    bytes the plan keeps resident at the node's scheduled position fits the budget.
    The reservation enforces both raw peak and represented allocation storage;
    storage omitted from those reports remains outside this accounting scope.
    That is the feasible maximum of the whole frontier, reached without compiling any
    candidate narrower than the winner; only a core that fits at no width compiles
    its entire frontier before failing.  A position whose resident bytes already
    reach the budget is refused before any candidate is compiled, since no width
    could serve it.

    ``resident_bytes_for`` may refine that lower bound for each executable, for
    example when compilation removes an argument whose owner stays live. It
    returns total external residency, never a replacement compiler peak. This
    lookup belongs to the current invocation, not to an executable cache.

    The returned executable is the exact object compiled for the selected candidate;
    the planner neither executes it nor recompiles the winner.
    """
    declared_axes = _validate_axes(axes=axes)
    widths_by_axis = _validate_fixed_widths(fixed_widths=fixed_widths)
    ceilings = _validate_width_ceilings(width_ceilings=width_ceilings)
    budget = _validate_budget(budget_bytes=budget_bytes)
    resident = _validate_resident_bytes(resident_bytes=resident_bytes)
    if not callable(compile_candidate):
        msg = "The workspace candidate compiler must be callable."
        raise TypeError(msg)
    if memory_for is not None and not callable(memory_for):
        msg = "The workspace memory lookup must be callable or None."
        raise TypeError(msg)
    if resident_bytes_for is not None and not callable(resident_bytes_for):
        raise TypeError("The workspace residency lookup must be callable or None.")

    candidates = _workspace_width_candidates(
        axes=declared_axes,
        fixed_widths=widths_by_axis,
        budget_bytes=budget,
        width_ceilings=ceilings,
        covered_axes=covered_axes,
    )

    if budget is None:
        widths = candidates[0]
        compiled = compile_candidate(widths)
        return WorkspacePlan(widths=widths, peak_bytes=None, compiled=compiled)

    if resident >= budget:
        raise ExecutionPlanningError(
            _resident_exhausts_budget_message(resident=resident, budget=budget)
        )

    least_total: int | None = None
    least_peak: int | None = None
    least_reservation: int | None = None
    least_resident = resident
    for widths in candidates:
        compiled = compile_candidate(widths)
        memory = _memory_for_candidate(
            compiled=compiled, widths=widths, memory_for=memory_for
        )
        candidate_resident = _resident_bytes_for_candidate(
            compiled=compiled,
            widths=widths,
            lower_bound=resident,
            resident_bytes_for=resident_bytes_for,
        )
        total = memory.reservation_bytes + candidate_resident
        if least_total is None or total < least_total:
            least_total = total
            least_peak = memory.peak_bytes
            least_reservation = memory.reservation_bytes
            least_resident = candidate_resident
        if total <= budget:
            return WorkspacePlan(
                widths=widths,
                peak_bytes=memory.peak_bytes,
                reservation_bytes=memory.reservation_bytes,
                compiled=compiled,
            )

    if declared_axes and all(axis.name in widths_by_axis for axis in declared_axes):
        msg = (
            "The explicitly requested workspace widths require "
            f"{least_reservation} reservation bytes (raw compiler peak {least_peak}), "
            f"exceeding the {budget}-byte budget "
            f"with {least_resident} resident bytes at the node's position."
        )
    else:
        msg = _no_candidate_fits_message(
            budget=budget,
            total=least_total,
            reservation=least_reservation,
            resident=least_resident,
            peak=least_peak,
        )
    raise ExecutionPlanningError(msg)


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
    cached_analysis_for: Callable[[Mapping[str, int]], object | None] | None = None,
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
        raise ExecutionPlanningError(
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

    def __init__(self, *, analysis: object) -> None:
        self.analysis = analysis

    def memory_analysis(self) -> object:
        """Return the report the caller handed the planner."""
        return self.analysis


@dataclass(frozen=True, slots=True, kw_only=True)
class WidthDecision:
    """One width mapping's accounting and the admission decision it earned."""

    widths: MappingProxyType[str, int]
    """The mapping evaluated, in axis declaration order."""
    reservation_bytes: int
    """Represented compiler requirement the report carried for this mapping."""
    resident_bytes: int
    """Bytes the plan keeps resident at the node's position for this mapping."""
    peak_bytes: int
    """Raw compiler peak the report carried for this mapping."""
    admitted: bool
    """Whether the budget admitted this mapping."""


@dataclass(kw_only=True)
class BoundedWidthSelector:
    """Propose one core's next width, decision by decision.

    The selector owns the bounded policy and nothing else: it never compiles,
    never reads a compiler report and never consults a budget. A caller asks it
    for the next width mapping to evaluate, evaluates that mapping however its
    own admission arithmetic requires, and hands back the reservation, the
    residency and the verdict. The walk is the policy's:

    - **seed** — a compatible `hint`, else the widest mapping under
      `seed="widest"`, else the conservative bootstrap anchor.
    - **shrink** — on refusal, halve the unfixed axis declaring the largest
      extent, ties going to declaration order, rounded down onto the widths that
      axis admits. An axis already at its narrowest is passed over, and the
      search is exhausted once every unfixed axis is.
    - **refine** — on the first admission, widen one axis at a time, most
      recently shrunk first, by bisecting between the admitted width and the
      last width refused on that axis.
    - **stop** — at `max_evaluations` decisions, at `refinement_share`
      refinement decisions, or when refinement cannot widen.

    Memory is treated as nonmonotone in the widths throughout: a refusal at one
    width says nothing about its neighbours, which is why every width the
    selector keeps was evaluated rather than inferred.
    """

    axes: tuple[ReducedAxis | TiledOutputAxis, ...]
    """The core's declared axes, in declaration order."""
    fixed_widths: Mapping[str, int]
    """Widths the caller pinned; a pinned axis is never proposed at another width."""
    width_ceilings: Mapping[str, int] = MappingProxyType({})
    """Upper bound per axis name; no proposal exceeds the ceiling an axis carries."""
    policy: WidthSearchPolicy
    """Which search to run and the evaluation budget it runs under."""
    hint: Mapping[str, int] | None = None
    """A mapping to evaluate first; a declaration that refuses it logs and skips it."""
    label: str = ""
    """What the diagnostics call this core."""
    covered_axes: Collection[str] = ()
    """Axes whose conservative seed is their whole extent where it is covered."""

    def __post_init__(self) -> None:
        """Announce the policy and compute the seed the first proposal carries."""
        self._declared = {axis.name: axis for axis in self.axes}
        self._width_ceiling_by_axis = {
            axis.name: _ceiling_for(axis=axis, width_ceilings=self.width_ceilings)
            for axis in self.axes
        }
        self._decisions: list[WidthDecision] = []
        self._seen: set[tuple[tuple[str, int], ...]] = set()
        self._last_refused: dict[str, int] = {}
        self._shrunk: list[str] = []
        self._best: WidthDecision | None = None
        self._refining = False
        self._refine_queue: list[str] = []
        self._ceiling: int | None = None
        self._spent = 0
        self._pending: MappingProxyType[str, int] | None = self._seed()
        _logger.info(
            "bounded width search %s: policy %s, seed rule %r, at most %d "
            "evaluations of which %d refine; first widths %r",
            self.label,
            self.policy.kind.value,
            self.policy.seed,
            self.policy.max_evaluations,
            self.policy.refinement_share,
            dict(self._pending),
        )

    @property
    def decisions(self) -> tuple[WidthDecision, ...]:
        """Return every decision recorded so far, in decision order."""
        return tuple(self._decisions)

    @property
    def selected(self) -> WidthDecision | None:
        """Return the widest admitted mapping, or `None` while none was admitted."""
        return self._best

    @property
    def refinement_decisions(self) -> int:
        """Return how many decisions were spent widening after the first admission."""
        return self._spent

    def propose(self) -> MappingProxyType[str, int] | None:
        """Return the next mapping to evaluate, or `None` when the search is done."""
        return self._pending

    def record(
        self,
        *,
        widths: Mapping[str, int],
        reservation_bytes: int,
        resident_bytes: int,
        peak_bytes: int,
        admitted: bool,
    ) -> None:
        """Take one mapping's verdict and compute what the search proposes next.

        Args:
            widths: The mapping that was evaluated, as `propose` handed it over.
            reservation_bytes: Represented compiler requirement of that mapping.
            resident_bytes: Bytes the plan keeps resident for that mapping.
            peak_bytes: Raw compiler peak of that mapping.
            admitted: Whether the caller's admission arithmetic kept it.

        Raises:
            ExecutionPlanningError: A shrink proposed a mapping already evaluated.

        """
        frozen = _width_mapping(
            axes=self.axes, values=tuple(widths[axis.name] for axis in self.axes)
        )
        decision = WidthDecision(
            widths=frozen,
            reservation_bytes=reservation_bytes,
            resident_bytes=resident_bytes,
            peak_bytes=peak_bytes,
            admitted=admitted,
        )
        self._decisions.append(decision)
        self._seen.add(_widths_key(widths=frozen))
        _logger.info(
            "bounded width search %s: evaluation %d at %r — %d reservation plus %d "
            "resident bytes, %s",
            self.label,
            len(self._decisions),
            dict(frozen),
            reservation_bytes,
            resident_bytes,
            "admitted" if admitted else "refused",
        )
        if self._refining:
            self._spent += 1
            if admitted:
                self._best = decision
            elif self._refine_queue:
                self._ceiling = frozen[self._refine_queue[0]]
            self._pending = self._next_refinement()
            return
        if admitted:
            self._best = decision
            self._refining = True
            self._refine_queue = [
                name
                for name in _most_recent_first(names=self._shrunk)
                if name in self._last_refused
            ]
            self._ceiling = (
                self._last_refused[self._refine_queue[0]]
                if self._refine_queue
                else None
            )
            self._pending = self._next_refinement()
            return
        self._pending = self._shrink_from(widths=frozen)

    def exhaustion_message(self, *, budget: int) -> str:
        """State that the search, not the model, ran out of candidates."""
        return _search_exhausted_message(
            budget=budget, policy=self.policy, decisions=self._decisions
        )

    def _admissible(self, *, axis: ReducedAxis | TiledOutputAxis, width: int) -> int:
        """Return the width one axis admits, under its pin and its ceiling.

        Args:
            axis: The axis a width is being proposed for.
            width: The proposal, before the axis's own policy narrows it.

        Returns:
            The pinned width of a pinned axis, else the proposal rounded onto
            the widths the axis admits; neither exceeds a declared ceiling.

        """
        ceiling = self._width_ceiling_by_axis[axis.name]
        if axis.name in self.fixed_widths:
            return _fixed_width(
                axis=axis, fixed_widths=self.fixed_widths, ceiling=ceiling
            )
        return _admissible_width(axis=axis, width=width, ceiling=ceiling)

    def _shrink_from(
        self, *, widths: MappingProxyType[str, int]
    ) -> MappingProxyType[str, int] | None:
        """Record the refusal on one axis and return the narrower mapping."""
        shrunk = self._shrink(widths=widths)
        if shrunk is None:
            return None
        name, narrower = shrunk
        self._last_refused[name] = widths[name]
        self._shrunk.append(name)
        if len(self._decisions) >= self.policy.max_evaluations:
            return None
        if _widths_key(widths=narrower) in self._seen:
            msg = (
                "The bounded width search proposed the already evaluated widths "
                f"{dict(narrower)!r} a second time. Every shrink narrows one axis "
                "and leaves the rest, so a repeat cannot arise."
            )
            raise ExecutionPlanningError(msg)
        return narrower

    def _seed(self) -> MappingProxyType[str, int]:
        """Return the first mapping to evaluate under the policy's seed rule."""
        if self.hint is not None:
            reason = self._hint_refusal(hint=self.hint)
            if reason is None:
                return _width_mapping(
                    axes=self.axes,
                    values=tuple(self.hint[axis.name] for axis in self.axes),
                )
            _logger.warning("hint incompatible: %s", reason)
        if self.policy.seed == "widest":
            return _width_mapping(
                axes=self.axes,
                values=tuple(
                    self._admissible(axis=axis, width=axis.extent) for axis in self.axes
                ),
            )
        return bootstrap_widths(
            axes=self.axes,
            fixed_widths=self.fixed_widths,
            width_ceilings=self.width_ceilings,
            covered_axes=self.covered_axes,
        )

    def _hint_refusal(self, *, hint: Mapping[str, int]) -> str | None:
        """Name what makes a hint unusable, or `None` when every axis admits it."""
        undeclared = tuple(sorted(set(hint) - set(self._declared)))
        if undeclared:
            return f"the core declares no axis named {undeclared[0]!r}."
        missing = tuple(axis.name for axis in self.axes if axis.name not in hint)
        if missing:
            return f"it names no width for axis {missing[0]!r}."
        for axis in self.axes:
            width = hint[axis.name]
            if axis.name in self.fixed_widths:
                fixed = self._admissible(axis=axis, width=axis.extent)
                if width != fixed:
                    return (
                        f"axis {axis.name!r} is fixed at {fixed} and the hint asks "
                        f"for {width}."
                    )
            elif self._admissible(axis=axis, width=width) != width:
                return (
                    f"axis {axis.name!r} admits no width {width} under alignment "
                    f"{axis.alignment}, floor {axis.minimum_width} and extent "
                    f"{axis.extent}."
                )
        return None

    def _shrink(
        self, *, widths: Mapping[str, int]
    ) -> tuple[str, MappingProxyType[str, int]] | None:
        """Halve the widest-extent unfixed axis that is not already at its floor.

        A covered axis refused at its whole extent steps to the power-of-two
        width `bootstrap_width` gives it before halving.
        """
        movable = sorted(
            (axis for axis in self.axes if axis.name not in self.fixed_widths),
            key=lambda axis: -axis.extent,
        )
        for axis in movable:
            current = widths[axis.name]
            proposal = (
                bootstrap_width(extent=axis.extent)
                if current == axis.extent
                and _covers(axis=axis, covered_axes=self.covered_axes)
                else current // 2
            )
            narrower = self._admissible(axis=axis, width=proposal)
            if narrower < current:
                return axis.name, _width_mapping(
                    axes=self.axes,
                    values=tuple(
                        narrower if other.name == axis.name else widths[other.name]
                        for other in self.axes
                    ),
                )
        return None

    def _next_refinement(self) -> MappingProxyType[str, int] | None:
        """Return the next widening proposal, or `None` when none can widen."""
        best = cast("WidthDecision", self._best)
        while self._refine_queue:
            if (
                self._spent >= self.policy.refinement_share
                or len(self._decisions) >= self.policy.max_evaluations
            ):
                return None
            name = self._refine_queue[0]
            floor = best.widths[name]
            ceiling = cast("int", self._ceiling)
            proposal = (
                0
                if ceiling <= floor
                else self._admissible(
                    axis=self._declared[name], width=(floor + ceiling) // 2
                )
            )
            if proposal > floor:
                widths = _width_mapping(
                    axes=self.axes,
                    values=tuple(
                        proposal if axis.name == name else best.widths[axis.name]
                        for axis in self.axes
                    ),
                )
                if _widths_key(widths=widths) not in self._seen:
                    return widths
            self._refine_queue.pop(0)
            self._ceiling = (
                self._last_refused[self._refine_queue[0]]
                if self._refine_queue
                else None
            )
        return None


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
    cached_analysis_for: Callable[[Mapping[str, int]], object | None] | None
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


def _most_recent_first(*, names: list[str]) -> tuple[str, ...]:
    """Return each axis name once, the most recently shrunk one first."""
    ordered: list[str] = []
    for name in reversed(names):
        if name not in ordered:
            ordered.append(name)
    return tuple(ordered)


def _widths_key(*, widths: Mapping[str, int]) -> tuple[tuple[str, int], ...]:
    """Return the hashable identity of one width mapping."""
    return tuple(sorted(widths.items()))


def _search_exhausted_message(
    *, budget: int, policy: WidthSearchPolicy, decisions: list[WidthDecision]
) -> str:
    """State that the search, not the model, ran out of candidates."""
    trail = "; ".join(
        f"{dict(item.widths)!r}: {item.reservation_bytes} reservation plus "
        f"{item.resident_bytes} resident bytes, "
        f"{'admitted' if item.admitted else 'refused'}"
        for item in decisions
    )
    return (
        "The bounded width search found no admitted candidate within the evaluation "
        f"budget of {policy.max_evaluations} against the {budget}-byte budget. "
        f"Seed rule {policy.seed!r}, refinement share {policy.refinement_share}. "
        f"Evaluations: {trail}. Those widths are not an exhaustive test of the "
        "frontier; a width the search never proposed may still fit."
    )


def plan_axis_free_workspace[Compiled](
    *,
    compile_candidate: Callable[[], Compiled],
    memory_for: Callable[[Compiled], CompilerMemoryReservation],
    budget_bytes: int,
    resident_bytes: int,
) -> WorkspacePlan[Compiled]:
    """Admit the single candidate of a program that declares no width axis.

    A pure host operation has one shape and therefore one workspace candidate, so
    `plan_workspace`'s frontier is the one-element map `{}`. This entry point is
    that specialization and nothing else: it applies the identical feasibility
    test — the candidate's represented reservation plus the bytes the caller keeps
    resident at the node's position must fit the budget — refuses an already
    exhausted position *before* compiling anything, and raises the same
    `ExecutionPlanningError` messages, built by the same message functions. It
    exists so a per-call dispatch need not re-validate an empty axis tuple or
    re-enumerate a one-element frontier; it grants no allocation `plan_workspace`
    would refuse and refuses none it would grant.
    """
    budget = _validate_budget(budget_bytes=budget_bytes)
    if budget is None:
        raise TypeError("An axis-free workspace plan requires an explicit budget.")
    resident = _validate_resident_bytes(resident_bytes=resident_bytes)
    if resident >= budget:
        raise ExecutionPlanningError(
            _resident_exhausts_budget_message(resident=resident, budget=budget)
        )
    compiled = compile_candidate()
    memory = _memory_for_candidate(
        compiled=compiled, widths=_NO_WIDTHS, memory_for=memory_for
    )
    total = memory.reservation_bytes + resident
    if total > budget:
        raise ExecutionPlanningError(
            _no_candidate_fits_message(
                budget=budget,
                total=total,
                reservation=memory.reservation_bytes,
                resident=resident,
                peak=memory.peak_bytes,
            )
        )
    return WorkspacePlan(
        widths=_NO_WIDTHS,
        peak_bytes=memory.peak_bytes,
        reservation_bytes=memory.reservation_bytes,
        compiled=compiled,
    )


def _resident_exhausts_budget_message(*, resident: int, budget: int) -> str:
    """State that a node's own residency leaves no workspace at any width."""
    return (
        f"The plan keeps {resident} bytes resident at the node's position, "
        f"leaving nothing of the {budget}-byte budget for a workspace."
    )


def _no_candidate_fits_message(
    *,
    budget: int,
    total: int | None,
    reservation: int | None,
    resident: int,
    peak: int | None,
) -> str:
    """State the cheapest total the frontier could offer against the budget."""
    return (
        "No workspace-width candidate fits the "
        f"{budget}-byte budget; the smallest total is {total} bytes "
        f"({reservation} compiler reservation plus {resident} resident "
        f"bytes; raw compiler peak {peak}, "
        "at the node's position)."
    )


def _validate_axes(
    *, axes: tuple[ReducedAxis | TiledOutputAxis, ...]
) -> tuple[ReducedAxis | TiledOutputAxis, ...]:
    """Validate planner-local width assumptions and preserve declaration order."""
    declared_axes = tuple(axes)
    for axis in declared_axes:
        if not isinstance(axis, (ReducedAxis, TiledOutputAxis)):
            msg = "Workspace axes must be reduced or tiled axis instances."
            raise TypeError(msg)
        _validate_axis(axis=axis)

    names = tuple(axis.name for axis in declared_axes)
    if len(names) != len(set(names)):
        msg = f"Workspace axes have duplicate names: {names!r}."
        raise ValueError(msg)
    return declared_axes


def _validate_axis(*, axis: ReducedAxis | TiledOutputAxis) -> None:
    """Validate planner-local assumptions about one axis."""
    if not isinstance(axis.name, str) or not axis.name:
        msg = "A workspace axis name must be a non-empty string."
        raise TypeError(msg)
    if isinstance(axis, ReducedAxis):
        _validate_coordinates(axis=axis)
    if axis.extent <= 1:
        msg = f"Workspace axis {axis.name!r} must have product extent greater than one."
        raise ValueError(msg)


def _validate_coordinates(*, axis: ReducedAxis) -> None:
    """Validate the coordinate product one reduced axis enumerates."""
    if len(axis.coordinate_names) != len(axis.coordinate_extents):
        msg = (
            f"Workspace axis {axis.name!r} coordinate names and extents must "
            "have the same length."
        )
        raise ValueError(msg)
    if not axis.coordinate_extents:
        msg = f"Workspace axis {axis.name!r} must declare coordinate extents."
        raise ValueError(msg)
    if any(
        isinstance(extent, bool) or not isinstance(extent, int)
        for extent in axis.coordinate_extents
    ):
        msg = f"Workspace axis {axis.name!r} extents must be integers."
        raise TypeError(msg)
    if any(extent <= 0 for extent in axis.coordinate_extents):
        msg = f"Workspace axis {axis.name!r} extents must be positive."
        raise ValueError(msg)


def _validate_fixed_widths(*, fixed_widths: Mapping[str, int]) -> Mapping[str, int]:
    """Require exact positive widths keyed by non-empty axis names."""
    widths = dict(fixed_widths)
    for name, width in widths.items():
        if type(name) is not str or not name:
            msg = "A fixed workspace width must be keyed by a non-empty axis name."
            raise TypeError(msg)
        if type(width) is not int:
            msg = f"Fixed width for workspace axis {name!r} must be an integer."
            raise TypeError(msg)
        if width <= 0:
            msg = f"Fixed width for workspace axis {name!r} must be positive."
            raise ValueError(msg)
    return MappingProxyType(widths)


def _validate_width_ceilings(*, width_ceilings: Mapping[str, int]) -> Mapping[str, int]:
    """Require exact positive ceilings keyed by non-empty axis names."""
    ceilings = dict(width_ceilings)
    for name, ceiling in ceilings.items():
        if type(name) is not str or not name:
            msg = "A workspace width ceiling must be keyed by a non-empty axis name."
            raise TypeError(msg)
        if type(ceiling) is not int:
            msg = f"Width ceiling for workspace axis {name!r} must be an integer."
            raise TypeError(msg)
        if ceiling <= 0:
            msg = f"Width ceiling for workspace axis {name!r} must be positive."
            raise ValueError(msg)
    return MappingProxyType(ceilings)


def _validate_budget(*, budget_bytes: int | None) -> int | None:
    """Require a positive exact-integer byte budget when one is supplied."""
    if budget_bytes is None:
        return None
    if type(budget_bytes) is not int:
        msg = "The workspace budget must be an integer number of bytes or None."
        raise TypeError(msg)
    if budget_bytes <= 0:
        msg = "The workspace budget must be positive."
        raise ValueError(msg)
    return budget_bytes


def _validate_resident_bytes(*, resident_bytes: int) -> int:
    """Require an exact non-negative count of bytes resident at the node."""
    if type(resident_bytes) is not int:
        msg = "The resident byte count must be an exact int."
        raise TypeError(msg)
    if resident_bytes < 0:
        msg = "The resident byte count cannot be negative."
        raise ValueError(msg)
    return resident_bytes


def bootstrap_width(*, extent: int, cap: int = BOOTSTRAP_WIDTH_CAP) -> int:
    """Return the width an axis streams at when no device-memory budget is declared.

    The width is the largest power of two strictly below the extent, capped at
    `cap` — `BOOTSTRAP_WIDTH_CAP` for a reduced axis, so an unbudgeted solve never
    lowers a whole action product, and a cap `bootstrap_widths` derives for a tiled
    output axis.  The full extent is reached only through a budget that shows it
    fits or through a fixed width.
    """
    if type(extent) is not int or extent <= 1:
        msg = f"An execution axis needs an exact int extent above one, got {extent!r}."
        raise ValueError(msg)
    if type(cap) is not int or cap < 1:
        msg = f"A bootstrap width cap must be an exact int above zero, got {cap!r}."
        raise ValueError(msg)
    upper_bound = min(cap, extent - 1)
    return 1 << (upper_bound.bit_length() - 1)


def _covers(
    *, axis: ReducedAxis | TiledOutputAxis, covered_axes: Collection[str]
) -> bool:
    """Report whether a named axis is seeded at its whole extent.

    It is when the extent is within `BOOTSTRAP_WIDTH_CAP` and the power-of-two
    `bootstrap_width` leaves a remainder of it, which a map at that width would
    trace as a second program.
    """
    return (
        axis.name in covered_axes
        and axis.extent <= BOOTSTRAP_WIDTH_CAP
        and axis.extent % bootstrap_width(extent=axis.extent) != 0
    )


def _tiled_bootstrap_cap(*, block: int) -> int:
    """Return the cap a tiled output axis bootstraps under, given the live block.

    The cap spends what `BOOTSTRAP_BLOCK_CAP` leaves after the widths already
    fixed for this candidate — reduced axes are declared first, so their blocks
    are counted before any tile widens — never exceeding
    `BOOTSTRAP_TILE_WIDTH_CAP` and never falling below `BOOTSTRAP_WIDTH_CAP`, so
    no axis is lowered narrower than the reduced rule alone would have lowered it.
    """
    return min(
        max(BOOTSTRAP_BLOCK_CAP // block, BOOTSTRAP_WIDTH_CAP),
        BOOTSTRAP_TILE_WIDTH_CAP,
    )


def bootstrap_widths(
    *,
    axes: tuple[ReducedAxis | TiledOutputAxis, ...],
    fixed_widths: Mapping[str, int] = MappingProxyType({}),
    width_ceilings: Mapping[str, int] = MappingProxyType({}),
    covered_axes: Collection[str] = (),
) -> MappingProxyType[str, int]:
    """Return the one width map an unbudgeted plan lowers, in declaration order.

    A named axis takes its fixed width.  A reduced axis takes `bootstrap_width`
    under `BOOTSTRAP_WIDTH_CAP`.  A tiled output axis takes it under the cap
    `_tiled_bootstrap_cap` derives from the widths already fixed, so the product of
    the map stays near `BOOTSTRAP_BLOCK_CAP` and the live block stays bounded by a
    fixed number of cells whatever the model.  Every width is the one the axis
    admits nearest the proposal, so alignment and `minimum_width` still hold.

    An unfixed axis `covered_axes` names takes its whole extent instead where
    `_covers` holds, so the map over it traces no remainder program; a ceiling
    below the extent still bounds it.
    """
    widths: dict[str, int] = {}
    block = 1
    for axis in axes:
        ceiling = _ceiling_for(axis=axis, width_ceilings=width_ceilings)
        if axis.name in fixed_widths:
            width = _fixed_width(axis=axis, fixed_widths=fixed_widths, ceiling=ceiling)
        else:
            cap = (
                _tiled_bootstrap_cap(block=block)
                if isinstance(axis, TiledOutputAxis)
                else BOOTSTRAP_WIDTH_CAP
            )
            width = _admissible_width(
                axis=axis,
                width=(
                    axis.extent
                    if _covers(axis=axis, covered_axes=covered_axes)
                    else bootstrap_width(extent=axis.extent, cap=cap)
                ),
                ceiling=ceiling,
            )
        widths[axis.name] = width
        block *= width
    return MappingProxyType(widths)


def _workspace_width_candidates(
    *,
    axes: tuple[ReducedAxis | TiledOutputAxis, ...],
    fixed_widths: Mapping[str, int],
    budget_bytes: int | None,
    width_ceilings: Mapping[str, int] = MappingProxyType({}),
    covered_axes: Collection[str] = (),
) -> tuple[MappingProxyType[str, int], ...]:
    """Enumerate one bootstrap width map or the budgeted frontier, widest first."""
    if budget_bytes is None:
        return (
            bootstrap_widths(
                axes=axes,
                fixed_widths=fixed_widths,
                width_ceilings=width_ceilings,
                covered_axes=covered_axes,
            ),
        )

    frontiers = tuple(
        _axis_frontier(
            axis=axis, fixed_widths=fixed_widths, width_ceilings=width_ceilings
        )
        for axis in axes
    )
    candidates = (
        _width_mapping(axes=axes, values=values)
        for values in itertools.product(*frontiers)
    )
    return tuple(sorted(candidates, key=_candidate_rank, reverse=True))


def _candidate_rank(widths: Mapping[str, int]) -> tuple[int, tuple[int, ...]]:
    """Rank a candidate by width product, then by its width tuple in axis order."""
    values = tuple(widths.values())
    return (math.prod(values), values)


def _axis_frontier(
    *,
    axis: ReducedAxis | TiledOutputAxis,
    fixed_widths: Mapping[str, int],
    width_ceilings: Mapping[str, int],
) -> tuple[int, ...]:
    """Return one fixed width, or the admissible 1/powers-of-two/full ladder."""
    ceiling = _ceiling_for(axis=axis, width_ceilings=width_ceilings)
    if axis.name in fixed_widths:
        return (_fixed_width(axis=axis, fixed_widths=fixed_widths, ceiling=ceiling),)

    widths = [1]
    power = 2
    while power < axis.extent:
        widths.append(power)
        power *= 2
    widths.append(axis.extent)
    admissible = {
        _admissible_width(axis=axis, width=width, ceiling=ceiling) for width in widths
    }
    return tuple(sorted(admissible))


def _fixed_width(
    *,
    axis: ReducedAxis | TiledOutputAxis,
    fixed_widths: Mapping[str, int],
    ceiling: int | None = None,
) -> int:
    """Return the fixed width of one axis under the width policy it declares."""
    return _admissible_width(
        axis=axis,
        width=min(fixed_widths[axis.name], axis.extent),
        ceiling=ceiling,
    )


def _ceiling_for(
    *,
    axis: ReducedAxis | TiledOutputAxis,
    width_ceilings: Mapping[str, int],
) -> int | None:
    """Return the declared upper bound on one axis's width, or `None`.

    Args:
        axis: The axis whose candidate widths are being enumerated.
        width_ceilings: The declared ceilings by axis name.

    Returns:
        The ceiling this axis is bound by, or `None` when none names it.

    Raises:
        ExecutionPlanningError: The ceiling lies below every legal width.

    """
    ceiling = width_ceilings.get(axis.name)
    if ceiling is None:
        return None
    smallest = _smallest_admissible_width(axis=axis)
    if ceiling < smallest:
        msg = (
            f"Workspace axis {axis.name!r} admits no width at or below the "
            f"declared ceiling {ceiling}: its narrowest legal width is "
            f"{smallest} under alignment {axis.alignment}, floor "
            f"{axis.minimum_width} and extent {axis.extent}."
        )
        raise ExecutionPlanningError(msg)
    return ceiling


def _admissible_width(
    *,
    axis: ReducedAxis | TiledOutputAxis,
    width: int,
    ceiling: int | None = None,
) -> int:
    """Return the width the axis admits nearest the proposal, preferring the shorter.

    An axis admits its full extent, whatever the alignment divides, plus every
    multiple of its alignment lying between its floor and that extent.  A proposal
    is rounded down onto that set; one that falls through it — below the floor, or
    below the alignment and so at zero — is lifted to the smallest width the set
    holds, which is the extent when no multiple of the alignment reaches the floor
    without passing the extent.

    A `ceiling` lowers the proposal before it is rounded, so the result never
    exceeds it.  The ceiling bounds the width only: the axis keeps its extent,
    its alignment and its floor, and a ceiling under that floor is refused by
    `_ceiling_for` before any proposal reaches here.
    """
    if ceiling is not None:
        width = min(width, ceiling)
    if width >= axis.extent:
        return axis.extent
    aligned = width - width % axis.alignment
    if aligned >= axis.minimum_width:
        return aligned
    return _smallest_admissible_width(axis=axis)


def _smallest_admissible_width(*, axis: ReducedAxis | TiledOutputAxis) -> int:
    """Return the narrowest width the axis admits: an aligned floor, else the extent."""
    lifted = -(-axis.minimum_width // axis.alignment) * axis.alignment
    return min(lifted, axis.extent)


def _width_mapping(
    *, axes: tuple[ReducedAxis | TiledOutputAxis, ...], values: tuple[int, ...]
) -> MappingProxyType[str, int]:
    """Bind a width tuple to axis names without losing declaration order."""
    return MappingProxyType(
        {axis.name: width for axis, width in zip(axes, values, strict=True)}
    )


def _memory_for_candidate[Compiled](
    *,
    compiled: Compiled,
    widths: Mapping[str, int],
    memory_for: Callable[[Compiled], CompilerMemoryReservation] | None,
) -> CompilerMemoryReservation:
    """Read complete candidate accounting directly or from its compiler cache."""
    if memory_for is None:
        return compiler_memory_reservation(compiled=compiled, widths=widths)

    try:
        value = memory_for(compiled)
    except Exception as exc:
        msg = f"Compiler memory reservation lookup failed for widths {dict(widths)!r}."
        raise ExecutionPlanningError(msg) from exc

    if not isinstance(value, CompilerMemoryReservation):
        msg = (
            "Compiler memory lookup returned no complete reservation for widths "
            f"{dict(widths)!r}."
        )
        raise ExecutionPlanningError(msg)
    return value


def _resident_bytes_for_candidate[Compiled](
    *,
    compiled: Compiled,
    widths: Mapping[str, int],
    lower_bound: int,
    resident_bytes_for: Callable[[Compiled], int] | None,
) -> int:
    """Validate a candidate-specific inventory without changing its raw peak."""
    if resident_bytes_for is None:
        return lower_bound
    try:
        resident = _validate_resident_bytes(resident_bytes=resident_bytes_for(compiled))
    except Exception as error:
        raise ExecutionPlanningError(
            f"Workspace residency lookup failed for widths {dict(widths)!r}."
        ) from error
    if resident < lower_bound:
        raise ExecutionPlanningError(
            "Workspace residency lookup returned fewer bytes than the declared "
            f"lower bound for widths {dict(widths)!r}."
        )
    return resident


def compiler_memory_reservation[Compiled](
    *, compiled: Compiled, widths: Mapping[str, int]
) -> CompilerMemoryReservation:
    """Reserve complete reported default-memory allocations and the raw peak.

    Accept one scalar record or a nonempty sequence/mapping of device records.
    Every record must supply nonnegative integral peak, argument, output, alias,
    and temporary counters, including the four host allocation counters. Nonzero
    host allocations are refused: the peak report does not reliably separate
    their memory space. Separately compiled CPU assembly remains a CPU profile.
    Generated-code metadata does not establish host allocation residency.

    The reservation enforces represented requirements. It does not establish a
    complete runtime upper bound for storage omitted by the compiler report.
    """
    analysis = _compiler_memory_analysis(compiled=compiled, widths=widths)
    try:
        records = _allocation_records(analysis=analysis)
        return CompilerMemoryReservation(
            records=tuple(_allocation_record(record=record) for record in records)
        )
    except Exception as exc:
        raise ExecutionPlanningError(
            "Compiler memory analysis returned no valid per-device reservation "
            f"for widths {dict(widths)!r}: {exc}"
        ) from exc


def compiler_peak_bytes[Compiled](
    *, compiled: Compiled, widths: Mapping[str, int]
) -> int:
    """Read and strictly normalize one candidate's compiler-reported peak."""
    analysis = _compiler_memory_analysis(compiled=compiled, widths=widths)
    try:
        return _peak_from_analysis(analysis=analysis)
    except Exception as exc:
        msg = (
            "Compiler memory analysis returned no valid per-device peak for widths "
            f"{dict(widths)!r}."
        )
        raise ExecutionPlanningError(msg) from exc


def _compiler_memory_analysis[Compiled](
    *, compiled: Compiled, widths: Mapping[str, int]
) -> object:
    """Read an executable report once without numerical dispatch."""
    try:
        analyze = cast("_MemoryAnalyzable", compiled).memory_analysis
    except Exception as exc:
        msg = f"Compiler memory analysis is unavailable for widths {dict(widths)!r}."
        raise ExecutionPlanningError(msg) from exc
    if not callable(analyze):
        msg = f"Compiler memory analysis is unavailable for widths {dict(widths)!r}."
        raise ExecutionPlanningError(msg)
    try:
        return analyze()
    except Exception as exc:
        msg = f"Compiler memory analysis failed for widths {dict(widths)!r}."
        raise ExecutionPlanningError(msg) from exc


def _allocation_records(*, analysis: object) -> tuple[object, ...]:
    """Keep fields paired within their record before reducing across devices."""
    if _peak_field(record=analysis) is not _MISSING:
        return (analysis,)
    if isinstance(analysis, Mapping):
        records = tuple(analysis.values())
    elif isinstance(analysis, (list, tuple)):
        records = tuple(analysis)
    else:
        raise TypeError("Memory analysis must expose a complete allocation record.")
    if not records:
        raise ValueError("Per-device memory analysis must not be empty.")
    return records


def _allocation_record(*, record: object) -> CompilerMemoryRecord:
    """Validate a scalar device record and its separately reported host space."""
    names = (
        "peak_memory_in_bytes",
        "argument_size_in_bytes",
        "output_size_in_bytes",
        "alias_size_in_bytes",
        "temp_size_in_bytes",
        "host_argument_size_in_bytes",
        "host_output_size_in_bytes",
        "host_alias_size_in_bytes",
        "host_temp_size_in_bytes",
    )
    fields = {
        name: record.get(name, _MISSING)
        if isinstance(record, Mapping)
        else getattr(record, name, _MISSING)
        for name in names
    }
    try:
        values = {
            name: _non_negative_bytes(value=value) for name, value in fields.items()
        }
        _fail_if_host_allocations(values=values)
        return CompilerMemoryRecord(
            peak_bytes=values["peak_memory_in_bytes"],
            argument_bytes=values["argument_size_in_bytes"],
            output_bytes=values["output_size_in_bytes"],
            alias_bytes=values["alias_size_in_bytes"],
            temporary_bytes=values["temp_size_in_bytes"],
        )
    except Exception as exc:
        snapshot = {
            name: "unavailable" if value is _MISSING else value
            for name, value in fields.items()
        }
        raise ValueError(f"{exc} Reported allocation fields: {snapshot!r}") from exc


def _fail_if_host_allocations(*, values: Mapping[str, int]) -> None:
    """Refuse a mixed-space peak whose default-memory share is unavailable."""
    if any(value for name, value in values.items() if name.startswith("host_")):
        raise ValueError("Mixed host/default allocation spaces are unsupported.")


def _peak_from_analysis(*, analysis: object) -> int:
    """Normalize one JAX-style record or nonempty per-device record collection."""
    peak = _peak_field(record=analysis)
    if peak is not _MISSING:
        return _normalize_peak_field(value=peak)

    if isinstance(analysis, Mapping):
        records = tuple(analysis.values())
    elif isinstance(analysis, (list, tuple)):
        records = tuple(analysis)
    else:
        msg = "Memory analysis must expose peak_memory_in_bytes."
        raise TypeError(msg)
    if not records:
        msg = "Per-device memory analysis must not be empty."
        raise ValueError(msg)
    return max(_peak_from_device_record(record=record) for record in records)


def _peak_from_device_record(*, record: object) -> int:
    """Read the required peak field from one per-device analysis record."""
    peak = _peak_field(record=record)
    if peak is _MISSING:
        msg = "Each per-device memory record must expose peak_memory_in_bytes."
        raise TypeError(msg)
    return _normalize_peak_field(value=peak)


def _peak_field(*, record: object) -> object:
    """Read a peak field from an attribute record or a string-keyed mapping."""
    if isinstance(record, Mapping):
        return record.get("peak_memory_in_bytes", _MISSING)
    return getattr(record, "peak_memory_in_bytes", _MISSING)


def _normalize_peak_field(*, value: object) -> int:
    """Normalize one integral peak or nonempty collection of per-device peaks."""
    if isinstance(value, Mapping):
        peaks = tuple(value.values())
    elif isinstance(value, (list, tuple)):
        peaks = tuple(value)
    else:
        return _non_negative_bytes(value=value)
    if not peaks:
        msg = "A per-device peak collection must not be empty."
        raise ValueError(msg)
    return max(_non_negative_bytes(value=peak) for peak in peaks)


def _non_negative_bytes(*, value: object) -> int:
    """Accept integer-like byte counts while rejecting booleans and lossy casts."""
    if isinstance(value, bool):
        msg = "A compiler memory counter must be an integer byte count, not bool."
        raise TypeError(msg)
    try:
        normalized = operator.index(cast("SupportsIndex", value))
    except TypeError as exc:
        msg = "A compiler memory counter must be an integer byte count."
        raise TypeError(msg) from exc
    if normalized < 0:
        msg = "A compiler memory counter must be non-negative."
        raise ValueError(msg)
    return normalized
