"""Lower age-indexed regime declarations into demand, support and engine laws.

A regime's `regime_transitions` declares where a local problem is available and
where it may go. This module is the single place that reads it for that purpose:

- `regime_transitions=None` is terminal and available at every age;
- a plain nonterminal law — a regime name, a `Choose`, a vector
  `MarkovTransition` with `targets`, or a per-target mapping — is available at
  every non-final age;
- a `ByAge` schedule is available at the non-final ages its cases select. A law
  it selects at the last age is unused: no continuation exists there.

Availability alone solves nothing. `resolve_demand` expands the declared starts
into the pairs a subject can visit and the pairs whose values those problems
read; that set is the coverage every later stage reads. Support comes only
from the declarations. A bare callable or a vector `MarkovTransition` without
`targets` names no support and is rejected.

The engine downstream reads one law per regime and phase. A schedule whose
covered periods select different laws is lowered into one equivalent law whose
cells dispatch on the `period` context argument; a cell whose target is outside
the selected law's support at a period evaluates to exactly zero there. Laws
shared across cases are reused unchanged, so no per-age closure is created.
"""

import dataclasses
import inspect
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import jax.numpy as jnp

from _lcm.typing import RegimeName
from lcm.ages import AgeGrid
from lcm.collective import ValueDependentTransition
from lcm.exceptions import ModelInitializationError, RegimeInitializationError
from lcm.phased import Phased
from lcm.transition import (
    ByAge,
    Choose,
    MarkovTransition,
    _freeze_selector,
    _select_periods,
)
from lcm.typing import Period

type PhaseKey = str
_PHASES: tuple[PhaseKey, PhaseKey] = ("solution", "simulation")


@dataclass(frozen=True, kw_only=True)
class RegimeSchedules:
    """Coverage and per-period support of every regime, from its declaration."""

    coverage_by_regime: MappingProxyType[RegimeName, tuple[int, ...]]
    """Periods at which each regime is solved: its available periods until
    `resolve_demand` restricts them to the demanded ones."""

    support_by_phase: MappingProxyType[
        PhaseKey, MappingProxyType[RegimeName, MappingProxyType[int, tuple[str, ...]]]
    ]
    """Per phase, per source regime, the declared targets at each covered period."""

    transitions: MappingProxyType[RegimeName, object]
    """Each regime's transition in the engine's period-independent vocabulary."""

    value_reads_by_phase: MappingProxyType[
        PhaseKey, MappingProxyType[RegimeName, MappingProxyType[int, tuple[str, ...]]]
    ] = MappingProxyType({})
    """Per phase, per source regime, the regimes whose next-age value a gated
    edge reads at each period: gate references, plus the solve fallbacks in
    the solution phase."""

    landings_by_regime: MappingProxyType[
        RegimeName, MappingProxyType[int, tuple[str, ...]]
    ] = MappingProxyType({})
    """Per source regime, the simulate fallbacks a gated edge may land a row in
    at the next age, at each period."""

    nodes: frozenset[tuple[int, RegimeName]] = frozenset()
    """The `(period, regime)` pairs whose value is required (S)."""

    visited_nodes: frozenset[tuple[int, RegimeName]] = frozenset()
    """The `(period, regime)` pairs a subject can physically visit (H)."""

    law_by_period_by_regime: MappingProxyType[
        RegimeName, MappingProxyType[int, object]
    ] = MappingProxyType({})
    """Per nonterminal regime, the declared law at each available period."""


def uses_declaration_vocabulary(transition: object) -> bool:
    """Whether a regime transition has a form the engine cannot read directly."""
    if isinstance(transition, ByAge | Choose | str):
        return True
    if isinstance(transition, MarkovTransition):
        return transition.targets is not None
    if isinstance(transition, Phased):
        return any(
            uses_declaration_vocabulary(side)
            for side in (transition.solve, transition.simulate)
        )
    return False


def declaration_view(transition: object) -> object:
    """Return a period-independent law in the engine vocabulary, for validation.

    A `ByAge` is replaced by the union of its laws: every cell each case declares
    for a target. A regime name and a `Choose` become probability cells when they
    meet a stochastic law, and deterministic otherwise. Cells are not masked by
    age, so the view is only for construction-time checks of the declared
    grammar, gates and state handoffs, never for evaluation.
    """
    if isinstance(transition, ByAge):
        return _union_law(laws=transition.laws, code_by_name=None, mask=None)
    if isinstance(transition, Phased) and not isinstance(transition.solve, ByAge):
        return Phased(
            solve=_plain_law(law=transition.solve, code_by_name=None),
            simulate=_plain_law(law=transition.simulate, code_by_name=None),
        )
    return _plain_law(law=transition, code_by_name=None)


def resolve_regime_schedules(
    *,
    user_regimes: Mapping[RegimeName, Any],
    ages: AgeGrid,
    regime_names_to_ids: Mapping[RegimeName, int],
) -> RegimeSchedules:
    """Resolve every regime's declared coverage, support and engine law."""
    _fail_if_support_is_undeclared(user_regimes=user_regimes)
    all_periods = tuple(range(ages.n_periods))
    coverage: dict[RegimeName, tuple[int, ...]] = {}
    support: dict[
        PhaseKey, dict[RegimeName, MappingProxyType[int, tuple[str, ...]]]
    ] = {phase: {} for phase in _PHASES}
    transitions: dict[RegimeName, object] = {}
    value_reads: dict[
        PhaseKey, dict[RegimeName, MappingProxyType[int, tuple[str, ...]]]
    ] = {phase: {} for phase in _PHASES}
    landings: dict[RegimeName, MappingProxyType[int, tuple[str, ...]]] = {}
    laws: dict[RegimeName, MappingProxyType[int, object]] = {}
    for name, regime in user_regimes.items():
        transition = regime.regime_transitions
        if transition is None:
            coverage[name] = all_periods
            transitions[name] = None
            continue
        law_by_period = (
            {
                period: law
                for period, law in transition.resolve(ages).law_by_period.items()
                if period < ages.n_periods - 1
            }
            if isinstance(transition, ByAge)
            else dict.fromkeys(all_periods[:-1], transition)
        )
        if not law_by_period:
            raise ModelInitializationError(
                f"Regime '{name}' declares a schedule that covers no age of the model."
            )
        coverage[name] = tuple(law_by_period)
        side_by_phase = {
            phase: {
                period: _phase_side(law=law, side=side)
                for period, law in law_by_period.items()
            }
            for phase, side in zip(_PHASES, ("solve", "simulate"), strict=True)
        }
        for phase, side_by_period in side_by_phase.items():
            support[phase][name] = MappingProxyType(
                {
                    period: _declared_support(law=law, all_regimes=user_regimes)
                    for period, law in side_by_period.items()
                }
            )
        solve_side, simulate_side = side_by_phase.values()
        value_reads["solution"][name] = MappingProxyType(
            {
                period: (*_gate_references(law), *_fallbacks(law=law, side="solve"))
                for period, law in solve_side.items()
            }
        )
        value_reads["simulation"][name] = MappingProxyType(
            {period: _gate_references(law) for period, law in simulate_side.items()}
        )
        landings[name] = MappingProxyType(
            {
                period: _fallbacks(law=law, side="simulate")
                for period, law in simulate_side.items()
            }
        )
        laws[name] = MappingProxyType(law_by_period)
        transitions[name] = _lower(
            law_by_period=law_by_period,
            solve_periods=tuple(law_by_period),
            simulate_periods=tuple(law_by_period),
            code_by_name=regime_names_to_ids,
        )
    return RegimeSchedules(
        coverage_by_regime=MappingProxyType(coverage),
        support_by_phase=MappingProxyType(
            {phase: MappingProxyType(by_regime) for phase, by_regime in support.items()}
        ),
        transitions=MappingProxyType(transitions),
        value_reads_by_phase=MappingProxyType(
            {
                phase: MappingProxyType(by_regime)
                for phase, by_regime in value_reads.items()
            }
        ),
        landings_by_regime=MappingProxyType(landings),
        law_by_period_by_regime=MappingProxyType(laws),
    )


def lower_demanded_transitions(
    *,
    schedules: RegimeSchedules,
    code_by_name: Mapping[str, int],
) -> MappingProxyType[RegimeName, object]:
    """Lower each regime's law over the periods demand requires, only.

    The solve side is lowered over the regime's solved periods (S) and the
    simulate side over its visited periods (H), falling back to S for a regime
    that is solved but never visited. A case selected only at undemanded ages
    contributes no cell, no argument and no parameter. A regime without
    demanded periods keeps its lowering over every available period: it is
    never executed, and `create_params_template` gives it no parameters.
    """
    lowered = dict(schedules.transitions)
    for name, law_by_period in schedules.law_by_period_by_regime.items():
        solve_periods = schedules.coverage_by_regime[name]
        if not solve_periods:
            continue
        visited = tuple(sorted(p for p, n in schedules.visited_nodes if n == name))
        lowered[name] = _lower(
            law_by_period=law_by_period,
            solve_periods=solve_periods,
            simulate_periods=visited or solve_periods,
            code_by_name=code_by_name,
        )
    return MappingProxyType(lowered)


def _lower(
    *,
    law_by_period: Mapping[int, object],
    solve_periods: tuple[int, ...],
    simulate_periods: tuple[int, ...],
    code_by_name: Mapping[str, int],
) -> object:
    """One engine law from the per-period laws at the given periods."""
    solve_side = {
        period: _phase_side(law=law_by_period[period], side="solve")
        for period in solve_periods
    }
    simulate_side = {
        period: _phase_side(law=law_by_period[period], side="simulate")
        for period in simulate_periods
    }
    solve_law = _lower_side(law_by_period=solve_side, code_by_name=code_by_name)
    # A schedule without phase variation lowers to one shared engine law.
    if all(
        law is _phase_side(law=law_by_period[period], side="solve")
        for period, law in simulate_side.items()
    ):
        return solve_law
    return Phased(
        solve=solve_law,
        simulate=_lower_side(law_by_period=simulate_side, code_by_name=code_by_name),
    )


def resolve_demand(
    *,
    schedules: RegimeSchedules,
    initial_nodes: frozenset[tuple[object, RegimeName]],
    same_period_refs_by_regime: Mapping[RegimeName, tuple[RegimeName, ...]],
    terminal_regimes: frozenset[RegimeName],
    ages: AgeGrid,
) -> RegimeSchedules:
    """Restrict `schedules` to the problems the starts require.

    Two roles are expanded to a fixed point from the starts:

    - a physical visit (H) requires the pair's value, its realized successors
      and gated landings as visits, and the values the realized routing reads;
    - a value (S) requires the values its backward problem reads: perceived
      continuation targets, gate references, solve fallbacks and same-period
      references. Realized routes out of a value-only pair are not followed.

    Every required pair must be an available problem: a terminal regime, or a
    nonterminal one with a law at a non-final age. The returned schedules
    cover exactly S; solution support is kept for S and simulation support for
    H.

    Raises:
        ModelInitializationError: If a start or a required read names a pair
            where no problem is available.

    """
    period_by_age: dict[object, int] = {
        age: period for period, age in enumerate(ages.exact_values)
    }
    available = {
        name: frozenset(periods)
        for name, periods in schedules.coverage_by_regime.items()
    }
    work: deque[tuple[bool, int, RegimeName, str]] = deque(
        (True, period_by_age[age], name, "`initial_regimes`")
        for age, name in sorted(initial_nodes, key=repr)
    )
    visited: set[tuple[int, RegimeName]] = set()
    valued: set[tuple[int, RegimeName]] = set()
    support = schedules.support_by_phase
    value_reads = schedules.value_reads_by_phase
    while work:
        physical, period, name, requester = work.popleft()
        done = visited if physical else valued
        if (period, name) in done:
            continue
        if name not in terminal_regimes and period not in available[name]:
            raise ModelInitializationError(
                _unavailable_message(
                    requester=requester, name=name, period=period, ages=ages
                )
            )
        done.add((period, name))
        here = f"the transition out of ({ages.exact_values[period]}, '{name}')"
        if physical:
            work.append((False, period, name, requester))
            work.extend(
                (True, period + 1, target, here)
                for target in (
                    *support["simulation"].get(name, {}).get(period, ()),
                    *schedules.landings_by_regime.get(name, {}).get(period, ()),
                )
            )
            reads = value_reads["simulation"].get(name, {}).get(period, ())
        else:
            reads = (
                *support["solution"].get(name, {}).get(period, ()),
                *value_reads["solution"].get(name, {}).get(period, ()),
            )
            work.extend(
                (
                    False,
                    period,
                    reference,
                    (
                        f"a same-period reference of ({ages.exact_values[period]}, "
                        f"'{name}')"
                    ),
                )
                for reference in same_period_refs_by_regime.get(name, ())
            )
        work.extend((False, period + 1, target, here) for target in reads)

    def _restricted(
        *,
        by_regime: Mapping[RegimeName, Mapping[int, tuple[str, ...]]],
        keep: set[tuple[int, RegimeName]],
    ) -> MappingProxyType[RegimeName, MappingProxyType[int, tuple[str, ...]]]:
        return MappingProxyType(
            {
                name: MappingProxyType(
                    {p: t for p, t in by_period.items() if (p, name) in keep}
                )
                for name, by_period in by_regime.items()
            }
        )

    return dataclasses.replace(
        schedules,
        coverage_by_regime=MappingProxyType(
            {
                name: tuple(sorted(p for p, n in valued if n == name))
                for name in schedules.coverage_by_regime
            }
        ),
        support_by_phase=MappingProxyType(
            {
                "solution": _restricted(by_regime=support["solution"], keep=valued),
                "simulation": _restricted(
                    by_regime=support["simulation"], keep=visited
                ),
            }
        ),
        nodes=frozenset(valued),
        visited_nodes=frozenset(visited),
    )


def _unavailable_message(
    *, requester: str, name: RegimeName, period: int, ages: AgeGrid
) -> str:
    age = ages.exact_values[period]
    reason = (
        "which is nonterminal at the last age: no next age exists"
        if period == ages.n_periods - 1
        else f"where '{name}' supplies no law"
    )
    return f"{requester} requires '{name}' at age {age}, {reason}."


def _gate_references(law: object) -> tuple[str, ...]:
    """The regimes whose value the gates of a per-target mapping read."""
    if not isinstance(law, Mapping):
        return ()
    return tuple(
        reference.regime
        for cell in law.values()
        if isinstance(cell, ValueDependentTransition)
        for reference in cell.gate_references.values()
    )


def _fallbacks(*, law: object, side: str) -> tuple[str, ...]:
    """The gate-closed regimes of a per-target mapping, for one phase side."""
    if not isinstance(law, Mapping):
        return ()
    return tuple(
        getattr(route, f"{side}_fallback").regime
        for cell in law.values()
        if isinstance(cell, ValueDependentTransition)
        for route in cell.routes.values()
    )


def resolve_initial_nodes(
    *,
    initial_regimes: object,
    regime_names: Sequence[RegimeName],
    ages: AgeGrid,
) -> frozenset[tuple[object, RegimeName]]:
    """Normalize `Model(initial_regimes=...)` to the exact admissible start pairs.

    `initial_regimes` maps age selectors (as in `ByAge`) to a regime name or a
    nonempty sequence of names. Each rule contributes the Cartesian product of
    its selected grid ages and names; rules are unioned. The result depends only
    on the declaration and the clock, never on solve coverage.
    """
    if not isinstance(initial_regimes, Mapping) or not initial_regimes:
        raise ModelInitializationError(
            "`initial_regimes` must be a nonempty mapping from age selectors to "
            f"regime names, e.g. `{{25: ('single', 'couple')}}`; got "
            f"{initial_regimes!r}."
        )
    period_by_age: dict[object, int] = {
        age: period for period, age in enumerate(ages.exact_values)
    }
    permitted: set[tuple[object, RegimeName]] = set()
    for selector, value in initial_regimes.items():
        names = _entry_names(value)
        _fail_if_unknown_entry_regimes(names=names, regime_names=regime_names)
        try:
            periods = _select_periods(
                selector=_freeze_selector(selector),
                ages=ages,
                period_by_age=period_by_age,
            )
        except RegimeInitializationError as error:
            raise ModelInitializationError(str(error)) from error
        if not periods:
            raise ModelInitializationError(
                f"The `initial_regimes` selector {selector!r} selects no age of "
                "the model."
            )
        permitted |= {
            (ages.exact_values[period], name) for period in periods for name in names
        }
    return frozenset(permitted)


def _entry_names(value: object) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value,)
    if (
        isinstance(value, Sequence)
        and value
        and all(isinstance(name, str) for name in value)
    ):
        return tuple(value)
    raise ModelInitializationError(
        "`initial_regimes` names regimes by a name or a nonempty sequence of "
        f"names; got {value!r}."
    )


def _fail_if_unknown_entry_regimes(
    *, names: tuple[str, ...], regime_names: Sequence[RegimeName]
) -> None:
    unknown = sorted(set(names) - set(regime_names))
    if unknown:
        raise ModelInitializationError(
            f"`initial_regimes` names unknown regime(s) {unknown}; regimes are "
            f"{sorted(regime_names)}."
        )


# keyword-only-exempt: primary-argument=func
def _with_signature(
    func: Callable[..., Any],
    *,
    names: tuple[str, ...],
    sources: tuple[Callable[..., Any], ...],
) -> Callable[..., Any]:
    """Expose `names` as keyword-only arguments annotated as in `sources`.

    The DAG machinery requires one annotation per argument name across a
    model, so each argument keeps the annotation of the law that reads it, and
    `period` is annotated as the period context argument.

    The `sources` are kept on the wrapper so parameter-indexing inspection
    reads the laws the user wrote rather than the wrapper's body.
    """
    annotations: dict[str, Any] = {"period": Period}
    for source in sources:
        for name, parameter in inspect.signature(source).parameters.items():
            if parameter.annotation is not inspect.Parameter.empty:
                annotations.setdefault(name, parameter.annotation)
    func.__signature__ = inspect.Signature(  # ty: ignore[unresolved-attribute]
        [
            inspect.Parameter(
                name,
                inspect.Parameter.KEYWORD_ONLY,
                annotation=annotations.get(name, inspect.Parameter.empty),
            )
            for name in names
        ]
    )
    func.__annotations__ = {
        name: annotations[name] for name in names if name in annotations
    }
    func.__lcm_sources__ = sources  # ty: ignore[unresolved-attribute]
    return func


def _constant(value: float) -> Callable[..., Any]:
    """A law that returns one fixed value and reads nothing."""

    def constant() -> Any:  # noqa: ANN401
        return jnp.asarray(value)

    return constant


def _one_hot(vector: tuple[float, ...]) -> Callable[..., Any]:
    """A constant probability vector."""

    def one_hot() -> Any:  # noqa: ANN401
        return jnp.asarray(vector)

    return one_hot


def _indicator(
    *, selector: Callable[..., Any], code: int, names: tuple[str, ...]
) -> Callable[..., Any]:
    """Probability one where a deterministic selector returns `code`."""

    def indicator(**kwargs: Any) -> Any:  # noqa: ANN401
        selected = selector(**{name: kwargs[name] for name in names})
        return jnp.asarray(selected == code, dtype=float)

    return _with_signature(indicator, names=names, sources=(selector,))


def _period_masked(
    *, cell: Callable[..., Any], periods: tuple[int, ...], names: tuple[str, ...]
) -> Callable[..., Any]:
    """A probability cell that is exactly zero outside `periods`."""

    def period_masked(**kwargs: Any) -> Any:  # noqa: ANN401
        value = cell(**{name: kwargs[name] for name in names})
        selected = jnp.isin(kwargs["period"], jnp.asarray(periods))
        return jnp.where(selected, value, jnp.zeros_like(value))

    return _with_signature(period_masked, names=_with_period(names), sources=(cell,))


def _period_dispatch(
    *,
    cases: tuple[Callable[..., Any], ...],
    case_names: tuple[tuple[str, ...], ...],
    case_by_period: tuple[int, ...],
) -> Callable[..., Any]:
    """Evaluate the case callable selected for the current period."""

    def period_dispatch(**kwargs: Any) -> Any:  # noqa: ANN401
        values = [
            jnp.asarray(case(**{name: kwargs[name] for name in names}))
            for case, names in zip(cases, case_names, strict=True)
        ]
        index = jnp.asarray(case_by_period)[kwargs["period"]]
        result = values[-1]
        for position in range(len(values) - 2, -1, -1):
            result = jnp.where(index == position, values[position], result)
        return result

    names = tuple(sorted({name for names in case_names for name in names}))
    return _with_signature(period_dispatch, names=_with_period(names), sources=cases)


def _with_period(names: tuple[str, ...]) -> tuple[str, ...]:
    return ("period", *(name for name in names if name != "period"))


def _signature(*, names: tuple[str, ...]) -> inspect.Signature:
    return inspect.Signature(
        [inspect.Parameter(name, inspect.Parameter.KEYWORD_ONLY) for name in names]
    )


def _argument_names(func: Callable[..., Any]) -> tuple[str, ...]:
    return tuple(inspect.signature(func).parameters)


def _phase_side(*, law: object, side: str) -> object:
    return getattr(law, side) if isinstance(law, Phased) else law


def _declared_support(
    *, law: object, all_regimes: Mapping[RegimeName, Any]
) -> tuple[str, ...]:
    """The targets one nonterminal law declares."""
    if isinstance(law, str):
        support: tuple[str, ...] = (law,)
    elif isinstance(law, Choose | MarkovTransition):
        support = tuple(law.targets or ())
    elif isinstance(law, Mapping):
        support = tuple(law)
    else:
        support = ()
    unknown = sorted(set(support) - set(all_regimes))
    if unknown:
        raise ModelInitializationError(
            f"A regime transition names unknown target regime(s) {unknown}; "
            f"regimes are {sorted(all_regimes)}."
        )
    return support


def _lower_side(
    *, law_by_period: Mapping[int, object], code_by_name: Mapping[str, int]
) -> object:
    """One phase's law as a single period-independent engine law."""
    distinct = _distinct(law_by_period.values())
    if len(distinct) == 1:
        return _plain_law(law=distinct[0], code_by_name=code_by_name)
    return _union_law(
        laws=distinct,
        code_by_name=code_by_name,
        mask={
            id(law): tuple(p for p, x in law_by_period.items() if x is law)
            for law in distinct
        },
        law_by_period=law_by_period,
    )


def _plain_law(*, law: object, code_by_name: Mapping[str, int] | None) -> object:
    """A single nonterminal law in the engine vocabulary."""
    if isinstance(law, str):
        return _constant(_code(name=law, code_by_name=code_by_name))
    if isinstance(law, Choose):
        return law.func
    return law


def _union_law(
    *,
    laws: tuple[object, ...],
    code_by_name: Mapping[str, int] | None,
    mask: Mapping[int, tuple[int, ...]] | None,
    law_by_period: Mapping[int, object] | None = None,
) -> object:
    """Several laws as one: dispatch deterministic laws, merge stochastic cells."""
    if isinstance(laws[0], Phased) or any(isinstance(law, Phased) for law in laws):
        solve = _union_law(
            laws=_distinct(_phase_side(law=law, side="solve") for law in laws),
            code_by_name=code_by_name,
            mask=None,
        )
        simulate = _union_law(
            laws=_distinct(_phase_side(law=law, side="simulate") for law in laws),
            code_by_name=code_by_name,
            mask=None,
        )
        return Phased(solve=solve, simulate=simulate)
    stochastic = [law for law in laws if isinstance(law, Mapping | MarkovTransition)]
    if not stochastic:
        return _deterministic_dispatch(
            laws=laws, code_by_name=code_by_name, law_by_period=law_by_period
        )
    if any(isinstance(law, MarkovTransition) for law in stochastic):
        return _vector_union(
            laws=laws, code_by_name=code_by_name, law_by_period=law_by_period
        )
    return _mapping_union(laws=laws, code_by_name=code_by_name, mask=mask)


def _deterministic_dispatch(
    *,
    laws: tuple[object, ...],
    code_by_name: Mapping[str, int] | None,
    law_by_period: Mapping[int, object] | None,
) -> object:
    cases = tuple(_plain_law(law=law, code_by_name=code_by_name) for law in laws)
    if law_by_period is None:
        return cases[0]
    return _dispatch(cases=cases, laws=laws, law_by_period=law_by_period)


def _dispatch(
    *,
    cases: tuple[Any, ...],
    laws: tuple[object, ...],
    law_by_period: Mapping[int, object],
) -> Callable[..., Any]:
    n_periods = max(law_by_period) + 2
    position = {id(law): index for index, law in enumerate(laws)}
    return _period_dispatch(
        cases=cases,
        case_names=tuple(_argument_names(case) for case in cases),
        case_by_period=tuple(
            position[id(law_by_period[period])] if period in law_by_period else 0
            for period in range(n_periods)
        ),
    )


def _vector_union(
    *,
    laws: tuple[object, ...],
    code_by_name: Mapping[str, int] | None,
    law_by_period: Mapping[int, object] | None,
) -> object:
    """Several vector laws and exits as one vector `MarkovTransition`."""
    if any(isinstance(law, Mapping) for law in laws):
        raise ModelInitializationError(
            "A schedule mixes a vector `MarkovTransition` with a per-target "
            "mapping. Use one form for every case: write the vector law's "
            "targets as a per-target mapping, or every case as a vector law."
        )
    targets = tuple(
        dict.fromkeys(target for law in laws for target in _law_targets(law))
    )
    if law_by_period is None or code_by_name is None:
        return next(law for law in laws if isinstance(law, MarkovTransition))
    n_regimes = max(code_by_name.values()) + 1
    cases = tuple(
        _vector_case(law=law, n_regimes=n_regimes, code_by_name=code_by_name)
        for law in laws
    )
    return MarkovTransition(
        func=_dispatch(cases=cases, laws=laws, law_by_period=law_by_period),
        targets=targets,
    )


def _vector_case(
    *, law: object, n_regimes: int, code_by_name: Mapping[str, int]
) -> Callable[..., Any]:
    if isinstance(law, MarkovTransition):
        return law.func
    if not isinstance(law, str):
        raise ModelInitializationError(
            f"A schedule mixes a vector `MarkovTransition` with {law!r}. Pair a "
            "vector law only with regime-name exits, or write every case as a "
            "per-target mapping."
        )
    one_hot = [0.0] * n_regimes
    one_hot[_code(name=law, code_by_name=code_by_name)] = 1.0
    return _one_hot(tuple(one_hot))


def _mapping_union(
    *,
    laws: tuple[object, ...],
    code_by_name: Mapping[str, int] | None,
    mask: Mapping[int, tuple[int, ...]] | None,
) -> MappingProxyType[str, object]:
    """Per-target mappings and exits merged; each target keeps one cell."""
    cells_by_target: dict[str, list[tuple[object, object]]] = {}
    for law in laws:
        for target, cell in _law_cells(law=law, code_by_name=code_by_name).items():
            cells_by_target.setdefault(target, []).append((law, cell))
    merged: dict[str, object] = {}
    for target, entries in cells_by_target.items():
        distinct_cells = _distinct(cell for _, cell in entries)
        if len(distinct_cells) > 1:
            if any(
                isinstance(cell, ValueDependentTransition) for cell in distinct_cells
            ):
                raise ModelInitializationError(
                    f"The transition into {target!r} is value-dependent in one "
                    "schedule case and declared differently in another. A gated "
                    "target keeps one `ValueDependentTransition` across every "
                    "case that names it."
                )
            if mask is None:
                merged[target] = distinct_cells[0]
                continue
            periods_by_cell = {
                id(cell): tuple(
                    period
                    for law, same in entries
                    if same is cell
                    for period in mask[id(law)]
                )
                for cell in distinct_cells
            }
            merged[target] = MarkovTransition(
                func=_period_sum(
                    tuple(
                        _masked(cell=cell, periods=periods_by_cell[id(cell)])
                        for cell in distinct_cells
                    )
                )
            )
            continue
        cell = distinct_cells[0]
        if mask is None:
            merged[target] = cell
            continue
        periods = tuple(
            sorted(period for law, _ in entries for period in mask[id(law)])
        )
        all_covered = tuple(sorted(period for law in laws for period in mask[id(law)]))
        merged[target] = (
            cell if periods == all_covered else _masked_cell(cell=cell, periods=periods)
        )
    return MappingProxyType(merged)


def _masked(*, cell: Any, periods: tuple[int, ...]) -> Callable[..., Any]:  # noqa: ANN401
    func = cell.func if isinstance(cell, MarkovTransition) else cell
    return _period_masked(
        cell=func, periods=tuple(sorted(periods)), names=_argument_names(func)
    )


def _period_sum(parts: tuple[Callable[..., Any], ...]) -> Callable[..., Any]:
    """Sum of period-masked cells whose periods never overlap."""
    part_names = tuple(_argument_names(part) for part in parts)

    def period_sum(**kwargs: Any) -> Any:  # noqa: ANN401
        values = [
            part(**{name: kwargs[name] for name in names})
            for part, names in zip(parts, part_names, strict=True)
        ]
        return sum(values[1:], values[0])

    names = tuple(sorted({name for names in part_names for name in names}))
    return _with_signature(period_sum, names=_with_period(names), sources=parts)


def _masked_cell(*, cell: object, periods: tuple[int, ...]) -> object:
    """Zero a cell outside `periods`; a gated cell keeps its gate and routes."""
    if isinstance(cell, ValueDependentTransition):
        probability = cell.probability
        func = (
            probability.func
            if isinstance(probability, MarkovTransition)
            else probability
        )
        return dataclasses.replace(
            cell, probability=MarkovTransition(func=_masked(cell=func, periods=periods))
        )
    return MarkovTransition(func=_masked(cell=cell, periods=periods))


def _law_cells(
    *, law: object, code_by_name: Mapping[str, int] | None
) -> Mapping[str, object]:
    if isinstance(law, Mapping):
        return law
    if isinstance(law, str):
        return {law: MarkovTransition(func=_constant(1.0))}
    if isinstance(law, Choose):
        names = _argument_names(law.func)
        return {
            target: MarkovTransition(
                func=_indicator(
                    selector=law.func,
                    code=_code(name=target, code_by_name=code_by_name),
                    names=names,
                )
            )
            for target in law.targets
        }
    raise ModelInitializationError(
        f"A schedule case {law!r} is not a nonterminal regime law. Use a regime "
        "name, `Choose(func=func, targets=...)`, "
        "`MarkovTransition(func=func, targets=...)` "
        "or a per-target mapping."
    )


def _law_targets(law: object) -> tuple[str, ...]:
    if isinstance(law, str):
        return (law,)
    if isinstance(law, Choose | MarkovTransition):
        return tuple(law.targets or ())
    return ()


def _code(*, name: str, code_by_name: Mapping[str, int] | None) -> int:
    if code_by_name is None:
        return 0
    if name not in code_by_name:
        raise ModelInitializationError(
            f"A regime transition names unknown target regime {name!r}; regimes "
            f"are {sorted(code_by_name)}."
        )
    return code_by_name[name]


def _distinct(values: Any) -> tuple[Any, ...]:  # noqa: ANN401
    """Distinct objects by identity, in first-seen order."""
    return tuple({id(value): value for value in values}.values())


def _fail_if_support_is_undeclared(*, user_regimes: Mapping[RegimeName, Any]) -> None:
    """Every nonterminal law names the regimes it may select."""
    errors = []
    for name, regime in user_regimes.items():
        transition = regime.regime_transitions
        laws = (
            transition.laws
            if isinstance(transition, ByAge)
            else ()
            if transition is None
            else (transition,)
        )
        for law in laws:
            for side in (
                (law.solve, law.simulate) if isinstance(law, Phased) else (law,)
            ):
                if isinstance(side, MarkovTransition) and side.targets is None:
                    errors.append(
                        f"Regime '{name}' declares a vector `MarkovTransition` "
                        "without `targets`. Name its support: "
                        "`MarkovTransition(func=func, targets=('a', 'b'))`."
                    )
                elif callable(side) and not isinstance(
                    side, Choose | MarkovTransition | Mapping
                ):
                    errors.append(
                        f"Regime '{name}' declares a bare deterministic "
                        "transition. Name its support: "
                        "`Choose(func=func, targets=('a', 'b'))`."
                    )
    if errors:
        raise ModelInitializationError("\n".join(errors))
