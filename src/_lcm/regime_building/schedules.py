"""Lower dated regime declarations into coverage, support and period-indexed laws.

A regime's `transition` declares where it is solved and where it may go. This
module is the single place that reads it for that purpose:

- `transition=None` is terminal and covered at every age;
- a plain nonterminal law — a regime name, a `Choose`, a vector
  `MarkovTransition` with `targets`, or a per-target mapping — covers every
  non-final age;
- a `ByAge` schedule covers exactly the ages its cases select.

A model is *dated* once any regime uses a regime name, `Choose`, `ByAge` or a
vector `MarkovTransition` with `targets`. A dated model reads coverage and
support only from these declarations. A model using none of them keeps its
`active` predicates and conservative coarse support.

The engine downstream reads one law per regime and phase. A schedule whose
covered periods select different laws is lowered into one equivalent law whose
cells dispatch on the `period` context argument; a cell whose target is outside
the selected law's support at a period evaluates to exactly zero there. Laws
shared across cases are reused unchanged, so no per-age closure is created.
"""

import dataclasses
import inspect
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

type PhaseKey = str
_PHASES: tuple[PhaseKey, PhaseKey] = ("solution", "simulation")


@dataclass(frozen=True, kw_only=True)
class RegimeSchedules:
    """Coverage and per-period support of every regime, from its declaration."""

    dated: bool
    """Whether coverage comes from the declarations rather than `active`."""

    coverage_by_regime: MappingProxyType[RegimeName, tuple[int, ...]]
    """Periods at which each regime is solved."""

    support_by_phase: MappingProxyType[
        PhaseKey, MappingProxyType[RegimeName, MappingProxyType[int, tuple[str, ...]]]
    ]
    """Per phase, per source regime, the declared targets at each covered period."""

    transitions: MappingProxyType[RegimeName, object]
    """Each regime's transition in the engine's period-independent vocabulary."""


def is_dated_declaration(transition: object) -> bool:
    """Whether a regime transition uses the dated vocabulary."""
    if isinstance(transition, ByAge | Choose | str):
        return True
    if isinstance(transition, MarkovTransition):
        return transition.targets is not None
    if isinstance(transition, Phased):
        return is_dated_declaration(transition.solve) or is_dated_declaration(
            transition.simulate
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
    dated = any(
        is_dated_declaration(regime.transition) for regime in user_regimes.values()
    )
    if not dated:
        coverage = {
            name: tuple(ages.get_periods_where(regime.active))
            for name, regime in user_regimes.items()
        }
        return RegimeSchedules(
            dated=False,
            coverage_by_regime=MappingProxyType(coverage),
            support_by_phase=MappingProxyType({}),
            transitions=MappingProxyType(
                {name: regime.transition for name, regime in user_regimes.items()}
            ),
        )

    _fail_if_legacy_declarations(user_regimes=user_regimes)
    all_periods = tuple(range(ages.n_periods))
    coverage: dict[RegimeName, tuple[int, ...]] = {}
    support: dict[
        PhaseKey, dict[RegimeName, MappingProxyType[int, tuple[str, ...]]]
    ] = {phase: {} for phase in _PHASES}
    transitions: dict[RegimeName, object] = {}
    for name, regime in user_regimes.items():
        transition = regime.transition
        if transition is None:
            coverage[name] = all_periods
            transitions[name] = None
            continue
        law_by_period = (
            dict(transition.resolve(ages).law_by_period)
            if isinstance(transition, ByAge)
            else dict.fromkeys(all_periods[:-1], transition)
        )
        if not law_by_period:
            raise ModelInitializationError(
                f"Regime '{name}' declares a schedule that covers no age of the model."
            )
        coverage[name] = tuple(law_by_period)
        lowered_sides = {}
        for phase, side in zip(_PHASES, ("solve", "simulate"), strict=True):
            side_by_period = {
                period: _phase_side(law=law, side=side)
                for period, law in law_by_period.items()
            }
            support[phase][name] = MappingProxyType(
                {
                    period: _declared_support(law=law, all_regimes=user_regimes)
                    for period, law in side_by_period.items()
                }
            )
            lowered_sides[phase] = _lower_side(
                law_by_period=side_by_period, code_by_name=regime_names_to_ids
            )
        transitions[name] = (
            lowered_sides["solution"]
            if lowered_sides["solution"] is lowered_sides["simulation"]
            else Phased(
                solve=lowered_sides["solution"], simulate=lowered_sides["simulation"]
            )
        )
    _fail_if_a_target_is_uncovered(
        support_by_phase=support, coverage_by_regime=coverage, ages=ages
    )
    return RegimeSchedules(
        dated=True,
        coverage_by_regime=MappingProxyType(coverage),
        support_by_phase=MappingProxyType(
            {phase: MappingProxyType(by_regime) for phase, by_regime in support.items()}
        ),
        transitions=MappingProxyType(transitions),
    )


def coverage_nodes(
    *,
    coverage_by_regime: Mapping[RegimeName, tuple[int, ...]],
    ages: AgeGrid,
) -> frozenset[tuple[object, RegimeName]]:
    """The exact `(age, regime)` pair of every covered problem."""
    return frozenset(
        (ages.exact_values[period], name)
        for name, periods in coverage_by_regime.items()
        for period in periods
    )


def resolve_initial_nodes(
    *,
    initial_regimes: object,
    coverage_by_regime: Mapping[RegimeName, tuple[int, ...]],
    ages: AgeGrid,
) -> frozenset[tuple[object, RegimeName]]:
    """Normalize `Model(initial_regimes=...)` to exact permitted entry pairs.

    - `None` ⇒ every covered pair;
    - a regime name or sequence of names ⇒ every covered age of those regimes;
    - a mapping from age selector to names ⇒ the Cartesian pairs of each rule,
      united across rules; every pair must be covered.
    """
    covered = coverage_nodes(coverage_by_regime=coverage_by_regime, ages=ages)
    if initial_regimes is None:
        return covered
    if not isinstance(initial_regimes, Mapping):
        names = _entry_names(initial_regimes)
        _fail_if_unknown_entry_regimes(
            names=names, coverage_by_regime=coverage_by_regime
        )
        return frozenset(pair for pair in covered if pair[1] in names)
    period_by_age: dict[object, int] = {
        age: period for period, age in enumerate(ages.exact_values)
    }
    permitted: set[tuple[object, RegimeName]] = set()
    for selector, value in initial_regimes.items():
        names = _entry_names(value)
        _fail_if_unknown_entry_regimes(
            names=names, coverage_by_regime=coverage_by_regime
        )
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
        pairs = {
            (ages.exact_values[period], name) for period in periods for name in names
        }
        uncovered = sorted(pairs - covered, key=repr)
        if uncovered:
            raise ModelInitializationError(
                f"`initial_regimes` permits entry at {uncovered}, where no problem "
                "is declared. Entry is only possible where a regime is solved."
            )
        permitted |= pairs
    return frozenset(permitted)


def _entry_names(value: object) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value,)
    if isinstance(value, Sequence) and all(isinstance(name, str) for name in value):
        return tuple(value)
    raise ModelInitializationError(
        "`initial_regimes` names regimes by a name or a sequence of names; got "
        f"{value!r}."
    )


def _fail_if_unknown_entry_regimes(
    *, names: tuple[str, ...], coverage_by_regime: Mapping[RegimeName, object]
) -> None:
    unknown = sorted(set(names) - set(coverage_by_regime))
    if unknown:
        raise ModelInitializationError(
            f"`initial_regimes` names unknown regime(s) {unknown}; regimes are "
            f"{sorted(coverage_by_regime)}."
        )


# keyword-only-exempt: primary-argument=func
def _with_signature(
    func: Callable[..., Any], *, names: tuple[str, ...]
) -> Callable[..., Any]:
    """Expose exactly `names` as keyword-only arguments to the DAG machinery."""
    func.__signature__ = _signature(names=names)  # ty: ignore[unresolved-attribute]
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

    return _with_signature(indicator, names=names)


def _period_masked(
    *, cell: Callable[..., Any], periods: tuple[int, ...], names: tuple[str, ...]
) -> Callable[..., Any]:
    """A probability cell that is exactly zero outside `periods`."""

    def period_masked(**kwargs: Any) -> Any:  # noqa: ANN401
        value = cell(**{name: kwargs[name] for name in names})
        selected = jnp.isin(kwargs["period"], jnp.asarray(periods))
        return jnp.where(selected, value, jnp.zeros_like(value))

    return _with_signature(period_masked, names=_with_period(names))


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
    return _with_signature(period_dispatch, names=_with_period(names))


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
        _dispatch(cases=cases, laws=laws, law_by_period=law_by_period),
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
                _period_sum(
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
    return _with_signature(period_sum, names=_with_period(names))


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
            cell, probability=MarkovTransition(_masked(cell=func, periods=periods))
        )
    return MarkovTransition(_masked(cell=cell, periods=periods))


def _law_cells(
    *, law: object, code_by_name: Mapping[str, int] | None
) -> Mapping[str, object]:
    if isinstance(law, Mapping):
        return law
    if isinstance(law, str):
        return {law: MarkovTransition(_constant(1.0))}
    if isinstance(law, Choose):
        names = _argument_names(law.func)
        return {
            target: MarkovTransition(
                _indicator(
                    selector=law.func,
                    code=_code(name=target, code_by_name=code_by_name),
                    names=names,
                )
            )
            for target in law.targets
        }
    raise ModelInitializationError(
        f"A schedule case {law!r} is not a nonterminal regime law. Use a regime "
        "name, `Choose(func, targets=...)`, `MarkovTransition(func, targets=...)` "
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


def _fail_if_legacy_declarations(*, user_regimes: Mapping[RegimeName, Any]) -> None:
    """A dated model reads coverage and support only from the declarations."""
    errors = []
    for name, regime in user_regimes.items():
        if regime.declares_active:
            errors.append(
                f"Regime '{name}' declares `active`. A dated model is solved "
                "where each transition declares a law: wrap the law in `ByAge` "
                "to restrict its ages, e.g. `ByAge({AgeRange(stop=65): law})`."
            )
        transition = regime.transition
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
                        "`MarkovTransition(func, targets=('a', 'b'))`."
                    )
                elif callable(side) and not isinstance(
                    side, Choose | MarkovTransition | Mapping
                ):
                    errors.append(
                        f"Regime '{name}' declares a bare deterministic "
                        "transition. Name its support: "
                        "`Choose(func, targets=('a', 'b'))`."
                    )
    if errors:
        raise ModelInitializationError("\n".join(errors))


def _fail_if_a_target_is_uncovered(
    *,
    support_by_phase: Mapping[
        PhaseKey, Mapping[RegimeName, Mapping[int, tuple[str, ...]]]
    ],
    coverage_by_regime: Mapping[RegimeName, tuple[int, ...]],
    ages: AgeGrid,
) -> None:
    """Every declared target must be solved at the next age."""
    errors = []
    covered = {name: frozenset(periods) for name, periods in coverage_by_regime.items()}
    for phase, by_regime in support_by_phase.items():
        for source, by_period in by_regime.items():
            for period, targets in by_period.items():
                errors.extend(
                    f"Regime '{source}' at age {ages.exact_values[period]} "
                    f"({phase}) declares target '{target}', which is not "
                    f"solved at age {ages.exact_values[period + 1]}. "
                    f"Extend '{target}''s schedule or remove the target "
                    "from this case."
                    for target in targets
                    if period + 1 not in covered[target]
                )
    if errors:
        raise ModelInitializationError("\n".join(sorted(set(errors))))
