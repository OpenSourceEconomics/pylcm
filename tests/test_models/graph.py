"""Translate older domain fixtures into explicit graphs for the public Model.

This is test authoring machinery, never a public construction fallback. Public
graph contract regressions call lcm.Model directly.
"""

from collections.abc import Mapping
from typing import Any

from lcm import Model, Phased
from lcm.ages import AgeGrid
from lcm.collective import ValueDependentTransition
from lcm.regime import Regime
from lcm.transition import (
    AgeRange,
    ByAge,
    DeterministicTransition,
    StochasticTransition,
)
from lcm.typing import UserAge


def _selected(*, selector: object, age: UserAge | float) -> bool:
    """Select metadata only; malformed selectors remain the Model's errors."""
    if isinstance(selector, AgeRange):
        return (selector.start is None or age >= selector.start) and (
            selector.exclusive_stop is None or age < selector.exclusive_stop
        )
    if isinstance(selector, tuple | range):
        return age in selector
    return age == selector


def _laws_at(*, law: object, age: UserAge | float, ages: AgeGrid) -> tuple[object, ...]:
    """Read the raw fixture schedule without prevalidating it."""
    if not isinstance(law, ByAge):
        return (law,)
    until = law._until
    if until is not None:
        values = ages.exact_values
        if until.stop_age_exclusive not in values:
            return (until.law, until.then)  # Model owns the invalid boundary.
        stop = values.index(until.stop_age_exclusive)
        period = values.index(age)
        if until.start_age_inclusive is not None and age < until.start_age_inclusive:
            return ()
        return (
            (until.then,)
            if period == stop - 1
            else ((until.law,) if period < stop - 1 else ())
        )
    selected = tuple(
        variant
        for selector, variant in law._cases
        if _selected(selector=selector, age=age)
    )
    default = law._default
    return selected or (() if default is None else (default,))


def _targets(
    *, law: object, age: UserAge | float, names: tuple[str, ...], side: str
) -> tuple[str, ...]:
    if isinstance(law, Phased):
        return _targets(law=getattr(law, side), age=age, names=names, side=side)
    if law is None:
        return ()
    if isinstance(law, str):
        return (law,)
    if isinstance(law, Mapping):
        targets = list(law)
        for cell in law.values():
            if isinstance(cell, ValueDependentTransition):
                targets.extend(
                    getattr(route, f"{side}_fallback").regime
                    for route in cell.routes.values()
                )
        return tuple(dict.fromkeys(targets))
    support = getattr(law, "targets", None)
    if isinstance(support, Mapping):
        return tuple(
            target
            for target, selector in support.items()
            if _selected(selector=selector, age=age)
        )
    return tuple(support) if support is not None else names


def _public_kernel(law: object) -> object:
    if isinstance(law, ByAge):
        return law.with_mapped_laws(func=_public_kernel)
    if isinstance(law, Phased):
        return Phased(
            solve=_public_kernel(law.solve), simulate=_public_kernel(law.simulate)
        )
    if isinstance(law, Mapping):
        return {target: _public_kernel(cell) for target, cell in law.items()}
    if isinstance(law, DeterministicTransition):
        return DeterministicTransition(func=law.func)
    if isinstance(law, StochasticTransition):
        return StochasticTransition(func=law.func, fixed_component=law.fixed_component)
    return law


def with_fixture_graph(
    *, regimes: Mapping[str, Regime], ages: AgeGrid, **kwargs: Any
) -> Model:
    """Materialize legacy fixture support, then call Model with explicit edges."""
    if (
        not isinstance(regimes, Mapping)
        or not isinstance(ages, AgeGrid)
        or any(not isinstance(regime, Regime) for regime in regimes.values())
    ):
        task_options: dict[str, Any] = {"edges": {}} | kwargs
        return Model(regimes=regimes, ages=ages, **task_options)
    public_regimes = {
        name: regime.replace(
            regime_transitions=_public_kernel(regime.regime_transitions)
        )
        for name, regime in regimes.items()
    }
    if "edges" in kwargs:
        return Model(regimes=public_regimes, ages=ages, **kwargs)
    names = tuple(regimes)
    by_phase = {}
    for side in ("solve", "simulate"):
        edges: dict[str, dict[str, tuple[UserAge | float, ...]]] = {}
        for name, regime in regimes.items():
            by_target: dict[str, list[UserAge | float]] = {}
            for age in ages.exact_values[:-1]:
                for law in _laws_at(law=regime.regime_transitions, age=age, ages=ages):
                    for target in _targets(law=law, age=age, names=names, side=side):
                        by_target.setdefault(target, []).append(age)
            if by_target:
                edges[name] = {
                    target: tuple(dict.fromkeys(selected))
                    for target, selected in by_target.items()
                }
        by_phase[side] = edges
    return Model(
        regimes=public_regimes,
        ages=ages,
        edges=Phased(solve=by_phase["solve"], simulate=by_phase["simulate"]),
        **kwargs,
    )
