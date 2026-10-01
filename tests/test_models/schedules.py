"""Dated regime transitions shared by the test models."""

from collections.abc import Mapping

from _lcm.regime_building.transition_support import (
    _SupportedDeterministicTransition,
    _SupportedStochasticTransition,
)
from lcm import ByAge, DeterministicTransition, Phased, StochasticTransition
from lcm.typing import UserAge


# keyword-only-exempt: primary-argument=boundary
def until_exit(
    boundary: UserAge | float,
    *,
    law: object,
    exits: tuple[str, ...],
    start: UserAge | float | None = None,
    stays: tuple[str, ...] | None = None,
) -> ByAge:
    """Apply `law` up to the age before `boundary`, then only its `exits`.

    The regime is solved at every age in `[start, boundary)`. On the last of
    them only the `exits` targets are declared, so a target that is not solved
    at `boundary` never appears there.

    - per-target mapping ⇒ the exit cells are the mapping's own cells
    - `DeterministicTransition` / vector `StochasticTransition` ⇒ the same
      function over `exits`
    - `Phased` ⇒ each phase's law restricted the same way

    `stays`, if given, restricts the earlier ages to those targets the same
    way, so a deterministic lifecycle declares one target per age.
    """
    return ByAge.until(
        stop_age_exclusive=boundary,
        law=law if stays is None else _restricted(law, exits=stays),
        then=_restricted(law, exits=exits),
        start_age_inclusive=start,
    )


# keyword-only-exempt: primary-argument=law
def _restricted(law: object, *, exits: tuple[str, ...]) -> object:
    if isinstance(law, Phased):
        return Phased(
            solve=_restricted(law.solve, exits=exits),
            simulate=_restricted(law.simulate, exits=exits),
        )
    if isinstance(law, Mapping):
        return {target: law[target] for target in exits}
    if isinstance(law, DeterministicTransition):
        return _SupportedDeterministicTransition(func=law.func, targets=exits)
    if isinstance(law, StochasticTransition):
        return _SupportedStochasticTransition(func=law.func, targets=exits)
    msg = f"No exit restriction for {law!r}."
    raise TypeError(msg)


# keyword-only-exempt: primary-argument=law
def choose_among(law: object, *, targets: tuple[str, ...]) -> object:
    """Declare the support of a deterministic selector, per phase if `Phased`.

    A per-target mapping or a `StochasticTransition` already declares its support
    and is returned unchanged.
    """
    if isinstance(law, Mapping | StochasticTransition):
        return law
    if isinstance(law, Phased):
        return Phased(
            solve=choose_among(law.solve, targets=targets),
            simulate=choose_among(law.simulate, targets=targets),
        )
    return _SupportedDeterministicTransition(func=law, targets=targets)  # ty: ignore[invalid-argument-type]
