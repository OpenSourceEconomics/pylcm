"""The `Decoupled` declaration for the Koopmans aggregator slot.

A thin leaf module, like `lcm.phased`: the class definition only, so the
user-facing `Regime`, the regime validators and the regime-building code can
all import it without an import cycle.
"""

from typing import Final, Literal

from beartype import beartype

from _lcm.beartype_conf import REGIME_CONF
from lcm.exceptions import RegimeInitializationError
from lcm.phased import Phased
from lcm.typing import UserFunction


class Decoupled:
    """Choose actions with one aggregator and store the value of another.

    Pass it as a regime's `koopmans_aggregator`. Both functions take the
    Koopmans aggregator's arguments, `W(utility, CE, ...)`, and see the same
    period utility and certainty-equivalent continuation:

    - `objective` ranks the actions: the action taken maximizes it over the
      feasible set, or, with taste shocks, it sets the choice probabilities.
      It may be `Phased`, so the solve and simulation can choose differently.
    - `value` is stored at the chosen action and passed back; simulation and
      policy lookup publish it. With taste shocks the stored value is its
      expectation at the noisy choice, including the chosen action's shock.

    Quasi-hyperbolic (naive, sophisticated and partially naive) agents are the
    leading use; `docs/methods/preferences.md` states the recipe and its
    caveats, and the beta-delta explanation notebook works it through.
    """

    __slots__ = ("objective", "value")

    @beartype(conf=REGIME_CONF)
    def __init__(
        self,
        *,
        objective: UserFunction | Phased[UserFunction, UserFunction],
        value: UserFunction,
    ) -> None:
        _fail_if_an_objective_variant_is_not_an_aggregator(objective=objective)
        self.objective: Final[UserFunction | Phased[UserFunction, UserFunction]] = (
            objective
        )
        self.value: Final[UserFunction] = value

    def __repr__(self) -> str:
        return f"Decoupled(objective={self.objective!r}, value={self.value!r})"


def _fail_if_an_objective_variant_is_not_an_aggregator(
    *, objective: UserFunction | Phased[UserFunction, UserFunction]
) -> None:
    """Refuse a `Phased` objective whose variants are not plain aggregators."""
    if isinstance(objective, Phased) and any(
        isinstance(variant, Decoupled) or not callable(variant)
        for variant in (objective.solve, objective.simulate)
    ):
        msg = "`Decoupled(objective=Phased(...))` needs an aggregator for each phase."
        raise RegimeInitializationError(msg)


type KoopmansDeclaration = (
    UserFunction | Phased[UserFunction, UserFunction] | Decoupled | None
)


def resolve_koopmans_aggregator(
    *, declaration: KoopmansDeclaration, phase: Literal["solve", "simulate"]
) -> tuple[UserFunction | None, UserFunction | None]:
    """Return the value aggregator and the objective aggregator one phase runs.

    The objective aggregator is `None` unless the slot declares a
    `Decoupled`; then the value aggregator also chooses.

    Args:
        declaration: The regime's `koopmans_aggregator` slot.
        phase: Which variant a `Phased` part resolves to.

    Returns:
        The value aggregator, or `None` when the slot is empty, and the
        objective aggregator, or `None`.

    """
    if isinstance(declaration, Decoupled):
        return declaration.value, _phase_variant(
            value=declaration.objective, phase=phase
        )
    if declaration is None:
        return None, None
    return _phase_variant(value=declaration, phase=phase), None


def koopmans_aggregator_variants(
    *,
    declaration: KoopmansDeclaration,
    phase: Literal["solve", "simulate"] | None = None,
) -> tuple[UserFunction, ...]:
    """Return every aggregator the slot runs, in one phase or in both.

    Each callable appears once, in the order value, solve objective, simulate
    objective, so a consumer that resolves a parameter by the first variant
    declaring it reads the value aggregator first.

    Args:
        declaration: The regime's `koopmans_aggregator` slot.
        phase: Restrict to the variants this phase runs; `None` for both.

    Returns:
        Tuple of the distinct aggregators.

    """
    phases: tuple[Literal["solve", "simulate"], ...] = (
        ("solve", "simulate") if phase is None else (phase,)
    )
    found: dict[int, UserFunction] = {}
    for current in phases:
        value, objective = resolve_koopmans_aggregator(
            declaration=declaration, phase=current
        )
        for func in (value, objective):
            if func is not None:
                found.setdefault(id(func), func)
    return tuple(found.values())


def _phase_variant(
    *,
    value: UserFunction | Phased[UserFunction, UserFunction],
    phase: Literal["solve", "simulate"],
) -> UserFunction:
    if isinstance(value, Phased):
        return value.solve if phase == "solve" else value.simulate
    return value
