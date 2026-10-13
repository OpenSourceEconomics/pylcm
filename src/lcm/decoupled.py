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
    - `value` gives the value stored and passed back: the solve stores `value`
      at the chosen action, and simulation and policy lookup publish it.

    With taste shocks the stored value is the expected `value` at the choice,
    including the chosen action's shock. Under extreme-value shocks of scale
    `scale` this is
    `scale * logsumexp(Q_objective / scale) + sum_k P_k (Q_value_k - Q_objective_k)`
    with `P = softmax(Q_objective / scale)`.

    `objective` may be `Phased`: its `solve` variant is how the selves the solve
    anticipates choose, its `simulate` variant how the acting self chooses.
    `value` is the same in both phases.

    The leading use is quasi-hyperbolic discounting. With
    `value = u + δ·CE`, a solve variant `u + β̃δ·CE` and a simulate variant
    `u + βδ·CE`, `β̃ = 1` is the naive agent, `β̃ = β` the sophisticated agent
    and `β < β̃ < 1` the partially naive agent. The stored value is then the
    long-run value of the policy the agent expects its future selves to follow.
    For naive and partially naive agents it is the agent's self-assessment;
    the policy it actually follows has a different long-run value.

    The stored value is read off at the objective's choice, away from the
    value's own maximum, so grid error in the chosen action enters it at first
    order. Expect to need finer grids than for an ordinary aggregator.

    A `Phased` Koopmans aggregator, `Phased(solve=f, simulate=g)`, is the
    special case `Decoupled(objective=Phased(solve=f, simulate=g), value=f)`
    for the choices it makes; it publishes `g` instead of `f` in simulation.
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
