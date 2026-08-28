"""Collective-regime models shared by the collective-regime tests.

Every factory returns a model small enough that its whole solution is an exact
small-integer expression, so a test can assert the value itself rather than the
absence of NaN. Each factory's hand computation is stated in its docstring and
the resulting arrays are published as module constants.

The economics are the same throughout: a two-stakeholder household choosing
between work and leisure, where the wife `f` values her own leisure and the
husband `m` values household consumption. The household maximizes the equally
weighted scalarization $O = (Q^f + Q^m) / 2$ and each partner reads off their
OWN value at that shared argmax, so the two stakeholder slices genuinely
differ and neither equals $O$.

A collective regime's stored value function carries one axis per solve state
plus one trailing stakeholder axis. A folded state is integrated out by
quadrature when the value is stored and contributes no axis at all. The two
never co-occur here: a collective regime may not declare a folded state, so
`make_folding_singleton_model` is a singleton and
`make_folding_collective_regime_kwargs` returns keyword arguments rather than a
built model.
"""

from typing import Any

import jax.numpy as jnp

from lcm import (
    AgeGrid,
    CollectiveUtility,
    DiscreteGrid,
    LinSpacedGrid,
    Model,
    NormalIIDProcess,
    Regime,
    categorical,
)
from lcm.typing import ContinuousState, DiscreteAction, FloatND, IntND, ScalarInt

# Nested `{regime: {function: {parameter: value}}}` params, as `solve` takes them.
type ParamsDict = dict[str, dict[str, dict[str, float]]]

# Flat `{state_name: column}` initial conditions, plus the `regime_id` column.
type InitialConditions = dict[str, FloatND | IntND]


@categorical(ordered=True)
class Work:
    """The single binary action every collective regime here offers."""

    leisure: ScalarInt  # code 0
    work: ScalarInt  # code 1


@categorical(ordered=False)
class CoupleRegimeId:
    """Regime ids of every collective model in this module."""

    couple: ScalarInt  # code 0
    couple_terminal: ScalarInt  # code 1


@categorical(ordered=False)
class ShockRegimeId:
    """Regime ids of the folding singleton model."""

    shocked: ScalarInt  # code 0
    shocked_terminal: ScalarInt  # code 1


# Discount factor every factory's params dict supplies.
DISCOUNT_FACTOR = 0.95

# Three periods: the source regime is active at age 0, the terminal from age 1.
AGES = AgeGrid(start=0, stop=2, step="Y")

# The collective regimes' only continuous state.
WAGE_GRID = LinSpacedGrid(start=8.0, stop=40.0, n_points=2)

# `WAGE_GRID`'s two nodes, so initial conditions can name them directly.
WAGE_GRID_POINTS = (8.0, 40.0)

# Payoff of the folding singleton model's stateless terminal regime.
FOLD_TERMINAL_PAYOFF = 4.0

# The folded IID shock of `make_folding_singleton_model`.
FOLDED_SHOCK = NormalIIDProcess(
    n_points=5, gauss_hermite=True, mu=0.0, sigma=2.0, fold=True
)

# `make_two_stakeholder_model`'s value functions, indexed (wage, stakeholder).
TWO_STAKEHOLDER_V_PERIOD_0 = ((46.0, 92.0), (78.0, 156.0))
TWO_STAKEHOLDER_V_TERMINAL = ((30.0, 0.0), (40.0, 80.0))

# `make_stateless_collective_target_model`'s value functions. The terminal one
# carries the stakeholder axis alone, the period-0 one (wage, stakeholder).
STATELESS_TARGET_V_TERMINAL = (10.0, 0.0)
STATELESS_TARGET_V_PERIOD_0 = ((39.5, 0.0), (49.5, 80.0))

# `make_folding_singleton_model`'s period-0 value: rank zero, the shock folded out.
FOLDING_SINGLETON_V_PERIOD_0 = 13.8


def make_two_stakeholder_model(
    *, n_subjects: int | None = None
) -> tuple[Model, ParamsDict]:
    """Build a two-stakeholder collective model over one 2-point wage state.

    `couple` is active at age 0 and transitions with probability one into
    `couple_terminal`, active from age 1 on. Both regimes carry the stakeholders
    `("f", "m")` and the same 2-point `wage` grid, so each stored value function
    has one wage axis plus the trailing stakeholder axis, shape `(2, 2)`.

    Hand computation, wage grid $\\{8, 40\\}$, $\\beta = 0.95$, and
    `next_wage = 40 * work + 8 * (1 - work)`:

    - Terminal, myopic argmax of $O$: at wage 8 leisure wins with $(30, 0)$;
      at wage 40 work wins with $(40, 80)$.
    - Period 0, $Q^s = u^s + 0.95 V'^s$: at wage 8 work wins with $(46, 92)$,
      at wage 40 work wins with $(78, 156)$. The continuation flips the low-wage
      argmax, so a dropped continuation is visible in the value.

    Args:
        n_subjects: Simulate batch size to compile ahead of time, or `None` to
            compile at runtime.

    Returns:
        Tuple of the model and the params dict that solves it.

    """
    couple = Regime(
        transition=_next_couple_regime,
        active=lambda age: age < 1,
        states={"wage": WAGE_GRID},
        state_transitions={"wage": _next_wage},
        actions={"work": DiscreteGrid(Work)},
        functions={
            "utility": CollectiveUtility(utilities={"f": _utility_f, "m": _utility_m})
        },
    )
    couple_terminal = Regime(
        transition=None,
        active=lambda age: age >= 1,
        states={"wage": WAGE_GRID},
        actions={"work": DiscreteGrid(Work)},
        functions={
            "utility": CollectiveUtility(utilities={"f": _utility_f, "m": _utility_m})
        },
    )
    model = Model(
        regimes={"couple": couple, "couple_terminal": couple_terminal},
        ages=AGES,
        regime_id_class=CoupleRegimeId,
        n_subjects=n_subjects,
    )
    return model, _couple_params()


def make_stateless_collective_target_model(
    *, n_subjects: int | None = None
) -> tuple[Model, ParamsDict]:
    """Build a collective model whose only target carries no state.

    `couple` is the same two-stakeholder wage regime as in
    `make_two_stakeholder_model`, but it transitions with probability one into a
    collective terminal regime declaring `states={}`. That target's value
    function is rank one — the stakeholder axis alone — with no state to
    interpolate at, and its value still belongs in the continuation.

    Hand computation, $\\beta = 0.95$:

    - Terminal: work gives $(10, 0)$ with $O = 5$, leisure gives $(0, 5)$ with
      $O = 2.5$, so the terminal value is $(10, 0)$.
    - Period 0, $Q^s = u^s + 0.95 \\cdot (10, 0)^s$: at wage 8 leisure wins with
      $(39.5, 0)$; at wage 40 work wins with $(49.5, 80)$.

    Dropping the continuation entirely would leave the flow payoffs at the same
    argmaxes, $((30, 0), (40, 80))$, which is why the terminal payoff is chosen
    small enough not to move either argmax.

    Args:
        n_subjects: Simulate batch size to compile ahead of time, or `None` to
            compile at runtime.

    Returns:
        Tuple of the model and the params dict that solves it.

    """
    couple = Regime(
        transition=_next_couple_regime,
        active=lambda age: age < 1,
        states={"wage": WAGE_GRID},
        state_transitions={"wage": _next_wage},
        actions={"work": DiscreteGrid(Work)},
        functions={
            "utility": CollectiveUtility(utilities={"f": _utility_f, "m": _utility_m})
        },
    )
    couple_terminal = Regime(
        transition=None,
        active=lambda age: age >= 1,
        states={},
        actions={"work": DiscreteGrid(Work)},
        functions={
            "utility": CollectiveUtility(
                utilities={"f": _stateless_utility_f, "m": _stateless_utility_m}
            )
        },
    )
    model = Model(
        regimes={"couple": couple, "couple_terminal": couple_terminal},
        ages=AGES,
        regime_id_class=CoupleRegimeId,
        n_subjects=n_subjects,
    )
    return model, _couple_params()


def make_folding_singleton_model(
    *, n_subjects: int | None = None
) -> tuple[Model, ParamsDict]:
    """Build a singleton model whose only state is a folded IID shock.

    `shocked` declares `wage_shock` as a `NormalIIDProcess(fold=True)` that
    enters only its own utility, so the quadrature is taken when the value is
    stored and the stored value function is rank zero. `shocked_terminal`
    declares no state and pays a constant, so the continuation is live.

    Hand computation: work pays $10 + \\varepsilon$ and leisure pays nothing, and
    $10 + \\varepsilon > 0$ at every node of a $\\sigma = 2$ shock, so work is
    chosen everywhere and the period-0 value is
    $E[10 + \\varepsilon] + 0.95 \\cdot 4 = 13.8$.

    Args:
        n_subjects: Simulate batch size to compile ahead of time, or `None` to
            compile at runtime.

    Returns:
        Tuple of the model and the params dict that solves it.

    """
    shocked = Regime(
        transition=_next_shock_regime,
        active=lambda age: age < 1,
        states={"wage_shock": FOLDED_SHOCK},
        actions={"work": DiscreteGrid(Work)},
        functions={"utility": _shock_utility},
    )
    shocked_terminal = Regime(
        transition=None,
        active=lambda age: age >= 1,
        functions={"utility": _fold_terminal_utility},
    )
    model = Model(
        regimes={"shocked": shocked, "shocked_terminal": shocked_terminal},
        ages=AGES,
        regime_id_class=ShockRegimeId,
        n_subjects=n_subjects,
    )
    params: ParamsDict = {
        "shocked": {"koopmans_aggregator": {"discount_factor": DISCOUNT_FACTOR}},
        "shocked_terminal": {},
    }
    return model, params


def make_folding_collective_regime_kwargs() -> dict[str, Any]:
    """Return the keyword arguments of a collective regime declaring a fold.

    A collective regime may not integrate a shock out of its stored value, so
    this combination is rejected when the model is built and no model can be
    handed out. The regime itself is well formed: two stakeholders, a binary
    action, and `wage_shock` as a `NormalIIDProcess(fold=True)` entering both
    partners' utilities.

    `make_folding_collective_regimes` pairs the regime these arguments describe
    with the terminal regime it transitions into, which is what `Model` needs.

    Returns:
        Dict of the keyword arguments `Regime` takes.

    """
    return {
        "transition": _next_couple_regime,
        "active": lambda age: age < 1,
        "stakeholders": ("f", "m"),
        "states": {"wage_shock": FOLDED_SHOCK},
        "actions": {"work": DiscreteGrid(Work)},
        "functions": {
            "utility_f": _shock_utility_f,
            "utility_m": _shock_utility_m,
        },
    }


def make_folding_collective_regimes() -> dict[str, Regime]:
    """Return the regimes of a model whose collective source declares a fold.

    Pass straight to `Model(regimes=..., ages=AGES,
    regime_id_class=CoupleRegimeId)`, which is where the combination is
    rejected.

    Returns:
        Dict of regime names to regimes: the collective source `couple` built
        from `make_folding_collective_regime_kwargs`, and the stateless
        collective terminal `couple_terminal` it transitions into.

    """
    couple_terminal = Regime(
        transition=None,
        active=lambda age: age >= 1,
        states={},
        actions={"work": DiscreteGrid(Work)},
        functions={
            "utility": CollectiveUtility(
                utilities={"f": _stateless_utility_f, "m": _stateless_utility_m}
            )
        },
    )
    return {
        "couple": Regime(**make_folding_collective_regime_kwargs()),
        "couple_terminal": couple_terminal,
    }


def folded_shock_nodes() -> FloatND:
    """Return `FOLDED_SHOCK`'s quadrature nodes at the ambient float precision.

    `make_folding_singleton_initial_conditions` seeds subject `i` at node
    `i % 5`, so a test computing that subject's expected utility reads the node
    from here rather than hardcoding a value only one precision can represent.

    Returns:
        Array of the five nodes, in the process's own order.

    """
    return FOLDED_SHOCK.to_jax()


def make_couple_initial_conditions(*, n_subjects: int = 2) -> InitialConditions:
    """Build initial conditions for the collective models in this module.

    Every subject starts at age 0 in `couple`, with wages cycling through
    `WAGE_GRID_POINTS` so that consecutive subjects sit on different nodes.

    Args:
        n_subjects: Number of subjects to simulate.

    Returns:
        Dict of `wage`, `age` and `regime_id` columns, each of length
        `n_subjects`.

    """
    wages = [
        WAGE_GRID_POINTS[index % len(WAGE_GRID_POINTS)] for index in range(n_subjects)
    ]
    return {
        "wage": jnp.asarray(wages),
        "age": jnp.zeros(n_subjects),
        "regime_id": jnp.full(n_subjects, CoupleRegimeId.couple, dtype=jnp.int32),
    }


def make_folding_singleton_initial_conditions(
    *, n_subjects: int = 2
) -> InitialConditions:
    """Build initial conditions for `make_folding_singleton_model`.

    Every subject starts at age 0 in `shocked`, seeded at a node of
    `FOLDED_SHOCK` — subject `i` at `folded_shock_nodes()[i % 5]` — so a test
    can state each subject's realized utility from the node it was given.

    Args:
        n_subjects: Number of subjects to simulate.

    Returns:
        Dict of `wage_shock`, `age` and `regime_id` columns, each of length
        `n_subjects`.

    """
    nodes = folded_shock_nodes()
    return {
        "wage_shock": jnp.asarray(
            [nodes[index % len(nodes)] for index in range(n_subjects)]
        ),
        "age": jnp.zeros(n_subjects),
        "regime_id": jnp.full(n_subjects, ShockRegimeId.shocked, dtype=jnp.int32),
    }


def _couple_params() -> ParamsDict:
    """Return the params dict every `couple` / `couple_terminal` model takes."""
    return {
        "couple": {"koopmans_aggregator": {"discount_factor": DISCOUNT_FACTOR}},
        "couple_terminal": {},
    }


def _utility_f(wage: ContinuousState, work: DiscreteAction) -> FloatND:
    """Wife: values her own leisure highly, also sees household consumption."""
    return wage * work + 30.0 * (1.0 - work)


def _utility_m(wage: ContinuousState, work: DiscreteAction) -> FloatND:
    """Husband: values household consumption, indifferent to leisure."""
    return 2.0 * (wage * work)


def _stateless_utility_f(work: DiscreteAction) -> FloatND:
    """Wife's payoff in the stateless terminal regime: 10 for work, 0 for leisure."""
    return 10.0 * work


def _stateless_utility_m(work: DiscreteAction) -> FloatND:
    """Husband's payoff in the stateless terminal regime: 5 for leisure, 0 for work."""
    return 5.0 * (1.0 - work)


def _next_wage(work: DiscreteAction) -> ContinuousState:
    """Deterministic wage law: working today yields the high wage tomorrow."""
    return 40.0 * work + 8.0 * (1.0 - work)


def _next_couple_regime() -> ScalarInt:
    """Regime transition: `couple` becomes `couple_terminal` with probability one."""
    return CoupleRegimeId.couple_terminal


def _shock_utility(wage_shock: FloatND, work: DiscreteAction) -> FloatND:
    """Working earns the base wage plus the shock; leisure earns nothing."""
    return work * (10.0 + wage_shock)


def _shock_utility_f(wage_shock: FloatND, work: DiscreteAction) -> FloatND:
    """Wife: the shocked wage when working, her leisure value otherwise."""
    return work * (10.0 + wage_shock) + 30.0 * (1.0 - work)


def _shock_utility_m(wage_shock: FloatND, work: DiscreteAction) -> FloatND:
    """Husband: twice the shocked wage when working, nothing otherwise."""
    return 2.0 * work * (10.0 + wage_shock)


def _fold_terminal_utility() -> FloatND:
    """Constant terminal payoff, so the folded regime has a live continuation."""
    return jnp.asarray(FOLD_TERMINAL_PAYOFF)


def _next_shock_regime() -> ScalarInt:
    """Regime transition: `shocked` becomes `shocked_terminal` with probability one."""
    return ShockRegimeId.shocked_terminal
