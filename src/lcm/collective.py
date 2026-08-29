"""Declarations a regime makes about collective and value-dependent choice.

Five objects, each declared inside a slot the regime already has: the
stakeholders and their trade-off in `functions["utility"]`, a value-reading
feasibility constraint in `constraints`, and a value-dependent route in
`transition`. `ProjectedRegimeValue` is the one thing the last two share — a
reading of another regime's value in the same period.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Literal, cast

from beartype import beartype

from _lcm.beartype_conf import REGIME_CONF
from _lcm.typing import RegimeName, StateName
from _lcm.utils.containers import ensure_containers_are_immutable
from lcm.exceptions import RegimeInitializationError
from lcm.phased import Phased
from lcm.transition import MarkovTransition
from lcm.typing import UserFunction


@beartype(conf=REGIME_CONF)
@dataclass(frozen=True, kw_only=True)
class ParetoObjective:
    r"""The household's Pareto scalarization of its stakeholders' action values.

    A collective regime takes one action for everybody, and this declares how
    the stakeholders' action values are traded off in taking it:

    ```{math}
        a^*(x) = \arg\max_{a\,:\,F(x,a)} \sum_s \lambda_s(x)\, Q^s(x, a),
        \qquad V^s(x) = Q^s(x, a^*(x)).
    ```

    Declaring it rather than writing the sum as an ordinary function is what
    lets the engine own what a Pareto weight means: one per stakeholder, finite
    and non-negative, with a strictly positive total, normalized cell by cell,
    and multiplied in zero-safely — so a stakeholder carrying no weight cannot
    decide the household's choice through an admissible `-inf` of her own.

    Omit it (the regime's default) for equal weights.
    """

    weights: Mapping[str, UserFunction | float]
    r"""One weight $\lambda_s$ per stakeholder, keyed by stakeholder name.

    A `float` is a constant. A callable is a function of the regime's STATES
    and of `period` / `age`; every other argument it names becomes a free
    scalar parameter under the regime's `pareto_objective` key in
    `get_params_template()`, so a weight is estimated like anything else. Note
    what that means for a name collision: an argument spelled like one of the
    regime's other functions does NOT receive that function's output — it
    becomes a parameter you must supply. A weight may not read an action: a
    weight that varies with the choice states a different objective per
    candidate, whose maximizer is a Pareto optimum of no fixed weighting.
    """

    normalization: str = "pointwise"
    """How the declared weights are turned into the weights actually used.

    - `"pointwise"` (the default) divides by the total at each cell, so the
      weights sum to one wherever the objective is evaluated and a
      state-dependent declaration keeps one scale across the grid.
    - `"none"` uses the declared weights as they stand. The scalarization is
      then not on the stakeholders' own scale, and comparing values across
      cells whose totals differ compares different objectives.
    """

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "weights", ensure_containers_are_immutable(self.weights)
        )
        if self.normalization not in {"pointwise", "none"}:
            msg = (
                f"`ParetoObjective.normalization` is {self.normalization!r}, "
                'which is neither "pointwise" nor "none".'
            )
            raise ValueError(msg)


@beartype(conf=REGIME_CONF)
@dataclass(frozen=True, kw_only=True)
class ProjectedRegimeValue:
    """Another regime's CURRENT-period value, read at mapped state coordinates.

    The reference regime is solved earlier in the same period — the solver
    orders each period's active regimes topologically by these declarations —
    and its value function is interpolated at the projected coordinates with
    the machinery the continuation uses, but on the current period's arrays.
    Reading the current period rather than the continuation is what a
    within-period participation constraint needs: a couple's period-$t$
    decision is checked against the values its members would have as singles in
    that same period $t$.

    Where the declaration sits fixes what its projection may read:

    - inside `ValueDependentConstraint.references` it projects from the
      DECLARING regime's state cell, receives that regime's `period` / `age`,
      and may introduce no free parameters;
    - inside `ValueDependentTransition` — as a `gate_references` entry or a
      route's `fallback` — it projects from the TARGET regime's grid, receives
      that target fold's `period` / `age`, and its free arguments are collected
      as that edge's own parameters.
    """

    regime: RegimeName
    """Name of the reference regime whose same-period value is read.

    Must be another regime of the model, active in every period the declaring
    regime is active. No transition edge between the two is required — a
    reference read works across otherwise unconnected regime "islands", which
    is what lets a constraint compare this regime's value against a regime it
    never transitions into.
    """

    projection: Mapping[StateName, UserFunction]
    """One entry per state of the REFERENCE regime: `state name -> function`.

    The reference value is interpolated at the resulting coordinates — linear
    on continuous axes, lookup on discrete axes.
    """

    stakeholder: str | None = None
    """Which stakeholder's value to read from a collective reference regime.

    Required when the reference regime is collective (its value carries a
    stakeholder axis); must be `None` when it is a singleton.
    """

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "projection", ensure_containers_are_immutable(self.projection)
        )


@beartype(conf=REGIME_CONF)
@dataclass(frozen=True, kw_only=True)
class CollectiveUtility:
    """A collective regime's stakeholders and how their values are traded off.

    Declared as the regime's `functions={"utility": CollectiveUtility(...)}`,
    which is what makes the regime collective: the `utilities` keys ARE its
    stakeholders, in insertion order, and that order fixes the trailing
    stakeholder axis of the regime's value function and of every published
    array.

    The household takes one action for everybody. It maximizes `objective`
    over the feasible actions and reads off each stakeholder's own action value
    at that common choice.
    """

    utilities: Mapping[str, UserFunction | Phased | None]
    """One flow-utility per stakeholder, keyed by stakeholder name.

    Three ways to name a stakeholder's utility, and the keys are the household
    either way:

    - a function — the ordinary case;
    - a `Phased` — the utility the household is solved against differs from the
      one simulation realizes;
    - `None` — the body arrives from elsewhere, under the regime's own
      `utility_<s>` entry. That is what lets a model declare a common utility
      at the model level and still name its households here; a stakeholder left
      undeclared in both places is an error naming the entry that is missing.
    """

    objective: ParetoObjective | None = None
    """How the stakeholders' action values are scalarized into the household's.

    `None` (the default) weights them equally. A `ParetoObjective` declares the
    weights, and declaring them rather than writing the sum as an ordinary
    function is what lets the engine own what a Pareto weight means — one per
    stakeholder, finite and non-negative, with a strictly positive total,
    normalized cell by cell, and multiplied in zero-safely.
    """

    def __post_init__(self) -> None:
        if not self.utilities:
            raise RegimeInitializationError(
                "A `CollectiveUtility` declares a household, so it needs at "
                "least one stakeholder: `utilities` may not be empty."
            )
        object.__setattr__(
            self, "utilities", ensure_containers_are_immutable(self.utilities)
        )


@beartype(conf=REGIME_CONF)
@dataclass(frozen=True, kw_only=True)
class ValueDependentConstraint:
    """A feasibility constraint that may read values, not only states.

    Declared inside the regime's `constraints`, beside the ordinary callables,
    so there is one constraint slot rather than two. The regime's mask is the
    AND of both kinds, and a cell whose mask is empty publishes the regime's
    dissolution flag.

    A participation constraint is the motivating case: a couple's cell is
    feasible only where each partner is at least as well off inside the
    household as at the outside option that partner's own regime offers in the
    same period.
    """

    predicate: UserFunction
    """Returns `True` where the cell is feasible.

    May read `Q_<s>` for each stakeholder of the declaring regime, each key of
    this constraint's OWN `references`, and ordinary states, actions,
    functions and parameters.
    """

    references: Mapping[str, ProjectedRegimeValue] = field(
        default_factory=lambda: MappingProxyType({})
    )
    """The same-period reference values `predicate` reads, keyed by the name
    each one enters the predicate under."""

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "references", ensure_containers_are_immutable(self.references)
        )


@beartype(conf=REGIME_CONF)
@dataclass(frozen=True, kw_only=True)
class StakeholderRoute:
    """Where one source stakeholder goes on each branch of a gated transition.

    A route owns four destinations, and simulation carries all four: the open
    branch's regime (the transition's own key) and role
    (`target_stakeholder`), and the closed branch's regime and role
    (`fallback.regime` and `fallback.stakeholder`). A row landing in a
    singleton regime carries no role.
    """

    fallback: ProjectedRegimeValue | Phased
    """The gate-closed branch: a reference regime's same-period value at a
    projection from the TARGET regime's grid.

    A bare `ProjectedRegimeValue` is both what the branch is worth and where it
    puts a row. `Phased(solve=..., simulate=...)` separates them, because what
    a household expects from leaving and what a settlement hands it are two
    objects:

    - the SOLVE leg prices the source's decision;
    - the SIMULATE leg supplies the regime, the role and the state coordinates
      a routed row actually lands on.

    Both sides must be `ProjectedRegimeValue`, and each is validated against
    the phase that reads it.
    """

    target_stakeholder: str | None = None
    """The gate-open branch's role — one of the target's stakeholders, or
    `None` for a singleton target."""

    @property
    def solve_fallback(self) -> ProjectedRegimeValue:
        """The reference whose value the gate-closed branch is priced at."""
        return (
            cast("ProjectedRegimeValue", self.fallback.solve)
            if isinstance(self.fallback, Phased)
            else self.fallback
        )

    @property
    def simulate_fallback(self) -> ProjectedRegimeValue:
        """The reference a routed row's regime, role and states come from."""
        return (
            cast("ProjectedRegimeValue", self.fallback.simulate)
            if isinstance(self.fallback, Phased)
            else self.fallback
        )

    @property
    def fallback_is_phased(self) -> bool:
        """Whether the two branches were declared separately."""
        return isinstance(self.fallback, Phased)


@beartype(conf=REGIME_CONF)
@dataclass(frozen=True, kw_only=True)
class ValueDependentTransition:
    """A transition into one target whose branch depends on values there.

    Declared inside the regime's `transition`, keyed by target regime name, so
    target selection and value-dependent routing are one declaration of one
    semantic transition rather than two.

    **The key is always the GATE-OPEN target** — the regime a row enters when
    the gate is true. A dissolution edge is therefore keyed by the CONTINUING
    collective regime under `gate = ~D_target`, with each partner's own regime
    as that partner's route fallback; keying it by one partner's regime would
    send both partners there whenever the couple stays together.

    `probability` and `gate` are two distinct operations: the first selects
    whether this target edge is attempted at all, the second keeps that target
    or takes the route's stakeholder-specific fallback.

    This is what unlocks mixed singleton/collective topologies — a singleton
    regime reaching a collective one under mutual consent, a collective regime
    routing per stakeholder into singleton ones on dissolution. A raw
    transition between regimes of different stakeholder structure stays
    rejected.
    """

    probability: UserFunction | MarkovTransition
    """Exactly what a plain `transition` entry for this target accepts."""

    gate: UserFunction
    """Boolean predicate on the TARGET regime's grid, in the target fold's context.

    May read the target's value — `V_target` for a singleton target,
    `V_target_<s>` per stakeholder for a collective one — the target's
    dissolution flag `D_target` (a collective target only; reading it on a
    singleton target is rejected while the model is built), each key of
    `gate_references`, ordinary target states and params, and the target fold's
    `period` / `age`. Mutual consent is the strict, unanimous gate
    `(V_target_f > V_single_f) & (V_target_m > V_single_m)`; "no dissolution
    this period" is `~D_target`. A gate returning a probability rather than a
    Boolean is rejected when the gate is evaluated, i.e. on the first `solve()`
    rather than at model build: the branch is selected with a strict `where`,
    in which every nonzero value is true.
    """

    routes: Mapping[str, StakeholderRoute]
    """One route per SOURCE stakeholder, keyed by stakeholder name.

    A singleton source declares exactly one route, under any key.
    """

    gate_references: Mapping[str, ProjectedRegimeValue] = field(
        default_factory=lambda: MappingProxyType({})
    )
    """The same-period reference values `gate` reads, projected from the target
    regime's grid."""

    off_grid: Literal["pointwise", "reject"] = "pointwise"
    """What the edge promises about a landing point between the target's nodes.

    - `"pointwise"` (the default) reads every operand at the landing point and
      applies the gate there, in both phases. The operands are interpolated, so
      the value carries the ordinary interpolation error of any continuation —
      but it is a value one branch really delivers, and the branch the solve
      priced is the branch simulation routes down.
    - `"reject"` demands that no such point exists: the model refuses to build
      unless the target regime's grid is reached exactly, i.e. it carries no
      continuous state. Declare it where a straddled gate would be an economic
      error rather than an approximation.
    """

    def __post_init__(self) -> None:
        object.__setattr__(self, "routes", ensure_containers_are_immutable(self.routes))
        object.__setattr__(
            self,
            "gate_references",
            ensure_containers_are_immutable(self.gate_references),
        )
