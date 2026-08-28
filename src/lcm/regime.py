"""The user-facing `Regime` definition.

The validators and the identity transition live behind a leading underscore in
`_lcm.user_regime_validation` and `_lcm.regime_building.transitions`. This
module is intentionally thin: the public class definition. A non-terminal
regime that declares no `koopmans_aggregator` takes the model-level one at
model build.

"""

import dataclasses
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, ClassVar, Literal, cast

from beartype import beartype

import lcm.solvers as _solvers
from _lcm.beartype_conf import REGIME_CONF
from _lcm.constraints.processed import ConstraintLike
from _lcm.grids import DiscreteGrid, Grid
from _lcm.regime_building.phases import normalize_regime_phases
from _lcm.regime_building.transitions import collect_state_transitions
from _lcm.typing import ActionName, ActiveFunction, FunctionName, RegimeName, StateName
from _lcm.user_regime_validation import (
    _validate_collective_regime,
    _validate_fold_declarations,
    _validate_gated_edges,
    _validate_logical_consistency,
    _validate_mapping_contents,
)
from _lcm.utils.containers import ensure_containers_are_immutable
from lcm.certainty_equivalent import CertaintyEquivalent
from lcm.collective import (
    CollectiveUtility,
    ParetoObjective,
    ProjectedRegimeValue,
    StakeholderRoute,
    ValueDependentConstraint,
    ValueDependentTransition,
)
from lcm.exceptions import RegimeInitializationError
from lcm.phased import Phased
from lcm.taste_shocks import ExtremeValueTasteShocks
from lcm.transition import AgeSpecializedGrid, JointTransition, MarkovTransition
from lcm.typing import UserFunction


@beartype(conf=REGIME_CONF)
@dataclass(frozen=True, kw_only=True)
class GatedEdge:
    """A gated edge routing a source regime's continuation into a target.

    The construct that unlocks MIXED singleton/collective
    regime topologies: a singleton regime may reach a collective regime (mutual
    consent marriage) and a collective regime may route per-stakeholder to
    singleton regimes (dissolution) — but only THROUGH a declared gated edge. Direct
    raw transitions between different-stakeholder regimes stay rejected.

    A source regime declares `gated_edges` as a mapping of TARGET regime name
    to `GatedEdge`. **The key is always the GATE-OPEN target** — the regime a
    row enters when the gate is true — and each leg's `fallback` is where the
    gate-false branch sends it. A dissolution edge is therefore keyed by the
    CONTINUING collective regime under `gate = ~D_target`, with each partner's
    single regime as that partner's leg fallback; keying it by one partner's
    single regime would send both partners there whenever the couple stays
    together.

    A leg thus owns four facts about where its rows go, and simulation carries
    all four: the open branch's regime (the key) and role
    (`StakeholderRoute.target_stakeholder`), and the closed branch's regime and role
    (`fallback.regime` and `fallback.stakeholder`). A row's own role —
    `initial_conditions["own_stakeholder"]`, published in the simulated frame —
    is what picks its leg, and it is updated to the branch's role as the row
    moves; a row landing in a singleton regime carries none.

    At the end of each period's solve, the engine folds, for
    each declared edge and each source stakeholder `s`, a gated continuation
    object on the target regime's grid:

        Wbar^s(x) = jnp.where(gate(x), V_target^{leg_s}(x), V_fallback^s(pi_s(x)))

    The source's continuation then reads `Wbar` in place of the raw target V,
    threaded through the ordinary transition machinery.

    - `gate` — a BOOLEAN user function evaluated pointwise on the target
      regime's grid. It may read the target regime's per-stakeholder value
      components under the names `V_target_<s>` (one per target stakeholder),
      the target's dissolution flag `D_target` (a collective target only), each
      key of `gate_refs` (a same-period reference value at its projection), and
      ordinary target states / params, plus the target fold's engine context
      (`period` / `age`). Mutual consent to marriage is the strict, unanimous gate
      `gate = (V_target_f > V_single_f_ref) & (V_target_m > V_single_m_ref)`;
      "no dissolution this period" is `gate = ~D_target`. Stochastic gates —
      a gate returning a probability in `(0, 1)` rather than a boolean — are
      not supported.
    - `legs` — one `StakeholderRoute` per SOURCE stakeholder (keyed by source
      stakeholder name; a singleton source declares exactly one leg under any
      key).
    - `gate_refs` — extra same-period reference values the `gate` reads,
      exactly like a regime's `same_period_refs` but projected from the TARGET
      regime's grid.
    - `off_grid` — what the edge promises about a landing point between the
      target's grid nodes. Both phases read the operands there and gate them
      there, so the value the source maximizes is one a branch really pays;
      `"reject"` additionally refuses, at model build, a target on whose grid
      such a point can occur at all.
    """

    gate: UserFunction
    """Boolean gate on the target grid, evaluated in the target fold's context."""

    legs: Mapping[str, StakeholderRoute]
    """One `StakeholderRoute` per source stakeholder.

    A singleton source declares exactly one, under any key.
    """

    gate_refs: Mapping[str, ProjectedRegimeValue] = field(
        default_factory=lambda: MappingProxyType({})
    )
    """Same-period reference values the `gate` reads (projected from the target
    grid)."""

    off_grid: Literal["pointwise", "reject"] = "pointwise"
    """How a landing point between the target's grid nodes is treated.

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
        object.__setattr__(self, "legs", ensure_containers_are_immutable(self.legs))
        object.__setattr__(
            self, "gate_refs", ensure_containers_are_immutable(self.gate_refs)
        )


@beartype(conf=REGIME_CONF)
@dataclass(frozen=True, kw_only=True)
class Regime:
    """User-facing regime definition.

    `Model` processes instances of this class into the canonical regime form
    (`_lcm.engine.Regime`) used internally by the solver and simulator.

    State transitions are specified via `state_transitions`, mapping state names to
    transition functions. A bare callable is deterministic; wrap in `MarkovTransition`
    for stochastic transitions. `fixed_transition(state_name)` marks a fixed state
    (identity law). Stochastic processes have intrinsic transitions and must not
    appear in `state_transitions`.

    The `transition` field on the regime itself is the *regime* transition function.
    A regime with `transition=None` is terminal — no separate `terminal` flag is
    needed.

    """

    _accepts_margin_solver: ClassVar[bool] = False

    # `UserFunction`/`Phased` inside the per-target dict pass the type check
    # so the validator can reject them with an explanation.
    transition: (
        UserFunction
        | MarkovTransition
        | Phased
        | Mapping[
            RegimeName,
            MarkovTransition | UserFunction | Phased | ValueDependentTransition,
        ]
        | None
    )
    """Regime transition, or `None` for terminal regimes.

    Three forms:

    - bare callable ⇒ deterministic, returns the target regime id
    - `MarkovTransition` ⇒ stochastic, returns a probability vector over all
      regimes
    - per-target dict ⇒ stochastic, maps target regime names to
      `MarkovTransition`-wrapped functions returning that target's
      probability. The key set declares the regime's reachable targets;
      omitted regimes are structurally unreachable.

    A bare callable or bare `MarkovTransition` declares conservative support
    over every regime active in the next period — every temporally
    compatible candidate must therefore have a valid state handoff (a
    carried state, a deterministic/stochastic law, or an explicit
    target-local/entry law). Use a per-target mapping to declare narrower
    support instead. Runtime-zero transition probabilities do not narrow
    this topology; only the declared form does.

    `Phased` gives each phase its own variant (matching form required; for
    per-target dicts, identical key sets).
    """

    active: ActiveFunction = lambda _age: True
    """Callable that takes age (float) and returns True if regime is active."""

    # `None` masks a model-level entry of the same name.
    states: Mapping[StateName, Grid | Phased | AgeSpecializedGrid | None] = field(
        default_factory=lambda: MappingProxyType({})
    )
    """Mapping of state variable names to grids or phase-variant declarations.

    A plain `Grid` value is a state shared by both phases.
    `Phased(solve=callable, simulate=Grid)` declares a carried state: a
    derived function (no grid axis) in the solve phase and a seeded, evolved
    state in the simulate phase, whose law of motion is its regular
    `state_transitions` entry.
    An `AgeSpecializedGrid` value is a continuous state whose grid bounds vary
    with age (fixed `n_points`); it is resolved to a concrete grid per period at
    model build.
    """

    state_transitions: Mapping[
        StateName,
        UserFunction
        | MarkovTransition
        | Phased
        # `Phased` inside a per-target dict passes the type check so the
        # validator can reject it with the outermost-only explanation.
        | Mapping[RegimeName, UserFunction | MarkovTransition | Phased]
        | None,
    ] = field(default_factory=lambda: MappingProxyType({}))
    """Mapping of state names to transition functions or per-target dicts.

    Every non-process target-state cell must have exactly one producer: an ordinary
    entry here or an output of `joint_transitions`. `fixed_transition(state_name)`
    marks a fixed state (identity law). Wrap in
    `MarkovTransition` for stochastic transitions. Per-target dicts map target
    regime names to transition functions — every reachable target must be listed.
    `Phased` gives each phase its own law of motion; it wraps the whole entry
    (outermost only, never inside a per-target dict).
    """

    joint_transitions: Mapping[RegimeName, Mapping[str, JointTransition | Phased]] = (
        field(default_factory=lambda: MappingProxyType({}))
    )
    """Correlated finite-support transitions owned by explicit target edges.

    The outer key names the target regime and the inner key names the sampled
    transition node supplied to every output law of that kernel. A
    ``JointTransition`` shares one probability draw across all of its output
    states. ``Phased`` may wrap the whole joint transition to provide matching
    solve and simulation declarations.
    """

    actions: Mapping[ActionName, Grid | None] = field(
        default_factory=lambda: MappingProxyType({})
    )
    """Mapping of action variable names to grid objects."""

    functions: Mapping[
        FunctionName, UserFunction | Phased | CollectiveUtility | None
    ] = field(default_factory=lambda: MappingProxyType({}))
    """Mapping of function names to callables; must include 'utility'.

    `Phased` gives each phase its own implementation.
    """

    # `Phased` passes the type check so the validator can reject it with an
    # explanation (constraints are phase-invariant).
    constraints: Mapping[
        FunctionName, ConstraintLike | Phased | ValueDependentConstraint | None
    ] = field(default_factory=lambda: MappingProxyType({}))
    """Mapping of constraint names to constraints.

    A constraint is either a `Condition` built from `lcm.ref`, or an ordinary
    predicate. The two mean the same thing and are evaluated identically; a
    condition additionally carries what it says, so a solver can prove or
    refuse it instead of only being able to call it.

    Constraints are phase-invariant: a phase-specific feasible set would let
    the simulated argmax range over actions the value function was never
    computed for, so `Phased` is rejected here.
    """

    derived_categoricals: Mapping[FunctionName, DiscreteGrid] = field(
        default_factory=lambda: MappingProxyType({})
    )
    """Categorical grids for DAG function outputs not in states/actions."""

    solver: _solvers.Solver = field(default_factory=_solvers.GridSearch)
    """Solution algorithm for this regime during backward induction.

    The solver must match the regime declaration that supplies its structural
    roles:

    - `Regime`: `GridSearch()` (the default), or another `Solver` that does
      not require margin binding.
    - `ConsumptionSavingsRegime`: `GridSearch()` or a `OneMarginSolver`,
      such as `EGM(...)`, `DCEGM(...)`, or `NBEGM(...)`.
    - `NestedConsumptionSavingsRegime`: `GridSearch()` or a
      `TwoMarginSolver`, such as `NEGM(...)` or `NNBEGM(...)`.

    Endogenous-grid solvers validate their structural contracts during
    `Model(...)`; the specialized regime owns the state, action, resources,
    and post-decision role names bound into the solver.
    """

    taste_shocks: ExtremeValueTasteShocks | None = None
    """EV1 taste shocks on the regime's discrete-action combinations.

    When set, the shock scale becomes the runtime param
    `{"taste_shocks": {"scale": ...}}` and the solve aggregates discrete
    actions via the smoothed expected maximum instead of the hard maximum.
    Requires at least one discrete action.
    """

    koopmans_aggregator: UserFunction | Phased | None = None
    """Combines current-period utility with the certainty equivalent into `Q`.

    Signature `W(utility, CE, ...)`; further arguments are runtime params
    under the pseudo-function name `koopmans_aggregator`, or outputs of
    regime functions of the same name. `Phased(solve=..., simulate=...)`
    gives the two phases different aggregators (a naive/sophisticated
    beta-delta split, say). `None` means the regime takes the model-level
    aggregator (`lcm.LinearAggregator` unless the `Model` says otherwise). Terminal
    regimes have no continuation and take none.
    """

    certainty_equivalent: CertaintyEquivalent | None = None
    """Nonlinear certainty equivalent over the next-period value distribution.

    When set, the solve aggregates the continuation as
    `g⁻¹(Σ_r p_r · E_w[g(V')])` instead of the linear expectation, and the
    transform parameters become runtime params under the pseudo-function
    name `certainty_equivalent`. Only non-terminal regimes solved by
    `GridSearch` support it.
    """

    description: str = ""
    """Description of the regime."""

    stakeholders: tuple[str, ...] | None = None
    """Names of the stakeholders whose individual values this regime carries.

    The lowered form of `CollectiveUtility.utilities`, whose keys are the
    stakeholders in the order they are written. A model declares the household
    in `functions["utility"]` and reads the set back here.

    `None` (the default) is the singleton case: the regime has one implicit
    stakeholder and one value function. A non-`None` tuple declares a
    *collective regime*: a couple (or other multi-party household) that solves
    one household argmax but reads off a per-stakeholder value at that common
    argmax, with value-aware feasibility and value-gated regime routing
    (consent / dissolution).

    A collective regime carries a per-stakeholder utility
    `functions["utility_<s>"]` for each stakeholder `<s>` and a
    `pareto_objective`. Its solve reads off each stakeholder's own value at the shared
    household argmax, and a non-terminal one aggregates the per-stakeholder
    continuation `Q^s = H(u^s, E[V'^s])`. A non-terminal collective regime's
    transition targets must all be collective regimes with the identical
    `stakeholders` tuple — per-stakeholder routing to different regimes goes
    through `gated_edges`. EV1 taste shocks, nonlinear certainty equivalents,
    and non-GridSearch solvers on a collective regime raise
    `NotImplementedError`.

    A shock declared `fold=True` is refused when the model is built, naming the
    regime and the state. A collective regime writes `-inf` where no action
    satisfies every stakeholder's participation constraint — a sentinel a gated
    edge resolves to the outside option, not a value on the household's own
    scale — and quadrature over that sentinel is not an expectation: a
    household dissolving at one node would be stored as dissolving at all of
    them. The same shock folds normally in a singleton regime.

    Three things to know before simulating one:

    - The population is one fixed-size cohort of independent rows. A
      dissolution does not split a row into two independently tracked
      households; each row records where every stakeholder would land and then
      continues as one of them.
    - Every row carries its own role. `simulate` reads it from
      `initial_conditions["own_stakeholder"]` and updates it wherever a gated
      edge routes the row, so one cohort may hold both partners.
    - The off-grid value gate is approximate: it interpolates the target's
      already-maximized value rather than recomputing the household maximum at
      the realized off-grid point (see `get_edge_simulate_gate_evaluator`).
    """

    pareto_objective: ParetoObjective | None = None
    """How this collective regime's household trades its stakeholders off.

    The lowered form of `CollectiveUtility.objective`.

    Used only when `stakeholders is not None`: the collective solve maximizes
    the household scalarization `O = Σ_s λ_s Q^s` over the feasible action set.
    When omitted (the default), equal weights `1/len(stakeholders)` are used —
    the symmetric-couple case, `λ = 0.5` on each partner. Declare a
    `ParetoObjective` to weigh one stakeholder more, to let the weights depend
    on a state, or to estimate them. Ignored — and must be `None` — for a
    singleton regime.
    """

    value_constraints: Mapping[FunctionName, UserFunction] = field(
        default_factory=lambda: MappingProxyType({})
    )
    """Value-aware feasibility predicates for a collective regime.

    The lowered form of the `ValueDependentConstraint` entries of `constraints`,
    which is where a model declares them — one constraint slot rather than two.

    Each entry maps a constraint name to a predicate returning `True` where the
    (state, action) combination is feasible. Unlike ordinary `constraints`
    (which are evaluated before and independently of `Q`), a value constraint
    is evaluated AFTER the per-stakeholder action values and may read, as named
    arguments:

    - `Q_<s>` for each stakeholder `<s>` — that stakeholder's own action value
      `Q^s(x, a)` (felicity plus discounted continuation) at the cell;
    - each key of `same_period_refs` — the reference regime's same-period value
      interpolated at the projected state (e.g. the dissolved single's value);
    - ordinary states, actions, regime functions, and parameters via the DAG
      (a predicate's own parameter surfaces in the params template under the
      constraint's name).

    The final action mask is the AND of ordinary constraints and all value
    constraints; the household argmax runs over the masked set, and a state
    cell whose mask is empty publishes the dissolution flag `D = True` (returned by
    the solve alongside V — never conflated with a numeric `-inf` value, which
    can occur on-path). A participation constraint takes the form
    `Q_j >= V_single_j(pi_j(x)) - Delta_j` for each stakeholder `j`.

    A TERMINAL collective regime may declare them too — a household's last
    period is a participation decision like any other — with two differences
    worth stating, because nothing in the arrays reveals them:

    - `Q_<s>` is stakeholder `s`'s terminal payoff. There is no continuation,
      so the value each partner weighs against the reference is what the cell
      itself delivers.
    - What a terminal regime publishes is a **flag**, not a resolved outcome.
      An empty feasible set sets `D = True` and leaves the `-inf` sentinel as
      the value; pylcm substitutes no outside option, because there is no
      continuation to route into. A caller that reads the value without the
      flag reads the sentinel. Deciding what a dissolved terminal household is
      worth is the model's business.

    Only collective regimes may declare value constraints at all: the
    predicates read `Q_<s>`, which a singleton regime does not carry.
    """

    gated_edges: Mapping[RegimeName, GatedEdge] = field(
        default_factory=lambda: MappingProxyType({})
    )
    """Gated edges routing this regime's continuation into a target regime.

    The lowered form of the `ValueDependentTransition` entries of `transition`,
    which is where a model declares them, so that target selection and
    value-dependent routing are one declaration rather than two.

    Maps a TARGET regime name to a `GatedEdge`. A gated edge lets this regime
    reach a target of a DIFFERENT stakeholder layout (a singleton regime into a
    collective one for mutual-consent marriage, or a collective regime into
    singleton regimes for dissolution) — the only way to cross the mixed-topology
    fence. When declared, the engine folds a gated continuation object
    `Wbar^s = jnp.where(gate, V_target, V_fallback)` on the target regime's
    grid at each period's end, and this regime's continuation reads `Wbar` in
    place of the raw target V. See `GatedEdge`. Only meaningful together
    with the corresponding `transition` / `state_transitions` into the target's
    state space; a target reached by a gated edge is exempt from the mixed-
    stakeholder rejection.
    """

    same_period_refs: Mapping[str, ProjectedRegimeValue] = field(
        default_factory=lambda: MappingProxyType({})
    )
    """Same-period cross-regime reference values read by `value_constraints`.

    The lowered form of `ValueDependentConstraint.references`, which is where a
    model declares them — local to the constraint that reads them.

    Maps each reference-value name (the argument name under which the
    interpolated value enters the predicates) to a `ProjectedRegimeValue` declaring
    the reference regime, the state projection, and — for a collective
    reference — the stakeholder. Reference regimes are solved earlier within
    the same period (topological order; cycles are rejected at model build).
    Only collective regimes that also declare `value_constraints` may declare
    references.
    """

    @property
    def terminal(self) -> bool:
        """Whether this is a terminal regime (derived from transition being None)."""
        return self.transition is None

    @property
    def stochastic_regime_transition(self) -> bool:
        """Whether the regime transition is stochastic.

        A `MarkovTransition` and a per-target dict are both stochastic.
        `Phased` variants must have matching forms, so the solve variant is
        representative.
        """
        transition = (
            self.transition.solve
            if isinstance(self.transition, Phased)
            else self.transition
        )
        return isinstance(transition, MarkovTransition | Mapping)

    def __post_init__(self) -> None:
        self._lower_value_dependent_declarations()
        self._fail_if_egm_solver_has_no_margin_declaration()
        # A collective regime's own declaration is validated here (the
        # `stakeholders` tuple, `weights`, the value-constraint grammar;
        # out-of-scope features — taste shocks, nonlinear certainty
        # equivalents, non-GridSearch solvers — are rejected). What a
        # model-level slot may still supply — the per-stakeholder
        # `utility_<s>` functions — is checked when the model finalizes its
        # regimes. The default `None`
        # (singleton) path never enters this branch.
        if self.gated_edges:
            _validate_gated_edges(self)
        if self.stakeholders is not None:
            _validate_collective_regime(self)
        elif self.pareto_objective is not None:
            raise RegimeInitializationError(
                "`pareto_objective` declares how a collective regime's "
                "household weighs its stakeholders; it is only meaningful "
                "together with `stakeholders`. Omit it for a singleton regime."
            )
        elif self.value_constraints:
            raise RegimeInitializationError(
                "`value_constraints` are value-aware feasibility predicates for "
                "a collective regime; they read the per-stakeholder action "
                "values `Q_<s>`, which only exist when `stakeholders` is set. "
                "Use ordinary `constraints` for a singleton regime."
            )
        elif self.same_period_refs:
            raise RegimeInitializationError(
                "`same_period_refs` declares same-period reference values for a "
                "collective regime's `value_constraints`; it is only "
                "meaningful together with `stakeholders`. Omit it for a "
                "singleton regime."
            )

        _validate_mapping_contents(self)
        _validate_logical_consistency(self)
        _validate_fold_declarations(self)

        def make_immutable(name: str) -> None:
            value = ensure_containers_are_immutable(getattr(self, name))
            object.__setattr__(self, name, value)

        # Completeness (a `utility` entry, aggregator injection, transition
        # coverage) is validated when the model finalizes its regimes
        # — model-level slots may still satisfy it after merging.
        make_immutable("functions")
        make_immutable("states")
        make_immutable("state_transitions")
        make_immutable("joint_transitions")
        make_immutable("actions")
        make_immutable("constraints")
        make_immutable("derived_categoricals")
        make_immutable("value_constraints")
        make_immutable("gated_edges")
        make_immutable("same_period_refs")

        # The phase grammar (states matrix, carried laws, regime-transition
        # variants) is validated by the normalizer; the per-phase spec it
        # builds is consumed during model processing.
        normalize_regime_phases(self)

    def _lower_value_dependent_declarations(self) -> None:
        """Decompose the collective declarations onto what the engine runs.

        `CollectiveUtility`, `ValueDependentConstraint` and
        `ValueDependentTransition` are declared inside the slots a regime
        already has — `functions`, `constraints` and `transition` — and each
        one carries several engine-side facts at once. Splitting them here, at
        construction, is what lets every later stage keep reading the fields it
        has always read.

        Nothing is lowered for a regime that declares none of the three, so a
        regime spelled out the long way passes through untouched.
        """
        self._lower_collective_utility()
        self._lower_value_dependent_constraints()
        self._lower_value_dependent_transitions()

    @property
    def decomposed_functions(
        self,
    ) -> Mapping[FunctionName, UserFunction | Phased | None]:
        """`functions` with any `CollectiveUtility` taken apart.

        A `CollectiveUtility` under `"utility"` is replaced by one
        `utility_<s>` entry per stakeholder, in the order the household
        declares them — so the mapping an engine stage reads never holds a
        declaration object, whatever order the entries reached the regime in. A
        stakeholder whose body is delegated keeps the entry the regime already
        carries; the delegation itself was validated at construction.

        Deterministic and idempotent: a regime that declares no household is
        returned unchanged, and reading the view never changes the regime.
        """
        return decompose_functions(self.functions)

    @property
    def decomposed_constraints(
        self,
    ) -> Mapping[FunctionName, ConstraintLike | Phased | None]:
        """`constraints` with every `ValueDependentConstraint` taken apart.

        A value-dependent constraint's predicate belongs to
        `value_constraints` and its projections to `same_period_refs`, so what
        this view holds is the ordinary constraints alone — the ones evaluated
        before and independently of the action values.

        Deterministic and idempotent, like the other two views.
        """
        return decompose_constraints(self.constraints)

    @property
    def decomposed_transition(
        self,
    ) -> (
        UserFunction
        | MarkovTransition
        | Phased
        | Mapping[RegimeName, MarkovTransition | UserFunction | Phased]
        | None
    ):
        """`transition` with every `ValueDependentTransition` taken apart.

        A value-dependent transition carries two facts at once: which target
        the regime selects, and how the household is routed once there. The
        second belongs to `gated_edges`; what stays here is the selection
        probability, in the per-target cell the canonical pipeline reads. A
        bare probability callable is wrapped, because that cell's grammar takes
        a `MarkovTransition`.

        Deterministic and idempotent, like the other two views.
        """
        return decompose_transition(self.transition)

    def _lower_collective_utility(self) -> None:
        """Split `functions["utility"]` into stakeholders and their utilities."""
        declaration = self.functions.get("utility")
        if not isinstance(declaration, CollectiveUtility):
            return
        if self.stakeholders not in (None, tuple(declaration.utilities)):
            raise RegimeInitializationError(
                "A regime declaring `functions={'utility': CollectiveUtility(...)}` "
                "names its stakeholders there — the `utilities` keys are the "
                "stakeholders — so `stakeholders` may not be declared as well."
            )
        functions = {
            name: func for name, func in self.functions.items() if name != "utility"
        }
        for stakeholder, utility in declaration.utilities.items():
            entry = f"utility_{stakeholder}"
            if utility is None:
                # The body is delegated: whatever already stands under the
                # stakeholder's own name is her utility, and there has to be
                # one for the household to be complete.
                if entry not in functions:
                    raise RegimeInitializationError(
                        f"The stakeholder {stakeholder!r} of this regime's "
                        f"`CollectiveUtility` declares no utility of her own, "
                        f"so hers is whatever the regime carries under "
                        f"{entry!r} — and it carries nothing. Write the body "
                        f"in the `CollectiveUtility`, or supply {entry!r}."
                    )
                continue
            if entry in functions and functions[entry] is not utility:
                raise RegimeInitializationError(
                    f"The stakeholder {stakeholder!r} of this regime's "
                    f"`CollectiveUtility` and the function {entry!r} would both "
                    f"supply {stakeholder}'s utility. Declare it once, in the "
                    "`CollectiveUtility`."
                )
            functions[entry] = utility
        object.__setattr__(self, "functions", functions)
        object.__setattr__(self, "stakeholders", tuple(declaration.utilities))
        if declaration.objective is not None:
            if self.pareto_objective not in (None, declaration.objective):
                raise RegimeInitializationError(
                    "This regime declares a `ParetoObjective` both as its "
                    "`CollectiveUtility`'s objective and as `pareto_objective`. "
                    "Declare it once, in the `CollectiveUtility`."
                )
            object.__setattr__(self, "pareto_objective", declaration.objective)

    def _lower_value_dependent_constraints(self) -> None:
        """Split each `ValueDependentConstraint` into predicate and references."""
        declarations = {
            name: constraint
            for name, constraint in self.constraints.items()
            if isinstance(constraint, ValueDependentConstraint)
        }
        if not declarations:
            return
        constraints = {
            name: constraint
            for name, constraint in self.constraints.items()
            if name not in declarations
        }
        value_constraints = dict(self.value_constraints)
        same_period_refs = dict(self.same_period_refs)
        for name, declaration in declarations.items():
            if value_constraints.get(name) not in (None, declaration.predicate):
                raise RegimeInitializationError(
                    f"The constraint {name!r} is declared both as a "
                    "`ValueDependentConstraint` and in `value_constraints`. "
                    "Declare it once."
                )
            value_constraints[name] = declaration.predicate
            for ref_name, reference in declaration.references.items():
                existing = same_period_refs.get(ref_name)
                if existing is not None and existing != reference:
                    raise RegimeInitializationError(
                        f"Two constraints of this regime read a reference value "
                        f"named {ref_name!r} but declare different references "
                        f"for it:\n  {existing}\n  {reference}\n"
                        "One name is one reference; rename one of them."
                    )
                same_period_refs[ref_name] = reference
        object.__setattr__(self, "constraints", constraints)
        object.__setattr__(self, "value_constraints", value_constraints)
        object.__setattr__(self, "same_period_refs", same_period_refs)

    def _lower_value_dependent_transitions(self) -> None:
        """Split each `ValueDependentTransition` into a target and its edge."""
        transition = self.transition
        if not isinstance(transition, Mapping):
            return
        declarations = {
            target: entry
            for target, entry in transition.items()
            if isinstance(entry, ValueDependentTransition)
        }
        if not declarations:
            return
        entries = dict(transition)
        gated_edges = dict(self.gated_edges)
        for target, declaration in declarations.items():
            lowered = GatedEdge(
                gate=declaration.gate,
                legs=declaration.routes,
                gate_refs=declaration.gate_references,
                off_grid=declaration.off_grid,
            )
            existing = gated_edges.get(target)
            if existing is not None and existing != lowered:
                raise RegimeInitializationError(
                    f"The transition into {target!r} is declared both as a "
                    "`ValueDependentTransition` and as a `gated_edges` entry, "
                    "and the two disagree. An edge is its gate, its routes, "
                    "its gate references and its off-grid contract together, "
                    "so agreeing on the gate alone is not agreeing. Declare "
                    "it once."
                )
            # The lowered per-target cell requires a `MarkovTransition`, so a
            # bare probability callable is wrapped rather than refused: the
            # declared type and the resulting grammar say the same thing.
            probability = declaration.probability
            entries[target] = (
                probability
                if isinstance(probability, MarkovTransition)
                else MarkovTransition(probability)
            )
            gated_edges[target] = lowered
        object.__setattr__(self, "transition", entries)
        object.__setattr__(self, "gated_edges", gated_edges)

    def _fail_if_egm_solver_has_no_margin_declaration(self) -> None:
        if self._accepts_margin_solver:
            return
        if isinstance(self.solver, _solvers.OneMarginSolver | _solvers.TwoMarginSolver):
            raise RegimeInitializationError(
                "EGM-family solvers require regime-owned margin declarations: use "
                "ConsumptionSavingsRegime for a OneMarginSolver or "
                "NestedConsumptionSavingsRegime for a TwoMarginSolver."
            )

    def _validate_finalized_structure(self, *, regime_name: RegimeName) -> None:
        """Validate subclass-owned structure after model-level slots are merged."""
        _ = regime_name

    def get_koopmans_aggregator(
        self,
        phase: Literal["solve", "simulate"] = "solve",
    ) -> UserFunction | None:
        """Get the Bellman aggregator this phase runs.

        Args:
            phase: Which variant to use when the declaration is `Phased`.

        Returns:
            The aggregator, or `None` when the regime declares none (a
            terminal regime, or one taking the model-level value).

        """
        if isinstance(self.koopmans_aggregator, Phased):
            variant = (
                self.koopmans_aggregator.solve
                if phase == "solve"
                else self.koopmans_aggregator.simulate
            )
            return cast("UserFunction", variant)
        return self.koopmans_aggregator

    def get_all_functions(
        self,
        phase: Literal["solve", "simulate"] = "solve",
    ) -> MappingProxyType[str, UserFunction]:
        """Get all regime functions including utility, constraints, and transitions.

        Collect functions from four sources:

        - `self.functions` (utility and helpers)
        - `self.constraints`
        - state transitions from `self.state_transitions`
        - the regime transition (`self.transition`, keyed as `"next_regime"`)

        For `Phased` entries, the variant matching `phase` is used. A
        carried-state declaration in `states` (`Phased(solve=...,
        simulate=Grid)`) contributes its `solve` variant as a derived
        function under the state's name and its law of motion under
        `next_<name>`, mirroring how ordinary state transitions are keyed.

        Args:
            phase: Which variant to use for phase-variant entries.

        Returns:
            Read-only mapping of all regime functions.

        """

        def resolve(value: object) -> UserFunction:
            if isinstance(value, Phased):
                value = value.solve if phase == "solve" else value.simulate
            return cast("UserFunction", value)

        result: dict[str, UserFunction] = {
            name: resolve(func) for name, func in self.decomposed_functions.items()
        }
        for name, spec in self.states.items():
            if isinstance(spec, Phased):
                # Carried state: the solve variant is its derived-function
                # imputation; the law of motion is its regular
                # `state_transitions` entry, collected below.
                result[name] = cast("UserFunction", spec.solve)
        result |= cast("Mapping[str, UserFunction]", self.decomposed_constraints)
        decomposed_transition = self.decomposed_transition
        if decomposed_transition is not None:
            joint_output_names = {
                state_name
                for kernels in self.joint_transitions.values()
                for raw in kernels.values()
                for joint in (
                    (raw.solve if phase == "solve" else raw.simulate)
                    if isinstance(raw, Phased)
                    else raw,
                )
                for state_name in cast("JointTransition", joint).outputs
            }
            collected = collect_state_transitions(
                self.states,
                self.state_transitions,
                joint_output_names=joint_output_names,
            )
            result |= {name: resolve(func) for name, func in collected.items()}
            transition = decomposed_transition
            if isinstance(transition, Phased):
                transition = (
                    transition.solve if phase == "solve" else transition.simulate
                )
            if isinstance(transition, Mapping):
                # Per-target regime transition: one entry per declared target,
                # mirroring how per-target state laws are keyed.
                for target_regime_name, cell in transition.items():
                    result[f"next_regime__{target_regime_name}"] = cast(
                        "UserFunction", cell
                    )
            else:
                result["next_regime"] = cast("UserFunction", transition)
        return MappingProxyType(result)

    def _augment_phase_functions(
        self, functions: dict[FunctionName, UserFunction]
    ) -> dict[FunctionName, UserFunction]:
        """Add internal functions required by a specialized regime declaration."""
        return functions

    def with_engine_functions(
        self,
        *,
        engine_functions: Mapping[FunctionName, UserFunction | Phased | None],
        **other_slots: Any,  # noqa: ANN401
    ) -> Regime:
        """Overlay engine-composed functions without disturbing the declarations.

        Regime building reads a regime's functions through
        `decomposed_functions`, composes more of them, and hands the result
        back. Writing that result straight into `functions` would put the
        decomposition where the declaration was, and the household would be
        gone. This writes back by provenance instead: an entry the declaration
        produced is the declaration's, and is left to it; everything else is
        the engine's, and is overlaid on the slot the author wrote.

        Args:
            engine_functions: The complete mapping the engine intends the
                regime's `decomposed_functions` to be.
            **other_slots: Further slots to replace, as `replace` takes them.

        Returns:
            A new regime carrying the engine's additions, whose declarations
            are the ones this regime was written with.

        Raises:
            RegimeInitializationError: If the engine mapping rewrites a
                stakeholder's declared utility, replaces a declaration object,
                or does not reproduce what the resulting regime decomposes to.
        """
        declaration = self.functions.get("utility")
        if not isinstance(declaration, CollectiveUtility):
            return self.replace(functions=engine_functions, **other_slots)

        declared_bodies = {
            f"utility_{stakeholder}": body
            for stakeholder, body in declaration.utilities.items()
            if body is not None
        }
        overlay: dict[FunctionName, UserFunction | Phased | None] = {}
        for name, func in engine_functions.items():
            if name == "utility":
                raise RegimeInitializationError(
                    "The engine mapping supplies a plain 'utility' for a regime "
                    "whose utility is a `CollectiveUtility`. Writing it back "
                    "would replace the household by a single utility."
                )
            if name in declared_bodies:
                if func is not declared_bodies[name]:
                    raise RegimeInitializationError(
                        f"The engine mapping supplies a different body for "
                        f"{name!r}, which this regime's `CollectiveUtility` "
                        f"declares. A stakeholder's utility is hers to declare."
                    )
                continue
            overlay[name] = func

        raw = {
            name: func
            for name, func in self.functions.items()
            if name not in declared_bodies
        }
        written = self.replace(
            functions=MappingProxyType({**raw, **overlay}), **other_slots
        )
        missing = sorted(set(engine_functions) - set(written.decomposed_functions))
        if missing:
            raise RegimeInitializationError(
                f"Writing the engine mapping back would lose {missing} from what "
                "the regime decomposes to. The write-back overlays engine "
                "additions on the declarations; it cannot express a removal."
            )
        return written

    def replace(self, **kwargs: Any) -> Regime:  # noqa: ANN401
        """Replace the attributes of the regime.

        Replacing a slot that carried a `CollectiveUtility`,
        `ValueDependentConstraint` or `ValueDependentTransition` does NOT undo
        what that declaration was decomposed into: the stakeholders, value
        constraints and gated edges stay as declared, and a replacement that
        contradicts them is refused rather than silently dropping one side.
        Build the regime outright where the two must differ.

        Args:
            **kwargs: Keyword arguments to replace the attributes of the regime.

        Returns:
            A new regime with the replaced attributes.

        """
        try:
            return dataclasses.replace(self, **kwargs)
        except TypeError as e:
            raise RegimeInitializationError(
                f"Failed to replace attributes of the regime. The error was: {e}"
            ) from e


def decompose_functions(
    functions: Mapping[FunctionName, UserFunction | Phased | CollectiveUtility | None],
) -> Mapping[FunctionName, UserFunction | Phased | None]:
    """Replace a `CollectiveUtility` by one utility entry per stakeholder.

    Args:
        functions: A regime's `functions` as declared.

    Returns:
        The same mapping with any `CollectiveUtility` under `"utility"`
        replaced by one `utility_<s>` entry per stakeholder, emitted in the
        order the household declares them so that the result does not depend on
        the order the entries reached the regime in. A stakeholder whose body
        is delegated keeps the entry the mapping already carries. A mapping
        declaring no household is returned unchanged, which makes the
        transformation idempotent.
    """
    declaration = functions.get("utility")
    if not isinstance(declaration, CollectiveUtility):
        return cast("Mapping[FunctionName, UserFunction | Phased | None]", functions)
    stakeholder_entries = {
        f"utility_{stakeholder}" for stakeholder in declaration.utilities
    }
    decomposed: dict[FunctionName, UserFunction | Phased | None] = {
        name: cast("UserFunction | Phased | None", func)
        for name, func in functions.items()
        if name != "utility" and name not in stakeholder_entries
    }
    for stakeholder, utility in declaration.utilities.items():
        entry = f"utility_{stakeholder}"
        decomposed[entry] = cast(
            "UserFunction | Phased | None",
            functions[entry] if utility is None else utility,
        )
    return MappingProxyType(decomposed)


def decompose_constraints(
    constraints: Mapping[
        FunctionName, ConstraintLike | Phased | ValueDependentConstraint | None
    ],
) -> Mapping[FunctionName, ConstraintLike | Phased | None]:
    """Drop the value-dependent declarations from a regime's constraints.

    Args:
        constraints: A regime's `constraints` as declared.

    Returns:
        The ordinary constraints alone — the ones evaluated before and
        independently of the action values. A value-dependent constraint's
        predicate belongs to `value_constraints` and its projections to
        `same_period_refs`, so neither appears here.
    """
    return MappingProxyType(
        {
            name: cast("ConstraintLike | Phased | None", constraint)
            for name, constraint in constraints.items()
            if not isinstance(constraint, ValueDependentConstraint)
        }
    )


def decompose_transition(
    transition: object,
) -> (
    UserFunction
    | MarkovTransition
    | Phased
    | Mapping[RegimeName, MarkovTransition | UserFunction | Phased]
    | None
):
    """Replace every `ValueDependentTransition` by the probability it declares.

    Args:
        transition: A regime's `transition` as declared, including the `Phased`
            form.

    Returns:
        The same transition with each value-dependent cell replaced by its
        selection probability. The routing half of the declaration belongs to
        `gated_edges` and does not appear here.
    """
    if isinstance(transition, Phased):
        solve = _decomposed_transition_side(transition.solve)
        simulate = _decomposed_transition_side(transition.simulate)
        if solve is transition.solve and simulate is transition.simulate:
            return transition
        return Phased(solve=solve, simulate=simulate)
    return _decomposed_transition_side(transition)


def _decomposed_transition_side(
    transition: object,
) -> (
    UserFunction
    | MarkovTransition
    | Phased
    | Mapping[RegimeName, MarkovTransition | UserFunction | Phased]
    | None
):
    """Replace one phase's `ValueDependentTransition` cells by their probabilities.

    Args:
        transition: One phase's regime transition — a per-target mapping, a
            coarse callable or `MarkovTransition`, or `None` for a terminal
            regime.

    Returns:
        The same transition with every `ValueDependentTransition` cell replaced
        by the selection probability it declares, wrapped in a
        `MarkovTransition` where the declaration gave a bare callable. Anything
        that is not a per-target mapping is returned unchanged.
    """
    if not isinstance(transition, Mapping):
        return cast(
            "UserFunction | MarkovTransition | Phased | None",
            transition,
        )
    if not any(
        isinstance(cell, ValueDependentTransition) for cell in transition.values()
    ):
        # Nothing to take apart. Returning the very same mapping keeps the
        # phase-variation scan able to ask whether the author wrote one object
        # for both phases, which a freshly built copy would always deny.
        return cast(
            "Mapping[RegimeName, MarkovTransition | UserFunction | Phased]", transition
        )
    return MappingProxyType(
        {
            target: (
                _as_markov_transition(cell.probability)
                if isinstance(cell, ValueDependentTransition)
                else cast("MarkovTransition | UserFunction | Phased", cell)
            )
            for target, cell in transition.items()
        }
    )


def _as_markov_transition(
    probability: UserFunction | MarkovTransition,
) -> MarkovTransition:
    """Wrap a bare probability callable in the cell grammar's `MarkovTransition`."""
    if isinstance(probability, MarkovTransition):
        return probability
    return MarkovTransition(probability)
