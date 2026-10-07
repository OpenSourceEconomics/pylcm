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
from _lcm.regime_building.transitions import collect_state_transitions
from _lcm.regime_law import RegimeLaw
from _lcm.typing import ActionName, FunctionName, RegimeName, StateName
from _lcm.user_regime_validation import validate_regime
from _lcm.utils.containers import ensure_containers_are_immutable
from lcm.certainty_equivalent import CertaintyEquivalent
from lcm.collective import (
    CollectiveUtility,
    ParetoObjective,
    ProjectedRegimeValue,
    ValueDependentConstraint,
)
from lcm.exceptions import RegimeInitializationError
from lcm.phased import Phased
from lcm.taste_shocks import ExtremeValueTasteShocks
from lcm.transition import (
    AgeSpecializedGrid,
    JointTransition,
    StochasticTransition,
)
from lcm.typing import UserFunction


@beartype(conf=REGIME_CONF)
@dataclass(frozen=True, kw_only=True)
class Regime:
    """User-facing regime definition.

    `Model` processes instances of this class into the canonical regime form
    (`_lcm.engine.Regime`) used internally by the solver and simulator.

    State transitions are specified via `state_transitions`, mapping state names to
    transition functions. A bare state callable is deterministic; wrap in
    `StochasticTransition` for stochastic transitions. `fixed_transition(state_name)`
    marks a fixed state
    (identity law). Stochastic processes have intrinsic transitions and must not
    appear in `state_transitions`.

    Movement between regimes is declared only in `Model(edges=...)`: a source
    with one outgoing edge per age moves along it, and a `Transition` carries
    the law wherever an age has several. A regime with no outgoing edges is
    terminal.

    """

    _accepts_margin_solver: ClassVar[bool] = False

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
        | StochasticTransition
        | Phased
        # `Phased` inside a per-target dict passes the type check so the
        # validator can reject it with the outermost-only explanation.
        | Mapping[RegimeName, UserFunction | StochasticTransition | Phased]
        | None,
    ] = field(default_factory=lambda: MappingProxyType({}))
    """Mapping of state names to transition functions or per-target dicts.

    Every non-process target-state cell must have exactly one producer: an ordinary
    entry here or an output of `joint_transitions`. `fixed_transition(state_name)`
    marks a fixed state (identity law). Wrap in
    `StochasticTransition` for stochastic transitions. Per-target dicts map target
    regime names to transition functions — every reachable target must be listed.
    `Phased` gives each phase its own law of motion; it wraps the whole entry
    (outermost only, never inside a per-target dict).
    """

    joint_transitions: Mapping[RegimeName, Mapping[str, JointTransition | Phased]] = (
        field(default_factory=lambda: MappingProxyType({}))
    )
    """Correlated finite-support transitions owned by explicit target edges.

    The outer key names a reachable target regime and the inner key names the
    sampled transition node supplied to every output law of that kernel. A
    `JointTransition` shares one probability draw across all of its output
    states. Terminal regimes declare none, and the current implementation is
    limited to `GridSearch` source regimes.

    `Phased` may wrap the whole joint transition. Its solve and simulation
    declarations must have identical output-state keys and `support_size`; two
    literal supports must also have identical pytree structure, leaf event
    shapes, and dtypes. Callable-support schemas must remain identical across
    active periods and both phases after params bind. Output laws are compiled
    against the applicable phase grids. Probability rows are numerically
    validated there, including carried-only simulation states, and must be
    finite, in `[0, 1]`, and unit mass.
    """

    actions: Mapping[ActionName, Grid | None] = field(
        default_factory=lambda: MappingProxyType({})
    )
    """Mapping of action variable names to grid objects."""

    functions: Mapping[
        FunctionName, UserFunction | Phased | CollectiveUtility | None
    ] = field(default_factory=lambda: MappingProxyType({}))
    """Mapping of function names to callables; must include 'utility'.

    `Phased` gives each phase its own implementation. A collective regime
    declares a `CollectiveUtility` under `"utility"`; that object remains in
    this raw mapping, while `decomposed_functions` exposes its per-stakeholder
    bodies under `utility_<s>` for the engine.
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

    The solve first maximizes the continuous actions conditional on each
    discrete-action combination, then replaces the hard maximum across those
    combinations by `scale * logsumexp(Q / scale)`. Simulation adds one
    mean-zero `scale * (Gumbel(0, 1) - EULER_GAMMA)` draw per discrete
    combination and subject before choosing the realized argmax. For a fixed
    candidate-value array, centering makes the expected latent perturbed
    maximum equal its log-sum. The shock affects the choice; simulation
    publishes the selected unshocked value. DCEGM simulation is
    grid-restricted, so its candidates need not equal the off-grid solve
    candidates.

    The shock scale is the runtime param
    `{"taste_shocks": {"scale": ...}}` and must be strictly positive; omit
    `taste_shocks` for a hard maximum. At least one discrete action is required.
    Taste shocks are currently supported by `GridSearch` and `DCEGM`. They are
    rejected on a collective regime, on the source of a
    `ValueDependentTransition`, with a folded IID state or nonlinear certainty
    equivalent, and with `NEGM`, `NBEGM`, or `NNBEGM`.
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
    name `certainty_equivalent`. `LinearExpectation()` is supported by every
    solver. `GridSearch` supports any otherwise-supported
    certainty-equivalent declaration. `NBEGM` and `NNBEGM` support the
    nonlinear `PowerMean()` paired with `CESAggregator()` only on ride-along
    routes that pass their remaining structural gates; the single-liquid
    route, current-period jumps, and liquid-dependent continuation reads are
    rejected. Other EGM-family solvers reject a nonlinear certainty
    equivalent. Terminal regimes have no continuation and cannot declare one.
    """

    description: str = ""
    """Description of the regime."""

    stakeholders: tuple[str, ...] | None = field(init=False, default=None)
    """Names of the stakeholders whose individual values this regime carries.

    Derived from `CollectiveUtility.utilities`, whose keys are the
    stakeholders in the order they are written. A model declares the household
    in the raw `functions["utility"]` slot and reads the set back here; the
    declaration itself is not replaced.

    `None` (the default) is the singleton case: the regime has one implicit
    stakeholder and one value function. A non-`None` tuple declares a
    *collective regime*: a couple (or other multi-party household) that solves
    one household argmax but reads off a per-stakeholder value at that common
    argmax, with value-aware feasibility and value-gated regime routing
    (consent / dissolution).

    A collective regime's `decomposed_functions` view carries a
    per-stakeholder `utility_<s>` for each stakeholder `<s>`, together with a
    `pareto_objective`. Its solve reads off each stakeholder's own value at the
    shared household argmax, and a non-terminal one aggregates the
    per-stakeholder continuation `Q^s = W(u^s, E[V'^s])`. A non-terminal
    collective regime's transition targets must all be collective regimes with
    the identical `stakeholders` tuple — per-stakeholder routing to different
    regimes goes through a `ValueDependentTransition` in the source's
    `Transition` law. EV1 taste shocks, nonlinear certainty
    equivalents, and non-GridSearch solvers on a collective regime raise
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

    pareto_objective: ParetoObjective | None = field(init=False, default=None)
    """How this collective regime's household trades its stakeholders off.

    Derived from `CollectiveUtility.objective`.

    Used only when `stakeholders is not None`: the collective solve maximizes
    the household scalarization `O = Σ_s λ_s Q^s` over the feasible action set.
    When omitted (the default), equal weights `1/len(stakeholders)` are used —
    the symmetric-couple case, `λ = 0.5` on each partner. Declare a
    `ParetoObjective` to weigh one stakeholder more, to let the weights depend
    on a state, or to estimate them. Ignored — and must be `None` — for a
    singleton regime.
    """

    value_constraints: Mapping[FunctionName, UserFunction] = field(
        init=False, default_factory=lambda: MappingProxyType({})
    )
    """Value-aware feasibility predicates for a collective regime.

    Derived from the `ValueDependentConstraint` entries of `constraints`,
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

    same_period_refs: Mapping[str, ProjectedRegimeValue] = field(
        init=False, default_factory=lambda: MappingProxyType({})
    )
    """Same-period cross-regime reference values read by `value_constraints`.

    Derived from `ValueDependentConstraint.references`, which is where a
    model declares them — local to the constraint that reads them.

    Maps each reference-value name (the argument name under which the
    interpolated value enters the predicates) to a `ProjectedRegimeValue` declaring
    the reference regime, the state projection, and — for a collective
    reference — the stakeholder. Reference regimes are solved earlier within
    the same period (topological order; cycles are rejected at model build).
    Only collective regimes that also declare `value_constraints` may declare
    references.
    """

    def _make_field_immutable(self, *, name: str) -> None:
        """Replace the named mapping field with its immutable form."""
        value = ensure_containers_are_immutable(getattr(self, name))
        object.__setattr__(self, name, value)

    def __post_init__(self) -> None:
        self._lower_value_dependent_declarations()
        self._fail_if_egm_solver_has_no_margin_declaration()
        # What depends on the law is validated once the model binds it from
        # `Model(edges=...)`; completeness (a `utility` entry, aggregator
        # injection, transition coverage) is validated when the model finalizes
        # its regimes, since model-level slots may still satisfy it.
        validate_regime(self)
        self._make_field_immutable(name="functions")
        self._make_field_immutable(name="states")
        self._make_field_immutable(name="state_transitions")
        self._make_field_immutable(name="joint_transitions")
        self._make_field_immutable(name="actions")
        self._make_field_immutable(name="constraints")
        self._make_field_immutable(name="derived_categoricals")
        self._make_field_immutable(name="value_constraints")
        self._make_field_immutable(name="same_period_refs")

    def _lower_value_dependent_declarations(self) -> None:
        """Derive the engine-facing views of the collective declarations.

        `CollectiveUtility` and `ValueDependentConstraint` are declared inside
        the slots a regime already has — `functions` and `constraints` — and
        each one carries
        several engine-side facts at once. Deriving those facts here, without
        replacing the raw declarations, lets every later stage read the fields
        and decomposed views it needs.

        Nothing is derived for a regime that declares neither, so a regime
        spelled out the long way passes through untouched.
        """
        self._lower_collective_utility()
        self._lower_value_dependent_constraints()

    @property
    def decomposed_functions(
        self,
    ) -> Mapping[FunctionName, UserFunction | Phased | None]:
        """`functions` with any `CollectiveUtility` taken apart.

        A `CollectiveUtility` under `"utility"` is replaced by one
        `utility_<s>` entry per stakeholder, in the order the household
        declares them — so the mapping an engine stage reads never holds a
        declaration object, whatever order the entries reached the regime in. A
        stakeholder whose body is delegated keeps an entry the regime already
        carries and is otherwise omitted. Model finalization merges any
        model-level body and then validates that every stakeholder is complete.

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

        Deterministic and idempotent, like `decomposed_functions`.
        """
        return decompose_constraints(self.constraints)

    def _lower_collective_utility(self) -> None:
        """Derive stakeholder metadata from `functions["utility"]`."""
        declaration = self.functions.get("utility")
        if not isinstance(declaration, CollectiveUtility):
            return
        for stakeholder, utility in declaration.utilities.items():
            entry = f"utility_{stakeholder}"
            # A delegated body may still arrive from the model level. Whether
            # one ever does is completeness, which the model reports once its
            # regimes are merged.
            if utility is None:
                continue
            if entry in self.functions and self.functions[entry] is not utility:
                raise RegimeInitializationError(
                    f"The stakeholder {stakeholder!r} of this regime's "
                    f"`CollectiveUtility` and the function {entry!r} would both "
                    f"supply {stakeholder}'s utility. Declare it once, in the "
                    "`CollectiveUtility`."
                )
        object.__setattr__(self, "stakeholders", tuple(declaration.utilities))
        if declaration.objective is not None:
            object.__setattr__(self, "pareto_objective", declaration.objective)

    def _lower_value_dependent_constraints(self) -> None:
        """Derive predicates and references from value-dependent constraints."""
        declarations = {
            name: constraint
            for name, constraint in self.constraints.items()
            if isinstance(constraint, ValueDependentConstraint)
        }
        if not declarations:
            return
        value_constraints: dict[FunctionName, UserFunction] = {}
        same_period_refs: dict[str, ProjectedRegimeValue] = {}
        for name, declaration in declarations.items():
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
        object.__setattr__(self, "value_constraints", value_constraints)
        object.__setattr__(self, "same_period_refs", same_period_refs)

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

    # keyword-only-exempt: primary-argument=phase
    def get_all_functions(
        self,
        phase: Literal["solve", "simulate"] = "solve",
        *,
        law: RegimeLaw | None = None,
    ) -> MappingProxyType[str, UserFunction]:
        """Get all regime functions including utility, constraints, and transitions.

        Collect functions from four sources:

        - `self.functions` (utility and helpers)
        - `self.constraints`
        - state transitions from `self.state_transitions`
        - the regime transition (`law`, keyed as `"next_regime"`, or
          `"next_regime__<target>"` per target)

        For `Phased` entries, the variant matching `phase` is used. A
        carried-state declaration in `states` (`Phased(solve=...,
        simulate=Grid)`) contributes its `solve` variant as a derived
        function under the state's name and its law of motion under
        `next_<name>`, mirroring how ordinary state transitions are keyed.

        Args:
            phase: Which variant to use for phase-variant entries.
            law: The law the model binds for this regime from its edges
                (`model.graph.laws[name]`). Without
                one the regime's state laws are collected and no regime
                transition; a terminal law contributes neither.

        Returns:
            Read-only mapping of all regime functions.

        """

        result: dict[str, UserFunction] = {
            name: _resolve_phase_variant(value=func, phase=phase)
            for name, func in self.decomposed_functions.items()
        }
        for name, spec in self.states.items():
            if isinstance(spec, Phased):
                # Carried state: the solve variant is its derived-function
                # imputation; the law of motion is its regular
                # `state_transitions` entry, collected below.
                result[name] = cast("UserFunction", spec.solve)
        result |= cast("Mapping[str, UserFunction]", self.decomposed_constraints)
        if law is None or not law.terminal:
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
                states=self.states,
                state_transitions=self.state_transitions,
                joint_output_names=joint_output_names,
                phase=phase,
            )
            result |= {
                name: _resolve_phase_variant(value=func, phase=phase)
                for name, func in collected.items()
            }
            transition = None if law is None else law.decomposed_transition
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
            elif transition is not None:
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
        disagreement = sorted(set(engine_functions) ^ set(written.decomposed_functions))
        if disagreement:
            raise RegimeInitializationError(
                f"Writing the engine mapping back would not reproduce it: "
                f"{disagreement} appears on one side only. The write-back "
                "overlays engine additions on the declarations, so it can add "
                "a function but never drop what a declaration produces."
            )
        return written

    def replace(self, **kwargs: Any) -> Regime:  # noqa: ANN401
        """Replace the attributes of the regime.

        Replacing a slot that carries a `CollectiveUtility` or
        `ValueDependentConstraint` replaces the declaration itself, and the
        stakeholders and value constraints are derived again from what the new
        slot says. `stakeholders`, `pareto_objective`, `value_constraints` and
        `same_period_refs` are those derived values, so naming one here is an
        error: they are read off a regime, never written to it.

        Args:
            **kwargs: Keyword arguments to replace the attributes of the regime.

        Returns:
            A new regime with the replaced attributes.

        """
        try:
            return dataclasses.replace(self, **kwargs)
        except (TypeError, ValueError) as e:
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
        if utility is None:
            # A delegated body that never arrived leaves no entry at all, so
            # completeness reports it by name when the model finalizes.
            if entry in functions:
                decomposed[entry] = cast(
                    "UserFunction | Phased | None", functions[entry]
                )
            continue
        decomposed[entry] = cast("UserFunction | Phased | None", utility)
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


def _resolve_phase_variant(
    *, value: object, phase: Literal["solve", "simulate"]
) -> UserFunction:
    """Return the variant of a possibly `Phased` entry that applies in `phase`."""
    if isinstance(value, Phased):
        value = value.solve if phase == "solve" else value.simulate
    return cast("UserFunction", value)
