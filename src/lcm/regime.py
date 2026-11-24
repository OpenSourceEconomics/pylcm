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
from functools import cached_property
from types import MappingProxyType
from typing import ClassVar, Literal, TypedDict, Unpack, cast

from beartype import beartype

import lcm.solvers as _solvers
from _lcm.beartype_conf import REGIME_CONF
from _lcm.constraints.processed import ConstraintLike
from _lcm.grids import Grid
from _lcm.post_decision_bound import _PostDecisionLowerBound
from _lcm.processes.joint_normal_lowering import lower_joint_normal_processes
from _lcm.regime_building.transitions import collect_state_transitions
from _lcm.regime_law import RegimeLaw
from _lcm.typing import ActionName, FunctionName, RegimeName, StateName
from _lcm.user_regime_validation import validate_regime
from _lcm.utils.containers import ensure_containers_are_immutable
from _lcm.utils.error_messages import path_segment_name_errors
from lcm.certainty_equivalent import CertaintyEquivalent
from lcm.collective import CollectiveUtility, ParetoObjective
from lcm.decoupled import (
    Decoupled,
    resolve_koopmans_aggregator,
)
from lcm.exceptions import RegimeInitializationError
from lcm.exits import Exit
from lcm.grids import DiscreteGrid
from lcm.phased import Phased
from lcm.processes import JointNormalIIDProcess
from lcm.shocks import ShockGroupName
from lcm.taste_shocks import ExtremeValueTasteShocks
from lcm.transition import (
    AgeSpecializedGrid,
    JointTransition,
    PeriodSpecializedGrid,
    StochasticTransition,
)
from lcm.typing import UserFunction

# One `Regime.states` value; `None` masks a model-level state of the same name.
type StateEntry = (
    Grid
    | JointNormalIIDProcess
    | Phased
    | AgeSpecializedGrid
    | PeriodSpecializedGrid
    | None
)

# One `Regime.state_transitions` value: a law, a stochastic law, a phase pair, or
# a per-target mapping of laws; `None` masks a model-level law. `Phased` inside a
# per-target mapping passes the type check so the validator can reject it with
# the outermost-only explanation.
type StateTransitionEntry = (
    UserFunction
    | StochasticTransition
    | Phased
    | Mapping[RegimeName, UserFunction | StochasticTransition | Phased]
    | None
)

# One `Regime.actions` value; `None` masks a model-level action.
type ActionEntry = Grid | None

# One `Regime.functions` value; `None` masks a model-level function.
type FunctionEntry = UserFunction | Phased | CollectiveUtility | None

# One `Regime.constraints` value; `None` masks a model-level constraint. `Phased`
# passes the type check so the validator can reject it with an explanation.
type ConstraintEntry = ConstraintLike | Phased | None


class _SlotsBesideFunctions(TypedDict, total=False):
    """The replaceable slots other than `functions`, each with its declared type.

    `Regime.with_engine_functions` writes `functions` itself and takes these
    beside it.
    """

    states: Mapping[StateName, StateEntry]
    state_transitions: Mapping[StateName, StateTransitionEntry]
    joint_transitions: Mapping[RegimeName, Mapping[str, JointTransition | Phased]]
    actions: Mapping[ActionName, ActionEntry]
    constraints: Mapping[FunctionName, ConstraintEntry]
    output_only_functions: tuple[FunctionName, ...]
    derived_categoricals: Mapping[FunctionName, DiscreteGrid]
    solver: _solvers.Solver
    taste_shocks: ExtremeValueTasteShocks | None
    shocks: tuple[ShockGroupName, ...]
    exit: Exit | None
    koopmans_aggregator: UserFunction | Phased | Decoupled | None
    certainty_equivalent: CertaintyEquivalent | None
    description: str


class RegimeReplacement(_SlotsBesideFunctions, total=False):
    """The slots `Regime.replace` may replace, each with its declared type.

    These are the slots a regime is constructed from. The derived slots
    (`members`, `member_axis` and `pareto_objective`) are read off a regime and
    are not among them.
    """

    functions: Mapping[FunctionName, FunctionEntry]


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

    states: Mapping[StateName, StateEntry] = field(
        default_factory=lambda: MappingProxyType({})
    )
    """Mapping of state variable names to grids or phase-variant declarations.

    A plain `Grid` value is a state shared by both phases.
    `Phased(solve=callable, simulate=Grid)` declares a carried state: a
    derived function (no grid axis) in the solve phase and a seeded, evolved
    state in the simulate phase, whose law of motion is its regular
    `state_transitions` entry.
    `Phased(solve=process, simulate=process)` keeps the perceived process for
    values and decisions and draws from the separate simulation process. Distinct
    phase laws use continuous draws; runtime parameter names shared by both
    declarations share one value, while constructor parameters may differ.
    An `AgeSpecializedGrid` value is a continuous state whose grid bounds vary
    with age (fixed `n_points`); it is resolved to a concrete grid per period at
    model build.
    A `JointNormalIIDProcess` value declares correlated physical shocks. It
    stays here as declared; `lower_joint_normal_processes` gives the
    coordinate states and generated functions the engine runs.
    """

    state_transitions: Mapping[StateName, StateTransitionEntry] = field(
        default_factory=lambda: MappingProxyType({})
    )
    """Mapping of state names to transition functions or per-target dicts.

    Every non-process target-state cell must have exactly one producer: an ordinary
    entry here or an output of `joint_transitions`. `fixed_transition(state_name)`
    marks a fixed state (identity law). Wrap in
    `StochasticTransition` for stochastic transitions. Per-target dicts map target
    regime names to transition functions. Every reachable target that carries the
    state must be listed, unless a joint transition produces its cell; a target
    that does not carry the state may be left out, and a law listed for one is
    unused.
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

    actions: Mapping[ActionName, ActionEntry] = field(
        default_factory=lambda: MappingProxyType({})
    )
    """Mapping of action variable names to grid objects."""

    functions: Mapping[FunctionName, FunctionEntry] = field(
        default_factory=lambda: MappingProxyType({})
    )
    """Mapping of function names to callables; must include 'utility'.

    `Phased` gives each phase its own implementation. A collective regime
    declares a `CollectiveUtility` under `"utility"`; that object remains in
    this raw mapping, while `decomposed_functions` exposes its per-member
    bodies under `utility_<s>` for the engine.
    """

    constraints: Mapping[FunctionName, ConstraintEntry] = field(
        default_factory=lambda: MappingProxyType({})
    )
    """Mapping of constraint names to constraints.

    A constraint is either a `Condition` built from `lcm.ref`, or an ordinary
    predicate. The two mean the same thing and are evaluated identically; a
    condition additionally carries what it says, so a solver can prove or
    refuse it instead of only being able to call it.

    Constraints are phase-invariant: a phase-specific feasible set would let
    the simulated argmax range over actions the value function was never
    computed for, so `Phased` is rejected here.
    """

    output_only_functions: tuple[FunctionName, ...] = ()
    """Functions retained solely for reporting after simulation.

    Declare each name on the model or on this regime. A function read by the
    model needs no declaration and cannot be named here.
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
    Taste shocks are currently supported by `GridSearch` and `DCEGM`. A household
    (a `CollectiveUtility`) admits them when exactly one member has a positive
    Pareto weight: that member decides, its value is its own smoothed maximum,
    and every other member's value is its mean action value under the
    decider's choice. They are rejected on a household with more than one
    positive weight, on a household that declares an `exit` or `shocks`, or whose
    decision another regime's exit reads, on the source of a `Gate`, with a nonlinear
    certainty equivalent, and with `NEGM`, `NBEGM`, or `NNBEGM`.
    """

    koopmans_aggregator: UserFunction | Phased | Decoupled | None = None
    """Combines current-period utility with the certainty equivalent into `Q`.

    Signature `W(utility, CE, ...)`; further arguments are runtime params
    under the pseudo-function name `koopmans_aggregator`, or outputs of
    regime functions of the same name. `Phased(solve=..., simulate=...)`
    gives the two phases different aggregators. `Decoupled(objective=...,
    value=...)` chooses actions with one aggregator and stores the value of
    another (time-inconsistent agents). `None` means the regime takes the model-level
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

    shocks: tuple[ShockGroupName, ...] = ()
    """Model-level IID shock-group references, separate from state processes.

    Each name references a `Model(shocks=...)` group. Reading a shock `x` reads
    this period's realization, known at the choice, and gives the group a node
    axis. A group the regime reads only as `next_x`, the draw after the choice,
    has no axis: the solve integrates it out of the continuation. Utility and
    constraints cannot read `next_x`. `repr(model)` lists, per regime, which
    groups carry an axis and which are integrated out.
    """

    exit: Exit | None = None
    """Current-period move to mapped destination policies, before the action."""

    description: str = ""
    """Description of the regime."""

    member_axis: tuple[str, ...] | None = field(init=False, default=None)
    """The members along the trailing value axis, or `None` for no such axis.

    The engine's marker for a household: equal to `members` for a regime
    declared with a `CollectiveUtility`, and `None` for a regime declared with
    a plain utility function, whose value carries no member axis. Derived from
    `CollectiveUtility.utilities`, whose keys are the members in the order they
    are written.

    `None` (the default) is the singleton case: the regime has the single
    member `"self"` and one value function. A non-`None` tuple declares a
    *collective regime*: a household, such as a couple, that solves
    one household argmax but reads off a per-member value at that common
    argmax, and whose exits compare each member's value (consent, dissolution).

    A collective regime's `decomposed_functions` view carries a
    per-member `utility_<s>` for each member `<s>`, together with a
    `pareto_objective`. Its solve reads off each member's own value at the
    shared household argmax, and a non-terminal one aggregates the
    per-member continuation `Q^s = W(u^s, E[V'^s])`. A non-terminal
    collective regime's transition targets must all be collective regimes with
    the identical `members` tuple — per-member current outside options
    are declared by an `Exit`. EV1 taste shocks with more than one member
    carrying a positive weight and nonlinear certainty equivalents on a
    collective regime raise `NotImplementedError`, as does a solver that does
    not declare `supports_several_members` on a regime with more than one
    member.

    An `Exit` compares each member's value with its outside value in the current
    period.

    Two things to know before simulating one:

    - The population is one fixed-size cohort of independent rows. A
      dissolution does not split a row into two independently tracked
      households; each row records where every member would land and then
      continues as one of them.
    - Every row carries its own role. `simulate` reads it from
      `initial_conditions["own_member"]`, so one cohort may hold both partners.
    """

    pareto_objective: ParetoObjective | None = field(init=False, default=None)
    """How this collective regime's household trades its members off.

    Derived from `CollectiveUtility.objective`.

    Used only when `member_axis is not None`: the collective solve maximizes
    the household scalarization `O = Σ_s λ_s Q^s` over the feasible action set.
    When omitted (the default), equal weights `1/len(members)` are used —
    the symmetric-couple case, `λ = 0.5` on each partner. Declare a
    `ParetoObjective` to weigh one member more, to let the weights depend
    on a state, or to estimate them. Ignored — and must be `None` — for a
    singleton regime.
    """

    def _make_field_immutable(self, *, name: str) -> None:
        """Replace the named mapping field with its immutable form."""
        value = ensure_containers_are_immutable(getattr(self, name))
        object.__setattr__(self, name, value)

    def __post_init__(self) -> None:
        if len(set(self.shocks)) != len(self.shocks) or any(
            not name.isidentifier()
            or path_segment_name_errors(kind="Shock group", names=(name,))
            for name in self.shocks
        ):
            raise RegimeInitializationError(
                "shocks must contain unique ordinary group names."
            )
        self._lower_collective_utility()
        self._fail_if_egm_solver_has_no_margin_declaration()
        # What depends on the law is validated once the model binds it from
        # `Model(edges=...)`; completeness (a `utility` entry, aggregator
        # injection, transition coverage) is validated when the model finalizes
        # its regimes, since model-level slots may still satisfy it. A regime
        # declaring a joint process is validated in the form the engine runs,
        # which constructing that form does.
        if self.lower_joint_normal_processes() is self:
            validate_regime(self)
        self._make_field_immutable(name="functions")
        self._make_field_immutable(name="states")
        self._make_field_immutable(name="state_transitions")
        self._make_field_immutable(name="joint_transitions")
        self._make_field_immutable(name="actions")
        self._make_field_immutable(name="constraints")
        self._make_field_immutable(name="derived_categoricals")

    def lower_joint_normal_processes(self) -> Regime:
        """Return the regime the engine runs for its joint normal declarations.

        Each `JointNormalIIDProcess` in `states` becomes one standard-normal
        coordinate state `<group>_normal_<d>` per dimension, and `functions`
        gains the generated `<group>` vector and one function per physical
        shock. The declarations themselves stay in this regime's `states` and
        `functions`, so `replace` swaps a joint process or a function without
        meeting the generated names.

        Returns:
            This regime when it declares no joint process; otherwise a new
            regime holding the lowered states and functions.

        Raises:
            RegimeInitializationError: If a generated name collides with a
                declaration of this regime.
        """
        if not any(
            isinstance(state, JointNormalIIDProcess) for state in self.states.values()
        ):
            return self
        states, functions = lower_joint_normal_processes(
            states=self.states,
            functions=self.functions,
            other_names=frozenset(self.actions)
            | frozenset(self.constraints)
            | frozenset(self.state_transitions),
        )
        return self.replace(states=states, functions=functions)

    @property
    def members(self) -> tuple[str, ...]:
        """Names of the members whose individual values this regime carries.

        Every regime has at least one member. A regime declared with a plain
        utility function has the single member `"self"`; a household declared
        with a `CollectiveUtility` has its `utilities` keys, in the order they
        are written (its `member_axis`).
        """
        return ("self",) if self.member_axis is None else self.member_axis

    @cached_property
    def decomposed_functions(
        self,
    ) -> Mapping[FunctionName, UserFunction | Phased | None]:
        """`functions` with any `CollectiveUtility` taken apart.

        A `CollectiveUtility` under `"utility"` is replaced by one
        `utility_<s>` entry per member, in the order the household
        declares them — so the mapping an engine stage reads never holds a
        declaration object, whatever order the entries reached the regime in. A
        member whose body is delegated keeps an entry the regime already
        carries and is otherwise omitted. Model finalization merges any
        model-level body and then validates that every member is complete.

        Deterministic and idempotent: a regime that declares no household is
        returned unchanged, and reading the view never changes the regime.
        """
        from _lcm.constraints.dynamic_bound import get_floor_node  # noqa: PLC0415
        from _lcm.constraints.processed import normalize_constraints  # noqa: PLC0415
        from lcm.condition import Ref  # noqa: PLC0415

        functions = dict(decompose_functions(self.functions))
        for name, constraint in self.constraints.items():
            if isinstance(constraint, _PostDecisionLowerBound) and callable(
                constraint.lower_bound
            ):
                target = f"{name}_lower"
                occupied = (
                    set(functions)
                    | set(self.constraints)
                    | set(self.states)
                    | set(self.actions)
                    | set(self.derived_categoricals)
                )
                occupied |= {
                    f"{prefix}_{state}"
                    for state in set(self.states) | set(self.state_transitions)
                    for prefix in ("next", "support_next", "weight_next", "key_next")
                }
                if target in occupied:
                    raise RegimeInitializationError(
                        f"Borrowing bound '{name}' generates '{target}', which "
                        f"collides with a declared node."
                    )
                functions[target] = get_floor_node(
                    function=cast("UserFunction", constraint.lower_bound),
                    constraint_name=name,
                )
        margin = getattr(self, "liquid", None)
        if margin is not None:
            normalized = normalize_constraints(
                constraints=cast(
                    "Mapping[FunctionName, ConstraintLike]", self.decomposed_constraints
                )
            )
            for name, constraint in normalized.items():
                surfaces = constraint.boundary_surfaces
                if (
                    surfaces is None
                    or len(surfaces) != 1
                    or isinstance(constraint.declaration, _PostDecisionLowerBound)
                ):
                    continue
                surface = surfaces[0]
                if (
                    surface.op != ">="
                    or not isinstance(surface.left, Ref)
                    or surface.left.name != margin.post_decision_state
                    or not isinstance(surface.right, Ref)
                ):
                    continue
                target = surface.right.name
                producer = functions.get(target)
                if isinstance(producer, Phased):
                    functions[target] = Phased(
                        solve=(
                            get_floor_node(
                                function=producer.solve, constraint_name=name
                            )
                            if callable(producer.solve)
                            else producer.solve
                        ),
                        simulate=(
                            get_floor_node(
                                function=producer.simulate, constraint_name=name
                            )
                            if callable(producer.simulate)
                            else producer.simulate
                        ),
                    )
                elif callable(producer):
                    functions[target] = get_floor_node(
                        function=producer, constraint_name=name
                    )
        return MappingProxyType(functions)

    @property
    def decomposed_constraints(
        self,
    ) -> MappingProxyType[FunctionName, ConstraintLike | Phased | None]:
        """`constraints` in the form the engine evaluates.

        A post-decision lower bound whose bound is a function names the target
        that publishes it. Deterministic and idempotent, like
        `decomposed_functions`.
        """
        return decompose_constraints(self.constraints)

    def _lower_collective_utility(self) -> None:
        """Derive member metadata from `functions["utility"]`."""
        declaration = self.functions.get("utility")
        if not isinstance(declaration, CollectiveUtility):
            return
        for member, utility in declaration.utilities.items():
            entry = f"utility_{member}"
            # A delegated body may still arrive from the model level. Whether
            # one ever does is completeness, which the model reports once its
            # regimes are merged.
            if utility is None:
                continue
            if entry in self.functions and self.functions[entry] is not utility:
                raise RegimeInitializationError(
                    f"The member {member!r} of this regime's "
                    f"`CollectiveUtility` and the function {entry!r} would both "
                    f"supply {member}'s utility. Declare it once, in the "
                    "`CollectiveUtility`."
                )
        object.__setattr__(self, "member_axis", tuple(declaration.utilities))
        if declaration.objective is not None:
            object.__setattr__(self, "pareto_objective", declaration.objective)

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
        value, _ = resolve_koopmans_aggregator(
            declaration=self.koopmans_aggregator, phase=phase
        )
        return value

    def get_objective_aggregator(
        self,
        phase: Literal["solve", "simulate"] = "solve",
    ) -> UserFunction | None:
        """Get the aggregator this phase chooses actions with, if separate.

        Args:
            phase: Which variant to use when the objective is `Phased`.

        Returns:
            The objective aggregator of a `Decoupled` declaration, or `None`
            when the Koopmans aggregator also chooses.

        """
        _, objective = resolve_koopmans_aggregator(
            declaration=self.koopmans_aggregator, phase=phase
        )
        return objective

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
            if isinstance(spec, Phased) and callable(spec.solve):
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
        from _lcm.constraints.processed import normalize_constraints  # noqa: PLC0415

        normalized = normalize_constraints(
            constraints=cast(
                "Mapping[FunctionName, ConstraintLike]", self.decomposed_constraints
            )
        )
        for name, constraint in normalized.items():
            if isinstance(constraint.declaration, _PostDecisionLowerBound) and callable(
                constraint.declaration.lower_bound
            ):
                result[name] = constraint.as_function(pool=result)
        return MappingProxyType(result)

    def _augment_phase_functions(
        self, functions: dict[FunctionName, UserFunction]
    ) -> dict[FunctionName, UserFunction]:
        """Add internal functions required by a specialized regime declaration."""
        return functions

    def _restore_floor_sources(
        self,
        engine_functions: Mapping[FunctionName, UserFunction | Phased | None],
    ) -> Mapping[FunctionName, UserFunction | Phased | None]:
        """Keep original declarations behind the engine's scalar checked views."""
        from _lcm.constraints.dynamic_bound import get_floor_source  # noqa: PLC0415

        restored = dict(engine_functions)
        for name, producer in engine_functions.items():
            raw = self.functions.get(name)
            if (
                callable(producer)
                and get_floor_source(function=cast("UserFunction", producer)) is raw
            ) or (
                isinstance(producer, Phased)
                and isinstance(raw, Phased)
                and all(
                    current is declared
                    or (
                        callable(current)
                        and get_floor_source(function=cast("UserFunction", current))
                        is declared
                    )
                    for current, declared in (
                        (producer.solve, raw.solve),
                        (producer.simulate, raw.simulate),
                    )
                )
            ):
                restored[name] = raw
        return MappingProxyType(restored)

    def with_engine_functions(
        self,
        *,
        engine_functions: Mapping[FunctionName, UserFunction | Phased | None],
        **other_slots: Unpack[_SlotsBesideFunctions],
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
                member's declared utility, replaces a declaration object,
                or does not reproduce what the resulting regime decomposes to.
        """
        declaration = self.functions.get("utility")
        engine_functions = self._restore_floor_sources(engine_functions)
        expected_function_names = frozenset(engine_functions)
        generated = {
            f"{name}_lower"
            for name, constraint in self.constraints.items()
            if isinstance(constraint, _PostDecisionLowerBound)
            and callable(constraint.lower_bound)
        }
        for target in generated:
            if (
                target in engine_functions
                and engine_functions[target] is not self.decomposed_functions[target]
            ):
                raise RegimeInitializationError(
                    f"Borrowing bound '{target.removesuffix('_lower')}' owns "
                    f"'{target}'; the engine mapping supplies a different producer."
                )
        engine_functions = MappingProxyType(
            {
                name: func
                for name, func in engine_functions.items()
                if name not in generated
            }
        )
        if not isinstance(declaration, CollectiveUtility):
            return self.replace(functions=engine_functions, **other_slots)

        declared_bodies = {
            f"utility_{member}": body
            for member, body in declaration.utilities.items()
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
                        f"declares. A member's utility is hers to declare."
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
        disagreement = sorted(
            expected_function_names ^ set(written.decomposed_functions)
        )
        if disagreement:
            raise RegimeInitializationError(
                f"Writing the engine mapping back would not reproduce it: "
                f"{disagreement} appears on one side only. The write-back "
                "overlays engine additions on the declarations, so it can add "
                "a function but never drop what a declaration produces."
            )
        return written

    def replace(self, **kwargs: Unpack[RegimeReplacement]) -> Regime:
        """Replace the attributes of the regime.

        Replacing a slot that carries a `CollectiveUtility` replaces the
        declaration itself, and the members are derived again from what the
        new slot says. `members`, `member_axis` and `pareto_objective` are those
        derived values, so naming one here is an error: they are read off a
        regime, never written to it.

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
    functions: Mapping[FunctionName, FunctionEntry],
) -> Mapping[FunctionName, UserFunction | Phased | None]:
    """Replace a `CollectiveUtility` by one utility entry per member.

    Args:
        functions: A regime's `functions` as declared.

    Returns:
        The same mapping with any `CollectiveUtility` under `"utility"`
        replaced by one `utility_<s>` entry per member, emitted in the
        order the household declares them so that the result does not depend on
        the order the entries reached the regime in. A member whose body
        is delegated keeps the entry the mapping already carries. A mapping
        declaring no household is returned unchanged, which makes the
        transformation idempotent.
    """
    declaration = functions.get("utility")
    if not isinstance(declaration, CollectiveUtility):
        return cast("Mapping[FunctionName, UserFunction | Phased | None]", functions)
    member_entries = {f"utility_{member}" for member in declaration.utilities}
    decomposed: dict[FunctionName, UserFunction | Phased | None] = {
        name: cast("UserFunction | Phased | None", func)
        for name, func in functions.items()
        if name != "utility" and name not in member_entries
    }
    for member, utility in declaration.utilities.items():
        entry = f"utility_{member}"
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
    constraints: Mapping[FunctionName, ConstraintEntry],
) -> MappingProxyType[FunctionName, ConstraintLike | Phased | None]:
    """Return a regime's constraints in the form the engine evaluates.

    Args:
        constraints: A regime's `constraints` as declared.

    Returns:
        The same constraints, with every post-decision lower bound whose bound
        is a function naming the target `<name>_lower` that publishes it.
    """
    return MappingProxyType(
        {
            name: (
                dataclasses.replace(constraint, lower_target=f"{name}_lower")
                if isinstance(constraint, _PostDecisionLowerBound)
                and callable(constraint.lower_bound)
                else cast("ConstraintLike | Phased | None", constraint)
            )
            for name, constraint in constraints.items()
        }
    )


def _resolve_phase_variant(
    *, value: UserFunction | Phased | None, phase: Literal["solve", "simulate"]
) -> UserFunction:
    """Return the variant of a possibly `Phased` entry that applies in `phase`."""
    if isinstance(value, Phased):
        value = value.solve if phase == "solve" else value.simulate
    return cast("UserFunction", value)
