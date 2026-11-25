"""Validation helpers for the user-facing `Regime`.

`validate_regime` backs `Regime.__post_init__` and checks what a regime declares
on its own. `validate_regime_law` runs once the model binds the regime's law and
checks what depends on it. The helpers raise nothing themselves — instead they
collect error messages, which are aggregated into a single
`RegimeInitializationError`.
Splitting the validators out of the public module keeps `lcm.regime` to class
definitions.

"""

import ast
import inspect
import math
import textwrap
from collections import Counter
from collections.abc import Iterator, Mapping
from types import MappingProxyType, UnionType
from typing import TYPE_CHECKING, cast

from beartype import beartype

from _lcm.beartype_conf import REGIME_CONF
from _lcm.certainty_equivalent import PowerMean, aggregates_nonlinearly
from _lcm.constraints.processed import ConstraintLike
from _lcm.grids import Grid
from _lcm.identity_transition import _IdentityTransition
from _lcm.processes.base import _ContinuousStochasticProcess
from _lcm.regime_building.collective import member_name
from _lcm.regime_building.phases import (
    normalize_regime_phases,
    validate_law_free_phase_grammar,
)
from _lcm.regime_law import DecomposedTransition, RegimeLaw, RegimeLaws
from _lcm.typing import (
    FunctionName,
    ProcessName,
    RegimeName,
    StateName,
    StateOrActionName,
)
from _lcm.utils.error_messages import format_messages, path_segment_name_errors
from _lcm.utils.functools import is_user_function
from lcm.certainty_equivalent import CertaintyEquivalent, LinearExpectation
from lcm.decoupled import Decoupled
from lcm.exceptions import RegimeInitializationError
from lcm.exits import Consent, Exit, MutualConsent, Unilateral
from lcm.grids import DiscreteGrid
from lcm.koopmans_aggregation import CESAggregator, LinearAggregator
from lcm.phased import Phased
from lcm.solvers import NBEGM, NNBEGM, GridSearch
from lcm.transition import (
    AgeSpecializedFunction,
    AgeSpecializedGrid,
    ByAge,
    JointTransition,
    StochasticTransition,
)
from lcm.typing import Phase, UserFunction

if TYPE_CHECKING:
    from lcm.regime import Regime, StateEntry, StateTransitionEntry
else:
    # `lcm.regime` imports this module, so neither `Regime` nor its field aliases
    # are importable here at runtime; a regime's own constructor checks its fields,
    # and the validators below report every malformed entry with their own message.
    type Regime = object
    type StateEntry = object
    type StateTransitionEntry = object

# A declared law or slot entry the walkers below unwrap: a regime law, a state
# law as declared, or a function or constraint. State laws reach the walkers
# before their entries are validated, so at runtime the alias admits any value.
type _TransitionValue = DecomposedTransition | StateTransitionEntry | ConstraintLike
# What `_iter_transition_nodes` yields once `Phased` sides and per-target cells
# are unwrapped.
type _TransitionLeaf = ConstraintLike | StochasticTransition | ByAge | None


@beartype(conf=REGIME_CONF)
def _grid_mapping_errors(
    *,
    attr_name: str,
    mapping: Mapping[StateOrActionName, StateEntry],
    allow_phase_variants: bool,
    allow_age_specialized_grid: bool = False,
) -> tuple[str, ...]:
    """Collect key/value type errors for a grid-valued mapping (states/actions)."""
    allowed: type[Grid] | UnionType = Grid | Phased if allow_phase_variants else Grid
    suffix = " or Phased" if allow_phase_variants else ""
    if allow_age_specialized_grid:
        # An age-varying continuous-state grid is a valid state (states only).
        allowed = allowed | AgeSpecializedGrid
        suffix += " or AgeSpecializedGrid"
    error_messages: list[str] = []
    for k, v in mapping.items():
        if not isinstance(k, str):
            error_messages.append(f"{attr_name} key {k!r} must be a string.")
        if v is None:
            # A mask of a model-level entry; bound at model build.
            continue
        if not allow_phase_variants and isinstance(v, Phased):
            error_messages.append(
                f"{attr_name}['{k}'] cannot be phase-variant: the simulated "
                f"argmax must range over the same menu the value function was "
                f"computed for."
            )
        elif not isinstance(v, allowed):
            error_messages.append(
                f"{attr_name} value {v!r} must be an LCM grid{suffix}."
            )
    return tuple(error_messages)


@beartype(conf=REGIME_CONF)
def _callable_mapping_errors(
    *,
    attr_name: str,
    mapping: Mapping[FunctionName, ConstraintLike | Phased | None],
    allow_phase_variants: bool,
) -> tuple[str, ...]:
    """Collect key/value type errors for a callable-valued mapping."""
    error_messages: list[str] = []
    for k, v in mapping.items():
        if not isinstance(k, str):
            error_messages.append(f"{attr_name} key {k!r} must be a string.")
        if v is None:
            # A mask of a model-level entry; bound at model build.
            continue
        if isinstance(v, Phased):
            if not allow_phase_variants:
                error_messages.append(
                    f"{attr_name}['{k}'] cannot be phase-variant: a "
                    f"phase-specific feasible set would let the simulated "
                    f"argmax range over actions the value function was never "
                    f"computed for."
                )
        elif not callable(v):
            error_messages.append(
                f"{attr_name} value {v!r} must be a Condition or a callable."
            )
    return tuple(error_messages)


def validate_regimes(*, regimes: Mapping[RegimeName, Regime], laws: RegimeLaws) -> None:
    """Validate every regime against the law its model binds for it."""
    for name, regime in regimes.items():
        validate_regime_law(regime, law=laws[name])


def validate_exit_targets(
    *, regimes: Mapping[RegimeName, Regime], laws: RegimeLaws
) -> None:
    """Bind declared destinations to real policy roles and complete phase states."""
    from _lcm.exits import exit_for_phase  # noqa: PLC0415 - phase validator cycle

    for source, regime in regimes.items():
        for phase in ("solve", "simulate"):
            declaration = exit_for_phase(declaration=regime.exit, phase=phase)
            if declaration is None:
                continue
            _validate_renegotiation(
                declaration=declaration, regime=regime, law=laws[source], phase=phase
            )
            _validate_optimum_comparison(declaration=declaration, regime=regime)
            _validate_settlement(declaration=declaration, regime=regime)
            for target in declaration.targets.values():
                context = f"Exit of {source}"
                if target.regime not in regimes:
                    raise RegimeInitializationError(
                        f"{context} names unknown destination {target.regime!r}."
                    )
                destination = regimes[target.regime]
                if target.role not in _declarable_roles(destination):
                    raise RegimeInitializationError(
                        f"{context} has invalid destination role {target.role!r}."
                    )
                _validate_exit_projection(
                    context=context,
                    projection=target.projection,
                    destination=destination,
                    law=laws[target.regime],
                    phase=phase,
                )
            _validate_counterparty(
                declaration=declaration,
                context=f"Exit of {source}, counterparty",
                regimes=regimes,
                laws=laws,
                phase=phase,
            )


def _validate_renegotiation(
    *, declaration: Exit, regime: Regime, law: RegimeLaw, phase: Phase
) -> None:
    """Require the ordered two-member domain before native queries are built."""
    if (
        not isinstance(declaration.rule, Unilateral)
        or declaration.rule.renegotiate is None
    ):
        return
    if len(regime.member_axis or ()) != 2:  # noqa: PLR2004 - directional two-member rule
        raise RegimeInitializationError("Renegotiate requires two collective members.")
    spec = normalize_regime_phases(regime, law=law)
    grid = getattr(
        spec, "solution" if phase == "solve" else "simulation"
    ).grid_states.get(declaration.rule.renegotiate.state)
    if not isinstance(grid, DiscreteGrid):
        raise RegimeInitializationError(
            "Renegotiate must name a declared discrete grid state."
        )
    if not grid.ordered:
        raise RegimeInitializationError("Renegotiate needs an ordered discrete state.")
    state = declaration.rule.renegotiate.state
    phase_spec = getattr(spec, "solution" if phase == "solve" else "simulation")
    transition = phase_spec.state_transitions.get(state)
    laws = transition.values() if isinstance(transition, Mapping) else (transition,)
    if any(
        value is not None and not isinstance(value, _IdentityTransition)
        for value in laws
    ):
        raise RegimeInitializationError(
            "Renegotiate carries its selected node: declare "
            f"fixed_transition('{state}') for its state law."
        )
    if any(
        state in kernel.outputs
        for kernels in phase_spec.joint_transitions.values()
        for kernel in kernels.values()
    ):
        raise RegimeInitializationError(
            "Renegotiate carries its selected node through fixed_transition, "
            "not through a joint state update."
        )


def _validate_optimum_comparison(*, declaration: Exit, regime: Regime) -> None:
    """Require a deterministic household optimum for a rule that compares there."""
    from _lcm.exits import compares_at_optimum  # noqa: PLC0415 - phase validator cycle

    if compares_at_optimum(declaration.rule) and regime.taste_shocks is not None:
        raise RegimeInitializationError(
            "An exit rule that compares at the household optimum needs a regime "
            "without taste shocks."
        )


def _validate_settlement(*, declaration: Exit, regime: Regime) -> None:
    """Require two members, a target that reads the settlement and a free name."""
    rule = declaration.rule
    if not isinstance(rule, MutualConsent) or rule.settle is None:
        return
    name = rule.settle.name
    if len(regime.member_axis or ()) != 2:  # noqa: PLR2004 - directional two-member rule
        raise RegimeInitializationError("Settlement requires two collective members.")
    # The author's names and the engine's: a household's `utility` is bound as
    # `utility_<s>` per member, and a settlement may shadow neither.
    taken = {
        *regime.states,
        *regime.actions,
        *regime.functions,
        *regime.decomposed_functions,
    }
    if name in taken:
        raise RegimeInitializationError(
            f"Settlement {name!r} has the name of a state, action or function of the "
            "regime."
        )
    callbacks = [
        function
        for target in declaration.targets.values()
        for function in (
            *cast("Mapping[str, UserFunction]", target.projection).values(),
            *(
                ()
                if target.utility_adjustment is None
                else (target.utility_adjustment,)
            ),
        )
    ]
    if not any(
        name in inspect.signature(function).parameters for function in callbacks
    ):
        raise RegimeInitializationError(
            f"Settlement {name!r} is read by no exit-target projection or utility "
            "adjustment."
        )


def _declarable_roles(regime: Regime) -> tuple[str | None, ...]:
    """Return the roles a declaration may name for a member of `regime`.

    A household's members by name. The member `"self"` is no role, so a
    declaration names it by naming none.
    """
    return tuple(None if member == "self" else member for member in regime.members)


def _validate_counterparty(
    *,
    declaration: Exit,
    context: str,
    regimes: Mapping[RegimeName, Regime],
    laws: RegimeLaws,
    phase: Phase,
) -> None:
    """Bind the other formation member to its own complete Raw stay context."""
    if not isinstance(declaration.rule, Consent):
        return
    counterparty = declaration.rule.counterparty
    if counterparty.regime not in regimes:
        raise RegimeInitializationError(
            f"{context} names unknown regime {counterparty.regime!r}."
        )
    # The role belongs to the formation destination, rather than its singleton Raw.
    for target in declaration.targets.values():
        destination = regimes[target.regime]
        if (
            counterparty.role not in _declarable_roles(destination)
            or counterparty.role == target.role
        ):
            raise RegimeInitializationError(
                f"{context} has invalid destination role {counterparty.role!r}."
            )
    _validate_exit_projection(
        context=context,
        projection=counterparty.projection,
        destination=regimes[counterparty.regime],
        law=laws[counterparty.regime],
        phase=phase,
    )


def _validate_exit_projection(
    *,
    context: str,
    projection: Mapping[StateName, UserFunction] | Phased,
    destination: Regime,
    law: RegimeLaw,
    phase: Phase,
) -> None:
    """Require a complete mapping into the actual phase's destination state."""
    spec = normalize_regime_phases(destination, law=law)
    expected = set(
        getattr(spec, "solution" if phase == "solve" else "simulation").grid_states
    )
    if not isinstance(projection, Mapping) or set(projection) != expected:
        raise RegimeInitializationError(
            f"{context} {phase} projection must name destination states "
            f"{sorted(expected)!r}."
        )


def validate_regime(regime: Regime) -> None:
    """Validate what a regime declares on its own, without its law.

    Runs whenever a `Regime` is constructed. What depends on the law the model
    binds from `Model(edges=...)` (terminality, the law's grammar, the variables
    it reads, gated edges) is checked by `validate_regime_law`.

    Args:
        regime: The regime to validate.

    Raises:
        RegimeInitializationError: On the first group of violated declarations.
    """
    _fail_if_solver_cannot_solve_the_members(regime)
    if regime.member_axis is not None:
        _validate_collective_regime(regime)
    elif regime.pareto_objective is not None:
        raise RegimeInitializationError(
            "`pareto_objective` declares how a collective regime's "
            "household weighs its members; it is only meaningful "
            "for a household declared with a `CollectiveUtility`. Omit it "
            "for a singleton regime."
        )
    member_mapping_errors = _member_mapping_errors(regime)
    if member_mapping_errors:
        raise RegimeInitializationError(format_messages(member_mapping_errors))

    _validate_mapping_contents(regime)
    _validate_exits(regime)
    _validate_logical_consistency(regime)
    validate_law_free_phase_grammar(regime)


def _member_mapping_errors(regime: Regime) -> tuple[str, ...]:
    """Collect errors for a Koopmans aggregator or certainty equivalent per member.

    A slot keyed by member names every member of a household, and a `Decoupled`
    aggregator chooses for the household as a whole, so no member declares one.
    """
    error_messages: list[str] = []
    for slot in ("koopmans_aggregator", "certainty_equivalent"):
        declared = getattr(regime, slot)
        if not isinstance(declared, Mapping):
            continue
        if regime.member_axis is None:
            error_messages.append(
                f"`{slot}` is keyed by member, which only a household declared "
                f"with a `CollectiveUtility` has. Declare one `{slot}` for this "
                "regime."
            )
        elif set(declared) != set(regime.members):
            error_messages.append(
                f"`{slot}` must be keyed by the members {regime.members!r}; got "
                f"{tuple(declared)!r}."
            )
    if isinstance(regime.koopmans_aggregator, Mapping) and any(
        isinstance(value, Decoupled) for value in regime.koopmans_aggregator.values()
    ):
        error_messages.append(
            "A `Decoupled` aggregator cannot be declared per member: the "
            "household chooses on the Pareto-weighted sum of its members' "
            "objectives. Declare `koopmans_aggregator=Decoupled(...)` for the "
            "household."
        )
    return tuple(error_messages)


def _fail_if_solver_cannot_solve_the_members(regime: Regime) -> None:
    """Refuse a regime of several members under a solver solving one only.

    Raises:
        NotImplementedError: If the regime has more than one member and its
            solver does not declare `supports_several_members`.
    """
    if (
        len(regime.members) > 1
        and not regime.solver.capabilities.supports_several_members
    ):
        raise NotImplementedError(
            f"The {type(regime.solver).__name__} solver solves a regime with one "
            f"member only; this regime has the members {regime.members!r}. Use a "
            "solver that supports several members, such as `solver=GridSearch()`."
        )


def _validate_exits(regime: Regime) -> None:
    """Require a destination for each of the source's members."""
    if regime.exit is None:
        return
    expected = set(regime.members)
    if set(regime.exit.targets) != expected:
        raise RegimeInitializationError(f"Exit targets must name {sorted(expected)!r}.")


# keyword-only-exempt: primary-argument=regime
def validate_regime_law(regime: Regime, *, law: RegimeLaw) -> None:
    """Validate a regime against the law its model binds for it.

    The regime itself has passed `validate_regime` when it was constructed;
    this checks only what depends on the law. The grammar and age-specialization
    checks share one period-independent view of the law
    (`RegimeLaw.decomposed_transition`).

    Args:
        regime: The regime the law moves.
        law: The law the model binds for the regime.

    Raises:
        RegimeInitializationError: On the first group of violated declarations.
    """
    if law.gated_edges:
        _validate_gated_edges(regime=regime, law=law)
    transition = law.decomposed_transition
    _validate_law_consistency(regime, transition=transition, terminal=law.terminal)
    # The law's phase grammar (its variants, terminality against carried
    # states and the aggregator) is validated by the normalizer.
    normalize_regime_phases(regime, law=law)


def fail_if_a_joint_target_is_unreachable(
    *,
    user_regimes: Mapping[RegimeName, Regime],
    targets_by_regime: Mapping[RegimeName, frozenset[RegimeName]],
) -> None:
    """Reject a joint kernel toward a target its source has no edge into.

    Args:
        user_regimes: The model's regimes, by name.
        targets_by_regime: Per source regime, every target the graph's edges
            lead to in either phase.

    Raises:
        RegimeInitializationError: If a regime declares `joint_transitions`
            toward a target outside its edges.
    """
    error_messages = [
        f"Regime '{name}': joint-transition targets must be reachable under the "
        f"declared edges {sorted(targets)}; got '{target_name}'."
        for name, regime in user_regimes.items()
        if regime.joint_transitions
        for targets in (targets_by_regime.get(name, frozenset()),)
        for target_name in regime.joint_transitions
        if target_name not in targets
    ]
    if error_messages:
        raise RegimeInitializationError(format_messages(error_messages))


def _validate_collective_regime(regime: Regime) -> None:
    """Validate a collective (member-valued) regime.

    Both the terminal and the non-terminal continuation cases are implemented.
    Checked here are the properties a bare `Regime` already
    determines on its own: a non-empty, duplicate-free `members` tuple, and — if
    `pareto_objective` is given — weight keys matching the member names,
    with every constant weight finite, non-negative, and a positive total (the
    zero-safe scalarization in `collective._weighted_sum` treats an exact-zero
    weight as a deliberate exclusion, so a non-finite or negative weight, or an
    all-zero declaration, would silently produce a meaningless or undefined
    household objective rather than erroring).

    What a collective regime needs but a model-level slot may supply — the
    per-member `utility_<s>` functions — is completeness, so
    `_validate_completeness` checks it once the regimes are finalized at model
    build. The household argmax runs over whatever action product the regime
    declares, which may be continuous or empty.

    EV1 taste shocks raise `NotImplementedError` unless exactly one member can
    carry a positive weight (`_fail_if_taste_shocks_lack_one_decider`). Whether
    the solver handles the member count is checked for every regime, by
    `_fail_if_solver_cannot_solve_the_members`.

    Called from `Regime.__post_init__` only when `member_axis is not None`, so
    the default singleton path never reaches it.
    """
    members = cast("tuple[str, ...]", regime.member_axis)

    _fail_if_taste_shocks_lack_one_decider(regime=regime, members=members)
    error_messages: list[str] = []
    error_messages.extend(_members_tuple_errors(members))
    error_messages.extend(path_segment_name_errors(kind="Member", names=members))
    error_messages.extend(_collective_weights_errors(regime=regime, members=members))

    if error_messages:
        raise RegimeInitializationError(format_messages(error_messages))


def _fail_if_taste_shocks_lack_one_decider(
    *, regime: Regime, members: tuple[str, ...]
) -> None:
    """Admit EV1 taste shocks in a household only where one member decides.

    The member with the positive Pareto weight decides, and the household's
    smoothed choice is that member's. What the declaration settles is checked
    here:

    - undeclared weights are equal, so only a household of one has one decider;
    - constant weights must have exactly one positive entry;
    - a weight declared as a function is admitted, and the solve checks at
      every cell that exactly one member's weight is positive.

    Raises:
        NotImplementedError: If the regime declares taste shocks and one of the
            cases above refuses them.
    """
    if regime.taste_shocks is None:
        return
    objective = regime.pareto_objective
    weights: Mapping[str, UserFunction | float] = (
        dict.fromkeys(members, 1.0 / len(members))
        if objective is None
        else objective.weights
    )
    constants = [weight for weight in weights.values() if not callable(weight)]
    n_positive = sum(weight > 0 for weight in constants)
    if n_positive > 1 or (len(constants) == len(weights) and n_positive != 1):
        raise NotImplementedError(
            "EV1 taste shocks in a household need exactly one member with a "
            "positive Pareto weight, the member who decides; got weights "
            f"{dict(weights)!r}. Give every other member weight 0, or drop "
            "`taste_shocks` on this regime."
        )


def _members_tuple_errors(members: tuple[str, ...]) -> tuple[str, ...]:
    """Collect errors for an empty or duplicate-containing `members` tuple."""
    if not members:
        return ("`members` must be a non-empty tuple for a collective regime.",)

    duplicate_counts = Counter(members)
    duplicates = sorted(name for name, count in duplicate_counts.items() if count > 1)
    if duplicates:
        return (
            (
                f"`members` must not contain duplicate names; got "
                f"{members} (duplicated: {duplicates})."
            ),
        )
    return ()


def _collective_weights_errors(
    *, regime: Regime, members: tuple[str, ...]
) -> tuple[str, ...]:
    """Collect errors for a collective regime's `pareto_objective`.

    Weight keys must match `members`, and a weight declared as a constant
    must be finite, non-negative, and — across the constants — leave a positive
    total. The zero-safe scalarization treats an exact-zero weight as a
    deliberate exclusion, so a non-finite or negative weight, or an all-zero
    declaration, would silently produce a meaningless or undefined household
    objective rather than erroring. A weight declared as a function is checked
    against the parameters it is solved with, where its values exist.
    """
    objective = regime.pareto_objective
    if objective is None:
        return ()

    declared = set(objective.weights)
    if declared != set(members):
        missing = sorted(set(members) - declared)
        extra = sorted(declared - set(members))
        return (
            (
                f"`pareto_objective.weights` weighs {sorted(declared)}, but "
                f"this regime's members are {sorted(members)}: "
                f"missing {missing}, unknown {extra}."
            ),
        )

    constants = {
        name: float(weight)
        for name, weight in objective.weights.items()
        if not callable(weight)
    }
    non_finite_or_negative = {
        name: weight
        for name, weight in constants.items()
        if not math.isfinite(weight) or weight < 0
    }
    if non_finite_or_negative:
        return (
            (
                "`pareto_objective.weights` must be finite and non-negative "
                f"for every member; got {non_finite_or_negative}."
            ),
        )
    if len(constants) == len(objective.weights) and sum(constants.values()) <= 0:
        return (
            (
                "`pareto_objective.weights` must leave a positive total; an "
                "all-zero declaration leaves the household scalarization "
                "identically zero for every action, so the argmax is undefined."
            ),
        )
    return ()


def _validate_gated_edges(*, regime: Regime, law: RegimeLaw) -> None:
    """Validate the gated edges a regime's law declares (regime-local part).

    Checks the properties knowable without the other
    regimes: the gate is a plain boolean callable (a stochastic, probabilistic
    gate — a `StochasticTransition` — is not implemented); the legs cover the
    SOURCE's member structure (exactly one leg for a singleton source, one
    per member for a collective source); route keys are valid
    parameter-path segments. Cross-regime properties — the
    target and fallback regimes exist, the member names resolve, the
    projections cover the reference states — are validated at model processing.

    A gated-edge SOURCE is restricted to the `GridSearch` solver with no taste
    shocks or certainty equivalent: the source reads the folded `Wbar` through
    the grid-search continuation machinery, which no DC-EGM, taste-shock, or
    certainty-equivalent source has.

    Called only for a law with gated edges, so the default path never reaches
    it.
    """
    _fail_if_gated_edge_source_out_of_scope(regime)

    error_messages: list[str] = []
    source_members = regime.members

    for target_name, edge in law.gated_edges.items():
        prefix = f"the value-dependent transition into {target_name!r}: "
        if isinstance(edge.gate, StochasticTransition):
            error_messages.append(
                f"{prefix}the gate must be a plain boolean function. A "
                "`StochasticTransition` (stochastic / probabilistic gate) is not "
                "implemented — a gated edge routes on a boolean gate."
            )
        elif not callable(edge.gate):
            error_messages.append(f"{prefix}the gate must be a callable.")
        error_messages.extend(
            path_segment_name_errors(kind=f"{prefix}route", names=edge.legs)
        )
        if not edge.legs:
            error_messages.append(f"{prefix}must declare at least one leg.")
        elif "self" in source_members:
            # The member `"self"` declares its one leg under any key.
            if len(edge.legs) != 1:
                error_messages.append(
                    f"{prefix}a singleton source regime must declare exactly one "
                    f"leg (got {sorted(edge.legs)})."
                )
        elif set(edge.legs) != set(source_members):
            error_messages.append(
                f"{prefix}the legs must be keyed by exactly the source "
                f"members {sorted(source_members)}; got "
                f"{sorted(edge.legs)}."
            )

    if error_messages:
        raise RegimeInitializationError(format_messages(error_messages))


def _fail_if_gated_edge_source_out_of_scope(regime: Regime) -> None:
    """Reject a gated-edge source outside the GridSearch / no-shock scope."""
    if not isinstance(regime.solver, GridSearch):
        raise NotImplementedError(
            "Gated edges are only implemented for GridSearch source "
            "regimes: the source reads the folded continuation through the "
            "grid-search machinery, which a DC-EGM regime does not have. Use "
            "`solver=GridSearch()`."
        )
    if regime.taste_shocks is not None:
        raise NotImplementedError(
            "Gated edges on a taste-shock source regime are not implemented."
        )
    if isinstance(regime.koopmans_aggregator, Decoupled):
        raise NotImplementedError(
            "Gated edges on a source regime with a `Decoupled` Koopmans "
            "aggregator are not implemented: which of its two values the gate "
            "compares is not yet defined."
        )
    if aggregates_nonlinearly(regime.certainty_equivalent):
        raise NotImplementedError(
            "Gated edges on a certainty-equivalent source regime are not implemented."
        )


def _validate_mapping_contents(regime: Regime) -> None:
    """Exhaustively check key/value types of `regime`'s mapping fields.

    Beartype on `Regime` catches top-level type mismatches and a sampled
    Mapping entry, but does not deep-check every key/value of a Mapping
    parameter — especially when the value type is a `Callable`/`Protocol`,
    which beartype skips entirely. This function fills that gap by
    iterating each entry and reporting all type violations together via
    the standard `error_messages` aggregator.

    """
    error_messages = [
        *_grid_mapping_errors(
            attr_name="states",
            mapping=regime.states,
            allow_phase_variants=True,
            allow_age_specialized_grid=True,
        ),
        *_grid_mapping_errors(
            attr_name="actions", mapping=regime.actions, allow_phase_variants=False
        ),
        *_callable_mapping_errors(
            attr_name="functions",
            mapping=regime.decomposed_functions,
            allow_phase_variants=True,
        ),
        *_callable_mapping_errors(
            attr_name="constraints",
            mapping=regime.decomposed_constraints,
            allow_phase_variants=False,
        ),
    ]

    if error_messages:
        msg = format_messages(error_messages)
        raise RegimeInitializationError(msg)


def _validate_logical_consistency(regime: Regime) -> None:
    """Validate the local, value-shape consistency of a regime.

    Completeness properties (a `utility` entry, state-transition coverage,
    state/action overlap, distributed-grid rules) may be satisfied only after
    model-level slots are merged in; `_validate_completeness` checks them when
    the regimes are finalized at model build.
    """
    error_messages: list[str] = []

    all_function_names = [
        *regime.decomposed_constraints.keys(),
        *regime.decomposed_functions.keys(),
    ]
    error_messages.extend(
        path_segment_name_errors(
            kind="Function and constraint", names=all_function_names
        )
    )

    next_prefixed = [name for name in all_function_names if name.startswith("next_")]
    if next_prefixed:
        error_messages.append(
            f"Function names must not start with 'next_' — this prefix is "
            f"reserved for auto-generated state transition functions. "
            f"Invalid names: {next_prefixed}.",
        )

    error_messages.extend(
        path_segment_name_errors(
            kind="State and action",
            names=[*regime.states.keys(), *regime.actions.keys()],
        )
    )

    error_messages.extend(_state_transition_grammar_errors(regime))
    error_messages.extend(_joint_transition_grammar_errors(regime))
    error_messages.extend(_state_transition_marker_errors(regime.state_transitions))

    if error_messages:
        msg = format_messages(error_messages)
        raise RegimeInitializationError(msg)


# keyword-only-exempt: primary-argument=regime
def _validate_law_consistency(
    regime: Regime, *, transition: DecomposedTransition, terminal: bool
) -> None:
    """Validate the regime's law and what the regime declares against it.

    Args:
        regime: The regime the law moves.
        transition: The law's period-independent, decomposed view.
        terminal: Whether the regime has no outgoing edges.
    """
    error_messages: list[str] = []
    if terminal and regime.joint_transitions:
        error_messages.append(
            _terminal_regime_message(
                slot="joint_transitions", names=sorted(regime.joint_transitions)
            )
        )
    error_messages.extend(_regime_transition_grammar_errors(transition))
    error_messages.extend(
        _age_specialized_scope_errors(
            transition=transition,
            functions=regime.decomposed_functions,
            constraints=regime.decomposed_constraints,
            terminal=terminal,
        )
    )

    if error_messages:
        msg = format_messages(error_messages)
        raise RegimeInitializationError(msg)


def _terminal_regime_message(*, slot: str, names: list[str]) -> str:
    """Explain why a regime is terminal and how to give it a next period."""
    return (
        "This regime is terminal: it has no outgoing edge in `Model(edges=...)`, "
        f"so no next period reads its `{slot}` {names}. Declare an edge from it "
        f"in `Model(edges=...)`, or drop its `{slot}` (or the regime)."
    )


def _iter_transition_nodes(value: _TransitionValue) -> Iterator[_TransitionLeaf]:
    """Yield leaf transition nodes, unwrapping `Phased` sides and per-target dicts."""
    if isinstance(value, Phased):
        yield from _iter_transition_nodes(value.solve)
        yield from _iter_transition_nodes(value.simulate)
    elif isinstance(value, Mapping):
        for cell in value.values():
            yield from _iter_transition_nodes(cell)
    else:
        yield value


@beartype(conf=REGIME_CONF)
def _state_transition_marker_errors(
    state_transitions: Mapping[StateName, StateTransitionEntry],
) -> tuple[str, ...]:
    """Collect errors for `AgeSpecializedFunction` markers inside state transitions."""
    error_messages: list[str] = []
    for name, value in state_transitions.items():
        for node in _iter_transition_nodes(value):
            if isinstance(node, StochasticTransition) and isinstance(
                node.func, AgeSpecializedFunction
            ):
                error_messages.append(
                    f"state_transitions['{name}']: a `StochasticTransition` "
                    "wrapping an "
                    f"`AgeSpecializedFunction` (a policy-specialized stochastic "
                    f"transition) is not supported.",
                )
            elif isinstance(node, AgeSpecializedFunction):
                error_messages.append(
                    f"state_transitions['{name}']: an `AgeSpecializedFunction` cannot "
                    f"be a state-transition value. Express the policy-dependent law of "
                    f"motion as a plain transition function that reads an "
                    f"`AgeSpecializedFunction` entry of `functions` instead.",
                )
    return tuple(error_messages)


def _first_age_specialized_ancestor_of_transition(
    *,
    transition: DecomposedTransition,
    functions: Mapping[FunctionName, UserFunction | Phased | None],
    constraints: Mapping[FunctionName, ConstraintLike | Phased | None],
) -> str | None:
    """Return the name of an `AgeSpecializedFunction` the transition reads, if any.

    Walks the regime transition's parameter names transitively through plain
    entries of `functions` and `constraints` (unwrapping `Phased` sides, per-target
    dicts, and `StochasticTransition` wrappers along the way) — both are documented,
    equally legal homes for an `AgeSpecializedFunction`. Returns the first
    specialized function name reached, or `None` when the transition's dependency
    graph is policy-free.
    """
    pool: dict[FunctionName, ConstraintLike | Phased | None] = {
        **functions,
        **constraints,
    }
    specialized_names = {
        name
        for name, value in pool.items()
        if any(
            isinstance(node, AgeSpecializedFunction)
            for node in _iter_transition_nodes(value)
        )
    }
    if transition is None or not specialized_names:
        return None

    seen: set[str] = set()
    stack = list(_transition_node_arg_names(transition))
    while stack:
        name = stack.pop()
        if name in seen:
            continue
        seen.add(name)
        if name in specialized_names:
            return name
        if name in pool:
            stack.extend(_transition_node_arg_names(pool[name]))
    return None


def _transition_node_arg_names(value: _TransitionValue) -> tuple[str, ...]:
    """List the parameter names of every callable node inside a transition value."""
    names: list[str] = []
    for node in _iter_transition_nodes(value):
        func = node.func if isinstance(node, StochasticTransition) else node
        if callable(func):
            try:
                names.extend(inspect.signature(func).parameters)
            except TypeError, ValueError:
                continue
    return tuple(names)


def _age_specialized_scope_errors(
    *,
    transition: DecomposedTransition,
    functions: Mapping[FunctionName, UserFunction | Phased | None],
    constraints: Mapping[FunctionName, ConstraintLike | Phased | None],
    terminal: bool,
) -> tuple[str, ...]:
    """Reject the `AgeSpecializedFunction` compositions that are out of scope.

    `AgeSpecializedFunction` is supported in `functions` and `constraints` of
    non-terminal regimes only. Rejected — loudly, before any per-period program
    is built:

    - a regime law that is (or contains) an
      `AgeSpecializedFunction` — a policy-specialized *regime* transition;
    - a regime transition whose dependency graph reads an `AgeSpecializedFunction`
      function (directly or through plain helper functions) — regime-transition
      probabilities are built once, not per period, so a policy-specialized
      value flowing into them would reuse one age's policy closure everywhere;
    - any `AgeSpecializedFunction` in a terminal regime — the terminal value program is
      built once and shared across all periods.

    The markers inside state transitions need no law and are rejected by
    `_state_transition_marker_errors`.
    """
    error_messages: list[str] = []

    if any(
        isinstance(node, AgeSpecializedFunction)
        or (
            isinstance(node, StochasticTransition)
            and isinstance(node.func, AgeSpecializedFunction)
        )
        for node in _iter_transition_nodes(transition)
    ):
        error_messages.append(
            "A regime transition law cannot be an `AgeSpecializedFunction`, bare or "
            "wrapped in `StochasticTransition`: age-specialized regime transitions are "
            "not supported. Specialize `functions` or `constraints` instead.",
        )

    specialized_ancestor = _first_age_specialized_ancestor_of_transition(
        transition=transition, functions=functions, constraints=constraints
    )
    if specialized_ancestor is not None:
        error_messages.append(
            f"The regime transition law depends on the `AgeSpecializedFunction` "
            f"'{specialized_ancestor}'. Regime-transition probabilities are built "
            f"once, not per period, so a policy-specialized value flowing into "
            f"them would silently reuse one age's policy closure across all "
            f"periods. Route the policy dependency through `functions` consumed "
            f"by utility, constraints, or state transitions instead.",
        )

    if terminal:
        for slot_name, slot in (("functions", functions), ("constraints", constraints)):
            for name, value in slot.items():
                if any(
                    isinstance(node, AgeSpecializedFunction)
                    for node in _iter_transition_nodes(value)
                ):
                    error_messages.append(
                        f"{slot_name}['{name}']: `AgeSpecializedFunction` is not "
                        f"supported in a terminal regime — the terminal value "
                        f"program is built once and shared across all periods.",
                    )

    return tuple(error_messages)


def _regime_transition_grammar_errors(
    transition: DecomposedTransition,
) -> tuple[str, ...]:
    """Validate the vocabulary of a regime's law between regimes.

    A `Phased` container's sides are each held to the bare vocabulary
    (callable, `StochasticTransition`, or a per-target dict); per-target cells
    must be `StochasticTransition`-wrapped probability functions.
    """
    error_messages: list[str] = []
    sides = (
        (
            (transition.solve, " solve variant"),
            (transition.simulate, " simulate variant"),
        )
        if isinstance(transition, Phased)
        else ((transition, ""),)
    )
    for side, label in sides:
        if not isinstance(side, Mapping):
            continue
        if not side:
            error_messages.append(
                f"transition{label}: an empty per-target dict declares no "
                f"candidate targets — a regime without outgoing edges in "
                f"`Model(edges=...)` is terminal.",
            )
        for target_regime_name, cell in side.items():
            if not isinstance(target_regime_name, str):
                error_messages.append(
                    f"transition{label} per-target dict key {target_regime_name!r} "
                    f"must be a string.",
                )
            if isinstance(cell, Phased):
                error_messages.append(
                    f"transition{label}['{target_regime_name}'] cannot be `Phased` — "
                    f"`Phased` is outermost-only: wrap the whole entry, e.g. "
                    f"`Phased(solve={{...}}, simulate={{...}})`.",
                )
            elif not isinstance(cell, StochasticTransition):
                error_messages.append(
                    f"transition{label}['{target_regime_name}'] must be a "
                    f"`StochasticTransition`-wrapped probability function — "
                    f"deterministic per-target regime transitions are not yet "
                    f"supported (use the coarse form, or `StochasticTransition` "
                    f"with indicator probabilities).",
                )
    return tuple(error_messages)


def _validate_completeness(
    *,
    regime: Regime,
    law: RegimeLaw,
    reserved_value_columns: frozenset[str],
) -> tuple[str, ...]:
    """Collect completeness errors for a finalized (post-merge) regime.

    Args:
        regime: The finalized regime to check.
        law: The regime's law, which says whether it is terminal.
        reserved_value_columns: The `value_<member>` names published by
            the model's collective regimes — the whole model's, not this
            regime's, because the published frame is one table over all of
            them.

    These properties hold only for the regime the model actually runs —
    a bare user `Regime` may legitimately lack them until model-level slots
    are merged in:

    - a `utility` entry in `functions` — a per-member `utility_<s>` for
      each member of a collective regime
    - no declaration under a name any of the model's collective regimes
      publishes as a `value_<member>` column
    - state-transition coverage (every non-process state has a law; terminal
      regimes have none; identity laws refer to existing states)
    - no state/action name overlap
    - distributed-grid rules
    - no function-output / discrete-grid-indexed-input name clash
    """
    error_messages: list[str] = []

    # Every member supplies its own `utility_<s>`; the member `"self"` supplies
    # `utility`.
    missing = [
        name
        for name in (
            member_name(base="utility", member=member) for member in regime.members
        )
        if name not in regime.decomposed_functions
    ]
    if missing == ["utility"]:
        error_messages.append(
            "A 'utility' function must be provided in the functions dictionary.",
        )
    elif missing:
        error_messages.append(
            "A collective regime must provide a per-member utility "
            f"function for each member. Missing: {missing}.",
        )

    error_messages.extend(
        _published_value_column_errors(
            regime=regime, reserved_value_columns=reserved_value_columns
        )
    )
    error_messages.extend(_state_transition_coverage_errors(regime, law=law))
    error_messages.extend(_validate_function_output_grid_indexing(regime, law=law))
    error_messages.extend(_koopmans_aggregator_errors(regime, law=law))
    error_messages.extend(_certainty_equivalent_errors(regime, law=law))
    error_messages.extend(_expected_utility_aggregator_errors(regime))

    states_and_actions_overlap = set(regime.states) & set(regime.actions)
    if states_and_actions_overlap:
        error_messages.append(
            "States and actions cannot have overlapping names. The following names "
            f"are used in both states and actions: {states_and_actions_overlap}.",
        )

    return tuple(error_messages)


def _published_value_column_errors(
    *, regime: Regime, reserved_value_columns: frozenset[str]
) -> tuple[str, ...]:
    """Collect errors for declarations that shadow a published value column.

    A collective regime publishes one `value_<member>` column per
    member, unconditionally, and the published frame is a single table
    over every regime — so a singleton regime standing beside a collective one
    claims the same column with a state of that name. A declaration that
    collides neither wins the column nor errors at publication: the frame
    carries the member's value under that name twice and the declared
    variable is absent. The collision is refused here, where the name is still
    the author's, and against the model's members rather than the
    declaring regime's.
    """
    declared = (
        set(regime.states)
        | set(regime.actions)
        | set(regime.functions)
        | set(regime.constraints)
        | set(regime.derived_categoricals)
    )
    colliding = sorted(reserved_value_columns & declared)
    if not colliding:
        return ()
    return (
        (
            "A collective regime publishes a `value_<member>` column for "
            "each of its members, so those names are reserved. Rename the "
            f"following declarations: {colliding}."
        ),
    )


# keyword-only-exempt: primary-argument=regime
def _koopmans_aggregator_errors(regime: Regime, *, law: RegimeLaw) -> tuple[str, ...]:
    """Collect errors for a regime's Koopmans aggregator declaration.

    - `CE` is the reserved name the aggregator receives the continuation under,
      so a variable of that name would be shadowed
    - terminal regimes have no continuation to aggregate
    """
    error_messages: list[str] = []
    # The aggregator is called as `koopmans_aggregator(utility=..., CE=...)`, so
    # a regime variable named `CE` never reaches it: the aggregator would read
    # the continuation while every other function in the regime read the
    # variable, and nothing would say so.
    colliding = sorted(
        {"CE"}
        & (
            set(regime.functions)
            | set(regime.states)
            | set(regime.actions)
            | set(regime.derived_categoricals)
        )
    )
    if colliding:
        error_messages.append(
            "'CE' is reserved: it is the name the Koopmans aggregator receives "
            "the certainty-equivalent continuation under, so a regime variable "
            "of that name would be shadowed inside the aggregator only. Rename "
            "it (`certainty_equivalent_value`, say)."
        )
    if law.terminal and regime.koopmans_aggregator is not None:
        error_messages.append(
            "A terminal regime cannot declare `koopmans_aggregator`: there is "
            "no continuation value to aggregate."
        )
    if isinstance(regime.koopmans_aggregator, Decoupled):
        error_messages.extend(_decoupled_errors(regime))
    return tuple(error_messages)


def _decoupled_errors(regime: Regime) -> tuple[str, ...]:
    """Collect errors for a `Decoupled` Koopmans aggregator.

    Only `GridSearch` reduces the action product in concrete values, so only it
    can choose with one aggregator and store another.
    """
    if isinstance(regime.solver, GridSearch):
        return ()
    return (
        (
            f"The {type(regime.solver).__name__} solver does not support a "
            "`Decoupled` Koopmans aggregator: choosing with one aggregator "
            "and storing another needs the full action product. Use "
            "`solver=GridSearch()` for this regime."
        ),
    )


def _expected_utility_aggregator_errors(regime: Regime) -> tuple[str, ...]:
    """Refuse a non-additive aggregator on the endogenous-grid EU route.

    Without a nonlinear certainty equivalent, NBEGM and NNBEGM run the
    expected-utility kernels, which invert the Euler equation of
    `W = utility + discount_factor * CE`. Any other solve-phase aggregator would
    be ignored rather than solved.
    """
    if not isinstance(regime.solver, (NBEGM, NNBEGM)) or aggregates_nonlinearly(
        regime.certainty_equivalent
    ):
        return ()
    solve_W = regime.get_koopmans_aggregator(phase="solve")
    if solve_W is None or isinstance(solve_W, LinearAggregator):
        return ()
    msg = (
        f"{type(regime.solver).__name__} without a nonlinear "
        "`certainty_equivalent` solves the additive aggregator "
        "`W = utility + discount_factor * CE` only, got "
        f"{type(solve_W).__name__}. Use `koopmans_aggregator=LinearAggregator()` "
        "(or leave it unset), pair a `CESAggregator` with `PowerMean()`, or "
        "solve the regime with GridSearch()."
    )
    return (msg,)


# keyword-only-exempt: primary-argument=regime
def _certainty_equivalent_errors(regime: Regime, *, law: RegimeLaw) -> tuple[str, ...]:
    """Collect errors for a regime's `certainty_equivalent` declaration.

    - terminal regimes have no continuation value to aggregate
    - a solver that reads a continuation payload inverts the Euler equation
      against it. That inversion assumes expected utility unless the solver
      declares `supports_nonlinear_certainty_equivalent`, so a nonlinear
      certainty equivalent is otherwise rejected rather than silently ignored;
      a solver that reads only the value array (grid search) aggregates any
      certainty equivalent in concrete values and is never subject to the rule
    - Epstein-Zin and extreme-value taste shocks do not compose: the taste-shock
      logsum is not invariant under the certainty-equivalent transform, so the
      combination is rejected

    `LinearExpectation` is the expected-utility default every solver and every
    taste-shock regime implements, so only a nonlinear certainty equivalent is
    subject to the composition rules.
    """
    declared = regime.certainty_equivalent
    if declared is None:
        return ()
    each_member = (
        tuple(declared.values()) if isinstance(declared, Mapping) else (declared,)
    )
    # Which class owns which reduction is a separate question from linearity, so
    # it is asked before the linear default is waved through: a subclass of
    # `LinearExpectation` that restates `aggregate` is an instance of it, and
    # would otherwise skip the very gate written to catch it.
    error_messages: list[str] = [
        message for each in each_member for message in _scaled_capability_errors(each)
    ]
    if law.terminal:
        error_messages.append(
            "A terminal regime cannot declare `certainty_equivalent`: there "
            "is no continuation value to aggregate."
        )
    if all(map(_is_plain_linear_expectation, each_member)):
        return tuple(error_messages)
    if (
        regime.solver.required_continuation_keys
        and not regime.solver.supports_nonlinear_certainty_equivalent
    ):
        error_messages.append(
            f"The {type(regime.solver).__name__} solver does not support a "
            "nonlinear `certainty_equivalent`: its Euler inversion assumes "
            "expected utility. Use GridSearch(), NBEGM(), or NNBEGM() for "
            "this regime."
        )
    if isinstance(regime.solver, (NBEGM, NNBEGM)):
        # The endogenous-grid kernels implement the Epstein-Zin recursion for
        # exactly one pairing: they read the power mean's `risk_aversion`
        # parameter for the transform partials and the aggregator's
        # intertemporal elasticity for the Euler inversion and period value.
        # NNBEGM's inner solve runs the same NBEGM kernels, so the contract
        # binds it identically. GridSearch aggregates any certainty
        # equivalent in concrete values, so only the endogenous-grid routes
        # are narrowed.
        solver_name = type(regime.solver).__name__
        if not isinstance(regime.certainty_equivalent, PowerMean):
            error_messages.append(
                f"{solver_name} implements the recursive certainty "
                f"equivalent for `PowerMean` only, got "
                f"{type(regime.certainty_equivalent).__name__}. Use "
                f"`certainty_equivalent=PowerMean()` or solve the regime with "
                f"GridSearch()."
            )
        if not isinstance(regime.koopmans_aggregator, CESAggregator):
            error_messages.append(
                f"{solver_name} with a `certainty_equivalent` requires the "
                "regime's aggregator to be a `CESAggregator` "
                "(`koopmans_aggregator=lcm.CESAggregator()`): the Euler "
                "inversion and period value read its intertemporal "
                "elasticity. With a different aggregator the kernels would "
                "solve a recursion the regime does not declare."
            )
    if regime.taste_shocks is not None:
        error_messages.append(
            "A regime cannot combine `certainty_equivalent` with "
            "`taste_shocks`: the extreme-value logsum is not invariant under "
            "the certainty-equivalent transform, so the Epstein-Zin recursion "
            "and taste shocks do not compose."
        )
    return tuple(error_messages)


# keyword-only-exempt: primary-argument=regime
def _validate_function_output_grid_indexing(
    regime: Regime, *, law: RegimeLaw
) -> tuple[str, ...]:
    """Detect the regime-function-output / discrete-grid-indexed-input name clash.

    The unsafe pattern is: a regime function `f` takes a discrete grid `g`
    (state, action, or derived categorical) as an input — so `f`'s output
    is a per-cell scalar — and a consumer then indexes `f[g]`. The
    consumer is indexing a 0-d array by a scalar integer, which raises
    `IndexError` at trace time. The fix is to drop the redundant `[g]`
    in the consumer (or refactor `f` not to take `g`).

    This check is deliberately best-effort: it catches the common
    `func_output[discrete_grid]` subscript form and nothing else. It is not
    meant to grow into a general correctness checker for user functions —
    if it ever produces false positives, prefer deleting it over hardening
    it to chase every way the pattern can hide.
    """
    function_output_names = set(regime.decomposed_functions)
    discrete_grid_names = (
        {name for name, grid in regime.states.items() if isinstance(grid, DiscreteGrid)}
        | {
            name
            for name, grid in regime.actions.items()
            if isinstance(grid, DiscreteGrid)
        }
        | set(regime.derived_categoricals)
    )
    if not function_output_names or not discrete_grid_names:
        return ()

    # Only treat `func_output[grid]` as unsafe when the producing function
    # *also* takes `grid` as an input — that is the case where the output
    # is per-cell scalar and the consumer's indexing is wrong. If the
    # producing function does not take `grid`, its output shape is
    # whatever it computed (typically an array indexable by `grid`) and
    # the consumer pattern is correct.
    function_inputs = _function_input_names(
        {
            name: func
            for name, func in regime.decomposed_functions.items()
            if func is not None
        }
    )
    consumers = _collect_indexing_consumers(regime, law=law)

    errors: list[str] = []
    for consumer_name, func in consumers:
        clashes = _find_function_output_grid_indexing(
            func=func,
            function_output_names=function_output_names,
            discrete_grid_names=discrete_grid_names,
        )
        for func_output_name, grid_name in clashes:
            if grid_name not in function_inputs.get(func_output_name, set()):
                continue
            errors.append(
                f"Consumer '{consumer_name}' indexes regime function output "
                f"'{func_output_name}' by discrete grid '{grid_name}' "
                f"(`{func_output_name}[{grid_name}]`), but '{func_output_name}' "
                f"already takes '{grid_name}' as input — its output is a "
                f"per-cell scalar, so the indexing raises IndexError at trace "
                f"time. Drop the redundant `[{grid_name}]` in '{consumer_name}', "
                f"or refactor '{func_output_name}' not to take '{grid_name}' "
                f"as input."
            )
    return tuple(errors)


def _function_input_names(
    functions: Mapping[FunctionName, UserFunction | Phased],
) -> MappingProxyType[FunctionName, frozenset[str]]:
    """Return each regime function's input-parameter names.

    A `Phased` contributes the union of both variants' parameters;
    unintrospectable callables contribute the empty set.
    """
    result: dict[FunctionName, frozenset[str]] = {}
    for name, func in functions.items():
        params: set[str] = set()
        for variant in _function_variants(func):
            try:
                params |= set(inspect.signature(variant).parameters)
            except ValueError, TypeError:
                continue
        result[name] = frozenset(params)
    return MappingProxyType(result)


# keyword-only-exempt: primary-argument=regime
def _collect_indexing_consumers(
    regime: Regime, *, law: RegimeLaw
) -> tuple[tuple[FunctionName, ConstraintLike], ...]:
    """Return `(name, callable)` pairs whose bodies are scanned for the clash.

    Functions and constraints contribute every variant of a
    `Phased`; the regime transition contributes itself.
    """
    consumers: list[tuple[FunctionName, ConstraintLike]] = []
    for name, func in regime.decomposed_functions.items():
        if func is None:
            continue
        consumers.extend((name, variant) for variant in _function_variants(func))
    for name, constraint in regime.decomposed_constraints.items():
        if constraint is None:
            continue
        consumers.extend((name, variant) for variant in _function_variants(constraint))
    if is_user_function(law.transition):
        consumers.append(("regime_transition", law.transition))
    return tuple(consumers)


def _function_variants(
    func: ConstraintLike | Phased,
) -> tuple[ConstraintLike, ...]:
    """Return the callable variants of a regime-function entry.

    A plain function is itself; a `Phased` yields its `solve` and `simulate`
    callables so both phases are scanned.
    """
    if isinstance(func, Phased):
        return (
            cast("ConstraintLike", func.solve),
            cast("ConstraintLike", func.simulate),
        )
    return (func,)


def _find_function_output_grid_indexing(
    *,
    func: ConstraintLike,
    function_output_names: set[str],
    discrete_grid_names: set[str],
) -> tuple[tuple[str, str], ...]:
    """Return `(function_output_name, grid_name)` clashes inside `func`'s body."""
    try:
        source = textwrap.dedent(inspect.getsource(func))
    except OSError, TypeError:
        return ()
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return ()

    clashes: list[tuple[str, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Subscript):
            continue
        if not isinstance(node.value, ast.Name):
            continue
        if node.value.id not in function_output_names:
            continue
        if not isinstance(node.slice, ast.Name):
            continue
        if node.slice.id not in discrete_grid_names:
            continue
        clashes.append((node.value.id, node.slice.id))
    return tuple(clashes)


def _state_transition_grammar_errors(regime: Regime) -> tuple[str, ...]:
    """Validate each `state_transitions` entry against the value vocabulary."""
    error_messages: list[str] = []
    for name, value in regime.state_transitions.items():
        error_messages.extend(
            _state_transition_value_errors(name=name, value=value, regime=regime)
        )
    return tuple(error_messages)


def _joint_transition_grammar_errors(regime: Regime) -> tuple[str, ...]:  # noqa: C901
    """Validate the locally knowable joint-kernel grammar.

    Whether a target is reachable is a question about the model's edges
    (`fail_if_a_joint_target_is_unreachable`); whether the regime may own joint
    kernels at all depends on its law (`_validate_law_consistency`).
    """
    if not regime.joint_transitions:
        return ()

    error_messages: list[str] = []
    # A joint node's name may not clash with anything already spoken for,
    # and that is both what the author wrote and what the engine binds: each
    # member's `utility_<s>` is a live name the author never wrote.
    declared_names = (
        set(regime.states)
        | set(regime.actions)
        | set(regime.functions)
        | set(regime.decomposed_functions)
        | set(regime.constraints)
        | set(regime.decomposed_constraints)
        | set(regime.derived_categoricals)
    )
    reserved_prefixes = ("next_", "support_", "weight_", "key_")
    for target_name, kernels in regime.joint_transitions.items():
        if not isinstance(target_name, str):
            error_messages.append(
                f"joint_transitions target key {target_name!r} must be a string."
            )
            continue
        if not isinstance(kernels, Mapping) or not kernels:
            error_messages.append(
                f"joint_transitions['{target_name}'] must be a nonempty mapping "
                "from node names to `JointTransition` declarations."
            )
            continue
        for kernel_name, raw in kernels.items():
            prefix = f"joint_transitions['{target_name}'][{kernel_name!r}]"
            if not isinstance(kernel_name, str):
                error_messages.append(f"{prefix}: the node name must be a string.")
            elif node_name_errors := path_segment_name_errors(
                kind=f"{prefix}: node", names=[kernel_name]
            ):
                error_messages.extend(node_name_errors)
            elif (
                reserved_prefix := next(
                    (
                        candidate
                        for candidate in reserved_prefixes
                        if kernel_name.startswith(candidate)
                    ),
                    None,
                )
            ) is not None:
                error_messages.append(
                    f"{prefix}: the joint node name {kernel_name!r} uses reserved "
                    f"transition prefix {reserved_prefix!r}."
                )
            else:
                colliding = sorted(
                    {kernel_name, f"support_{kernel_name}"} & declared_names
                )
                if colliding:
                    error_messages.append(
                        f"{prefix}: the joint node name {kernel_name!r} collides "
                        "with source state, action, function, constraint, derived "
                        f"categorical, or synthesized support name(s): {colliding}."
                    )
            variants = (raw.solve, raw.simulate) if isinstance(raw, Phased) else (raw,)
            for variant in variants:
                if not isinstance(variant, JointTransition):
                    error_messages.append(
                        f"{prefix}: each value must be a `JointTransition`, or "
                        "`Phased` wrapping one for each phase."
                    )
                    break
    return tuple(error_messages)


# keyword-only-exempt: primary-argument=regime
def _state_transition_coverage_errors(
    regime: Regime, *, law: RegimeLaw
) -> tuple[str, ...]:
    """Validate that ordinary or joint laws cover the regime's states."""
    error_messages: list[str] = []

    process_names: set[ProcessName] = {
        name
        for name, grid in regime.states.items()
        if isinstance(grid, _ContinuousStochasticProcess)
        or (
            isinstance(grid, Phased)
            and isinstance(grid.solve, _ContinuousStochasticProcess)
        )
    }
    non_process_names: set[StateName] = set(regime.states) - process_names

    # Keys not in states are allowed only with actual transitions. A
    # `fixed_transition` entry is an identity law, which requires the state
    # to exist in this regime.
    extra_keys = set(regime.state_transitions) - set(regime.states)
    for key in extra_keys:
        value = regime.state_transitions[key]
        if isinstance(value, _IdentityTransition):
            error_messages.append(
                f"state_transitions['{key}'] is `fixed_transition` but '{key}' "
                f"is not in states. Identity laws require the state to exist "
                f"in this regime.",
            )

    process_in_transitions = process_names & set(regime.state_transitions)
    if process_in_transitions:
        error_messages.append(
            f"Stochastic process states have intrinsic transitions and must not "
            f"appear in state_transitions: {process_in_transitions}.",
        )

    if law.terminal:
        if regime.state_transitions:
            error_messages.append(
                _terminal_regime_message(
                    slot="state_transitions", names=sorted(regime.state_transitions)
                )
            )
        return tuple(error_messages)

    joint_output_names = {
        output_name
        for kernels in regime.joint_transitions.values()
        for raw in kernels.values()
        for joint in ((raw.solve, raw.simulate) if isinstance(raw, Phased) else (raw,))
        if isinstance(joint, JointTransition)
        for output_name in joint.outputs
    }
    missing = non_process_names - set(regime.state_transitions) - joint_output_names
    if missing:
        error_messages.append(
            f"Every non-process state must have an entry in state_transitions. "
            f"Missing: {missing}. Use `fixed_transition(state_name)` for "
            f"fixed states.",
        )

    return tuple(error_messages)


def _phased_per_target_shape_mismatch(
    *, name: StateName, value: Phased, regime: Regime
) -> tuple[str, ...]:
    """Constrain how the two variants of a `Phased` state law may be shaped.

    A `Phased` whose variants are per-target dicts is normalized into one entry per
    target, each holding that target's two laws. `Phased` is outermost-only, so this
    is the only spelling for a per-target law that varies by phase.

    Accepted:

    - **Both bare** — one coarse law per phase, sharing a single parameter leaf.
    - **Both per-target** — each phase declares its own target laws.
    - **Per-target on one side, a parameter-free bare law on the other** — the bare
      law broadcasts over the other side's targets, exactly as a bare state law does
      outside `Phased`. Carrying no parameter, it contributes no template leaf, so
      nothing is replicated.

    Rejected:

    - **A parameterized bare law opposite a per-target dict** — the params template
      would show one leaf per target while the law binds a single node, so all but
      one leaf is dead and setting them differently is silently ignored. Spell it
      either as bare in both phases (one shared parameter) or as an explicit
      per-target dict on that side (one parameter per target).
    """
    solve_per_target = isinstance(value.solve, Mapping)
    simulate_per_target = isinstance(value.simulate, Mapping)
    if solve_per_target and simulate_per_target:
        return ()
    if solve_per_target == simulate_per_target:
        # Both bare: one coarse node (parameterized or not) — nothing to reject.
        return ()
    # Map-vs-bare: one side per-target, the other bare. The bare side broadcasts; a
    # free parameter on it would be replicated per target with only the first leaf live.
    bare_side, phase_label = (
        (value.simulate, "simulate") if solve_per_target else (value.solve, "solve")
    )
    if _law_has_free_parameter(law=bare_side, regime=regime):
        return (
            (
                f"state_transitions['{name}']: the {phase_label} variant is a bare "
                f"(coarse) law with a free parameter, opposite a per-target dict "
                f"(map-vs-bare). A parameterized coarse law broadcast over targets "
                f"would have its parameter replicated per target with only one "
                f"binding live, so this shape is not supported. Spell it as a "
                f"both-bare `Phased` (coarse in both phases, one shared parameter) "
                f"or as an explicit per-target dict on the {phase_label} side (one "
                f"parameter per target)."
            ),
        )
    return ()


def _law_has_free_parameter(*, law: StateTransitionEntry, regime: Regime) -> bool:
    """Whether a bare state-transition law reads a free parameter (a template leaf).

    A free parameter is any argument that is not a state, action, `next_<state>` node,
    or reserved name (`period`, `age`, `E_next_V`) — i.e., an argument that would appear
    in the params template. `fixed_transition` identities and any callable whose
    signature cannot be read are treated as parameter-free (nothing to replicate).
    """
    if isinstance(law, _IdentityTransition) or not callable(law):
        return False
    try:
        arg_names = set(inspect.signature(law).parameters)
    except TypeError, ValueError:
        return False
    variables = (
        set(regime.states)
        | set(regime.actions)
        | set(regime.decomposed_functions)
        | {f"next_{state_name}" for state_name in regime.states}
        | {f"next_{state_name}" for state_name in regime.state_transitions}
        | {"period", "age", "E_next_V"}
    )
    return bool(arg_names - variables)


def _state_transition_value_errors(
    *, name: StateName, value: StateTransitionEntry, regime: Regime
) -> tuple[str, ...]:
    """Validate one `state_transitions` entry against the value vocabulary.

    Each variant of a `Phased` entry is held to the vocabulary of a bare value —
    callable, `StochasticTransition`, or a per-target Mapping. A stochastic variant is
    allowed: the solve variant is the perceived law that prices the continuation,
    the simulate variant the true law the next state is drawn from.

    The two variants need not agree on whether the law is stochastic. A deterministic
    law is a degenerate kernel, so the state has the same domain either way, and an
    agent may perceive risk where there is none or believe a transition certain that
    is not.

    Which combinations of bare and per-target shapes are allowed is decided by
    `_phased_per_target_shape_mismatch`.

    `None` is not a law of motion; the error points to `fixed_transition`.
    """
    error_messages: list[str] = []
    phase_variant = isinstance(value, Phased)
    if phase_variant:
        error_messages.extend(
            _phased_per_target_shape_mismatch(name=name, value=value, regime=regime)
        )
    for variant, label in _state_transition_variants(value):
        if variant is None:
            if phase_variant:
                error_messages.append(
                    f"state_transitions['{name}']{label}: a mask is "
                    f"whole-entry only — `None` cannot appear inside "
                    f"`Phased`.",
                )
            # Bare `None` masks a model-level law; bound at model build.
            continue
        error_messages.extend(
            _fixed_transition_name_mismatch(state_name=name, value=variant, label=label)
        )
        if callable(variant):
            continue
        if isinstance(variant, Mapping):
            error_messages.extend(
                _validate_per_target_dict(state_name=name, targets=variant)
            )
        else:
            error_messages.append(
                f"state_transitions['{name}']{label} must be callable, "
                f"StochasticTransition, `fixed_transition(...)`, or a per-target "
                f"Mapping, got {type(variant).__name__}.",
            )
    return tuple(error_messages)


def _fixed_transition_name_mismatch(
    *, state_name: StateName, value: StateTransitionEntry, label: str
) -> tuple[str, ...]:
    """Reject a `fixed_transition` whose argument differs from its dict key."""
    if (
        isinstance(value, _IdentityTransition) and value._state_name != state_name  # noqa: SLF001
    ):
        return (
            (
                f"state_transitions['{state_name}']{label}: "
                f"`fixed_transition('{value._state_name}')` is assigned to state "  # noqa: SLF001
                f"'{state_name}' — the names must match."
            ),
        )
    return ()


def _state_transition_variants(
    value: StateTransitionEntry,
) -> tuple[tuple[StateTransitionEntry, str], ...]:
    """Return a state-transition entry's per-phase variants with display labels.

    A bare value is its own single variant; a `Phased` yields its solve and
    simulate variants, each validated against the same vocabulary.
    """
    if isinstance(value, Phased):
        return ((value.solve, " solve variant"), (value.simulate, " simulate variant"))
    return ((value, ""),)


@beartype(conf=REGIME_CONF)
def _validate_per_target_dict(
    *, state_name: StateName, targets: Mapping[RegimeName, StateTransitionEntry]
) -> tuple[str, ...]:
    """Validate a per-target transition dict for stochastic consistency and types."""
    error_messages: list[str] = []
    markov_count = 0
    for target_regime_name, law in targets.items():
        if not isinstance(target_regime_name, str):
            error_messages.append(
                f"state_transitions['{state_name}'] per-target dict key "
                f"{target_regime_name!r} must be a string.",
            )
        if isinstance(law, Phased):
            error_messages.append(
                f"state_transitions['{state_name}']['{target_regime_name}'] cannot "
                f"be `Phased` — `Phased` is outermost-only: wrap the whole "
                f"entry, e.g. `Phased(solve={{...}}, simulate={{...}})`.",
            )
        elif isinstance(law, StochasticTransition):
            markov_count += 1
        elif isinstance(law, _IdentityTransition):
            error_messages.extend(
                _fixed_transition_name_mismatch(
                    state_name=state_name,
                    value=law,
                    label=f"['{target_regime_name}']",
                )
            )
        elif not callable(law):
            error_messages.append(
                f"state_transitions['{state_name}']['{target_regime_name}'] must be "
                f"callable or StochasticTransition, got "
                f"{type(law).__name__}.",
            )
    if 0 < markov_count < len(targets):
        error_messages.append(
            f"state_transitions['{state_name}'] per-target dict must be "
            f"consistently stochastic: either all values are "
            f"StochasticTransition or none are.",
        )
    return tuple(error_messages)


def _is_plain_linear_expectation(
    certainty_equivalent: CertaintyEquivalent,
) -> bool:
    """Whether continuation solvers may use their plain expected-value route.

    A subclass is plain only when it inherits both ordinary and scaled
    semantics unchanged from `LinearExpectation`. A subclass that overrides
    both methods is internally consistent but states a different mean, which a
    continuation solver would otherwise accept and silently replace by expected
    utility.
    """
    mean_type = type(certainty_equivalent)
    return (
        isinstance(certainty_equivalent, LinearExpectation)
        and _method_owner(mean_type=mean_type, name="aggregate") is LinearExpectation
        and _method_owner(mean_type=mean_type, name="aggregate_scaled")
        is LinearExpectation
    )


def _scaled_capability_errors(
    certainty_equivalent: CertaintyEquivalent,
) -> tuple[str, ...]:
    """Collect errors for a certainty equivalent that cannot take scaled weights.

    A continuation lottery's weights reach the certainty equivalent as
    `(coefficient, shift)` pairs, because a joint probability built from several
    rare factors sits further below the likeliest node than the exponent field
    spans and no ordinary weight vector states it. `aggregate_scaled` is the
    method that reads the pair.

    Which implementation runs is decided by ownership rather than presence, and
    two cases are wrong:

    - the most-derived owner of `aggregate_scaled` is the base class, which
      cannot reduce a scaled lottery for an algebra it has never seen and says
      so rather than approximating one;
    - `aggregate` is owned further down the class hierarchy than
      `aggregate_scaled`, so a subclass has changed what the mean *means* while
      inheriting a scaled reduction written for its parent's meaning. The two
      would then disagree, and which one ran would depend on whether a lottery
      happened to carry scales.

    Inherited semantics with an inherited scaled method agree by construction,
    and an explicit override of both is the author's own business.
    """
    mean_type = type(certainty_equivalent)
    ordinary_owner = _method_owner(mean_type=mean_type, name="aggregate")
    scaled_owner = _method_owner(mean_type=mean_type, name="aggregate_scaled")
    remedy = (
        f"Implement `aggregate_scaled` on `{mean_type.__name__}` so it reduces "
        "the lottery with its scales still present, or use one of the shipped "
        "certainty equivalents (`LinearExpectation`, `QuasiArithmeticMean`, "
        "`PowerMean`), which do."
    )
    if scaled_owner is CertaintyEquivalent:
        boundary = (
            f"`{mean_type.__name__}` defines `aggregate` but inherits "
            "`aggregate_scaled` from `CertaintyEquivalent`, which is a "
            "capability boundary rather than an implementation: a weight below "
            "the smallest positive float cannot be handed to an arbitrary "
            f"`aggregate` as an ordinary number. {remedy}"
        )
        return (boundary,)
    order = mean_type.__mro__
    if order.index(ordinary_owner) < order.index(scaled_owner):
        mismatch = (
            f"`{mean_type.__name__}` overrides `aggregate` but inherits "
            f"`aggregate_scaled` from `{scaled_owner.__name__}`, whose scaled "
            "reduction states the parent's mean rather than this one. The two "
            "would disagree, and which of them ran would depend on whether the "
            f"lottery carried scales. {remedy}"
        )
        return (mismatch,)
    return ()


def _method_owner(*, mean_type: type[CertaintyEquivalent], name: str) -> type:  # noqa: PAN006 - an MRO entry may be any class, a mixin included
    """Return the most-derived class in `mean_type`'s MRO that defines `name`."""
    return next(klass for klass in mean_type.__mro__ if name in vars(klass))
