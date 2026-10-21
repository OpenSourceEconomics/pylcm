"""Hand user functions the numbers of value grids instead of their codes.

A `DiscreteGrid(values=...)` variable is a code `0..n-1` everywhere inside the
engine. `adapt_user_function` wraps a user function so that it receives the
values of the value-grid variables it reads, and, for a deterministic law or a
derived categorical of such a grid, so that the value it returns becomes the
code of the node it names. A value on no node becomes `MISSING_CAT_CODE`.

An argument `<name>_code` of any discrete variable `<name>`, declared by values
or by a category class, receives the code itself. For a category class the code
is what `<name>` holds as well. A stochastic process state `<name>` holds the
value of its node, and `<name>_code` receives that node's int32 position.

The wrapper publishes `DiscreteState` for every decoded argument and encoded
return, the annotation the engine's own nodes carry for codes, so the DAG sees
one annotation per node whichever function names it.
"""

import copy
import dataclasses
import inspect
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import cast, no_type_check

import jax.numpy as jnp
from dags import get_annotations, get_free_arguments

from _lcm.constraints.ir import Condition
from _lcm.gated_edge import GatedEdge
from _lcm.grids import DiscreteGrid, Grid
from _lcm.identity_transition import _IdentityTransition
from _lcm.processes import _ContinuousStochasticProcess
from _lcm.processes.joint_normal import JointNormalIIDProcess
from _lcm.regime_building.schedules import RegimeTransitionLaw
from _lcm.regime_law import RegimeLaw, RegimeLaws
from _lcm.typing import EconFunctionArg, FunctionName, RegimeName
from _lcm.utils.ast_inspection import value_indexed_arrays
from lcm.collective import (
    CollectiveUtility,
    ProjectedRegimeValue,
    ValueDependentConstraint,
)
from lcm.decoupled import Decoupled
from lcm.exceptions import RegimeInitializationError
from lcm.phased import Phased
from lcm.regime import (
    ConstraintEntry,
    FunctionEntry,
    StateEntry,
    StateTransitionEntry,
)
from lcm.regime import Regime as UserRegime
from lcm.transition import (
    AgeSpecializedFunction,
    DeterministicTransition,
    JointTransition,
    StochasticTransition,
    _AgeSpecialized,
)
from lcm.typing import StateOrActionName, UserFunction, ValueND

_CODE_ANNOTATION = "DiscreteState"


@dataclass(frozen=True)
class UnreadableProcessCode:
    """A process state whose node position a reading regime cannot supply."""

    reason: str
    """Why the reading regime has no node position to give."""


# The grid a user function's argument is read through: a discrete grid, whose
# code the engine holds, or the reading regime's process, whose node value the
# engine holds.
type AdapterGrid = DiscreteGrid | _ContinuousStochasticProcess | UnreadableProcessCode

# The suffix of an argument that reads the code of a discrete variable.
CODE_SUFFIX = "_code"

# One entry of a regime's `functions`, `states`, `state_transitions`,
# `constraints` or `joint_transitions`.
type AdaptableEntry = (
    FunctionEntry
    | StateEntry
    | StateTransitionEntry
    | ConstraintEntry
    | Decoupled
    | JointTransition
    | Mapping[str, JointTransition | Phased]
)

# The adapters built for one model: each authored function and encoding grid
# (by identity) to the function and its adapter.
type AdapterMemo = dict[
    tuple[int, int, tuple[tuple[str, int], ...]], tuple[UserFunction, ValueAdapter]
]

# The attributes an adapter publishes itself and never takes from its function.
_ADAPTER_OWN_ATTRIBUTES: frozenset[str] = frozenset(
    {
        "func",
        "decoded",
        "encode",
        "__signature__",
        "__wrapped__",
        "__annotations__",
        "__annotate__",
        "__dict__",
        "__lcm_source__",
        "_reads",
    }
)


def collect_value_grids(
    *,
    regimes: Mapping[RegimeName, UserRegime],
    derived_categoricals: Mapping[FunctionName, DiscreteGrid],
) -> MappingProxyType[StateOrActionName, DiscreteGrid]:
    """Return the value grid of every variable a model declares by its values.

    A name declared with values in one place must be declared with the same
    values, numeric kind and tolerance wherever it appears, so one grid encodes
    a law's result whichever regime reads it next.

    Raises:
        RegimeInitializationError: If a name is a value grid in one declaration
            and differs in another, or a value grid meets a declaration that
            would read its codes.
        GridInitializationError: If a grid's values do not survive the
            precision in force, for instance collide once rounded to float32.

    """
    declarations: dict[StateOrActionName, list[tuple[RegimeName, DiscreteGrid]]] = {}
    for regime_name, regime in regimes.items():
        for name, entry in {
            **dict(regime.states),
            **dict(regime.actions),
            **dict(regime.derived_categoricals),
        }.items():
            # A carried state declares its grid on one side of `Phased`.
            sides = (
                (entry.solve, entry.simulate) if isinstance(entry, Phased) else (entry,)
            )
            for side in sides:
                if isinstance(side, DiscreteGrid):
                    declarations.setdefault(name, []).append((regime_name, side))
    for name, grid in derived_categoricals.items():
        declarations.setdefault(name, []).append(("model", grid))
    value_grids = {
        name: value_grid
        for name, grids in declarations.items()
        for value_grid in [next((g for _, g in grids if g.values is not None), None)]
        if value_grid is not None
    }
    for name, value_grid in value_grids.items():
        expected_key = _value_grid_key(grid=value_grid)
        mismatched = [
            regime_name
            for regime_name, grid in declarations[name]
            if _value_grid_key(grid=grid) != expected_key
        ]
        if mismatched:
            raise RegimeInitializationError(
                f"'{name}' is a `DiscreteGrid(values=...)` with values "
                f"{value_grid.values} and tolerance {value_grid.tolerance}, but "
                f"{mismatched} declare it differently. Declare it with the same "
                "values and tolerance everywhere. The numeric kind (all integers "
                "or all floats) must also match."
            )
        # A grid built at another precision holds its nodes at the model's
        # from here on; values that do not survive that precision raise now.
        value_grid.values_to_jax()
    if value_grids:
        _fail_if_unsupported_reader(
            regimes=regimes,
            value_grids=value_grids,
            model_derived_categoricals=derived_categoricals,
        )
    return MappingProxyType(value_grids)


def collect_adapter_grids(
    *,
    regimes: Mapping[RegimeName, UserRegime],
    derived_categoricals: Mapping[FunctionName, DiscreteGrid],
) -> MappingProxyType[StateOrActionName, AdapterGrid]:
    """Return the grid of every discrete variable a user function can read.

    A variable declared by its values maps to its value grid, which decodes and
    encodes it. Every other discrete variable maps to one of its category grids,
    which only marks that `<name>_code` reads its code. A stochastic process
    state maps to a declaration of its process, on whose nodes `<name>_code`
    reads the position.

    Raises:
        RegimeInitializationError: If a regime declares a variable, function or
            constraint named `<name>_code` for a discrete variable `<name>`, or
            as `collect_value_grids` does.

    """
    value_grids = collect_value_grids(
        regimes=regimes, derived_categoricals=derived_categoricals
    )
    grids: dict[StateOrActionName, AdapterGrid] = {
        **_declared_process_grids(regimes=regimes),
        **dict(
            _declared_discrete_grids(
                regimes=regimes, derived_categoricals=derived_categoricals
            )
        ),
    }
    _fail_if_a_name_spells_a_code(regimes=regimes, discrete_names=frozenset(grids))
    return MappingProxyType({**grids, **value_grids})


def bind_reading_regimes(
    *,
    value_grids: Mapping[StateOrActionName, AdapterGrid],
    regimes: Mapping[RegimeName, UserRegime],
) -> MappingProxyType[RegimeName, MappingProxyType[StateOrActionName, AdapterGrid]]:
    """Return, per reading regime, `value_grids` with each process bound to its own.

    A `<name>_code` argument reads the position of the value the reading regime
    holds, so its nodes are that regime's. Where the regime holds no process of
    that name, the name maps to an `UnreadableProcessCode`, which refuses a code
    read.

    Each regime binds its own copy of a process, so an adapter's process says
    which regime reads it, even where regimes share one declaration.
    """
    process_names = tuple(
        name
        for name, grid in value_grids.items()
        if isinstance(grid, _ContinuousStochasticProcess | UnreadableProcessCode)
    )
    return MappingProxyType(
        {
            regime_name: MappingProxyType(
                {
                    **value_grids,
                    **{
                        name: _local_process(regime=regime, name=name)
                        for name in process_names
                    },
                }
            )
            for regime_name, regime in regimes.items()
        }
    )


def process_code_reads(
    *,
    value_grids_by_regime: Mapping[RegimeName, Mapping[StateOrActionName, AdapterGrid]],
    adapters: AdapterMemo,
) -> MappingProxyType[
    RegimeName, MappingProxyType[StateOrActionName, _ContinuousStochasticProcess]
]:
    """Return, per regime, the process states some adapter reads the position of.

    A regime's code readers are its own functions and the gates, fallbacks and
    references of the gated edges landing in it; each binds that regime's copy.
    """
    read = {
        id(process)
        for _, adapter in adapters.values()
        for process in adapter.positions.values()
    }
    return MappingProxyType(
        {
            regime_name: MappingProxyType(
                {
                    name: grid
                    for name, grid in grids.items()
                    if isinstance(grid, _ContinuousStochasticProcess)
                    and id(grid) in read
                }
            )
            for regime_name, grids in value_grids_by_regime.items()
        }
    )


def _local_process(
    *, regime: UserRegime, name: StateOrActionName
) -> _ContinuousStochasticProcess | UnreadableProcessCode:
    """Return a copy of the process `regime` holds under `name`, or why it has none."""
    entry = regime.states.get(name)
    sides = (entry.solve, entry.simulate) if isinstance(entry, Phased) else (entry,)
    processes = [
        side for side in sides if isinstance(side, _ContinuousStochasticProcess)
    ]
    if not processes or len(processes) != len(sides):
        return UnreadableProcessCode(reason=f"the regime holds no process `{name}`")
    # Validation refuses two node-drawing phases with different laws, so the
    # phases of a process that stands on nodes share its nodes.
    unsafe = [process for process in processes if not _stands_on_nodes(process)]
    return copy.copy(unsafe[0] if unsafe else processes[0])


def declared_discrete_names(
    *,
    regimes: Mapping[RegimeName, UserRegime],
    derived_categoricals: Mapping[FunctionName, DiscreteGrid],
) -> frozenset[StateOrActionName]:
    """Return the name of every variable whose code `<name>_code` is readable.

    Those are the declared discrete variables and the stochastic process states.
    """
    return frozenset(
        {
            *(
                name
                for name, _ in _declared_discrete_grids(
                    regimes=regimes, derived_categoricals=derived_categoricals
                )
            ),
            *_declared_process_grids(regimes=regimes),
        }
    )


def _declared_process_grids(
    *, regimes: Mapping[RegimeName, UserRegime]
) -> dict[StateOrActionName, _ContinuousStochasticProcess]:
    """Return one declaration per process state name, both sides of `Phased`.

    A `<name>_code` read takes the node position from the value the process
    holds, which is exact only where that value is a node, so a declaration
    that simulates off its nodes or lacks fixed nodes is the one returned.
    """
    found: dict[StateOrActionName, _ContinuousStochasticProcess] = {}
    for regime in regimes.values():
        for name, entry in regime.states.items():
            sides = (
                (entry.solve, entry.simulate) if isinstance(entry, Phased) else (entry,)
            )
            for side in sides:
                if not isinstance(side, _ContinuousStochasticProcess):
                    continue
                if name not in found or not _stands_on_nodes(side):
                    found[name] = side
    return found


def _stands_on_nodes(process: _ContinuousStochasticProcess) -> bool:
    """Whether every value the process holds is one of its fixed nodes."""
    return process.is_fully_specified and process.simulate_draws == "nodes"


def _declared_discrete_grids(
    *,
    regimes: Mapping[RegimeName, UserRegime],
    derived_categoricals: Mapping[FunctionName, DiscreteGrid],
) -> list[tuple[StateOrActionName, DiscreteGrid]]:
    """Return every declared discrete grid with its name, both sides of `Phased`."""
    found = list(derived_categoricals.items())
    for regime in regimes.values():
        for name, entry in {
            **dict(regime.states),
            **dict(regime.actions),
            **dict(regime.derived_categoricals),
        }.items():
            sides = (
                (entry.solve, entry.simulate) if isinstance(entry, Phased) else (entry,)
            )
            found.extend(
                (name, side) for side in sides if isinstance(side, DiscreteGrid)
            )
    return found


def _fail_if_a_name_spells_a_code(
    *, regimes: Mapping[RegimeName, UserRegime], discrete_names: frozenset[str]
) -> None:
    """Raise where a declared name is `<name>_code` of a discrete variable."""
    collisions = sorted(
        f"'{name}' in regime '{regime_name}'"
        for regime_name, regime in regimes.items()
        for name in (
            *regime.states,
            *regime.actions,
            *regime.derived_categoricals,
            *regime.decomposed_functions,
            *regime.constraints,
        )
        if name.endswith(CODE_SUFFIX)
        and name.removesuffix(CODE_SUFFIX) in discrete_names
    )
    if collisions:
        raise RegimeInitializationError(
            f"{', '.join(collisions)} collides with the code of a discrete "
            f"variable: an argument `<name>{CODE_SUFFIX}` reads the code of the "
            "discrete variable `<name>`. Rename it."
        )


def _value_grid_key(
    *, grid: DiscreteGrid
) -> tuple[tuple[int | float, ...] | None, float, bool | None]:
    """Return values, tolerance and numeric kind for one declared grid."""
    values = grid.values
    # Validated value grids are nonempty and hold only Python ints or floats.
    is_integer = None if values is None else isinstance(values[0], int)
    return values, grid.tolerance, is_integer


def adapt_user_function(
    *,
    func: UserFunction,
    value_grids: Mapping[StateOrActionName, AdapterGrid],
    encode: DiscreteGrid | None,
    adapters: AdapterMemo | None = None,
) -> UserFunction:
    """Wrap `func` to read values and, given `encode`, to return a code.

    Returns `func` itself when it reads no value-grid variable and `encode` is
    `None`.

    Args:
        func: The user function.
        value_grids: The value grid of every variable declared by its values.
        encode: The grid whose code the function's returned value names, for a
            deterministic law or a derived categorical of a value grid.
        adapters: The adapters already built for this model, so one authored
            function with one encoding gets one adapter wherever it appears.

    Returns:
        The function the engine calls with codes.

    """
    # A finalized regime already holds adapters; re-adapting starts from the
    # user's function, so a regime reused in another model is wrapped once.
    if isinstance(func, ValueAdapter):
        func = func.func
    reads = _read_names(func)
    decoded = {
        name: cast("DiscreteGrid", value_grids[grid_name])
        for name in reads
        if (grid_name := _value_grid_name(name=name, value_grids=value_grids))
        is not None
    }
    codes = {
        name: name.removesuffix(CODE_SUFFIX)
        for name in reads
        if name.endswith(CODE_SUFFIX)
        and name.removesuffix(CODE_SUFFIX) in value_grids
        and name not in value_grids
    }
    positions = {
        name: grid
        for name, variable in codes.items()
        if isinstance(
            grid := value_grids[variable],
            _ContinuousStochasticProcess | UnreadableProcessCode,
        )
    }
    if not decoded and not codes and encode is None:
        return func
    _fail_if_an_array_is_indexed_by_a_value(func=func, decoded=decoded)
    _fail_if_a_position_has_no_node(func=func, positions=positions, codes=codes)
    # One callable read in two regimes binds each regime's own nodes.
    key = (
        id(func),
        id(encode),
        tuple(sorted((name, id(grid)) for name, grid in positions.items())),
    )
    if adapters is not None and key in adapters:
        return adapters[key][1]
    adapter = ValueAdapter(
        func=func,
        decoded=MappingProxyType(decoded),
        codes=MappingProxyType(codes),
        positions=MappingProxyType(
            cast("dict[str, _ContinuousStochasticProcess]", positions)
        ),
        encode=encode,
    )
    if adapters is not None:
        # The function is kept with its adapter, so its id cannot be reused.
        adapters[key] = (func, adapter)
    return adapter


def adapt_regime_laws(
    *,
    laws: RegimeLaws,
    value_grids_by_regime: Mapping[RegimeName, Mapping[StateOrActionName, AdapterGrid]],
    adapters: AdapterMemo | None = None,
) -> MappingProxyType[RegimeName, RegimeLaw]:
    """Adapt every regime transition law to read the values of value grids.

    A law still returns a regime code or probabilities over regimes, so nothing
    it returns is encoded. A regime name and a terminal law read nothing. The
    transition reads the source regime's states; a gated edge's gate, fallbacks
    and references read the states of the target it lands in.
    """
    adapters = {} if adapters is None else adapters
    return MappingProxyType(
        {
            name: law
            if not value_grids_by_regime[name] or law.terminal
            else _adapt_regime_law(
                law=law,
                source=name,
                value_grids_by_regime=value_grids_by_regime,
                adapters=adapters,
            )
            for name, law in laws.items()
        }
    )


def _adapt_regime_law(
    *,
    law: RegimeLaw,
    source: RegimeName,
    value_grids_by_regime: Mapping[RegimeName, Mapping[StateOrActionName, AdapterGrid]],
    adapters: AdapterMemo,
) -> RegimeLaw:
    """Adapt one law's transition and gated edges; keep the law if neither changes."""
    transition = (
        law.transition
        if isinstance(law.transition, str)
        else cast(
            "RegimeTransitionLaw",
            adapt_entry(
                entry=cast("AdaptableEntry", law.transition),
                value_grids=value_grids_by_regime[source],
                encode=None,
                adapters=adapters,
            ),
        )
    )
    gated_edges = {
        target: _adapt_gated_edge(
            edge=edge,
            value_grids=value_grids_by_regime[target],
            adapters=adapters,
        )
        for target, edge in law.gated_edges.items()
    }
    if transition is law.transition and all(
        gated_edges[target] is edge for target, edge in law.gated_edges.items()
    ):
        return law
    return dataclasses.replace(
        law, transition=transition, gated_edges=MappingProxyType(gated_edges)
    )


def adapt_collective_utility(
    *,
    declaration: CollectiveUtility,
    engine_functions: Mapping[FunctionName, UserFunction | Phased | None],
    value_grids: Mapping[StateOrActionName, AdapterGrid],
    adapters: AdapterMemo,
) -> CollectiveUtility:
    """Return the household declaration with the bodies the engine runs.

    Each stakeholder's body is the adapted `utility_<s>` the engine holds, so the
    declaration and the engine name one object, and each Pareto weight reads
    values. A declaration that reads no value grid is returned as declared.
    """
    utilities = {
        stakeholder: None
        if body is None
        else cast(
            "UserFunction | Phased",
            engine_functions.get(f"utility_{stakeholder}", body),
        )
        for stakeholder, body in declaration.utilities.items()
    }
    objective = declaration.objective
    if objective is not None:
        weights = {
            stakeholder: weight
            if isinstance(weight, float | int)
            else adapt_user_function(
                func=weight, value_grids=value_grids, encode=None, adapters=adapters
            )
            for stakeholder, weight in objective.weights.items()
        }
        if any(weights[s] is not w for s, w in objective.weights.items()):
            objective = dataclasses.replace(objective, weights=weights)
    if objective is declaration.objective and all(
        utilities[s] is body for s, body in declaration.utilities.items()
    ):
        return declaration
    return dataclasses.replace(declaration, utilities=utilities, objective=objective)


def _adapt_gated_edge(
    *,
    edge: GatedEdge,
    value_grids: Mapping[StateOrActionName, AdapterGrid],
    adapters: AdapterMemo,
) -> GatedEdge:
    """Adapt a gated edge's gate, its routes' fallbacks and its references.

    The edge is kept as declared if none of them changes.
    """
    gate = adapt_user_function(
        func=edge.gate, value_grids=value_grids, encode=None, adapters=adapters
    )
    fallbacks = {
        name: _adapt_reference(
            reference=route.fallback, value_grids=value_grids, adapters=adapters
        )
        for name, route in edge.legs.items()
    }
    gate_refs = {
        name: cast(
            "ProjectedRegimeValue",
            _adapt_reference(
                reference=reference, value_grids=value_grids, adapters=adapters
            ),
        )
        for name, reference in edge.gate_refs.items()
    }
    if (
        gate is edge.gate
        and all(fallbacks[name] is route.fallback for name, route in edge.legs.items())
        and all(gate_refs[name] is ref for name, ref in edge.gate_refs.items())
    ):
        return edge
    return dataclasses.replace(
        edge,
        gate=gate,
        legs=MappingProxyType(
            {
                name: dataclasses.replace(route, fallback=fallbacks[name])
                for name, route in edge.legs.items()
            }
        ),
        gate_refs=MappingProxyType(gate_refs),
    )


def _adapt_reference(
    *,
    reference: ProjectedRegimeValue | Phased,
    value_grids: Mapping[StateOrActionName, AdapterGrid],
    adapters: AdapterMemo,
) -> ProjectedRegimeValue | Phased:
    """Adapt a projection: it reads values and returns each value-grid state's code.

    A projection lands on the reference regime's state of the same name, whose
    grid is the model's one grid of that name.
    """
    if isinstance(reference, Phased):
        solve, simulate = (
            _adapt_reference(reference=side, value_grids=value_grids, adapters=adapters)
            for side in (reference.solve, reference.simulate)
        )
        if solve is reference.solve and simulate is reference.simulate:
            return reference
        return Phased(solve=solve, simulate=simulate)
    projection = {
        name: adapt_user_function(
            func=func,
            value_grids=value_grids,
            encode=encoding_grid(value_grids=value_grids, name=name),
            adapters=adapters,
        )
        for name, func in reference.projection.items()
    }
    if all(projection[name] is func for name, func in reference.projection.items()):
        return reference
    return dataclasses.replace(reference, projection=projection)


def adapt_entry(  # noqa: PLR0911
    *,
    entry: AdaptableEntry,
    value_grids: Mapping[StateOrActionName, AdapterGrid],
    encode: DiscreteGrid | None,
    adapters: AdapterMemo | None = None,
) -> AdaptableEntry:
    """Adapt every user function inside one regime slot entry.

    Walks `Phased` sides, per-target mappings, both functions of a `Decoupled`
    Koopmans aggregator, the function of a `DeterministicTransition`, the
    predicate and projections of a `ValueDependentConstraint`, the callables of
    a `JointTransition`, and the probability function of a
    `StochasticTransition`, which is decoded only: it returns probabilities over
    codes. With `adapters`, one authored function gets one adapter wherever it
    appears, so a function shared by both phases stays one function.
    """
    # Without value grids nothing changes. A grid declares no function. A
    # condition is a tree of names, not a function, and an age-specialized
    # marker builds its functions per age, so neither can be wrapped here; the
    # model refuses them beside a value grid. A household's stakeholder bodies
    # are adapted as the regime's `utility_<s>` functions. An identity law maps
    # codes to codes exactly as it maps values to values.
    if not value_grids or isinstance(
        entry,
        Condition
        | CollectiveUtility
        | _IdentityTransition
        | _AgeSpecialized
        | Grid
        | JointNormalIIDProcess
        | None,
    ):
        return entry
    adapters = {} if adapters is None else adapters
    if isinstance(entry, ValueDependentConstraint):
        return _adapt_value_dependent_constraint(
            entry=entry, value_grids=value_grids, adapters=adapters
        )
    if isinstance(entry, Phased):
        return _adapt_phased(
            entry=entry, value_grids=value_grids, encode=encode, adapters=adapters
        )
    if isinstance(entry, Mapping):
        adapted = {
            key: adapt_entry(
                entry=value, value_grids=value_grids, encode=encode, adapters=adapters
            )
            for key, value in entry.items()
        }
        if all(adapted[key] is value for key, value in entry.items()):
            return entry
        return cast("AdaptableEntry", MappingProxyType(adapted))
    if isinstance(entry, Decoupled):
        return _adapt_decoupled(entry=entry, value_grids=value_grids, adapters=adapters)
    if isinstance(entry, JointTransition):
        return _adapt_joint(entry=entry, value_grids=value_grids, adapters=adapters)
    if isinstance(entry, StochasticTransition | DeterministicTransition):
        # A stochastic law returns probabilities over codes, never a value.
        func = adapt_user_function(
            func=entry.func,
            value_grids=value_grids,
            encode=None if isinstance(entry, StochasticTransition) else encode,
            adapters=adapters,
        )
        return entry if func is entry.func else dataclasses.replace(entry, func=func)
    return adapt_user_function(
        func=cast("UserFunction", entry),
        value_grids=value_grids,
        encode=encode,
        adapters=adapters,
    )


# A container whose parts all come back unchanged is returned as declared: other
# checks compare declarations, such as a stakeholder's utility, by identity.


def _adapt_phased(
    *,
    entry: Phased,
    value_grids: Mapping[StateOrActionName, AdapterGrid],
    encode: DiscreteGrid | None,
    adapters: AdapterMemo,
) -> Phased:
    """Adapt both sides of a `Phased` entry, keeping the entry if neither changes."""
    solve, simulate = (
        adapt_entry(
            entry=side, value_grids=value_grids, encode=encode, adapters=adapters
        )
        for side in (entry.solve, entry.simulate)
    )
    if solve is entry.solve and simulate is entry.simulate:
        return entry
    return Phased(solve=solve, simulate=simulate)


def _adapt_value_dependent_constraint(
    *,
    entry: ValueDependentConstraint,
    value_grids: Mapping[StateOrActionName, AdapterGrid],
    adapters: AdapterMemo,
) -> ValueDependentConstraint:
    """Adapt the predicate and the projections, keeping the entry if neither changes."""
    predicate = adapt_user_function(
        func=entry.predicate, value_grids=value_grids, encode=None, adapters=adapters
    )
    references = {
        name: cast(
            "ProjectedRegimeValue",
            _adapt_reference(
                reference=reference, value_grids=value_grids, adapters=adapters
            ),
        )
        for name, reference in entry.references.items()
    }
    if predicate is entry.predicate and all(
        references[name] is reference for name, reference in entry.references.items()
    ):
        return entry
    return dataclasses.replace(entry, predicate=predicate, references=references)


def _adapt_joint(
    *,
    entry: JointTransition,
    value_grids: Mapping[StateOrActionName, AdapterGrid],
    adapters: AdapterMemo,
) -> JointTransition:
    """Adapt a joint lottery: each output returns its state's code.

    The probabilities and a callable support read values; a literal support is
    the author's own numbers, which the outputs read as declared.
    """
    probabilities = adapt_user_function(
        func=entry.probabilities,
        value_grids=value_grids,
        encode=None,
        adapters=adapters,
    )
    support = (
        adapt_user_function(
            func=cast("UserFunction", entry.support),
            value_grids=value_grids,
            encode=None,
            adapters=adapters,
        )
        if callable(entry.support)
        else entry.support
    )
    outputs = {
        name: adapt_user_function(
            func=func,
            value_grids=value_grids,
            encode=encoding_grid(value_grids=value_grids, name=name),
            adapters=adapters,
        )
        for name, func in entry.outputs.items()
    }
    if (
        probabilities is entry.probabilities
        and support is entry.support
        and all(outputs[name] is func for name, func in entry.outputs.items())
    ):
        return entry
    return dataclasses.replace(
        entry, probabilities=probabilities, support=support, outputs=outputs
    )


def _adapt_decoupled(
    *,
    entry: Decoupled,
    value_grids: Mapping[StateOrActionName, AdapterGrid],
    adapters: AdapterMemo,
) -> Decoupled:
    """Adapt the objective and the value, keeping the entry if neither changes."""
    objective, value = (
        adapt_entry(entry=part, value_grids=value_grids, encode=None, adapters=adapters)
        for part in (entry.objective, entry.value)
    )
    if objective is entry.objective and value is entry.value:
        return entry
    return Decoupled(
        objective=cast("UserFunction | Phased", objective),
        value=cast("UserFunction", value),
    )


@dataclass(frozen=True, kw_only=True, eq=False)
class ValueAdapter:
    """A user function the engine calls with the codes of value grids.

    A frozen callable instance rather than a nested function: the beartype claw
    memoizes every function it decorates, so a function defined per model build
    would pin that build's user function after the model is dropped. It sets no
    `__wrapped__`, so nothing that unwraps callables reaches the function that
    reads values.
    """

    func: UserFunction
    """The user function, which reads values."""
    decoded: MappingProxyType[str, DiscreteGrid]
    """The grid of each argument the function reads as a value."""
    codes: MappingProxyType[str, StateOrActionName] = MappingProxyType({})
    """The variable each `<name>_code` argument reads the code of."""
    positions: MappingProxyType[str, _ContinuousStochasticProcess] = MappingProxyType(
        {}
    )
    """The process of each `<name>_code` argument whose variable holds a node value."""
    encode: DiscreteGrid | None
    """The grid whose code the returned value names, if any."""

    def __post_init__(self) -> None:
        # The function's own attributes, declaration markers such as a case
        # piece's among them, stay visible on the adapter that stands in for it.
        for name, value in getattr(self.func, "__dict__", {}).items():
            if name not in _ADAPTER_OWN_ATTRIBUTES:
                object.__setattr__(self, name, value)
        for name in ("__module__", "__name__", "__qualname__", "__doc__"):
            value = getattr(self.func, name, None)
            if value is not None:
                object.__setattr__(self, name, value)
        # Array subscripts are read off the source of the function the user wrote.
        object.__setattr__(self, "__lcm_source__", self.func)
        # A `partial` keeps the keywords it fixes in its signature; the adapter
        # publishes only the arguments the engine supplies, and a `<name>_code`
        # argument as the variable `<name>` whose code the engine holds.
        free = _read_names(self.func)
        object.__setattr__(self, "_reads", free)
        code_reads = {
            *self.decoded,
            *(
                variable
                for name, variable in self.codes.items()
                if name not in self.positions
            ),
        }
        annotations = {
            self.codes.get(name, name): _CODE_ANNOTATION
            if self.codes.get(name, name) in code_reads
            else annotation
            for name, annotation in get_annotations(self.func).items()
            if name in free
        }
        return_annotation = get_annotations(self.func).get("return")
        if self.encode is not None:
            return_annotation = _CODE_ANNOTATION
        if return_annotation is not None:
            annotations["return"] = return_annotation
        signature = inspect.signature(self.func)
        published: dict[str, inspect.Parameter] = {}
        for parameter in signature.parameters.values():
            name = self.codes.get(parameter.name, parameter.name)
            if parameter.name in free and name not in published:
                published[name] = parameter.replace(
                    name=name,
                    annotation=annotations.get(name, parameter.annotation),
                )
        object.__setattr__(
            self,
            "__signature__",
            signature.replace(
                parameters=list(published.values()),
                return_annotation=annotations.get(
                    "return", signature.return_annotation
                ),
            ),
        )
        # `concatenate_functions` reads `__annotations__`, not `__signature__`.
        object.__setattr__(self, "__annotations__", annotations)

    @no_type_check
    def __call__(self, **kwargs: EconFunctionArg) -> ValueND:
        result = self.func(
            **{
                name: _node_position(
                    process=self.positions[name], value=kwargs[self.codes[name]]
                )
                if name in self.positions
                else kwargs[self.codes[name]]
                if name in self.codes
                else self.decoded[name].decode(kwargs[name])
                if name in self.decoded
                else kwargs[name]
                for name in self._reads
            }
        )
        return result if self.encode is None else self.encode.encode(result)


def _node_position(*, process: _ContinuousStochasticProcess, value: ValueND) -> ValueND:
    """Return the int32 position of the node a process value stands on."""
    coordinate = jnp.clip(process.get_coordinate(value), 0, process.n_points - 1)
    return jnp.rint(coordinate).astype(jnp.int32)


def _fail_if_a_position_has_no_node(
    *,
    func: UserFunction,
    positions: Mapping[str, _ContinuousStochasticProcess | UnreadableProcessCode],
    codes: Mapping[str, StateOrActionName],
) -> None:
    """Raise where `func` reads the node position of a process that may hold none.

    A process holds a node value while the model solves. A simulation drawing
    from the continuous law realises values between the nodes, and a process
    parameterized at runtime has no nodes while the model builds.
    """
    problems = [
        f"`{name}` ({process.reason})"
        if isinstance(process, UnreadableProcessCode)
        else f"`{name}` (fix the parameters of `{codes[name]}` at construction)"
        if not process.is_fully_specified
        else f"`{name}` (declare `{codes[name]}` with `simulate_draws='nodes'`)"
        for name, process in positions.items()
        if isinstance(process, UnreadableProcessCode) or not _stands_on_nodes(process)
    ]
    if problems:
        func_name = getattr(func, "__name__", type(func).__name__)
        raise RegimeInitializationError(
            f"'{func_name}' reads the node position of a process that does not "
            f"always stand on a node: {', '.join(problems)}."
        )


def _read_names(func: UserFunction) -> tuple[str, ...]:
    """Return the names a user function reads, without a `partial`'s fixed ones."""
    return tuple(get_free_arguments(func))


def _value_grid_name(
    *, name: str, value_grids: Mapping[StateOrActionName, AdapterGrid]
) -> StateOrActionName | None:
    """Return the value grid an argument reads: `x` or the next state `next_x`."""
    for grid_name in (name, name.removeprefix("next_")):
        grid = value_grids.get(grid_name)
        if isinstance(grid, DiscreteGrid) and grid.values is not None:
            return grid_name
    return None


def encoding_grid(
    *, value_grids: Mapping[StateOrActionName, AdapterGrid], name: str
) -> DiscreteGrid | None:
    """Return the value grid a returned value of `name` is encoded on, if any."""
    grid = value_grids.get(name)
    return grid if isinstance(grid, DiscreteGrid) and grid.values is not None else None


def _fail_if_an_array_is_indexed_by_a_value(
    *, func: UserFunction, decoded: Mapping[str, DiscreteGrid]
) -> None:
    """Raise where `func` indexes an array argument by a value-grid variable's value.

    An array is laid out over a variable's codes, so its position is the code;
    reading `<name>_code` gives it.
    """
    found = value_indexed_arrays(
        func=func,
        names=frozenset(name for name in decoded if not name.startswith("next_")),
    )
    if found:
        func_name = getattr(func, "__name__", type(func).__name__)
        reads = ", ".join(
            f"`{array}[{name}]` (write `{array}[{name}{CODE_SUFFIX}]`)"
            for array, name in found
        )
        raise RegimeInitializationError(
            f"'{func_name}' indexes an array by the value of a value grid: {reads}. "
            f"An array's positions are a variable's codes; read the code as the "
            f"argument `<name>{CODE_SUFFIX}`."
        )


def _fail_if_unsupported_reader(
    *,
    regimes: Mapping[RegimeName, UserRegime],
    value_grids: Mapping[StateOrActionName, DiscreteGrid],
    model_derived_categoricals: Mapping[FunctionName, DiscreteGrid],
) -> None:
    """Raise where a value grid meets a declaration that would read its codes."""
    problems: list[str] = []
    for regime_name, regime in regimes.items():
        declared = {
            name
            for name in (
                *regime.states,
                *regime.actions,
                *regime.derived_categoricals,
                *model_derived_categoricals,
            )
            if name in value_grids
        }
        problems.extend(
            f"the imputation of the carried state '{name}' of '{regime_name}'"
            for name, entry in regime.states.items()
            # A marker's published *args/**kwargs do not describe the
            # concrete reader, so no adapter can be built for it.
            if isinstance(entry, Phased)
            and bool(declared)
            and _contains_age_specialized(entry)
        )
        problems.extend(
            f"the condition constraint '{name}' of '{regime_name}'"
            for name, entry in regime.constraints.items()
            if isinstance(entry, Condition) and entry.dependencies & set(value_grids)
        )
        if declared:
            for slot_name, slot in (
                ("function", regime.decomposed_functions),
                ("constraint", regime.constraints),
                ("state transition", regime.state_transitions),
            ):
                problems.extend(
                    f"the age-specialized {slot_name} '{name}' of '{regime_name}'"
                    for name, entry in slot.items()
                    if _contains_age_specialized(entry)
                )
    if problems:
        raise RegimeInitializationError(
            "Value grids are not supported where these declarations read them: "
            + "; ".join(problems)
            + ". Declare these variables with a category class."
        )


def _contains_age_specialized(entry: AdaptableEntry | StateEntry) -> bool:
    """Inspect marker leaves, never the generic signature of a build-time marker."""
    if isinstance(entry, Phased):
        return _contains_age_specialized(entry.solve) or _contains_age_specialized(
            entry.simulate
        )
    if isinstance(entry, Mapping):
        return any(_contains_age_specialized(cell) for cell in entry.values())
    if isinstance(entry, StochasticTransition):
        return _contains_age_specialized(entry.func)
    return isinstance(entry, AgeSpecializedFunction)


def _value_names(value_grids: Mapping[StateOrActionName, DiscreteGrid]) -> set[str]:
    """Return every argument name that reads a value grid: `x` and `next_x`."""
    return {*value_grids, *(f"next_{name}" for name in value_grids)}
