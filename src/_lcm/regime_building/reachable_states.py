"""Reachable tuples of a declared set of deterministic discrete states.

The enumeration reads only the engine's own laws, constraint pools and fixed
parameters; it never solves. From declared start codes at the admissible initial
nodes, each period pushes every reachable tuple of the named states through their
laws toward every target the static regime graph of either phase retains:

- every code of each other discrete state or discrete action that the laws or the
  evaluable constraints read is tried, so the sets do not depend on probabilities;
- a constraint is evaluable when it reads only discrete variables, the period, the
  age and fixed parameters; any other constraint is skipped, which can only enlarge
  the sets, and is reported;
- a law that reads anything else, a stochastic law, or a state held on a
  non-discrete grid is refused, naming the law or state.

The union of every node's tuples is one code set for all periods; a tuple in it is
reachable at the periods whose nodes list it.
"""

import inspect
import itertools
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

import jax
import jax.numpy as jnp
import numpy as np
from dags import concatenate_functions

from _lcm.engine import Regime, SimulationPhase, SolutionPhase
from _lcm.grids import DiscreteGrid, Grid
from _lcm.time import TimeAxis
from _lcm.typing import (
    FlatRegimeParams,
    FunctionName,
    ReferenceName,
    RegimeName,
    StateName,
)
from lcm.exceptions import ModelInitializationError
from lcm.typing import UserAge

type Phase = SolutionPhase | SimulationPhase
type Codes = tuple[int, ...]

_MAX_ROWS_PER_EVALUATION = 1 << 20
_TIME_ARGUMENTS = frozenset({"period", "age"})


@dataclass(frozen=True, kw_only=True)
class ReachableStates:
    """Reachable code tuples of the named states, per node and over all periods."""

    state_names: tuple[StateName, ...]
    """The enumerated states, in tuple order."""

    by_node: MappingProxyType[tuple[int, RegimeName], frozenset[Codes]]
    """Per `(period, regime)` carrying the states, the tuples reachable there."""

    union: tuple[Codes, ...]
    """Sorted union of every node's tuples."""

    carriers: frozenset[RegimeName]
    """Regimes that carry every named state."""

    skipped_constraints: MappingProxyType[
        tuple[RegimeName, FunctionName], tuple[ReferenceName, ...]
    ]
    """Constraints that read a non-enumerable input, with those inputs."""


def enumerate_reachable_states(
    *,
    regimes: Mapping[RegimeName, Regime],
    initial_nodes: frozenset[tuple[UserAge, RegimeName]],
    ages: TimeAxis,
    state_names: Sequence[StateName],
    start_codes: Mapping[StateName, Sequence[int]],
) -> ReachableStates:
    """Enumerate the tuples of `state_names` reachable from the declared starts.

    Args:
        regimes: Mapping of regime names to the canonical engine regimes.
        initial_nodes: The admissible `(age, regime)` starts.
        ages: The model's time axis.
        state_names: The deterministic discrete states to enumerate jointly.
        start_codes: Mapping of each named state to its admissible start codes.

    Returns:
        The reachable tuples per node and their union.

    Raises:
        ModelInitializationError: If a named state is not discrete in a regime
            carrying it, a regime carries only some of them, a regime without them
            leads into one with them, a start code is off the grid, or a law of a
            named state is stochastic or reads a non-enumerable input.

    """
    members = tuple(state_names)
    carriers = _carriers(regimes=regimes, members=members)
    _fail_if_start_codes_are_invalid(
        regimes=regimes, carriers=carriers, members=members, start_codes=start_codes
    )
    period_of_age = {age: period for period, age in enumerate(ages.exact_values)}
    evaluator = _SuccessorEvaluator(regimes=regimes, members=members, ages=ages)
    by_node: dict[tuple[int, RegimeName], set[Codes]] = {}
    starts = set(itertools.product(*(tuple(start_codes[m]) for m in members)))
    for age, name in initial_nodes:
        if name in carriers:
            by_node.setdefault((period_of_age[age], name), set()).update(starts)
    edges = _edges(regimes=regimes, carriers=carriers)
    for period in range(len(ages.exact_values) - 1):
        for source, target, phase_name in edges.get(period, ()):
            tuples = by_node.get((period, source))
            if tuples:
                by_node.setdefault((period + 1, target), set()).update(
                    evaluator.successors(
                        phase_name=phase_name,
                        source=source,
                        target=target,
                        period=period,
                        tuples=tuples,
                    )
                )
    return ReachableStates(
        state_names=members,
        by_node=MappingProxyType(
            {node: frozenset(tuples) for node, tuples in sorted(by_node.items())}
        ),
        union=tuple(sorted(set().union(*by_node.values()))),
        carriers=carriers,
        skipped_constraints=MappingProxyType(dict(sorted(evaluator.skipped.items()))),
    )


def _carriers(
    *, regimes: Mapping[RegimeName, Regime], members: tuple[StateName, ...]
) -> frozenset[RegimeName]:
    """Return regimes carrying every member; refuse partial or non-discrete ones."""
    carriers: set[RegimeName] = set()
    for name, regime in sorted(regimes.items()):
        grids = regime.solution.grids
        held = [m for m in members if m in grids]
        if not held:
            continue
        if len(held) != len(members):
            missing = sorted(set(members) - set(held))
            msg = (
                f"Regime {name!r} carries {held} but not {missing}; every regime "
                "must carry all enumerated states or none of them."
            )
            raise ModelInitializationError(msg)
        for member in members:
            if not isinstance(grids[member], DiscreteGrid):
                msg = (
                    f"State {member!r} in regime {name!r} is held on "
                    f"{type(grids[member]).__name__}; only states on a discrete "
                    "grid can be enumerated."
                )
                raise ModelInitializationError(msg)
        carriers.add(name)
    return frozenset(carriers)


def _fail_if_start_codes_are_invalid(
    *,
    regimes: Mapping[RegimeName, Regime],
    carriers: frozenset[RegimeName],
    members: tuple[StateName, ...],
    start_codes: Mapping[StateName, Sequence[int]],
) -> None:
    """Require nonempty start codes on every member's grid."""
    if not carriers:
        msg = f"No regime carries the states {list(members)}."
        raise ModelInitializationError(msg)
    grids = regimes[min(carriers)].solution.grids
    for member in members:
        codes = tuple(start_codes.get(member, ()))
        valid = set(_codes(grids[member]))
        if not codes or not set(codes) <= valid:
            msg = (
                f"Start codes of {member!r} must be a nonempty subset of its codes "
                f"{sorted(valid)}; got {codes!r}."
            )
            raise ModelInitializationError(msg)


def _edges(
    *, regimes: Mapping[RegimeName, Regime], carriers: frozenset[RegimeName]
) -> dict[int, tuple[tuple[RegimeName, RegimeName, str], ...]]:
    """Return each period's carrier-to-carrier edges of either phase.

    Refuses an edge entering a carrier from a regime without the states, since the
    enumeration has no tuple to start that edge from.
    """
    edges: dict[int, set[tuple[RegimeName, RegimeName, str]]] = {}
    for phase_name in ("solution", "simulation"):
        reachability = getattr(next(iter(regimes.values())), phase_name).reachability
        for period in range(reachability.n_periods - 1):
            for source in sorted(reachability.active_regimes_by_period[period]):
                for target in reachability.targets(period=period, source=source):
                    if target not in carriers:
                        continue
                    if source not in carriers:
                        msg = (
                            f"Edge {source} -> {target} at period {period} enters a "
                            "regime carrying the enumerated states from one without "
                            "them."
                        )
                        raise ModelInitializationError(msg)
                    edges.setdefault(period, set()).add((source, target, phase_name))
    return {period: tuple(sorted(e)) for period, e in sorted(edges.items())}


class _SuccessorEvaluator:
    """Evaluate member laws and evaluable constraints on enumerated rows."""

    def __init__(
        self,
        *,
        regimes: Mapping[RegimeName, Regime],
        members: tuple[StateName, ...],
        ages: TimeAxis,
    ) -> None:
        self._regimes = regimes
        self._members = members
        self._ages = ages
        self._programs: dict[tuple[str, RegimeName, RegimeName, int], _Program] = {}
        self.skipped: dict[tuple[RegimeName, FunctionName], tuple[ReferenceName, ...]]
        self.skipped = {}

    def successors(
        self,
        *,
        phase_name: str,
        source: RegimeName,
        target: RegimeName,
        period: int,
        tuples: set[Codes],
    ) -> set[Codes]:
        """Return the member tuples entering `target` from `tuples` at `period`."""
        program = self._program(
            phase_name=phase_name, source=source, target=target, period=period
        )
        rows = np.asarray(sorted(tuples), dtype=np.int32).reshape(
            len(tuples), len(self._members)
        )
        out: set[Codes] = set()
        chunk = max(1, _MAX_ROWS_PER_EVALUATION // len(program.other_rows))
        for start in range(0, len(rows), chunk):
            out.update(program.evaluate(rows[start : start + chunk]))
        return out

    def _program(
        self, *, phase_name: str, source: RegimeName, target: RegimeName, period: int
    ) -> _Program:
        key = (phase_name, source, target, period)
        if key not in self._programs:
            regime = self._regimes[source]
            phase: Phase = getattr(regime, phase_name)
            self._programs[key] = _build_program(
                regime_name=source,
                phase=phase,
                target=target,
                period=period,
                age=self._ages.exact_values[period],
                fixed_params=MappingProxyType(
                    {
                        **regime.resolved_fixed_params,
                        **getattr(phase, "resolved_fixed_params", {}),
                    }
                ),
                members=self._members,
                skipped=self.skipped,
            )
        return self._programs[key]


@dataclass(frozen=True, kw_only=True)
class _Program:
    """One source-target-period evaluation over members times other discrete codes."""

    func: Callable[..., tuple[dict[str, jax.Array], jax.Array]]
    members: tuple[StateName, ...]
    member_reads: tuple[StateName, ...]
    other_names: tuple[ReferenceName, ...]
    other_rows: np.ndarray
    law_members: tuple[StateName, ...]

    def evaluate(self, rows: np.ndarray) -> set[Codes]:
        """Return the successors of `rows` under every allowed code of the others."""
        n_other = len(self.other_rows)
        histories = np.repeat(rows, n_other, axis=0)
        others = np.tile(self.other_rows, (len(rows), 1))
        kwargs = {
            name: jnp.asarray(histories[:, self.members.index(name)])
            for name in self.member_reads
        } | {name: jnp.asarray(others[:, i]) for i, name in enumerate(self.other_names)}
        nexts, feasible = jax.vmap(self.func)(**kwargs) if kwargs else self.func()
        feasible = np.broadcast_to(np.asarray(feasible), (len(histories),))
        successors = histories.copy()
        for member in self.law_members:
            successors[:, self.members.index(member)] = np.broadcast_to(
                np.asarray(nexts[f"next_{member}"]), (len(histories),)
            )
        return set(map(tuple, successors[feasible].tolist()))


def _build_program(
    *,
    regime_name: RegimeName,
    phase: Phase,
    target: RegimeName,
    period: int,
    age: UserAge,
    fixed_params: FlatRegimeParams,
    members: tuple[StateName, ...],
    skipped: dict[tuple[RegimeName, FunctionName], tuple[ReferenceName, ...]],
) -> _Program:
    """Compose the member laws toward `target` with the evaluable constraints."""
    functions, constraints = phase.feasibility_pools_by_period[period]
    discrete = frozenset(
        name for name, grid in phase.grids.items() if isinstance(grid, DiscreteGrid)
    )

    def offending(
        func: Callable[..., Mapping[str, jax.Array] | jax.Array],
    ) -> tuple[ReferenceName, ...]:
        return tuple(
            sorted(
                arg
                for arg in inspect.signature(func).parameters
                if arg not in discrete
                and arg not in _TIME_ARGUMENTS
                and arg not in fixed_params
            )
        )

    plan = phase.transition_plans[target]
    laws = {}
    for member in members:
        law_name = f"next_{member}"
        output = plan.outputs.get(member)
        if plan.is_lottery(law_name) or (
            output is not None and output.lottery_dependencies
        ):
            msg = (
                f"The law {law_name} of {member!r} from regime {regime_name!r} to "
                f"{target!r} is stochastic; only deterministic states can be "
                "enumerated."
            )
            raise ModelInitializationError(msg)
        laws[law_name] = phase.transitions[target][law_name]
    law_func = concatenate_functions(
        functions=dict(functions) | laws,
        targets=list(laws),
        return_type="dict",
        enforce_signature=False,
    )
    if bad := offending(law_func):
        msg = (
            f"The laws {sorted(laws)} from regime {regime_name!r} to {target!r} "
            f"read {list(bad)}, which are neither discrete variables, the period, "
            "the age nor fixed parameters, so their support cannot be enumerated."
        )
        raise ModelInitializationError(msg)
    usable = {}
    for name, constraint in constraints.items():
        reads = concatenate_functions(
            functions=dict(functions) | {name: constraint},
            targets=[name],
            enforce_signature=False,
        )
        if bad := offending(reads):
            skipped.setdefault((regime_name, name), bad)
        else:
            usable[name] = constraint
    feasibility = (
        concatenate_functions(
            functions=dict(functions) | usable,
            targets=list(usable),
            aggregator=jnp.logical_and,
            enforce_signature=False,
        )
        if usable
        else None
    )
    law_args = tuple(inspect.signature(law_func).parameters)
    feasibility_args = (
        () if feasibility is None else tuple(inspect.signature(feasibility).parameters)
    )
    leaves = set(law_args) | set(feasibility_args)
    scalars = {
        name: value for name, value in fixed_params.items() if name in leaves
    } | {
        name: value
        for name, value in (("period", jnp.int32(period)), ("age", jnp.asarray(age)))
        if name in leaves
    }
    member_reads = tuple(m for m in members if m in leaves)
    other_names = tuple(
        sorted(name for name in leaves if name in discrete and name not in members)
    )
    other_rows = np.asarray(
        list(itertools.product(*(_codes(phase.grids[name]) for name in other_names))),
        dtype=np.int32,
    ).reshape(-1, len(other_names))

    def evaluate(**row: jax.Array) -> tuple[dict[str, jax.Array], jax.Array]:
        values = row | scalars
        nexts = law_func(**{arg: values[arg] for arg in law_args})
        if feasibility is None:
            return nexts, jnp.ones((), dtype=bool)
        allowed = feasibility(**{arg: values[arg] for arg in feasibility_args})
        return nexts, jnp.asarray(allowed)

    return _Program(
        func=evaluate,
        members=members,
        member_reads=member_reads,
        other_names=other_names,
        other_rows=other_rows,
        law_members=tuple(m for m in members if f"next_{m}" in laws),
    )


def _codes(grid: Grid) -> tuple[int, ...]:
    """Return a discrete grid's codes as Python integers."""
    if not isinstance(grid, DiscreteGrid):
        msg = f"Expected a discrete grid, got {type(grid).__name__}."
        raise TypeError(msg)
    return tuple(np.asarray(grid.codes).tolist())
