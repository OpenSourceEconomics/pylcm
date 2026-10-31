"""Infer which `Model(shocks=...)` groups a regime reads only as their draw.

A regime lists a group in `Regime(shocks=...)`. Reading a shock `x` means this
period's realization, known at the choice; reading `next_x` means the draw after
the choice. A group whose shocks the regime reads only as `next_x` is integrated
out: each shock becomes a fixed node set, the solve sums the source's
continuation over its nodes with its weights, and simulation samples one node
per subject from the same weights. The regime stores no axis for it.
"""

import inspect
from collections.abc import Mapping
from dataclasses import dataclass, fields
from types import MappingProxyType
from typing import ClassVar, Literal, cast

import jax
import jax.numpy as jnp

from _lcm.processes import _IIDProcess
from _lcm.regime_building.broadcast import root_functions
from _lcm.regime_law import RegimeLaws
from _lcm.typing import PRNGKeyND
from _lcm.utils.error_messages import format_messages
from lcm.exceptions import ModelInitializationError
from lcm.phased import Phased
from lcm.regime import Regime
from lcm.shocks import (
    Discrete,
    MonteCarlo,
    Product,
    QuasiMonteCarlo,
    ShockGroup,
    ShockGroupName,
    ShockName,
    _Independence,
)
from lcm.solvers import GridSearch
from lcm.typing import (
    Float1D,
    FloatND,
    FunctionName,
    RegimeName,
    ScalarFloat,
    ScalarInt,
    UserFunction,
)

type PostDecisionShocks = MappingProxyType[
    RegimeName, MappingProxyType[ShockName, PostDecisionShock]
]


def lower_post_decision_shocks(
    *,
    regimes: Mapping[RegimeName, Regime],
    laws: RegimeLaws,
    groups: Mapping[ShockGroupName, ShockGroup],
) -> tuple[MappingProxyType[RegimeName, Regime], PostDecisionShocks]:
    """Take every group a regime reads only as its draw out of `Regime.shocks`.

    Returns:
        Tuple of the regimes without their post-decision groups and, per regime that
        has any, its post-decision shocks lowered to their nodes.

    Raises:
        ModelInitializationError: If a regime reads a group both as its value and
            as its draw, or a post-decision group is unusable: also read as a value
            in another regime, not lowerable to one fixed node set per shock, or
            declared on a regime whose solver cannot integrate it.

    """
    post_decision_groups = {
        name: _post_decision_groups(
            regime_name=name, regime=regime, laws=laws, groups=groups
        )
        for name, regime in regimes.items()
    }
    _fail_if_groups_are_unusable(
        regimes=regimes, groups=groups, post_decision_groups=post_decision_groups
    )
    lowered = {
        group_name: _lower_group(group_name=group_name, group=groups[group_name])
        for names in post_decision_groups.values()
        for group_name in names
    }
    stripped = {
        name: regime.replace(
            shocks=_without(declared=regime.shocks, removed=post_decision_groups[name])
        )
        if post_decision_groups[name]
        else regime
        for name, regime in regimes.items()
    }
    return MappingProxyType(stripped), MappingProxyType(
        {
            regime_name: MappingProxyType(
                {
                    shock: process
                    for group_name in names
                    for shock, process in lowered[group_name].items()
                }
            )
            for regime_name, names in post_decision_groups.items()
            if names
        }
    )


def _without(
    *, declared: tuple[ShockGroupName, ...], removed: tuple[ShockGroupName, ...]
) -> tuple[ShockGroupName, ...]:
    if isinstance(declared, Phased):
        return cast(
            "tuple[ShockGroupName, ...]",
            Phased(
                solve=_without(declared=declared.solve, removed=removed),
                simulate=_without(declared=declared.simulate, removed=removed),
            ),
        )
    return tuple(name for name in declared if name not in removed)


def _post_decision_groups(
    *,
    regime_name: RegimeName,
    regime: Regime,
    laws: RegimeLaws,
    groups: Mapping[ShockGroupName, ShockGroup],
) -> tuple[ShockGroupName, ...]:
    """Name the listed groups the regime reads as `next_<shock>` and never as a value.

    Raises:
        ModelInitializationError: If a group is read both ways.

    """
    names = (
        (*regime.shocks.solve, *regime.shocks.simulate)
        if isinstance(regime.shocks, Phased)
        else regime.shocks
    )
    listed = {name: groups[name] for name in dict.fromkeys(names) if name in groups}
    if not listed:
        return ()
    reads: set[str] = set()
    for phase in ("solve", "simulate"):
        functions = _phase_functions(regime=regime, phase=phase)
        roots = dict(
            root_functions(
                regime_name=regime_name, regime=regime, laws=laws, phase=phase
            )
        ) | _law_roots(regime=regime, phase=phase)
        for root_name, root in roots.items():
            reads |= _reads(root_name=root_name, root=root, functions=functions)
    post_decision_groups: list[ShockGroupName] = []
    errors: list[str] = []
    for group_name, group in listed.items():
        values = sorted(shock for shock in group.shocks if shock in reads)
        draws = sorted(
            f"next_{shock}" for shock in group.shocks if f"next_{shock}" in reads
        )
        if values and draws:
            errors.append(
                f"Regime '{regime_name}' reads {values} and {draws} of shock group "
                f"'{group_name}'. A group's value is integrated within the period "
                "it is realized in, so no later period can condition on the draw. "
                "To carry the realization into the next period, declare the shock "
                "as an IID process state, such as `NormalIIDProcess`, in "
                "`states=`; reading both its value and its draw keeps its axis."
            )
        elif draws:
            post_decision_groups.append(group_name)
    if errors:
        raise ModelInitializationError(format_messages(errors))
    return tuple(post_decision_groups)


def _law_roots(
    *, regime: Regime, phase: Literal["solve", "simulate"]
) -> dict[str, UserFunction]:
    """Key every law of motion the regime declares, toward any target."""
    roots: dict[str, UserFunction] = {}
    for state, raw in regime.state_transitions.items():
        entry = (
            (raw.solve if phase == "solve" else raw.simulate)
            if isinstance(raw, Phased)
            else raw
        )
        cells = entry.items() if isinstance(entry, Mapping) else (("", entry),)
        for target, cell in cells:
            law = (
                (cell.solve if phase == "solve" else cell.simulate)
                if isinstance(cell, Phased)
                else cell
            )
            if callable(law):
                roots[f"__law_{state}_{target}"] = cast("UserFunction", law)
    return roots


@dataclass(frozen=True, kw_only=True)
class PostDecisionShock(_IIDProcess):
    """One post-decision shock's physical nodes and weights, fixed at model build."""

    nodes: tuple[float, ...]
    """Physical values, in the group's node order."""
    weights: tuple[float, ...]
    """Integration weights, one per node, summing to one."""

    _NON_PARAM_FIELDS: ClassVar[frozenset[str]] = (
        _IIDProcess._NON_PARAM_FIELDS | {"nodes", "weights"}  # noqa: SLF001
    )

    def compute_gridpoints(self, **kwargs: ScalarFloat | ScalarInt) -> Float1D:  # noqa: ARG002
        return jnp.asarray(self.nodes)

    def compute_transition_probs(self, **kwargs: ScalarFloat | ScalarInt) -> FloatND:  # noqa: ARG002
        return jnp.full(
            (self.n_points, self.n_points), fill_value=jnp.asarray(self.weights)
        )

    def draw_shock(
        self,
        *,
        params: MappingProxyType[str, ScalarFloat | ScalarInt],  # noqa: ARG002
        key: PRNGKeyND,
    ) -> ScalarFloat:
        index = jax.random.choice(key, self.n_points, p=jnp.asarray(self.weights))
        return jnp.asarray(self.nodes)[index]


def _fail_if_groups_are_unusable(
    *,
    regimes: Mapping[RegimeName, Regime],
    groups: Mapping[ShockGroupName, ShockGroup],
    post_decision_groups: Mapping[RegimeName, tuple[ShockGroupName, ...]],
) -> None:
    """Refuse a post-decision group on a non-GridSearch regime, read as a value in
    another regime, or without a fixed node set."""
    current = {
        name
        for regime_name, regime in regimes.items()
        for name in (
            (*regime.shocks.solve, *regime.shocks.simulate)
            if isinstance(regime.shocks, Phased)
            else regime.shocks
        )
        if name not in post_decision_groups[regime_name]
    }
    errors: list[str] = []
    for regime_name, regime in regimes.items():
        if post_decision_groups[regime_name] and not isinstance(
            regime.solver, GridSearch
        ):
            errors.append(
                f"Regime '{regime_name}' reads shock group(s) "
                f"{list(post_decision_groups[regime_name])} only as their draw "
                "`next_<shock>`. "
                "Only the `GridSearch` solver integrates such a draw."
            )
        for group_name in post_decision_groups[regime_name]:
            group = groups[group_name]
            if group_name in current:
                errors.append(
                    f"Shock group '{group_name}' is read only as its draw in regime "
                    f"'{regime_name}' and as its value in another regime. Declare a "
                    "separate group for each timing."
                )
            elif problem := _unlowerable(group):
                errors.append(
                    f"Shock group '{group_name}', read only as its draw in regime "
                    f"'{regime_name}', {problem}"
                )
    if errors:
        raise ModelInitializationError(format_messages(sorted(set(errors))))


def _unlowerable(group: ShockGroup) -> str | None:
    """Explain why a group has no single fixed node set per shock, else `None`."""
    laws = tuple(group.shocks.values())
    problems = (
        (
            any(isinstance(law, Discrete) for law in laws),
            "declares a discrete law; a post-decision shock must be continuous.",
        ),
        (
            any(
                not isinstance(getattr(law, item.name), float)
                for law in laws
                for item in fields(law)
            ),
            (
                "has a runtime or state-dependent law parameter; every "
                "post-decision law parameter must be a fixed number."
            ),
        ),
        (
            group.correlation is not _Independence.DEFAULT,
            "declares a correlation; post-decision shocks must be independent.",
        ),
        (
            isinstance(group.nodes, MonteCarlo | QuasiMonteCarlo),
            (
                "uses a sampled node rule, which differs by age; use a "
                "deterministic rule such as `GaussHermite`."
            ),
        ),
        (group.signed, "has signed weights, which simulation cannot sample."),
        (
            len(laws) > 1 and not isinstance(group.nodes, Product),
            "combines several shocks without a `Product` rule.",
        ),
    )
    return next((message for failed, message in problems if failed), None)


def _lower_group(
    *, group_name: ShockGroupName, group: ShockGroup
) -> dict[ShockName, PostDecisionShock]:
    """Realise each shock of an independent group on its own one-dimensional rule."""
    lowered: dict[ShockName, PostDecisionShock] = {}
    for shock, law in group.shocks.items():
        single = (
            group
            if len(group.shocks) == 1
            else ShockGroup(
                shocks={shock: law},
                nodes=Product(**{shock: cast("Product", group.nodes).rules[shock]}),
            )
        )
        realised = single.realise(
            nodes=single.resolve_nodes(group_name=group_name, age=0)
        )
        realised.raise_if_invalid()
        weights = tuple(float(w) for w in realised.weights)
        lowered[shock] = PostDecisionShock(
            n_points=len(weights),
            nodes=tuple(float(v) for v in realised.values[shock]),
            weights=weights,
        )
    return lowered


def _phase_functions(
    *, regime: Regime, phase: Literal["solve", "simulate"]
) -> dict[FunctionName, UserFunction]:
    return {
        name: cast(
            "UserFunction",
            (func.solve if phase == "solve" else func.simulate)
            if isinstance(func, Phased)
            else func,
        )
        for name, func in regime.decomposed_functions.items()
    }


def _reads(
    *,
    root_name: str,
    root: UserFunction,
    functions: Mapping[FunctionName, UserFunction],
) -> set[str]:
    """Collect every argument name `root`'s closure through regime functions reads."""
    found: set[str] = set()
    frontier: list[UserFunction] = [root]
    visited = {root_name}
    while frontier:
        for arg in inspect.signature(frontier.pop()).parameters:
            found.add(arg)
            if arg in functions and arg not in visited:
                visited.add(arg)
                frontier.append(functions[arg])
    return found
