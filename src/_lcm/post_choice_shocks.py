"""Lower `Regime(post_choice_shocks=...)` onto transition-local draws.

A post-choice group is realised between a regime's choice and its next states, so
only the regime's state transitions read it, as `next_<shock>`. Each shock becomes
a fixed node set: the solve sums the source's continuation over its nodes with
its weights, and simulation samples one node per subject from the same weights.
The regime stores no axis for it.
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
    ReferenceName,
    RegimeName,
    ScalarFloat,
    ScalarInt,
    UserFunction,
)

type PostChoiceShocks = MappingProxyType[
    RegimeName, MappingProxyType[ShockName, PostChoiceShock]
]


def lower_post_choice_shocks(
    *,
    regimes: Mapping[RegimeName, Regime],
    laws: RegimeLaws,
    groups: Mapping[ShockGroupName, ShockGroup],
) -> PostChoiceShocks:
    """Validate every regime's post-choice groups and lower each shock to its nodes.

    Returns:
        Immutable mapping of regime names to that regime's post-choice shocks.

    Raises:
        ModelInitializationError: If a group is unknown, also realised before a
            choice, not lowerable to one fixed node set per shock, or read by
            something evaluated before the draw.

    """
    _fail_if_groups_are_unusable(regimes=regimes, groups=groups)
    _fail_if_read_before_the_draw(regimes=regimes, laws=laws, groups=groups)
    lowered = {
        group_name: _lower_group(group_name=group_name, group=groups[group_name])
        for regime in regimes.values()
        for group_name in regime.post_choice_shocks
    }
    return MappingProxyType(
        {
            regime_name: MappingProxyType(
                {
                    shock: process
                    for group_name in regime.post_choice_shocks
                    for shock, process in lowered[group_name].items()
                }
            )
            for regime_name, regime in regimes.items()
            if regime.post_choice_shocks
        }
    )


@dataclass(frozen=True, kw_only=True)
class PostChoiceShock(_IIDProcess):
    """One post-choice shock's physical nodes and weights, fixed at model build."""

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
) -> None:
    """Refuse a non-GridSearch regime, and a group that is unknown, current
    elsewhere, or has no fixed node set."""
    current = {
        name
        for regime in regimes.values()
        for name in (
            (*regime.shocks.solve, *regime.shocks.simulate)
            if isinstance(regime.shocks, Phased)
            else regime.shocks
        )
    }
    errors: list[str] = []
    for regime_name, regime in regimes.items():
        if regime.post_choice_shocks and not isinstance(regime.solver, GridSearch):
            errors.append(
                f"Regime '{regime_name}' declares post-choice shocks, which only "
                "the `GridSearch` solver supports."
            )
        for group_name in regime.post_choice_shocks:
            group = groups.get(group_name)
            if group is None:
                errors.append(
                    f"Regime '{regime_name}' declares post-choice shock group "
                    f"'{group_name}', which `Model(shocks=...)` does not define."
                )
            elif group_name in current:
                errors.append(
                    f"Shock group '{group_name}' is post-choice in regime "
                    f"'{regime_name}' and current in another regime. Declare a "
                    "separate group for each timing."
                )
            elif problem := _unlowerable(group):
                errors.append(
                    f"Post-choice shock group '{group_name}' of regime "
                    f"'{regime_name}' {problem}"
                )
    if errors:
        raise ModelInitializationError(format_messages(sorted(set(errors))))


def _unlowerable(group: ShockGroup) -> str | None:
    """Explain why a group has no single fixed node set per shock, else `None`."""
    laws = tuple(group.shocks.values())
    problems = (
        (
            any(isinstance(law, Discrete) for law in laws),
            "declares a discrete law; post-choice shocks must be continuous.",
        ),
        (
            any(
                not isinstance(getattr(law, item.name), float)
                for law in laws
                for item in fields(law)
            ),
            (
                "has a runtime or state-dependent law parameter; every "
                "post-choice law parameter must be a fixed number."
            ),
        ),
        (
            group.correlation is not _Independence.DEFAULT,
            "declares a correlation; post-choice shocks must be independent.",
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
) -> dict[ShockName, PostChoiceShock]:
    """Realise each shock of an independent group on its own one-dimensional rule."""
    lowered: dict[ShockName, PostChoiceShock] = {}
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
        lowered[shock] = PostChoiceShock(
            n_points=len(weights),
            nodes=tuple(float(v) for v in realised.values[shock]),
            weights=weights,
        )
    return lowered


def _fail_if_read_before_the_draw(
    *,
    regimes: Mapping[RegimeName, Regime],
    laws: RegimeLaws,
    groups: Mapping[ShockGroupName, ShockGroup],
) -> None:
    """Refuse a read of a post-choice shock from anything evaluated before the draw.

    The roots are what the regime evaluates within its period (utility,
    constraints, the regime transition, ...); every regime function they reach is
    walked, so a helper that reads the shock is caught as well.
    """
    errors: list[str] = []
    for regime_name, regime in regimes.items():
        shock_args = {
            arg: (shock, group_name)
            for group_name in regime.post_choice_shocks
            for shock in groups[group_name].shocks
            for arg in (shock, f"next_{shock}")
        }
        if not shock_args:
            continue
        for phase in ("solve", "simulate"):
            functions = _phase_functions(regime=regime, phase=phase)
            for key, root in root_functions(
                regime_name=regime_name, regime=regime, laws=laws, phase=phase
            ).items():
                root_name = _root_name(key)
                for func_name, arg in _reads(
                    root_name=root_name,
                    root=root,
                    functions=functions,
                    names=shock_args,
                ):
                    shock, group_name = shock_args[arg]
                    via = (
                        ""
                        if func_name == root_name
                        else f", and '{root_name}' depends on '{func_name}'"
                    )
                    errors.append(
                        f"Regime '{regime_name}': '{func_name}' reads '{arg}', the "
                        f"post-choice shock '{shock}' of group '{group_name}'{via}. "
                        "A post-choice shock is drawn after the choice, so only "
                        f"state transitions may read it, as 'next_{shock}'. Read it "
                        "in a state transition or a function only state transitions "
                        "use, or declare the group in `shocks=` to realise it "
                        "before the choice."
                    )
    if errors:
        raise ModelInitializationError(format_messages(sorted(set(errors))))


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


def _root_name(key: str) -> str:
    """Name a root by the user's slot name where it has one."""
    kind, _, name = key.removeprefix("__").partition("__")
    if name and kind in {"utility", "constraint", "derived_categorical", "output_only"}:
        return name
    return key.removeprefix("__")


def _reads(
    *,
    root_name: str,
    root: UserFunction,
    functions: Mapping[FunctionName, UserFunction],
    names: Mapping[ReferenceName, tuple[ShockName, ShockGroupName]],
) -> list[tuple[str, ReferenceName]]:
    """List `(function, argument)` pairs where `root`'s closure reads one of `names`."""
    found: list[tuple[str, ReferenceName]] = []
    frontier: list[tuple[str, UserFunction]] = [(root_name, root)]
    visited = {root_name}
    while frontier:
        func_name, func = frontier.pop()
        for arg in inspect.signature(func).parameters:
            if arg in names:
                found.append((func_name, arg))
            elif arg in functions and arg not in visited:
                visited.add(arg)
                frontier.append((arg, functions[arg]))
    return found
