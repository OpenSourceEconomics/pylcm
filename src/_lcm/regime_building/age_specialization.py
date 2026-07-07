"""Resolve and fingerprint per-age-specialized DAG nodes.

`AgeSpecializedFunction` marks a function whose closure is bound per age at build time.
Before the DAG is traced, pylcm resolves each marked node to its concrete function
for the period's age (`resolve_node` / `resolve_tree`) and fingerprints the closure
(`node_signature` / `tree_signature`), so periods that resolve to the same program
share a single compiled `Q_and_F` and never false-share across different policies.

The tree helpers recurse into nested mappings — pylcm's processed transitions are
`{target_regime: {transition_name: fn}}` — emitting sorted `(path, signature)` pairs
so a marked node nested under one key cannot collide with one under another.
"""

from collections.abc import Callable, Hashable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Final, cast

from _lcm.grids.continuous import ContinuousGrid
from _lcm.typing import EconFunction
from lcm.exceptions import RegimeInitializationError
from lcm.transition import AgeSpecializedFunction, AgeSpecializedGrid


class _Invariant:
    """Singleton signature for nodes whose closure does not vary with age."""

    __slots__ = ()

    def __repr__(self) -> str:
        return "INVARIANT"


INVARIANT: Final[Hashable] = _Invariant()


@dataclass(frozen=True)
class _SpecializedEconFunction:
    """A per-age-specialized node after `_process_regime_core` has processed it.

    Carries the same closure identity as the user's `AgeSpecializedFunction`, but its
    `build` returns a fully processed `EconFunction` (params already renamed to
    qnames) rather than a user function, so the per-period builders can resolve
    it directly into the DAG.
    """

    build: Callable[[float], EconFunction]
    """Return the processed `EconFunction` for a given age."""

    signature: Callable[[float], Hashable]
    """Return a hashable identity of the age's closure; the dedup key."""

    def __call__(self, *args: Any, **kwargs: Any) -> Any:  # noqa: ANN401, ARG002
        # Satisfies the `EconFunction` protocol (so the marker rides in an
        # `EconFunctionsMapping` past the beartype claw) while making it a loud
        # error if a builder feeds the marker to the DAG without resolving it.
        msg = (
            "_SpecializedEconFunction is a build-time marker and must be resolved "
            "to a concrete function via build(age) before it is called."
        )
        raise TypeError(msg)


# Node classes whose closure is bound per age: the user-facing marker and its
# processed counterpart. Both expose `build(age)` and `signature(age)`.
_SPECIALIZED_TYPES = (AgeSpecializedFunction, _SpecializedEconFunction)


def resolve_node(node: object, age: float) -> object:
    """Return the concrete function for `age`, or the node if age-invariant."""
    if isinstance(node, _SPECIALIZED_TYPES):
        return node.build(age)
    return node


def node_signature(node: object, age: float) -> Hashable:
    """Fingerprint `node`'s closure at `age`.

    `INVARIANT` for a plain callable; `node.signature(age)` for a specialized node.
    """
    if isinstance(node, _SPECIALIZED_TYPES):
        return node.signature(age)
    return INVARIANT


def resolve_specialized_nodes(
    mapping: Mapping[str, object], age: float
) -> Mapping[str, object]:
    """Resolve every `_SpecializedEconFunction` in a flat mapping at `age`.

    Return the input mapping unchanged when it holds no specialized node, so an
    age-invariant model builds byte-identically to one with no per-age wiring.
    """
    if not any(isinstance(node, _SpecializedEconFunction) for node in mapping.values()):
        return mapping
    return MappingProxyType(
        {name: resolve_node(node, age) for name, node in mapping.items()}
    )


def resolve_tree(tree: Mapping[str, object], age: float) -> Mapping[str, object]:
    """Resolve every `AgeSpecializedFunction` leaf of a (possibly nested) mapping at `age`."""
    resolved: dict[str, object] = {}
    for key, value in tree.items():
        if isinstance(value, Mapping):
            # Nested node trees ({target_regime: {name: fn}}) are always str-keyed.
            resolved[key] = resolve_tree(cast("Mapping[str, object]", value), age)
        else:
            resolved[key] = resolve_node(value, age)
    return resolved


def tree_signature(tree: Mapping[str, object], age: float) -> Hashable:
    """Fingerprint a (possibly nested) mapping of nodes at `age`.

    Recurse into `Mapping` values and emit sorted `(path, signature)` pairs, so a
    marked node nested under one key cannot collide with one under another.
    """
    pairs: list[tuple[str, Hashable]] = []
    for key in sorted(tree):
        value = tree[key]
        if isinstance(value, Mapping):
            nested = cast("Mapping[str, object]", value)
            pairs.append((key, tree_signature(nested, age)))
        else:
            pairs.append((key, node_signature(value, age)))
    return tuple(pairs)


# --------------------------------------------------------------------------- #
# Age-specialized GRIDS (shape-invariant continuous-state grids)              #
# --------------------------------------------------------------------------- #
def has_age_specialized_grid(states: Mapping[str, object]) -> bool:
    """Whether any state in the mapping is an `AgeSpecializedGrid` marker."""
    return any(isinstance(spec, AgeSpecializedGrid) for spec in states.values())


def age_specialized_grid_names(states: Mapping[str, object]) -> frozenset[str]:
    """Names of the states that are `AgeSpecializedGrid` markers."""
    return frozenset(
        name for name, spec in states.items() if isinstance(spec, AgeSpecializedGrid)
    )


def resolve_grid(spec: object, age: float) -> object:
    """Resolve an `AgeSpecializedGrid` to its concrete grid at `age`; else identity."""
    if isinstance(spec, AgeSpecializedGrid):
        return spec.build(age)
    return spec


def resolve_state_grids(states: Mapping[str, object], age: float) -> Mapping[str, object]:
    """Resolve every `AgeSpecializedGrid` in a states mapping at `age`.

    Returns the input unchanged when it holds no age-specialized grid, so an
    age-invariant model builds byte-identically to one with no per-age grids.
    """
    if not has_age_specialized_grid(states):
        return states
    return MappingProxyType(
        {name: resolve_grid(spec, age) for name, spec in states.items()}
    )


def grid_signature(spec: object, age: float) -> Hashable:
    """Fingerprint a state grid at `age`.

    `INVARIANT` for a plain grid; the marker's `signature(age)` for an
    `AgeSpecializedGrid`.
    """
    if isinstance(spec, AgeSpecializedGrid):
        return spec.signature(age)
    return INVARIANT


def state_grids_signature(states: Mapping[str, object], age: float) -> Hashable:
    """Fingerprint the age-varying grids in a states mapping at `age`.

    Only `AgeSpecializedGrid` states contribute a non-`INVARIANT` fingerprint, so
    two periods share a program iff their age-varying grids resolve identically.
    """
    pairs: list[tuple[str, Hashable]] = []
    for name in sorted(states):
        sig = grid_signature(states[name], age)
        if sig is not INVARIANT:
            pairs.append((name, sig))
    return tuple(pairs)


def validate_age_specialized_grids(
    states: Mapping[str, object], ages: object
) -> None:
    """Validate the shape-invariance contract of every `AgeSpecializedGrid`.

    Across every period's age, each marker's `build(age)` must return the **same
    grid class** and the **same `n_points`**, and a `ContinuousGrid`. `ages` is an
    `AgeGrid` (imported loosely to avoid a cycle); its `period_to_age` /
    `n_periods` are used.

    Raises:
        RegimeInitializationError: if the contract is violated.
    """
    period_to_age = ages.period_to_age
    n_periods = ages.n_periods
    errors: list[str] = []
    for name, spec in states.items():
        if not isinstance(spec, AgeSpecializedGrid):
            continue
        cls0 = npoints0 = None
        for period in range(n_periods):
            grid = spec.build(period_to_age(period))
            if not isinstance(grid, ContinuousGrid):
                errors.append(
                    f"AgeSpecializedGrid '{name}' build(age) must return a "
                    f"ContinuousGrid; got {type(grid).__name__} at period {period}."
                )
                break
            npoints = int(grid.n_points)
            if cls0 is None:
                cls0, npoints0 = type(grid), npoints
            elif type(grid) is not cls0 or npoints != npoints0:
                errors.append(
                    f"AgeSpecializedGrid '{name}' is not shape-invariant: period 0 "
                    f"gives {cls0.__name__}(n_points={npoints0}) but period {period} "
                    f"gives {type(grid).__name__}(n_points={npoints}). Age-varying "
                    f"grids must keep the same class and n_points at every age."
                )
                break
    if errors:
        raise RegimeInitializationError("\n".join(errors))
