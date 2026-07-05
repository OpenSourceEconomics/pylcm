"""Resolve and fingerprint per-age-specialized DAG nodes.

`AgeSpecialized` marks a function whose closure is bound per age at build time.
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

from _lcm.typing import EconFunction
from lcm.transition import AgeSpecialized


class _Invariant:
    """Singleton signature for nodes whose closure does not vary with age."""

    __slots__ = ()

    def __repr__(self) -> str:
        return "INVARIANT"


INVARIANT: Final[Hashable] = _Invariant()


@dataclass(frozen=True)
class _SpecializedEconFunction:
    """A per-age-specialized node after `_process_regime_core` has processed it.

    Carries the same closure identity as the user's `AgeSpecialized`, but its
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
_SPECIALIZED_TYPES = (AgeSpecialized, _SpecializedEconFunction)


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
    """Resolve every `AgeSpecialized` leaf of a (possibly nested) mapping at `age`."""
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
