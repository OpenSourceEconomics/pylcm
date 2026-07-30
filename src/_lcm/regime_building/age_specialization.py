"""Per-age-specialized node resolution and grid-shape validation helpers.

`AgeSpecializedFunction` marks a function whose closure is bound per age at build time.
`resolve_node` returns a marked node's concrete function for a given age, and
`node_signature` / `tree_signature` fingerprint the closure so nodes that resolve to
the same program can share a compiled `Q_and_F`. These are used by the model-level
dependency analysis, which runs on the raw (un-normalized) user regimes.

The model-processing pipeline itself normalizes age specialization early
(`age_normalization.normalize_age_specialization`) and no longer resolves markers
lazily; this module also hosts the grid shape-invariance traits (`_grid_traits` and
`_GridTraits`) that the normalizer reuses to validate `AgeSpecializedGrid` families.
"""

from collections.abc import Hashable, Mapping
from dataclasses import dataclass
from typing import Any, Final, cast

import numpy as np

from _lcm.grids.continuous import ContinuousGrid
from lcm.transition import AgeSpecializedFunction
from lcm.typing import Float1D


class _Invariant:
    """Singleton signature for nodes whose closure does not vary with age."""

    __slots__ = ()

    def __repr__(self) -> str:
        return "INVARIANT"


INVARIANT: Final[Hashable] = _Invariant()


def resolve_node(node: object, age: float) -> object:
    """Return the concrete function for `age`, or the node if age-invariant."""
    if isinstance(node, AgeSpecializedFunction):
        return node.build(age)
    return node


def node_signature(node: object, age: float) -> Hashable:
    """Fingerprint `node`'s closure at `age`.

    `INVARIANT` for a plain callable; `node.signature(age)` for a specialized node.
    """
    if isinstance(node, AgeSpecializedFunction):
        return node.signature(age)
    return INVARIANT


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


class _GridTraitsError(Exception):
    """A resolved grid is internally inconsistent (message is user-facing)."""


@dataclass(frozen=True)
class _GridTraits:
    """What must not change across a grid's active ages.

    `shape`/`dtype` come from the resolved `to_jax()` array and are `None` only for
    runtime-supplied grids, whose nodes do not exist at build time.

    `dtype` holds the exact `np.dtype` object, not `dtype.str`, which is not injective
    over JAX's extended floating types and would let a dtype change past the validator.
    `weak_type` is JAX array metadata that `np.asarray` drops, yet it steers promotion
    in the shared trace, so a `weak_type` change across ages is rejected as a
    shape-invariance violation.
    """

    cls: type
    batch_size: int
    pass_points_at_runtime: bool
    n_points: int
    shape: tuple[int, ...] | None
    dtype: np.dtype[Any] | None
    weak_type: bool | None


def _grid_traits(grid: ContinuousGrid, *, nodes: Float1D | None = None) -> _GridTraits:
    """Resolve the invariants of one concrete grid; raise if self-inconsistent.

    Pass `nodes` when the caller already resolved `grid.to_jax()`, so it is not
    recomputed here.
    """
    runtime = bool(getattr(grid, "pass_points_at_runtime", False))
    declared = getattr(grid, "n_points", None)
    if runtime:
        if declared is None:
            msg = (
                "a grid whose points are supplied at runtime must declare n_points; "
                f"{type(grid).__name__} declares none, so its axis shape is unknown "
                "at build time."
            )
            raise _GridTraitsError(msg)
        return _GridTraits(
            cls=type(grid),
            batch_size=int(grid.batch_size),
            pass_points_at_runtime=True,
            n_points=int(declared),
            shape=None,
            dtype=None,
            weak_type=None,
        )
    # Concrete grid: the resolved array is the source of truth. `n_points` is not part
    # of the `Grid` base contract, but `to_jax()` is.
    if nodes is None:
        nodes = grid.to_jax()
    arr = np.asarray(nodes)
    if arr.ndim != 1:
        msg = (
            f"{type(grid).__name__}.to_jax() must return a 1-D array of nodes; got "
            f"shape {arr.shape}."
        )
        raise _GridTraitsError(msg)
    if declared is not None and int(declared) != arr.shape[0]:
        msg = (
            f"{type(grid).__name__} declares n_points={int(declared)} but its "
            f"to_jax() returns {arr.shape[0]} nodes."
        )
        raise _GridTraitsError(msg)
    return _GridTraits(
        cls=type(grid),
        batch_size=int(grid.batch_size),
        pass_points_at_runtime=False,
        n_points=arr.shape[0],
        shape=arr.shape,
        dtype=arr.dtype,
        weak_type=bool(getattr(nodes, "weak_type", False)),
    )


def _mode(traits: _GridTraits) -> str:
    return "at runtime" if traits.pass_points_at_runtime else "concretely"


# (field, label, renderer) per trait, in the order the message should prefer. Every
# field of `_GridTraits` appears exactly once, so a trait added to the dataclass without
# a row here is caught by `test_every_grid_trait_is_described`.
_TRAIT_DESCRIPTIONS: Final = (
    ("cls", "grid class", lambda t: t.cls.__name__),
    ("pass_points_at_runtime", "points supplied", _mode),
    ("batch_size", "batch_size", lambda t: t.batch_size),
    ("n_points", "n_points", lambda t: t.n_points),
    ("shape", "resolved node shape", lambda t: t.shape),
    ("dtype", "resolved node dtype", lambda t: t.dtype),
    ("weak_type", "resolved node weak_type", lambda t: t.weak_type),
)


def _describe_trait_mismatch(first: _GridTraits, other: _GridTraits) -> str:
    """One sentence naming the first trait that differs."""
    for field, label, render in _TRAIT_DESCRIPTIONS:
        if getattr(first, field) != getattr(other, field):
            return f"{label} {render(first)} -> {render(other)}."
    # Unreachable: the caller only calls this once two traits compare unequal, and
    # every field is covered above.
    msg = "grid traits differ but no described trait does"
    raise AssertionError(msg)
