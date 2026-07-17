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

from collections.abc import Callable, Hashable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Final, cast

import numpy as np

from _lcm.grids.continuous import ContinuousGrid
from _lcm.typing import EconFunction
from lcm.ages import AgeGrid
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
    """Resolve every `AgeSpecializedFunction` leaf of a (possibly nested) mapping.

    Resolution happens at `age`.
    """
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


def resolve_state_grids(
    states: Mapping[str, object], age: float
) -> Mapping[str, object]:
    """Resolve every `AgeSpecializedGrid` in a states mapping at `age`.

    Returns the input unchanged when it holds no age-specialized grid, so an
    age-invariant model builds byte-identically to one with no per-age grids.
    """
    if not has_age_specialized_grid(states):
        return states
    return MappingProxyType(
        {name: resolve_grid(spec, age) for name, spec in states.items()}
    )


# NOTE: there is deliberately no `grid_signature` / `state_grids_signature` here. Grids
# are deduplicated on their *resolved nodes* (`processing._grid_identity`), not on the
# user's `signature(age)` — a grid, unlike a closure, can be asked what it actually is,
# so nothing hand-written is load-bearing for grid correctness. The two helpers that
# fingerprinted grids via `signature(age)` were never called and were removed (round-3
# re-review F3) rather than left to imply a rule the code does not follow.


class _GridTraitsError(Exception):
    """A resolved grid is internally inconsistent (message is user-facing)."""


@dataclass(frozen=True)
class _GridTraits:
    """What must not change across a grid's active ages.

    `shape`/`dtype` come from the resolved `to_jax()` array and are `None` only for
    runtime-supplied grids, whose nodes do not exist at build time.

    `dtype` holds the exact `np.dtype` object, not `dtype.str`, which is not injective
    over JAX's extended floating types and would let a dtype change past the validator
    (audit round-4 F1). `weak_type` is JAX array metadata that `np.asarray` drops, yet
    it steers promotion in the shared trace (audit round-5 hardening note). Both mirror
    `processing._node_fingerprint`, so the validator and the cache key agree on what
    identity means.
    """

    cls: type
    batch_size: int
    pass_points_at_runtime: bool
    n_points: int
    shape: tuple[int, ...] | None
    dtype: np.dtype[Any] | None
    weak_type: bool | None


def _grid_traits(grid: ContinuousGrid) -> _GridTraits:
    """Resolve the invariants of one concrete grid; raise if self-inconsistent."""
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
    # of the `Grid` base contract, but `to_jax()` is (re-review F2).
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


def validate_age_specialized_grids(
    states: Mapping[str, object],
    ages: AgeGrid,
    *,
    active_periods: Sequence[int] | None = None,
) -> None:
    """Validate the shape-invariance contract of every `AgeSpecializedGrid`.

    Across the owning regime's active ages, each marker's `build(age)` must return a
    `ContinuousGrid` of the **same class**, with the same **`batch_size`**, the same
    **points mode** (concrete vs supplied-at-runtime), and — for concrete grids — the
    same resolved **node-array shape and dtype**. Only a grid's bounds or node *values*
    may vary with age. `ages` is an `AgeGrid` (imported loosely to avoid a cycle); its
    `period_to_age` / `n_periods` are used.

    Class and `n_points` alone are not enough (audit F2): a `batch_size` change silently
    alters the execution layout, and a grid whose points are concrete at one age but
    supplied at runtime at another passes a class/`n_points` check yet blows up later,
    deep in period-axis construction, with an error about the wrong abstraction level.

    Nor is *declared* `n_points` enough (round-3 re-review F2). The invariant the solver
    actually needs is the compiled input signature: `_compile_all_functions` lowers one
    shared kernel against the **representative** state axis, and the solve then feeds it
    each period's axis, so a differing shape *or dtype* is rejected by the compiled
    executable. `n_points` is not part of the `Grid` base contract (only `to_jax()` is),
    so `getattr(grid, "n_points", 0)` silently agreed at 0 for two custom grids of
    different actual length. Concrete grids are therefore validated on their **resolved
    `to_jax()` array** — the same source of truth `_grid_identity` keys on — and any
    declared `n_points` must agree with it. Runtime grids cannot resolve nodes at build
    time, so only their declared `n_points` is checked here; their runtime points are a
    parameter-boundary concern.

    `active_periods` restricts the check to the ages where the owning regime is active;
    a builder may be deliberately undefined (raise) outside them, and building it there
    would turn a valid terminal-only/age-limited grid into a construction failure (audit
    F2). It defaults to the full horizon so the function stays back-compatible when
    called without a regime context.

    Raises:
        RegimeInitializationError: if the contract is violated.
    """
    period_to_age = ages.period_to_age
    # Only the owning regime's ACTIVE ages (audit F2); full horizon iff not given.
    active = range(ages.n_periods) if active_periods is None else active_periods
    errors: list[str] = []
    for name, spec in states.items():
        if not isinstance(spec, AgeSpecializedGrid):
            continue
        # The first active age's traits; every later active age must match them.
        first: _GridTraits | None = None
        first_period: int | None = None
        for active_period in active:
            grid = spec.build(period_to_age(active_period))
            if not isinstance(grid, ContinuousGrid):
                errors.append(
                    f"AgeSpecializedGrid '{name}' build(age) must return a "
                    f"ContinuousGrid; got {type(grid).__name__} at period "
                    f"{active_period}."
                )
                break
            try:
                traits = _grid_traits(grid)
            except _GridTraitsError as error:
                errors.append(
                    f"AgeSpecializedGrid '{name}' at period {active_period}: {error}"
                )
                break
            if first is None:
                first, first_period = traits, active_period
                continue
            if first == traits:
                continue
            errors.append(
                f"AgeSpecializedGrid '{name}' is not shape-invariant: "
                f"{_describe_trait_mismatch(first, traits)} The first active age is "
                f"period {first_period}, the offending one is period {active_period}. "
                f"Age-varying grids must keep the same class, batch_size, points mode "
                f"and resolved node shape/dtype at every active age; only their bounds "
                f"or node values may vary."
            )
            break
    if errors:
        raise RegimeInitializationError("\n".join(errors))
