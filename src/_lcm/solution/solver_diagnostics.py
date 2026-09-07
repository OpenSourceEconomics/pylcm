"""Solver-published numerical diagnostics.

A continuous-outer solve is only inference-grade when its numerical error is
*observable*: how far the interpolant strays from exact solves, how wide the
final refinement brackets are, how much population-relevant state mass sits at
bounds or branch ties. Solvers publish those observations as a
`SolverDiagnostics` on the auxiliary channel of their `KernelOutput`; the
engine treats the payload as opaque cargo (no solver-type switch), and
downstream release gates read it. A retained payload is described by a
model-verifiable artifact descriptor the solve generates from the payload
itself, so it persists with the result and reads back without a model.

The finite-grid solvers publish nothing (`None`); the continuous-outer solver
fills the fields it can measure. Every field is optional-by-shape rather than
optional-by-`None` inside one payload: a period that measures nothing simply
does not publish a payload at all.
"""

import dataclasses
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Literal

import jax
import jax.numpy as jnp
import numpy as np

from lcm.solver_api import (
    SOLVER_DIAGNOSTICS,
    ArtifactChannel,
    ArtifactDescriptor,
    LeafDescriptor,
    PersistencePolicy,
    _CanonicalArtifactTemplate,
    _same_exact_artifact_contract,
    _snapshot_artifact_template_once,
)
from lcm.typing import BoolND, FloatND, IntND

# How much diagnostic cargo a solve retains: "none" keeps no diagnostic
# arrays; "summary" keeps scalar maxima/quantiles/counts; "full" keeps
# state-array diagnostics (research/debugging).
type DiagnosticLevel = Literal["none", "summary", "full"]


@dataclass(frozen=True, kw_only=True)
class SolverDiagnostics:
    """Per-period numerical diagnostics a solver publishes with its solve.

    Shapes are the solver's choice per its declared `DiagnosticLevel`:
    state-shaped arrays at `"full"`, 0-d summaries at `"summary"`. Consumers
    must not assume a particular shape — only that larger is worse for error
    fields and that masks mark cells needing attention.
    """

    max_outer_interpolation_error: FloatND
    """Largest validated interpolant-vs-exact-solve gap as a dimensionless
    normalized validation ratio `|exact - interp| / (value_atol + value_rtol *
    scale)`, NOT a value-unit gap: a value at or below 1 is within the
    configured mesh tolerance."""

    max_outer_bracket_width: FloatND
    """Widest refinement bracket around a selected outer optimum."""

    outer_nodes_used: IntND
    """Number of exact outer candidate nodes the solve ended with."""

    outer_at_lower_bound: BoolND
    """Where the selected outer action sits at the domain's lower bound."""

    outer_at_upper_bound: BoolND
    """Where the selected outer action sits at the domain's upper bound."""

    keeper_adjuster_margin: FloatND
    """Winning-branch margin `|V_keeper - V_adjuster|` at the optimum."""

    best_second_best_margin: FloatND
    """Margin between the best and second-best outer candidates."""

    policy_fallback_mask: BoolND
    """Where a policy read fell back rather than reading the optimum."""

    unresolved_mask: BoolND
    """Where the solve could not certify its outer optimum to tolerance."""

    n_outer_all_invalid_cells: IntND
    """Count of state cells with no finite outer candidate on any mesh node
    (nor any sampled midpoint). Under NB-EGM these are cells where *adjusting*
    is infeasible everywhere and the keeper legitimately wins, so the count is
    reported rather than raised on; a release gate that knows the reachable
    support consults it to distinguish a benign infeasible padding cell from a
    reachable cell that should have resolved."""

    adjustment_probability: FloatND | None = None
    """Analytic per-cell adjuster-branch probability under a uniform observed
    fixed cost (`UniformObservedFixedCost`), or `None` under the
    deterministic maximum. An analytic moment: the moment engine reads it
    directly instead of estimating adjustment frequencies from draws."""


_DIAGNOSTICS_FIELDS: tuple[str, ...] = tuple(
    field.name for field in dataclasses.fields(SolverDiagnostics)
)
SOLVER_DIAGNOSTICS_TYPE_ID = (
    f"{SolverDiagnostics.__module__}.{SolverDiagnostics.__qualname__}"
)

# The NumPy dtype kinds each field may carry: error levels and margins are
# floating point, counts are integers, masks are Boolean.
_LEAF_DTYPE_KINDS: dict[str, frozenset[str]] = {
    "max_outer_interpolation_error": frozenset("f"),
    "max_outer_bracket_width": frozenset("f"),
    "outer_nodes_used": frozenset("iu"),
    "outer_at_lower_bound": frozenset("b"),
    "outer_at_upper_bound": frozenset("b"),
    "keeper_adjuster_margin": frozenset("f"),
    "best_second_best_margin": frozenset("f"),
    "policy_fallback_mask": frozenset("b"),
    "unresolved_mask": frozenset("b"),
    "n_outer_all_invalid_cells": frozenset("iu"),
    "adjustment_probability": frozenset("f"),
}
# Fields a payload may leave `None`; every other field is a numerical leaf.
_OPTIONAL_FIELDS = frozenset({"adjustment_probability"})
assert set(_LEAF_DTYPE_KINDS) == set(_DIAGNOSTICS_FIELDS)  # noqa: S101


def _flatten_diagnostics(
    diagnostics: SolverDiagnostics,
) -> tuple[tuple[Any, ...], None]:
    return tuple(getattr(diagnostics, name) for name in _DIAGNOSTICS_FIELDS), None


def _flatten_diagnostics_with_keys(
    diagnostics: SolverDiagnostics,
) -> tuple[tuple[tuple[jax.tree_util.GetAttrKey, Any], ...], None]:
    """Flatten with field-named keys so a leaf path reads `.unresolved_mask`."""
    return (
        tuple(
            (jax.tree_util.GetAttrKey(name), getattr(diagnostics, name))
            for name in _DIAGNOSTICS_FIELDS
        ),
        None,
    )


# keyword-only-exempt: library-callback=jax.tree_util.register_pytree_with_keys
def _unflatten_diagnostics(_aux: None, children: Iterable[Any]) -> SolverDiagnostics:
    diagnostics = object.__new__(SolverDiagnostics)
    for name, child in zip(_DIAGNOSTICS_FIELDS, children, strict=True):
        object.__setattr__(diagnostics, name, child)
    return diagnostics


jax.tree_util.register_pytree_with_keys(
    SolverDiagnostics,
    _flatten_diagnostics_with_keys,
    _unflatten_diagnostics,
    _flatten_diagnostics,
)


def diagnostics_template_from_descriptor(  # noqa: C901
    *, descriptor: ArtifactDescriptor, label: str
) -> SolverDiagnostics:
    """Rebuild the zero-valued payload a diagnostics descriptor describes.

    A result carries the descriptor of every diagnostics payload its solve
    retained. The model cannot re-derive it without solving, so a consumer
    admits it as solution-owned data after checking what a published payload
    must satisfy: the standard key on the diagnostic channel, one leaf per
    field of `SolverDiagnostics` with no unknown or duplicated field, every
    non-optional field present, and each leaf carrying the dtype kind its field
    is declared with. Shapes are the solver's choice and are taken as declared.

    Args:
        descriptor: The diagnostics descriptor a result presents.
        label: Where the descriptor sits, for error messages.

    Returns:
        A `SolverDiagnostics` of zeros with exactly the described leaves.

    Raises:
        TypeError: If the descriptor is not a diagnostics descriptor or a leaf
            has a type or shape no field could carry.
        ValueError: If a field is unknown, repeated, or missing.

    """
    if not _same_exact_artifact_contract(
        actual=descriptor.key, expected=SOLVER_DIAGNOSTICS
    ):
        raise TypeError(f"{label} is not described under the diagnostics key.")
    if (
        descriptor.channel is not ArtifactChannel.DIAGNOSTIC
        or descriptor.persistence is not PersistencePolicy.MODEL_VERIFIABLE
        or descriptor.payload_type_id != SOLVER_DIAGNOSTICS_TYPE_ID
        or descriptor.state_roles
        or descriptor.action_roles
        or descriptor.categorical_domains
        or descriptor.required_for
        or descriptor.required
    ):
        raise TypeError(
            f"{label} is not a model-verifiable diagnostic payload descriptor."
        )
    leaves: dict[str, LeafDescriptor] = {}
    for leaf in descriptor.leaf_descriptors:
        if (
            type(leaf) is not LeafDescriptor
            or len(leaf.path) != 1
            or not leaf.path[0].startswith("attribute:")
        ):
            raise TypeError(f"{label} has a leaf outside the payload's fields.")
        name = leaf.path[0].removeprefix("attribute:")
        if name not in _LEAF_DTYPE_KINDS:
            raise ValueError(f"{label} declares the unknown field {name!r}.")
        if name in leaves:
            raise ValueError(f"{label} declares the field {name!r} twice.")
        if type(leaf.shape) is not tuple or any(
            type(size) is not int or size < 0 for size in leaf.shape
        ):
            raise TypeError(f"{label} field {name!r} has an invalid shape.")
        try:
            kind = np.dtype(leaf.dtype).kind
        except TypeError as error:
            raise TypeError(f"{label} field {name!r} has an invalid dtype.") from error
        if kind not in _LEAF_DTYPE_KINDS[name]:
            raise TypeError(
                f"{label} field {name!r} declares dtype {leaf.dtype!r}, which the "
                "field cannot carry."
            )
        leaves[name] = leaf
    missing = tuple(
        name
        for name in _DIAGNOSTICS_FIELDS
        if name not in leaves and name not in _OPTIONAL_FIELDS
    )
    if missing:
        raise ValueError(f"{label} is missing the fields {missing!r}.")
    fields: dict[str, Any] = {
        name: (
            None
            if name not in leaves
            else _zero_leaf(leaf=leaves[name], name=name, label=label)
        )
        for name in _DIAGNOSTICS_FIELDS
    }
    return SolverDiagnostics(**fields)


def _zero_leaf(*, leaf: LeafDescriptor, name: str, label: str) -> jax.Array:
    """Return zeros of one described leaf, refusing a dtype the profile narrows."""
    zeros = jnp.zeros(leaf.shape, dtype=np.dtype(leaf.dtype))
    if np.dtype(zeros.dtype) != np.dtype(leaf.dtype):
        raise TypeError(
            f"{label} field {name!r} has dtype {leaf.dtype!r}, which the active JAX "
            f"configuration would materialize as {zeros.dtype!s}. Enable the "
            "matching JAX dtype configuration instead of narrowing the solution."
        )
    return zeros


def diagnostics_template_snapshot(
    *, descriptor: ArtifactDescriptor, label: str
) -> _CanonicalArtifactTemplate:
    """Return the reconstruction plan for the payload a descriptor describes."""
    template = diagnostics_template_from_descriptor(descriptor=descriptor, label=label)
    snapshot, _containers = _snapshot_artifact_template_once(
        template=template,
        payload_runtime_type=SolverDiagnostics,
    )
    return snapshot
