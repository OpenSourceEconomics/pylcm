"""Process user-provided params into internal params.

`process_params` resolves user-supplied parameters against the model's
template, then runs a boundary-cast pass that normalises every numeric
leaf to a canonical pylcm dtype:

- Python `bool` (and `np.bool_` arrays) cast to `jnp.bool_`.
- Python `int` and typed integer arrays cast to `jnp.int32`. Out-of-
  range values surface as `ValueError`.
- Python `float` and typed float arrays cast to `canonical_float_dtype()`.
  Down-cast overflow surfaces as `OverflowError`.
- `UserMappingLeaf` / `UserSequenceLeaf` containers (covering both the
  user-input variant and the canonical narrow variant) recurse, always
  emitting a canonical `MappingLeaf` / `SequenceLeaf`.

The pass runs as the *last* step over `flat_params` — `pd.Series`
leaves are reshaped to JAX arrays via `convert_series_in_params`
beforehand, so by the time the cast walks the tree, every numeric leaf
is either a JAX array, a numpy array, or a Python scalar.

Anything else (`pd.Series` (defensive), strings, complex/object arrays,
custom objects) raises `InvalidParamsError` with the offending leaf's
qualified name.
"""

from collections.abc import Mapping
from types import MappingProxyType
from typing import Any, cast

import jax.numpy as jnp
import numpy as np
import pandas as pd
from dags.tree import QNAME_DELIMITER, qname_from_tree_path, tree_path_from_qname
from jax import Array

from _lcm.dtypes import CanonicalArrayWriter, safe_to_float_dtype, safe_to_int_dtype
from _lcm.engine import Regime
from _lcm.params.edges import EDGES, user_path
from _lcm.params.mapping_leaf import MappingLeaf, UserMappingLeaf
from _lcm.params.sequence_leaf import SequenceLeaf, UserSequenceLeaf
from _lcm.typing import (
    EdgeParamsTemplate,
    FlatEdgeParams,
    FlatParams,
    FlatRegimeParams,
    ParamsTemplate,
    QualifiedName,
    RegimeName,
    RegimeParamsTemplateNode,
)
from _lcm.utils.containers import ensure_containers_are_immutable
from _lcm.utils.error_messages import path_segment_name_errors
from _lcm.utils.namespace import ParamsQnameDepth, flatten_regime_namespace
from lcm.exceptions import InvalidNameError, InvalidParamsError
from lcm.typing import FunctionName, ParameterName, ReferenceName, UserParams


def process_params(
    *,
    params: UserParams,
    params_template: ParamsTemplate,
) -> FlatParams:
    """Process user-provided params into internal params.

    Users can provide parameters at exactly one of these levels:

    - Model level: `{"arg_0": 0.0}` — propagates to all functions needing arg_0
    - Regime level: `{"regime_0": {"arg_0": 0.0}}` — propagates within regime_0
    - Function level: `{"regime_0": {"func": {"arg_0": 0.0}}}` — direct
      specification; for per-target transition params this broadcasts over
      the target regimes
    - Target-regime level —
      `{"regime_0": {"target_regime_0": {"func": {"arg_0": 0.0}}}}` is the
      target-regime-specific value for a per-target transition function

    Parameters of the callables `Model(edges=...)` declares sit at their
    declaration path under `{"edges": {source: ...}}`; `find_param_candidates`
    lists their levels.

    The output always matches the params_template skeleton. Every numeric
    leaf — Python `bool` / `int` / `float`, typed JAX or numpy arrays, and
    numerics inside `UserMappingLeaf` / `UserSequenceLeaf` (or their
    canonical narrow subclasses) — is cast to the canonical pylcm dtype
    so the AOT signature is stable across calls.

    Callers that pass `pd.Series` leaves should orchestrate the steps
    themselves: `broadcast_to_template` (resolve), `convert_series_in_params`
    (multi-index reshape), then `cast_params_to_canonical_dtypes`. The
    one-shot `process_params` raises on `pd.Series` because the dtype
    cast does not know how to reshape multi-index data.

    Args:
        params: User-provided parameters dictionary.
        params_template: Template from `model.get_params_template()`.

    Returns:
        Immutable mapping with the same structure as params_template.

    Raises:
        InvalidParamsError: If params contains unexpected keys, type
            mismatches, or unsupported leaf types.
        InvalidNameError: If the same parameter is specified at multiple levels.
        ValueError: If a typed integer leaf carries a value outside the
            int32 range; the message names the offending parameter qname.
        OverflowError: If a typed float leaf would saturate to `±inf` on
            down-cast to `float32`; the message names the offending qname.

    """
    internal = broadcast_to_template(
        params=params, template=params_template, required=True
    )
    return cast_params_to_canonical_dtypes(internal)


def broadcast_to_template(
    *,
    params: Mapping,
    template: Mapping[str, Mapping],
    required: bool = True,
    already_consumed: frozenset[str] = frozenset(),
) -> FlatParams:
    """Broadcast user params to template shape via most-to-least-specific resolution.

    For each template qname, search for a matching user value at:

    1. Exact match: `regime__function__param`, or for per-target transition
       params `regime__target__function__param`
    2. Coarse function level (per-target qnames only):
       `regime__function__param` — one value broadcasts over the targets
    3. Regime level: `regime__param`
    4. Model level: `param`

    Returns the resolved structure with leaves left as the user supplied
    them; dtype canonicalisation is a separate step
    (`cast_params_to_canonical_dtypes`).

    Args:
        params: User-provided values at any nesting depth.
        template: Target structure defining all valid keys.
        required: If True, raise when any template key has no match.
        already_consumed: Flat keys an earlier stage resolved and acted on, so
            that a value which pinned something no longer in the template is not
            reported as unknown. A process law bound into its grid is the case:
            binding removes its slot, and a broadcast that bound it may serve
            other slots too, so the key stays in the params it is passed here.

    Returns:
        Immutable mapping from regime name to mapping of `func__param`
        keys to resolved values. All regime keys from the template are
        present (possibly with empty inner mappings when `required` is
        False).

    Raises:
        InvalidParamsError: On missing required keys or unknown user keys.
        InvalidNameError: On ambiguous multi-level specification.

    """
    template_flat = flatten_regime_namespace(template)
    params_flat = flatten_regime_namespace(params)

    # User values, unvalidated until `cast_params_to_canonical_dtypes`.
    result: dict[RegimeName, dict[str, object]] = {
        name: {} for name in template if name != EDGES
    }
    edge_result: dict[RegimeName, dict[str, object]] = {
        source: {} for source in template.get(EDGES, {})
    }
    used_keys: set[str] = set()
    missing: list[str] = []

    for qname in template_flat:
        candidates = find_param_candidates(qname=qname, params_flat=params_flat)

        if len(candidates) > 1:
            raise InvalidNameError(
                f"Ambiguous parameter specification for {qname!r}. "
                f"Found values at: {candidates}"
            )

        path = tree_path_from_qname(qname)
        if candidates:
            chosen = candidates[0]
            regime, remainder = path[0], qname_from_tree_path(path[1:])
            if regime == EDGES:
                source, slot = path[1], qname_from_tree_path(path[2:])
                edge_result[source][slot] = params_flat[chosen]
            else:
                result[regime][remainder] = params_flat[chosen]
            used_keys.add(chosen)
        elif required:
            missing.append(user_path(path=path) if path[0] == EDGES else qname)

    unknown = set(params_flat) - used_keys - already_consumed
    if missing or unknown:
        messages = []
        if missing:
            messages.append(f"Missing required parameter(s): {missing}")
        if unknown:
            messages.append(f"Unknown keys: {sorted(unknown)}")
            messages.extend(
                _edge_slot_hints(unknown=unknown, template_flat=template_flat)
            )
        raise InvalidParamsError(" ".join(messages))

    frozen: dict[RegimeName, MappingProxyType[str, object]] = {
        k: MappingProxyType(v) for k, v in result.items()
    }
    if EDGES in template:
        frozen[EDGES] = MappingProxyType(
            {source: MappingProxyType(v) for source, v in edge_result.items()}
        )
    return cast("FlatParams", MappingProxyType(frozen))


def _edge_slot_hints(
    *, unknown: set[str], template_flat: Mapping[str, str]
) -> list[str]:
    """Point each unknown key written under a source regime at its edge slots.

    A value written under a regime never reaches a callable `Model(edges=...)`
    declares. When an unknown key under a regime ends in an argument name that
    an edge slot of that source reads, the slot's own path is what was meant.

    Args:
        unknown: The user's flat keys no template slot consumed.
        template_flat: The flattened template.

    Returns:
        One message per unknown key with matching edge slots, in key order.

    """
    slots: dict[tuple[RegimeName, ParameterName], list[str]] = {}
    for qname in template_flat:
        path = tree_path_from_qname(qname)
        if path[0] == EDGES:
            slots.setdefault((path[1], path[-1]), []).append(user_path(path=path))
    hints = []
    for key in sorted(unknown):
        path = tree_path_from_qname(key)
        matches = slots.get((path[0], path[-1]), []) if len(path) > 1 else []
        if matches:
            hints.append(
                f"{key!r} is written under the regime {path[0]!r}, which feeds "
                "only that regime's own functions. A parameter of a callable "
                "declared in `Model(edges=...)` lives at its declaration path "
                f"({', '.join(matches)}), once for every callable of the source "
                f"at {user_path(path=(EDGES, path[0], path[-1]))}, or at the "
                "model level."
            )
    return hints


def materialize_granular_transition_params(
    *,
    flat_params: FlatParams,
    expansions: Mapping[str, Mapping[str, tuple[str, ...]]],
) -> FlatParams:
    """Expand coarse transition-law params to their per-target qnames.

    Canonical flat params always key transition-law params per target
    (`<target>__<law>__<param>`), matching the engine's target-prefixed
    function qnames. A coarse user spelling resolves against the coarse
    template key first; this pass replaces each such entry with one entry
    per granular prefix, every target sharing the same leaf object —
    bit-identical arithmetic and no per-target copies.

    Args:
        flat_params: Template-shaped output of `broadcast_to_template`
            (after dtype canonicalisation).
        expansions: Per regime, mapping of coarse law key to its granular
            qname prefixes (`Regime.granular_param_expansions`).

    Returns:
        New immutable mapping with coarse transition-law entries replaced
        by their granular spellings.

    """
    result: dict[str, MappingProxyType[str, object]] = {}
    for regime_name, leaves in flat_params.items():
        if regime_name == EDGES:
            result[EDGES] = leaves
            continue
        regime_expansions = expansions.get(regime_name, {})
        materialized: dict[str, object] = {}
        for param_qname, value in leaves.items():
            path = tree_path_from_qname(param_qname)
            prefixes = regime_expansions.get(path[0])
            if len(path) == ParamsQnameDepth.REGIME__FUNC__PARAM - 1 and prefixes:
                for prefix in prefixes:
                    materialized[qname_from_tree_path((prefix, path[1]))] = value
            else:
                materialized[param_qname] = value
        result[regime_name] = MappingProxyType(materialized)
    return cast("FlatParams", MappingProxyType(result))


# keyword-only-exempt: primary-argument=flat_params
def cast_params_to_canonical_dtypes(
    flat_params: FlatParams, *, array_writer: CanonicalArrayWriter | None = None
) -> FlatParams:
    """Cast every numeric leaf of `flat_params` to its canonical pylcm dtype.

    Runs as a separate pass so the orchestrator can interpose
    `convert_series_in_params` between broadcast and cast — by the time
    this pass walks the tree, no `pd.Series` leaf should remain.

    Args:
        flat_params: Output of `broadcast_to_template`, optionally
            after `convert_series_in_params`.
        array_writer: Optional owner admitting each validated canonical leaf before
            its device upload; traversal and top-level identity memo stay unchanged.

    Returns:
        New immutable mapping with every leaf cast to its canonical dtype.

    """
    # One cast per distinct input object: a value broadcast into several
    # slots (e.g. a coarse value resolved into per-target template slots)
    # stays one shared leaf, so downstream consumers can deduplicate by
    # identity and large array leaves are not copied per slot.
    memo: dict[int, Any] = {}

    return MappingProxyType(
        {
            regime: (
                MappingProxyType(
                    {
                        source: _cast_flat_leaves(
                            leaves=source_leaves,
                            prefix=f"{EDGES}{QNAME_DELIMITER}{source}",
                            memo=memo,
                            array_writer=array_writer,
                        )
                        for source, source_leaves in cast(
                            "FlatEdgeParams", leaves
                        ).items()
                    }
                )
                if regime == EDGES
                else _cast_flat_leaves(
                    leaves=leaves,
                    prefix=regime,
                    memo=memo,
                    array_writer=array_writer,
                )
            )
            for regime, leaves in flat_params.items()
        }
    )


def _cast_flat_leaves(
    *,
    # User values, cast here; unvalidated until then.
    leaves: Mapping[str, object],
    prefix: str,
    # Shared with `_cast_shared`, whose memo holds the canonical leaves.
    memo: dict[int, Any],
    array_writer: CanonicalArrayWriter | None,
) -> FlatRegimeParams:
    """Cast one flat mapping's leaves, naming each by `prefix` and its key."""
    return MappingProxyType(
        {
            param_qname: _cast_shared(
                value=value,
                name=f"{prefix}{QNAME_DELIMITER}{param_qname}",
                memo=memo,
                array_writer=array_writer,
            )
            for param_qname, value in leaves.items()
        }
    )


def _cast_shared(
    *,
    value: Any,
    name: str,
    memo: dict[int, Any],
    array_writer: CanonicalArrayWriter | None,
) -> Any:
    """Cast `value` once per distinct input object, memoized by identity in `memo`."""
    key = id(value)
    if key not in memo:
        memo[key] = _cast_leaves_to_canonical_dtype(
            value=value, name=name, array_writer=array_writer
        )
    return memo[key]


def _cast_leaves_to_canonical_dtype(  # noqa: C901, PLR0911
    *,
    value: Any,
    name: str,
    array_writer: CanonicalArrayWriter | None,
) -> Any:
    """Cast a single params leaf to its canonical pylcm dtype.

    Strict whitelist — every code path either casts or raises.

    Casts:

    - `UserMappingLeaf` / `UserSequenceLeaf` (covers both wide user and
      canonical narrow variants): recurse on contents, always emit the
      canonical `MappingLeaf` / `SequenceLeaf`.
    - Python `bool`: `jnp.bool_(value)` (must come before `int` —
      `True` is a Python `int` subclass).
    - Python `int`: `safe_to_int_dtype(value)` → `jnp.int32`.
    - Python `float`: `safe_to_float_dtype(value)` → canonical float.
    - JAX or numpy array, dispatch on `dtype.kind`:
      - `"b"` (bool) → `jnp.asarray(..., dtype=jnp.bool_)`.
      - `"i"` / `"u"` (signed/unsigned int) → `safe_to_int_dtype`.
      - `"f"` (float) → `safe_to_float_dtype`.

    Raises `InvalidParamsError` for:

    - `pd.Series`: defensive — the orchestrator must run
      `convert_series_in_params` before this pass.
    - Array dtypes other than bool/int/float (e.g. complex, object,
      string).
    - Anything else (`str`, `None`, `dict`, lists, custom objects).

    """
    # `UserMappingLeaf` covers both user (wide) and canonical (`MappingLeaf`)
    # variants — recursing always emits a canonical `MappingLeaf`.
    if isinstance(value, UserMappingLeaf):
        return MappingLeaf(
            {
                k: _cast_leaves_to_canonical_dtype(
                    value=v, name=f"{name}.{k}", array_writer=array_writer
                )
                for k, v in value.data.items()
            }
        )
    if isinstance(value, UserSequenceLeaf):
        return SequenceLeaf(
            [
                _cast_leaves_to_canonical_dtype(
                    value=v, name=f"{name}[{i}]", array_writer=array_writer
                )
                for i, v in enumerate(value.data)
            ]
        )
    if isinstance(value, pd.Series):
        msg = (
            f"{name!r}: pd.Series leaf reached the dtype cast — "
            f"`convert_series_in_params` must run between "
            f"`broadcast_to_template` and `cast_params_to_canonical_dtypes`."
        )
        raise InvalidParamsError(msg)
    # `bool` before `int` — `True` is a Python `int` subclass.
    if isinstance(value, bool):
        if array_writer is not None:
            return array_writer(
                value=np.asarray(value), dtype=np.dtype(np.bool_), name=name
            )
        return jnp.bool_(value)
    if isinstance(value, int):
        return safe_to_int_dtype(value=value, name=name, array_writer=array_writer)
    if isinstance(value, float):
        return safe_to_float_dtype(value=value, name=name, array_writer=array_writer)
    if isinstance(value, (Array, np.ndarray)):
        kind = value.dtype.kind
        if kind == "b":
            if array_writer is not None:
                return array_writer(value=value, dtype=np.dtype(np.bool_), name=name)
            return jnp.asarray(value, dtype=jnp.bool_)
        if kind in ("i", "u"):
            return safe_to_int_dtype(value=value, name=name, array_writer=array_writer)
        if kind == "f":
            return safe_to_float_dtype(
                value=value, name=name, array_writer=array_writer
            )
        msg = (
            f"{name!r}: array dtype {value.dtype} not supported "
            f"(expected bool / int / float)."
        )
        raise InvalidParamsError(msg)
    msg = (
        f"{name!r}: unsupported leaf type {type(value).__name__} "
        f"(expected bool / int / float / numpy or JAX array / "
        f"UserMappingLeaf / UserSequenceLeaf)."
    )
    raise InvalidParamsError(msg)


def find_param_candidates(
    *,
    qname: QualifiedName,
    params_flat: Mapping[str, object],
) -> list[str]:
    """Find candidate matches for a template qname, most to least specific.

    This is the project's one resolution rule. Every consumer that asks where a
    user wrote a value calls it — the params template and the process-law binder
    alike — so a spelling one accepts cannot be a spelling another refuses.

    Resolution levels of a regime function's slot:

    1. Exact match: `regime__function__param`, or for per-target transition
       params `regime__target__function__param`
    2. Coarse function level (per-target qnames only):
       `regime__function__param` — one value broadcasts over the targets
    3. Regime level: `regime__param`
    4. Model level: `param`

    Resolution levels of an edge slot, `edges__source__<declaration path>__param`:

    1. Exact match
    2. Source level: `edges__source__param` — one value for every callable the
       source's edges declare
    3. Model level: `param`

    A regime-level value never fills an edge slot, and the coarse function level
    does not apply to one: its second segment is a source regime, not a target.

    Args:
        qname: Qualified name of the template slot to fill.
        params_flat: Flattened user params, keyed by qualified name.

    Returns:
        List of the matching keys of `params_flat`, most specific first. More
        than one entry means the user wrote the value at several levels, which
        the caller reports as ambiguous.

    """
    tree_path = tree_path_from_qname(qname)
    param_name = tree_path[-1]
    candidates: list[str] = []

    if qname in params_flat:
        candidates.append(qname)

    if tree_path[0] == EDGES:
        source_level_qname = qname_from_tree_path((EDGES, tree_path[1], param_name))
        if source_level_qname != qname and source_level_qname in params_flat:
            candidates.append(source_level_qname)
        if param_name in params_flat:
            candidates.append(param_name)
        return candidates

    if len(tree_path) == ParamsQnameDepth.REGIME__TARGETREGIME__FUNC__PARAM:
        coarse_qname = qname_from_tree_path((tree_path[0], *tree_path[2:]))
        if coarse_qname in params_flat:
            candidates.append(coarse_qname)

    if len(tree_path) >= ParamsQnameDepth.REGIME__FUNC__PARAM:
        regime_level_qname = qname_from_tree_path((tree_path[0], param_name))
        if regime_level_qname in params_flat:
            candidates.append(regime_level_qname)

    if param_name in params_flat:
        candidates.append(param_name)

    return candidates


def create_params_template(
    regimes: MappingProxyType[RegimeName, Regime],
) -> ParamsTemplate:
    """Create params_template from internal regimes and validate name uniqueness.

    This function validates that regime names, function names, and argument names
    are disjoint sets to enable unambiguous parameter propagation, and that none
    of them is `edges`, the root of the edge namespace. The template holds one
    branch per regime and, when any source's edges declare a parameter, an
    `edges` branch keyed by source regime.

    Args:
        regimes: Immutable mapping of regime names to Regime
            instances.

    Returns:
        The parameter template.

    Raises:
        InvalidNameError: If names are not disjoint, contain the separator, or
            name the edge namespace.

    """
    template: dict[str, Any] = {}
    regime_names: set[RegimeName] = set(regimes)
    function_names: set[FunctionName] = set()
    arg_names = _edge_arg_names(regimes)

    for name, regime in regimes.items():
        # A regime no required problem solves reads no parameter.
        regime_template = (
            dict(regime.regime_params_template) if regime.active_periods else {}
        )
        template[name] = regime_template

        for key, val in regime_template.items():
            if not isinstance(val, (dict, Mapping)):
                raise InvalidNameError(
                    f"Parameter {key!r} in regime {name!r} must be nested under "
                    f"a function name, e.g., {{'function_name': {{'{key}': type}}}}"
                )
            if key in regime_names:
                # A target branch: per-target transition params nested under
                # the target regime's name.
                for func_key, func_val in val.items():
                    if not isinstance(func_val, Mapping):
                        raise InvalidNameError(
                            f"{key!r} in regime {name!r} is a regime name, so "
                            f"its entries must be per-target transition "
                            f"functions with nested params; {func_key!r} maps "
                            f"to a bare leaf."
                        )
                    function_names.add(func_key)
                    nested_roles = {
                        role
                        for role, role_val in func_val.items()
                        if isinstance(role_val, Mapping)
                    }
                    if nested_roles:
                        if set(func_val) != {"support", "probabilities"}:
                            raise InvalidNameError(
                                f"Joint transition kernel {func_key!r} in regime "
                                f"{name!r} must contain exactly 'support' and "
                                "'probabilities' role mappings."
                            )
                        for role, role_params in func_val.items():
                            arg_names |= _validated_arg_names(
                                func_name=f"{func_key}.{role}",
                                params=cast("Mapping", role_params),
                                regime_name=name,
                            )
                    else:
                        arg_names |= _validated_arg_names(
                            func_name=func_key, params=func_val, regime_name=name
                        )
            else:
                function_names.add(key)
                arg_names |= _validated_arg_names(
                    func_name=key, params=val, regime_name=name
                )

    _fail_if_template_names_invalid(
        regime_names=regime_names,
        function_names=function_names,
        arg_names=arg_names,
    )

    return cast(
        "ParamsTemplate",
        ensure_containers_are_immutable(template | _edges_branch(regimes)),
    )


def _edge_arg_names(regimes: Mapping[RegimeName, Regime]) -> set[ParameterName]:
    """Return the argument names every source's edge slots read."""
    return {
        path[-1]
        for regime in regimes.values()
        for path in _leaf_paths(regime.edge_params_template)
    }


def _edges_branch(
    regimes: Mapping[RegimeName, Regime],
) -> dict[str, dict[RegimeName, EdgeParamsTemplate]]:
    """Return the template's `edges` branch, or nothing when no source has a slot."""
    sources = {
        name: regime.edge_params_template
        for name, regime in regimes.items()
        if regime.edge_params_template
    }
    return {EDGES: sources} if sources else {}


def _leaf_paths(
    branch: Mapping[str, RegimeParamsTemplateNode],
) -> list[tuple[str, ...]]:
    """Return the key path of every leaf below a nested template branch."""
    return [
        (name, *path)
        for name, value in branch.items()
        for path in (_leaf_paths(value) if isinstance(value, Mapping) else [()])
    ]


def _validated_arg_names(
    *,
    func_name: FunctionName,
    params: Mapping,
    regime_name: RegimeName,
) -> set[str]:
    """Return a function entry's argument names, validating each leaf.

    Argument names must be valid parameter-path segments and map to bare leaves
    — a nested mapping at this depth means the user nested params one level too
    deep.
    """
    if errors := path_segment_name_errors(kind=f"{func_name!r} argument", names=params):
        raise InvalidNameError(errors[0])
    arg_names: set[ReferenceName] = set()
    for arg_name, leaf in params.items():
        if isinstance(leaf, Mapping):
            raise InvalidNameError(
                f"Parameter {arg_name!r} in regime {regime_name!r} is "
                f"nested too deeply."
            )
        arg_names.add(arg_name)
    return arg_names


def _fail_if_template_names_invalid(
    *,
    regime_names: set[RegimeName],
    function_names: set[FunctionName],
    arg_names: set[ReferenceName],
) -> None:
    """Validate the form and disjointness of template name sets.

    Regime and function names must be valid parameter-path segments, and
    regime names must be disjoint from both function and argument names so
    parameter propagation stays unambiguous. Function names CAN overlap with
    argument names across regimes — a function output in one regime may be a
    parameter in another (e.g. `labor_income` is a function in `working` but
    a param in `retired`).
    """
    for kind, names in (
        ("Regime", regime_names),
        ("Function", function_names),
        ("Argument", arg_names),
    ):
        if EDGES in names:
            raise InvalidNameError(
                f"{kind} name {EDGES!r} is reserved: `params[{EDGES!r}]` holds the "
                "parameters of the callables `Model(edges=...)` declares. Rename it."
            )

    if errors := [
        *path_segment_name_errors(kind="Regime", names=sorted(regime_names)),
        *path_segment_name_errors(kind="Function", names=sorted(function_names)),
    ]:
        raise InvalidNameError(errors[0])

    regime_func_overlap = regime_names & function_names
    if regime_func_overlap:
        raise InvalidNameError(
            f"Regime names and function names must be disjoint. "
            f"Overlap: {sorted(regime_func_overlap)}"
        )

    regime_arg_overlap = regime_names & arg_names
    if regime_arg_overlap:
        raise InvalidNameError(
            f"Regime names and argument names must be disjoint. "
            f"Overlap: {sorted(regime_arg_overlap)}"
        )


def get_flat_param_names(
    regime_params_template: Mapping[str, RegimeParamsTemplateNode],
) -> set[str]:
    """Get all flat parameter names from a regime params template.

    Converts nested template entries like `{"utility": {"risk_aversion": type}}`
    to flat names like `utility__risk_aversion`; per-target branches like
    `{"retired": {"next_wealth": {"exit_tax": type}}}` yield
    `retired__next_wealth__exit_tax`.

    """
    result: set[str] = set()
    for key, value in regime_params_template.items():
        _collect_flat_param_names(prefix=(key,), node=value, result=result)
    return result


def _collect_flat_param_names(
    *, prefix: tuple[str, ...], node: object, result: set[str]
) -> None:
    """Add the qualified name of every leaf under `node` to `result`."""
    if isinstance(node, Mapping):
        for name, child in node.items():
            _collect_flat_param_names(prefix=(*prefix, name), node=child, result=result)
    else:
        result.add(qname_from_tree_path(prefix))
