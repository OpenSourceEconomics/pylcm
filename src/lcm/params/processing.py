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

from lcm.dtypes import safe_to_float_dtype, safe_to_int_dtype
from lcm.exceptions import InvalidNameError, InvalidParamsError
from lcm.interfaces import Regime
from lcm.params.mapping_leaf import MappingLeaf, UserMappingLeaf
from lcm.params.sequence_leaf import SequenceLeaf, UserSequenceLeaf
from lcm.typing import (
    FlatParams,
    ParamsTemplate,
    RegimeName,
    RegimeParamsTemplate,
    UserParams,
)
from lcm.utils.containers import ensure_containers_are_immutable
from lcm.utils.namespace import flatten_regime_namespace

_NUM_PARTS_FUNCTION_PARAM = 3


def process_params(
    *,
    params: UserParams,
    params_template: ParamsTemplate,
) -> FlatParams:
    """Process user-provided params into internal params.

    Users can provide parameters at exactly one of three levels:

    - Model level: `{"arg_0": 0.0}` — propagates to all functions needing arg_0
    - Regime level: `{"regime_0": {"arg_0": 0.0}}` — propagates within regime_0
    - Function level: `{"regime_0": {"func": {"arg_0": 0.0}}}` — direct specification

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
) -> FlatParams:
    """Broadcast user params to template shape via 3-level resolution.

    For each template qname, search for a matching user value at:

    1. Exact match: `regime__function__param`
    2. Regime level: `regime__param`
    3. Model level: `param`

    Returns the resolved structure with leaves left as the user supplied
    them; dtype canonicalisation is a separate step
    (`cast_params_to_canonical_dtypes`).

    Args:
        params: User-provided values at any nesting depth.
        template: Target structure defining all valid 3-part keys.
        required: If True, raise when any template key has no match.

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

    result: dict[str, dict[str, object]] = {name: {} for name in template}
    used_keys: set[str] = set()

    for qname in template_flat:
        candidates = _find_candidates(qname=qname, params_flat=params_flat)

        if len(candidates) > 1:
            raise InvalidNameError(
                f"Ambiguous parameter specification for {qname!r}. "
                f"Found values at: {candidates}"
            )

        if candidates:
            chosen = candidates[0]
            path = tree_path_from_qname(qname)
            regime = path[0]
            remainder = qname_from_tree_path(path[1:])
            result[regime][remainder] = params_flat[chosen]
            used_keys.add(chosen)
        elif required:
            raise InvalidParamsError(f"Missing required parameter: {qname!r}")

    unknown = set(params_flat) - used_keys
    if unknown:
        raise InvalidParamsError(f"Unknown keys: {sorted(unknown)}")

    return cast(
        "FlatParams",
        MappingProxyType({k: MappingProxyType(v) for k, v in result.items()}),
    )


def cast_params_to_canonical_dtypes(flat_params: FlatParams) -> FlatParams:
    """Cast every numeric leaf of `flat_params` to its canonical pylcm dtype.

    Runs as a separate pass so the orchestrator can interpose
    `convert_series_in_params` between broadcast and cast — by the time
    this pass walks the tree, no `pd.Series` leaf should remain.

    Args:
        flat_params: Output of `broadcast_to_template`, optionally
            after `convert_series_in_params`.

    Returns:
        New immutable mapping with every leaf cast to its canonical dtype.

    """
    return cast(
        "FlatParams",
        MappingProxyType(
            {
                regime: MappingProxyType(
                    {
                        param_qname: _cast_leaves_to_canonical_dtype(
                            value, name=f"{regime}{QNAME_DELIMITER}{param_qname}"
                        )
                        for param_qname, value in leaves.items()
                    }
                )
                for regime, leaves in flat_params.items()
            }
        ),
    )


def _cast_leaves_to_canonical_dtype(value: Any, *, name: str) -> Any:  # noqa: ANN401, C901, PLR0911
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
                k: _cast_leaves_to_canonical_dtype(v, name=f"{name}.{k}")
                for k, v in value.data.items()
            }
        )
    if isinstance(value, UserSequenceLeaf):
        return SequenceLeaf(
            [
                _cast_leaves_to_canonical_dtype(v, name=f"{name}[{i}]")
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
        return jnp.bool_(value)
    if isinstance(value, int):
        return safe_to_int_dtype(value, name=name)
    if isinstance(value, float):
        return safe_to_float_dtype(value, name=name)
    if isinstance(value, (Array, np.ndarray)):
        kind = value.dtype.kind
        if kind == "b":
            return jnp.asarray(value, dtype=jnp.bool_)
        if kind in ("i", "u"):
            return safe_to_int_dtype(value, name=name)
        if kind == "f":
            return safe_to_float_dtype(value, name=name)
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


def _find_candidates(
    *,
    qname: str,
    params_flat: Mapping[str, object],
) -> list[str]:
    """Find candidate matches for a template qname at exact / regime / model levels."""
    tree_path = tree_path_from_qname(qname)
    param_name = tree_path[-1]
    candidates: list[str] = []

    if qname in params_flat:
        candidates.append(qname)

    if len(tree_path) == _NUM_PARTS_FUNCTION_PARAM:
        regime_level_qname = qname_from_tree_path((tree_path[0], param_name))
        if regime_level_qname in params_flat:
            candidates.append(regime_level_qname)

    if param_name in params_flat:
        candidates.append(param_name)

    return candidates


def create_params_template(  # noqa: C901
    regimes: MappingProxyType[RegimeName, Regime],
) -> ParamsTemplate:
    """Create params_template from internal regimes and validate name uniqueness.

    This function validates that regime names, function names, and argument names
    are disjoint sets to enable unambiguous parameter propagation.

    Args:
        regimes: Immutable mapping of regime names to Regime
            instances.

    Returns:
        The parameter template.

    Raises:
        InvalidNameError: If names are not disjoint or contain the separator.

    """
    template: dict[str, Any] = {}
    regime_names: set[str] = set()
    function_names: set[str] = set()
    arg_names: set[str] = set()

    for name, regime in regimes.items():
        regime_names.add(name)
        regime_template = dict(regime.regime_params_template)
        template[name] = regime_template

        for key, val in regime_template.items():
            if isinstance(val, (dict, Mapping)):
                function_names.add(key)
                for arg_name in val:
                    # Check for separator in argument names
                    if QNAME_DELIMITER in arg_name:
                        raise InvalidNameError(
                            f"Argument name {arg_name!r} in function {key!r} "
                            f"cannot contain the separator '{QNAME_DELIMITER}'"
                        )
                    arg_names.add(arg_name)
            else:
                raise InvalidNameError(
                    f"Parameter {key!r} in regime {name!r} must be nested under "
                    f"a function name, e.g., {{'function_name': {{'{key}': type}}}}"
                )

    # Check for separator in regime names
    for name in regime_names:
        if QNAME_DELIMITER in name:
            raise InvalidNameError(
                f"Regime name {name!r} cannot contain the separator '{QNAME_DELIMITER}'"
            )

    # Check for separator in function names
    for name in function_names:
        if QNAME_DELIMITER in name:
            raise InvalidNameError(
                f"Function name {name!r} cannot contain the separator "
                f"'{QNAME_DELIMITER}'"
            )

    # Check that names are disjoint
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

    # Note: Function names CAN overlap with argument names across regimes.
    # This happens when a function output in one regime is a parameter in another.
    # E.g., labor_income is a function in 'working' but a param in 'retired'.

    return ensure_containers_are_immutable(template)


def get_flat_param_names(regime_params_template: RegimeParamsTemplate) -> set[str]:
    """Get all flat parameter names from a regime params template.

    Converts nested template entries like {"utility": {"risk_aversion": type}} to
    flat names like "utility__risk_aversion".

    """
    result = set()
    for key, value in regime_params_template.items():
        if isinstance(value, Mapping):
            for param_name in value:
                result.add(qname_from_tree_path((key, param_name)))
    return result
