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

from _lcm.dtypes import safe_to_float_dtype, safe_to_int_dtype
from _lcm.engine import Regime
from _lcm.params.mapping_leaf import MappingLeaf, UserMappingLeaf
from _lcm.params.sequence_leaf import SequenceLeaf, UserSequenceLeaf
from _lcm.typing import FlatParams, ParamsTemplate, RegimeName, RegimeParamsTemplate
from _lcm.utils.containers import ensure_containers_are_immutable
from _lcm.utils.namespace import ParamsQnameDepth, flatten_regime_namespace
from lcm.exceptions import InvalidNameError, InvalidParamsError
from lcm.typing import UserParams


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
    missing: list[str] = []

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
            missing.append(qname)

    unknown = set(params_flat) - used_keys
    if missing or unknown:
        messages = []
        if missing:
            messages.append(f"Missing required parameter(s): {missing}")
        if unknown:
            messages.append(f"Unknown keys: {sorted(unknown)}")
        raise InvalidParamsError(" ".join(messages))

    return cast(
        "FlatParams",
        MappingProxyType({k: MappingProxyType(v) for k, v in result.items()}),
    )


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
    # One cast per distinct input object: a value broadcast into several
    # slots (e.g. a coarse value resolved into per-target template slots)
    # stays one shared leaf, so downstream consumers can deduplicate by
    # identity and large array leaves are not copied per slot.
    memo: dict[int, Any] = {}

    def _cast_shared(value: Any, *, name: str) -> Any:  # noqa: ANN401
        key = id(value)
        if key not in memo:
            memo[key] = _cast_leaves_to_canonical_dtype(value, name=name)
        return memo[key]

    return cast(
        "FlatParams",
        MappingProxyType(
            {
                regime: MappingProxyType(
                    {
                        param_qname: _cast_shared(
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
    """Find candidate matches for a template qname, most to least specific.

    Resolution levels:

    1. Exact match: `regime__function__param`, or for per-target transition
       params `regime__target__function__param`
    2. Coarse function level (per-target qnames only):
       `regime__function__param` — one value broadcasts over the targets
    3. Regime level: `regime__param`
    4. Model level: `param`

    """
    tree_path = tree_path_from_qname(qname)
    param_name = tree_path[-1]
    candidates: list[str] = []

    if qname in params_flat:
        candidates.append(qname)

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
    regime_names: set[str] = set(regimes)
    function_names: set[str] = set()
    arg_names: set[str] = set()

    for name, regime in regimes.items():
        regime_template = dict(regime.regime_params_template)
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

    return ensure_containers_are_immutable(template)


def _validated_arg_names(
    *,
    func_name: str,
    params: Mapping,
    regime_name: str,
) -> set[str]:
    """Return a function entry's argument names, validating each leaf.

    Argument names must be separator-free and map to bare leaves — a nested
    mapping at this depth means the user nested params one level too deep.
    """
    arg_names: set[str] = set()
    for arg_name, leaf in params.items():
        if QNAME_DELIMITER in arg_name:
            raise InvalidNameError(
                f"Argument name {arg_name!r} in function {func_name!r} "
                f"cannot contain the separator '{QNAME_DELIMITER}'"
            )
        if isinstance(leaf, Mapping):
            raise InvalidNameError(
                f"Parameter {arg_name!r} in regime {regime_name!r} is "
                f"nested too deeply."
            )
        arg_names.add(arg_name)
    return arg_names


def _fail_if_template_names_invalid(
    *,
    regime_names: set[str],
    function_names: set[str],
    arg_names: set[str],
) -> None:
    """Validate separator-freedom and disjointness of template name sets.

    Regime and function names must not contain the qname separator, and
    regime names must be disjoint from both function and argument names so
    parameter propagation stays unambiguous. Function names CAN overlap with
    argument names across regimes — a function output in one regime may be a
    parameter in another (e.g. `labor_income` is a function in `working` but
    a param in `retired`).
    """
    for name in regime_names:
        if QNAME_DELIMITER in name:
            raise InvalidNameError(
                f"Regime name {name!r} cannot contain the separator '{QNAME_DELIMITER}'"
            )

    for name in function_names:
        if QNAME_DELIMITER in name:
            raise InvalidNameError(
                f"Function name {name!r} cannot contain the separator "
                f"'{QNAME_DELIMITER}'"
            )

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


def get_flat_param_names(regime_params_template: RegimeParamsTemplate) -> set[str]:
    """Get all flat parameter names from a regime params template.

    Converts nested template entries like `{"utility": {"risk_aversion": type}}`
    to flat names like `utility__risk_aversion`; per-target branches like
    `{"retired": {"next_wealth": {"exit_tax": type}}}` yield
    `retired__next_wealth__exit_tax`.

    """
    result = set()
    for key, value in regime_params_template.items():
        if not isinstance(value, Mapping):
            continue
        for inner_name, inner_value in value.items():
            if isinstance(inner_value, Mapping):
                result.update(
                    qname_from_tree_path((key, inner_name, param_name))
                    for param_name in inner_value
                )
            else:
                result.add(qname_from_tree_path((key, inner_name)))
    return result
