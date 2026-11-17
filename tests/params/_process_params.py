"""The one-shot user-params processing the unit tests drive.

Production (`Model._process_params`) runs the same steps with
`convert_series_in_params` in between; tests whose params hold no `pd.Series`
take them in one call. Each slot is cast in the numeric role its template leaf
declares, as in production.
"""

from collections.abc import Mapping
from types import MappingProxyType
from typing import cast

from _lcm.params.numeric_roles import NumericRole, TemplateAnnotation, template_roles
from _lcm.params.processing import (
    broadcast_to_template,
    cast_params_to_canonical_dtypes,
)
from _lcm.typing import FlatParams, ParamsTemplate
from _lcm.utils.containers import ensure_containers_are_immutable
from lcm.typing import UserParams

type TemplateNode = str | Mapping[str, TemplateNode]
type RoledNode = TemplateAnnotation | dict[str, RoledNode]

# The role production reads off these annotation spellings; any other keeps the
# kind of the value it receives.
_ROLE_OF_TEXT = MappingProxyType({"float": NumericRole.FLOAT, "int": NumericRole.INT})


def as_template(plain: Mapping[str, TemplateNode]) -> ParamsTemplate:
    """Freeze a plain nested dict into a `ParamsTemplate` whose leaves carry roles.

    A leaf `"float"` or `"int"` takes that role, as an argument annotated so does;
    any other leaf keeps the kind of the value it receives.
    """
    return cast(
        "ParamsTemplate",
        ensure_containers_are_immutable(
            {key: _with_roles(child) for key, child in plain.items()}
        ),
    )


def _with_roles(node: TemplateNode) -> RoledNode:
    if isinstance(node, str):
        return TemplateAnnotation(
            node, role=_ROLE_OF_TEXT.get(node, NumericRole.AS_GIVEN)
        )
    return {key: _with_roles(child) for key, child in node.items()}


def process_params(
    *,
    params: UserParams,
    params_template: ParamsTemplate,
) -> FlatParams:
    """Resolve user params against the template and cast them, as one step.

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
    return cast_params_to_canonical_dtypes(
        internal, roles=template_roles(params_template)
    )
