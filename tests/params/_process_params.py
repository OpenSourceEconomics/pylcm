"""The one-shot user-params processing the unit tests drive.

Production (`Model._process_params`) runs the same steps with
`convert_series_in_params` in between; tests whose params hold no `pd.Series`
take them in one call.
"""

from _lcm.params.processing import (
    broadcast_to_template,
    cast_params_to_canonical_dtypes,
)
from _lcm.typing import FlatParams, ParamsTemplate
from lcm.typing import UserParams


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
    return cast_params_to_canonical_dtypes(internal)
