"""Declared signatures carried by scheduled role wrappers retain durable identity."""

import inspect

import pytest

from _lcm.output_roles import OutputRole, in_role
from _lcm.regime_building.schedules import _period_dispatch
from _lcm.regime_building.transition_support import _SupportedStochasticTransition
from _lcm.solution.fingerprint import _semantic_fingerprint


def _probability(*, state: float | int) -> float:  # noqa: PYI041 - Test exact union metadata stored in inspect.Signature.
    return 0.5 if state else 0.25


def test_declared_union_signature_is_supported() -> None:
    assert _semantic_fingerprint(inspect.signature(_probability))


@pytest.mark.parametrize("changed", ["parameter", "return", "optional"])
def test_union_signature_identity_binds_annotation_members(changed: str) -> None:
    baseline = inspect.signature(_probability)
    parameter = baseline.parameters["state"]
    if changed == "return":
        alternative = baseline.replace(return_annotation=float | int)
    else:
        annotation = float | str if changed == "parameter" else float | int | None
        alternative = baseline.replace(
            parameters=[parameter.replace(annotation=annotation)]
        )

    assert _semantic_fingerprint(baseline) != _semantic_fingerprint(alternative)


def test_period_dispatch_of_role_wrapped_laws_is_fingerprintable() -> None:
    case = in_role(func=_probability, role=OutputRole.FLOAT, kind="Law")
    dispatch = _period_dispatch(
        cases=(case,), case_names=(("state",),), case_by_period=(0, 0)
    )
    transition = _SupportedStochasticTransition(func=dispatch, targets=("active",))

    assert _semantic_fingerprint(transition)


def test_opaque_signature_default_remains_rejected() -> None:
    signature = inspect.Signature(
        parameters=[
            inspect.Parameter(
                "state",
                inspect.Parameter.KEYWORD_ONLY,
                default=object(),
                annotation=int,
            )
        ]
    )

    with pytest.raises(TypeError, match="opaque semantic value"):
        _semantic_fingerprint(signature)
