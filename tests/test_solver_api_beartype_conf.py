from collections.abc import Callable
from dataclasses import replace
from typing import cast

import jax.numpy as jnp
import pytest
from beartype import beartype

from _lcm.typing import ArtifactPayload
from lcm._solver_api.beartype_conf import SOLVER_API_CONF
from lcm.exceptions import SolverAPITypeError
from lcm.solver_api import ArtifactKey, ArtifactRef, ArtifactStore, ValueStore


@beartype(conf=SOLVER_API_CONF)
def _echo_int(*, value: int) -> int:
    return value


@beartype(conf=SOLVER_API_CONF)
def _misreport_int(*, value: int) -> int:
    return str(value)  # ty: ignore[invalid-return-type]


def test_solver_api_conf_reports_a_wrong_argument_as_a_type_error() -> None:
    """A wrongly typed argument at a solver-API boundary is a `TypeError`."""
    with pytest.raises(TypeError, match="value"):
        _echo_int(value="1")  # ty: ignore[invalid-argument-type]


def test_solver_api_conf_reports_a_wrong_return_as_the_solver_api_error() -> None:
    """A wrongly typed return at a solver-API boundary is a `SolverAPITypeError`."""
    with pytest.raises(SolverAPITypeError):
        _misreport_int(value=1)


def _stores() -> tuple[ValueStore, ArtifactStore, ArtifactRef]:
    ref = ArtifactRef(
        period=0, regime="alive", key=ArtifactKey(type_id="example.policy")
    )
    values = ValueStore({(0, "alive"): jnp.asarray([1.0])})
    artifacts = ArtifactStore({ref: cast("ArtifactPayload", object())})
    return values, artifacts, ref


@pytest.mark.parametrize(
    ("read", "parameter"),
    [
        (lambda values, _artifacts: values["0"], "period"),
        (lambda values, _artifacts: values[0][1], "regime"),
        (lambda _values, artifacts: artifacts.load_state("ref"), "ref"),
        (lambda _values, artifacts: artifacts.project("key"), "key"),
    ],
    ids=["value-period", "value-regime", "artifact-ref", "artifact-key"],
)
def test_store_reads_report_a_wrongly_typed_coordinate_as_the_solver_api_error(
    *,
    read: Callable[[ValueStore, ArtifactStore], object],
    parameter: str,
) -> None:
    """A store read with a wrongly typed coordinate names it in a `TypeError`."""
    values, artifacts, _ = _stores()
    with pytest.raises(SolverAPITypeError, match=rf"parameter {parameter}="):
        read(values, artifacts)


_WRONGLY_TYPED_REF_FIELDS = [("period", "0"), ("regime", 0), ("key", "example")]


@pytest.mark.parametrize(("field", "value"), _WRONGLY_TYPED_REF_FIELDS)
def test_artifact_store_load_state_names_a_wrongly_typed_ref_field(
    *, field: str, value: object
) -> None:
    """An exact `ArtifactRef` holding a wrongly typed field is refused by name."""
    _, artifacts, ref = _stores()
    hostile = replace(ref)
    object.__setattr__(hostile, field, value)
    with pytest.raises(SolverAPITypeError, match=rf"parameter {field}="):
        artifacts.load_state(hostile)


@pytest.mark.parametrize(("field", "value"), _WRONGLY_TYPED_REF_FIELDS)
def test_artifact_store_membership_is_false_for_a_wrongly_typed_ref_field(
    *, field: str, value: object
) -> None:
    """Membership of an exact `ArtifactRef` holding a wrongly typed field is False."""
    _, artifacts, ref = _stores()
    hostile = replace(ref)
    object.__setattr__(hostile, field, value)
    assert hostile not in artifacts
