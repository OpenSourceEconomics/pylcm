"""Declared payload types admit the values their producers actually write.

The CI tooling exchanges JSON reports whose shapes are declared as `TypedDict`s.
Each test feeds a real producer's output through a recursive conformance check,
so a declaration that disagrees with the wire format fails here even though no
runtime checker inspects `TypedDict` fields.
"""

import json
import subprocess
import sys
from pathlib import Path
from typing import (
    Literal,
    NotRequired,
    TypeAliasType,
    TypedDict,
    Union,
    get_args,
    get_origin,
    get_type_hints,
    is_typeddict,
)

import pytest

from tests.candidate_certificate.verify import _CliVerificationReport
from tests.ci.ci_workloads import WorkloadManifest, load_manifest
from tests.ci.pytest_policy import _SelectionRecord

_REPO_ROOT = Path(__file__).resolve().parents[2]
_VERIFIER = _REPO_ROOT / "tests" / "candidate_certificate" / "verify.py"
type _ScalarType = type[bool | int | float | str | None]

# The JSON scalar types each declared scalar type admits.
_SCALAR_TYPES: tuple[tuple[_ScalarType | None, frozenset[_ScalarType]], ...] = (
    (None, frozenset({type(None)})),
    (type(None), frozenset({type(None)})),
    (bool, frozenset({bool})),
    (int, frozenset({int})),
    (float, frozenset({int, float})),
    (str, frozenset({str})),
)


class _Probe(TypedDict):
    version: int
    tags: list[str]
    note: NotRequired[str | None]


@pytest.mark.parametrize(
    ("payload", "conforms"),
    [
        ({"version": 2, "tags": ["a"]}, True),
        ({"version": 2, "tags": ["a"], "note": None}, True),
        ({"version": "2", "tags": ["a"]}, False),
        ({"version": 2, "tags": "a"}, False),
        ({"version": 2}, False),
        ({"version": 2, "tags": [], "extra": 1}, False),
    ],
)
def test_conformance_check_tells_matching_from_mismatching_payloads(
    *,
    payload: dict[str, int | str | list[str] | None],
    conforms: bool,
) -> None:
    """Types, nesting, required keys and unknown keys all decide conformance."""
    assert _conforms(value=payload, hint=_Probe) is conforms


def test_selection_records_declare_the_precision_the_option_produces(
    pytestconfig: pytest.Config,
) -> None:
    """The `--precision` value a selection record copies fits its declared type."""
    precision = pytestconfig.getoption("--precision")

    assert _conforms(
        value=precision, hint=get_type_hints(_SelectionRecord)["precision"]
    )


def test_the_committed_workload_manifest_matches_its_declared_shape() -> None:
    """Every field of the committed CI workload manifest fits its declaration."""
    assert _conforms(value=load_manifest(), hint=WorkloadManifest)


def test_the_verifier_cli_report_matches_the_shape_its_adapters_declare() -> None:
    """The JSON the certificate verifier prints is what its mutation adapters parse."""
    completed = subprocess.run(  # noqa: S603
        [sys.executable, str(_VERIFIER), "--repo-root", str(_REPO_ROOT)],
        capture_output=True,
        check=False,
        text=True,
    )

    assert _conforms(value=json.loads(completed.stdout), hint=_CliVerificationReport)


def _conforms(
    *,
    value: object,  # noqa: PAN001 - any decoded JSON or Python payload is probed
    hint: object,  # noqa: PAN001 - a type expression of any typing form
) -> bool:
    """Return whether `value` fits `hint`, recursing through fields and items."""
    if isinstance(hint, TypeAliasType):
        hint = hint.__value__
    if is_typeddict(hint):
        fields = get_type_hints(hint)
        required = hint.__required_keys__  # ty: ignore[unresolved-attribute]
        return (
            isinstance(value, dict)
            and required <= value.keys() <= fields.keys()
            and all(_conforms(value=value[key], hint=fields[key]) for key in value)
        )
    origin, arguments = get_origin(hint), get_args(hint)
    if origin is Union:
        return any(_conforms(value=value, hint=argument) for argument in arguments)
    if origin is Literal:
        return any(
            type(value) is type(argument) and value == argument
            for argument in arguments
        )
    if origin in {list, dict}:
        return _conforms_container(value=value, origin=origin, arguments=arguments)
    admitted = next(
        (scalars for declared, scalars in _SCALAR_TYPES if hint is declared), None
    )
    if admitted is not None:
        return type(value) in admitted
    msg = f"no conformance rule for {hint!r}"
    raise TypeError(msg)


def _conforms_container(
    *,
    value: object,  # noqa: PAN001 - any decoded JSON or Python payload is probed
    origin: type[list[object] | dict[object, object]],  # noqa: PAN001 - the element types are checked per item below
    arguments: tuple[object, ...],  # noqa: PAN001 - type expressions of any typing form
) -> bool:
    if origin is list:
        return isinstance(value, list) and all(
            _conforms(value=item, hint=arguments[0]) for item in value
        )
    return isinstance(value, dict) and all(
        _conforms(value=key, hint=arguments[0])
        and _conforms(value=item, hint=arguments[1])
        for key, item in value.items()
    )
