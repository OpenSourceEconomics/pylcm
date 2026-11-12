"""Check source-cell block admission directly from its normative data file.

This interpreter imports no producer admission, search, diagnostic or cost builder.
"""

from collections.abc import Mapping, Sequence
from functools import cache
import hashlib
from importlib.resources import files
import json
from typing import TypedDict

from .records import JSONValue


class _BlockRules(TypedDict):
    equal_fields: list[str]
    missing_field_values: Mapping[str, JSONValue]
    same_actor_required: bool
    one_choice_per_period_required: bool
    uniform_terminal_required: bool
    uniform_role_required: bool
    uniform_ownership_required: bool


class _DispatchRules(TypedDict):
    key: list[str]
    preserve_fields: list[str]


class _SpecificationData(TypedDict):
    schema: str
    block: _BlockRules
    source_dispatch: _DispatchRules


def checked_context_key(cell: Mapping[str, JSONValue]) -> str:
    """Derive the common interface independently of the producer."""
    specification = _specification()
    fields = specification["block"]["equal_fields"]
    defaults = specification["block"]["missing_field_values"]
    values = [cell[name] if name in cell else defaults[name] for name in fields]
    return json.dumps(values, sort_keys=True, separators=(",", ":"), allow_nan=False)


def expected_admission(cells: Sequence[Mapping[str, JSONValue]]) -> dict[str, JSONValue]:
    """Recompute every original context binding without producer helpers."""
    if len(cells) == 0:
        raise ValueError("Empty allocation block")
    reference = checked_context_key(cells[0])
    specification = _specification()
    bindings: dict[str, JSONValue] = {}
    for cell in cells:
        if checked_context_key(cell) != reference:
            raise ValueError("Conflicting block interface field")
        identifier = cell.get("id")
        if not isinstance(identifier, str) or identifier in bindings:
            raise ValueError("Source-cell coverage is not unique")
        entries = []
        dispatch = specification["source_dispatch"]
        for field in dispatch["key"] + dispatch["preserve_fields"]:
            entries.append([field, field in cell, cell.get(field)])
        serialized = json.dumps(entries, sort_keys=True, separators=(",", ":"), allow_nan=False)
        bindings[identifier] = hashlib.sha256(serialized.encode()).hexdigest()
    return {"schema": specification["schema"],
            "specification_sha256": hashlib.sha256(_PAYLOAD).hexdigest(),
            "source_contexts": bindings}


@cache
def _specification() -> _SpecificationData:
    value: _SpecificationData = json.loads(_PAYLOAD)
    if value["schema"] != "lcm-regime-block-admission/v1":
        raise ValueError("Unsupported independent block-admission specification")
    block = value["block"]
    if any(required is not False for required in (
        block["same_actor_required"], block["one_choice_per_period_required"],
        block["uniform_terminal_required"], block["uniform_role_required"],
        block["uniform_ownership_required"],
    )):
        raise ValueError("Invalid source-dispatch rule")
    return value


_PAYLOAD = files(__package__).joinpath("block_admission_v1.json").read_bytes()
