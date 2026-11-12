"""Interpret allocation-block admission while retaining original source contexts."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
from importlib.resources import files
import json
from types import MappingProxyType

from .records import JSONValue


@dataclass(frozen=True)
class _Specification:
    schema: str
    """Versioned data contract shared with the independent interpreter."""
    digest: str
    """Digest of the exact normative JSON bytes."""
    equal_fields: tuple[str, ...]
    """Execution-interface fields that must agree across a block."""
    defaults: Mapping[str, JSONValue]
    """Values of absent execution-interface fields."""
    context_fields: tuple[str, ...]
    """Original per-cell facts preserved by dispatch."""


def context_key(cell: Mapping[str, JSONValue]) -> str:
    """Return the declared common execution-interface key, without actor pooling."""
    values = tuple(cell.get(name, _SPEC.defaults[name]) for name in _SPEC.equal_fields)
    return json.dumps(values, sort_keys=True, separators=(",", ":"), allow_nan=False)


def admission_record(cells: Sequence[Mapping[str, JSONValue]]) -> dict[str, JSONValue]:
    """Bind each block member to its own original context and terminal semantics."""
    if not cells or len({context_key(cell) for cell in cells}) != 1:
        raise ValueError("Block members have incompatible execution interfaces")
    bindings: dict[str, JSONValue] = {}
    for cell in cells:
        identifier = cell.get("id")
        if not isinstance(identifier, str) or identifier in bindings:
            raise ValueError("Block source-cell identifiers must be unique strings")
        values = tuple((name, name in cell, cell.get(name)) for name in _SPEC.context_fields)
        payload = json.dumps(values, sort_keys=True, separators=(",", ":"), allow_nan=False)
        bindings[identifier] = hashlib.sha256(payload.encode()).hexdigest()
    return {"schema": _SPEC.schema, "specification_sha256": _SPEC.digest,
            "source_contexts": bindings}


def _names(value: JSONValue) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(isinstance(name, str) for name in value):
        raise ValueError("Admission field names must be a list of strings")
    return tuple(name for name in value if isinstance(name, str))


def _load() -> _Specification:
    payload = files(__package__).joinpath("block_admission_v1.json").read_bytes()
    data = json.loads(payload)
    if data["schema"] != "lcm-regime-block-admission/v1":
        raise ValueError("Unsupported block-admission specification")
    block = data["block"]
    if any(block[name] for name in ("same_actor_required", "one_choice_per_period_required",
                                    "uniform_terminal_required", "uniform_role_required",
                                    "uniform_ownership_required")):
        raise ValueError("Source-cell dispatch contract was changed")
    return _Specification(schema=data["schema"], digest=hashlib.sha256(payload).hexdigest(),
                          equal_fields=_names(block["equal_fields"]),
                          defaults=MappingProxyType(block["missing_field_values"]),
                          context_fields=(_names(data["source_dispatch"]["key"])
                                          + _names(data["source_dispatch"]["preserve_fields"])))


_SPEC = _load()
