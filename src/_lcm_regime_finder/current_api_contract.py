"""Source authorities for native participation declarations.

These hashes bind the original class source, not arbitrary names. They do not
prove callback evaluation, information equivalence, or numerical allocation.
The independent checker separately validates each original payload authority.
"""
from __future__ import annotations

_PARTICIPATION_FILE = "19cc45ca193a562403f57f5d96525cf1e68641bef68abc3d4f4f6ea352d8301a"
_AUTHORITIES = {
    "lcm.participation.ParticipationRead": (
        "4d036ef554fa2b2e93e8b1f8da3366c701700c4e439006944674570d37e0d428",
        (_PARTICIPATION_FILE,),
    ),
    "lcm.participation.ParticipationCheck": (
        "44c8bb9ec6f04e6e31bd557fe5f58bfbbcb3bb5d8d2ccfedd0e86ba38c38fd36",
        (_PARTICIPATION_FILE,),
    ),
    "lcm.transition.Transition": (
        "46088bb83593235ee33e56c2f0b92a29ba51396e91d38467dac62058008c2245",
        (
            "587518f47dc4909c070ffd72e161cec0be134919a7d89f480976e6c1b0d1c496",
            # Specialization types changed in the same file. Transition is identical.
            "bee97101a92d339e52e930066ac63b88158ca786f3d63cbb562d129ea10c1383",
        ),
    ),
}


def matches_authority(*, name: str, observed: object) -> bool:
    """Bind both the captured class and its containing source file."""
    if name not in _AUTHORITIES or not isinstance(observed, dict):
        return False
    module, qualname = name.rsplit(".", 1)
    source_sha256, file_hashes = _AUTHORITIES[name]
    return (
        observed.get("module") == module
        and observed.get("qualname") == qualname
        and observed.get("source_sha256") == source_sha256
        and observed.get("file_sha256") in file_hashes
    )


def inspect_authority(cls):
    import hashlib
    import inspect
    from pathlib import Path
    name = cls.__module__ + "." + cls.__qualname__
    if name not in _AUTHORITIES:
        return None
    try:
        source = inspect.getsource(cls).strip()
        filename = inspect.getsourcefile(cls)
        file_hash = hashlib.sha256(Path(filename).read_bytes()).hexdigest()
    except (TypeError, OSError):
        return None
    return {"module": cls.__module__, "qualname": cls.__qualname__,
            "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
            "file_sha256": file_hash}


def validate_original_types(value):
    """Refuse a modern spelling without the corresponding original source."""
    if isinstance(value, dict):
        name = value.get("type", "")
        if value.get("kind") not in ("dataclass", "object") or not isinstance(name, str):
            name = ""
        modern = name.rsplit(".", 1)[-1] in {"ParticipationRead", "ParticipationCheck"}
        modern_transition = name.rsplit(".", 1)[-1] == "Transition" and bool(value.get("fields", {}).get("participation"))
        if modern or modern_transition:
            if not matches_authority(name=name, observed=value.get("canonical_type_source")):
                raise ValueError("Current participation declaration lacks verified canonical class source: " + name)
        for child in value.values():
            validate_original_types(child)
    elif isinstance(value, list):
        for child in value:
            validate_original_types(child)


def gate_maps(data, *names):
    result = {}
    for name in names:
        values = data.get(name, {})
        if not isinstance(values, dict):
            raise ValueError("A transition participation/gate mapping is not captured")
        if result.keys() & values.keys():
            raise ValueError("A scheduled target has conflicting gate/participation declarations")
        result.update(values)
    return result
