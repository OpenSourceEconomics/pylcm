"""A public capture's archive records are admitted only at their declared types."""

import re

import pytest

from _lcm.solution import public_period_replay

_SHARDING = {
    "kind": "named",
    "device_ids": [0, 1],
    "partition_spec": ["cell", ["a", "b"], None],
    "mesh_axis_names": ["cell"],
    "mesh_axis_sizes": [2],
    "memory_kind": "device",
}
_LEAF = {
    "tree_path": "['next_regime_to_V_arr']['work']",
    "shape": [3],
    "dtype": "float64",
    "weak_type": False,
    "committed": True,
    "sharding": _SHARDING,
}
_TRANSFER = {
    "kind": "reshard",
    "target": "target",
    "source": "source",
    "stored_sharding": _SHARDING,
    "source_sharding": _SHARDING,
    "expected_shape": [3],
    "expected_dtype": "float64",
}
_CORE = {
    "name": "main",
    "lowered_out_shardings": [_SHARDING],
    "compiled_input_shardings": [["x", _SHARDING]],
    "compiled_output_shardings": [["out", _SHARDING]],
    "input_transfer_plan": [_TRANSFER],
    "donated_arguments": [],
    "variant": "non_donating",
}
_LAYOUTS = {
    "route": "package.Kernel",
    "device_ids": [0, 1],
    "leaves": [_LEAF],
    "cores": {"main": _CORE, "unused": None},
}
_ADMISSION = {
    "budget_bytes": 1024,
    "resident_bytes": 64,
    "reservation_bytes": 128,
    "peak_bytes": 128,
}
_METADATA = {
    "regime": "work",
    "period": 0,
    "layouts": _LAYOUTS,
    "widths": {"main": {"cell": 2}},
    "admission": {"main": _ADMISSION},
    "optimized_hlo": {"main": {"fingerprint": "abc"}},
    "retain_replay": False,
}

_RECORDS = {
    "_is_sharding_record": _SHARDING,
    "_is_leaf_record": _LEAF,
    "_is_transfer_record": _TRANSFER,
    "_is_core_record": _CORE,
    "_is_layouts_record": _LAYOUTS,
    "_is_admission_record": _ADMISSION,
    "_is_capture_metadata": _METADATA,
}

_KEY_CASES = [
    pytest.param(validator, key, id=f"{validator}-{key}")
    for validator, record in _RECORDS.items()
    for key in record
]


@pytest.mark.parametrize("validator", _RECORDS)
def test_record_validator_accepts_a_well_formed_record(validator: str) -> None:
    """A record carrying every entry at its declared type is admitted."""
    assert getattr(public_period_replay, validator)(_RECORDS[validator])


@pytest.mark.parametrize(("validator", "key"), _KEY_CASES)
def test_record_validator_names_a_missing_key(*, validator: str, key: str) -> None:
    """A record lacking an entry is refused, naming that entry."""
    record = {name: value for name, value in _RECORDS[validator].items() if name != key}

    with pytest.raises(ValueError, match=re.escape(repr(key))):
        getattr(public_period_replay, validator)(record)


@pytest.mark.parametrize(("validator", "key"), _KEY_CASES)
def test_record_validator_names_a_wrong_typed_key(*, validator: str, key: str) -> None:
    """A record holding an entry at the wrong type is refused, naming that entry."""
    record = {**_RECORDS[validator], key: 0.5}

    with pytest.raises(TypeError, match=re.escape(repr(key))):
        getattr(public_period_replay, validator)(record)


def test_layouts_validator_names_a_key_missing_from_a_nested_leaf() -> None:
    """A layout block is refused when one of its leaves lacks an entry."""
    leaf = {name: value for name, value in _LEAF.items() if name != "dtype"}

    with pytest.raises(ValueError, match="'dtype'"):
        public_period_replay._is_layouts_record({**_LAYOUTS, "leaves": [leaf]})
