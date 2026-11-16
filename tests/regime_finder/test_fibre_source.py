"""Original primitive metadata controls fibre reads and rectangle exclusions."""

from collections.abc import Callable, Iterator, Mapping
from copy import deepcopy
from fractions import Fraction
import inspect
from pathlib import Path
import json
import os
import gc
import subprocess
import textwrap
from types import FunctionType, GenericAlias, MappingProxyType, ModuleType, UnionType
from typing import TYPE_CHECKING, Literal, TypeAliasType, cast
from unittest.mock import patch

import pytest
import jax.numpy as jnp
from jax import Array
from beartype import beartype
from lcm import categorical
from lcm.typing import FloatND, ScalarInt, UserAge
import numpy as np
from numpy.typing import NDArray

from _lcm_regime_finder.declarations import normalize_declarations
from _lcm_regime_finder.callable_identity import capture_session
from _lcm_regime_finder.callable_identity import _beartype_interface
from _lcm_regime_finder.fibre_source import FibreSource, TargetSupport
from _lcm_regime_finder.fibre_facts import FibreLiveness, calculate_fibre_liveness
from _lcm_regime_finder.records import JSONValue
from _lcm_regime_finder.model_snapshot import normalize
from _lcm_regime_finder.semantic_check import (
    SourceProof,
    _CallableRecords,
    _UNKNOWN,
    _native_wrapper_inputs,
)
from _lcm_regime_finder.structural_arrays import NativeWriter
from _lcm_regime_finder.structural_check import Reader
from _lcm_regime_finder.structural_context import descriptor, named_roots, register
from _lcm_regime_finder.structural_events import compare_source, source_anchor
from tests.regime_finder.test_model_adapter import function, grid, snapshot
from tests.regime_finder.test_query_owners import policy_snapshot
from tests.regime_finder.test_round4_semantic import captured_gate_graph
from tests.regime_finder.test_graph_builtin_processes import _target_process_capture

if TYPE_CHECKING:
    from lcm.typing import RegimeName, UserFunction


def _object(value: JSONValue) -> dict[str, JSONValue]:
    assert isinstance(value, dict)
    return value


def _name(value: JSONValue) -> str:
    assert isinstance(value, str)
    return value


def _rows(value: JSONValue) -> list[JSONValue]:
    assert isinstance(value, list)
    return value


def _original() -> dict[str, JSONValue]:
    source = snapshot()
    _fields(source)["constraints"] = {}
    return source


def _native_history_input(history: float) -> float:
    raise AssertionError("Economic callback must not be evaluated")


def _native_history_code_input(history_code: int) -> float:
    raise AssertionError("Economic callback must not be evaluated")


def _native_ignored_history_input(x: float, history: float) -> float:
    return x


def _native_wrapper(
    kind: Literal["value", "role"], *, code_alias: bool = False
) -> UserFunction:
    if kind == "role":
        roles = pytest.importorskip("_lcm.output_roles")
        return roles.in_role(
            func=_native_history_input, role=roles.OutputRole.FLOAT, kind="Utility"
        )
    from _lcm.regime_building.value_adapter import ValueAdapter

    return cast(
        "UserFunction",
        ValueAdapter(
            func=_native_history_code_input if code_alias else _native_history_input,
            decoded=MappingProxyType({}),
            codes=MappingProxyType({"history_code": "history"} if code_alias else {}),
            encode=None,
        ),
    )


def _capture_native_wrapper(
    wrapper: UserFunction, *, declared: bool = True
) -> tuple[dict[str, JSONValue], dict[str, JSONValue]]:
    from lcm.declarations import FunctionDeclaration

    with capture_session() as session:
        if declared:
            function = getattr(wrapper, "func", None)
            declared_functions = (wrapper,) if function is None else (wrapper, function)
            session.declaration_interfaces = MappingProxyType(
                {
                    id(item): FunctionDeclaration(
                        function=item, signature=inspect.signature(item)
                    )
                    for item in declared_functions
                }
            )
        original = normalize(wrapper)
    return original, session.records


def _native_wrapper_source(
    wrapper: UserFunction, *, declared: bool = True
) -> tuple[dict[str, JSONValue], dict[str, JSONValue]]:
    native, records = _capture_native_wrapper(wrapper, declared=declared)
    original = _original()
    original["callable_records"] = records
    _object(_fields(original)["functions"])["utility"] = native
    return original, native


@pytest.mark.parametrize(
    ("kind", "code_alias"), [("value", False), ("value", True), ("role", False)]
)
def test_native_wrapper_contract_retains_complete_inputs_without_evaluation(
    kind: Literal["value", "role"], code_alias: bool
) -> None:
    wrapper = _native_wrapper(kind, code_alias=code_alias)
    original, native = _native_wrapper_source(wrapper)
    answer = SourceProof(
        {"utility": {"original": native}},
        callable_records=original["callable_records"],
    ).callback("utility")
    assert answer.proved
    assert answer.reads == frozenset({"history"})
    assert answer.value is _UNKNOWN
    assert _source(original).get_direct_reads(
        cell_id=_cell_id(original), statuses={}, removal_order=()
    ).live_coordinates == frozenset({"history"})


@pytest.mark.parametrize("kind", ["value", "role"])
def test_native_wrapper_helper_has_an_authenticated_finite_interface(
    kind: Literal["value", "role"],
) -> None:
    original, native = _native_wrapper_source(_native_wrapper(kind), declared=False)
    answer = SourceProof(
        {"utility": {"original": native}},
        callable_records=original["callable_records"],
    ).callback("utility")
    assert answer.proved
    assert answer.reads == frozenset({"history"})
    assert answer.value is _UNKNOWN


@pytest.mark.parametrize("kind", ["value", "role"])
def test_native_wrapper_contract_restores_all_effective_inputs_when_child_reads_less(
    kind: Literal["value", "role"],
) -> None:
    # Use the real native constructor so the published signature
    # and stored free-input names describe the complete original interface.
    if kind == "value":
        from _lcm.regime_building.value_adapter import ValueAdapter

        wrapper = cast(
            "UserFunction",
            ValueAdapter(
                func=_native_ignored_history_input,
                decoded=MappingProxyType({}),
                encode=None,
            ),
        )
    else:
        roles = pytest.importorskip("_lcm.output_roles")
        wrapper = roles.in_role(
            func=_native_ignored_history_input,
            role=roles.OutputRole.FLOAT,
            kind="Utility",
        )
    original, native = _native_wrapper_source(wrapper)
    answer = SourceProof(
        {"utility": {"original": native}},
        callable_records=original["callable_records"],
    ).callback("utility")
    assert answer.proved
    assert answer.reads == frozenset({"x", "history"})
    assert answer.value is _UNKNOWN
    source = _source(original)
    callback = next(
        record
        for record in source._callbacks.values()
        if record.get("source_path")
        == "/user_regimes/original/fields/functions/utility"
    )
    assert callback["arguments"] == ["x", "history"]
    assert callback["reads"] == ["x", "history"]


@pytest.mark.parametrize("kind", ["value", "role"])
def test_native_wrapper_rejects_an_advertised_signature_that_omits_an_operand(
    kind: Literal["value", "role"],
) -> None:
    wrapper = _native_wrapper(kind)
    object.__setattr__(wrapper, "__signature__", inspect.Signature())
    original, native = _native_wrapper_source(wrapper)
    answer = SourceProof(
        {"utility": {"original": native}},
        callable_records=original["callable_records"],
    ).callback("utility")
    assert not answer.proved
    assert _source(original).get_direct_reads(
        cell_id=_cell_id(original), statuses={}, removal_order=()
    ).live_coordinates == frozenset({"x", "history"})


@pytest.mark.parametrize("field", ["_reads", "codes", "func"])
def test_native_value_wrapper_rejects_changed_forwarding_fields(field: str) -> None:
    wrapper = _native_wrapper("value")
    if field == "_reads":
        object.__setattr__(wrapper, "_reads", ())
    elif field == "codes":
        object.__setattr__(wrapper, "codes", MappingProxyType({"history": "x"}))
    else:
        object.__setattr__(wrapper, "func", _native_history_code_input)
    original, native = _native_wrapper_source(wrapper)
    answer = SourceProof(
        {"utility": {"original": native}},
        callable_records=original["callable_records"],
    ).callback("utility")
    assert not answer.proved
    assert _source(original).get_direct_reads(
        cell_id=_cell_id(original), statuses={}, removal_order=()
    ).live_coordinates == frozenset({"x", "history"})


def test_native_wrapper_rejects_a_subclass_that_copies_the_native_name() -> None:
    from _lcm.regime_building.value_adapter import ValueAdapter

    copied_name = type(
        "ValueAdapter", (ValueAdapter,), {"__module__": ValueAdapter.__module__}
    )
    wrapper = copied_name(
        func=_native_history_input, decoded=MappingProxyType({}), encode=None
    )
    original, native = _native_wrapper_source(wrapper)
    answer = SourceProof(
        {"utility": {"original": native}},
        callable_records=original["callable_records"],
    ).callback("utility")
    assert not answer.proved
    assert _source(original).get_direct_reads(
        cell_id=_cell_id(original), statuses={}, removal_order=()
    ).live_coordinates == frozenset({"x", "history"})


def _changed_native_call(
    self: object,  # noqa: PAN001 - a forged native method may receive any receiver
    **kwargs: object,  # noqa: PAN001 - malformed forwarding may contain arbitrary arguments
) -> float:
    raise AssertionError("Changed native implementation must not be evaluated")


def test_native_wrapper_rejects_a_changed_actual_call_implementation() -> None:
    wrapper = _native_wrapper("value")
    with patch.object(type(wrapper), "__call__", _changed_native_call):
        original, native = _native_wrapper_source(wrapper)
    answer = SourceProof(
        {"utility": {"original": native}},
        callable_records=original["callable_records"],
    ).callback("utility")
    assert not answer.proved
    assert _source(original).get_direct_reads(
        cell_id=_cell_id(original), statuses={}, removal_order=()
    ).live_coordinates == frozenset({"x", "history"})


def test_native_wrapper_rejects_an_in_place_code_swap_with_original_source_location() -> (
    None
):
    wrapper = _native_wrapper("value")
    method = inspect.getattr_static(type(wrapper), "__call__")
    assert isinstance(method, FunctionType)
    before = method.__code__
    changed = _changed_native_call.__code__.replace(
        co_filename=before.co_filename,
        co_firstlineno=before.co_firstlineno,
        co_name=before.co_name,
        co_qualname=before.co_qualname,
    )
    method.__code__ = changed
    try:
        assert inspect.getattr_static(type(wrapper), "__call__") is method
        original, native = _native_wrapper_source(wrapper)
    finally:
        method.__code__ = before
    assert method.__code__ is before
    records, meta = _temporal_records(original, native)
    assert meta.get("native_input_source") is None
    assert _native_wrapper_inputs(native, records) is None
    answer = SourceProof(
        {"utility": {"original": native}}, callable_records=original["callable_records"]
    ).callback("utility")
    assert not answer.proved
    assert _source(original).get_direct_reads(
        cell_id=_cell_id(original), statuses={}, removal_order=()
    ).live_coordinates == frozenset({"x", "history"})


def _source_none_defaults(
    value: object = None,  # noqa: PAN001 - the source authenticator inspects arbitrary default values
    *,
    where: object = None,  # noqa: PAN001 - the source authenticator inspects arbitrary default values
) -> float:
    raise AssertionError("Source/default authentication must not execute a body")


def test_source_compiled_code_contract_accepts_only_literal_none_defaults() -> None:
    from _lcm_regime_finder.current_api_contract import inspect_source_bound_function
    import hashlib

    kwargs = {
        "module_name": __name__,
        "qualname": _source_none_defaults.__qualname__,
        "file_sha256": hashlib.sha256(
            Path(__file__).read_bytes().replace(b"\r\n", b"\n")
        ).hexdigest(),
    }
    assert (
        inspect_source_bound_function(_source_none_defaults, **kwargs)
        is _source_none_defaults
    )
    with patch.object(_source_none_defaults, "__defaults__", (False,)):
        assert inspect_source_bound_function(_source_none_defaults, **kwargs) is None
    with patch.object(_source_none_defaults, "__kwdefaults__", {"where": False}):
        assert inspect_source_bound_function(_source_none_defaults, **kwargs) is None
    before = _source_none_defaults.__code__
    changed = before.replace(co_consts=(*before.co_consts, "changed code"))
    _source_none_defaults.__code__ = changed
    try:
        assert inspect_source_bound_function(_source_none_defaults, **kwargs) is None
    finally:
        _source_none_defaults.__code__ = before
    assert _source_none_defaults.__code__ is before


def test_source_compiled_code_contract_authenticates_variadic_native_body_without_finite_inputs() -> (
    None
):
    import hashlib
    from lcm.transition import StochasticTransition
    from _lcm_regime_finder.callable_identity import _beartype_forwarding
    from _lcm_regime_finder.current_api_contract import inspect_source_bound_function

    method = inspect.getattr_static(StochasticTransition, "__call__")
    assert _beartype_interface(method) is None
    forwarding = _beartype_forwarding(method)
    assert forwarding is not None
    filename = inspect.getsourcefile(StochasticTransition)
    assert isinstance(filename, str)
    source = Path(filename)
    authored = inspect_source_bound_function(
        method,
        module_name="lcm.transition",
        qualname="StochasticTransition.__call__",
        file_sha256=hashlib.sha256(
            source.read_bytes().replace(b"\r\n", b"\n")
        ).hexdigest(),
    )
    assert authored is forwarding[0]
    assert any(
        parameter.kind in (parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD)
        for parameter in forwarding[1].parameters.values()
    )


def test_native_keywordable_default_set_matches_the_authored_interface_without_mutation() -> (
    None
):
    from lcm.transition import StochasticTransition
    from _lcm_regime_finder.callable_identity import _beartype_forwarding

    method = inspect.getattr_static(StochasticTransition, "__call__")
    defaults = method.__kwdefaults__
    assert defaults is not None
    names = defaults["__beartype_args_name_keywordable"]
    assert type(names) is set and names == {"self"}
    before = frozenset(names)
    forwarding = _beartype_forwarding(method)
    assert forwarding is not None
    assert _beartype_interface(method) is None
    assert method.__kwdefaults__ is defaults
    assert defaults["__beartype_args_name_keywordable"] is names
    assert frozenset(names) == before


class _KeywordableSetSubclass(set[str]):
    def __iter__(self) -> Iterator[str]:
        raise AssertionError("A generated-name set subclass must not be inspected")

    def __eq__(self, other: object) -> bool:
        raise AssertionError("A generated-name set subclass must not be compared")


class _KeywordableStringSubclass(str):
    def __eq__(self, other: object) -> bool:
        raise AssertionError("A generated-name string subclass must not be compared")

    __hash__ = str.__hash__


@pytest.mark.parametrize(
    "replacement",
    (
        set(),
        {"kwargs"},
        {"self", "extra"},
        {1},
        frozenset({"self"}),
        ("self",),
        _KeywordableSetSubclass({"self"}),
        {_KeywordableStringSubclass("self")},
    ),
)
def test_native_keywordable_default_rejects_changed_names_types_and_containers(
    replacement: object,  # noqa: PAN001 - malformed wrapper defaults may contain arbitrary objects
) -> None:
    from lcm.transition import StochasticTransition
    from _lcm_regime_finder.callable_identity import _beartype_forwarding

    method = inspect.getattr_static(StochasticTransition, "__call__")
    before = method.__kwdefaults__
    assert before is not None
    method.__kwdefaults__ = {**before, "__beartype_args_name_keywordable": replacement}
    try:
        assert _beartype_forwarding(method) is None
    finally:
        method.__kwdefaults__ = before
    assert method.__kwdefaults__ is before


def _source_positive_zero() -> float:
    return 0.0


def _source_integer_membership(
    value: object,  # noqa: PAN001 - source-authentication fixtures admit arbitrary comparison operands
) -> bool:
    return value in {1}


@pytest.mark.parametrize(
    ("function", "constant_type", "replacement"),
    [
        (_source_positive_zero, float, -0.0),
        (_source_integer_membership, frozenset, frozenset({True})),
        (_source_integer_membership, frozenset, frozenset({1.0})),
    ],
)
def test_source_compiled_code_rejects_changed_constant_bits_or_types(
    function, constant_type, replacement
) -> None:
    import hashlib
    from _lcm_regime_finder.current_api_contract import inspect_source_bound_function

    kwargs = {
        "module_name": __name__,
        "qualname": function.__qualname__,
        "file_sha256": hashlib.sha256(
            Path(__file__).read_bytes().replace(b"\r\n", b"\n")
        ).hexdigest(),
    }
    assert inspect_source_bound_function(function, **kwargs) is function
    before = function.__code__
    positions = [
        index
        for index, value in enumerate(before.co_consts)
        if type(value) is constant_type
    ]
    assert len(positions) == 1
    values = list(before.co_consts)
    values[positions[0]] = replacement
    function.__code__ = before.replace(co_consts=tuple(values))
    try:
        assert inspect_source_bound_function(function, **kwargs) is None
    finally:
        function.__code__ = before
    assert function.__code__ is before


def test_native_code_contract_is_bound_to_the_actual_captured_call_code() -> None:
    from _lcm_regime_finder.current_api_contract import matches_native_code

    original, native = _native_wrapper_source(_native_wrapper("value"))
    records, meta = _temporal_records(original, native)
    authority = meta.get("native_input_source")
    assert isinstance(authority, Mapping)
    assert matches_native_code(authority=authority, code_sha256=meta.get("code_sha256"))
    assert not matches_native_code(
        authority={**authority, "call_code_sha256": "0" * 64},
        code_sha256=meta.get("code_sha256"),
    )
    assert _native_wrapper_inputs(native, records) == ("history",)


def _schedule_x(x: float) -> float:
    raise AssertionError("A schedule input check must never evaluate its case")


def _schedule_variadic(**kwargs: float) -> float:
    raise AssertionError("A variadic case is not an authenticated finite interface")


def _native_schedule():
    from _lcm.regime_building.schedules import _period_dispatch

    return _period_dispatch(
        cases=(_native_history_input, _schedule_x),
        case_names=(("history",), ("x",)),
        case_by_period=(0, 1),
    )


def _schedule_answer(wrapper):
    original, native = _native_wrapper_source(cast("UserFunction", wrapper))
    records = _CallableRecords(original["callable_records"])
    assert not records.errors
    answer = SourceProof(
        {"utility": {"original": native}}, callable_records=original["callable_records"]
    ).callback("utility")
    return original, native, records, answer


@pytest.mark.parametrize("supported", [False, True])
def test_native_schedule_contract_keeps_every_case_and_period_input(
    supported: bool,
) -> None:
    schedule = _native_schedule()
    wrapper = schedule
    if supported:
        from _lcm.regime_building.transition_support import (
            _SupportedStochasticTransition,
        )

        wrapper = _SupportedStochasticTransition(func=schedule, targets=("original",))
    original, native, records, answer = _schedule_answer(wrapper)
    assert _native_wrapper_inputs(native, records) == ("period", "history", "x")
    assert answer.proved and answer.value is _UNKNOWN
    assert answer.reads == frozenset({"period", "history", "x"})
    assert _source(original).get_direct_reads(
        cell_id=_cell_id(original), statuses={}, removal_order=()
    ).live_coordinates == frozenset({"history", "x"})


@pytest.mark.parametrize("role", [False, True])
def test_native_schedule_constant_has_no_inputs_and_no_numeric_value_proof(
    role: bool,
) -> None:
    from _lcm.regime_building import schedules

    cls = getattr(
        schedules, "_ProbabilityConstant", getattr(schedules, "_Constant", None)
    )
    assert cls is not None
    wrapper = cls(value=(0.0, 1.0))
    if role:
        roles = pytest.importorskip("_lcm.output_roles")
        wrapper = roles.in_role(
            func=wrapper, role=roles.OutputRole.FLOAT, kind="Probability"
        )
    _, native, records, answer = _schedule_answer(wrapper)
    assert _native_wrapper_inputs(native, records) == ()
    assert answer.proved and answer.value is _UNKNOWN and not answer.reads


@pytest.mark.parametrize(
    "field",
    ["case_names", "case_by_period", "cases", "__signature__", "__lcm_sources__"],
)
def test_native_schedule_rejects_changed_case_tables_and_hidden_case_inputs(
    field: str,
) -> None:
    schedule = _native_schedule()
    changes = {
        "case_names": (("x",), ("x",)),
        "case_by_period": (0, 9),
        "cases": (_schedule_variadic, _schedule_x),
        "__signature__": inspect.Signature(
            parameters=[
                inspect.Parameter("period", kind=inspect.Parameter.KEYWORD_ONLY)
            ]
        ),
        "__lcm_sources__": (),
    }
    object.__setattr__(schedule, field, changes[field])
    _, native, records, answer = _schedule_answer(schedule)
    assert _native_wrapper_inputs(native, records) is None
    assert not answer.proved


def test_native_schedule_rejects_changed_actual_method_code_and_restores_it() -> None:
    schedule = _native_schedule()
    method = inspect.getattr_static(type(schedule), "__call__")
    before = method.__code__
    method.__code__ = before.replace(
        co_consts=(*before.co_consts, "changed schedule code")
    )
    try:
        _, native, records, answer = _schedule_answer(schedule)
    finally:
        method.__code__ = before
    assert method.__code__ is before
    assert _native_wrapper_inputs(native, records) is None and not answer.proved


@pytest.mark.parametrize("field", ["targets", "fixed_component", "__wrapped__"])
def test_native_supported_probability_law_rejects_changed_receiver_metadata(
    field: str,
) -> None:
    from _lcm.regime_building.transition_support import _SupportedStochasticTransition

    schedule = _native_schedule()
    wrapper = _SupportedStochasticTransition(func=schedule, targets=("original",))
    object.__setattr__(
        wrapper,
        field,
        {"targets": (), "fixed_component": (0,), "__wrapped__": _schedule_x}[field],
    )
    _, native, records, answer = _schedule_answer(wrapper)
    assert _native_wrapper_inputs(native, records) is None and not answer.proved


@pytest.mark.parametrize("field", ["cases", "case_names", "case_by_period"])
def test_native_schedule_rejects_outer_receiver_disagreement(field: str) -> None:
    original, native, _, _ = _schedule_answer(_native_schedule())
    # Keep each compact payload internally valid; their operative disagreement
    # must be rejected independently of a changed integrity digest.
    native["fields"] = dict(native["fields"])
    native["fields"][field] = None
    records = _CallableRecords(original["callable_records"])
    assert not records.errors
    assert _native_wrapper_inputs(native, records) is None


@pytest.mark.parametrize("kind", ["dispatch", "constant", "supported"])
def test_native_schedule_wire_reconciles_actual_inline_and_reference_encodings(
    kind: str,
) -> None:
    from _lcm.regime_building import schedules
    from _lcm.regime_building.transition_support import _SupportedStochasticTransition

    if kind == "constant":
        cls = getattr(
            schedules, "_ProbabilityConstant", getattr(schedules, "_Constant", None)
        )
        assert cls is not None
        wrapper = cls(value=(0.0, 1.0))
        field = "value"
    else:
        wrapper = _native_schedule()
        field = "cases"
        if kind == "supported":
            wrapper = _SupportedStochasticTransition(
                func=wrapper, targets=("original",)
            )
            field = "targets"
    _, native, records, _ = _schedule_answer(wrapper)
    meta = records.resolve(native["callable"])
    assert isinstance(native["fields"][field], list)
    receiver = meta["bound_self"]["fields"][field]
    assert receiver["kind"] == "callable-reference"
    assert records.resolve(receiver)["binding_type"] == "builtins.tuple"
    assert _native_wrapper_inputs(native, records) is not None


@pytest.mark.parametrize("replacement", [False, 0, -0.0])
def test_native_schedule_wire_preserves_scalar_types_and_signed_zero(
    replacement,
) -> None:
    from _lcm.regime_building import schedules

    cls = getattr(
        schedules, "_ProbabilityConstant", getattr(schedules, "_Constant", None)
    )
    assert cls is not None
    _, native, records, _ = _schedule_answer(cls(value=(0.0, 1.0)))
    assert _native_wrapper_inputs(native, records) == ()
    changed = deepcopy(native)
    changed["fields"]["value"][0] = replacement
    assert _native_wrapper_inputs(changed, records) is None


def test_native_schedule_wire_rejects_forged_nested_dispatch_fields() -> None:
    from _lcm.regime_building.transition_support import _SupportedStochasticTransition

    _, native, records, _ = _schedule_answer(
        _SupportedStochasticTransition(func=_native_schedule(), targets=("original",))
    )
    assert _native_wrapper_inputs(native, records) == ("period", "history", "x")
    changed = deepcopy(native)
    cases = changed["fields"]["func"]["fields"]["cases"]
    cases[0] = cases[1]
    # The nested primary callable ref is unchanged. Its contradictory duplicate
    # fields must still reject rather than disappear during wire projection.
    assert _native_wrapper_inputs(changed, records) is None


def test_native_schedule_wire_rejects_forged_nested_role_fields() -> None:
    from _lcm.regime_building.schedules import _period_dispatch

    roles = pytest.importorskip("_lcm.output_roles")
    role = roles.in_role(
        func=_native_history_input, role=roles.OutputRole.FLOAT, kind="Probability"
    )
    wrapper = _period_dispatch(
        cases=(role, _schedule_x),
        case_names=(("history",), ("x",)),
        case_by_period=(0, 1),
    )
    _, native, records, _ = _schedule_answer(wrapper)
    assert _native_wrapper_inputs(native, records) == ("period", "history", "x")
    changed = deepcopy(native)
    cases = changed["fields"]["cases"]
    cases[0]["fields"]["func"] = cases[1]
    assert _native_wrapper_inputs(changed, records) is None


def test_native_schedule_wire_never_substitutes_source_equivalent_callable_identity() -> (
    None
):
    from _lcm.regime_building.schedules import _period_dispatch

    clone = FunctionType(
        _native_history_input.__code__,
        _native_history_input.__globals__,
        name=_native_history_input.__name__,
    )
    clone.__annotations__ = dict(_native_history_input.__annotations__)
    clone.__qualname__ = _native_history_input.__qualname__
    clone.__module__ = _native_history_input.__module__
    wrapper = _period_dispatch(
        cases=(_native_history_input, clone),
        case_names=(("history",), ("history",)),
        case_by_period=(0, 1),
    )
    _, native, records, _ = _schedule_answer(wrapper)
    assert _native_wrapper_inputs(native, records) == ("period", "history")
    changed = deepcopy(native)
    cases = changed["fields"]["cases"]
    assert cases[0]["capture_ref"] != cases[1]["capture_ref"]
    cases[0] = cases[1]
    assert _native_wrapper_inputs(changed, records) is None


def test_native_schedule_wire_rejects_untracked_references_and_recursive_wire_data() -> (
    None
):
    from _lcm_regime_finder.semantic_check import _same_native_wire

    records = _CallableRecords({})
    assert not _same_native_wire(
        {"kind": "callable-reference", "capture_ref": "missing"}, None, records
    )
    assert not _same_native_wire(
        {"kind": "callable", "arguments": []},
        {"kind": "callable", "arguments": []},
        records,
    )
    cycle: list[JSONValue] = []
    cycle.append(cycle)
    assert not _same_native_wire(cycle, cycle, records)
    assert not _same_native_wire([0] * 8193, [0] * 8193, records)


def _native_wire_retyped_reference(records, reference, binding_type):
    """Reseal a genuine captured field after changing only its type label."""
    from _lcm_regime_finder.callable_identity import _digest, callable_record_digest

    table = deepcopy(records.records)
    table[reference["capture_ref"]]["binding_type"] = binding_type
    hashes = {ref: callable_record_digest(record) for ref, record in table.items()}
    graph = _digest(hashes)
    for ref, record in table.items():
        record["record_sha256"] = hashes[ref]
        record["capture_graph_sha256"] = graph
        record["identity_sha256"] = _digest(
            {
                "capture_ref": ref,
                "record_sha256": hashes[ref],
                "capture_graph_sha256": graph,
            }
        )
    pending = list(table.values())
    while pending:
        value = pending.pop()
        if isinstance(value, list):
            pending.extend(value)
        elif isinstance(value, dict):
            if value.get("kind") == "callable-reference":
                target = table[value["capture_ref"]]
                value["identity_sha256"] = target["identity_sha256"]
                value["capture_graph_sha256"] = graph
            else:
                pending.extend(value.values())
    target = table[reference["capture_ref"]]
    changed = {
        "kind": "callable-reference",
        "capture_ref": reference["capture_ref"],
        "identity_sha256": target["identity_sha256"],
        "capture_graph_sha256": graph,
    }
    checked = _CallableRecords(table)
    assert not checked.errors
    return checked, changed


@pytest.mark.parametrize(
    "binding_type",
    [
        "unsupported.NativeTuple",
        "builtins.dict",
        "inspect.Signature",
        "_lcm.output_roles.OutputRole",
    ],
)
def test_native_schedule_wire_rejects_resealed_unknown_or_mismatched_tuple_binding_type(
    binding_type: str,
) -> None:
    from _lcm.regime_building import schedules
    from _lcm_regime_finder.semantic_check import _same_native_wire

    cls = getattr(
        schedules, "_ProbabilityConstant", getattr(schedules, "_Constant", None)
    )
    assert cls is not None
    _, native, records, _ = _schedule_answer(cls(value=(0.0, 1.0)))
    meta = records.resolve(native["callable"])
    reference = meta["bound_self"]["fields"]["value"]
    outer = native["fields"]["value"]
    assert _same_native_wire(outer, reference, records)
    changed_records, changed_ref = _native_wire_retyped_reference(
        records, reference, binding_type
    )
    assert not _same_native_wire(outer, changed_ref, changed_records)


@pytest.mark.parametrize("field", ["role", "signature"])
def test_native_schedule_wire_rejects_resealed_descriptor_as_plain_mapping(
    field: str,
) -> None:
    from _lcm_regime_finder.semantic_check import _same_native_wire

    roles = pytest.importorskip("_lcm.output_roles")
    wrapper = roles.in_role(
        func=_native_history_input, role=roles.OutputRole.FLOAT, kind="Probability"
    )
    _, native, records, _ = _schedule_answer(wrapper)
    meta = records.resolve(native["callable"])
    reference = meta["bound_self"]["fields"][field]
    outer = native["fields"][field]
    assert _same_native_wire(outer, reference, records)
    changed_records, changed_ref = _native_wire_retyped_reference(
        records, reference, "builtins.dict"
    )
    assert not _same_native_wire(outer, changed_ref, changed_records)


def test_native_schedule_wire_retains_actual_value_adapter_grid_and_mapping_forms() -> (
    None
):
    from lcm import DiscreteGrid
    from _lcm.regime_building.value_adapter import ValueAdapter
    from _lcm_regime_finder.semantic_check import _same_native_wire

    grid = DiscreteGrid(values=np.array([1.0, 2.0]))
    wrapper = ValueAdapter(
        func=_native_history_input,
        decoded=MappingProxyType({"history": grid}),
        encode=grid,
    )
    _, native, records, _ = _schedule_answer(wrapper)
    meta = records.resolve(native["callable"])
    for field in ("decoded", "codes", "positions", "encode"):
        assert _same_native_wire(
            native["fields"][field], meta["bound_self"]["fields"][field], records
        )


def test_native_schedule_wire_grid_type_and_payload_spelling_must_agree() -> None:
    from lcm import DiscreteGrid
    from _lcm.regime_building.value_adapter import ValueAdapter
    from _lcm_regime_finder.semantic_check import _same_native_wire

    grid = DiscreteGrid(values=np.array([1.0, 2.0]))
    wrapper = ValueAdapter(
        func=_native_history_input,
        decoded=MappingProxyType({"history": grid}),
        encode=grid,
    )
    _, native, records, _ = _schedule_answer(wrapper)
    meta = records.resolve(native["callable"])
    reference = meta["bound_self"]["fields"]["encode"]
    actual_type = records.resolve(reference)["binding_type"]
    assert actual_type in {"lcm.grids.DiscreteGrid", "_lcm.grids.discrete.DiscreteGrid"}
    assert _same_native_wire(native["fields"]["encode"], reference, records)
    other_type = (
        "_lcm.grids.discrete.DiscreteGrid"
        if actual_type == "lcm.grids.DiscreteGrid"
        else "lcm.grids.DiscreteGrid"
    )
    for replacement in (other_type, "_lcm.grids.discrete.FakeGrid"):
        changed_records, changed_ref = _native_wire_retyped_reference(
            records, reference, replacement
        )
        assert not _same_native_wire(
            native["fields"]["encode"], changed_ref, changed_records
        )


@pytest.mark.parametrize("name", ["name_of", "_first_array", "Array"])
def test_native_supported_probability_law_rejects_rebound_conversion_helpers(
    name: str,
) -> None:
    import lcm.transition as transition
    from _lcm.regime_building.transition_support import _SupportedStochasticTransition

    dtypes = pytest.importorskip("_lcm.dtypes")
    if not hasattr(transition, "as_canonical_float"):
        pytest.skip(
            "The original profile returns probabilities without output conversion"
        )
    schedule = _native_schedule()
    wrapper = _SupportedStochasticTransition(func=schedule, targets=("original",))
    module = transition if name == "name_of" else dtypes
    with patch.object(module, name, _schedule_variadic):
        _, native, records, answer = _schedule_answer(wrapper)
    assert _native_wrapper_inputs(native, records) is None and not answer.proved


@pytest.mark.parametrize("module_name", ["numpy", "jax.numpy"])
def test_native_supported_probability_law_rejects_rebound_array_constructor(
    module_name: str,
) -> None:
    import importlib
    from _lcm.regime_building.transition_support import _SupportedStochasticTransition

    wrapper = _SupportedStochasticTransition(
        func=_native_schedule(), targets=("original",)
    )
    with patch.object(
        importlib.import_module(module_name), "asarray", _schedule_variadic
    ):
        _, native, records, answer = _schedule_answer(wrapper)
    assert _native_wrapper_inputs(native, records) is None and not answer.proved


def test_native_role_wrapper_rejects_an_unapproved_conversion_binding() -> None:
    roles = pytest.importorskip("_lcm.output_roles")
    wrapper = _native_wrapper("role")
    with patch.dict(roles._ADAPTERS, {roles.OutputRole.FLOAT: _native_history_input}):
        original, native = _native_wrapper_source(wrapper)
    answer = SourceProof(
        {"utility": {"original": native}},
        callable_records=original["callable_records"],
    ).callback("utility")
    assert not answer.proved
    assert _source(original).get_direct_reads(
        cell_id=_cell_id(original), statuses={}, removal_order=()
    ).live_coordinates == frozenset({"x", "history"})


def _native_spoofed_input(history: float) -> float:
    raise AssertionError("An advertised interface is not its implementation")


def test_native_wrapper_rejects_a_forwarded_function_with_an_unreconciled_signature() -> (
    None
):
    from _lcm.regime_building.value_adapter import ValueAdapter

    with patch.object(
        _native_spoofed_input, "__signature__", inspect.Signature(), create=True
    ):
        wrapper = cast(
            "UserFunction",
            ValueAdapter(
                func=_native_spoofed_input,
                decoded=MappingProxyType({}),
                encode=None,
            ),
        )
        original, native = _native_wrapper_source(wrapper)
    answer = SourceProof(
        {"utility": {"original": native}},
        callable_records=original["callable_records"],
    ).callback("utility")
    assert not answer.proved
    assert _source(original).get_direct_reads(
        cell_id=_cell_id(original), statuses={}, removal_order=()
    ).live_coordinates == frozenset({"x", "history"})


def _native_converter_copy() -> FunctionType:
    pytest.importorskip("_lcm.output_roles")
    from beartype import beartype
    from _lcm import dtypes

    as_bool = getattr(dtypes, "as_bool", None)

    assert isinstance(as_bool, FunctionType)
    defaults = as_bool.__kwdefaults__
    assert isinstance(defaults, dict)
    target = defaults["__beartype_func"]
    assert isinstance(target, FunctionType)
    clone = FunctionType(
        target.__code__,
        target.__globals__,
        target.__name__,
        target.__defaults__,
        target.__closure__,
    )
    clone.__qualname__, clone.__module__ = target.__qualname__, target.__module__
    clone.__kwdefaults__ = dict(target.__kwdefaults__ or {})
    clone.__annotations__ = dict(target.__annotations__)
    wrapper = beartype(conf=defaults["__beartype_conf"])(clone)
    assert isinstance(wrapper, FunctionType)
    return wrapper


def _replace_generated_type_tuple(wrapper: FunctionType, mutation: str) -> None:
    defaults = dict(wrapper.__kwdefaults__ or {})
    name, predicate = next(
        (name, value)
        for name, value in defaults.items()
        if type(value) is tuple
        and len(value) > 1
        and all(isinstance(item, type) for item in value)
        and name == "__beartype_object_" + str(id(value))
    )
    replacement = tuple(item for item in predicate)
    if mutation == "member":
        assert all(item is not object for item in predicate)
        replacement = (object, *predicate[1:])
    elif mutation == "order":
        replacement = tuple(reversed(predicate))
    new_name = "__beartype_object_" + str(id(replacement))
    if mutation == "identifier":
        new_name = "__beartype_object_0"
    defaults.pop(name)
    defaults[new_name] = replacement
    wrapper.__kwdefaults__ = defaults
    wrapper.__code__ = wrapper.__code__.replace(
        co_varnames=tuple(
            new_name if item == name else item for item in wrapper.__code__.co_varnames
        )
    )


def test_real_native_output_converters_have_authenticated_forwarding() -> None:
    pytest.importorskip("_lcm.output_roles")
    from _lcm import dtypes

    for name in ("as_bool", "as_canonical_float", "as_code"):
        function = getattr(dtypes, name, None)
        assert isinstance(function, FunctionType)
        defaults = function.__kwdefaults__
        assert isinstance(defaults, dict)
        target = defaults["__beartype_func"]
        assert isinstance(target, FunctionType)
        before = (dict(vars(target)), dict(target.__annotations__))
        interface = _beartype_interface(function)
        assert interface is not None
        assert interface[0] is target
        assert tuple(interface[1].parameters) == ("value", "source")
        assert (vars(target), target.__annotations__) == before


def test_generated_type_tuple_slot_rename_keeps_the_bound_predicate() -> None:
    wrapper = _native_converter_copy()
    _replace_generated_type_tuple(wrapper, "same")
    assert _beartype_interface(wrapper) is not None


def test_generated_slot_rename_rejects_an_extra_colliding_default_key() -> None:
    from _lcm_regime_finder.callable_identity import _beartype_default_renames

    actual = _native_converter_copy()
    reference = _native_converter_copy()
    _replace_generated_type_tuple(actual, "same")
    public_names = frozenset({"value", "source"})
    renames = _beartype_default_renames(actual, reference, public_names)
    assert renames is not None
    defaults = dict(actual.__kwdefaults__ or {})
    expected = reference.__kwdefaults__
    assert isinstance(expected, dict)
    reference_name = next(iter(renames.values()))
    assert reference_name not in defaults
    defaults[reference_name] = expected[reference_name]
    actual.__kwdefaults__ = defaults
    assert _beartype_default_renames(actual, reference, public_names) is None


@pytest.mark.parametrize(
    "mutation", ["member", "order", "identifier", "ordinary", "code"]
)
def test_generated_wrapper_slot_reconciliation_rejects_changed_bindings_or_code(
    mutation: str,
) -> None:
    wrapper = _native_converter_copy()
    _replace_generated_type_tuple(wrapper, mutation)
    if mutation == "ordinary":
        wrapper.__code__ = wrapper.__code__.replace(
            co_varnames=tuple(
                "changed_kwargs" if item == "kwargs" else item
                for item in wrapper.__code__.co_varnames
            )
        )
    elif mutation == "code":
        wrapper.__code__ = wrapper.__code__.replace(
            co_consts=(*wrapper.__code__.co_consts, "changed code constant")
        )
    assert _beartype_interface(wrapper) is None


def _temporal_input(history: float, schedule: float) -> float:
    raise AssertionError("Temporal authored callback must not be evaluated")


def _temporal_period_input(history: float, schedule: float, period: int) -> float:
    raise AssertionError("Temporal authored callback must not be evaluated")


def _temporal_other_input(history: float, other: float) -> float:
    raise AssertionError("Changed temporal callback must not be evaluated")


def _temporal_wrapper(*, passes_period: bool = False) -> UserFunction:
    from lcm.temporal import time_varying_params

    return time_varying_params("schedule")(
        _temporal_period_input if passes_period else _temporal_input
    )


def _temporal_records(
    original: dict[str, JSONValue],
    native: dict[str, JSONValue],
) -> tuple[_CallableRecords, Mapping[str, JSONValue]]:
    records = _CallableRecords(original["callable_records"])
    assert not records.errors
    meta = records.resolve(native.get("callable", native))
    assert isinstance(meta, Mapping)
    return records, meta


def test_temporal_ordinary_callbacks_skip_native_source_inspection() -> None:
    from _lcm_regime_finder import current_api_contract

    with patch.object(
        current_api_contract,
        "inspect_authority",
        side_effect=AssertionError("Ordinary callbacks must not inspect native source"),
    ):
        assert (
            current_api_contract.inspect_temporal_input_authority(_temporal_input)
            is None
        )


@pytest.mark.parametrize("passes_period", [False, True])
@pytest.mark.parametrize("declared", [False, True])
def test_native_temporal_contract_keeps_schedule_and_period_inputs(
    passes_period: bool,
    declared: bool,
) -> None:
    wrapper = _temporal_wrapper(passes_period=passes_period)
    original, native = _native_wrapper_source(wrapper, declared=declared)
    records, meta = _temporal_records(original, native)
    authority = meta.get("native_temporal_source")
    assert isinstance(authority, Mapping)
    assert (
        authority.get("template_code_contract")
        == "source-bound-native-temporal-template/v1"
    )
    assert authority.get("actual_code_sha256") == authority.get("template_code_sha256")
    assert isinstance(meta.get("temporal_target"), Mapping)
    assert _native_wrapper_inputs(native, records) == ("history", "schedule", "period")
    answer = SourceProof(
        {"temporal": {"original": native}},
        callable_records=original["callable_records"],
    ).callback("temporal")
    assert answer.proved
    assert answer.reads == frozenset({"history", "schedule", "period"})
    assert answer.value is _UNKNOWN
    assert _source(original).get_direct_reads(
        cell_id=_cell_id(original), statuses={}, removal_order=()
    ).live_coordinates == frozenset({"history"})


def test_native_value_adapter_accepts_an_authenticated_temporal_free_interface() -> (
    None
):
    from _lcm.regime_building.value_adapter import ValueAdapter

    wrapper = cast(
        "UserFunction",
        ValueAdapter(
            func=_temporal_wrapper(),
            decoded=MappingProxyType({}),
            encode=None,
        ),
    )
    original, native = _native_wrapper_source(wrapper)
    records, meta = _temporal_records(original, native)
    receiver = _object(meta["bound_self"])
    target = records.resolve(_object(receiver["fields"])["func"])
    assert isinstance(target, Mapping)
    assert isinstance(target.get("native_temporal_source"), Mapping)
    assert isinstance(target.get("temporal_target"), Mapping)
    assert _native_wrapper_inputs(native, records) == ("history", "schedule", "period")
    answer = SourceProof(
        {"temporal": {"original": native}},
        callable_records=original["callable_records"],
    ).callback("temporal")
    assert answer.proved
    assert answer.reads == frozenset({"history", "schedule", "period"})
    assert answer.value is _UNKNOWN


@pytest.mark.parametrize(
    "mutation",
    [
        "signature",
        "time_names",
        "names",
        "parameter_names",
        "defaults",
        "passes_period",
        "func",
        "wrapped",
        "code",
        "global",
    ],
)
def test_native_temporal_contract_rejects_changed_source_or_closure(
    mutation: str,
) -> None:
    import lcm.temporal as temporal

    wrapper = _temporal_wrapper()
    assert isinstance(wrapper, FunctionType)
    cells = dict(
        zip(wrapper.__code__.co_freevars, wrapper.__closure__ or (), strict=True)
    )
    if mutation == "signature":
        object.__setattr__(wrapper, "__signature__", inspect.Signature())
    elif mutation == "time_names":
        object.__setattr__(wrapper, "__lcm_time_params__", ("history",))
    elif mutation == "wrapped":
        object.__setattr__(wrapper, "__wrapped__", _temporal_other_input)
    elif mutation == "code":
        wrapper.__code__ = wrapper.__code__.replace(
            co_consts=(*wrapper.__code__.co_consts, "changed temporal code")
        )
    elif mutation != "global":
        changes = {
            "names": ("history",),
            "parameter_names": ("history", "schedule"),
            "defaults": (("schedule", 0.0),),
            "passes_period": True,
            "func": _temporal_other_input,
        }
        cells[mutation].cell_contents = changes[mutation]
    with patch.object(
        temporal, "np", object() if mutation == "global" else temporal.np
    ):
        original, native = _native_wrapper_source(wrapper)
    records, meta = _temporal_records(original, native)
    assert meta.get("native_temporal_source") is None
    assert _native_wrapper_inputs(native, records) is None
    answer = SourceProof(
        {"temporal": {"original": native}},
        callable_records=original["callable_records"],
    ).callback("temporal")
    assert not answer.proved
    assert _source(original).get_direct_reads(
        cell_id=_cell_id(original), statuses={}, removal_order=()
    ).live_coordinates == frozenset({"x", "history"})


def _fake_temporal_body(
    *args: object,  # noqa: PAN001 - a forged temporal callback may receive arbitrary arguments
    **kwargs: object,  # noqa: PAN001 - a forged temporal callback may receive arbitrary arguments
) -> float:
    raise AssertionError("A marker does not authenticate a user wrapper")


def test_temporal_marker_and_finite_signature_alone_do_not_grant_input_authority() -> (
    None
):
    genuine = _temporal_wrapper()
    fake = FunctionType(_fake_temporal_body.__code__, _fake_temporal_body.__globals__)
    object.__setattr__(fake, "__signature__", inspect.signature(genuine))
    object.__setattr__(fake, "__wrapped__", _temporal_input)
    object.__setattr__(fake, "__lcm_time_params__", ("schedule",))
    original, native = _native_wrapper_source(fake)
    records, meta = _temporal_records(original, native)
    assert meta.get("native_temporal_source") is None
    assert _native_wrapper_inputs(native, records) is None
    answer = SourceProof(
        {"temporal": {"original": native}},
        callable_records=original["callable_records"],
    ).callback("temporal")
    assert not answer.proved
    assert _source(original).get_direct_reads(
        cell_id=_cell_id(original), statuses={}, removal_order=()
    ).live_coordinates == frozenset({"x", "history"})


def test_native_schedule_read_regenerated_aliases_keep_source_bound_body() -> None:
    import hashlib
    from _lcm.regime_building import schedules
    from _lcm_regime_finder.callable_identity import _beartype_forwarding
    from _lcm_regime_finder.current_api_contract import inspect_source_bound_function

    if "_read" not in vars(schedules):
        pytest.skip("This pinned native profile has no schedules._read helper")
    _read = vars(schedules)["_read"]
    assert inspect.isfunction(_read)
    forwarding = _beartype_forwarding(_read)
    assert forwarding is not None
    authored = forwarding[0]
    filename = inspect.getsourcefile(authored)
    assert filename is not None
    path = Path(filename)
    assert (
        inspect_source_bound_function(
            _read,
            module_name="_lcm.regime_building.schedules",
            qualname="_read",
            file_sha256=hashlib.sha256(
                path.read_bytes().replace(b"\r\n", b"\n")
            ).hexdigest(),
        )
        is authored
    )


def test_exact_generic_aliases_match_origins_and_identity_argument_leaves() -> None:
    from _lcm_regime_finder.callable_identity import _same_annotation_identity

    left = GenericAlias(dict, (str, GenericAlias(tuple, (int, Ellipsis))))
    right = GenericAlias(dict, (str, GenericAlias(tuple, (int, Ellipsis))))
    assert left is not right
    assert _same_annotation_identity(left, right)
    assert _same_annotation_identity(left, left)


class _AliasEqualityTrap:
    def __eq__(self, other: object) -> bool:
        raise AssertionError("Annotation leaf equality is not authority")

    def __repr__(self) -> str:
        raise AssertionError("Annotation leaf representation is not authority")

    @property
    def __value__(self) -> object:  # noqa: PAN001 - opaque alias metadata may hold any value
        raise AssertionError("Deferred alias values must not be evaluated")


class _AliasSubclass(GenericAlias):
    def __eq__(self, other: object) -> bool:
        raise AssertionError("GenericAlias subclasses must not be compared")


def test_alias_changes_and_opaque_containers_reject_without_application_equality() -> (
    None
):
    from _lcm_regime_finder.callable_identity import _same_annotation_identity

    for left, right in (
        (GenericAlias(list, int), GenericAlias(tuple, int)),
        (GenericAlias(list, int), GenericAlias(list, str)),
        (GenericAlias(dict, (str, int)), GenericAlias(dict, (int, str))),
        (GenericAlias(list, int), (list, int)),
        (_AliasSubclass(list, int), _AliasSubclass(list, int)),
        (
            GenericAlias(list, _AliasEqualityTrap()),
            GenericAlias(list, _AliasEqualityTrap()),
        ),
        (GenericAlias(list, (tuple([int]),)), GenericAlias(list, (tuple([int]),))),
        (GenericAlias(list, [int]), GenericAlias(list, [int])),
    ):
        assert not _same_annotation_identity(left, right)
    leaf = _AliasEqualityTrap()
    assert _same_annotation_identity(GenericAlias(list, leaf), GenericAlias(list, leaf))


def test_alias_comparison_bounds_deep_wide_and_mutable_cycle_shapes() -> None:
    from _lcm_regime_finder.callable_identity import _same_annotation_identity

    left: object = int  # noqa: PAN001 - alias nesting crosses arbitrary metadata types
    right: object = int  # noqa: PAN001 - alias nesting crosses arbitrary metadata types
    for _ in range(5):
        left, right = GenericAlias(list, left), GenericAlias(list, right)
    assert _same_annotation_identity(left, right)
    for _ in range(20):
        left, right = GenericAlias(list, left), GenericAlias(list, right)
    assert not _same_annotation_identity(left, right)
    assert not _same_annotation_identity(
        GenericAlias(tuple, (int,) * 129), GenericAlias(tuple, (int,) * 129)
    )
    left_cycle: list[object] = []  # noqa: PAN001 - malformed aliases may hold arbitrary mutable objects
    right_cycle: list[object] = []  # noqa: PAN001 - malformed aliases may hold arbitrary mutable objects
    left_cycle.append(left_cycle)
    right_cycle.append(right_cycle)
    assert not _same_annotation_identity(
        GenericAlias(list, left_cycle), GenericAlias(list, right_cycle)
    )


def test_unpacked_alias_flag_is_not_erased_by_equal_origin_and_arguments() -> None:
    from _lcm_regime_finder.callable_identity import _same_annotation_identity

    ordinary = GenericAlias(tuple, (int, Ellipsis))
    unpacked = next(iter(ordinary))
    assert type(unpacked) is GenericAlias
    assert unpacked.__origin__ is ordinary.__origin__
    assert not _same_annotation_identity(ordinary, unpacked)


def _fields(
    source: dict[str, JSONValue], regime: RegimeName = "original"
) -> dict[str, JSONValue]:
    return _object(_object(_object(source["user_regimes"])[regime])["fields"])


def _normalized(source: dict[str, JSONValue]) -> dict[str, JSONValue]:
    return normalize_declarations(source)


def _source(original: dict[str, JSONValue]) -> FibreSource:
    assert _rows(_normalized(original)["cells"])
    return FibreSource(original)


def _cell_id(
    source: dict[str, JSONValue], period: int = 0, regime: RegimeName = "original"
) -> str:
    cells = [
        _object(row)
        for row in _rows(_normalized(source)["cells"])
        if _object(row)["period"] == period and _object(row)["regime"] == regime
    ]
    assert len(cells) == 1
    identifier = cells[0]["id"]
    assert isinstance(identifier, str)
    return identifier


def _relation_id(
    source: dict[str, JSONValue], *, kind: str = "transition", phase: str = "solve"
) -> str:
    cid = _cell_id(source)
    relations = [
        _object(row)
        for row in _rows(_normalized(source)["relations"])
        if _object(row)["source"] == cid
        and _object(row)["kind"] == kind
        and _object(row)["phase"] == phase
    ]
    assert len(relations) == 1
    identifier = relations[0]["id"]
    assert isinstance(identifier, str)
    return identifier


def test_source_ids_are_original_declaration_order_and_input_is_unchanged() -> None:
    original = _original()
    before = deepcopy(original)
    declarations = _normalized(original)
    source = _source(original)
    assert source.get_cell_ids() == tuple(
        _object(c)["id"] for c in _rows(declarations["cells"])
    )
    assert source.get_relation_ids() == tuple(
        _object(r)["id"] for r in _rows(declarations["relations"])
    )
    assert original == before


def test_solve_only_and_simulation_only_consumers_both_remain_live() -> None:
    original = _original()
    _object(_fields(original)["functions"])["utility"] = {
        "kind": "dataclass",
        "type": "lcm.Phased",
        "fields": {
            "solve": function("def utility(x):\n    return x\n", ["x"]),
            "simulate": function(
                "def utility(history):\n    return history\n", ["history"]
            ),
        },
    }
    result = _source(original).get_direct_reads(
        cell_id=_cell_id(original), statuses={}, removal_order=()
    )
    assert result.live_coordinates == frozenset({"x", "history"})


def test_reported_output_is_a_direct_consumer() -> None:
    original = _original()
    fields = _fields(original)
    _object(fields["functions"])["reported_history"] = function(
        "def reported_history(history):\n    return history\n", ["history"]
    )
    fields["output_only_functions"] = ["reported_history"]
    result = _source(original).get_direct_reads(
        cell_id=_cell_id(original), statuses={}, removal_order=()
    )
    assert result.live_coordinates == frozenset({"x", "history"})


def test_transitive_function_and_category_code_alias_keep_the_state() -> None:
    original = _original()
    functions = _object(_fields(original)["functions"])
    functions["utility"] = function(
        "def utility(intermediate):\n    return intermediate\n", ["intermediate"]
    )
    functions["intermediate"] = function(
        "def intermediate(history_code):\n    return history_code\n", ["history_code"]
    )
    result = _source(original).get_direct_reads(
        cell_id=_cell_id(original), statuses={}, removal_order=()
    )
    assert result.live_coordinates == frozenset({"history"})


@pytest.mark.parametrize("registered", [True, False])
def test_only_registered_conditional_shock_laws_are_direct_consumers(
    registered: bool,
) -> None:
    original = _original()
    _fields(original)["shocks"] = ["offers"] if registered else []
    original["shocks"] = {
        "offers": {
            "kind": "dataclass",
            "type": "lcm.shocks.ShockGroup",
            "fields": {
                "shocks": {
                    "offer": {
                        "kind": "dataclass",
                        "type": "lcm.shocks.Discrete",
                        "fields": {
                            "probs": function(
                                "def probs(history):\n    return history\n", ["history"]
                            )
                        },
                    }
                }
            },
        }
    }
    result = _source(original).get_direct_reads(
        cell_id=_cell_id(original), statuses={}, removal_order=()
    )
    assert result.live_coordinates == frozenset(
        {"x", "history"} if registered else {"x"}
    )


@pytest.mark.parametrize(
    ("status", "expected"),
    [(0, frozenset({"history"})), (1, frozenset({"history", "x"}))],
)
def test_status_branch_only_removes_ordinary_state_with_joint_proof(
    status: int, expected: frozenset[str]
) -> None:
    original = _original()
    _object(_fields(original)["functions"])["utility"] = function(
        "def utility(history, x):\n    return 0 if history == 0 else x\n",
        ["history", "x"],
    )
    result = _source(original).get_direct_reads(
        cell_id=_cell_id(original),
        statuses={"history": status},
        removal_order=("history",),
    )
    assert result.live_coordinates == expected
    fibre = {"history": status, "x": 0}
    for history in (0, 1):
        for wealth in (0, 1):
            candidate = {"history": history, "x": wealth}
            if all(candidate[k] == fibre[k] for k in result.live_coordinates):
                assert (0 if history == 0 else wealth) == (0 if status == 0 else 0)


def test_date_specialization_uses_original_period_and_keeps_both_branches_where_needed() -> (
    None
):
    original = _original()
    _object(_fields(original)["functions"])["utility"] = function(
        "def utility(history, x, period):\n    return x if period == 0 else history\n",
        ["history", "x", "period"],
    )
    source = _source(original)
    assert source.get_direct_reads(
        cell_id=_cell_id(original, 0), statuses={}, removal_order=()
    ).live_coordinates == frozenset({"x"})
    assert source.get_direct_reads(
        cell_id=_cell_id(original, 1), statuses={}, removal_order=()
    ).live_coordinates == frozenset({"history"})


def test_opaque_source_retains_all_states_and_smaller_producer_reads_are_not_authority() -> (
    None
):
    original = _original()
    callback = function("def utility(x):\n    return unknown_operation(x)\n", ["x"])
    callback.update(reads=[], expression="0", status="proved")
    _object(_fields(original)["functions"])["utility"] = callback
    result = _source(original).get_direct_reads(
        cell_id=_cell_id(original), statuses={}, removal_order=()
    )
    assert result.live_coordinates == frozenset({"x", "history"})


@pytest.mark.parametrize("missing", [False, True])
def test_update_reads_follow_original_law_and_missing_law_is_conservative(
    missing: bool,
) -> None:
    original = _original()
    transitions = _object(_fields(original)["state_transitions"])
    if missing:
        transitions.pop("x")
    else:
        transitions["x"] = function(
            "def next_x(history):\n    return history\n", ["history"]
        )
    result = _source(original).get_update_reads(
        relation_id=_relation_id(original),
        target_coordinate="x",
        statuses={},
        removal_order=(),
    )
    assert result.live_coordinates == frozenset(
        {"x", "history"} if missing else {"history"}
    )


def test_same_period_value_read_incidence_preserves_projector_inputs() -> None:
    original = _original()
    _fields(original)["same_period_refs"] = {
        "current": {
            "kind": "dataclass",
            "type": "lcm.ProjectedRegimeValue",
            "fields": {
                "regime": "original",
                "kind": "Raw",
                "field": "value",
                "observes": [],
                "projection": {
                    "x": function(
                        "def project(history):\n    return history\n", ["history"]
                    ),
                    "history": function(
                        "def keep(history):\n    return history\n", ["history"]
                    ),
                },
            },
        }
    }
    source = _source(original)
    relation = _relation_id(original, kind="value_read")
    assert relation in source.get_relation_ids()
    assert source.get_update_reads(
        relation_id=relation, target_coordinate="x", statuses={}, removal_order=()
    ).live_coordinates == frozenset({"history"})


@pytest.mark.parametrize(
    ("stochastic", "expected"),
    [
        (False, TargetSupport(frozenset({0}), True)),
        (True, TargetSupport(frozenset({0, 1}), False)),
    ],
)
def test_stochastic_scalar_zero_is_not_a_deterministic_target_code(
    stochastic: bool, expected: TargetSupport
) -> None:
    original = _original()
    callback = function("def next_x():\n    return 0\n", [])
    _object(_fields(original)["state_transitions"])["x"] = (
        {
            "kind": "dataclass",
            "type": "lcm.StochasticTransition",
            "fields": {"func": callback},
        }
        if stochastic
        else callback
    )
    result = _source(original).get_target_support(
        relation_id=_relation_id(original), target_coordinate="x", statuses={}
    )
    assert result == expected


def test_unknown_target_support_keeps_entire_original_grid() -> None:
    original = _original()
    _object(_fields(original)["state_transitions"])["x"] = function(
        "def next_x(x):\n    return unknown_operation(x)\n", ["x"]
    )
    result = _source(original).get_target_support(
        relation_id=_relation_id(original), target_coordinate="x", statuses={}
    )
    assert result == TargetSupport(frozenset({0, 1}), False)


@pytest.mark.parametrize(
    ("rectangle", "excluded"),
    [
        ({"x": (0,), "history": (0,)}, frozenset({0})),
        ({"x": (0, 1), "history": (0, 1)}, frozenset()),
    ],
)
def test_action_exclusion_checks_full_rectangle_including_nonmember_points(
    rectangle: Mapping[str, tuple[int, ...]], excluded: frozenset[int]
) -> None:
    original = _original()
    _fields(original)["constraints"] = {
        "allowed": function(
            "def allowed(choice, x, history):\n    return (choice != 0) | (x != history)\n",
            ["choice", "x", "history"],
        )
    }
    result = _source(original).get_excluded_action_values(
        cell_id=_cell_id(original), action="choice", status_values=rectangle
    )
    assert result == excluded
    for action in result:
        assert all(
            not ((action != 0) or (x != history))
            for x in rectangle["x"]
            for history in rectangle["history"]
        )
    member_fibres = ((0, 0), (1, 1))
    assert all(not ((0 != 0) or (x != history)) for x, history in member_fibres)
    assert (0 != 0) or (0 != 1)


def test_other_actions_ordinary_states_and_runtime_inputs_stay_free_for_exclusion() -> (
    None
):
    original = _original()
    _object(_fields(original)["actions"])["other"] = grid([0, 1])
    _fields(original)["constraints"] = {
        "allowed": function(
            "def allowed(choice, other, x, beta):\n    return (choice != 0) | (other == 1) | (x == 1) | (beta > 0)\n",
            ["choice", "other", "x", "beta"],
        )
    }
    result = _source(original).get_excluded_action_values(
        cell_id=_cell_id(original), action="choice", status_values={"history": (0,)}
    )
    assert result == frozenset()


@pytest.mark.parametrize(
    "defect",
    [
        "schema",
        "cell",
        "relation",
        "target_coordinate",
        "status_coordinate",
        "status_code",
        "action",
        "rectangle_code",
        "empty_rectangle",
    ],
)
def test_malformed_original_or_lookup_fails_clearly(defect: str) -> None:
    original = _original()
    if defect == "schema":
        original["schema"] = "producer-declarations"
        with pytest.raises(ValueError):
            FibreSource(original)
        return
    source = _source(original)
    with pytest.raises(ValueError):
        if defect == "cell":
            source.get_direct_reads(cell_id="missing", statuses={}, removal_order=())
        elif defect == "relation":
            source.get_update_reads(
                relation_id="missing",
                target_coordinate="x",
                statuses={},
                removal_order=(),
            )
        elif defect == "target_coordinate":
            source.get_update_reads(
                relation_id=_relation_id(original),
                target_coordinate="missing",
                statuses={},
                removal_order=(),
            )
        elif defect in {"status_coordinate", "status_code"}:
            statuses = (
                {"missing": 0} if defect == "status_coordinate" else {"history": 7}
            )
            source.get_direct_reads(
                cell_id=_cell_id(original),
                statuses=statuses,
                removal_order=tuple(statuses),
            )
        else:
            source.get_excluded_action_values(
                cell_id=_cell_id(original),
                action="missing" if defect == "action" else "choice",
                status_values={"history": () if defect == "empty_rectangle" else (7,)},
            )


@pytest.mark.parametrize("composed", [False, True])
def test_update_draws_use_each_target_and_phase_law_context(composed: bool) -> None:
    conditions = {
        "target_a": ("a_solve", "a_simulate"),
        "target_b": ("b_solve", "b_simulate"),
    }
    original = _target_process_capture(
        source_condition="source_history",
        target_conditions=conditions,
        composed=composed,
    )
    declarations = _normalized(original)
    cells = {
        _name(_object(cell)["id"]): _object(cell)
        for cell in _rows(declarations["cells"])
    }
    source = _source(original)
    matched = 0
    for raw in _rows(declarations["relations"]):
        relation = _object(raw)
        if relation["kind"] != "transition":
            continue
        targets = _rows(relation["targets"])
        assert len(targets) == 1
        target_id = targets[0]
        assert isinstance(target_id, str)
        target = cells[target_id]["regime"]
        assert isinstance(target, str)
        condition = conditions[target][0 if relation["phase"] == "solve" else 1]
        relation_id = relation["id"]
        assert isinstance(relation_id, str)
        result = source.get_update_reads(
            relation_id=relation_id,
            target_coordinate="x" if composed else "z",
            statuses={},
            removal_order=(),
        )
        assert result.live_coordinates == frozenset({condition})
        assert "source_history" not in result.live_coordinates
        matched += 1
    assert matched == 4


def test_raising_original_callback_is_inspected_without_execution() -> None:
    original = _original()
    _object(_fields(original)["functions"])["utility"] = function(
        'def utility(x, history):\n    raise AssertionError("Economic callback executed")\n',
        ["x", "history"],
    )
    result = _source(original).get_direct_reads(
        cell_id=_cell_id(original), statuses={}, removal_order=()
    )
    assert result.live_coordinates == frozenset({"x", "history"})


@pytest.mark.parametrize(
    ("fixed", "expected"),
    [
        ({}, TargetSupport(frozenset({0, 1}), False)),
        ({"history": 1}, TargetSupport(frozenset({1}), True)),
    ],
)
def test_typed_identity_support_uses_only_the_fixed_original_status(
    fixed: Mapping[str, int], expected: TargetSupport
) -> None:
    original = _original()
    _object(_fields(original)["state_transitions"])["history"] = {
        "kind": "object",
        "type": "_lcm.identity_transition._IdentityTransition",
        "fields": {"state_name": "history"},
    }
    result = _source(original).get_target_support(
        relation_id=_relation_id(original), target_coordinate="history", statuses=fixed
    )
    assert result == expected


def test_missing_update_does_not_invent_identity_target_support() -> None:
    original = _original()
    _object(_fields(original)["state_transitions"]).pop("history")
    result = _source(original).get_target_support(
        relation_id=_relation_id(original),
        target_coordinate="history",
        statuses={"history": 1},
    )
    assert result == TargetSupport(frozenset({0, 1}), False)


def test_numeric_discrete_action_values_are_distinct_from_excluded_codes() -> None:
    original = _original()
    _object(_fields(original)["actions"])["choice"] = grid([10.0, 20.0])
    _fields(original)["constraints"] = {
        "allowed": function(
            "def allowed(choice_code):\n    return choice_code != 0\n", ["choice_code"]
        )
    }
    result = _source(original).get_excluded_action_values(
        cell_id=_cell_id(original), action="choice", status_values={}
    )
    assert result == frozenset({0})


def _stored_category_source(
    *, storage: str, duplicate_values: bool = False
) -> tuple[dict[str, JSONValue], Reader]:
    codes = [7, 11, 19]
    values = [2.5, 2.5 if duplicate_values else 5.5, 9.5]
    roots: dict[
        str,
        list[int]
        | list[float]
        | NDArray[np.int64 | np.uint64 | np.float32 | np.float64],
    ] = {"codes": codes, "values": values}
    if storage != "sequence":
        array_codes = np.array(codes, dtype=">i8")
        array_values = np.array(values, dtype=">f8")
        if storage == "reversed":
            array_codes = np.array(codes[::-1], dtype=">i8")[::-1]
            array_values = np.array(values[::-1], dtype=">f8")[::-1]
        elif storage == "strided":
            array_codes = np.array([7, 0, 11, 0, 19, 0], dtype="<u8")[::2]
            array_values = np.repeat(np.array(values, dtype="<f4"), 2)[::2]
        roots = {"codes": array_codes, "values": array_values}
    anchor = source_anchor(
        roots,
        source_context={"fixture": "categorical-code-value-payload"},
        transaction={"kind": "exclusive-owner", "authority": "test-owned roots"},
    )
    reader = Reader(NativeWriter().finish(roots, anchor))
    reader.decode()
    compare_source(reader, anchor)
    register(reader)
    references = {
        name: {
            "kind": "structural-reference",
            "store_root": reader.doc["root_digest"],
            "root": name,
            **descriptor(reader, token),
        }
        for name, token in named_roots(reader).items()
    }
    original = _original()
    original["structural_store"] = {
        "schema": "regime-structural-capture/v1",
        "manifest": reader.doc,
        "source_comparison": "PASS",
        "complete_roots": list(roots),
    }
    _object(_fields(original)["states"])["history"] = {
        "kind": "object",
        "type": "lcm.DiscreteGrid",
        "fields": {**references, "categories": ["low", "middle", "high"]},
    }
    return original, reader


@pytest.mark.parametrize("storage", ["sequence", "native", "strided", "reversed"])
def test_checked_category_payload_binds_original_values_and_target_codes(
    storage: str,
) -> None:
    original, reader = _stored_category_source(storage=storage)
    _object(_fields(original)["functions"])["utility"] = function(
        "def utility(history, x):\n    return 0 if history == 5.5 else x\n",
        ["history", "x"],
    )
    source = _source(original)
    cell = _cell_id(original)
    assert source.get_direct_reads(
        cell_id=cell, statuses={"history": 11}, removal_order=("history",)
    ).live_coordinates == frozenset({"history"})
    assert source.get_target_support(
        relation_id=_relation_id(original),
        target_coordinate="history",
        statuses={"history": 11},
    ) == TargetSupport(frozenset({11}), True)
    report = source.get_report()
    assert not any(
        "payload" in str(item).lower() for item in _rows(report["source_obligations"])
    )
    assert report["transformation_authorized"] is False
    assert reader.doc["status"] == "COMPLETE"


@pytest.mark.parametrize("stored", [False, True])
def test_fixed_nominal_code_is_not_recovered_from_ambiguous_numeric_value(
    stored: bool,
) -> None:
    original = _original()
    reader = None
    if stored:
        original, reader = _stored_category_source(
            storage="native", duplicate_values=True
        )
    else:
        _object(_fields(original)["states"])["history"] = grid(
            [2.5, 2.5, 9.5], codes=[7, 11, 19]
        )
    _object(_fields(original)["functions"])["utility"] = function(
        "def utility(history_code, x):\n    return 0 if history_code == 11 else x\n",
        ["history_code", "x"],
    )
    assert _source(original).get_direct_reads(
        cell_id=_cell_id(original),
        statuses={"history": 11},
        removal_order=("history",),
    ).live_coordinates == frozenset({"history"})
    assert reader is None or reader.doc["status"] == "COMPLETE"


def test_action_exclusion_keeps_codes_when_numeric_action_values_repeat() -> None:
    original = _original()
    _object(_fields(original)["actions"])["choice"] = grid(
        [2.5, 2.5, 9.5], codes=[7, 11, 19]
    )
    _fields(original)["constraints"] = {
        "allowed": function(
            "def allowed(choice_code):\n    return choice_code != 11\n", ["choice_code"]
        )
    }
    assert _source(original).get_excluded_action_values(
        cell_id=_cell_id(original), action="choice", status_values={}
    ) == frozenset({11})


def test_decoded_category_values_survive_the_capture_reader_lifetime() -> None:
    original, reader = _stored_category_source(storage="reversed")
    source = _source(original)
    relation = _relation_id(original)
    del original, reader
    gc.collect()
    assert source.get_target_support(
        relation_id=relation,
        target_coordinate="history",
        statuses={"history": 11},
    ) == TargetSupport(frozenset({11}), True)


def test_stored_category_descriptor_stays_bound_to_its_verified_payload() -> None:
    original, reader = _stored_category_source(storage="native")
    states = _object(_fields(original)["states"])
    fields = _object(_object(states["history"])["fields"])
    _object(fields["values"])["dtype"] = "<i8"
    with pytest.raises(ValueError, match="descriptor differs"):
        _source(original)
    assert reader.doc["status"] == "COMPLETE"


@pytest.mark.parametrize("phased", [False, True])
def test_stored_category_class_without_numeric_values_uses_its_original_codes(
    phased: bool,
) -> None:
    original, reader = _stored_category_source(storage="native")
    states = _object(_fields(original)["states"])
    category = _object(states["history"])
    _object(category["fields"])["values"] = None
    if phased:
        states["history"] = {
            "kind": "dataclass",
            "type": "lcm.Phased",
            "fields": {"solve": category, "simulate": category},
        }
    _object(_fields(original)["functions"])["utility"] = function(
        "def utility(history_code, x):\n    return 0 if history_code == 11 else x\n",
        ["history_code", "x"],
    )
    source = _source(original)
    assert source.get_direct_reads(
        cell_id=_cell_id(original),
        statuses={"history": 11},
        removal_order=("history",),
    ).live_coordinates == frozenset({"history"})
    assert source.get_target_support(
        relation_id=_relation_id(original),
        target_coordinate="history",
        statuses={"history": 11},
    ) == TargetSupport(frozenset({11}), True)
    assert reader.doc["status"] == "COMPLETE"


@pytest.mark.parametrize("mutation", ["payload", "semantics"])
def test_stored_code_as_value_replay_rejects_forged_normalized_semantics(
    mutation: str,
) -> None:
    original, reader = _stored_category_source(storage="native")
    category = _object(_object(_fields(original)["states"])["history"])
    fields = _object(category["fields"])
    wrong_values = fields["values"]
    fields["values"] = None
    normalized = _normalized(original)
    cell = _object(_rows(normalized["cells"])[0])
    identity = _name(_object(cell["states"])["history"])
    domain = _object(_object(normalized["domains"])[identity])
    if mutation == "payload":
        domain["structural_values"] = wrong_values
    else:
        domain["value_semantics"] = "invented_numeric_values"
    with (
        patch(
            "_lcm_regime_finder.fibre_source.normalize_declarations",
            return_value=normalized,
        ),
        pytest.raises(ValueError, match="Original capture authentication failed"),
    ):
        FibreSource(original)
    assert reader.doc["status"] == "COMPLETE"


@pytest.mark.parametrize("field", ["structural_values", "structural_codes", "size"])
def test_phased_grid_replay_rejects_a_forged_copy_of_the_verified_child(
    field: str,
) -> None:
    original, reader = _stored_category_source(storage="native")
    states = _object(_fields(original)["states"])
    category = states["history"]
    states["history"] = {
        "kind": "dataclass",
        "type": "lcm.Phased",
        "fields": {"solve": category, "simulate": category},
    }
    normalized = _normalized(original)
    cell = _object(_rows(normalized["cells"])[0])
    domain = _object(
        _object(normalized["domains"])[_name(_object(cell["states"])["history"])]
    )
    domain[field] = (
        4
        if field == "size"
        else _object(_object(category)["fields"])[
            "values" if field == "structural_codes" else "codes"
        ]
    )
    with (
        patch(
            "_lcm_regime_finder.fibre_source.normalize_declarations",
            return_value=normalized,
        ),
        pytest.raises(ValueError, match="original inherited grid payload changed"),
    ):
        FibreSource(original)
    assert reader.doc["status"] == "COMPLETE"


@pytest.mark.parametrize("field", ["size", "conditional_declaration"])
def test_conditional_action_replay_rejects_changed_grid_or_activation_metadata(
    field: str,
) -> None:
    original = _original()
    _object(_fields(original)["actions"])["savings"] = {
        "kind": "object",
        "type": "lcm.ConditionalAction",
        "fields": {
            "grid": {
                "kind": "object",
                "type": "lcm.LinSpacedGrid",
                "fields": {
                    "start": 0.0,
                    "stop": 1.0,
                    "n_points": 3,
                },
            },
            "active_when": {"choice": [0]},
            "inactive_value": 0.0,
            "inactive_index": 0,
        },
    }
    FibreSource(original)
    normalized = _normalized(original)
    cell = _object(_rows(normalized["cells"])[0])
    domain = _object(
        _object(normalized["domains"])[_name(_object(cell["actions"])["savings"])]
    )
    domain[field] = 2 if field == "size" else {}
    with (
        patch(
            "_lcm_regime_finder.fibre_source.normalize_declarations",
            return_value=normalized,
        ),
        pytest.raises(
            ValueError,
            match="original (inherited grid payload|conditional action declaration) changed",
        ),
    ):
        FibreSource(original)


def test_native_capture_identity_and_raising_callbacks_are_metadata_only() -> None:
    script = textwrap.dedent(
        """
        from lcm import DiscreteGrid, InitialNodes, Model, Periods, Regime, categorical, fixed_transition
        from lcm.typing import DiscreteState, FloatND, ScalarInt
        from _lcm_regime_finder.declarations import normalize_declarations
        from _lcm_regime_finder.fibre_source import FibreSource, TargetSupport
        from _lcm_regime_finder.model_snapshot import read_model

        @categorical(ordered=False)
        class Binary:
            low: ScalarInt
            high: ScalarInt

        @categorical(ordered=False)
        class Regimes:
            alive: ScalarInt
            final: ScalarInt

        def utility(x: DiscreteState, history: DiscreteState) -> FloatND:
            raise AssertionError("Economic callback executed")

        model = Model(
            n_periods=2,
            regime_id_class=Regimes,
            regimes={"alive": Regime(
                states={"x": DiscreteGrid(Binary), "history": DiscreteGrid(Binary)},
                functions={"utility": utility},
                state_transitions={"x": fixed_transition("x"), "history": fixed_transition("history")},
            ), "final": Regime(
                states={"x": DiscreteGrid(Binary), "history": DiscreteGrid(Binary)},
                functions={"utility": utility},
            )},
            initial_nodes=InitialNodes(by_period={0: "alive"}),
            edges={"alive": {"final": Periods(values=(0,))}},
            enable_jit=False,
        )
        original = read_model(model)
        declarations = normalize_declarations(original)
        source = FibreSource(original)
        cell = next(row for row in declarations["cells"] if row["period"] == 0)
        relation = next(row for row in declarations["relations"] if row["source"] == cell["id"] and row["phase"] == "solve")
        assert source.get_direct_reads(cell_id=cell["id"], statuses={}, removal_order=()).live_coordinates == frozenset({"x", "history"})
        assert source.get_target_support(relation_id=relation["id"], target_coordinate="history", statuses={"history": 1}) == TargetSupport(frozenset({1}), True)
        """
    )
    command = [
        "pixi",
        "run",
        "--as-is",
        "--manifest-path",
        str(Path(__file__).resolve().parents[2] / "pyproject.toml"),
        "-e",
        os.environ.get("PIXI_ENVIRONMENT_NAME", "type-checking"),
        "python",
        "-c",
        script,
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    assert completed.returncode == 0, completed.stdout + completed.stderr


@pytest.mark.parametrize("slot", ["utility", "update"])
def test_unknown_process_namespace_does_not_grant_helper_input_authority(
    slot: str,
) -> None:
    original = _original()
    unknown: dict[str, JSONValue] = {
        "kind": "dataclass",
        "type": "_lcm.processes.CustomUnspecifiedProcess",
        "fields": {"helper": function("def helper(x):\n    return x\n", ["x"])},
    }
    if slot == "utility":
        _object(_fields(original)["functions"])["utility"] = unknown
    else:
        _object(_fields(original)["state_transitions"])["x"] = unknown
    source = _source(original)
    result = (
        source.get_direct_reads(
            cell_id=_cell_id(original), statuses={}, removal_order=()
        )
        if slot == "utility"
        else source.get_update_reads(
            relation_id=_relation_id(original),
            target_coordinate="x",
            statuses={},
            removal_order=(),
        )
    )
    assert result.live_coordinates == frozenset({"x", "history"})


@pytest.mark.parametrize(
    ("simulation_values", "expected"),
    [([0], frozenset({"history"})), ([0, 1], frozenset({"history", "x"}))],
)
@pytest.mark.parametrize("statuses", [{}, {"history": 1, "x": 0}])
def test_solve_imputation_reads_history_without_replacing_realised_carried_state(
    simulation_values: list[int], expected: frozenset[str], statuses: Mapping[str, int]
) -> None:
    original = _original()
    fields = _fields(original)
    _object(fields["states"])["x"] = {
        "kind": "dataclass",
        "type": "lcm.Phased",
        "fields": {
            "solve": function(
                "def imputed(history):\n    return history\n", ["history"]
            ),
            "simulate": grid(simulation_values),
        },
    }
    _object(fields["functions"])["utility"] = function(
        "def utility(x):\n    return x\n", ["x"]
    )
    result = _source(original).get_direct_reads(
        cell_id=_cell_id(original), statuses=statuses, removal_order=tuple(statuses)
    )
    assert result.live_coordinates == expected


@pytest.mark.parametrize("missing_update", [False, True])
def test_cell_facts_preserve_original_order_and_literal_multistep_dependencies(
    missing_update: bool,
) -> None:
    original = _original()
    transitions = _object(_fields(original)["state_transitions"])
    if missing_update:
        transitions.pop("x")
    else:
        transitions["x"] = function(
            "def next_x(history):\n    return history\n", ["history"]
        )
    declarations = _normalized(original)
    source = _source(original)
    facts = source.get_cell_facts()
    original_ids = tuple(
        _name(_object(cell)["id"]) for cell in _rows(declarations["cells"])
    )
    assert tuple(fact.identifier for fact in facts) == original_ids
    assert all(fact.coordinates == frozenset({"x", "history"}) for fact in facts)
    assert all(fact.direct.live_coordinates == frozenset({"x"}) for fact in facts)
    assert all(fact.direct.removal_order == () for fact in facts)
    expected = tuple(
        FibreLiveness(
            identifier,
            frozenset({"x", "history"} if index < len(original_ids) - 1 else {"x"}),
        )
        for index, identifier in enumerate(original_ids)
    )
    assert calculate_fibre_liveness(facts) == expected
    first = facts[0]
    assert len(first.links) == 2
    assert {link.phase for link in first.links} == {"solve", "simulation"}
    for link in first.links:
        updates = {row.coordinate: row.reads.live_coordinates for row in link.updates}
        assert updates == {
            "x": frozenset({"x", "history"} if missing_update else {"history"}),
            "history": frozenset({"history"}),
        }


def test_joint_and_ordinary_producers_form_one_complete_link_per_phase() -> None:
    original = _original()
    fields = _fields(original)
    _object(fields["state_transitions"]).pop("x")
    fields["joint_transitions"] = {
        "original": {
            "lottery": {
                "kind": "dataclass",
                "type": "lcm.JointTransition",
                "fields": {
                    "support_size": 2,
                    "support": [0, 1],
                    "probabilities": function(
                        "def probabilities():\n    return [0.5, 0.5]\n", []
                    ),
                    "outputs": {
                        "x": function(
                            "def draw(history, lottery):\n    return history\n",
                            ["history", "lottery"],
                        )
                    },
                },
            }
        }
    }
    declarations = _normalized(original)
    source = _source(original)
    first = source.get_cell_facts()[0]
    assert len(first.links) == 2
    assert {link.phase for link in first.links} == {"solve", "simulation"}
    for link in first.links:
        expected_relations = tuple(
            _name(_object(row)["id"])
            for row in _rows(declarations["relations"])
            if _object(row)["source"] == first.identifier
            and _object(row)["targets"] == [link.target]
            and _object(row)["phase"] == link.phase
        )
        assert link.source_relations == expected_relations
        assert len(link.source_relations) == 2
        assert {update.coordinate for update in link.updates} == {"x", "history"}
        assert {
            update.coordinate: update.reads.live_coordinates for update in link.updates
        } == {"x": frozenset({"history"}), "history": frozenset({"history"})}
    assert calculate_fibre_liveness(source.get_cell_facts())[
        0
    ].live_coordinates == frozenset({"x", "history"})


def test_cell_links_preserve_target_phase_specific_process_certificates() -> None:
    conditions = {
        "target_a": ("a_solve", "a_simulate"),
        "target_b": ("b_solve", "b_simulate"),
    }
    original = _target_process_capture(
        source_condition="source_history", target_conditions=conditions
    )
    declarations = _normalized(original)
    cells = {
        _name(_object(cell)["id"]): _object(cell)
        for cell in _rows(declarations["cells"])
    }
    facts = _source(original).get_cell_facts()
    source = next(
        fact for fact in facts if cells[fact.identifier]["regime"] == "source"
    )
    assert len(source.links) == 4
    for link in source.links:
        target = cells[link.target]
        regime = _name(target["regime"])
        assert link.phase in {"solve", "simulation"}
        expected_condition = conditions[regime][0 if link.phase == "solve" else 1]
        updates = {
            update.coordinate: update.reads.live_coordinates for update in link.updates
        }
        assert set(updates) == set(_object(target["states"]))
        assert updates["z"] == frozenset({expected_condition})
        assert all(
            not reads for coordinate, reads in updates.items() if coordinate != "z"
        )
    expected_source_live = frozenset(
        {"z", "a_solve", "a_simulate", "b_solve", "b_simulate"}
    )
    assert (
        next(
            row
            for row in calculate_fibre_liveness(facts)
            if row.identifier == source.identifier
        ).live_coordinates
        == expected_source_live
    )


@pytest.mark.parametrize("kind", ["value_read", "exit"])
def test_current_value_and_exit_targets_keep_explicit_projector_links(
    kind: str,
) -> None:
    original = _original()
    target: dict[str, JSONValue] = {
        "kind": "dataclass",
        "type": "lcm.ExitTarget" if kind == "exit" else "lcm.ProjectedRegimeValue",
        "fields": {
            "regime": "original",
            "kind": "Raw",
            "field": "value",
            "observes": [],
            "projection": {
                "x": function(
                    "def project(history):\n    return history\n", ["history"]
                ),
                "history": function(
                    "def keep(history):\n    return history\n", ["history"]
                ),
            },
        },
    }
    if kind == "exit":
        _fields(original)["exits"] = {
            "leave": {
                "kind": "dataclass",
                "type": "lcm.Exit",
                "fields": {
                    "targets": {"self": target},
                    "rule": "unilateral",
                    "observes": [],
                },
            }
        }
    else:
        _fields(original)["same_period_refs"] = {"current": target}
    declarations = _normalized(original)
    facts = _source(original).get_cell_facts()
    first = facts[0]
    desired = {
        _name(_object(row)["id"])
        for row in _rows(declarations["relations"])
        if _object(row)["source"] == first.identifier
        and _object(row)["kind"] == kind
        and _object(row)["targets"]
    }
    assert desired
    selected = tuple(
        link for link in first.links if set(link.source_relations) & desired
    )
    assert {rid for link in selected for rid in link.source_relations} == desired
    assert all(link.target == first.identifier for link in selected)
    assert all(
        {update.coordinate for update in link.updates} == {"x", "history"}
        for link in selected
    )
    assert all(
        next(
            update.reads.live_coordinates
            for update in link.updates
            if update.coordinate == "x"
        )
        == frozenset({"history"})
        for link in selected
    )
    assert calculate_fibre_liveness(facts)[0].live_coordinates == frozenset(
        {"x", "history"}
    )


def test_fallback_is_a_value_link_but_gate_computation_is_not_a_self_link() -> None:
    original = captured_gate_graph()
    declarations = _normalized(original)
    source = _source(original)
    facts = source.get_cell_facts()
    by_relation = {
        _name(_object(row)["id"]): _object(row)
        for row in _rows(declarations["relations"])
    }
    linked_relations = {
        rid for fact in facts for link in fact.links for rid in link.source_relations
    }
    fallback_ids = {
        rid for rid, row in by_relation.items() if row["kind"] == "fallback"
    }
    assert fallback_ids and fallback_ids <= linked_relations
    assert all(
        by_relation[rid]["kind"] not in {"gate", "arrival"} for rid in linked_relations
    )
    assert all(link.target != fact.identifier for fact in facts for link in fact.links)
    landing = next(
        fact
        for fact in facts
        if any(
            by_relation[rid]["kind"] == "fallback"
            for link in fact.links
            for rid in link.source_relations
        )
    )
    assert landing.direct.live_coordinates == frozenset({"history"})


def test_imputation_stays_a_provider_instead_of_an_artificial_self_value_link() -> None:
    original = _original()
    _object(_fields(original)["states"])["x"] = {
        "kind": "dataclass",
        "type": "lcm.Phased",
        "fields": {
            "solve": function(
                "def imputed(history):\n    return history\n", ["history"]
            ),
            "simulate": grid([0, 1]),
        },
    }
    declarations = _normalized(original)
    facts = _source(original).get_cell_facts()
    imputation_ids = {
        _name(_object(row)["id"])
        for row in _rows(declarations["relations"])
        if _object(row)["kind"] == "imputation"
    }
    assert imputation_ids
    assert all(
        not (set(link.source_relations) & imputation_ids)
        for fact in facts
        for link in fact.links
    )
    assert all(link.target != fact.identifier for fact in facts for link in fact.links)
    assert all(
        fact.direct.live_coordinates == frozenset({"history", "x"}) for fact in facts
    )


def test_query_only_target_qualification_retains_every_original_source_coordinate() -> (
    None
):
    original = policy_snapshot()
    declarations = _normalized(original)
    source = _source(original)
    facts = source.get_cell_facts()
    assert any(
        "policy query" in qualification for qualification in source.source_obligations
    )
    assert tuple(fact.identifier for fact in facts) == source.get_cell_ids()
    assert all(
        fact.direct.live_coordinates == frozenset(_object(_object(cell)["states"]))
        for fact, cell in zip(facts, _rows(declarations["cells"]), strict=True)
    )
    query_ids = {
        _name(_object(row)["id"])
        for row in _rows(declarations["relations"])
        if _object(row)["kind"] == "policy_query"
    }
    assert query_ids
    assert all(
        not (set(link.source_relations) & query_ids)
        for fact in facts
        for link in fact.links
    )


def test_new_target_coordinate_has_an_explicit_constant_update() -> None:
    original = _original()
    original["n_periods"] = 2
    fields = _fields(original)
    fields["functions"] = {"utility": function("def utility():\n    return 0\n", [])}
    fields["state_transitions"] = {
        "fresh": function("def fresh():\n    return 0\n", [])
    }
    _object(original["user_regimes"])["target"] = {
        "kind": "dataclass",
        "type": "lcm.Regime",
        "fields": {
            "states": {"fresh": grid([0, 1])},
            "actions": {},
            "constraints": {},
            "functions": {
                "utility": function(
                    "def utility(fresh):\n    return fresh\n", ["fresh"]
                )
            },
            "state_transitions": {},
        },
    }
    reach = _object(_object(original["reachability"])["fields"])
    reach["nodes"] = [[0, "original"], [1, "target"]]
    for phase in ("solution", "simulation"):
        _object(_object(reach[phase])["fields"])["targets_by_period"] = [
            {"original": ["target"]}
        ]
    before = deepcopy(original)
    facts = _source(original).get_cell_facts()
    assert isinstance(facts, tuple)
    assert len(facts) == 2
    assert facts[0].direct.live_coordinates == frozenset()
    assert all(
        link.updates[0].coordinate == "fresh"
        and link.updates[0].reads.live_coordinates == frozenset()
        for link in facts[0].links
    )
    assert calculate_fibre_liveness(facts) == (
        FibreLiveness(facts[0].identifier, frozenset()),
        FibreLiveness(facts[1].identifier, frozenset({"fresh"})),
    )
    assert original == before


@pytest.mark.parametrize("period", [0, 1])
@pytest.mark.parametrize(
    "clock", ["half_year_ages", "integer_ages", "periods", "periods_free_age"]
)
def test_date_specialization_uses_numeric_original_age_or_actual_period(
    clock: str, period: int
) -> None:
    original = _original()
    original["n_periods"] = 2
    dates: list[JSONValue]
    if clock == "half_year_ages":
        dates = [{"fraction": "1/2"}, {"fraction": "3/2"}]
    else:
        dates = [0, 1]
    uses_age_grid = clock in {"half_year_ages", "integer_ages"}
    original["ages"] = (
        {"kind": "object", "type": "lcm.AgeGrid", "fields": {"exact_values": dates}}
        if uses_age_grid
        else None
    )
    graph = _object(_object(original["graph"])["fields"])
    graph["coordinate_kind"] = "age" if uses_age_grid else "period"
    graph["initial_nodes"] = [[dates[0], "original"]]
    reach = _object(_object(original["reachability"])["fields"])
    reach["nodes"] = [[date, "original"] for date in dates]
    for phase in ("solution", "simulation"):
        _object(_object(reach[phase])["fields"])["targets_by_period"] = [
            {"original": ["original"]}
        ]
    argument = "period" if clock == "periods" else "age"
    _object(_fields(original)["functions"])["utility"] = function(
        f"def utility(x, history, {argument}):\n    return x if {argument} < 1 else history\n",
        ["x", "history", argument],
    )
    declarations = _normalized(original)
    cell = next(
        _object(row)
        for row in _rows(declarations["cells"])
        if _object(row)["period"] == period
    )
    if clock == "half_year_ages":
        assert cell["age"] == ("1/2", "3/2")[period]
        assert isinstance(cell["age"], str)
    elif clock == "integer_ages":
        assert cell["age"] == period and type(cell["age"]) is int
    else:
        assert cell["age"] is None
    result = _source(original).get_direct_reads(
        cell_id=_name(cell["id"]), statuses={}, removal_order=()
    )
    expected = frozenset(
        {"x", "history"}
        if clock == "periods_free_age"
        else {"x"}
        if period == 0
        else {"history"}
    )
    assert result.live_coordinates == expected


def _report_original(*, numeric_values: bool = False) -> dict[str, JSONValue]:
    original: dict[str, JSONValue] = snapshot()
    _fields(original)["constraints"] = {
        "allowed": function(
            "def allowed(choice_code, period):\n    return (choice_code == 0) | (period < 2)\n",
            ["choice_code", "period"],
        )
    }
    if numeric_values:
        _object(_fields(original)["states"])["x"] = grid([10.0, 20.0])
        _object(_fields(original)["actions"])["choice"] = grid([10.0, 20.0])
        _object(_fields(original)["state_transitions"])["x"] = function(
            "def next_x():\n    return 20.0\n", []
        )
    return original


def test_source_report_binds_joint_status_screen_to_original_primitives() -> None:
    original = _original()
    before = deepcopy(original)
    report = _source(original).get_report(status_axes={"original": ("history",)})
    screen = _object(report["status_screen"])
    assert screen["conditions_pass"] is True
    assert screen["raw_lanes"] == 2
    assert screen["projected_lanes"] == 1
    assert screen["forcing"] == "no forced-action facts"
    assert screen["transformation_authorized"] is False
    assert original == before


def test_source_report_keeps_full_grid_action_exclusion_counterexample() -> None:
    original = _original()
    _fields(original)["constraints"] = {
        "allowed": function(
            "def allowed(history, choice):\n    return choice == history\n",
            ["history", "choice"],
        )
    }
    screen = _object(
        _source(original).get_report(status_axes={"original": ("history",)})[
            "status_screen"
        ]
    )
    assert screen["conditions_pass"] is False
    assert _object(_object(screen["premises"])["I2"])["status"] == "refuted"
    assert screen["projected_lanes"] is None


def test_status_projection_requires_every_original_part_reachable() -> None:
    original = _original()
    _object(_fields(original)["state_transitions"])["history"] = function(
        "def next_history():\n    return 0\n", []
    )
    screen = _object(
        _source(original).get_report(status_axes={"original": ("history",)})[
            "status_screen"
        ]
    )
    premises = _object(screen["premises"])
    assert _object(premises["I1"])["status"] == "refuted"
    assert _object(premises["I3"])["status"] == "proved"
    assert screen["conditions_pass"] is False
    assert screen["projected_lanes"] is None


@pytest.mark.parametrize(
    "axes",
    (
        {"missing": ("history",)},
        {"original": ("missing",)},
        {"original": ("history", "history")},
    ),
)
def test_source_report_rejects_unknown_or_repeated_status_axes(
    axes: dict[str, tuple[str, ...]],
) -> None:
    with pytest.raises(ValueError):
        _source(_original()).get_report(status_axes=axes)


def _assert_source_report(
    report: dict[str, JSONValue],
    original: dict[str, JSONValue],
    *,
    conservative_update: bool = False,
) -> None:
    assert report["schema"] == "regime-source-facts/v1"
    assert report["status"] == "SOURCE_FACTS_BUILT"
    assert report["transformation_authorized"] is False
    assert isinstance(json.dumps(report, allow_nan=False), str)
    declarations = _normalized(original)
    cells = tuple(_object(row) for row in _rows(report["cells"]))
    assert tuple(row["id"] for row in cells) == tuple(
        _object(row)["id"] for row in _rows(declarations["cells"])
    )
    assert tuple(_object(row)["id"] for row in _rows(report["relations"])) == tuple(
        _object(row)["id"] for row in _rows(declarations["relations"])
    )
    exclusions = _object(report["excluded_action_values"])
    for cell, declared in zip(cells, _rows(declarations["cells"]), strict=True):
        assert cell["direct_reads"] == ["x"]
        has_successor = any(
            _object(row)["source"] == cell["id"] and bool(_object(row)["targets"])
            for row in _rows(declarations["relations"])
        )
        assert cell["live_coordinates"] == (
            ["history", "x"] if conservative_update and has_successor else ["x"]
        )
        period = _object(declared)["period"]
        assert isinstance(period, int)
        expected = [1] if period >= 2 else []
        assert _object(exclusions[_name(cell["id"])])["choice"] == expected
    assert all(isinstance(item, str) for item in _rows(report["source_obligations"]))
    assert "recommendation" not in report
    assert "optimality_certificate" not in report


def test_source_report_is_json_safe_original_facts_without_layout_search() -> None:
    from _lcm_regime_finder import frontier, production

    original = _report_original()
    before = deepcopy(original)
    with (
        patch.object(
            frontier,
            "search_frontier",
            side_effect=AssertionError("Source facts must not run a layout search"),
        ),
        patch.object(
            production.MetadataAnalysis,
            "run",
            side_effect=AssertionError("Source facts must not run an optimizer"),
        ),
    ):
        report = _source(original).get_report()
    _assert_source_report(report, original)
    assert original == before


@pytest.mark.parametrize("stochastic", [False, True])
def test_source_report_keeps_nominal_codes_and_unknown_categorical_support(
    stochastic: bool,
) -> None:
    original = _report_original(numeric_values=True)
    if stochastic:
        _object(_fields(original)["state_transitions"])["x"] = {
            "kind": "dataclass",
            "type": "lcm.StochasticTransition",
            "fields": {"func": function("def probabilities():\n    return 0\n", [])},
        }
    report = _source(original).get_report()
    _assert_source_report(report, original, conservative_update=stochastic)
    first = next(
        _object(row)
        for row in _rows(report["relations"])
        if _object(row)["id"] == _relation_id(original)
    )
    support = _object(_object(first["categorical_support"])["x"])
    assert support["possible_codes"] == ([0, 1] if stochastic else [1])
    assert support["proved"] is (not stochastic)


def test_source_report_preserves_query_qualification_and_conservative_live_states() -> (
    None
):
    original = policy_snapshot()
    report = _source(original).get_report()
    assert report["status"] == "SOURCE_FACTS_BUILT"
    assert report["transformation_authorized"] is False
    assert any(
        "policy query" in _name(item) for item in _rows(report["source_obligations"])
    )
    assert all(
        set(_rows(_object(cell)["live_coordinates"])) == {"x", "history"}
        for cell in _rows(report["cells"])
    )


def test_source_report_uses_each_targets_original_categorical_domain() -> None:
    original = _report_original()
    original["n_periods"] = 2
    regimes = _object(original["user_regimes"])
    for target, values in (("small", [0, 1]), ("large", [10, 20, 30])):
        regimes[target] = deepcopy(regimes["original"])
        _object(_fields(original, target)["states"])["x"] = grid(values)
    _fields(original)["state_transitions"] = {
        "x": {
            "small": function("def next_x():\n    return 0\n", []),
            "large": function("def next_x():\n    return 20\n", []),
        },
        "history": function(
            "def next_history(history):\n    return history\n", ["history"]
        ),
    }
    reach = _object(_object(original["reachability"])["fields"])
    reach["nodes"] = [[0, "original"], [1, "small"], [1, "large"]]
    for phase in ("solution", "simulation"):
        _object(_object(reach[phase])["fields"])["targets_by_period"] = [
            {"original": ["small", "large"]}
        ]
    report = _source(original).get_report()
    cells = {_name(_object(row)["id"]): _object(row) for row in _rows(report["cells"])}
    checked = set()
    for row in _rows(report["relations"]):
        relation = _object(row)
        if "x" not in _object(relation["categorical_support"]):
            continue
        targets = _rows(relation["targets"])
        assert len(targets) == 1
        target = _name(cells[_name(targets[0])]["regime"])
        support = _object(_object(relation["categorical_support"])["x"])
        assert support["proved"] is True
        assert support["possible_codes"] == ([0] if target == "small" else [1])
        checked.add((target, _name(relation["phase"])))
    assert checked == {
        (target, phase)
        for target in ("small", "large")
        for phase in ("solve", "simulation")
    }
    assert report["transformation_authorized"] is False


def test_source_report_qualifies_uninterpretable_categorical_payload() -> None:
    original = _report_original()
    unavailable: JSONValue = {"kind": "opaque", "type": "UnavailableCategoricalPayload"}
    for slot, coordinate in (("states", "x"), ("actions", "choice")):
        domain = _object(_object(_fields(original)[slot])[coordinate])
        _object(domain["fields"])["values"] = unavailable
    report = _source(original).get_report()
    assert report["status"] == "SOURCE_FACTS_BUILT"
    assert report["transformation_authorized"] is False
    assert any(
        "payload needs interpretation" in _name(item).lower()
        for item in _rows(report["source_obligations"])
    )
    for cell in _rows(report["cells"]):
        assert (
            _object(
                _object(report["excluded_action_values"])[_name(_object(cell)["id"])]
            )["choice"]
            == []
        )
    supports = tuple(
        _object(_object(_object(row)["categorical_support"])["x"])
        for row in _rows(report["relations"])
    )
    assert supports
    assert all(
        item["proved"] is False and "target_domains" in item for item in supports
    )
    assert all("possible_codes" not in item for item in supports)


def test_native_first_array_has_source_authenticated_alias_reference_forwarding() -> (
    None
):
    from _lcm import dtypes
    from _lcm_regime_finder.callable_identity import _beartype_forwarding

    wrapper = vars(dtypes).get("_first_array")
    if wrapper is None:
        pytest.skip("This pinned native profile has no _first_array helper")
    assert type(wrapper) is FunctionType
    defaults = wrapper.__kwdefaults__
    assert type(defaults) is dict
    target = defaults["__beartype_func"]
    assert type(target) is FunctionType
    annotations = target.__annotations__
    before = dict(annotations)
    attributes = dict(vars(target))
    interface = _beartype_forwarding(wrapper)
    assert interface is not None
    assert interface[0] is target
    assert tuple(interface[1].parameters) == ("value",)
    assert target.__annotations__ is annotations
    assert annotations.keys() == before.keys()
    assert all(annotations[key] is value for key, value in before.items())
    assert vars(target).keys() == attributes.keys()
    assert all(vars(target)[key] is value for key, value in attributes.items())


def _native_first_array_metadata() -> tuple[
    ModuleType, ModuleType, FunctionType, FunctionType, TypeAliasType, FunctionType
]:
    from _lcm import dtypes
    from lcm import typing as native_typing

    wrapper = vars(dtypes).get("_first_array")
    if wrapper is None:
        pytest.skip("This pinned native profile has no _first_array helper")
    assert type(wrapper) is FunctionType
    defaults = wrapper.__kwdefaults__
    assert type(defaults) is dict
    target = defaults["__beartype_func"]
    assert type(target) is FunctionType
    alias = vars(native_typing)["ValueND"]
    assert isinstance(alias, TypeAliasType)
    evaluator = alias.evaluate_value
    assert type(evaluator) is FunctionType
    return dtypes, native_typing, wrapper, target, alias, evaluator


def _must_not_evaluate_native_alias(_format: int = 1) -> None:
    raise AssertionError("Unauthenticated alias evaluator body was called")


def test_native_first_array_return_expands_only_its_source_bound_alias_metadata() -> (
    None
):
    from _lcm_regime_finder.current_api_contract import (
        inspect_native_first_array_return,
    )

    _, _, wrapper, target, alias, _ = _native_first_array_metadata()
    before = dict(target.__annotations__)
    result = inspect_native_first_array_return(wrapper, target)
    assert result is not None
    original_hint, expanded_hint = result
    assert isinstance(original_hint, UnionType) and isinstance(expanded_hint, UnionType)
    assert original_hint is before["return"]
    assert original_hint.__args__[1] is alias
    assert expanded_hint.__args__[0] is original_hint.__args__[0]
    assert isinstance(expanded_hint.__args__[1], type)
    assert expanded_hint.__args__[1] is alias.__value__
    assert target.__annotations__.keys() == before.keys()
    assert all(target.__annotations__[name] is value for name, value in before.items())


@pytest.mark.parametrize(
    "mutation",
    ("body", "literal", "format_default", "keyword_default", "positional_interface"),
)
def test_native_alias_evaluator_protocol_mutations_reject_before_body_execution(
    mutation: str,
) -> None:
    from _lcm_regime_finder.current_api_contract import (
        inspect_native_first_array_return,
    )

    _, _, wrapper, target, _, evaluator = _native_first_array_metadata()
    code, defaults, keyword = (
        evaluator.__code__,
        evaluator.__defaults__,
        evaluator.__kwdefaults__,
    )
    try:
        if mutation == "body":
            evaluator.__code__ = _must_not_evaluate_native_alias.__code__
        elif mutation == "literal":
            assert any(
                type(value) is str and value == "..." for value in code.co_consts
            )
            evaluator.__code__ = code.replace(
                co_consts=tuple(
                    "changed_metadata_shape"
                    if type(value) is str and value == "..."
                    else value
                    for value in code.co_consts
                )
            )
        elif mutation == "format_default":
            evaluator.__defaults__ = (True,)
        elif mutation == "keyword_default":
            evaluator.__kwdefaults__ = {"format": 1}
        else:
            evaluator.__code__ = code.replace(co_posonlyargcount=0)
        result = inspect_native_first_array_return(wrapper, target)
    finally:
        evaluator.__code__, evaluator.__defaults__, evaluator.__kwdefaults__ = (
            code,
            defaults,
            keyword,
        )
    assert result is None
    assert (
        evaluator.__code__ is code
        and evaluator.__defaults__ is defaults
        and evaluator.__kwdefaults__ is keyword
    )


class _NativeAliasEqualityTrap:
    def __eq__(self, _other: object) -> bool:
        raise AssertionError("Alias authority used application equality")

    def __repr__(self) -> str:
        raise AssertionError("Alias authority rendered an unknown value")


@pytest.mark.parametrize(
    "owner,name", (("typing", "Shaped"), ("typing", "Array"), ("dtypes", "ValueND"))
)
def test_native_alias_global_rebinding_rejects_without_interpreting_unknown_objects(
    owner: str, name: str
) -> None:
    from _lcm_regime_finder.current_api_contract import (
        inspect_native_first_array_return,
    )

    dtypes, native_typing, wrapper, target, _, _ = _native_first_array_metadata()
    module = native_typing if owner == "typing" else dtypes
    before = vars(module)[name]
    try:
        setattr(module, name, _NativeAliasEqualityTrap())
        result = inspect_native_first_array_return(wrapper, target)
    finally:
        setattr(module, name, before)
    assert result is None
    assert vars(module)[name] is before


@pytest.mark.parametrize(
    "mutation", ("return_order", "return_member", "unknown_alias", "parametric_alias")
)
def test_native_first_array_reference_rejects_unbound_or_changed_return_aliases(
    mutation: str,
) -> None:
    import typing
    from operator import or_
    from typing import TypeAliasType, TypeVar
    from _lcm_regime_finder.current_api_contract import (
        inspect_native_first_array_return,
    )

    dtypes, native_typing, wrapper, target, alias, _ = _native_first_array_metadata()
    annotations = target.__annotations__
    before = dict(annotations)
    try:
        if mutation == "return_order":
            annotations["return"] = or_(alias, np.ndarray)
        elif mutation == "return_member":
            annotations["return"] = or_(object, alias)
        else:
            # These are runtime counterexample objects, not static type aliases.
            assert vars(typing)["TypeAliasType"] is TypeAliasType
            assert vars(typing)["TypeVar"] is TypeVar
            params: tuple[TypeVar, ...] = ()
            if mutation == "parametric_alias":
                parameter: TypeVar = vars(typing)["TypeVar"]("T")
                params = (parameter,)
            fake: TypeAliasType = vars(typing)["TypeAliasType"](
                "ValueND",
                object,
                type_params=params,
            )
            vars(native_typing)["ValueND"] = vars(dtypes)["ValueND"] = fake
            annotations["return"] = or_(np.ndarray, fake)
        result = inspect_native_first_array_return(wrapper, target)
    finally:
        vars(native_typing)["ValueND"] = vars(dtypes)["ValueND"] = alias
        annotations.clear()
        annotations.update(before)
    assert result is None
    assert all(annotations[name] is value for name, value in before.items())


class _NativeCachedShapeTrap(str):
    def __eq__(self, _other: object) -> bool:
        raise AssertionError("Cached native shape validation used application equality")

    def __repr__(self) -> str:
        raise AssertionError("Cached native shape validation rendered an unknown value")


@pytest.mark.parametrize(
    "field,mutation",
    (
        ("_skip_instancecheck", "extra"),
        ("dtype", "object"),
        ("array_type", "object"),
        ("dim_str", "wrong_shape"),
        ("dim_str", "opaque_shape"),
        ("dtypes", "object"),
        ("dims", "empty_dims"),
        ("dims", "wrong_dimension"),
        ("index_variadic", "boolean_index"),
        ("index_variadic", "wrong_index"),
        ("__module__", "wrong_module"),
        ("__doc__", "extra_document"),
    ),
)
def test_cached_native_value_nd_class_rejects_mutated_or_extra_predicate_metadata(
    field: str, mutation: str
) -> None:
    from _lcm_regime_finder.current_api_contract import (
        inspect_native_first_array_return,
    )

    _, _, wrapper, target, alias, _ = _native_first_array_metadata()
    authenticated = inspect_native_first_array_return(wrapper, target)
    assert authenticated is not None
    # Lookup is permitted only after the exact native evaluator auth succeeded.
    cached = alias.__value__
    assert isinstance(cached, type)
    fields = type.__getattribute__(cached, "__dict__")
    assert isinstance(fields, MappingProxyType)
    existed = field in fields
    before = fields.get(field)
    if mutation == "extra":
        assert not existed
        replacement: object = True  # noqa: PAN001 - Malformed cache fields intentionally carry heterogeneous sentinel values.
    elif mutation == "wrong_shape":
        replacement = "_"
    elif mutation == "opaque_shape":
        replacement = _NativeCachedShapeTrap("...")
    elif mutation == "empty_dims":
        replacement = ()
    elif mutation == "wrong_dimension":
        replacement = (object(),)
    elif mutation == "boolean_index":
        replacement = False
    elif mutation == "wrong_index":
        replacement = 1
    elif mutation == "wrong_module":
        replacement = "not_the_native_array_module"
    elif mutation == "extra_document":
        replacement = "not_a_generated_array_class"
    else:
        replacement = object()
    try:
        setattr(cached, field, replacement)
        result = inspect_native_first_array_return(wrapper, target)
    finally:
        if existed:
            setattr(cached, field, before)
        else:
            delattr(cached, field)
    assert result is None
    final = type.__getattribute__(cached, "__dict__")
    assert isinstance(final, MappingProxyType)
    assert (field in final) is existed
    if existed:
        assert final[field] is before


def test_explicit_builtin_union_operator_preserves_native_class_operand_order() -> None:
    import typing
    from operator import or_

    forward = type.__or__(int, str)
    reverse = type.__or__(str, int)
    assert isinstance(forward, UnionType) and isinstance(reverse, UnionType)
    assert forward.__args__[0] is int and forward.__args__[1] is str
    assert reverse.__args__[0] is str and reverse.__args__[1] is int
    ordinary = or_(int, str)
    assert isinstance(ordinary, UnionType)
    assert ordinary.__args__[0] is int and ordinary.__args__[1] is str
    assert vars(typing)["TypeAliasType"] is TypeAliasType
    alias: TypeAliasType = vars(typing)["TypeAliasType"]("UnionOperatorControl", int)
    for left, right in ((alias, np.ndarray), (np.ndarray, alias), (object, alias)):
        union = or_(left, right)
        assert isinstance(union, UnionType)
        assert union.__args__[0] is left and union.__args__[1] is right


@beartype
def _source_builtin_slice_fixture(value: FloatND) -> int:
    return value[:, 0].size


def _source_plain_slice_fixture(value: FloatND) -> int:
    return value[:, 0].size


def _changed_compiler_slices(
    value: object,  # noqa: PAN001 - compiler constants can include deliberately unsupported application objects
) -> object:  # noqa: PAN001 - preserve arbitrary untouched compiler constants in this trust-boundary fixture
    if type(value) is slice:
        return slice(1, value.stop, value.step)
    if type(value) is tuple:
        return tuple(_changed_compiler_slices(item) for item in value)
    return value


@pytest.mark.parametrize(
    "callback", (_source_builtin_slice_fixture, _source_plain_slice_fixture)
)
def test_source_compiled_code_contract_accepts_real_builtin_slice_constants(
    callback: Callable[[FloatND], int],
) -> None:
    import hashlib
    from _lcm_regime_finder.current_api_contract import (
        _native_constant_equal,
        inspect_source_bound_function,
    )

    interface = _beartype_interface(callback)
    target = interface[0] if interface is not None else callback
    assert isinstance(target, FunctionType)
    source_hash = hashlib.sha256(
        Path(__file__).read_bytes().replace(b"\r\n", b"\n")
    ).hexdigest()
    assert (
        inspect_source_bound_function(
            target,
            module_name=__name__,
            qualname=target.__qualname__,
            file_sha256=source_hash,
        )
        is target
    )
    before = target.__code__
    changed_constants = tuple(
        _changed_compiler_slices(value) for value in before.co_consts
    )
    assert not _native_constant_equal(before.co_consts, changed_constants)
    target.__code__ = before.replace(co_consts=changed_constants)
    try:
        assert (
            inspect_source_bound_function(
                target,
                module_name=__name__,
                qualname=target.__qualname__,
                file_sha256=source_hash,
            )
            is None
        )
    finally:
        target.__code__ = before
    assert target.__code__ is before


@pytest.mark.parametrize(
    ("left", "right", "expected"),
    (
        (slice(None), slice(None), True),
        (slice(None, 3, 1), slice(None, 3, 1), True),
        (slice(None, 3, 1), slice(None, 4, 1), False),
        (slice(True, None), slice(1, None), False),
        (slice(1.0, None), slice(1, None), False),
        (slice(-0.0, None), slice(0.0, None), False),
        ((slice(None), 0), (slice(None), 0), True),
        ((slice(None), True), (slice(None), 1), False),
    ),
)
def test_source_constant_slice_comparison_preserves_exact_types_and_bits(
    left: slice | tuple[slice, int],
    right: slice | tuple[slice, int],
    expected: bool,
) -> None:
    from _lcm_regime_finder.current_api_contract import _native_constant_equal

    assert _native_constant_equal(left, right) is expected


class _SliceEqualityPoison:
    def __eq__(self, other: object) -> bool:
        raise AssertionError(
            "Compiler constant authentication must not call application equality"
        )


def test_source_constant_slices_reject_unknown_components_without_equality() -> None:
    from _lcm_regime_finder.current_api_contract import _native_constant_equal

    poison = _SliceEqualityPoison()
    assert not _native_constant_equal(slice(poison), slice(poison))
    assert not _native_constant_equal(slice([1]), slice([1]))
    assert not _native_constant_equal(slice(None), poison)


def _plain_stochastic_probability(history: float) -> FloatND:
    raise AssertionError("A probability input check must never evaluate its law")


def _plain_stochastic_other_probability(x: float) -> FloatND:
    raise AssertionError("A changed probability interface must never be evaluated")


@pytest.mark.parametrize("declared", [False, True])
def test_plain_stochastic_forwarder_retains_all_inputs_and_no_output_proof(
    declared: bool,
) -> None:
    from lcm.transition import StochasticTransition

    original, native = _native_wrapper_source(
        StochasticTransition(func=_plain_stochastic_probability), declared=declared
    )
    records = _CallableRecords(original["callable_records"])
    assert not records.errors
    meta = records.resolve(native["callable"])
    assert isinstance(meta, Mapping)
    assert isinstance(meta.get("native_input_source"), Mapping)
    assert _native_wrapper_inputs(native, records) == ("history",)
    answer = SourceProof(
        {"law": {"original": native}}, callable_records=records
    ).callback("law")
    assert answer.proved and answer.value is _UNKNOWN
    assert answer.reads == frozenset({"history"})


def test_plain_stochastic_capture_to_target_update_avoids_unrelated_state_fallback() -> (
    None
):
    from lcm.transition import StochasticTransition

    original, native = _native_wrapper_source(
        StochasticTransition(func=_plain_stochastic_probability)
    )
    _object(_fields(original)["state_transitions"])["history"] = native
    source = _source(original)
    cell_id = _cell_id(original)
    relation = next(
        row
        for row in source._relations.values()
        if row["source"] == cell_id and row["phase"] == "simulation"
    )
    answer = source.get_update_reads(
        relation_id=_name(relation["id"]),
        target_coordinate="history",
        statuses={},
        removal_order=(),
    )
    assert answer.live_coordinates == frozenset({"history"})


def test_plain_stochastic_forwarder_retains_every_nested_schedule_input() -> None:
    from lcm.transition import StochasticTransition

    _, native, records, answer = _schedule_answer(
        StochasticTransition(func=_native_schedule())
    )
    assert _native_wrapper_inputs(native, records) == ("period", "history", "x")
    assert answer.proved and answer.value is _UNKNOWN
    assert answer.reads == frozenset({"period", "history", "x"})


@pytest.mark.parametrize("field", ["func", "__wrapped__", "__signature__"])
def test_plain_stochastic_forwarder_rejects_mismatched_receiver_interfaces(
    field: str,
) -> None:
    from lcm.transition import StochasticTransition

    wrapper = StochasticTransition(func=_plain_stochastic_probability)
    replacement = (
        inspect.Signature(
            parameters=[inspect.Parameter("x", kind=inspect.Parameter.KEYWORD_ONLY)]
        )
        if field == "__signature__"
        else _plain_stochastic_other_probability
    )
    object.__setattr__(wrapper, field, replacement)
    _, native, records, answer = _schedule_answer(wrapper)
    assert _native_wrapper_inputs(native, records) is None and not answer.proved


def _plain_stochastic_hidden_input(history: float, hidden: float) -> FloatND:
    raise AssertionError("Hidden probability operands must never be evaluated")


def test_plain_stochastic_forwarder_rejects_a_hidden_authored_input() -> None:
    from lcm.transition import StochasticTransition

    before = vars(_plain_stochastic_hidden_input).copy()
    vars(_plain_stochastic_hidden_input)["__signature__"] = inspect.Signature(
        parameters=[inspect.Parameter("history", kind=inspect.Parameter.KEYWORD_ONLY)]
    )
    try:
        _, native, records, answer = _schedule_answer(
            StochasticTransition(func=_plain_stochastic_hidden_input)
        )
    finally:
        vars(_plain_stochastic_hidden_input).clear()
        vars(_plain_stochastic_hidden_input).update(before)
    assert vars(_plain_stochastic_hidden_input) == before
    assert _native_wrapper_inputs(native, records) is None and not answer.proved


def test_plain_stochastic_forwarder_rejects_a_native_subclass_override() -> None:
    from lcm.transition import StochasticTransition

    class ChangedTransition(StochasticTransition):
        def __call__(
            self,
            *args: object,  # noqa: PAN001 - native override accepts arbitrary positional operands
            **kwargs: object,  # noqa: PAN001 - native override accepts arbitrary keyword operands
        ) -> FloatND:
            raise AssertionError("Subclass implementations must not be evaluated")

    _, native, records, answer = _schedule_answer(
        ChangedTransition(func=_plain_stochastic_probability)
    )
    assert _native_wrapper_inputs(native, records) is None and not answer.proved


def test_plain_stochastic_forwarder_rejects_changed_live_method_code_and_restores_it() -> (
    None
):
    from lcm.transition import StochasticTransition

    wrapper = StochasticTransition(func=_plain_stochastic_probability)
    method = inspect.getattr_static(StochasticTransition, "__call__")
    before = method.__code__
    method.__code__ = before.replace(
        co_consts=(*before.co_consts, "changed native stochastic body")
    )
    try:
        _, native, records, answer = _schedule_answer(wrapper)
    finally:
        method.__code__ = before
    assert method.__code__ is before
    assert _native_wrapper_inputs(native, records) is None and not answer.proved


@pytest.mark.parametrize("field", ["func", "fixed_component"])
def test_plain_stochastic_forwarder_rejects_outer_receiver_field_disagreement(
    field: str,
) -> None:
    from lcm.transition import StochasticTransition

    _, native, records, _ = _schedule_answer(
        StochasticTransition(func=_plain_stochastic_probability)
    )
    changed = deepcopy(native)
    _object(changed["fields"])[field] = None if field == "func" else [0]
    assert not records.errors
    assert _native_wrapper_inputs(changed, records) is None


@pytest.mark.parametrize("name", ["as_canonical_float", "name_of", "_first_array"])
def test_plain_stochastic_forwarder_rejects_rebound_conversion_functions(
    name: str,
) -> None:
    import lcm.transition as transition

    dtypes = pytest.importorskip("_lcm.dtypes")
    if not hasattr(transition, "as_canonical_float"):
        pytest.skip("The original native profile has no output conversion")
    wrapper = transition.StochasticTransition(func=_plain_stochastic_probability)
    module = dtypes if name == "_first_array" else transition
    with patch.object(module, name, _schedule_variadic):
        _, native, records, answer = _schedule_answer(wrapper)
    assert _native_wrapper_inputs(native, records) is None and not answer.proved


def _phase_target_update_snapshot(
    form: Literal["outer", "internal", "plain", "bare"],
) -> dict[str, JSONValue]:
    """Internal target-side phases are reader compatibility, not native user syntax."""
    original = _original()
    original["n_periods"] = 2
    regimes = _object(original["user_regimes"])
    for target in ("small", "large"):
        regimes[target] = deepcopy(regimes["original"])
    phases: dict[str, JSONValue] = {}
    for phase in ("solve", "simulate"):
        laws: dict[str, JSONValue] = {}
        for target in ("small", "large"):
            coordinate = (
                "x"
                if (target == "small") == (phase == "solve" or form == "plain")
                else "history"
            )
            laws[target] = function(
                f"def next_{target}_{phase}({coordinate}):\n    return {coordinate}\n",
                [coordinate],
            )
        phases[phase] = laws
    if form == "outer":
        update: JSONValue = {
            "kind": "object",
            "type": "lcm.Phased",
            "fields": phases,
        }
    elif form == "internal":
        update = {
            target: {
                "kind": "object",
                "type": "lcm.Phased",
                "fields": {
                    phase: _object(phases[phase])[target]
                    for phase in ("solve", "simulate")
                },
            }
            for target in ("small", "large")
        }
    elif form == "plain":
        update = phases["solve"]
    else:
        update = function("def next_x(x):\n    return x\n", ["x"])
    _fields(original)["state_transitions"] = {
        "x": update,
        "history": function(
            "def next_history(history):\n    return history\n", ["history"]
        ),
    }
    reach = _object(_object(original["reachability"])["fields"])
    reach["nodes"] = [[0, "original"], [1, "small"], [1, "large"]]
    for phase in ("solution", "simulation"):
        _object(_object(reach[phase])["fields"])["targets_by_period"] = [
            {"original": ["small", "large"]}
        ]
    return original


@pytest.mark.parametrize("form", ["outer", "internal", "plain", "bare"])
def test_phase_target_update_selection_retains_exact_law_and_provenance(
    form: Literal["outer", "internal", "plain", "bare"],
) -> None:
    original = _phase_target_update_snapshot(form)
    normalized = _normalized(original)
    callbacks = _object(normalized["callbacks"])
    cells = {
        _name(_object(row)["id"]): _object(row) for row in _rows(normalized["cells"])
    }
    source = _source(original)
    checked: set[tuple[str, str]] = set()
    for row in _rows(normalized["relations"]):
        relation = _object(row)
        if relation["kind"] != "transition":
            continue
        target = _name(cells[_name(_rows(relation["targets"])[0])]["regime"])
        phase = _name(relation["phase"])
        coordinate = (
            "x"
            if form == "bare"
            or (target == "small") == (phase == "solve" or form == "plain")
            else "history"
        )
        callback = _object(callbacks[_name(_object(relation["state_updates"])["x"])])
        selected = _object(callback["original"])
        assert selected["kind"] == "callable"
        assert selected["arguments"] == [coordinate]
        prefix = "/user_regimes/original/fields/state_transitions/x"
        suffix = (
            f"/fields/{'solve' if phase == 'solve' else 'simulate'}/{target}"
            if form == "outer"
            else f"/{target}/fields/{'solve' if phase == 'solve' else 'simulate'}"
            if form == "internal"
            else f"/{target}"
            if form == "plain"
            else ""
        )
        assert callback["source_path"] == prefix + suffix
        assert _object(callback["provenance"])["source_path"] == prefix + suffix
        reads = source.get_update_reads(
            relation_id=_name(relation["id"]),
            target_coordinate="x",
            statuses={},
            removal_order=(),
        )
        assert reads.live_coordinates == frozenset({coordinate})
        checked.add((target, phase))
    assert checked == {
        (target, phase)
        for target in ("small", "large")
        for phase in ("solve", "simulation")
    }


@pytest.mark.parametrize("missing", ["phase", "target", "internal_phase"])
def test_phase_target_update_missing_law_does_not_borrow_another_side(
    missing: Literal["phase", "target", "internal_phase"],
) -> None:
    original = _phase_target_update_snapshot(
        "internal" if missing == "internal_phase" else "outer"
    )
    update = _object(_object(_fields(original)["state_transitions"])["x"])
    if missing == "internal_phase":
        _object(_object(update["small"])["fields"])["simulate"] = None
    elif missing == "phase":
        _object(update["fields"])["simulate"] = None
    else:
        _object(_object(update["fields"])["simulate"]).pop("small")
    normalized = _normalized(original)
    cells = {
        _name(_object(row)["id"]): _object(row) for row in _rows(normalized["cells"])
    }
    source = _source(original)
    checked = False
    for row in _rows(normalized["relations"]):
        relation = _object(row)
        if (
            relation["kind"] != "transition"
            or relation["phase"] != "simulation"
            or cells[_name(_rows(relation["targets"])[0])]["regime"] != "small"
        ):
            continue
        assert "x" not in _object(relation["state_updates"])
        assert source.get_update_reads(
            relation_id=_name(relation["id"]),
            target_coordinate="x",
            statuses={},
            removal_order=(),
        ).live_coordinates == frozenset({"x", "history"})
        checked = True
    assert checked


def test_phase_target_update_missing_callable_reference_fails_closed() -> None:
    from _lcm_regime_finder.declarations import DeclarationError

    original = _phase_target_update_snapshot("outer")
    update = _object(_object(_fields(original)["state_transitions"])["x"])
    _object(_object(update["fields"])["simulate"])["small"] = {
        "kind": "callable-reference",
        "capture_ref": "missing-target-law",
    }
    with pytest.raises(
        DeclarationError, match="callable reference is missing or changed"
    ):
        _normalized(original)


@pytest.mark.parametrize("mutation", ["phase", "target", "omitted", "payload"])
def test_phase_target_update_independent_check_rejects_changed_consumer(
    mutation: Literal["phase", "target", "omitted", "payload"],
) -> None:
    from _lcm_regime_finder.semantic_check import _check_capture

    original = _phase_target_update_snapshot("outer")
    normalized = _normalized(original)
    cells = {
        _name(_object(row)["id"]): _object(row) for row in _rows(normalized["cells"])
    }
    transitions = {
        (
            _name(cells[_name(_rows(_object(row)["targets"])[0])]["regime"]),
            _name(_object(row)["phase"]),
        ): _object(row)
        for row in _rows(normalized["relations"])
        if _object(row)["kind"] == "transition"
    }
    consumer = _object(transitions["small", "simulation"]["state_updates"])
    if mutation == "omitted":
        consumer.pop("x")
    elif mutation == "payload":
        callbacks = _object(normalized["callbacks"])
        wrong = transitions["small", "solve"]
        replacement = _object(callbacks[_name(_object(wrong["state_updates"])["x"])])
        _object(callbacks[_name(consumer["x"])])["original"] = deepcopy(
            replacement["original"]
        )
    else:
        wrong = transitions[
            ("small", "solve") if mutation == "phase" else ("large", "simulation")
        ]
        consumer["x"] = _object(wrong["state_updates"])["x"]
    failures: list[str] = []
    qualifications: list[str] = []

    def fail(kind: str, **_details: JSONValue) -> None:
        failures.append(kind)

    def unknown(kind: str, **_details: JSONValue) -> None:
        qualifications.append(kind)

    _check_capture(original, normalized, fail, unknown)
    assert "original state update consumer changed" in failures


def test_native_wire_rejects_an_opaque_type_with_spoofing_metaclass_equality() -> None:
    from _lcm_regime_finder.semantic_check import _same_native_wire

    class SpoofingMeta(
        type[object],  # noqa: PAN001 - arbitrary opaque fixture class metadata boundary
    ):
        def __eq__(cls, _other: object) -> bool:
            return True

        def __hash__(cls) -> int:
            raise AssertionError("Opaque metaclass hashing must not run")

    class Opaque(metaclass=SpoofingMeta):
        def __iter__(self) -> Iterator[int]:
            raise AssertionError("Opaque wire iteration must not run")

    value = Opaque()
    assert not _same_native_wire(value, value, _CallableRecords({}))


def test_native_wire_type_guards_never_call_metaclass_equality_or_hashing() -> None:
    from _lcm_regime_finder.semantic_check import _same_native_wire

    class TrappingMeta(
        type[object],  # noqa: PAN001 - arbitrary opaque fixture class metadata boundary
    ):
        def __eq__(cls, _other: object) -> bool:
            raise AssertionError("Opaque metaclass equality must not run")

        def __hash__(cls) -> int:
            raise AssertionError("Opaque metaclass hashing must not run")

    class Opaque(metaclass=TrappingMeta):
        def __iter__(self) -> Iterator[int]:
            raise AssertionError("Opaque wire iteration must not run")

    class ListSubclass(list[int], metaclass=TrappingMeta):
        def __iter__(self) -> Iterator[int]:
            raise AssertionError("Wire subclass iteration must not run")

    records = _CallableRecords({})
    opaque = Opaque()
    subclass = ListSubclass([1])
    assert not _same_native_wire(opaque, opaque, records)
    assert not _same_native_wire(subclass, subclass, records)


def test_native_wire_exact_builtin_and_subclass_neighbors() -> None:
    from _lcm_regime_finder.semantic_check import _same_native_wire

    records = _CallableRecords({})
    assert _same_native_wire(None, None, records)
    assert _same_native_wire(True, True, records)
    assert _same_native_wire(1, 1, records)
    assert _same_native_wire("wire", "wire", records)
    assert _same_native_wire([1, "wire"], (1, "wire"), records)
    assert _same_native_wire({"axis": [1, 2]}, {"axis": (1, 2)}, records)
    assert not _same_native_wire(True, 1, records)
    assert not _same_native_wire(0.0, -0.0, records)

    class ListSubclass(list[int]):
        pass

    class TupleSubclass(tuple[int, ...]):
        pass

    class DictSubclass(dict[str, int]):
        pass

    assert not _same_native_wire(ListSubclass([1]), [1], records)
    assert not _same_native_wire(TupleSubclass((1,)), (1,), records)
    assert not _same_native_wire(DictSubclass(axis=1), {"axis": 1}, records)


def test_native_wire_rejects_opaque_nested_field_keys_without_hashing() -> None:
    from _lcm_regime_finder.semantic_check import _same_native_wire

    field_name = "cases"
    stored_hash = hash(field_name)

    class OpaqueKey:
        armed: bool = False

        def __hash__(self) -> int:
            if self.armed:
                raise AssertionError("Opaque field-key hashing must not run")
            return stored_hash

        def __eq__(self, _other: object) -> bool:
            if self.armed:
                raise AssertionError("Opaque field-key equality must not run")
            return False

    _, native, records, _ = _schedule_answer(_native_schedule())
    assert _same_native_wire(native, native, records)
    original_fields = native["fields"]
    assert isinstance(original_fields, dict)
    assert set(original_fields) == {"cases", "case_names", "case_by_period"}
    key = OpaqueKey()
    fields = {
        key if name == field_name else name: value
        for name, value in original_fields.items()
    }
    assert len(fields) == len(original_fields)
    changed = {**native, "fields": fields}
    key.armed = True
    assert not _same_native_wire(changed, changed, records)


@pytest.mark.parametrize("location", ["fields", "payload"])
def test_native_wire_rejects_mutated_categorical_field_keys_without_hashing(
    location: Literal["fields", "payload"],
) -> None:
    from _lcm.regime_building.value_adapter import ValueAdapter
    from _lcm_regime_finder.semantic_check import _same_native_wire
    from lcm import DiscreteGrid

    field_name = "categories" if location == "fields" else "kind"
    stored_hash = hash(field_name)

    class OpaqueKey:
        armed: bool = False

        def __hash__(self) -> int:
            if self.armed:
                raise AssertionError("Opaque categorical-key hashing must not run")
            return stored_hash

        def __eq__(self, _other: object) -> bool:
            if self.armed:
                raise AssertionError("Opaque categorical-key equality must not run")
            return False

    grid = DiscreteGrid(values=np.array([1.0, 2.0]))
    wrapper = ValueAdapter(
        func=_native_history_input,
        decoded=MappingProxyType({"history": grid}),
        encode=grid,
    )
    _, native, records, _ = _schedule_answer(wrapper)
    meta = records.resolve(native["callable"])
    reference = meta["bound_self"]["fields"]["encode"]
    outer = deepcopy(native["fields"]["encode"])
    assert _same_native_wire(outer, reference, records)
    record = records.resolve(reference)
    assert isinstance(record, dict)
    payload = record["value"]
    assert isinstance(payload, dict)
    fields = payload["fields"]
    assert isinstance(fields, dict)
    assert set(fields) == {"categories", "codes", "ordered", "tolerance", "values"}
    assert set(payload) == {"kind", "type", "fields"}
    mapping = fields if location == "fields" else payload
    key = OpaqueKey()
    changed = {
        key if name == field_name else name: value for name, value in mapping.items()
    }
    assert len(changed) == len(mapping)
    try:
        # Replacing owned metadata after validation is not a newly admitted capture.
        if location == "fields":
            payload["fields"] = changed
        else:
            record["value"] = changed
        key.armed = True
        assert not _same_native_wire(outer, reference, records)
    finally:
        payload["fields"] = fields
        record["value"] = payload


def _support_image_dispatch(choice: int) -> Array:
    destination = jnp.where(choice == 1, 2, 1)
    return jnp.where(destination == 0, 0, jnp.where(destination == 2, 2, 1))


def _support_image_statement(choice: int) -> int:
    if choice:
        return 1
    return 2


def _support_image_expression(choice: int) -> int:
    return 1 if choice else 2


def _support_image_age_choice(*, age: UserAge, choice: int) -> Array:
    return jnp.where(
        age == 79, jnp.where(choice == 1, 4, 3), jnp.where(choice == 1, 2, 1)
    )


def _support_image_bool(choice: int) -> Array:
    return jnp.where(choice == 1, True, False)


def _support_image_legacy_boolop(choice: int) -> Array:
    return jnp.where(choice == 1, (0 or 2), False)


def _support_image_legacy_same_value(choice: int) -> Array:
    return jnp.where(choice == 1, True, (0 or 2))


def _support_image_legacy_same_nested(*, choice: int, history: int) -> Array:
    value = True if choice else (0 or 2)
    return jnp.where(history == 1, value, False)


def _support_image_legacy_condition(choice: int) -> Array:
    return jnp.where(
        (0 or 2) == 1, jnp.where(choice == 1, 1, 2), jnp.where(choice == 1, 3, 4)
    )


def _support_image_legacy_if(choice: int) -> Array:
    if (0 or 2) == 1:
        return jnp.where(choice == 1, 1, 2)
    return jnp.where(choice == 1, 3, 4)


def _support_image_legacy_ifexp(choice: int) -> Array:
    return (
        jnp.where(choice == 1, 1, 2) if (0 or 2) == 1 else jnp.where(choice == 1, 3, 4)
    )


def _support_image_float(choice: int) -> Array:
    return jnp.where(choice == 1, 1.0, 2.0)


def _support_image_mixed(choice: int) -> Array:
    return jnp.where(choice == 1, True, 2)


_SUPPORT_ARRAY_0D: NDArray[np.int8] = np.asarray(1, dtype=np.int8)


def _support_image_array_binding(choice: int) -> Array:
    return jnp.where(choice == 1, _SUPPORT_ARRAY_0D, 2)


def _support_image_cast(choice: int) -> Array:
    return jnp.asarray(jnp.where(choice == 1, 1, 2))


def _support_image_astype(choice: int) -> Array:
    return jnp.where(choice == 1, 1, 2).astype(jnp.int32)


def _support_image_add(choice: int) -> Array:
    return jnp.where(choice == 1, 1, 2) + 0


def _support_image_negative(choice: int) -> Array:
    return -jnp.where(choice == 1, 1, 2)


def _support_image_invert(choice: int) -> Array:
    return ~jnp.where(choice == 1, 1, 2)


def _support_image_not(choice: int) -> bool:
    return not jnp.where(choice == 1, 1, 2)


def _support_image_unmapped(choice: int) -> Array:
    return jnp.where(choice == 1, 1, 99)


def _support_image_hidden(history: int) -> Array:
    raise AssertionError("Support analysis must not execute a callback")


def _support_image_opaque(*, choice: int, history: int) -> Array:
    return jnp.where(choice == 1, 1, _support_image_hidden(history))


def _support_image_physical(choice: int) -> Array:
    return jnp.where(choice == 1, 10, 20)


def _support_image_code(physical_code: int) -> int:
    return physical_code


_SUPPORT_FRACTION = Fraction(1)


def _support_image_fraction_output(choice: int) -> Fraction | int:
    return _SUPPORT_FRACTION if choice else 2


def _support_image_original(
    *,
    callback: UserFunction,
    values: tuple[int | bool, ...] = (0, 1, 2),
) -> tuple[dict[str, JSONValue], dict[str, JSONValue]]:
    native, records = _capture_native_wrapper(callback, declared=False)
    original = _original()
    original["callable_records"] = records
    _object(_fields(original)["states"])["x"] = grid(list(values))
    _object(_fields(original)["state_transitions"])["x"] = native
    return original, native


@pytest.mark.parametrize(
    "callback",
    (
        _support_image_dispatch,
        _support_image_statement,
        _support_image_expression,
    ),
)
def test_support_images_prove_complete_control_alternatives_without_scalar_value(
    callback: UserFunction,
) -> None:
    original, native = _support_image_original(callback=callback)
    assert _source(original).get_target_support(
        relation_id=_relation_id(original),
        target_coordinate="x",
        statuses={},
    ) == TargetSupport(frozenset({1, 2}), True)
    answer = SourceProof(
        {"root": {"original": native}},
        callable_records=_object(original["callable_records"]),
    ).callback("root")
    assert answer.proved
    assert answer.value is _UNKNOWN
    assert answer.reads == frozenset({"choice"})
    assert answer.support_images is not None
    assert set(answer.support_images) == {1, 2}


def test_support_images_optimized_compiler_matches_literal_reference() -> None:
    from _lcm_regime_finder.implicit.implicit_core import Support
    from _lcm_regime_finder.implicit.implicit_native import CellCompiler

    original, _ = _support_image_original(callback=_support_image_dispatch)
    source = _source(original)
    compiler = CellCompiler(source=source, cell_id=_cell_id(original), status_axes=())
    assert compiler.cache_purity["proved"] is True
    relation_id = _relation_id(original)
    root = compiler.support(relation_id=relation_id, coordinate="x")
    optimized = compiler.diagram.evaluate(root, {})
    assert optimized == Support(frozenset({1, 2}), True)
    reference = source.get_target_support(
        relation_id=relation_id, target_coordinate="x", statuses={}
    )
    assert reference == TargetSupport(frozenset({1, 2}), True)
    assert isinstance(optimized, Support)
    assert (optimized.possible_codes, optimized.proved) == (
        reference.possible_codes,
        reference.proved,
    )


@pytest.mark.parametrize(
    "callback",
    (
        _support_image_float,
        _support_image_mixed,
        _support_image_cast,
        _support_image_astype,
        _support_image_add,
        _support_image_opaque,
        _support_image_array_binding,
        _support_image_legacy_same_value,
        _support_image_legacy_same_nested,
    ),
)
def test_support_images_keep_unsupported_transformations_conservative(
    callback: UserFunction,
) -> None:
    original, _ = _support_image_original(callback=callback)
    assert _source(original).get_target_support(
        relation_id=_relation_id(original),
        target_coordinate="x",
        statuses={},
    ) == TargetSupport(frozenset({0, 1, 2}), False)


@pytest.mark.parametrize(
    "callback",
    (
        _support_image_negative,
        _support_image_invert,
        _support_image_not,
    ),
)
def test_support_images_do_not_forward_through_unknown_unary_operations(
    callback: UserFunction,
) -> None:
    values = (-3, -2, -1, 0, 1, 2)
    original, native = _support_image_original(callback=callback, values=values)
    answer = SourceProof(
        {"root": {"original": native}},
        callable_records=_object(original["callable_records"]),
    ).callback("root")
    assert answer.proved and answer.value is _UNKNOWN
    assert answer.reads == frozenset({"choice"})
    assert answer.support_images is None
    assert _source(original).get_target_support(
        relation_id=_relation_id(original),
        target_coordinate="x",
        statuses={},
    ) == TargetSupport(frozenset(range(len(values))), False)


def test_support_images_unmapped_alternative_is_not_a_model_counterexample() -> None:
    original, _ = _support_image_original(callback=_support_image_unmapped)
    assert _source(original).get_target_support(
        relation_id=_relation_id(original),
        target_coordinate="x",
        statuses={},
    ) == TargetSupport(frozenset({0, 1, 2}), False)
    _object(_fields(original)["state_transitions"])["x"] = function(
        "def next_x():\n    return 99\n",
        [],
    )
    with pytest.raises(ValueError, match="deterministic update leaves target grid"):
        _source(original).get_target_support(
            relation_id=_relation_id(original),
            target_coordinate="x",
            statuses={},
        )


def test_support_images_budget_overflow_keeps_the_entire_original_domain() -> None:
    original = _original()
    values = tuple(range(65))
    _object(_fields(original)["states"])["x"] = grid(list(values))
    expression = "64"
    for value in reversed(range(64)):
        expression = f"{value} if choice == {value} else ({expression})"
    _object(_fields(original)["state_transitions"])["x"] = function(
        f"def next_x(choice):\n    return {expression}\n",
        ["choice"],
    )
    assert _source(original).get_target_support(
        relation_id=_relation_id(original),
        target_coordinate="x",
        statuses={},
    ) == TargetSupport(frozenset(values), False)


def test_support_images_are_cleared_by_nonidentity_derived_category_encoding() -> None:
    # Capture roots must share one graph so unrelated C-identifiers cannot alias.
    with capture_session() as session:
        physical = normalize(_support_image_physical)
        code = normalize(_support_image_code)
    answer = SourceProof(
        {"physical": {"original": physical}, "root": {"original": code}},
        providers={
            "physical": "physical",
            "physical_code": {
                "derived_code_of": "physical",
                "values": [10, 20],
                "codes": [0, 1],
            },
        },
        callable_records=session.records,
    ).callback("root")
    assert answer.proved and answer.value is _UNKNOWN
    assert answer.reads == frozenset({"choice"})
    assert answer.support_images is None
    original = _original()
    original["callable_records"] = session.records
    _object(_fields(original)["states"])["x"] = grid([0, 1, 10, 20])
    _fields(original)["derived_categoricals"] = {"physical": grid([10, 20])}
    _object(_fields(original)["functions"])["physical"] = physical
    _object(_fields(original)["state_transitions"])["x"] = code
    assert _source(original).get_target_support(
        relation_id=_relation_id(original),
        target_coordinate="x",
        statuses={},
    ) == TargetSupport(frozenset({0, 1, 2, 3}), False)


class _SupportImageEqualityTrap:
    def __eq__(self, _other: object) -> bool:
        raise AssertionError("Opaque support image equality must not run")

    def __hash__(self) -> int:
        raise AssertionError("Opaque support image hashing must not run")


def test_support_images_reject_opaque_and_typed_scalar_neighbors_without_equality() -> (
    None
):
    from _lcm_regime_finder.semantic_check import _support_literal, _support_union

    assert _support_literal(_SupportImageEqualityTrap()) is None
    assert _support_literal(1.0) is None
    assert _support_literal(2**31) is None
    assert _support_literal(-(2**31) - 1) is None
    assert _support_union((True,), (1,)) is None
    assert _support_union((False,), (True,)) == (False, True)
    assert _support_union((-(2**31),), (2**31 - 1,)) == (-(2**31), 2**31 - 1)
    assert _support_literal(Fraction(62)) is None
    assert _support_literal(Fraction(3, 2)) is None


def test_support_images_preserve_the_actual_fraction_clock_outer_select() -> None:
    native, records = _capture_native_wrapper(_support_image_age_choice, declared=False)
    answer = SourceProof(
        {"root": {"original": native}},
        static={"age": Fraction(62)},
        callable_records=records,
    ).callback("root")
    assert answer.proved and answer.value is _UNKNOWN
    assert answer.reads == frozenset({"choice"})
    assert answer.support_images is not None and set(answer.support_images) == {1, 2}


def test_support_images_preserve_boolean_element_types() -> None:
    original, _ = _support_image_original(
        callback=_support_image_bool, values=(False, True)
    )
    assert _source(original).get_target_support(
        relation_id=_relation_id(original),
        target_coordinate="x",
        statuses={},
    ) == TargetSupport(frozenset({0, 1}), True)


def test_support_images_do_not_reinterpret_captured_fraction_outputs_as_ints() -> None:
    with capture_session() as session:
        native = normalize(_support_image_fraction_output)
    answer = SourceProof(
        {"root": {"original": native}}, callable_records=session.records
    ).callback("root")
    assert answer.proved and answer.value is _UNKNOWN
    assert answer.reads == frozenset({"choice"})
    assert answer.support_images is None
    original = _original()
    original["callable_records"] = session.records
    _object(_fields(original)["states"])["x"] = grid([0, 1, 2])
    _object(_fields(original)["state_transitions"])["x"] = native
    assert _source(original).get_target_support(
        relation_id=_relation_id(original),
        target_coordinate="x",
        statuses={},
    ) == TargetSupport(frozenset({0, 1, 2}), False)


def test_support_images_do_not_seed_computed_legacy_boolean_values() -> None:
    original, _ = _support_image_original(
        callback=_support_image_legacy_boolop, values=(False, True, 2)
    )
    assert _source(original).get_target_support(
        relation_id=_relation_id(original),
        target_coordinate="x",
        statuses={},
    ) == TargetSupport(frozenset({0, 1, 2}), False)


@pytest.mark.parametrize(
    "callback", (_support_image_legacy_if, _support_image_legacy_ifexp)
)
def test_support_images_do_not_trust_legacy_known_python_conditions(
    callback: UserFunction,
) -> None:
    original, native = _support_image_original(
        callback=callback, values=(0, 1, 2, 3, 4)
    )
    answer = SourceProof(
        {"root": {"original": native}},
        callable_records=_object(original["callable_records"]),
    ).callback("root")
    assert answer.proved and answer.value is _UNKNOWN
    assert answer.reads == frozenset({"choice"})
    assert answer.support_images is None
    assert _source(original).get_target_support(
        relation_id=_relation_id(original),
        target_coordinate="x",
        statuses={},
    ) == TargetSupport(frozenset({0, 1, 2, 3, 4}), False)


def test_support_images_union_both_select_branches_when_legacy_truth_is_unproved() -> (
    None
):
    original, native = _support_image_original(
        callback=_support_image_legacy_condition, values=(0, 1, 2, 3, 4)
    )
    answer = SourceProof(
        {"root": {"original": native}},
        callable_records=_object(original["callable_records"]),
    ).callback("root")
    assert answer.proved and answer.value is _UNKNOWN
    assert answer.reads == frozenset({"choice"})
    assert answer.support_images is not None and set(answer.support_images) == {
        1,
        2,
        3,
        4,
    }
    assert _source(original).get_target_support(
        relation_id=_relation_id(original),
        target_coordinate="x",
        statuses={},
    ) == TargetSupport(frozenset({1, 2, 3, 4}), True)


@pytest.mark.parametrize(
    ("expression", "expected"),
    (
        ("0 or 2", 2),
        ("1 or 2", 1),
        ("0 and 2", 0),
        ("1 and 2", 2),
        ("'' or 'kept'", "kept"),
        ("None or False", False),
    ),
)
def test_support_images_boolop_scalar_semantics_returns_the_actual_operand(
    *,
    expression: str,
    expected: JSONValue,
) -> None:
    answer = SourceProof({}).callback(
        function(f"def root():\n    return {expression}\n", [])
    )
    assert answer.proved and not answer.reads
    assert type(answer.value) is type(expected) and answer.value == expected
    assert answer.support_images is None


class _SupportTruthTrap:
    def __bool__(self) -> bool:
        raise AssertionError("Unsupported scalar truth must not execute user code")


def test_support_images_boolop_unknown_and_unsupported_truth_remain_conservative() -> (
    None
):
    callback = function(
        "def root(value, history):\n    return value or history\n", ["value", "history"]
    )
    unknown = SourceProof({}).callback(callback)
    assert unknown.value is _UNKNOWN and unknown.reads == frozenset(
        {"value", "history"}
    )
    assert unknown.support_images is None
    opaque = SourceProof({}, static={"value": _SupportTruthTrap()}).callback(callback)
    assert not opaque.proved and opaque.value is _UNKNOWN
    assert opaque.support_images is None


@categorical(ordered=True)
class _SupportImageNativeChoice:
    decline: ScalarInt
    apply: ScalarInt


@categorical(ordered=False)
class _SupportImageNativeRegime:
    before: ScalarInt
    unclaimed: ScalarInt
    claimed: ScalarInt
    terminal_unclaimed: ScalarInt
    terminal_claimed: ScalarInt


def _support_image_native_destination(*, age: UserAge, choice: int) -> Array:
    return jnp.where(
        age == 79,
        jnp.where(
            choice == _SupportImageNativeChoice.apply,
            _SupportImageNativeRegime.terminal_claimed,
            _SupportImageNativeRegime.terminal_unclaimed,
        ),
        jnp.where(
            choice == _SupportImageNativeChoice.apply,
            _SupportImageNativeRegime.claimed,
            _SupportImageNativeRegime.unclaimed,
        ),
    )


def _support_image_native_member(original_destination: Array) -> Array:
    return jnp.where(
        original_destination == 0, 0, jnp.where(original_destination == 2, 2, 1)
    )


def _support_image_native_dispatch(*, age: UserAge, choice: int, x: int) -> Array:
    destination = _support_image_native_destination(age=age, choice=choice)
    member = _support_image_native_member(destination)
    return jnp.where(x == 1, member, 0)


def test_support_images_capture_native_categorical_members_through_dispatch() -> None:
    from _lcm_regime_finder.implicit.implicit_core import Support
    from _lcm_regime_finder.implicit.implicit_native import CellCompiler

    original, native = _support_image_original(callback=_support_image_native_dispatch)
    source = _source(original)
    relation_id = _relation_id(original)
    expected = TargetSupport(frozenset({1, 2}), True)
    assert (
        source.get_target_support(
            relation_id=relation_id, target_coordinate="x", statuses={"x": 1}
        )
        == expected
    )
    answer = SourceProof(
        {"root": {"original": native}},
        static={"age": Fraction(62), "x": 1},
        callable_records=_object(original["callable_records"]),
    ).callback("root")
    assert answer.proved and answer.value is _UNKNOWN
    assert answer.reads == frozenset({"choice"})
    assert answer.support_images is not None and set(answer.support_images) == {1, 2}
    compiler = CellCompiler(
        source=source, cell_id=_cell_id(original), status_axes=("x",)
    )
    assert compiler.cache_purity["proved"] is True
    root = compiler.support(relation_id=relation_id, coordinate="x")
    assert compiler.diagram.evaluate(root, {"x": 1}) == Support(
        expected.possible_codes, expected.proved
    )


@pytest.mark.parametrize(
    ("field", "replacement"),
    (
        ("kind", "type"),
        ("dtype", "int8"),
        ("dtype", "int64"),
        ("dtype", "float32"),
        ("shape", [1]),
        ("shape", None),
        ("values", True),
        ("values", 1.0),
        ("values", [1]),
        ("values", 0),
        ("values", 2**31),
        ("values", -(2**31) - 1),
        ("sha256", "0" * 64),
        ("sha256", None),
        ("extra", 0),
    ),
)
def test_support_images_reject_changed_native_categorical_member_wire(
    *, field: str, replacement: JSONValue
) -> None:
    from _lcm_regime_finder.semantic_check import _support_categorical_int32

    captured = normalize(_SupportImageNativeRegime)
    member = _object(_object(captured["members"])["unclaimed"])
    assert _support_categorical_int32(member) == (1,)
    changed = deepcopy(member)
    changed[field] = replacement
    assert _support_categorical_int32(changed) is None


def test_support_images_reject_opaque_categorical_payload_without_equality() -> None:
    from _lcm_regime_finder.semantic_check import _support_categorical_int32

    captured = normalize(_SupportImageNativeRegime)
    member = _object(_object(captured["members"])["unclaimed"])
    assert _support_categorical_int32(member) == (1,)
    assert (
        _support_categorical_int32(
            {key: value for key, value in member.items() if key != "sha256"}
        )
        is None
    )
    malformed: dict[str, object] = dict(member)  # noqa: PAN001 - intentional arbitrary malformed categorical payload boundary
    malformed["values"] = _SupportImageEqualityTrap()
    assert _support_categorical_int32(malformed) is None
    malformed["values"] = 1
    malformed["sha256"] = _SupportImageEqualityTrap()
    assert _support_categorical_int32(malformed) is None


class _SupportImageIntSubclass(int):
    def __eq__(self, _other: object) -> bool:
        raise AssertionError("A categorical int subclass must not enter equality")


def test_support_images_reject_categorical_int_subclasses() -> None:
    from _lcm_regime_finder.semantic_check import _support_categorical_int32

    captured = normalize(_SupportImageNativeRegime)
    member = _object(_object(captured["members"])["unclaimed"])
    assert _support_categorical_int32(member) == (1,)
    member["values"] = _SupportImageIntSubclass(1)
    assert _support_categorical_int32(member) is None


_SUPPORT_ARRAY_INT32: NDArray[np.int32] = np.asarray(1, dtype=np.int32)


def _support_image_generic_int32_array(choice: int) -> Array:
    return jnp.where(choice == 1, _SUPPORT_ARRAY_INT32, 2)


def test_support_images_do_not_seed_generic_int32_array_bindings() -> None:
    original, _ = _support_image_original(callback=_support_image_generic_int32_array)
    assert _source(original).get_target_support(
        relation_id=_relation_id(original), target_coordinate="x", statuses={}
    ) == TargetSupport(frozenset({0, 1, 2}), False)
