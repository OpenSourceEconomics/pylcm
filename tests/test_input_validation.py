"""Validated construction of user-facing configuration records."""

import dataclasses
import inspect
from collections.abc import Callable
from decimal import Decimal
from fractions import Fraction

import cloudpickle
import numpy as np
import pytest
from beartype.roar import BeartypeCallHintViolation
from pydantic import ValidationError

from _lcm.egm.branch_aggregation import UniformObservedFixedCost
from _lcm.egm.outer_search import AdaptiveOuterMesh
from _lcm.input_validation import format_validation_error
from _lcm.solution.dcegm import DCEGM, ExactEnvelope, FUESEnvelope, RFCEnvelope
from lcm import LinSpacedGrid
from lcm.exceptions import RegimeInitializationError
from lcm.execution import ExecutionConfig, WidthSearchPolicy

_GRID = LinSpacedGrid(start=0.0, stop=1.0, n_points=5)

type _Built = (
    ExactEnvelope
    | FUESEnvelope
    | RFCEnvelope
    | DCEGM
    | AdaptiveOuterMesh
    | UniformObservedFixedCost
    | WidthSearchPolicy
    | ExecutionConfig
)
# Constructors are called with keyword arguments assembled per case.
type _Record = Callable[..., _Built]
type _Required = dict[str, LinSpacedGrid | str | float]


class _IntSubclass(int):
    pass


class _FloatLike:
    def __float__(self) -> float:
        return 2.0


type _Inadmissible = (
    bool | int | float | str | np.generic | Decimal | Fraction | _FloatLike
)


# Every migrated floating option, with the error a refused value raises.
_FLOAT_OPTIONS = (
    pytest.param(FUESEnvelope, {}, "jump_thresh", RegimeInitializationError, id="fues"),
    pytest.param(RFCEnvelope, {}, "jump_thresh", RegimeInitializationError, id="rfc"),
    pytest.param(
        DCEGM,
        {"savings_grid": _GRID},
        "refined_grid_factor",
        RegimeInitializationError,
        id="dcegm",
    ),
    pytest.param(
        AdaptiveOuterMesh,
        {"initial_grid": _GRID},
        "value_atol",
        RegimeInitializationError,
        id="outer_mesh",
    ),
    pytest.param(
        UniformObservedFixedCost,
        {"shock_name": "chi", "scale_function": "scale", "upper": 3.0},
        "lower",
        RegimeInitializationError,
        id="fixed_cost",
    ),
    pytest.param(
        ExecutionConfig,
        {},
        "device_memory_headroom_fraction",
        ValueError,
        id="headroom",
    ),
)

# Every migrated count, with the error a refused value raises.
_COUNT_OPTIONS = (
    pytest.param(ExactEnvelope, {}, "max_runs", RegimeInitializationError, id="exact"),
    pytest.param(RFCEnvelope, {}, "search_radius", RegimeInitializationError, id="rfc"),
    pytest.param(
        DCEGM,
        {"savings_grid": _GRID},
        "n_constrained_points",
        RegimeInitializationError,
        id="dcegm",
    ),
    pytest.param(
        AdaptiveOuterMesh,
        {"initial_grid": _GRID},
        "max_nodes",
        RegimeInitializationError,
        id="outer_mesh",
    ),
    pytest.param(WidthSearchPolicy, {}, "max_evaluations", ValueError, id="search"),
    pytest.param(ExecutionConfig, {}, "device_memory_bytes", ValueError, id="budget"),
)

_NON_BUILTIN_REALS = (
    pytest.param(True, id="bool"),
    pytest.param("2", id="str"),
    pytest.param(np.float64(2.0), id="numpy_float"),
    pytest.param(np.int64(2), id="numpy_int"),
    pytest.param(Decimal(2), id="decimal"),
    pytest.param(Fraction(2), id="fraction"),
    pytest.param(_FloatLike(), id="float_protocol"),
)

_NON_EXACT_INTS = (
    pytest.param(True, id="bool"),
    pytest.param(4.0, id="float"),
    pytest.param("4", id="str"),
    pytest.param(_IntSubclass(4), id="int_subclass"),
    pytest.param(np.int64(4), id="numpy_int"),
)


@pytest.mark.parametrize(("record", "required", "name", "_error"), _FLOAT_OPTIONS)
def test_float_option_stores_an_integer_as_a_python_float(
    *, record: _Record, required: _Required, name: str, _error: type[Exception]
) -> None:
    """`2` is admitted and stored as the Python float `2.0`."""
    value = 0 if name == "device_memory_headroom_fraction" else 2
    stored = getattr(record(**required, **{name: value}), name)
    assert (type(stored), stored) == (float, float(value))


@pytest.mark.parametrize("value", _NON_BUILTIN_REALS)
@pytest.mark.parametrize(("record", "required", "name", "error"), _FLOAT_OPTIONS)
def test_float_option_refuses_a_non_builtin_real(
    *,
    record: _Record,
    required: _Required,
    name: str,
    error: type[Exception],
    value: _Inadmissible,
) -> None:
    """Only a builtin `int` or `float` is admitted, and the error names the field."""
    with pytest.raises(error, match=name):
        record(**required, **{name: value})


@pytest.mark.parametrize("value", _NON_EXACT_INTS)
@pytest.mark.parametrize(("record", "required", "name", "error"), _COUNT_OPTIONS)
def test_count_option_refuses_anything_but_an_exact_int(
    *,
    record: _Record,
    required: _Required,
    name: str,
    error: type[Exception],
    value: _Inadmissible,
) -> None:
    """A count is a builtin `int`; booleans, floats and subclasses are refused."""
    with pytest.raises(error, match=name):
        record(**required, **{name: value})


@pytest.mark.parametrize(
    "record",
    [ExactEnvelope, FUESEnvelope, RFCEnvelope, WidthSearchPolicy, ExecutionConfig],
)
def test_record_defaults_pass_their_own_validation(*, record: _Record) -> None:
    """Defaults are validated like supplied values and construct a record."""
    assert dataclasses.is_dataclass(record())


def test_execution_config_default_search_policy_is_a_policy_instance() -> None:
    assert type(ExecutionConfig().width_search) is WidthSearchPolicy


def test_refused_field_data_chains_the_validation_error() -> None:
    with pytest.raises(RegimeInitializationError) as raised:
        FUESEnvelope(jump_thresh=-1.0)
    assert isinstance(raised.value.__cause__, ValidationError)


@pytest.mark.parametrize(
    ("args", "kwargs"),
    [
        pytest.param((2.0,), {}, id="positional"),
        pytest.param((), {"jump_threshold": 2.0}, id="unexpected"),
    ],
)
def test_wrong_call_shape_raises_type_error(
    *, args: tuple[float, ...], kwargs: dict[str, float]
) -> None:
    with pytest.raises(TypeError):
        FUESEnvelope(*args, **kwargs)  # ty: ignore[invalid-argument-type]


def test_missing_required_argument_raises_type_error() -> None:
    with pytest.raises(TypeError, match="shock_name"):
        UniformObservedFixedCost(scale_function="scale")  # ty: ignore[missing-argument]


def test_ordinary_functions_keep_beartype_checking() -> None:
    """The package claw still checks functions in the validation module."""
    with pytest.raises(BeartypeCallHintViolation):
        format_validation_error(class_name=1, error=None, inputs={})  # ty: ignore[invalid-argument-type]


@pytest.mark.parametrize(
    "record", [ExactEnvelope, FUESEnvelope, RFCEnvelope, ExecutionConfig]
)
def test_record_is_frozen(*, record: _Record) -> None:
    instance = record()
    name = dataclasses.fields(instance)[0].name
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(instance, name, getattr(instance, name))


def test_replace_validates_the_changed_field() -> None:
    with pytest.raises(RegimeInitializationError, match=r"FUESEnvelope\.jump_thresh"):
        dataclasses.replace(FUESEnvelope(), jump_thresh=0.0)


def test_replace_normalizes_the_changed_field() -> None:
    replaced = dataclasses.replace(FUESEnvelope(), jump_thresh=3)
    assert type(replaced.jump_thresh) is float


def test_record_survives_a_cloudpickle_round_trip() -> None:
    """A configuration travels to a worker process unchanged."""
    config = ExecutionConfig(axis_widths={"cell": 4, "point": {"working": 2}})
    assert cloudpickle.loads(cloudpickle.dumps(config)) == config


@pytest.mark.parametrize(
    ("record", "module"),
    [
        (FUESEnvelope, "_lcm.solution.dcegm"),
        (ExecutionConfig, "lcm.execution"),
        (AdaptiveOuterMesh, "_lcm.egm.outer_search"),
        (UniformObservedFixedCost, "_lcm.egm.branch_aggregation"),
    ],
)
def test_record_keeps_its_defining_module(*, record: type[_Built], module: str) -> None:
    assert (record.__module__, record.__qualname__) == (module, record.__name__)


def test_record_signature_is_keyword_only() -> None:
    parameters = inspect.signature(FUESEnvelope).parameters.values()
    assert {parameter.kind for parameter in parameters} == {
        inspect.Parameter.KEYWORD_ONLY
    }


def test_an_error_path_names_the_field_not_the_union_members() -> None:
    """Each line names the field the caller wrote, never a schema member label."""
    with pytest.raises(ValueError, match="Invalid ExecutionConfig") as raised:
        ExecutionConfig(value_storage="scratch")  # ty: ignore[invalid-argument-type]
    assert str(raised.value) == (
        "Invalid ExecutionConfig:\n"
        "  ExecutionConfig.value_storage: Input should be 'device' or 'host'"
        " (got 'scratch')\n"
        "  ExecutionConfig.value_storage: Input should be an instance of Path"
        " (got 'scratch')"
    )


@pytest.mark.parametrize(
    ("kwargs", "path"),
    [
        pytest.param({"device_memory_bytes": "x"}, ".device_memory_bytes:", id="tag"),
        pytest.param(
            {"axis_widths": {"b": {"working": 0}}},
            ".axis_widths['b']['working']:",
            id="nested_union",
        ),
        pytest.param({"axis_widths": {"": 2}}, ".axis_widths['']:", id="key"),
    ],
)
def test_an_error_path_follows_the_supplied_containers(
    *, kwargs: dict[str, str | dict[str, int] | dict[str, dict[str, int]]], path: str
) -> None:
    with pytest.raises(ValueError, match="Invalid ExecutionConfig") as raised:
        ExecutionConfig(**kwargs)  # ty: ignore[invalid-argument-type]
    lines = str(raised.value).splitlines()[1:]
    assert lines
    assert all(line.startswith(f"  ExecutionConfig{path}") for line in lines), lines


def test_an_error_path_names_a_refused_inner_key() -> None:
    with pytest.raises(ValueError, match="Invalid WidthSearchPolicy") as raised:
        WidthSearchPolicy(hints={"working": {"": 1}})
    assert (
        str(raised.value)
        .splitlines()[1]
        .startswith("  WidthSearchPolicy.hints['working']['']:")
    ), str(raised.value)


@pytest.mark.parametrize(
    ("record", "args", "kwargs", "name"),
    [
        pytest.param(ExecutionConfig, (5,), {}, "ExecutionConfig", id="positional"),
        pytest.param(
            UniformObservedFixedCost,
            (),
            {},
            "UniformObservedFixedCost",
            id="missing",
        ),
        pytest.param(
            FUESEnvelope, (), {"jump_threshold": 2.0}, "FUESEnvelope", id="unexpected"
        ),
    ],
)
def test_a_wrong_call_shape_names_the_record(
    *,
    record: _Record,
    args: tuple[int, ...],
    kwargs: dict[str, float],
    name: str,
) -> None:
    with pytest.raises(TypeError, match=rf"^{name}\b"):
        record(*args, **kwargs)


@pytest.mark.parametrize(
    "support",
    [
        pytest.param({"lower": -np.inf}, id="lower"),
        pytest.param({"upper": np.inf}, id="upper"),
        pytest.param({"lower": -np.inf, "upper": np.inf}, id="both"),
    ],
)
def test_fixed_cost_support_must_be_finite(*, support: dict[str, float]) -> None:
    with pytest.raises(RegimeInitializationError, match="finite"):
        UniformObservedFixedCost(shock_name="chi", scale_function="scale", **support)


@pytest.mark.parametrize("name", ["value_atol", "value_rtol"])
def test_outer_mesh_tolerances_must_be_finite(*, name: str) -> None:
    with pytest.raises(RegimeInitializationError, match=rf"AdaptiveOuterMesh\.{name}"):
        AdaptiveOuterMesh(initial_grid=_GRID, **{name: float("inf")})  # ty: ignore[invalid-argument-type]


def test_outer_mesh_lipschitz_bound_may_be_infinite() -> None:
    mesh = AdaptiveOuterMesh(initial_grid=_GRID, outer_lipschitz_bound=float("inf"))
    assert mesh.outer_lipschitz_bound == float("inf")
