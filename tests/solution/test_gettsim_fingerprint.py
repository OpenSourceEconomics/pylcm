"""Exercise durable identity through GETTSIM's public JAX graph builder."""

import dataclasses
import importlib
import importlib.metadata
import os
import subprocess
import sys
from collections.abc import Callable
from functools import partial
from pathlib import Path
from types import CodeType, FunctionType, ModuleType, SimpleNamespace
from typing import Any, cast

import cloudpickle
import dags.tree as dt
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.solution import external_fingerprint
from _lcm.solution.fingerprint import _semantic_fingerprint
from lcm import (
    AgeGrid,
    LinSpacedGrid,
    Model,
    categorical,
    fixed_transition,
    load_solution,
)
from lcm.exceptions import InvalidSimulationInputError, ModelInitializationError
from lcm.regime import Regime
from lcm.typing import ScalarFloat, ScalarInt

gettsim = pytest.importorskip("gettsim")
tt = pytest.importorskip("gettsim.tt")
param_objects = pytest.importorskip("ttsim.tt.param_objects")
rounding = pytest.importorskip("ttsim.tt.rounding")
type_resolution = pytest.importorskip("ttsim.tt.type_resolution")
time_converters = pytest.importorskip("ttsim.time_converters")

_UNIT = tt.TTSIMUnit.DIMENSIONLESS
_UNIT_NAMESPACE = tt.TTSIMUnit
FloatColumn = pytest.importorskip("ttsim.typing").FloatColumn
IntColumn = pytest.importorskip("ttsim.typing").IntColumn

_BACKEND_DEPENDENCIES = ("ttsim-backend", "numpy", "jax", "jaxlib")
_OPERATION_DEPENDENCIES = (
    ("dags.tree.tree_utils.flatten_to_qnames", ("dags", "flatten-dict")),
    *(
        (
            f"ttsim.tt.aggregation.{name}",
            (*_BACKEND_DEPENDENCIES, "numpy-groupies", "numba", "llvmlite"),
        )
        for name in (
            "grouped_any",
            "grouped_count",
            "grouped_min",
            "grouped_sum",
            "sum_by_p_id",
        )
    ),
    (
        "ttsim.tt.column_objects_param_function.ColumnFunction.__call__",
        ("ttsim-backend",),
    ),
    ("ttsim.tt.column_objects_param_function.reorder_ids", _BACKEND_DEPENDENCIES),
    ("ttsim.tt.piecewise_polynomial.piecewise_polynomial", _BACKEND_DEPENDENCIES),
    ("ttsim.tt.shared.join", _BACKEND_DEPENDENCIES),
    *(
        (f"ttsim.time_converters.{name}", ("ttsim-backend", "pint"))
        for name in (
            "m_to_y",
            "per_m_to_per_y",
            "per_w_to_per_y",
            "per_y_to_per_m",
            "y_to_m",
        )
    ),
    ("ttsim.tt.units.cast_ttsim_unit", ("ttsim-backend",)),
)


@pytest.mark.parametrize(
    ("operation", "dependency"),
    [
        (operation, dependency)
        for operation, dependencies in _OPERATION_DEPENDENCIES
        for dependency in dependencies
    ],
)
@pytest.mark.parametrize("defect", ["unsupported", "editable", "missing"])
def test_external_operation_rejects_unreviewed_dependency(
    *, operation: str, dependency: str, defect: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Each installed implementation requires its reviewed dependency closure."""
    function = _resolve_external_operation(operation)
    _replace_installation_metadata(
        dependency=dependency, defect=defect, monkeypatch=monkeypatch
    )
    _clear_external_metadata_caches()
    try:
        with pytest.raises(TypeError, match=f"{dependency}.*durable identity"):
            _semantic_fingerprint(function)
    finally:
        _clear_external_metadata_caches()


@pytest.mark.parametrize("kind", ["lookup", "polynomial", "backend"])
@pytest.mark.parametrize("dependency", ["numpy", "jax", "jaxlib"])
@pytest.mark.parametrize("defect", ["unsupported", "editable", "missing"])
def test_jax_parameter_and_backend_require_reviewed_implementation(
    *, kind: str, dependency: str, defect: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """JAX carrier and backend bindings require reviewed numerical libraries."""
    if kind == "lookup":
        value = param_objects.ConsecutiveIntLookupTableParamValue(
            xnp=jnp,
            values_to_look_up=jnp.array([1.0]),
            bases_to_subtract=jnp.array([0]),
        )
    elif kind == "polynomial":
        value = param_objects.PiecewisePolynomialParamValue(
            thresholds=jnp.array([0.0, 1.0]),
            intercepts=jnp.array([1.0]),
            coefficients=jnp.array([[2.0]]),
        )
    else:

        def policy(*, value: FloatColumn, xnp: ModuleType) -> FloatColumn:
            return xnp.sqrt(value)

        declaration = tt.policy_function(
            vectorization_strategy="not_required", unit=_UNIT
        )(policy)
        value = partial(declaration, xnp=jnp)
    _replace_installation_metadata(
        dependency=dependency, defect=defect, monkeypatch=monkeypatch
    )
    _clear_external_metadata_caches()
    try:
        with pytest.raises(TypeError, match=f"{dependency}.*durable identity"):
            _semantic_fingerprint(value)
    finally:
        _clear_external_metadata_caches()


@pytest.mark.parametrize("defect", ["unsupported", "editable", "missing"])
def test_policy_record_requires_reviewed_package_metadata(
    *, defect: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A genuine policy record requires an installed reviewed policy package."""
    record = _wohngeld_record()
    _replace_installation_metadata(
        dependency="gettsim", defect=defect, monkeypatch=monkeypatch
    )
    _clear_external_metadata_caches()
    try:
        with pytest.raises(TypeError, match=r"gettsim.*durable identity"):
            _semantic_fingerprint(record)
    finally:
        _clear_external_metadata_caches()


def test_standalone_flattening_keeps_ttsim_optional() -> None:
    """Standalone DAG flattening needs no optional tax-transfer libraries."""
    script = """
import importlib.metadata
import sys
original = importlib.metadata.distribution
optional = {"gettsim", "ttsim-backend", "numpy-groupies", "numba", "llvmlite", "pint"}
def distribution(name):
    if name in optional:
        raise importlib.metadata.PackageNotFoundError(name)
    return original(name)
importlib.metadata.distribution = distribution
import dags.tree as dt
from _lcm.solution.fingerprint import _semantic_fingerprint
assert "ttsim" not in sys.modules
assert dt.flatten_to_qnames({"outer": {"inner": 3}}) == {"outer__inner": 3}
first = _semantic_fingerprint(dt.flatten_to_qnames)
assert first == _semantic_fingerprint(dt.flatten_to_qnames)
assert "ttsim" not in sys.modules
print("standalone-flattening-supported")
"""
    output = subprocess.check_output(  # noqa: S603, fixed interpreter and literal script
        [sys.executable, "-c", script],
        text=True,
        timeout=120,
    )
    assert output.strip() == "standalone-flattening-supported"


def test_grouped_count_keeps_numpy_fallback_without_numba() -> None:
    """The reviewed NumPy aggregation fallback works without compiler libraries."""
    script = """
import builtins
import importlib.metadata
original_import = builtins.__import__
original_distribution = importlib.metadata.distribution
def guarded_import(name, *args, **kwargs):
    if name.split(".", 1)[0] in {"numba", "llvmlite"}:
        raise ImportError(name)
    return original_import(name, *args, **kwargs)
def distribution(name):
    if name in {"numba", "llvmlite"}:
        raise importlib.metadata.PackageNotFoundError(name)
    return original_distribution(name)
builtins.__import__ = guarded_import
importlib.metadata.distribution = distribution
import numpy as np
import numpy_groupies as npg
from ttsim.tt.aggregation import grouped_count
from _lcm.solution.fingerprint import _semantic_fingerprint
assert npg.aggregate.__module__ == "numpy_groupies.aggregate_numpy"
np.testing.assert_array_equal(grouped_count(np.array([0, 0, 1]), 2, "numpy"), [2, 2, 1])
assert _semantic_fingerprint(grouped_count) == _semantic_fingerprint(grouped_count)
print("numpy-fallback-supported")
"""
    output = subprocess.check_output(  # noqa: S603, fixed interpreter and literal script
        [sys.executable, "-c", script],
        text=True,
        timeout=120,
    )
    assert output.strip() == "numpy-fallback-supported"


def test_generated_gettsim_graph_has_repeatable_semantic_identity() -> None:
    """Independent grouped-income graphs agree in output and durable identity."""
    first = _build_income_function()
    second = _build_income_function()
    np.testing.assert_array_equal(
        first({k: v for k, v in _input_data().items() if k != "p_id"})["total"],
        [10.0, 10.0],
    )
    assert _semantic_fingerprint(first) == _semantic_fingerprint(second)


def test_real_kindergeld_policy_graph_has_numerical_and_durable_identity() -> None:
    """The selected 2025 child-benefit policy yields 255 for an eligible child."""
    inputs = {
        "alter": jnp.array([10]),
        "arbeitsstunden_w": jnp.array([0.0]),
        "kindergeld__in_ausbildung": jnp.array([False]),
        "kindergeld__p_id_empfänger": jnp.array([0]),
        "p_id": jnp.array([0]),
    }

    def build() -> Callable:
        return gettsim.main(
            main_target=gettsim.MainTarget.tt_function,
            policy_date_str="2025-01-01",
            input_data=gettsim.InputData.tree(dt.unflatten_from_qnames(inputs)),
            tt_targets=gettsim.TTTargets.qname(["kindergeld__betrag_m"]),
            backend="jax",
            include_fail_nodes=False,
            include_warn_nodes=False,
        )

    first = build()
    np.testing.assert_array_equal(first(inputs)["kindergeld__betrag_m"], [255])
    assert _semantic_fingerprint(first) == _semantic_fingerprint(build())


def test_real_pension_lookup_graph_has_numerical_and_durable_identity() -> None:
    """The selected 2025 pension lookup yields the age-65 tax share."""
    target = (
        "einkommensteuer__einkünfte__sonstige__rente__"
        "ertragsanteil_sonstige_private_vorsorge"
    )
    inputs = {
        "einkommensteuer__einkünfte__sonstige__rente__"
        "alter_beginn_leistungsbezug_sonstige_private_vorsorge": jnp.array([65])
    }
    build_data = {**inputs, "p_id": jnp.array([0])}

    def build() -> Callable:
        return gettsim.main(
            main_target=gettsim.MainTarget.tt_function,
            policy_date_str="2025-01-01",
            input_data=gettsim.InputData.tree(dt.unflatten_from_qnames(build_data)),
            tt_targets=gettsim.TTTargets.qname([target]),
            backend="jax",
            include_fail_nodes=False,
            include_warn_nodes=False,
        )

    first = build()
    np.testing.assert_allclose(first(inputs)[target], [0.18], rtol=1e-6)
    assert _semantic_fingerprint(first) == _semantic_fingerprint(build())


def test_real_wage_tax_polynomial_graph_has_numerical_and_durable_identity() -> None:
    """The selected 2025 wage-tax tariff yields its expected annual value."""
    target = "lohnsteuer__basistarif_y"
    inputs = {
        "alter": jnp.array([30]),
        "einnahmen__bruttolohn_m": jnp.array([3000.0]),
        "familie__p_id_elternteil_1": jnp.array([-1]),
        "familie__p_id_elternteil_2": jnp.array([-1]),
        "lohnsteuer__steuerklasse": jnp.array([1]),
        "p_id": jnp.array([0]),
        "sozialversicherung__pflege__beitrag__hat_kinder": jnp.array([False]),
    }

    def build() -> Callable:
        return gettsim.main(
            main_target=gettsim.MainTarget.tt_function,
            policy_date_str="2025-01-01",
            input_data=gettsim.InputData.tree(dt.unflatten_from_qnames(inputs)),
            tt_targets=gettsim.TTTargets.qname([target]),
            backend="jax",
            include_fail_nodes=False,
            include_warn_nodes=False,
        )

    first = build()
    expected = 3618.4880603278016 if jax.config.x64_enabled else 3618.7637
    np.testing.assert_allclose(first(inputs)[target], [expected], rtol=1e-6)
    assert _semantic_fingerprint(first) == _semantic_fingerprint(build())


def test_gettsim_lookup_parameters_bind_values_and_index_origins() -> None:
    """A compiled lookup binds its table entries and the integer index origin."""

    def lookup(*, index: int, table: Any) -> object:
        return table.look_up(index)

    def build(*, values: list[float], origin: int) -> Callable:
        bound = partial(
            lookup,
            table=param_objects.ConsecutiveIntLookupTableParamValue(
                xnp=jnp,
                values_to_look_up=jnp.array(values),
                bases_to_subtract=jnp.array([origin]),
            ),
        )

        def function(index: int) -> object:
            return bound(index=index)

        return jax.jit(function)

    baseline = build(values=[10.0, 20.0], origin=1)
    np.testing.assert_array_equal(baseline(1), 10.0)
    assert _semantic_fingerprint(baseline) == _semantic_fingerprint(
        build(values=[10.0, 20.0], origin=1)
    )
    assert _semantic_fingerprint(baseline) != _semantic_fingerprint(
        build(values=[11.0, 20.0], origin=1)
    )
    assert _semantic_fingerprint(baseline) != _semantic_fingerprint(
        build(values=[10.0, 20.0], origin=0)
    )


def test_unreviewed_partial_rejects_backend_module() -> None:
    """A bound backend module requires a supported generated wrapper."""

    def function(*, value: float, xnp: ModuleType) -> object:
        return xnp.sqrt(value)

    wrapped = partial(function, xnp=jnp)
    np.testing.assert_array_equal(wrapped(value=9.0), 3.0)
    with pytest.raises(TypeError, match="bound keyword argument 'xnp'"):
        _semantic_fingerprint(wrapped)


def test_policy_function_can_be_captured_directly() -> None:
    """A generated wrapper can retain its decorated policy callable."""

    @tt.policy_function(vectorization_strategy="not_required", unit=_UNIT)
    def policy(value: FloatColumn) -> FloatColumn:
        return value * 2

    def wrapper(value: float) -> float:
        return policy(value)

    np.testing.assert_array_equal(wrapper(3.0), 6.0)
    assert _semantic_fingerprint(wrapper) == _semantic_fingerprint(wrapper)


def test_rounded_policy_retains_policy_and_rounding_semantics() -> None:
    """Rounding a policy callable binds its closure and rounding rule."""

    def build(*, scale: float, base: float) -> Callable:
        @tt.policy_function(vectorization_strategy="not_required", unit=_UNIT)
        def policy(value: FloatColumn) -> FloatColumn:
            return value * scale

        return rounding.RoundingSpec(base=base, direction="up").apply_rounding(
            policy, jnp
        )

    baseline = build(scale=2.0, base=1.0)
    np.testing.assert_array_equal(baseline(jnp.array([1.1])), [3.0])
    assert _semantic_fingerprint(baseline) == _semantic_fingerprint(
        build(scale=2.0, base=1.0)
    )
    assert _semantic_fingerprint(baseline) != _semantic_fingerprint(
        build(scale=3.0, base=1.0)
    )
    assert _semantic_fingerprint(baseline) != _semantic_fingerprint(
        build(scale=2.0, base=2.0)
    )


def test_installed_policy_parameter_records_bind_nested_tables() -> None:
    """GETTSIM's frozen policy records retain all nested numerical parameters."""

    policy = importlib.import_module("gettsim.germany.wohngeld.wohngeld")
    table = param_objects.ConsecutiveIntLookupTableParamValue(
        xnp=jnp, values_to_look_up=jnp.array([1.0]), bases_to_subtract=jnp.array([0])
    )

    def function(*, value: float, params: Any) -> float:
        return value * params.skalierungsfaktor

    baseline = partial(
        function, params=policy.BasisformelParamValues(1.0, table, table, table)
    )
    changed = partial(
        function, params=policy.BasisformelParamValues(2.0, table, table, table)
    )
    assert _semantic_fingerprint(baseline) != _semantic_fingerprint(changed)


def test_generated_model_rejects_unreviewed_aggregation_dependency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Model construction rejects an unreviewed aggregation implementation."""
    original = importlib.metadata.distribution

    def distribution(name: str) -> object:
        installed = original(name)
        if name != "numpy-groupies":
            return installed
        return SimpleNamespace(
            version="0.0.0+unreviewed",
            locate_file=installed.locate_file,
            read_text=installed.read_text,
        )

    monkeypatch.setattr(importlib.metadata, "distribution", distribution)
    _clear_external_metadata_caches()
    try:
        with pytest.raises(
            ModelInitializationError, match="Unsupported numpy-groupies version"
        ):
            _build_lcm_model()
    finally:
        _clear_external_metadata_caches()


def test_unsupported_ttsim_release_has_no_durable_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A carrier adapter requires an implementation version it understands."""
    table = param_objects.ConsecutiveIntLookupTableParamValue(
        xnp=jnp, values_to_look_up=jnp.array([1.0]), bases_to_subtract=jnp.array([0])
    )
    contract = external_fingerprint._capture_ttsim_contract()
    assert contract is not None
    monkeypatch.setattr(
        external_fingerprint,
        "_capture_ttsim_contract",
        lambda: dataclasses.replace(contract, version="9.9"),
    )
    with pytest.raises(TypeError, match="Unsupported ttsim-backend version"):
        _semantic_fingerprint(table)


def test_editable_ttsim_installation_has_no_durable_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An editable backend requires an implementation receipt before durable use."""
    table = param_objects.ConsecutiveIntLookupTableParamValue(
        xnp=jnp, values_to_look_up=jnp.array([1.0]), bases_to_subtract=jnp.array([0])
    )
    original = importlib.metadata.distribution

    def distribution(name: str) -> object:
        installed = original(name)
        if name != "ttsim-backend":
            return installed
        return SimpleNamespace(
            version=installed.version,
            locate_file=installed.locate_file,
            read_text=lambda filename: (
                '{"dir_info": {"editable": true}}'
                if filename == "direct_url.json"
                else installed.read_text(filename)
            ),
        )

    monkeypatch.setattr(importlib.metadata, "distribution", distribution)
    external_fingerprint._installed_package.cache_clear()
    external_fingerprint._capture_ttsim_contract.cache_clear()
    try:
        with pytest.raises(TypeError, match="Editable ttsim-backend installation"):
            _semantic_fingerprint(table)
    finally:
        external_fingerprint._installed_package.cache_clear()
        external_fingerprint._capture_ttsim_contract.cache_clear()


def test_unsupported_gettsim_release_has_no_durable_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A policy record requires an implementation version it understands."""
    policy = importlib.import_module("gettsim.germany.wohngeld.wohngeld")
    table = param_objects.ConsecutiveIntLookupTableParamValue(
        xnp=jnp, values_to_look_up=jnp.array([1.0]), bases_to_subtract=jnp.array([0])
    )
    record = policy.BasisformelParamValues(1.0, table, table, table)
    original = external_fingerprint._installed_package

    def changed_version(
        *, distribution_name: str, package: str
    ) -> tuple[Path | None, str]:
        root, version = original(distribution_name=distribution_name, package=package)
        if distribution_name == "gettsim":
            version = "9.9"
        return root, version

    monkeypatch.setattr(external_fingerprint, "_installed_package", changed_version)
    with pytest.raises(TypeError, match="Unsupported gettsim version"):
        _semantic_fingerprint(record)


def test_rounding_wrapper_survives_serialization() -> None:
    """Serialized generated wrappers keep their durable semantic identity."""

    def body(value: float) -> float:
        return value * 2

    # Model a generated policy module with ordinary, serializable globals.
    function = FunctionType(body.__code__, {"__name__": "policy_fixture"})
    function.__annotations__ = {"value": "FloatColumn", "return": "FloatColumn"}
    policy = tt.policy_function(vectorization_strategy="not_required", unit=_UNIT)(
        function
    )

    wrapped = rounding.RoundingSpec(base=1.0, direction="up").apply_rounding(
        policy, jnp
    )
    restored = cloudpickle.loads(cloudpickle.dumps(wrapped))
    assert _semantic_fingerprint(wrapped) == _semantic_fingerprint(restored)


def test_external_record_rejects_mutable_nested_objects() -> None:

    @dataclasses.dataclass
    class Mutable:
        scale: float

    policy = importlib.import_module("gettsim.germany.wohngeld.wohngeld")

    def function(params: Any) -> object:
        return params.skalierungsfaktor

    table = param_objects.ConsecutiveIntLookupTableParamValue(
        xnp=jnp, values_to_look_up=jnp.array([1.0]), bases_to_subtract=jnp.array([0])
    )
    record = policy.BasisformelParamValues(2.0, table, table, table)
    object.__setattr__(record, "skalierungsfaktor", Mutable(2.0))
    with pytest.raises(TypeError, match="durably fingerprint"):
        _semantic_fingerprint(partial(function, params=record))


@pytest.mark.parametrize(
    "changes",
    [
        {"scale": 3.0},
        {"year": 2019},
        {"targets": ("income", "total")},
        {"start_date": "2000-01-01"},
    ],
)
def test_generated_graph_binds_policy_and_output_declarations(
    changes: dict[str, Any],
) -> None:
    """Policy coefficients, dates and target schemas each participate in identity."""
    baseline = _build_income_function()
    changed = _build_income_function(**changes)
    assert _semantic_fingerprint(baseline) != _semantic_fingerprint(changed)


def test_generated_graph_ignores_declaration_insertion_order() -> None:
    """Equivalent unordered policy environments have one identity."""
    assert _semantic_fingerprint(_build_income_function()) == _semantic_fingerprint(
        _build_income_function(reverse=True)
    )


def test_foreign_key_declarations_have_distinct_identities() -> None:
    """The three foreign-key meanings remain distinguishable for the same body."""

    def key(person_id: IntColumn) -> IntColumn:
        return person_id

    declarations = [
        tt.policy_function(
            foreign_key_type=kind, vectorization_strategy="not_required", unit=_UNIT
        )(key)
        for kind in tt.FKType
    ]
    assert (
        len({_semantic_fingerprint(declaration) for declaration in declarations}) == 3
    )


@pytest.mark.parametrize(
    "opaque", [object(), SimpleNamespace(scale=2.0), ModuleType("jax.numpy")]
)
def test_generated_policy_rejects_unsupported_captured_objects(opaque: object) -> None:
    """Generated policy wrappers cannot hide opaque state or fake backend modules."""

    @tt.policy_function(vectorization_strategy="not_required", unit=_UNIT)
    def policy(_value: FloatColumn) -> object:
        return opaque

    with pytest.raises(TypeError, match="durably fingerprint"):
        _semantic_fingerprint(policy)


def test_library_name_claim_does_not_hide_user_function_state() -> None:
    """An infrastructure module name grants no trust to a user-created function."""
    state = SimpleNamespace(scale=2.0)

    def function(_value: float) -> object:
        return state

    function.__module__ = "dags.tree.tree_utils"
    function.__qualname__ = "flatten_to_qnames"
    with pytest.raises(TypeError, match="durably fingerprint"):
        _semantic_fingerprint(function)


def test_unreviewed_installed_library_function_has_no_durable_identity() -> None:
    """A genuine library function needs a reviewed operation contract."""
    with pytest.raises(TypeError, match="Unreviewed external operation"):
        _semantic_fingerprint(tt.policy_function)


def test_adapter_does_not_discover_new_declaration_classes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The declaration contract contains only reviewed TTSIM classes."""
    columns = importlib.import_module("ttsim.tt.column_objects_param_function")

    @dataclasses.dataclass(frozen=True)
    class SyntheticColumn(columns.ColumnObject):
        pass

    monkeypatch.setattr(columns, "SyntheticColumn", SyntheticColumn, raising=False)
    external_fingerprint._capture_ttsim_contract.cache_clear()
    try:
        contract = external_fingerprint._capture_ttsim_contract()
        assert contract is not None
        assert SyntheticColumn not in contract.columns
    finally:
        external_fingerprint._capture_ttsim_contract.cache_clear()


def test_optional_ttsim_adapter_loads_only_when_used() -> None:
    """Importing the core fingerprint walker leaves optional TTSIM modules unloaded."""
    script = (
        "import sys; import _lcm.solution.fingerprint; "
        "print('ttsim.tt.param_objects' in sys.modules)"
    )
    loaded = subprocess.check_output(  # noqa: S603, fixed interpreter and literal script
        [sys.executable, "-c", script], text=True
    ).strip()
    assert loaded == "False"


def test_generated_identity_agrees_between_fresh_processes() -> None:
    """Process-local hashes and object addresses do not enter generated identity."""
    script = (
        "from tests.solution.test_gettsim_fingerprint import _build_income_function; "
        "from _lcm.solution.fingerprint import _semantic_fingerprint; "
        "print(_semantic_fingerprint(_build_income_function()))"
    )
    digests = [
        subprocess.check_output(  # noqa: S603, fixed interpreter and literal script
            [sys.executable, "-c", script],
            text=True,
            env={
                **os.environ,
                "PYTHONHASHSEED": seed,
                "JAX_ENABLE_X64": str(jax.config.x64_enabled).lower(),
            },
        ).strip()
        for seed in ("17", "91")
    ]
    assert len(digests[0]) == 64
    assert digests[0] == digests[1]


def test_generated_gettsim_solution_replays_only_in_an_equivalent_model(
    tmp_path: Path,
) -> None:
    """An archived GETTSIM-backed solution retains its mathematical model identity."""
    model = _build_lcm_model()
    params = {
        "working": {"koopmans_aggregator": {"discount_factor": 0.9}},
        "retired": {},
    }
    solution = model.solve(params=params, log_level="debug")
    np.testing.assert_allclose(solution.values[0]["working"], [4.0, 8.0])
    solution.save(path=tmp_path / "solution")
    loaded = load_solution(path=tmp_path / "solution")
    initial = {
        "wealth": jnp.array([1.0, 2.0]),
        "regime_id": jnp.array([0, 0]),
        "age": jnp.array([0.0, 0.0]),
    }
    replay = _build_lcm_model().simulate(
        params=params, initial_conditions=initial, solution=loaded, log_level="debug"
    )
    np.testing.assert_allclose(
        np.sort(replay.to_dataframe(additional_targets=["utility"])["utility"]),
        [0.0, 0.0, 4.0, 8.0],
    )
    with pytest.raises(InvalidSimulationInputError, match="fingerprint"):
        _build_lcm_model(scale=3.0).simulate(
            params=params,
            initial_conditions=initial,
            solution=loaded,
            log_level="debug",
        )


@pytest.mark.parametrize("field", ["thresholds", "intercepts", "coefficients"])
def test_polynomial_parameters_bind_each_numerical_field(field: str) -> None:
    """Every polynomial axis and coefficient is part of the parameter identity."""
    parameter = param_objects.PiecewisePolynomialParamValue(
        thresholds=jnp.array([0.0, 1.0, 2.0]),
        intercepts=jnp.array([10.0, 20.0]),
        coefficients=jnp.array([[1.0], [2.0]]),
    )
    changed = dataclasses.replace(parameter, **{field: getattr(parameter, field) + 1})

    def evaluate(*, value: float, params: Any) -> object:
        return params.intercepts[0] + params.coefficients[0, 0] * value

    baseline = partial(evaluate, params=parameter)
    np.testing.assert_array_equal(baseline(value=0.5), 10.5)
    assert _semantic_fingerprint(baseline) != _semantic_fingerprint(
        partial(evaluate, params=changed)
    )


@pytest.mark.parametrize("defect", ["extra-field", "non-jax-array"])
def test_polynomial_parameters_reject_unsupported_state(defect: str) -> None:
    """Exact parameter types provide no escape hatch for opaque or mutable data."""
    parameter = param_objects.PiecewisePolynomialParamValue(
        thresholds=jnp.array([0.0, 1.0]),
        intercepts=jnp.array([10.0]),
        coefficients=jnp.array([[1.0]]),
    )
    if defect == "extra-field":
        object.__setattr__(parameter, "opaque", object())
    else:
        object.__setattr__(parameter, "intercepts", np.array([10.0]))
    with pytest.raises(TypeError, match="durably fingerprint"):
        _semantic_fingerprint(parameter)


def test_unit_declarations_have_distinct_identities() -> None:
    """A declaration's unit participates in its identity."""

    def amount(value: FloatColumn) -> FloatColumn:
        return value

    first, second = (
        tt.policy_function(vectorization_strategy="not_required", unit=unit)(amount)
        for unit in (tt.TTSIMUnit.CURRENCY, tt.TTSIMUnit.CURRENCY.PER_MONTH)
    )
    assert _semantic_fingerprint(first) != _semantic_fingerprint(second)


def test_typed_forwarder_is_identified_by_its_callee() -> None:
    """The generated column-typed forwarder binds the callable it forwards to."""
    assert _semantic_fingerprint(_typed_forwarder(scale=2.0)) != (
        _semantic_fingerprint(_typed_forwarder(scale=3.0))
    )


def test_typed_forwarder_rejects_a_modified_body() -> None:
    """A forwarder whose generated body does more than forward is refused."""
    guard = _typed_forwarder(scale=2.0)
    forwarder = guard.__dict__.get("__wrapped__", guard)
    source = "def _typed_body(value):\n    return _ttsim_wrapped_impl(value) + 1\n"
    code = next(
        constant
        for constant in compile(source, "<ttsim-typed-wrapper>", "exec").co_consts
        if isinstance(constant, CodeType)
    )
    modified = FunctionType(code, forwarder.__globals__, "body")
    modified.__module__ = "ttsim.typing"
    with pytest.raises(TypeError, match="modified ttsim typed forwarder"):
        _semantic_fingerprint(modified)


def test_typed_forwarder_guard_must_regenerate_from_its_hints() -> None:
    """A guard whose recorded hints do not reproduce its checks is refused."""
    guard = _typed_forwarder(scale=2.0)
    if guard.__dict__.get("__beartype_wrapper") is not True:
        pytest.skip("ttsim runs without its import-time beartype claw.")
    forged = FunctionType(guard.__code__, guard.__globals__, guard.__name__)
    forged.__kwdefaults__ = dict(guard.__kwdefaults__)
    forged.__dict__.update(guard.__dict__)
    forged.__dict__["__beartype_annotations"] = {"value": int, "return": int}
    with pytest.raises(TypeError, match="does not regenerate"):
        _semantic_fingerprint(forged)


def _typed_forwarder(*, scale: float) -> Any:
    def body(value: FloatColumn) -> FloatColumn:
        return value * scale

    return type_resolution.build_beartype_checkable_wrapper(
        body,
        annotations={"value": "FloatColumn", "return": "FloatColumn"},
        node_name="body",
    )


@pytest.mark.parametrize(
    ("target", "inputs", "expected"),
    [
        pytest.param(
            "sozialversicherung__rente__alter_bei_renteneintritt",
            {
                "geburtsjahr": [1960],
                "geburtsmonat": [1],
                "sozialversicherung__rente__jahr_renteneintritt": [2025],
                "sozialversicherung__rente__monat_renteneintritt": [7],
            },
            [65 + 5 / 12],
            id="months-to-years",
        ),
        pytest.param(
            "sozialversicherung__rente__altersrente__langjährig__altersgrenze",
            {"geburtsjahr": [1965], "geburtsmonat": [1]},
            [67.0],
            id="years-to-months",
        ),
        pytest.param(
            "anzahl_personen_hh",
            {"hh_id": [0, 0, 1]},
            [2, 2, 1],
            id="group-count",
        ),
        pytest.param(
            "familie__alter_monate_jüngstes_mitglied_fg",
            {"alter_monate": [400, 30, 500], "fg_id": [0, 0, 1]},
            [30, 30, 500],
            id="group-minimum",
        ),
        pytest.param(
            "familie__alleinerziehend_sn",
            {"familie__alleinerziehend": [True, False, False], "sn_id": [0, 0, 1]},
            [True, True, False],
            id="group-any",
        ),
        pytest.param(
            "ehe_id",
            {"p_id": [0, 1, 2], "familie__p_id_ehepartner": [1, 0, -1]},
            [0, 0, 1],
            id="reordered-group-creation",
        ),
        pytest.param(
            "familie__ist_kind_in_familiengemeinschaft",
            {
                "p_id": [0, 1, 2],
                "familie__p_id_elternteil_1": [-1, 0, 5],
                "familie__p_id_elternteil_2": [-1, -1, -1],
                "fg_id": [0, 0, 1],
            },
            [False, True, False],
            id="foreign-key-join",
        ),
    ],
)
def test_real_policy_graph_operations_have_numerical_and_durable_identity(
    *, target: str, inputs: dict[str, list[float]], expected: list[float]
) -> None:
    """Reviewed time, group and join operations reached by 2025 GETTSIM graphs."""
    n_obs = len(next(iter(inputs.values())))
    data = {name: jnp.array(values) for name, values in inputs.items()}
    build_data = {"p_id": jnp.arange(n_obs), **data}

    def build() -> Callable:
        return gettsim.main(
            main_target=gettsim.MainTarget.tt_function,
            policy_date_str="2025-01-01",
            input_data=gettsim.InputData.tree(dt.unflatten_from_qnames(build_data)),
            tt_targets=gettsim.TTTargets.qname([target]),
            backend="jax",
            include_fail_nodes=False,
            include_warn_nodes=False,
        )

    first = build()
    np.testing.assert_allclose(first(data)[target], expected, rtol=1e-6)
    assert _semantic_fingerprint(first) == _semantic_fingerprint(build())


def _monthly_via_module(value: float) -> object:
    return tt.cast_ttsim_unit(value, unit=tt.TTSIMUnit.CURRENCY.PER_MONTH)


def _yearly_via_module(value: float) -> object:
    return tt.cast_ttsim_unit(value, unit=tt.TTSIMUnit.CURRENCY.PER_YEAR)


def _monthly_via_namespace(value: float) -> object:
    return tt.cast_ttsim_unit(value, unit=_UNIT_NAMESPACE.CURRENCY.PER_MONTH)


def _yearly_via_namespace(value: float) -> object:
    return tt.cast_ttsim_unit(value, unit=_UNIT_NAMESPACE.CURRENCY.PER_YEAR)


@pytest.mark.parametrize(
    ("monthly", "yearly"),
    [
        pytest.param(_monthly_via_module, _yearly_via_module, id="module"),
        pytest.param(_monthly_via_namespace, _yearly_via_namespace, id="namespace"),
    ],
)
def test_units_spelled_in_policy_code_bind_identity(
    *, monthly: Callable, yearly: Callable
) -> None:
    """A unit spelled off the builder namespace inside a body is bound by value."""
    assert _semantic_fingerprint(monthly) != _semantic_fingerprint(yearly)


def _hours_per_year_from_weekly(hours: float) -> float:
    return time_converters.per_w_to_per_y(hours)


def _hours_per_year_from_monthly(hours: float) -> float:
    return time_converters.per_m_to_per_y(hours)


def test_weekly_flow_conversion_binds_its_period() -> None:
    """Weekly and monthly flow conversions called from model code are told apart."""
    assert _semantic_fingerprint(_hours_per_year_from_weekly) != (
        _semantic_fingerprint(_hours_per_year_from_monthly)
    )


def test_policy_record_from_a_reexecuted_module_keeps_its_identity() -> None:
    """A record survives GETTSIM re-executing its policy module on a later load."""
    stale = _wohngeld_record()
    gettsim.main(
        main_target=gettsim.MainTarget.policy_environment,
        policy_date_str="2025-01-01",
    )
    current = _wohngeld_record()
    assert type(stale) is not type(current)
    assert _semantic_fingerprint(stale) == _semantic_fingerprint(current)


def test_policy_record_imitating_a_reexecuted_module_is_not_a_policy_record() -> None:
    """A record whose class only borrows a GETTSIM name is not a policy record."""
    sample = _wohngeld_record()
    genuine = cast("Any", type(sample))
    imitation = dataclasses.make_dataclass(
        genuine.__name__,
        [(field.name, field.type) for field in dataclasses.fields(genuine)],
        frozen=True,
        namespace={"skaliert": lambda self: self.skalierungsfaktor * 2},
    )
    imitation.__module__ = genuine.__module__
    record = imitation(
        *(getattr(sample, field.name) for field in dataclasses.fields(genuine))
    )
    assert external_fingerprint.external_policy_record_version(record) is None


def _clear_external_metadata_caches() -> None:
    """Refresh installed-package observations between metadata boundary controls."""
    for value in vars(external_fingerprint).values():
        clear = getattr(value, "cache_clear", None)
        if callable(clear):
            clear()


def _resolve_external_operation(operation: str) -> object:
    """Resolve an exported function or class method at its installed location."""
    parts = operation.split(".")
    for stop in range(len(parts) - 1, 0, -1):
        module_name = ".".join(parts[:stop])
        try:
            value = importlib.import_module(module_name)
        except ModuleNotFoundError as error:
            if error.name is None or not (
                error.name == module_name or module_name.startswith(error.name + ".")
            ):
                raise
            continue
        for name in parts[stop:]:
            value = getattr(value, name)
        return value
    raise AssertionError(f"No installed export {operation}.")


def _replace_installation_metadata(
    *, dependency: str, defect: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Substitute metadata at the installed-distribution boundary."""
    original = importlib.metadata.distribution

    def distribution(name: str) -> object:
        if name == dependency and defect == "missing":
            raise importlib.metadata.PackageNotFoundError(name)
        installed = original(name)
        if name != dependency:
            return installed

        def read_text(filename: str) -> str | None:
            if filename == "direct_url.json" and defect == "editable":
                return '{"dir_info":{"editable":true}}'
            return installed.read_text(filename)

        return SimpleNamespace(
            version="0.0.0+unreviewed"
            if defect == "unsupported"
            else installed.version,
            locate_file=installed.locate_file,
            read_text=read_text,
        )

    monkeypatch.setattr(importlib.metadata, "distribution", distribution)


def _wohngeld_record() -> object:
    policy = importlib.import_module("gettsim.germany.wohngeld.wohngeld")
    table = param_objects.ConsecutiveIntLookupTableParamValue(
        xnp=jnp, values_to_look_up=jnp.array([1.0]), bases_to_subtract=jnp.array([0])
    )
    return policy.BasisformelParamValues(1.0, table, table, table)


@categorical(ordered=False)
class _RegimeId:
    working: ScalarInt
    retired: ScalarInt


def _build_lcm_model(*, scale: float = 2.0) -> Model:
    function = _build_income_function(scale=scale)

    def utility(wealth: ScalarFloat) -> ScalarFloat:
        return function({"wage": jnp.repeat(wealth, 2), "hh_id": jnp.array([0, 0])})[
            "total"
        ][0]

    return Model(
        ages=AgeGrid(start=0, inclusive_stop=1, step="Y"),
        regime_id_class=_RegimeId,
        initial_nodes=((0, "working"),),
        edges={"working": {"retired": 0}},
        regimes={
            "working": Regime(
                states={"wealth": LinSpacedGrid(start=1.0, stop=2.0, n_points=2)},
                state_transitions={"wealth": fixed_transition("wealth")},
                functions={"utility": utility},
            ),
            "retired": Regime(
                states={"wealth": LinSpacedGrid(start=1.0, stop=2.0, n_points=2)},
                functions={"utility": lambda wealth: wealth * 0.0},
            ),
        },
    )


def _input_data() -> dict[str, object]:
    return {
        "p_id": jnp.array([0, 1]),
        "hh_id": jnp.array([0, 0]),
        "wage": jnp.array([2.0, 3.0]),
    }


def _build_income_function(
    *,
    scale: float = 2.0,
    year: int = 2018,
    targets: tuple[str, ...] = ("total",),
    reverse: bool = False,
    start_date: str = "1900-01-01",
) -> Callable:
    def income(*, wage: float, policy_year: int) -> float:
        return wage * scale + (policy_year - 2018)

    income.__annotations__ = {
        "wage": "FloatColumn",
        "policy_year": int,
        "return": "FloatColumn | IntColumn | BoolColumn",
    }
    policy_income = tt.policy_function(
        vectorization_strategy="not_required", start_date=start_date, unit=_UNIT
    )(income)

    @tt.agg_by_group_function(agg_type=tt.AggType.SUM, unit=_UNIT)
    def total(*, income: float, hh_id: int) -> float:  # noqa: ARG001
        # GETTSIM consumes this declaration's signature.
        """Declare the household sum of income."""
        raise AssertionError("GETTSIM replaces aggregation declaration bodies.")

    environment = {
        "policy_year": year,
        "policy_month": 1,
        "policy_day": 1,
        "income": policy_income,
        "total": total,
    }
    if reverse:
        environment = dict(reversed(tuple(environment.items())))
    return gettsim.main(
        main_target=gettsim.MainTarget.tt_function,
        policy_environment=environment,
        policy_date_str=f"{year}-01-01",
        input_data=gettsim.InputData.tree(_input_data()),
        tt_targets=gettsim.TTTargets.tree(dict.fromkeys(targets, True)),
        include_warn_nodes=False,
        include_fail_nodes=False,
        backend="jax",
    )
