"""Original constructor prefix shares entry admission; dated factories stay closed."""

import sys
from dataclasses import replace
from types import MappingProxyType
from typing import Never

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import _lcm.grids.discrete as discrete_module
from _lcm import model_processing
from _lcm.execution.execution_plan import (
    resolve_execution_config,
    resolve_execution_hardware,
)
from _lcm.grids import DiscreteGrid
from _lcm.regime_building.fixed_regime_support import _evaluate_fixed_function
from _lcm.regime_building.value_adapter import collect_value_grids
from _lcm.simulation.declaration_inputs import (
    _DeclarationRootObserver,
    open_declaration_allocations,
)
from _lcm.time import ModelTime
from _lcm.typing import PytreeValue
from lcm import ExecutionConfig, Model, Regime
from lcm.exceptions import (
    ExecutionPlanningError,
    ModelInitializationError,
    UnsupportedOperationError,
)
from lcm.model import _ModelConstructorInputs
from lcm.typing import FloatND, ScalarInt
from tests.egm.test_current_iid_call_owned_generation import _declarations, _params
from tests.egm.test_current_iid_generation_materialization import _owner
from tests.egm.test_current_iid_parent_demands import _RegimeId

pytestmark = pytest.mark.coverage(backends=("cpu",), precisions="both")

_GLOBAL_ONLY_ROOT: FloatND | None = None


def _original_log1p_declaration(*, value: FloatND) -> FloatND:
    return jnp.log1p(value)


def _passive_callback_must_not_execute(*, _value: FloatND) -> Never:
    raise AssertionError("Passive declaration custody invoked its callback.")


@pytest.mark.parametrize("weak", [False, True])
def test_passive_original_jit_keeps_roots_and_original_numeric_boundary(
    *, weak: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Inventory canonical callbacks without tracing or executing declarations."""
    from _lcm.egm import validation_stages  # noqa: PLC0415

    value = jnp.asarray(0.25) if weak else jnp.asarray([0.25])
    expected = _original_log1p_declaration(value=value)
    expected_bytes = np.asarray(expected).tobytes()
    root = jnp.ones((17,))
    monkeypatch.setattr(sys.modules[__name__], "_GLOBAL_ONLY_ROOT", root)
    config = ExecutionConfig(device_memory_bytes=1 << 24)
    hardware = resolve_execution_hardware(
        config=config,
        visible_device_ids=(jax.devices("cpu")[0].id,),
        device_pool_limit_bytes={},
    )

    def must_not_trace_or_build(*_args: PytreeValue, **_kwargs: PytreeValue) -> Never:
        raise AssertionError("Passive declaration custody traced or built values.")

    with monkeypatch.context() as patch:
        patch.setattr(jax, "make_jaxpr", must_not_trace_or_build)
        patch.setattr(
            validation_stages, "prepare_validation_recipe", must_not_trace_or_build
        )
        patch.setattr(np, "asarray", must_not_trace_or_build)
        owner = open_declaration_allocations(
            hardware=hardware,
            declarations=(
                _original_log1p_declaration,
                _global_only_callback,
                _passive_callback_must_not_execute,
            ),
        )
    assert owner is not None
    try:
        assert owner.original_inputs is not None
        assert any(array is root for array in owner.original_inputs.arrays)
        roots: list[jax.Array] = []
        _DeclarationRootObserver(array_recorder=roots.append).visit(
            value=(_original_log1p_declaration, _global_only_callback)
        )
        assert any(array is root for array in roots)
        actual = _original_log1p_declaration(value=value)
        assert actual.dtype == expected.dtype
        assert actual.weak_type == expected.weak_type
        assert np.asarray(actual).tobytes() == expected_bytes
    finally:
        owner.close()


@pytest.mark.parametrize("mutation", ["fun", "info", "code", "rebound", "forged"])
def test_passive_jit_original_origin_refuses_changed_or_foreign_callback(
    *, mutation: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Passive census keeps exact native authority, not arbitrary EARLY trust."""
    import inspect  # noqa: PLC0415
    from types import FunctionType  # noqa: PLC0415

    from jax._src.api import Inline  # noqa: PLC0415

    original = jnp.log1p
    raw = vars(original)["_fun"]
    assert isinstance(raw, FunctionType)
    if mutation == "fun":
        monkeypatch.setattr(original, "_fun", _global_only_callback)
    elif mutation == "info":
        monkeypatch.setattr(original, "_jit_info", None)
    elif mutation == "code":
        replacement_raw = inspect.unwrap(_passive_callback_must_not_execute)
        assert isinstance(replacement_raw, FunctionType)
        monkeypatch.setattr(raw, "__code__", replacement_raw.__code__)
    else:
        replacement = jax.jit(_global_only_callback, inline=Inline.JAX_EARLY)
        if mutation == "forged":
            monkeypatch.setattr(replacement, "__module__", original.__module__)
            monkeypatch.setattr(replacement, "__qualname__", original.__qualname__)
        monkeypatch.setattr(jnp, "log1p", replacement)
    with pytest.raises(ExecutionPlanningError, match=r"JAX_EARLY|Original JIT"):
        _DeclarationRootObserver().visit(value=_original_log1p_declaration)


def _bare_model(declarations: _ModelConstructorInputs) -> Model:
    """Return an unbuilt Model that retains `declarations` as its constructor."""
    model = Model.__new__(Model)
    model._constructor_inputs = declarations
    return model


def test_passive_canonical_jit_does_not_admit_executable_validator_early_stage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The unchanged canonical callback still has its executable-stage refusal."""
    from _lcm.egm import validation_stages  # noqa: PLC0415

    def must_not_trace(**_: PytreeValue) -> Never:
        raise AssertionError("Canonical EARLY validator traced before refusal.")

    monkeypatch.setattr(validation_stages, "prepare_validation_recipe", must_not_trace)
    value = jnp.asarray(0.25)
    owner = _owner(roots=(value,))
    try:
        with pytest.raises(ExecutionPlanningError, match="JAX_EARLY"):
            owner.evaluate_validation(
                function=_original_log1p_declaration, arguments={"value": value}
            )
        assert not owner.operations.cache
        assert not owner._pending
    finally:
        owner.close()


def _global_only_callback(*, value: FloatND) -> FloatND:
    if _GLOBAL_ONLY_ROOT is None:
        raise ValueError("The original callback buffer is not bound.")
    return value + _GLOBAL_ONLY_ROOT


def _referenced_category_code() -> ScalarInt:
    return _RegimeId.child


def _direct_category_construction() -> ScalarInt:
    return _RegimeId(
        parent=_RegimeId.parent, child=_RegimeId.child, done=_RegimeId.done
    ).child


def test_explicit_category_slot_keeps_actual_codes_without_class_behavior() -> None:
    """Declared category codes are owned; direct class behavior stays refused."""
    config = ExecutionConfig(device_memory_bytes=1 << 24)
    hardware = resolve_execution_hardware(
        config=config,
        visible_device_ids=(jax.devices("cpu")[0].id,),
        device_pool_limit_bytes={},
    )
    owner = open_declaration_allocations(
        hardware=hardware,
        declarations=(_RegimeId, _referenced_category_code),
        regime_id_class=_RegimeId,
    )
    assert owner is not None
    try:
        assert owner.original_inputs is not None
        for code in (_RegimeId.parent, _RegimeId.child, _RegimeId.done):
            assert any(root is code for root in owner.original_inputs.arrays)
    finally:
        owner.close()
    # An explicit declaration slot does not authorize using that class as code.
    with pytest.raises(TypeError, match="direct class dependency"):
        open_declaration_allocations(
            hardware=hardware,
            declarations=(_RegimeId, _direct_category_construction),
            regime_id_class=_RegimeId,
        )
    with pytest.raises(TypeError, match="direct class dependency"):
        _DeclarationRootObserver().visit(value=_RegimeId)


def test_passive_constructor_roots_include_referenced_global_and_existing_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = jnp.ones((17,))
    grid = DiscreteGrid(values=np.asarray([0.125, 0.75, 2.0]))
    cached = grid.materialized_values
    assert isinstance(cached, jax.Array)
    (nodes,) = grid.materialized_array_roots
    midpoints = nodes.midpoints
    assert isinstance(midpoints, jax.Array)
    monkeypatch.setattr(sys.modules[__name__], "_GLOBAL_ONLY_ROOT", root)
    config = ExecutionConfig(device_memory_bytes=1 << 24)
    hardware = resolve_execution_hardware(
        config=config,
        visible_device_ids=(jax.devices("cpu")[0].id,),
        device_pool_limit_bytes={},
    )

    def must_not_copy_or_build(**_: PytreeValue) -> Never:
        raise AssertionError("Passive constructor custody invoked a numeric factory.")

    with monkeypatch.context() as patch:
        patch.setattr(np, "asarray", must_not_copy_or_build)
        patch.setattr(discrete_module, "_build_nodes", must_not_copy_or_build)
        owner = open_declaration_allocations(
            hardware=hardware, declarations=(_global_only_callback, grid)
        )
    assert owner is not None
    try:
        assert owner.original_inputs is not None
        assert any(value is root for value in owner.original_inputs.arrays)
        assert any(value is cached for value in owner.original_inputs.arrays)
        assert any(value is midpoints for value in owner.original_inputs.arrays)
        owner.publish(stage="params", tree=MappingProxyType({}))
        assert any(
            value is root for value in jax.tree.leaves(owner.solve_input_roots())
        )
        assert any(
            value is cached for value in jax.tree.leaves(owner.solve_input_roots())
        )
        assert any(
            value is midpoints for value in jax.tree.leaves(owner.solve_input_roots())
        )
        # Omitting the midpoint bank would leave room for this next upload.
        owner.budget_bytes = root.nbytes + cached.nbytes + midpoints.nbytes - 1
        with pytest.raises(ExecutionPlanningError, match="before allocation"):
            owner(value=np.ones(1), dtype=np.dtype(root.dtype), name="next_parameter")
    finally:
        owner.close()


def test_missing_precision_cache_refuses_before_original_node_factory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    current = bool(jax.config.x64_enabled)
    with jax.enable_x64(new_val=not current):
        grid = DiscreteGrid(values=np.asarray([0.125, 0.75, 2.0]))
    assert grid.materialized_values is None
    roots: list[jax.Array] = []

    def must_not_build(**_: PytreeValue) -> Never:
        raise AssertionError("Missing precision cache was populated before admission.")

    monkeypatch.setattr(discrete_module, "_build_nodes", must_not_build)
    _DeclarationRootObserver(array_recorder=roots.append).visit(value=grid)
    assert roots
    owner = _owner(roots=tuple(roots))
    try:
        with pytest.raises(ExecutionPlanningError, match="precision-cache"):
            collect_value_grids(
                regimes={}, derived_categoricals={"signal": grid}, materializer=owner
            )
        assert grid.materialized_values is None
    finally:
        owner.close()


def test_actual_private_prefix_refuses_residency_before_declaration_effects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    declarations = replace(
        _declarations(runtime_child_grid=True),
        execution_config=ExecutionConfig(device_memory_bytes=1),
    )

    def must_not_prepare(self: Model, **_: PytreeValue) -> Never:
        del self
        raise AssertionError("Declaration preparation preceded source admission.")

    monkeypatch.setattr(Model, "_prepare_declarations", must_not_prepare)
    with pytest.raises(ExecutionPlanningError, match="budget before allocation"):
        Model._from_call_owned_params(
            declarations=declarations,
            params=_params(child_points=np.asarray([1.0, 12.5])),
        )


def _unsupported_fixed_probability(*, value: FloatND) -> FloatND:
    return jnp.sin(value)


def _invalid_fixed_probability(*, value: FloatND) -> FloatND:
    del value
    raise ValueError("An ordinary invalid fixed law proves no structural zero.")


def _terminal_utility() -> float:
    return 0.0


def test_fixed_zero_pruning_preserves_ordinary_error_but_propagates_admission() -> None:
    regime = Regime(functions={"utility": _terminal_utility})
    fixed = MappingProxyType({"r__law__value": np.asarray(0.5)})
    owner = _owner()
    try:
        assert (
            _evaluate_fixed_function(
                func=_invalid_fixed_probability,
                path=("r", "law"),
                side="solve",
                regime=regime,
                fixed_flat=fixed,
                ancestors=(),
            )
            is None
        )
        assert (
            _evaluate_fixed_function(
                func=_invalid_fixed_probability,
                materializer=owner,
                path=("r", "law"),
                side="solve",
                regime=regime,
                fixed_flat=fixed,
                ancestors=(),
            )
            is None
        )
        with pytest.raises(ExecutionPlanningError, match="JAX_EARLY"):
            _evaluate_fixed_function(
                func=_unsupported_fixed_probability,
                materializer=owner,
                path=("r", "law"),
                side="solve",
                regime=regime,
                fixed_flat=fixed,
                ancestors=(),
            )
    finally:
        owner.close()


def test_hardware_prefix_reuses_pool_rule_and_final_name_validation_order() -> None:
    config = ExecutionConfig(
        devices=(3,),
        device_memory_bytes=2048,
        device_memory_headroom_fraction=0.25,
    )
    hardware = resolve_execution_hardware(
        config=config, visible_device_ids=(7, 3), device_pool_limit_bytes={3: 1024}
    )
    plain = resolve_execution_config(
        config=config,
        visible_device_ids=(7, 3),
        device_pool_limit_bytes={3: 1024},
        state_names=frozenset(),
    )
    prepared = resolve_execution_config(
        config=config,
        visible_device_ids=(7, 3),
        device_pool_limit_bytes={3: 1024},
        state_names=frozenset(),
        hardware=hardware,
    )
    assert prepared == plain
    assert prepared.device_memory_bytes == 768
    bad = ExecutionConfig(devices=(9,), sharded_states=("missing",))
    with pytest.raises(ExecutionPlanningError, match="sharded_states"):
        resolve_execution_config(
            config=bad, visible_device_ids=(3,), state_names=frozenset()
        )


def test_cold_metadata_uses_actual_prefix_owner_and_declared_support() -> None:
    """Reach the real fixed/schema builder without replacing it with a test hook.

    This prefix-only control establishes the preprogram connection. Complete
    budgeted construction and solving have their separate genuine Model controls.
    """
    declarations = replace(
        _declarations(),
        execution_config=ExecutionConfig(
            devices=(jax.devices("cpu")[0].id,), device_memory_bytes=1 << 28
        ),
    )
    hardware = resolve_execution_hardware(
        config=declarations.execution_config,
        visible_device_ids=(jax.devices("cpu")[0].id,),
        device_pool_limit_bytes={},
    )
    entry = open_declaration_allocations(
        hardware=hardware,
        declarations=(*declarations._input_bindings, _params()),
        regime_id_class=declarations.regime_id_class,
    )
    assert entry is not None
    model = _bare_model(declarations)
    try:
        prepared = declarations.prepare(
            model=model, dated=True, materializer=entry, hardware=hardware
        )
        metadata = model._prepare_dated_metadata(prepared=prepared, materializer=entry)
        assert metadata.builder.representative_user_regimes is (
            prepared.prepared_structure.representative_user_regimes
        )
        parent = metadata.builder.state_action_spaces["parent"]
        child = metadata.builder.state_action_spaces["child"]
        np.testing.assert_array_equal(parent.states["wealth"], np.array([1.0, 4.0]))
        np.testing.assert_array_equal(child.states["wealth"], np.array([1.0, 12.5]))
        assert model._simulate_entry_operations is entry.operations
        roots = jax.tree.leaves(entry.solve_input_roots())
        assert any(value is parent.states["wealth"] for value in roots)
        assert any(value is child.states["wealth"] for value in roots)
    finally:
        entry.close()


def test_unsupported_cold_metadata_refuses_before_builder_without_changing_ordinary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A real eager declaration keeps its ordinary builder/error boundary."""
    declarations = replace(_declarations(), enable_jit=False)
    model = _bare_model(declarations)
    prepared = declarations.prepare(model=model, dated=True)
    owner = _owner()

    def original_builder(**_: PytreeValue) -> Never:
        raise AssertionError("The original metadata builder was entered.")

    monkeypatch.setattr(model_processing, "prepare_regime_builder", original_builder)
    try:
        with pytest.raises(UnsupportedOperationError, match="unsharded compiled"):
            model._prepare_dated_metadata(prepared=prepared, materializer=owner)
        with pytest.raises(AssertionError, match="original metadata builder"):
            model._prepare_dated_metadata(prepared=prepared)
    finally:
        owner.close()


def test_cold_period_clock_refuses_before_its_lazy_original_factory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Validate the horizon first; do not invoke the unadmitted arange reader."""
    declarations = replace(_declarations(), ages=None, n_periods=3)
    owner = _owner()

    def unadmitted_coordinate_factory(_clock: ModelTime) -> Never:
        raise AssertionError("Original period coordinate producer was invoked.")

    monkeypatch.setattr(ModelTime, "values", property(unadmitted_coordinate_factory))
    horizonless = replace(declarations, n_periods=0)
    try:
        with pytest.raises(ModelInitializationError, match="positive integer"):
            horizonless.prepare(
                model=_bare_model(horizonless), dated=True, materializer=owner
            )
        with pytest.raises(UnsupportedOperationError, match="period clocks"):
            declarations.prepare(
                model=_bare_model(declarations), dated=True, materializer=owner
            )
    finally:
        owner.close()
