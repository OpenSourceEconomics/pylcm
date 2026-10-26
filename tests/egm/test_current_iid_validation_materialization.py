"""Original eager validator arithmetic and ownership under actual admission."""

from collections.abc import Callable
from functools import partial
from types import MappingProxyType

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax._src.api import Inline

from _lcm.dtypes import canonical_float_dtype
from _lcm.egm.validation import (
    _admitted_validation_unique,
    validate_dcegm_regime,
)
from _lcm.execution.workspace_planning import WorkspaceBudgetError
from _lcm.solution.generation import canonical_generation_roots
from _lcm.typing import PytreeValue
from lcm.exceptions import ExecutionPlanningError, ModelInitializationError
from lcm.typing import FloatND
from tests.egm.test_current_iid_generation_materialization import _owner
from tests.egm.test_current_iid_parent_demands import (
    _current_resources,
    _inverse,
    _resources,
    _savings,
    _utility,
)
from tests.solution.test_post_decision_lower_bound import _model as _bounded_model

pytestmark = pytest.mark.coverage(backends=("cpu",), precisions="both")


@pytest.mark.parametrize(
    ("function", "values"),
    [
        (_utility, {"consumption": 0.75}),
        (_resources, {"wealth": 1.25}),
        (_current_resources, {"wealth": 1.25, "eps": -0.125}),
        (_savings, {"resources": 1.25, "consumption": 0.75}),
        (_inverse, {"marginal_continuation": 1.25}),
    ],
)
def test_original_economic_callback_bytes_and_weak_type(
    *, function: Callable[..., PytreeValue], values: dict[str, float]
) -> None:
    arguments = {name: jnp.asarray(value) for name, value in values.items()}
    expected = function(**arguments)
    owner = _owner(roots=(*arguments.values(), expected))
    try:
        actual = owner.evaluate_validation(function=function, arguments=arguments)
        assert isinstance(expected, jax.Array)
        assert isinstance(actual, jax.Array)
        assert actual.dtype == expected.dtype
        assert actual.weak_type == expected.weak_type
        assert np.asarray(actual).tobytes() == np.asarray(expected).tobytes()
    finally:
        owner.close()


@jax.jit
def _opaque_product(*, left: FloatND, right: FloatND) -> FloatND:
    return left * right


def _separate_dispatches(*, left: FloatND, right: FloatND, offset: FloatND) -> FloatND:
    return _opaque_product(left=left, right=right) + offset


def test_original_nested_jit_remains_a_separate_rounding_boundary() -> None:
    dtype = np.dtype(canonical_float_dtype())
    eps = np.finfo(dtype).eps
    arguments = {
        "left": jnp.asarray(dtype.type(1.0 + eps)),
        "right": jnp.asarray(dtype.type(1.0 - eps)),
        "offset": jnp.asarray(dtype.type(-1.0)),
    }
    expected = _separate_dispatches(**arguments)
    owner = _owner(roots=(*arguments.values(), expected))
    try:
        actual = owner.evaluate_validation(
            function=_separate_dispatches, arguments=arguments
        )
        assert np.asarray(expected).tobytes() == np.asarray(dtype.type(0.0)).tobytes()
        assert np.asarray(actual).tobytes() == np.asarray(expected).tobytes()
    finally:
        owner.close()


def _product_for_origin_control(*, left: FloatND, right: FloatND) -> FloatND:
    return left * right


def _original_nextafter(*, left: FloatND, right: FloatND) -> FloatND:
    return jnp.nextafter(left, right)


@pytest.mark.parametrize("weak", [False, True])
@pytest.mark.parametrize("direction", [-np.inf, np.inf])
def test_original_nextafter_keeps_bytes_dtype_and_weak_type(
    *, weak: bool, direction: float
) -> None:
    dtype = np.dtype(canonical_float_dtype())
    arguments = {
        "left": jnp.asarray(1.0, dtype=None if weak else dtype),
        "right": jnp.asarray(direction, dtype=None if weak else dtype),
    }
    expected = _original_nextafter(**arguments)
    owner = _owner(roots=(*arguments.values(), expected))
    try:
        actual = owner.evaluate_validation(
            function=_original_nextafter, arguments=arguments
        )
        assert actual.dtype == expected.dtype
        assert actual.weak_type == expected.weak_type
        assert np.asarray(actual).tobytes() == np.asarray(expected).tobytes()
        assert (
            np.asarray(actual).tobytes()
            == np.asarray(
                np.nextafter(dtype.type(1.0), dtype.type(direction))
            ).tobytes()
        )
    finally:
        owner.close()


def test_rebound_nextafter_refuses_before_callback_tracing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from _lcm.egm import validation_stages  # noqa: PLC0415

    replacement = jax.jit(_product_for_origin_control, inline=Inline.JAX_EARLY)
    monkeypatch.setattr(jnp, "nextafter", replacement)

    def must_not_trace(**_: PytreeValue) -> PytreeValue:
        raise AssertionError("Callback tracing preceded rebound nextafter refusal.")

    monkeypatch.setattr(validation_stages, "prepare_validation_recipe", must_not_trace)
    arguments = {"left": jnp.asarray(1.0), "right": jnp.asarray(2.0)}
    owner = _owner(roots=tuple(arguments.values()))
    try:
        with pytest.raises(ExecutionPlanningError, match="JAX_EARLY"):
            owner.evaluate_validation(function=_original_nextafter, arguments=arguments)
        assert not owner.operations.cache
        assert not owner._pending
    finally:
        owner.close()


@pytest.mark.parametrize("mutation", ["none", "info", "fun", "malformed"])
def test_early_jit_origin_refuses_before_callback_tracing(
    *, mutation: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    import sys  # noqa: PLC0415

    from _lcm.egm import validation_stages  # noqa: PLC0415

    early = jax.jit(_product_for_origin_control, inline=Inline.JAX_EARLY)
    auto = jax.jit(_product_for_origin_control, inline=Inline.AUTO)
    if mutation == "info":
        monkeypatch.setattr(early, "_jit_info", vars(auto)["_jit_info"])
    elif mutation == "fun":
        monkeypatch.setattr(early, "_fun", _separate_dispatches)
    elif mutation == "malformed":
        monkeypatch.setattr(early, "_jit_info", None)
    monkeypatch.setattr(sys.modules[__name__], "_opaque_product", early)

    def must_not_trace(**_: PytreeValue) -> PytreeValue:
        raise AssertionError("Callback tracing preceded the original JIT refusal.")

    monkeypatch.setattr(validation_stages, "prepare_validation_recipe", must_not_trace)
    arguments = {
        "left": jnp.asarray(0.5),
        "right": jnp.asarray(0.25),
        "offset": jnp.asarray(-1.0),
    }
    owner = _owner(roots=tuple(arguments.values()))
    try:
        with pytest.raises(ExecutionPlanningError, match=r"JAX_EARLY|Original JIT"):
            owner.evaluate_validation(
                function=_separate_dispatches, arguments=arguments
            )
        assert not owner.operations.cache
        assert not owner._pending
    finally:
        owner.close()


def _multi_operation_canonical_log(*, value: FloatND) -> FloatND:
    return jnp.log10(value)


def test_canonical_multi_operation_early_jit_refuses_before_callback_tracing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from _lcm.egm import validation_stages  # noqa: PLC0415

    def must_not_trace(**_: PytreeValue) -> PytreeValue:
        raise AssertionError("Callback tracing preceded canonical EARLY refusal.")

    monkeypatch.setattr(validation_stages, "prepare_validation_recipe", must_not_trace)
    value = jnp.asarray(0.5)
    owner = _owner(roots=(value,))
    try:
        with pytest.raises(ExecutionPlanningError, match="JAX_EARLY"):
            owner.evaluate_validation(
                function=_multi_operation_canonical_log, arguments={"value": value}
            )
        assert not owner.operations.cache
        assert not owner._pending
    finally:
        owner.close()


def _unsupported_after_add(*, value: FloatND) -> FloatND:
    return jax.lax.sin(value + 1.0)


def test_unsupported_later_primitive_refuses_before_any_dispatch() -> None:
    value = jnp.asarray(0.25)
    owner = _owner(roots=(value,))
    try:
        with pytest.raises(ExecutionPlanningError, match="unsupported eager primitive"):
            owner.evaluate_validation(
                function=_unsupported_after_add, arguments={"value": value}
            )
        assert not owner.operations.cache
        assert not owner._pending
    finally:
        owner.close()


def _eager_trace_escape(*, value: FloatND) -> FloatND:
    with jax.ensure_compile_time_eval():
        captured = jnp.ones((4096,))
    return value + captured


def test_eager_trace_escape_refuses_before_callback_tracing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from _lcm.egm import validation_stages  # noqa: PLC0415

    def must_not_trace(**_: PytreeValue) -> PytreeValue:
        raise AssertionError("Callback tracing preceded the eager-escape refusal.")

    monkeypatch.setattr(validation_stages, "prepare_validation_recipe", must_not_trace)
    value = jnp.asarray(0.25)
    owner = _owner(roots=(value,))
    try:
        with pytest.raises(ExecutionPlanningError, match="compile-time eager"):
            owner.evaluate_validation(
                function=_eager_trace_escape, arguments={"value": value}
            )
        assert not owner.operations.cache
        assert not owner._pending
    finally:
        owner.close()


_CAPTURED_BIAS: jax.Array | None = None


def _captured_array(*, value: FloatND) -> FloatND:
    if _CAPTURED_BIAS is None:
        raise ValueError("The captured callback bias is not bound.")
    return value + _CAPTURED_BIAS


def test_referenced_callback_array_survives_unrelated_publish(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sys  # noqa: PLC0415

    bias = jnp.ones((4096,))
    value = jnp.asarray(0.25)
    monkeypatch.setattr(sys.modules[__name__], "_CAPTURED_BIAS", bias)
    owner = _owner(roots=(value,))
    try:
        actual = owner.evaluate_validation(
            function=_captured_array, arguments={"value": value}
        )
        np.testing.assert_array_equal(actual, np.full(4096, 1.25))
        del actual
        owner.publish(stage="params", tree=MappingProxyType({}))
        assert any(root is bias for root in jax.tree.leaves(owner.solve_input_roots()))
        owner.budget_bytes = bias.nbytes - 1
        with pytest.raises(ExecutionPlanningError, match="before allocation"):
            owner(value=np.ones(1), dtype=np.dtype(bias.dtype), name="next_leaf")
    finally:
        owner.close()


def test_actual_integer_unique_cardinality_keeps_original_JAX_values() -> None:
    indices = jnp.asarray([0, 0, 2, 2, 3], dtype=jnp.int32)
    expected = jnp.unique(indices)
    owner = _owner(roots=(indices, expected))
    try:
        actual = _admitted_validation_unique(indices=indices, materializer=owner)
        assert actual.shape == (3,)
        assert actual.dtype == indices.dtype
        assert np.asarray(actual).tobytes() == np.asarray(expected).tobytes()
    finally:
        owner.close()


def _broad_then_add(*, value: FloatND) -> FloatND:
    broad = jnp.broadcast_to(value, (4096,))
    return broad + 1.0


def test_actual_first_stage_fits_and_later_allocation_refuses() -> None:
    value = jnp.asarray(0.25)
    probe = _owner(roots=(value,))
    result = probe.evaluate_validation(
        function=_broad_then_add, arguments={"value": value}
    )
    records = tuple(probe.operations.cache.values())
    first = records[0].memory.reservation_bytes
    assert first < records[-1].memory.reservation_bytes
    operations = probe.operations
    del result
    probe.close()
    owner = _owner(roots=(value,))
    owner.operations = operations
    owner.budget_bytes = first
    try:
        with pytest.raises((ExecutionPlanningError, WorkspaceBudgetError)):
            owner.evaluate_validation(
                function=_broad_then_add, arguments={"value": value}
            )
        assert not owner._pending
        np.testing.assert_array_equal(value, np.asarray(0.25))
    finally:
        owner.close()


def _wrong_inverse(*, marginal_continuation: FloatND) -> FloatND:
    return 2.0 / marginal_continuation


@pytest.mark.parametrize("wrong_inverse", [False, True])
def test_original_real_DCEGM_validator_keeps_outcome_and_diagnostic(
    *, wrong_inverse: bool
) -> None:
    model = _bounded_model(savings_grid_start=-2.0, declared_lower_bound=-2.0)
    prepared = model._declaration_preparation
    assert prepared is not None
    regime = model.user_regimes["saving"]
    if wrong_inverse:
        regime = regime.replace(
            functions={
                **regime.functions,
                "inverse_marginal_utility": _wrong_inverse,
            }
        )
    regimes = {**model.user_regimes, "saving": regime}
    laws = prepared.prepared_structure.laws
    validate = partial(
        validate_dcegm_regime,
        regime_name="saving",
        user_regime=regime,
        user_regimes=regimes,
        laws=laws,
        solution_reachability=prepared.prepared_structure.reachability.solution,
    )
    owner = _owner(
        roots=(
            *model._generation_roots(),
            *canonical_generation_roots(declarations=(regime,)),
        )
    )
    try:
        if wrong_inverse:
            with pytest.raises(ModelInitializationError) as expected:
                validate()
            with pytest.raises(ModelInitializationError) as actual:
                validate(materializer=owner)
            assert str(actual.value) == str(expected.value)
            assert "inconsistent" in str(actual.value)
        else:
            assert validate() is None
            assert validate(materializer=owner) is None
    finally:
        owner.close()
