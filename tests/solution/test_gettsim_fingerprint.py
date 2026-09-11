"""Exercise durable identity through GETTSIM's public JAX graph builder."""

import dataclasses
import importlib
import os
import subprocess
import sys
from collections.abc import Callable
from functools import partial
from pathlib import Path
from types import FunctionType, ModuleType, SimpleNamespace
from typing import Any

import cloudpickle
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.solution.fingerprint import _semantic_fingerprint
from lcm import (
    AgeGrid,
    LinSpacedGrid,
    Model,
    categorical,
    fixed_transition,
    load_solution,
)
from lcm.exceptions import InvalidSimulationInputError
from lcm.regime import Regime
from lcm.typing import ScalarFloat, ScalarInt

gettsim = pytest.importorskip("gettsim")
tt = pytest.importorskip("gettsim.tt")
param_objects = pytest.importorskip("ttsim.tt.param_objects")
rounding = pytest.importorskip("ttsim.tt.rounding")


def test_generated_gettsim_graph_has_repeatable_semantic_identity() -> None:
    """Independent grouped-income graphs agree in output and durable identity."""
    first = _build_income_function()
    second = _build_income_function()
    np.testing.assert_array_equal(
        first({k: v for k, v in _input_data().items() if k != "p_id"})["total"],
        [10.0, 10.0],
    )
    assert _semantic_fingerprint(first) == _semantic_fingerprint(second)


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


def test_generated_callable_accepts_exact_jax_backend_parameter() -> None:
    """A generated backend parameter has a versioned numerical identity."""

    def function(*, value: float, xnp: ModuleType) -> object:
        return xnp.sqrt(value)

    wrapped = partial(function, xnp=jnp)
    np.testing.assert_array_equal(wrapped(value=9.0), 3.0)
    assert _semantic_fingerprint(wrapped) == _semantic_fingerprint(
        partial(function, xnp=jnp)
    )


def test_policy_function_can_be_captured_directly() -> None:
    """A generated wrapper can retain its decorated policy callable."""

    @tt.policy_function(vectorization_strategy="not_required")
    def policy(value: float) -> float:
        return value * 2

    def wrapper(value: float) -> float:
        return policy(value)

    np.testing.assert_array_equal(wrapper(3.0), 6.0)
    assert _semantic_fingerprint(wrapper) == _semantic_fingerprint(wrapper)


def test_rounded_policy_retains_policy_and_rounding_semantics() -> None:
    """Rounding a policy callable binds its closure and rounding rule."""

    def build(*, scale: float, base: float) -> Callable:
        @tt.policy_function(vectorization_strategy="not_required")
        def policy(value: float) -> float:
            return value * scale

        return rounding.RoundingSpec(base=base, direction="up").apply_rounding(
            policy, jnp
        )

    baseline = build(scale=2.0, base=1.0)
    np.testing.assert_array_equal(baseline(1.1), 3.0)
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


def test_rounding_wrapper_survives_serialization() -> None:
    """Serialized generated wrappers keep their durable semantic identity."""

    def body(value: float) -> float:
        return value * 2

    # Model a generated policy module with ordinary, serializable globals.
    function = FunctionType(body.__code__, {"__name__": "policy_fixture"})
    policy = tt.policy_function(vectorization_strategy="not_required")(function)

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

    def key(person_id: int) -> int:
        return person_id

    declarations = [
        tt.policy_function(
            foreign_key_type=kind, vectorization_strategy="not_required"
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

    @tt.policy_function(vectorization_strategy="not_required")
    def policy(_value: float) -> object:
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

    def transition() -> ScalarInt:
        return _RegimeId.retired

    return Model(
        ages=AgeGrid(start=0, stop=1, step="Y"),
        regime_id_class=_RegimeId,
        regimes={
            "working": Regime(
                transition=transition,
                active=lambda age: age == 0,
                states={"wealth": LinSpacedGrid(start=1.0, stop=2.0, n_points=2)},
                state_transitions={"wealth": fixed_transition("wealth")},
                functions={"utility": utility},
            ),
            "retired": Regime(
                transition=None,
                active=lambda age: age == 1,
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
        vectorization_strategy="not_required", start_date=start_date
    )(income)

    @tt.agg_by_group_function(agg_type=tt.AggType.SUM)
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
        input_data=gettsim.InputData.tree(_input_data()),
        tt_targets=gettsim.TTTargets.tree(dict.fromkeys(targets, True)),
        include_warn_nodes=False,
        include_fail_nodes=False,
        backend="jax",
    )
