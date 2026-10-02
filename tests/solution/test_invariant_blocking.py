"""Opt-in type-local GridSearch: one invariant code at a time, same numbers.

`ExecutionConfig(invariant_block_widths={"pref_type": 1})` solves every regime
carrying `pref_type` one preference type at a time, reading each continuation
through the selected block of that type. Values, policies and the public result
schema equal the unblocked solve of the same model bit for bit, and an unsafe or
unsupported request is refused before anything is dispatched.
"""

import logging
from collections.abc import Mapping
from types import MappingProxyType
from typing import Any

import jax.numpy as jnp
import numpy as np
import pytest
from beartype.roar import BeartypeCallHintParamViolation

from _lcm.execution.core_program import core_program_graph
from _lcm.execution.execution_plan import CorePlanRecord
from _lcm.solution import backward_induction
from lcm import (
    AgeGrid,
    ByAge,
    Choose,
    DiscreteGrid,
    ExecutionConfig,
    LinSpacedGrid,
    Model,
    categorical,
    fixed_transition,
)
from lcm.exceptions import ExecutionPlanningError
from lcm.regime import Regime
from lcm.typing import (
    BoolND,
    ContinuousAction,
    ContinuousState,
    DiscreteState,
    FloatND,
    ScalarInt,
)
from tests.test_models import independent_types


def _blocked() -> ExecutionConfig:
    return ExecutionConfig(invariant_block_widths={"pref_type": 1})


_N_TYPES = len(independent_types.TYPE_PARAMS["weight"])


@categorical(ordered=False)
class _Sector:
    public: ScalarInt
    private: ScalarInt


@categorical(ordered=False)
class _RegimeId:
    working: ScalarInt
    terminal: ScalarInt


def _sector_utility(
    *,
    consumption: ContinuousAction,
    sector: DiscreteState,
    pref_type: DiscreteState,
    weight: FloatND,
    exponent: FloatND,
) -> FloatND:
    return (weight[pref_type] + 0.25 * sector) * (1.0 + consumption) ** exponent[
        pref_type
    ]


def _typed_bequest(
    *,
    wealth: ContinuousState,
    pref_type: DiscreteState,
    bequest_weight: FloatND,
    exponent: FloatND,
) -> FloatND:
    return bequest_weight[pref_type] * (1.0 + wealth) ** exponent[pref_type]


def _type_free_bequest(*, wealth: ContinuousState) -> FloatND:
    return 0.7 * (1.0 + wealth) ** 0.5


def _next_wealth(
    *, wealth: ContinuousState, consumption: ContinuousAction
) -> ContinuousState:
    return wealth - consumption


def _affordable(*, wealth: ContinuousState, consumption: ContinuousAction) -> BoolND:
    return consumption <= wealth


def _next_sector(*, sector: DiscreteState) -> DiscreteState:
    return 1 - sector


def _sector_model(
    *, typed_terminal: bool, execution_config: ExecutionConfig | None = None
) -> Model:
    """A model whose `pref_type` axis follows a moving discrete `sector` axis.

    With `typed_terminal`, the terminal regime keeps `pref_type`, so every
    continuation read selects one type's block. Without it, the terminal value
    is type-free and every reader shares it whole.
    """
    ages = AgeGrid(start=0, stop=3, step="Y")
    wealth = LinSpacedGrid(start=0, stop=10, n_points=11)
    pref_type = DiscreteGrid(category_class=independent_types.PrefType)
    last_age = ages.exact_values[-1]
    working = Regime(
        regime_transitions=ByAge.until(
            stop_age_exclusive=last_age,
            law=Choose(func=lambda: _RegimeId.working, targets=("working",)),
            then=Choose(func=lambda: _RegimeId.terminal, targets=("terminal",)),
        ),
        # `sector` is declared first, so it leads the discrete axes and
        # `pref_type` is the second value axis.
        states={
            "sector": DiscreteGrid(category_class=_Sector),
            "pref_type": pref_type,
            "wealth": wealth,
        },
        state_transitions={
            "sector": _next_sector,
            "pref_type": fixed_transition("pref_type"),
            "wealth": _next_wealth,
        },
        actions={"consumption": wealth},
        functions={"utility": _sector_utility},
        constraints={"affordable": _affordable},
    )
    terminal = Regime(
        regime_transitions=None,
        states=(
            {"pref_type": pref_type, "wealth": wealth}
            if typed_terminal
            else {"wealth": wealth}
        ),
        functions={"utility": _typed_bequest if typed_terminal else _type_free_bequest},
    )
    return Model(
        regimes={"working": working, "terminal": terminal},
        ages=ages,
        regime_id_class=_RegimeId,
        initial_regimes={ages.exact_values[0]: "working"},
        execution_config=execution_config or ExecutionConfig(),
    )


def _sector_params(*, typed_terminal: bool, scale: float = 1.0) -> dict:
    weight = jnp.asarray(independent_types.TYPE_PARAMS["weight"]) * scale
    exponent = jnp.asarray(independent_types.TYPE_PARAMS["exponent"])
    terminal = (
        {
            "utility": {
                "bequest_weight": jnp.asarray(
                    independent_types.TYPE_PARAMS["bequest_weight"]
                ),
                "exponent": exponent,
            }
        }
        if typed_terminal
        else {}
    )
    return {
        "discount_factor": 0.9,
        "working": {"utility": {"weight": weight, "exponent": exponent}},
        "terminal": terminal,
    }


def _independent_types_model(*, execution_config: ExecutionConfig) -> Model:
    model = independent_types.get_model()
    return Model(
        regimes=model.user_regimes,
        ages=model.ages,
        regime_id_class=independent_types.RegimeId,
        initial_regimes={model.ages.exact_values[0]: "working"},
        execution_config=execution_config,
    )


def _workload(*, name: str, execution_config: ExecutionConfig) -> tuple[Model, dict]:
    if name == "independent_types":
        return (
            _independent_types_model(execution_config=execution_config),
            independent_types.get_params(),
        )
    typed_terminal = name == "sector_typed_terminal"
    return (
        _sector_model(typed_terminal=typed_terminal, execution_config=execution_config),
        _sector_params(typed_terminal=typed_terminal),
    )


_WORKLOADS = ("independent_types", "sector_typed_terminal", "sector_type_free_terminal")


def _values(*, model: Model, params: dict) -> Mapping:
    return model.solve(params=params, log_level="off").values


def _assert_values_bitwise_equal(*, got: Mapping, expected: Mapping) -> None:
    assert {period: tuple(by_regime) for period, by_regime in got.items()} == {
        period: tuple(by_regime) for period, by_regime in expected.items()
    }
    mismatched = [
        (period, regime)
        for period, by_regime in expected.items()
        for regime, value in by_regime.items()
        if not (
            np.asarray(got[period][regime]).dtype == np.asarray(value).dtype
            and np.array_equal(
                np.asarray(got[period][regime]), np.asarray(value), equal_nan=True
            )
        )
    ]
    assert mismatched == []


def test_execution_config_invariant_block_widths_default_blocks_nothing() -> None:
    """Without a request no state is blocked."""
    assert dict(ExecutionConfig().invariant_block_widths) == {}


def test_execution_config_invariant_block_widths_are_frozen() -> None:
    """The request is held in an immutable mapping."""
    config = ExecutionConfig(invariant_block_widths={"pref_type": 1})

    assert isinstance(config.invariant_block_widths, MappingProxyType)


@pytest.mark.parametrize(
    ("widths", "error"),
    [
        ({"pref_type": 0}, ValueError),
        ({"pref_type": -1}, ValueError),
        ({"pref_type": True}, TypeError),
        ({"pref_type": 1.0}, BeartypeCallHintParamViolation),
        ({"": 1}, TypeError),
    ],
)
def test_execution_config_refuses_an_unusable_block_width(
    *, widths: dict, error: type[Exception]
) -> None:
    """A width is a positive exact integer keyed by a non-empty state name."""
    with pytest.raises(error):
        ExecutionConfig(invariant_block_widths=widths)


@pytest.mark.parametrize("workload", _WORKLOADS)
def test_blocked_solve_equals_the_unblocked_solve_bitwise(workload: str) -> None:
    """Every published value array is the unblocked one, bit for bit."""
    model, params = _workload(name=workload, execution_config=_blocked())
    reference, _ = _workload(name=workload, execution_config=ExecutionConfig())

    _assert_values_bitwise_equal(
        got=_values(model=model, params=params),
        expected=_values(model=reference, params=params),
    )


def test_blocked_solve_matches_the_enumerated_oracle() -> None:
    """Each type's blocked values equal the independent enumeration."""
    model = _independent_types_model(execution_config=_blocked())
    oracle, _ = independent_types.solve_by_enumeration()
    values = _values(model=model, params=independent_types.get_params())

    np.testing.assert_allclose(
        np.stack([np.asarray(values[period]["working"]) for period in (0, 1, 2)]),
        np.asarray([oracle[period] for period in (0, 1, 2)]),
        rtol=1e-5,
    )


def test_blocked_solve_with_changed_params_equals_the_unblocked_one() -> None:
    """A second blocked solve with new parameters reuses nothing stale."""
    model = _sector_model(typed_terminal=True, execution_config=_blocked())
    reference = _sector_model(typed_terminal=True)
    base = _values(model=model, params=_sector_params(typed_terminal=True))
    changed_params = _sector_params(typed_terminal=True, scale=1.3)

    changed = _values(model=model, params=changed_params)

    _assert_values_bitwise_equal(
        got=changed, expected=_values(model=reference, params=changed_params)
    )
    assert not np.array_equal(
        np.asarray(changed[0]["working"]), np.asarray(base[0]["working"])
    )


def test_blocked_simulation_equals_the_unblocked_panel() -> None:
    """Simulated choices and states from a blocked solve are the unblocked ones."""
    n_wealth = independent_types.N_WEALTH_POINTS
    types = np.repeat(np.arange(_N_TYPES), n_wealth)
    initial_conditions = {
        "regime_id": jnp.full(types.size, independent_types.RegimeId.working),
        "age": jnp.zeros(types.size),
        "wealth": jnp.asarray(np.tile(np.arange(n_wealth, dtype=float), _N_TYPES)),
        "pref_type": jnp.asarray(types, dtype=jnp.int32),
    }
    frames = [
        _independent_types_model(execution_config=config)
        .simulate(
            params=independent_types.get_params(),
            initial_conditions=initial_conditions,
            seed=0,
            log_level="off",
        )
        .to_dataframe()
        for config in (_blocked(), ExecutionConfig())
    ]

    assert frames[0].equals(frames[1])


def test_block_programs_keep_the_original_type_codes() -> None:
    """Each block program of a carrier is bound to one original code, in order."""
    model = _independent_types_model(execution_config=_blocked())
    graph = core_program_graph(
        kernel=model._regimes["working"].solution.period_kernels[0]
    )

    bindings = [program.invariant_binding for program in graph.values()]

    assert [
        (binding.state_name, binding.code)
        for binding in bindings
        if binding is not None
    ] == [("pref_type", code) for code in range(_N_TYPES)]
    assert None not in bindings


def test_an_unblocked_carrier_keeps_one_main_program() -> None:
    """Without a request the carrier keeps its single unbound program."""
    model = _independent_types_model(execution_config=ExecutionConfig())
    graph = core_program_graph(
        kernel=model._regimes["working"].solution.period_kernels[0]
    )

    assert [(name, program.invariant_binding) for name, program in graph.items()] == [
        ("main", None)
    ]


def _count_compiles(
    *, monkeypatch: pytest.MonkeyPatch, model: Model, params: dict
) -> int:
    calls: list[object] = []
    compile_and_log = backward_induction._compile_and_log

    def counted(**kwargs: Any) -> object:
        calls.append(kwargs["lowering_key"])
        return compile_and_log(**kwargs)

    monkeypatch.setattr(backward_induction, "_compile_and_log", counted)
    model.solve(params=params, log_level="off")
    monkeypatch.undo()
    return len(calls)


@pytest.mark.parametrize("workload", _WORKLOADS)
def test_blocked_solve_compiles_no_program_per_type(
    *, monkeypatch: pytest.MonkeyPatch, workload: str
) -> None:
    """Blocking compiles exactly as many programs as the unblocked solve."""
    blocked, params = _workload(name=workload, execution_config=_blocked())
    reference, _ = _workload(name=workload, execution_config=ExecutionConfig())

    assert _count_compiles(
        monkeypatch=monkeypatch, model=blocked, params=params
    ) == _count_compiles(monkeypatch=monkeypatch, model=reference, params=params)


class _PlanRecords(logging.Handler):
    """Collect the plan records a debug solve emits."""

    def __init__(self) -> None:
        super().__init__(level=logging.DEBUG)
        self.records: list[CorePlanRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        plan_record = getattr(record, "core_plan_record", None)
        if plan_record is not None:
            self.records.append(plan_record)


def _plan_records(*, model: Model, params: dict) -> list[CorePlanRecord]:
    handler = _PlanRecords()
    logger = logging.getLogger("lcm")
    logger.addHandler(handler)
    try:
        model.solve(params=params, log_level="debug")
    finally:
        logger.removeHandler(handler)
    return handler.records


def test_plan_records_name_one_selected_type_per_block() -> None:
    """Every carrier block records the one-code interval it evaluates."""
    records = _plan_records(
        model=_independent_types_model(execution_config=_blocked()),
        params=independent_types.get_params(),
    )

    assert sorted(
        (record.period, dict(record.selected_block or {}).get("pref_type"))
        for record in records
        if record.regime == "working"
    ) == sorted(
        (period, (code, code + 1)) for period in (0, 1, 2) for code in range(_N_TYPES)
    )


def test_a_block_moves_one_type_of_each_continuation() -> None:
    """A block's transfers hold one type's share of the typed continuation."""
    model = _sector_model(typed_terminal=True, execution_config=_blocked())
    params = _sector_params(typed_terminal=True)
    records = _plan_records(model=model, params=params)
    terminal_value = np.asarray(_values(model=model, params=params)[3]["terminal"])

    assert {
        record.transfer_workspace_bytes
        for record in records
        if record.regime == "working" and record.period == 2
    } == {terminal_value.nbytes // _N_TYPES}


@pytest.mark.parametrize(
    ("widths", "sharded", "match"),
    [
        ({"absent": 1}, (), "absent"),
        ({"wealth": 1}, (), "wealth"),
        ({"pref_type": 2}, (), "width"),
        ({"pref_type": 1}, ("pref_type",), "sharded"),
    ],
)
def test_an_unsafe_or_unsupported_request_is_refused_at_construction(
    *, widths: dict, sharded: tuple[str, ...], match: str
) -> None:
    """An invalid request raises before any program is lowered or dispatched."""
    with pytest.raises(ExecutionPlanningError, match=match):
        _independent_types_model(
            execution_config=ExecutionConfig(
                invariant_block_widths=widths, sharded_states=sharded
            )
        )


def test_a_reset_of_the_blocked_state_is_refused_at_construction() -> None:
    """A state whose law is not the identity everywhere cannot be blocked."""
    with pytest.raises(ExecutionPlanningError, match="sector"):
        _sector_model(
            typed_terminal=True,
            execution_config=ExecutionConfig(invariant_block_widths={"sector": 1}),
        )
