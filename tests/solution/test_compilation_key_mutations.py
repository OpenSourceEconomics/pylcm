"""What the lowering key separates, and what it shares, under one-fact mutations.

Each case mutates one fact of a tiny grid-search model and reads the lowering
keys the solve publishes. A fact that changes the traced program — a shape, a
dtype, an axis order, a device placement, a donation set, a static width, a
function body, the x64 flag — must give the mutated solve a disjoint key set. A
parameter value is a traced scalar argument, so it leaves every component of
the key untouched except the model fingerprint, which digests the canonical
solution parameters. The keys are built from durable data only: two processes
publish the same keys, and no key keeps its model alive.

`tests/solution/test_compilation_identity.py` holds the unit-level cases for
compiler options, donation, placement and solver group keys.
"""

import gc
import json
import os
import subprocess
import sys
import weakref
from collections.abc import Hashable, Mapping
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

from _lcm.solution import backward_induction
from lcm import AgeGrid, DiscreteGrid, ExecutionConfig, LinSpacedGrid, Model
from lcm.solvers import GridSearch
from lcm.typing import FloatND, UserParams
from tests.solution.test_compilation_identity import _capture_lowering_keys, _model
from tests.test_models import nbegm_ride_along_toy
from tests.test_models.deterministic.regression import (
    START_AGE,
    LaborSupply,
    RegimeId,
    dead,
    get_params,
    working_life,
)

_N_PERIODS = 3
_REPO_ROOT = Path(__file__).resolve().parents[2]
# Position inside a lowering key of the donated-argument tuple.
_DONATED_POSITION = 5
type _Key = tuple[tuple[str, ...], *tuple[Hashable, ...]]
type _Candidate = tuple[tuple[str, int, str], Hashable]


def _reordered_actions_model() -> Model:
    """The identity toy with its two actions declared in the opposite order."""
    final_age_alive = START_AGE + _N_PERIODS - 2
    return Model(
        regimes={
            "working_life": working_life.replace(
                active=lambda age: age <= final_age_alive,
                states={"wealth": LinSpacedGrid(start=1, stop=3, n_points=3)},
                actions={
                    "consumption": LinSpacedGrid(start=1, stop=3, n_points=3),
                    "labor_supply": DiscreteGrid(category_class=LaborSupply),
                },
                solver=GridSearch(),
            ),
            "dead": dead,
        },
        ages=AgeGrid(start=START_AGE, stop=final_age_alive + 1, step="Y"),
        regime_id_class=RegimeId,
    )


def _steeper_wage(age: float) -> float | FloatND:
    """A wage profile twice as steep in age as the identity toy's."""
    return 1 + 0.2 * age


def _rewaged_model() -> Model:
    """The identity toy with a different wage function body."""
    final_age_alive = START_AGE + _N_PERIODS - 2
    functions = {**working_life.functions, "wage": _steeper_wage}
    return Model(
        regimes={
            "working_life": working_life.replace(
                active=lambda age: age <= final_age_alive,
                states={"wealth": LinSpacedGrid(start=1, stop=3, n_points=3)},
                actions={
                    "labor_supply": DiscreteGrid(category_class=LaborSupply),
                    "consumption": LinSpacedGrid(start=1, stop=3, n_points=3),
                },
                functions=functions,
                solver=GridSearch(),
            ),
            "dead": dead,
        },
        ages=AgeGrid(start=START_AGE, stop=final_age_alive + 1, step="Y"),
        regime_id_class=RegimeId,
    )


def _fixed_discount_model(*, discount_factor: float) -> Model:
    """The identity toy with its discount factor fixed at construction."""
    return Model(
        regimes={
            "working_life": working_life.replace(
                active=lambda age: age <= START_AGE + _N_PERIODS - 2,
                states={"wealth": LinSpacedGrid(start=1, stop=3, n_points=3)},
                actions={
                    "labor_supply": DiscreteGrid(category_class=LaborSupply),
                    "consumption": LinSpacedGrid(start=1, stop=3, n_points=3),
                },
                solver=GridSearch(),
            ),
            "dead": dead,
        },
        ages=AgeGrid(start=START_AGE, stop=START_AGE + _N_PERIODS - 1, step="Y"),
        regime_id_class=RegimeId,
        fixed_params={"discount_factor": discount_factor},
    )


def _nbegm_model(*, donate_buffers: bool) -> Model:
    """The ride-along tax toy, whose NBEGM dispatch donates its marginal leaf."""
    return nbegm_ride_along_toy.build_model(
        variant="nbegm",
        n_periods=_N_PERIODS,
        n_liquid=12,
        n_savings=12,
        n_consumption=12,
        execution_config=ExecutionConfig(donate_buffers=donate_buffers),
    )


def _keys_of(
    *, model: Model, params: UserParams, regime: str, monkeypatch: pytest.MonkeyPatch
) -> set[Hashable]:
    """The keys the solve publishes for the cores of one regime."""
    captured = _capture_lowering_keys(monkeypatch=monkeypatch)
    model.solve(params=params, log_level="off")
    return {
        key
        for candidate, key in captured[0].items()
        if cast("_Candidate", candidate)[0][0] == regime
    }


_TOY_PARAMS = get_params(n_periods=_N_PERIODS)
_TOY_PARAMS_WITHOUT_DISCOUNT = {
    name: value for name, value in _TOY_PARAMS.items() if name != "discount_factor"
}
# (baseline, mutant, params, regime whose cores the mutated fact reaches)
_IN_PROCESS_MUTATIONS = [
    pytest.param(
        _model,
        lambda: _model(n_wealth_points=4),
        _TOY_PARAMS,
        "working_life",
        id="shape",
    ),
    pytest.param(
        _model, _reordered_actions_model, _TOY_PARAMS, "working_life", id="axis_order"
    ),
    pytest.param(
        _model,
        lambda: _model(execution_config=ExecutionConfig(axis_widths={"cell": 1})),
        _TOY_PARAMS,
        "working_life",
        id="static_width",
    ),
    pytest.param(
        _model, _rewaged_model, _TOY_PARAMS, "working_life", id="function_body"
    ),
    pytest.param(
        lambda: _fixed_discount_model(discount_factor=0.95),
        lambda: _fixed_discount_model(discount_factor=0.9),
        _TOY_PARAMS_WITHOUT_DISCOUNT,
        "working_life",
        id="fixed_param",
    ),
]


@pytest.mark.parametrize(
    ("build_baseline", "build_mutant", "params", "regime"), _IN_PROCESS_MUTATIONS
)
def test_a_program_changing_fact_shares_no_lowering_key(
    *,
    build_baseline: Any,
    build_mutant: Any,
    params: UserParams,
    regime: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A mutant whose traced program differs is lowered under its own keys."""
    baseline = _keys_of(
        model=build_baseline(), params=params, regime=regime, monkeypatch=monkeypatch
    )
    mutant = _keys_of(
        model=build_mutant(), params=params, regime=regime, monkeypatch=monkeypatch
    )

    assert (baseline & mutant, len(baseline) > 0) == (set(), True)


def test_disabling_donation_relowers_every_donating_core(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A core that donates under the default keeps none of its keys once it may not.

    Cores that donate nothing either way are lowered under one key in both
    arms; only the donating cores' keys must vanish from the non-donating solve.
    """
    params = nbegm_ride_along_toy.build_params()
    donating = _keys_of(
        model=_nbegm_model(donate_buffers=True),
        params=params,
        regime="alive",
        monkeypatch=monkeypatch,
    )
    keeping = _keys_of(
        model=_nbegm_model(donate_buffers=False),
        params=params,
        regime="alive",
        monkeypatch=monkeypatch,
    )
    donors = {key for key in donating if cast("_Key", key)[_DONATED_POSITION]}

    assert (len(donors) > 0, donors & keeping) == (True, set())


def test_a_parameter_value_change_keeps_every_lowering_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two parameter vectors of one model publish identical lowering keys.

    Every parameter is a traced argument of the lowered program, so nothing a
    key names — program identity, argument description, specialization,
    layout, donation, placement, compiler options — depends on its value.
    """
    captured = _capture_lowering_keys(monkeypatch=monkeypatch)
    _model().solve(params=get_params(n_periods=_N_PERIODS), log_level="off")
    _model().solve(
        params=get_params(n_periods=_N_PERIODS, discount_factor=0.9),
        log_level="off",
    )
    baseline, mutant = captured

    assert baseline == mutant


def test_a_parameter_value_change_keeps_every_lowered_program(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every lowered program is byte-identical across two parameter vectors.

    This is what licenses a parameter-free program identity: no parameter
    value is baked into a lowered module as a constant.
    """
    captured = _capture_lowered_text(monkeypatch=monkeypatch)
    _model().solve(params=get_params(n_periods=_N_PERIODS), log_level="off")
    _model().solve(
        params=get_params(n_periods=_N_PERIODS, discount_factor=0.9),
        log_level="off",
    )
    baseline, mutant = captured

    assert (len(baseline) > 0, baseline) == (True, mutant)


def test_a_parameter_value_change_separates_the_model_fingerprint() -> None:
    """Two parameter vectors digest to two solution-artifact fingerprints."""
    baseline = _model().solve(params=get_params(n_periods=_N_PERIODS), log_level="off")
    mutant = _model().solve(
        params=get_params(n_periods=_N_PERIODS, discount_factor=0.9),
        log_level="off",
    )

    assert baseline.metadata.model_fingerprint != mutant.metadata.model_fingerprint


def _capture_lowered_text(*, monkeypatch: pytest.MonkeyPatch) -> list[dict[str, str]]:
    """Collect every solve's lowered modules, keyed by the engine's program label."""
    captured: list[dict[str, str]] = []
    original = backward_induction._assert_lowered_output_roles
    original_planning = backward_induction._resolve_output_layouts_and_lowering_keys

    def _new_solve(**kwargs: Any) -> tuple:
        captured.append({})
        return original_planning(**kwargs)

    def _spy(**kwargs: Any) -> None:
        captured[-1][kwargs["label"]] = kwargs["lowered"].as_text()
        original(**kwargs)

    monkeypatch.setattr(
        backward_induction, "_resolve_output_layouts_and_lowering_keys", _new_solve
    )
    monkeypatch.setattr(backward_induction, "_assert_lowered_output_roles", _spy)
    return captured


def test_a_parameter_value_change_moves_the_solved_value() -> None:
    """A lower discount factor changes the first-period value function."""
    baseline = _model().solve(params=get_params(n_periods=_N_PERIODS), log_level="off")
    mutant = _model().solve(
        params=get_params(n_periods=_N_PERIODS, discount_factor=0.9),
        log_level="off",
    )
    gap = np.max(
        np.abs(
            np.asarray(baseline.values[0]["working_life"])
            - np.asarray(mutant.values[0]["working_life"])
        )
    )
    assert gap > 1e-3


def test_admission_is_checked_per_candidate_not_per_lowering_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every admitted candidate is measured, also when its key was compiled already.

    The toy's five candidates lower to three executables (two working-life
    periods share one, two dead periods another), yet residency depends on the
    candidate's position in the plan, so each candidate is measured against the
    budget on its own.
    """
    measured: list[Hashable] = []
    lowered: list[Hashable] = []
    original_measure = backward_induction._measure_variant
    original_wave = backward_induction._lower_and_compile_wave

    def _count_measure(**kwargs: Any) -> int:
        measured.append(kwargs["variant_key"])
        return original_measure(**kwargs)

    def _count_wave(**kwargs: Any) -> None:
        lowered.extend(kwargs["new_lowerings"])
        original_wave(**kwargs)

    monkeypatch.setattr(backward_induction, "_measure_variant", _count_measure)
    monkeypatch.setattr(backward_induction, "_lower_and_compile_wave", _count_wave)
    model = _model(execution_config=ExecutionConfig(device_memory_bytes=2**32))
    model.solve(params=get_params(n_periods=_N_PERIODS), log_level="off")

    assert (len(measured), set(measured) == set(lowered), len(lowered)) == (5, True, 3)


def test_a_budget_change_reruns_admission_on_the_same_keys(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A second budgeted solve measures every candidate again, keys unchanged."""
    measured: list[Hashable] = []
    original_measure = backward_induction._measure_variant

    def _count_measure(**kwargs: Any) -> int:
        measured.append(kwargs["variant_key"])
        return original_measure(**kwargs)

    monkeypatch.setattr(backward_induction, "_measure_variant", _count_measure)
    params = get_params(n_periods=_N_PERIODS)
    _model(execution_config=ExecutionConfig(device_memory_bytes=2**32)).solve(
        params=params, log_level="off"
    )
    first = list(measured)
    measured.clear()
    _model(execution_config=ExecutionConfig(device_memory_bytes=2**31)).solve(
        params=params, log_level="off"
    )

    assert (first == measured, len(first)) == (True, 5)


def test_no_lowering_key_keeps_its_model_alive(monkeypatch: pytest.MonkeyPatch) -> None:
    """Dropping a model after its solve collects it while its keys are still held."""
    model = _model()
    keys = _keys_of(
        model=model, params=_TOY_PARAMS, regime="working_life", monkeypatch=monkeypatch
    )
    ghost = weakref.ref(model)
    del model
    gc.collect()

    assert (ghost(), len(keys)) == (None, 2)


_DUMP_KEYS = """
import json, sys
import pytest
from tests.solution.test_compilation_identity import _capture_lowering_keys, _model
from tests.test_models.deterministic.regression import get_params
from lcm import ExecutionConfig
raw = json.loads(sys.argv[1])
config = ExecutionConfig(
    **{k: tuple(v) if isinstance(v, list) else v for k, v in raw.items()}
)
monkeypatch = pytest.MonkeyPatch()
captured = _capture_lowering_keys(monkeypatch=monkeypatch)
_model(execution_config=config).solve(params=get_params(n_periods=3), log_level="off")
print(json.dumps({repr(c): repr(k) for c, k in captured[0].items()}, sort_keys=True))
"""


def _keys_in_fresh_process(
    *, env: Mapping[str, str] = {}, config: Mapping[str, object] = {}
) -> dict[str, str]:
    """Solve the identity toy in a new interpreter and return its keys by `repr`.

    The baseline pins the x64 flag off, so a module imported earlier in this
    interpreter that exports `JAX_ENABLE_X64` cannot reach the child.
    """
    result = subprocess.run(  # noqa: S603
        [sys.executable, "-c", _DUMP_KEYS, json.dumps(config)],
        capture_output=True,
        text=True,
        cwd=_REPO_ROOT,
        env={**os.environ, "JAX_PLATFORMS": "cpu", "JAX_ENABLE_X64": "0", **env},
        check=False,
        timeout=600,
    )
    assert result.returncode == 0, result.stderr[-4000:]
    return json.loads(result.stdout.splitlines()[-1])


def test_two_processes_publish_the_same_lowering_keys() -> None:
    """The key holds no process-local object: two interpreters agree on it."""
    first = _keys_in_fresh_process()
    second = _keys_in_fresh_process()

    assert first == second


def test_lowering_keys_spell_no_object_address_or_callable() -> None:
    """A key's spelling names no `0x…` address and no function object."""
    spelled = "".join(_keys_in_fresh_process().values())

    assert ("0x" in spelled, "<function" in spelled, "<bound" in spelled) == (
        False,
        False,
        False,
    )


_FRESH_PROCESS_MUTATIONS = [
    pytest.param({"JAX_ENABLE_X64": "1"}, {}, id="jax_enable_x64"),
    pytest.param(
        {"XLA_FLAGS": "--xla_force_host_platform_device_count=2"},
        {"devices": (0, 1)},
        id="device_count",
    ),
]


@pytest.mark.parametrize(("env", "config"), _FRESH_PROCESS_MUTATIONS)
def test_a_process_level_fact_shares_no_lowering_key(
    *, env: Mapping[str, str], config: Mapping[str, object]
) -> None:
    """The x64 flag and the device topology each separate every key."""
    baseline = _keys_in_fresh_process()
    mutant = _keys_in_fresh_process(env=env, config=config)

    assert not set(baseline.values()) & set(mutant.values())
