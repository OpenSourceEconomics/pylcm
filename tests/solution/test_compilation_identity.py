"""Executables are keyed by what a program computes, not by which object computes it.

A compiled solve program is identified by the model it belongs to, the regime and
core it serves, the engine's grouping of its period, and the solver's own grouping
of that period. None of those is a `id()` of a Python callable, so the key two
constructions of one model produce is the same key, and two periods a solver built
from identical inputs reach one executable.
"""

from collections.abc import Hashable

import pytest

from _lcm.solution import backward_induction
from _lcm.solution.backward_induction import _lowering_key, _program_identity
from lcm import AgeGrid, DiscreteGrid, LinSpacedGrid, Model
from lcm.exceptions import ExecutionPlanningError
from lcm.solvers import GridSearch
from tests.solution.test_dcegm_age_specialized_function import _twin
from tests.test_models import n_nbegm_toy
from tests.test_models.dcegm_paper_twin import get_params as twin_params
from tests.test_models.deterministic.regression import (
    START_AGE,
    LaborSupply,
    RegimeId,
    dead,
    get_params,
    working_life,
)

_N_PERIODS = 3
_IDENTITY = ("program", "fingerprint", "working_life", "main", ("signature",), None)


def _model(*, n_wealth_points: int = 3) -> Model:
    """A two-regime grid-search toy whose wealth grid size is a build input."""
    final_age_alive = START_AGE + _N_PERIODS - 2
    return Model(
        regimes={
            "working_life": working_life.replace(
                active=lambda age: age <= final_age_alive,
                states={
                    "wealth": LinSpacedGrid(start=1, stop=3, n_points=n_wealth_points)
                },
                actions={
                    "labor_supply": DiscreteGrid(category_class=LaborSupply),
                    "consumption": LinSpacedGrid(start=1, stop=3, n_points=3),
                },
                solver=GridSearch(),
            ),
            "dead": dead,
        },
        ages=AgeGrid(start=START_AGE, stop=final_age_alive + 1, step="Y"),
        regime_id_class=RegimeId,
    )


def _capture_lowering_keys(monkeypatch) -> list[dict[Hashable, Hashable]]:
    """Collect the lowering keys of every solve run while the spy is installed."""
    captured: list[dict[Hashable, Hashable]] = []
    original = backward_induction._resolve_output_layouts_and_lowering_keys

    def _spy(**kwargs):
        result = original(**kwargs)
        captured.append(dict(result[1]))
        return result

    monkeypatch.setattr(
        backward_induction, "_resolve_output_layouts_and_lowering_keys", _spy
    )
    return captured


def test_lowering_key_ignores_callable_identity() -> None:
    """One program identity yields one key, whatever object carries the program."""
    first = _lowering_key(
        program_identity=_IDENTITY,
        layout_key=("layout",),
        arguments=None,
        specialization_key=("specialization",),
        output_roles=None,
    )
    second = _lowering_key(
        program_identity=_IDENTITY,
        layout_key=("layout",),
        arguments=None,
        specialization_key=("specialization",),
        output_roles=None,
    )
    assert first == second


def test_lowering_key_separates_distinct_program_identities() -> None:
    """Two programs of different identity never share one key."""
    other = ("program", "fingerprint", "working_life", "main", ("other",), None)
    first = _lowering_key(program_identity=_IDENTITY, layout_key=("layout",))
    second = _lowering_key(program_identity=other, layout_key=("layout",))

    assert first != second


def test_program_identity_separates_distinct_solver_group_keys() -> None:
    """A solver grouping finer than the engine's splits the identity."""
    common = {
        "model_fingerprint": "fingerprint",
        "regime_name": "working_life",
        "core_name": "main",
        "period_signature": ("period-signature", 1),
    }
    first = _program_identity(**common, solver_group_key=("egm", 0))
    second = _program_identity(**common, solver_group_key=("egm", 1))

    assert first != second


def test_program_identity_separates_distinct_model_fingerprints() -> None:
    """Two models never share a program identity, however alike their regimes."""
    common = {
        "regime_name": "working_life",
        "core_name": "main",
        "period_signature": ("period-signature", 1),
        "solver_group_key": None,
    }
    first = _program_identity(**common, model_fingerprint="first")
    second = _program_identity(**common, model_fingerprint="second")

    assert first != second


def test_two_constructions_of_one_model_produce_equal_compilation_keys(
    monkeypatch,
) -> None:
    """Rebuilding a model reproduces every compilation key it had before."""
    captured = _capture_lowering_keys(monkeypatch)
    params = get_params(n_periods=_N_PERIODS)
    _model().solve(params=params, log_level="off")
    _model().solve(params=params, log_level="off")

    first, second = captured
    assert first == second


def test_models_differing_in_a_grid_size_share_no_compilation_key(monkeypatch) -> None:
    """A model built from different inputs gets its own executables."""
    captured = _capture_lowering_keys(monkeypatch)
    params = get_params(n_periods=_N_PERIODS)
    _model().solve(params=params, log_level="off")
    _model(n_wealth_points=4).solve(params=params, log_level="off")

    first, second = captured
    assert not set(first.values()) & set(second.values())


def test_equal_keys_over_different_callables_are_refused(monkeypatch) -> None:
    """A program identity coarser than the solver's specialization is a defect."""
    model = _twin(solver_kind="brute_force", age_specialized=True)
    monkeypatch.setattr(
        backward_induction,
        "_program_identity",
        lambda **_kwargs: ("program", "constant"),
    )
    with pytest.raises(ExecutionPlanningError, match="different callables"):
        model.solve(params=twin_params(), log_level="off")


def test_a_grouping_solver_publishes_one_group_key_per_active_period() -> None:
    """An EGM-family regime reports the group key it built each period under."""
    model = n_nbegm_toy.build_model(variant="negm", n_periods=3)
    regime = model._regimes["alive"]

    assert set(regime.solution.solver_period_group_keys) == set(regime.active_periods)


def test_grid_search_publishes_no_solver_period_group_key() -> None:
    """A solver whose periods are grouped by the engine alone reports nothing."""
    regime = _model()._regimes["working_life"]

    assert dict(regime.solution.solver_period_group_keys) == {}
