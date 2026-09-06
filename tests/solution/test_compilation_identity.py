"""Executables are keyed by what a program computes, not by which object computes it.

A compiled solve program is identified by the model it belongs to, the regime and
core it serves, the engine's grouping of its period, and the solver's own grouping
of that period. None of those is a `id()` of a Python callable, so the key two
constructions of one model produce is the same key, and two periods a solver built
from identical inputs reach one executable.
"""

from collections.abc import Callable, Hashable
from typing import Any

import pytest

from _lcm.solution import backward_induction
from _lcm.solution.backward_induction import _lowering_key, _program_identity
from lcm import AgeGrid, DiscreteGrid, LinSpacedGrid, Model
from lcm.exceptions import ExecutionPlanningError
from lcm.solvers import GridSearch
from lcm.typing import RegimeName
from tests.conftest import EXACT_KERNEL_SKIP_REASON
from tests.solution.test_dcegm_age_specialized_function import _twin
from tests.test_models import n_nbegm_toy, nbegm_ride_along_toy
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
_NNBEGM_PARAMS = {"discount_factor": 0.95}


def _negm_model() -> Model:
    """The two-asset toy solved by `NEGM`."""
    return n_nbegm_toy.build_model(variant="negm", n_periods=_N_PERIODS)


def _nnbegm_model() -> Model:
    """The two-asset toy solved by `NNBEGM` over an `NBEGM` inner margin."""
    return n_nbegm_toy.build_model(variant="n_nbegm", n_periods=_N_PERIODS)


def _nbegm_model() -> Model:
    """The ride-along tax toy solved by `NBEGM`, at its smallest grids."""
    return nbegm_ride_along_toy.build_model(
        variant="nbegm",
        n_periods=_N_PERIODS,
        n_liquid=12,
        n_savings=12,
        n_consumption=12,
    )


def _dcegm_model() -> Model:
    """The DC-EGM paper twin, with no age specialization."""
    return _twin(solver_kind="dcegm", age_specialized=False)


# One case per solver that publishes a group key, so every publisher is
# exercised rather than only the one that happened to be reachable.
_GROUPING_SOLVER_CASES = [
    pytest.param(_negm_model, "alive", id="negm"),
    pytest.param(_nnbegm_model, "alive", id="n_nbegm"),
    pytest.param(_nbegm_model, "alive", id="nbegm"),
    pytest.param(
        _dcegm_model,
        "working_life",
        marks=pytest.mark.requires_exact_affine_kernel(reason=EXACT_KERNEL_SKIP_REASON),
        id="dcegm",
    ),
]


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


def _capture_lowering_keys(
    *, monkeypatch: pytest.MonkeyPatch
) -> list[dict[Hashable, Hashable]]:
    """Collect the lowering keys of every solve run while the spy is installed."""
    captured: list[dict[Hashable, Hashable]] = []
    original = backward_induction._resolve_output_layouts_and_lowering_keys

    # `Any` rather than `object`: the spy forwards its keywords untouched to a
    # strictly typed function, so narrowing them here would be a claim it does
    # not make.
    def _spy(**kwargs: Any) -> tuple:
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
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Rebuilding a model reproduces every compilation key it had before."""
    captured = _capture_lowering_keys(monkeypatch=monkeypatch)
    params = get_params(n_periods=_N_PERIODS)
    _model().solve(params=params, log_level="off")
    _model().solve(params=params, log_level="off")

    first, second = captured
    assert first == second


def test_models_differing_in_a_grid_size_share_no_compilation_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A model built from different inputs gets its own executables."""
    captured = _capture_lowering_keys(monkeypatch=monkeypatch)
    params = get_params(n_periods=_N_PERIODS)
    _model().solve(params=params, log_level="off")
    _model(n_wealth_points=4).solve(params=params, log_level="off")

    first, second = captured
    assert not set(first.values()) & set(second.values())


def test_equal_keys_over_different_callables_are_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A program identity coarser than the solver's specialization is a defect."""
    model = _twin(solver_kind="brute_force", age_specialized=True)
    monkeypatch.setattr(
        backward_induction,
        "_program_identity",
        lambda **_kwargs: ("program", "constant"),
    )
    with pytest.raises(ExecutionPlanningError) as refusal:
        model.solve(params=twin_params(), log_level="off")

    # Both colliding addresses are named, so a reader can see which two
    # programs the identity failed to tell apart.
    message = str(refusal.value)
    assert message.count("regime 'working_life', core 'main', period ") == 2


def test_the_key_refusal_names_the_equivalent_callable_case(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The refusal states that a per-period equivalent callable also reaches it.

    A solver whose group key is right but which builds a fresh, equivalent closure
    per period is refused by the same guard, so the message names that possibility
    beside a group key that is too coarse.
    """
    model = _twin(solver_kind="brute_force", age_specialized=True)
    monkeypatch.setattr(
        backward_induction,
        "_program_identity",
        lambda **_kwargs: ("program", "constant"),
    )
    with pytest.raises(ExecutionPlanningError) as refusal:
        model.solve(params=twin_params(), log_level="off")

    assert "equivalent" in str(refusal.value)


@pytest.mark.parametrize(("build_model", "regime_name"), _GROUPING_SOLVER_CASES)
def test_a_grouping_solver_publishes_one_group_key_per_active_period(
    *, build_model: Callable[[], Model], regime_name: RegimeName
) -> None:
    """An EGM-family regime reports the group key it built each period under."""
    regime = build_model()._regimes[regime_name]

    assert set(regime.solution.solver_period_group_keys) == set(regime.active_periods)


def test_two_constructions_of_a_grouped_model_produce_equal_compilation_keys(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A solver-grouped model's compilation keys survive being rebuilt.

    The grid-search companion above cannot see this: its regime publishes no
    solver group key, so nothing of the solver's own grouping — and none of the
    constraint-plan digest inside it — reaches the compilation key there.
    """
    captured = _capture_lowering_keys(monkeypatch=monkeypatch)
    _nnbegm_model().solve(params=_NNBEGM_PARAMS, log_level="off")
    _nnbegm_model().solve(params=_NNBEGM_PARAMS, log_level="off")

    first, second = captured
    assert first == second


def test_two_constructions_publish_equal_solver_period_group_keys() -> None:
    """A solver's own per-period grouping is reproduced when the model is rebuilt.

    Non-empty is asserted alongside equality: two empty mappings are equal, and
    a key the solver never published would make the claim vacuous.
    """
    first = _nnbegm_model()._regimes["alive"]
    second = _nnbegm_model()._regimes["alive"]

    assert (
        dict(first.solution.solver_period_group_keys)
        == dict(second.solution.solver_period_group_keys)
        != {}
    )


def test_grid_search_publishes_no_solver_period_group_key() -> None:
    """A solver whose periods are grouped by the engine alone reports nothing."""
    regime = _model()._regimes["working_life"]

    assert dict(regime.solution.solver_period_group_keys) == {}
