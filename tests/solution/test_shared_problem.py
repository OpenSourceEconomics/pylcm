"""Regimes that hold one problem solve and simulate like their hand-declared twins.

Two couples hold one collective problem and differ only in the rule of their exit:
one dissolves unless both members are willing to stay, the other only if neither
is. Two singles hold one single's problem without an exit. The couple's problem
reads a current shock and a model-level state. In the second and last period every
regime moves on to a terminal regime of its kind that pays the same flow again. The
twin model declares every slot by hand on each regime, so any difference between
the two is the language's.

Closed form at the couple's cells in the first period, with an outside value of
`2 * wealth` per period for each member and a mean-zero shock in the wife's utility
(the discounted second period scales every comparison alike):

- the wife gets `3 * wealth` in the couple and never wants to leave;
- the husband gets `wealth + skill`, so he wants to leave everywhere except at
  `wealth = 1, skill = 1`, where he is indifferent and stays;
- the unilateral couple therefore dissolves where the husband wants to leave, and
  the mutual-consent couple never does.
"""

import dataclasses
from collections.abc import Callable, Mapping

import cloudpickle
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

from _lcm.regime_law import bind_regime_law
from _lcm.solution.fingerprint import (
    _project_user_regime_declaration,
    _semantic_fingerprint,
)
from lcm import (
    CollectiveUtility,
    DiscreteGrid,
    ExecutionConfig,
    Exit,
    ExitTarget,
    InitialNodes,
    LinSpacedGrid,
    Model,
    MutualConsent,
    Periods,
    Problem,
    Regime,
    Unilateral,
    categorical,
    fixed_transition,
)
from lcm.exceptions import RegimeInitializationError
from lcm.exits import ExitRule
from lcm.result import SimulationResult
from lcm.shocks import GaussHermite, Normal, Product, ShockGroup
from lcm.solver_api import SolutionResult
from lcm.typing import ContinuousState, DiscreteState, FloatND, RegimeName, ScalarInt

_PARAMS = {"discount_factor": 0.9}
_COUPLES = ("couple_u", "couple_m")
_FIRST_PERIOD = (*_COUPLES, "single_f", "single_m")
_LAST_PERIOD = ("couple_end", "single_end")
_REGIME_NAMES = (*_FIRST_PERIOD, *_LAST_PERIOD)
_EDGES = {
    "couple_u": {"couple_end": Periods(values=(0,))},
    "couple_m": {"couple_end": Periods(values=(0,))},
    "single_f": {"single_end": Periods(values=(0,))},
    "single_m": {"single_end": Periods(values=(0,))},
}


@categorical(ordered=False)
class _Household:
    couple_u: ScalarInt
    couple_m: ScalarInt
    single_f: ScalarInt
    single_m: ScalarInt
    couple_end: ScalarInt
    single_end: ScalarInt


@categorical(ordered=True)
class _Skill:
    low: ScalarInt
    high: ScalarInt


def _wife(*, wealth: ContinuousState, match: FloatND) -> FloatND:
    return 3.0 * wealth + match


def _skilled_husband(*, wealth: ContinuousState, skill: DiscreteState) -> FloatND:
    return wealth + skill


def _husband(wealth: ContinuousState) -> FloatND:
    return wealth


def _outside(wealth: ContinuousState) -> FloatND:
    return 2.0 * wealth


def _wealth() -> LinSpacedGrid:
    return LinSpacedGrid(start=1.0, stop=2.0, n_points=2)


def _reluctant_husband(wealth: ContinuousState) -> FloatND:
    return 0.5 * wealth


def _variant_household() -> CollectiveUtility:
    return CollectiveUtility(utilities={"f": _wife, "m": _reluctant_husband})


def _household(*, skill: bool) -> CollectiveUtility:
    return CollectiveUtility(
        utilities={"f": _wife, "m": _skilled_husband if skill else _husband}
    )


def _couple_problem(*, last: bool, skill: bool) -> Problem:
    return Problem(
        states={"wealth": _wealth()},
        state_transitions={} if last else {"wealth": fixed_transition("wealth")},
        functions={"utility": _household(skill=skill)},
        shocks=("match",),
    )


def _single_problem(*, last: bool) -> Problem:
    return Problem(
        states={"wealth": _wealth()},
        state_transitions={} if last else {"wealth": fixed_transition("wealth")},
        functions={"utility": _outside},
    )


def _dissolution(*, rule: ExitRule, skill: bool) -> Exit:
    return Exit(
        rule=rule,
        targets={
            member: ExitTarget(
                regime=f"single_{member}",
                projection={"wealth": fixed_transition("wealth")}
                | ({"skill": fixed_transition("skill")} if skill else {}),
            )
            for member in ("f", "m")
        },
    )


def _shared_regimes(*, skill: bool) -> dict[RegimeName, Regime]:
    couple = _couple_problem(last=False, skill=skill)
    single = _single_problem(last=False)
    return {
        "couple_u": Regime(
            problem=couple, exit=_dissolution(rule=Unilateral(), skill=skill)
        ),
        "couple_m": Regime(
            problem=couple, exit=_dissolution(rule=MutualConsent(), skill=skill)
        ),
        "single_f": Regime(problem=single),
        "single_m": Regime(problem=single),
        "couple_end": Regime(problem=_couple_problem(last=True, skill=skill)),
        "single_end": Regime(problem=_single_problem(last=True)),
    }


def _twin_regimes() -> dict[RegimeName, Regime]:
    def couple(*, last: bool, rule: ExitRule | None) -> Regime:
        return Regime(
            states={"wealth": _wealth()},
            state_transitions={} if last else {"wealth": fixed_transition("wealth")},
            functions={"utility": _household(skill=True)},
            shocks=("match",),
            exit=None if rule is None else _dissolution(rule=rule, skill=True),
        )

    def single(*, last: bool) -> Regime:
        return Regime(
            states={"wealth": _wealth()},
            state_transitions={} if last else {"wealth": fixed_transition("wealth")},
            functions={"utility": _outside},
        )

    return {
        "couple_u": couple(last=False, rule=Unilateral()),
        "couple_m": couple(last=False, rule=MutualConsent()),
        "single_f": single(last=False),
        "single_m": single(last=False),
        "couple_end": couple(last=True, rule=None),
        "single_end": single(last=True),
    }


def _model(*, regimes: Mapping[RegimeName, Regime], skill: bool) -> Model:
    """Build the model, with the model-level state `skill` or without it."""
    return Model(
        regimes=regimes,
        states={"skill": DiscreteGrid(category_class=_Skill)} if skill else {},
        state_transitions={"skill": fixed_transition("skill")} if skill else {},
        shocks={
            "match": ShockGroup(
                shocks={"match": Normal(mu=0.0, sigma=0.5)},
                nodes=Product(match=GaussHermite(n=2)),
            )
        },
        n_periods=2,
        regime_id_class=_Household,
        initial_nodes=InitialNodes(by_period={0: _COUPLES}),
        edges=_EDGES,
        execution_config=ExecutionConfig(device_memory_bytes=None),
    )


@pytest.fixture(scope="module")
def shared_regimes() -> dict[RegimeName, Regime]:
    return _shared_regimes(skill=True)


@pytest.fixture(scope="module")
def shared(shared_regimes: dict[RegimeName, Regime]) -> Model:
    return _model(regimes=shared_regimes, skill=True)


@pytest.fixture(scope="module")
def twin() -> Model:
    return _model(regimes=_twin_regimes(), skill=True)


@pytest.fixture(scope="module")
def shared_solution(shared: Model) -> SolutionResult:
    return shared.solve(params=_PARAMS, log_level="off")


@pytest.fixture(scope="module")
def twin_solution(twin: Model) -> SolutionResult:
    return twin.solve(params=_PARAMS, log_level="off")


def _simulate(*, model: Model, solution: SolutionResult) -> pd.DataFrame:
    roles = [model.member_names_to_ids[member] for member in ("f", "m")] * 4
    result: SimulationResult = model.simulate(
        params=_PARAMS,
        solution=solution,
        initial_conditions={
            "period": jnp.zeros(8, dtype=jnp.int32),
            "wealth": jnp.asarray([1.0, 1.0, 2.0, 2.0] * 2),
            "skill": jnp.asarray([1, 1, 0, 0, 1, 1, 0, 0], dtype=jnp.int32),
            "regime_id": jnp.asarray(
                [_Household.couple_u] * 4 + [_Household.couple_m] * 4,
                dtype=jnp.int32,
            ),
            "own_member": jnp.asarray(roles, dtype=jnp.int32),
        },
        seed=0,
        log_level="off",
    )
    return result.to_dataframe()


@pytest.mark.parametrize(
    ("period", "regime_name"),
    [(0, name) for name in _COUPLES] + [(1, name) for name in _LAST_PERIOD],
)
def test_shared_problem_solves_bitwise_like_its_hand_declared_twin(
    *,
    shared_solution: SolutionResult,
    twin_solution: SolutionResult,
    period: int,
    regime_name: RegimeName,
) -> None:
    np.testing.assert_array_equal(
        np.asarray(shared_solution.values[period][regime_name]),
        np.asarray(twin_solution.values[period][regime_name]),
    )


def test_shared_problem_has_its_twin_s_model_fingerprint(
    *, shared_solution: SolutionResult, twin_solution: SolutionResult
) -> None:
    assert (
        shared_solution.metadata.model_fingerprint
        == twin_solution.metadata.model_fingerprint
    )


def test_shared_problem_simulates_like_its_hand_declared_twin(
    *,
    shared: Model,
    twin: Model,
    shared_solution: SolutionResult,
    twin_solution: SolutionResult,
) -> None:
    pd.testing.assert_frame_equal(
        _simulate(model=shared, solution=shared_solution),
        _simulate(model=twin, solution=twin_solution),
    )


def test_the_exit_rule_of_each_holder_decides_its_value(
    shared_solution: SolutionResult,
) -> None:
    """The unilateral couple dissolves where the husband would leave; the
    mutual-consent couple does not, so their values differ there."""
    values = shared_solution.values[0]
    assert not np.array_equal(
        np.asarray(values["couple_u"]), np.asarray(values["couple_m"])
    )


@pytest.mark.parametrize("slot", ["states", "state_transitions", "functions", "shocks"])
def test_two_couples_hold_one_problem_s_objects(slot: str) -> None:
    regimes = _shared_regimes(skill=True)
    assert getattr(regimes["couple_u"], slot) is getattr(regimes["couple_m"], slot)


def test_shared_problem_has_its_twin_s_params_template(
    *, shared: Model, twin: Model
) -> None:
    assert shared.get_params_template() == twin.get_params_template()


def test_results_name_regimes_not_problems(shared_solution: SolutionResult) -> None:
    """The singles are only read as outside values, so no value of theirs is kept;
    the two couples holding one problem keep one each, by regime name."""
    assert set(shared_solution.values[0]) == set(_COUPLES)


def test_regime_ids_name_regimes_not_problems(shared: Model) -> None:
    assert set(shared.regime_names_to_ids) == set(_REGIME_NAMES)


def test_edges_name_regimes_not_problems(shared: Model) -> None:
    assert shared.edges == _EDGES


def test_simulated_rows_name_regimes_not_problems(
    *, shared: Model, shared_solution: SolutionResult
) -> None:
    frame = _simulate(model=shared, solution=shared_solution)
    assert set(frame["regime_name"]) <= set(_REGIME_NAMES)


@pytest.mark.parametrize("regime_name", _COUPLES)
def test_a_finalized_holder_holds_no_problem(
    *, shared: Model, regime_name: RegimeName
) -> None:
    """Finalization writes the model's own slots into each regime, so a finalized
    regime declares its slots itself and no longer holds the problem."""
    assert shared.user_regimes[regime_name].problem is None


def _holders(
    problems: Mapping[RegimeName, Problem | None],
) -> set[frozenset[RegimeName]]:
    """Group the regimes by the problem object they map to, leaving out those
    without one."""
    return {
        frozenset(name for name, other in problems.items() if other is problem)
        for problem in problems.values()
        if problem is not None
    }


_HOLDERS = {
    frozenset(_COUPLES),
    frozenset({"single_f", "single_m"}),
    frozenset({"couple_end"}),
    frozenset({"single_end"}),
}


@pytest.mark.parametrize("regime_name", _REGIME_NAMES)
def test_the_model_names_the_problem_each_regime_was_given(
    *, shared: Model, shared_regimes: dict[RegimeName, Regime], regime_name: RegimeName
) -> None:
    assert shared.problems[regime_name] is shared_regimes[regime_name].problem


def test_the_model_names_a_problem_for_every_regime(shared: Model) -> None:
    assert set(shared.problems) == set(_REGIME_NAMES)


def test_regimes_holding_one_problem_map_to_one_object(shared: Model) -> None:
    """The couples hold one problem, the singles another, and each terminal regime
    a problem of its own."""
    assert _holders(shared.problems) == _HOLDERS


@pytest.fixture(scope="module")
def mixed() -> Model:
    """The terminal regimes declare their slots themselves."""
    regimes = _shared_regimes(skill=True)
    for name in _LAST_PERIOD:
        regimes[name] = regimes[name].replace(problem=None)
    return _model(regimes=regimes, skill=True)


def test_a_regime_without_a_problem_maps_to_none(mixed: Model) -> None:
    assert {name: mixed.problems[name] for name in _LAST_PERIOD} == dict.fromkeys(
        _LAST_PERIOD
    )
    assert _holders(mixed.problems) == {
        frozenset(_COUPLES),
        frozenset({"single_f", "single_m"}),
    }


def test_a_model_of_regimes_declaring_their_slots_knows_no_problems(
    twin: Model,
) -> None:
    assert dict(twin.problems) == dict.fromkeys(_REGIME_NAMES)


@pytest.mark.parametrize("regime_name", _REGIME_NAMES)
def test_the_model_names_the_problem_its_finalized_regime_no_longer_holds(
    *, shared: Model, regime_name: RegimeName
) -> None:
    assert (
        shared.user_regimes[regime_name].problem,
        shared.problems[regime_name] is not None,
    ) == (None, True)


def test_the_problems_of_a_model_cannot_be_changed(shared: Model) -> None:
    with pytest.raises(TypeError):
        shared.problems["couple_u"] = None  # ty: ignore[invalid-assignment]
    with pytest.raises(AttributeError):
        shared.problems = {}  # ty: ignore[invalid-assignment]


@pytest.fixture(scope="module")
def restored(shared: Model) -> Model:
    return cloudpickle.loads(cloudpickle.dumps(shared))


def test_a_pickled_model_keeps_which_regimes_share_a_problem(restored: Model) -> None:
    assert _holders(restored.problems) == _HOLDERS


@pytest.mark.parametrize("regime_name", _REGIME_NAMES)
def test_a_pickled_model_keeps_the_problem_of_each_regime(
    *, shared: Model, restored: Model, regime_name: RegimeName
) -> None:
    """The restored problem is a copy that declares what the original declares.

    A fixed state's law compares by identity, so a copy of a problem holding one
    is never `==` to the original; the two are compared as the model fingerprints
    a regime that holds each.
    """
    original, copy = shared.problems[regime_name], restored.problems[regime_name]
    assert original is not None
    assert copy is not None
    law = bind_regime_law(None)
    assert (
        copy is not original,
        _semantic_fingerprint(
            _project_user_regime_declaration(Regime(problem=copy), law=law)
        ),
    ) == (
        True,
        _semantic_fingerprint(
            _project_user_regime_declaration(Regime(problem=original), law=law)
        ),
    )


def test_a_model_archive_without_problems_knows_none(shared: Model) -> None:
    """An archive written by a model that recorded no problems restores one that
    names none."""
    state = shared.__getstate__()
    del state["_problems"]
    model = Model.__new__(Model)
    model.__setstate__(state)
    assert dict(model.problems) == dict.fromkeys(_REGIME_NAMES)


@pytest.fixture(scope="module")
def detached_solution() -> SolutionResult:
    """The shared model built from its regimes after each let go of its problem."""
    regimes = {
        name: regime.replace(problem=None)
        for name, regime in _shared_regimes(skill=True).items()
    }
    return _model(regimes=regimes, skill=True).solve(params=_PARAMS, log_level="off")


def test_the_problems_of_a_model_do_not_enter_its_fingerprint(
    *, shared_solution: SolutionResult, detached_solution: SolutionResult
) -> None:
    assert (
        shared_solution.metadata.model_fingerprint
        == detached_solution.metadata.model_fingerprint
    )


@pytest.fixture(scope="module")
def without_skill() -> Model:
    return _model(regimes=_shared_regimes(skill=False), skill=False)


@pytest.fixture(scope="module")
def rebuilt(without_skill: Model) -> Model:
    """The model built again from its own finalized regimes.

    It has no model-level state: a finalized regime carries such a state as its
    own, and the rebuilt model would then declare it on both levels.
    """
    return _model(regimes=without_skill.user_regimes, skill=False)


@pytest.mark.parametrize("regime_name", _COUPLES)
def test_a_model_rebuilt_from_finalized_holders_holds_no_problem(
    *, rebuilt: Model, regime_name: RegimeName
) -> None:
    assert rebuilt.user_regimes[regime_name].problem is None


def test_a_model_rebuilt_from_finalized_holders_knows_no_problems(
    rebuilt: Model,
) -> None:
    """The finalized regimes hold no problem, so the model they build names none."""
    assert dict(rebuilt.problems) == dict.fromkeys(_REGIME_NAMES)


@pytest.mark.parametrize("regime_name", _COUPLES)
def test_finalized_regimes_rebuild_the_model(
    *, without_skill: Model, rebuilt: Model, regime_name: RegimeName
) -> None:
    """Finalized holders build the same model again and solve to the same values."""
    np.testing.assert_array_equal(
        np.asarray(
            rebuilt.solve(params=_PARAMS, log_level="off").values[0][regime_name]
        ),
        np.asarray(
            without_skill.solve(params=_PARAMS, log_level="off").values[0][regime_name]
        ),
    )


_REPLACE_APIS = {
    "dataclasses.replace": dataclasses.replace,
    "Regime.replace": Regime.replace,
}


@pytest.mark.parametrize("replace", _REPLACE_APIS.values(), ids=_REPLACE_APIS)
def test_a_couple_given_a_variant_of_its_problem_is_refused(
    replace: Callable[..., Regime],
) -> None:
    """The variant keeps every object of the couple's problem, its solver
    included, except the utility, which the couple carries from its first
    problem."""
    couple = _couple_problem(last=False, skill=True)
    holder = Regime(problem=couple, exit=_dissolution(rule=MutualConsent(), skill=True))
    variant = dataclasses.replace(couple, functions={"utility": _variant_household()})
    with pytest.raises(
        RegimeInitializationError, match=r"but this one also declares \['functions'\]"
    ):
        replace(holder, problem=variant)


@pytest.fixture(scope="module")
def variant_solution() -> SolutionResult:
    """`couple_m` holds a variant of the couples' problem with another husband."""
    regimes = _shared_regimes(skill=True)
    couple = regimes["couple_u"].problem
    assert couple is not None
    regimes["couple_m"] = Regime(
        problem=dataclasses.replace(
            couple, functions={"utility": _variant_household()}
        ),
        exit=_dissolution(rule=MutualConsent(), skill=True),
    )
    return _model(regimes=regimes, skill=True).solve(params=_PARAMS, log_level="off")


@pytest.fixture(scope="module")
def variant_twin_solution() -> SolutionResult:
    regimes = _twin_regimes()
    regimes["couple_m"] = regimes["couple_m"].replace(
        functions={"utility": _variant_household()}
    )
    return _model(regimes=regimes, skill=True).solve(params=_PARAMS, log_level="off")


@pytest.mark.parametrize(
    ("period", "regime_name"),
    [(0, name) for name in _COUPLES] + [(1, name) for name in _LAST_PERIOD],
)
def test_a_holder_of_a_variant_problem_solves_like_the_variant_s_twin(
    *,
    variant_solution: SolutionResult,
    variant_twin_solution: SolutionResult,
    period: int,
    regime_name: RegimeName,
) -> None:
    np.testing.assert_array_equal(
        np.asarray(variant_solution.values[period][regime_name]),
        np.asarray(variant_twin_solution.values[period][regime_name]),
    )


def test_a_holder_of_a_variant_problem_has_the_variant_s_twin_s_fingerprint(
    *, variant_solution: SolutionResult, variant_twin_solution: SolutionResult
) -> None:
    assert (
        variant_solution.metadata.model_fingerprint
        == variant_twin_solution.metadata.model_fingerprint
    )


def test_the_variant_problem_changes_the_couple_s_value(
    *, variant_solution: SolutionResult, twin_solution: SolutionResult
) -> None:
    """The variant's husband values the couple less than the first problem's."""
    assert not np.array_equal(
        np.asarray(variant_solution.values[0]["couple_m"]),
        np.asarray(twin_solution.values[0]["couple_m"]),
    )
