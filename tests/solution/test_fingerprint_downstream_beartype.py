"""Durable fingerprints of model functions guarded by a downstream beartype claw.

A package that builds models on pylcm may install its own beartype claw, so every
function it hands to a `Regime` arrives wrapped in a beartype guard that pylcm never
captured at import. The guard is transparent to the economics: the fingerprint has to
see through it to the callee, while a wrapper that merely wears beartype's marks stays
refused.
"""

from collections.abc import Callable
from pathlib import Path

import jax.numpy as jnp
import pytest
from beartype import BeartypeConf, BeartypeStrategy, beartype
from pandas.testing import assert_frame_equal

from _lcm.solution import fingerprint as fingerprints
from lcm import AgeGrid, DiscreteGrid, LinSpacedGrid, Model
from lcm.exceptions import InvalidSimulationInputError
from lcm.persistence import load_solution
from lcm.typing import FloatND, UserInitialConditions, UserParams
from lcm_examples.mortality import LaborSupply
from tests.test_models.deterministic.regression import (
    START_AGE,
    RegimeId,
    dead,
    get_params,
    working_life,
)

# The configuration a downstream package's claw applies to every function it
# defines; the exact settings are not what matters, only that they are not pylcm's.
_DOWNSTREAM_CONF = BeartypeConf(
    is_color=False,
    is_pep484_tower=True,
    strategy=BeartypeStrategy.On,
)


def _downstream_terminal_utility(scale: float) -> Callable[[], FloatND]:
    """Terminal utility `scale`, guarded the way a downstream claw guards it."""

    def utility() -> FloatND:
        return jnp.asarray(scale, dtype=float)

    return beartype(conf=_DOWNSTREAM_CONF)(utility)


def _downstream_model(scale: float) -> Model:
    """Two-period GridSearch model whose terminal payoff is a guarded closure."""
    final_age_alive = START_AGE
    grid = LinSpacedGrid(start=1, stop=3, n_points=3)
    return Model(
        regimes={
            "working_life": working_life.replace(
                active=lambda age: age <= final_age_alive,
                states={"wealth": grid},
                actions={
                    "labor_supply": DiscreteGrid(category_class=LaborSupply),
                    "consumption": grid,
                },
            ),
            "dead": dead.replace(
                functions={"utility": _downstream_terminal_utility(scale)}
            ),
        },
        ages=AgeGrid(start=START_AGE, stop=final_age_alive + 1, step="Y"),
        regime_id_class=RegimeId,
    )


def _downstream_model_inputs() -> tuple[UserParams, UserInitialConditions]:
    params = get_params(n_periods=2)
    initial_conditions: UserInitialConditions = {
        "wealth": jnp.asarray([2.0]),
        "age": jnp.asarray([18.0]),
        "regime_id": jnp.asarray([RegimeId.working_life], dtype=jnp.int32),
    }
    return params, initial_conditions


def test_downstream_guard_is_a_beartype_wrapper_pylcm_never_captured() -> None:
    """The specimen is a genuine beartype wrapper around a distinct callee."""
    guarded = _downstream_terminal_utility(1.0)
    captured = {
        wrapper
        for wrapper, _code, _wrapped in fingerprints._TRUSTED_BEARTYPE_WRAPPER_CAPTURES
    }
    assert (
        guarded.__dict__.get("__beartype_wrapper"),
        guarded.__wrapped__ is guarded,  # ty: ignore[unresolved-attribute]
        guarded in captured,
    ) == (True, False, False)


def test_downstream_beartype_guards_fingerprint_deterministically() -> None:
    """Two guards around equal callees carry the same durable identity."""
    assert fingerprints._semantic_fingerprint(
        _downstream_terminal_utility(2.0)
    ) == fingerprints._semantic_fingerprint(_downstream_terminal_utility(2.0))


def test_downstream_beartype_guards_are_transparent_to_their_callee() -> None:
    """What the guarded callee computes is what enters the identity."""
    assert fingerprints._semantic_fingerprint(
        _downstream_terminal_utility(2.0)
    ) != fingerprints._semantic_fingerprint(_downstream_terminal_utility(3.0))


def test_downstream_guard_whose_callee_mark_was_repointed_is_refused() -> None:
    """A guard that no longer names one bound callee cannot be seen through."""
    guarded = _downstream_terminal_utility(1.0)
    guarded.__wrapped__ = _downstream_terminal_utility(  # ty: ignore[unresolved-attribute]
        2.0
    ).__wrapped__

    with pytest.raises(TypeError, match="uncaptured transparent beartype wrapper"):
        fingerprints._semantic_fingerprint(guarded)


def test_downstream_guarded_model_replays_its_archive_in_a_rebuilt_model(
    tmp_path: Path,
) -> None:
    """A model guarded by a downstream claw solves, persists, and replays."""
    params, initial_conditions = _downstream_model_inputs()
    model = _downstream_model(1.0)
    solved = model.solve(params=params, log_level="off")
    restored = load_solution(path=solved.save(path=tmp_path / "guarded.lcm"))

    expected = model.simulate(
        params=params,
        initial_conditions=initial_conditions,
        solution=solved,
        log_level="off",
    ).to_dataframe()
    actual = (
        _downstream_model(1.0)
        .simulate(
            params=params,
            initial_conditions=initial_conditions,
            solution=restored,
            log_level="off",
        )
        .to_dataframe()
    )

    assert_frame_equal(actual, expected)


def test_downstream_guarded_model_refuses_an_archive_of_a_different_callee(
    tmp_path: Path,
) -> None:
    """The guard does not hide a changed payoff from the model fingerprint."""
    params, initial_conditions = _downstream_model_inputs()
    solved = _downstream_model(1.0).solve(params=params, log_level="off")
    restored = load_solution(path=solved.save(path=tmp_path / "guarded.lcm"))

    with pytest.raises(InvalidSimulationInputError, match="model_fingerprint"):
        _downstream_model(2.0).simulate(
            params=params,
            initial_conditions=initial_conditions,
            solution=restored,
            log_level="off",
        )
