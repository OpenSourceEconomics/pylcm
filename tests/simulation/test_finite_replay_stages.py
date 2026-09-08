"""Pure replay stages keep complete-bank ranking and diagnostic precedence."""

import functools
import importlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.utils.logging import get_logger
from lcm.exceptions import UnrepresentableOuterCandidateError
from tests.simulation import test_nnbegm_candidate_ranking as specimens

_replay = importlib.import_module("_lcm.simulation.simulate")


def test_reconstruction_and_ranking_compile_with_dynamic_payloads(monkeypatch) -> None:
    """Both numerical stages trace without host reads and rank the complete bank."""
    observed = []

    def compile_stage(*, name, metadata_names):
        original = getattr(_replay, name)

        def compiled(**arguments):
            metadata = {key: arguments.pop(key) for key in metadata_names}
            executable = (
                jax.jit(functools.partial(original, **metadata))
                .lower(**arguments)
                .compile()
            )
            result = executable(**arguments)
            observed.append(name)
            return result

        monkeypatch.setattr(_replay, name, compiled)

    compile_stage(
        name="_prepare_nnbegm_candidate_bank",
        metadata_names=("regime", "period"),
    )
    compile_stage(
        name="_rank_nnbegm_candidate_bank",
        metadata_names=("regime", "period", "action_names"),
    )

    def q_and_f(*, inner, outer, state, next_regime_to_V_arr, period, age):
        del outer, state, next_regime_to_V_arr, period, age
        return inner, jnp.ones_like(inner, dtype=bool)

    actions, values = specimens._synthetic_replay(
        inner=[1.0, 7.0, 7.0],
        outer=[10.0, 20.0, 30.0],
        marker=[100.0, 0.0, 200.0],
        q_and_f=q_and_f,
    )
    np.testing.assert_array_equal(actions["inner"], [7.0])
    np.testing.assert_array_equal(actions["outer"], [20.0])
    np.testing.assert_array_equal(values, [7.0])
    assert observed == [
        "_prepare_nnbegm_candidate_bank",
        "_rank_nnbegm_candidate_bank",
    ]


def test_debug_refuses_unrepresented_candidates_before_scoring(monkeypatch) -> None:
    """A replay-domain error precedes the canonical-Q evaluation as before."""
    scoring_calls = []

    def q_and_f(*, inner, outer, state, next_regime_to_V_arr, period, age):
        del outer, state, next_regime_to_V_arr, period, age
        scoring_calls.append(True)
        return inner, jnp.ones_like(inner, dtype=bool)

    def debug_logger(*, log_level):
        del log_level
        return get_logger(log_level="debug")

    monkeypatch.setattr(specimens, "get_logger", debug_logger)
    with pytest.raises(UnrepresentableOuterCandidateError):
        specimens._synthetic_replay(
            inner=[1.0, 2.0],
            outer=[10.0, 201.0],
            marker=[0.0, 0.0],
            q_and_f=q_and_f,
        )
    assert scoring_calls == []
