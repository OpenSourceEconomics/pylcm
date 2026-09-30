from typing import Any

import jax.numpy as jnp
import pytest

from _lcm import transition_checks
from lcm.exceptions import InvalidRegimeTransitionProbabilitiesError
from lcm_examples.mortality import get_model, get_params

N_PERIODS = 4


@pytest.fixture
def single_calls(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Record every per-period regime-transition validation call."""
    calls: list[int] = []
    original = transition_checks._validate_regime_transition_single

    def counting(**kwargs: Any) -> None:
        calls.append(1)
        original(**kwargs)

    monkeypatch.setattr(
        transition_checks, "_validate_regime_transition_single", counting
    )
    return calls


def test_process_params_skips_transition_validation_for_repeated_params(
    single_calls: list[int],
) -> None:
    """A second call with identical params runs no per-period transition checks."""
    model = get_model(N_PERIODS)
    params = get_params(n_periods=N_PERIODS)
    model._process_params(params)
    n_first = len(single_calls)
    model._process_params(get_params(n_periods=N_PERIODS))
    assert (n_first, len(single_calls)) == (n_first, n_first)


def test_process_params_revalidates_changed_params(
    single_calls: list[int],
) -> None:
    """Changed params values are validated afresh."""
    model = get_model(N_PERIODS)
    model._process_params(get_params(n_periods=N_PERIODS))
    n_first = len(single_calls)
    model._process_params(get_params(n_periods=N_PERIODS, discount_factor=0.9))
    assert len(single_calls) == 2 * n_first


def test_process_params_raises_on_every_call_with_invalid_params() -> None:
    """Invalid regime transitions raise the same error on each repeated call."""
    model = get_model(N_PERIODS)
    params = get_params(
        n_periods=N_PERIODS, survival_probs=jnp.array([2.0] + [0.0] * (N_PERIODS - 2))
    )
    messages = []
    for _ in range(2):
        with pytest.raises(InvalidRegimeTransitionProbabilitiesError) as error:
            model._process_params(params)
        messages.append(str(error.value))
    assert messages[0] == messages[1]
