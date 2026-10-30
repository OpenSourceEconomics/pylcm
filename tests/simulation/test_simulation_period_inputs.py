"""Concrete raw operands and selected replay occurrences in one regime unit."""

import jax.numpy as jnp
import pytest

from _lcm.egm.published_policy import NBEGMGridPolicy
from _lcm.simulation.period_inputs import (
    POLICY_SCORE,
    unit_value_reads,
)
from lcm.solver_api import SIMULATION_POLICY, ActionOutput
from tests.regime_building.test_collective_regime_simulate import _solve_dissolution


@pytest.fixture(scope="module")
def dissolution():
    """A solved participation arrival, whose couple dissolves at the middle wage."""
    return _solve_dissolution()


@pytest.mark.parametrize("external", [False, True])
def test_selected_external_reader_omits_unused_legacy_policy_occurrences(
    *,
    dissolution,
    external: bool,
) -> None:
    _, regimes, _, _, values = dissolution

    def reader(*, states, fallback_actions):
        del states
        return ActionOutput(actions=fallback_actions)

    reads = unit_value_reads(
        regime=regimes["married"],
        name="married",
        period=0,
        values=values,
        policy=NBEGMGridPolicy(action=jnp.array([1.0, 2.0]), state_names=("wage",)),
        reader=reader if external else None,
    )
    policy_reads = [
        read for read in reads if read.target.artifact_key == SIMULATION_POLICY
    ]
    score_reads = [read for read in reads if read.source.core_key == POLICY_SCORE]
    if external:
        assert not policy_reads
        assert not score_reads
        assert any(
            read.source.core_key == "simulation_external_replay_score" for read in reads
        )
    else:
        assert len(policy_reads) == 1
        assert policy_reads[0].source.argument == "payload"
        assert policy_reads[0].source.path == (0,)
        assert score_reads
