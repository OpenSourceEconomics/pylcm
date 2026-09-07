"""Every shipped reduction's block fold reproduces its dense one-pass result.

The planner may cut a reduced axis into blocks of any width, so each shipped
specification is folded over two partitions that both end in a short block and
compared against a scalar loop over the whole axis.
"""

import jax.numpy as jnp
import numpy as np
import pytest
from beartype.roar import BeartypeCallHintViolation
from numpy.testing import assert_array_almost_equal as aaae

from _lcm.execution.reductions import (
    WEIGHTED_EXPECTATION_REDUCTION,
    WeightedExpectationResult,
    HARD_MAX_WITH_CARRY_REDUCTION,
    OuterCandidateAccumulator,
)
from _lcm.solution.action_reduction import HARD_MAX_REDUCTION, HardMaxResult
from _lcm.solution.collective_action_reduction import (
    COLLECTIVE_HARD_MAX_REDUCTION,
    CollectiveHardMaxResult,
)
from _lcm.solution.logsumexp_action_reduction import (
    LOGSUMEXP_REDUCTION,
    LogSumExpResult,
)
from lcm.typing import BoolND, FloatND, Int1D
from tests.conftest import DECIMAL_PRECISION, assert_agrees_to_ulp

_N_STATES = 3
_N_ACTIONS = 10
_N_STAKEHOLDERS = 2
_SCALE = 0.5
_PARTITIONS = ((4, 4, 2), (3, 3, 3, 1))


def _blocks(*, partition: tuple[int, ...]) -> list[tuple[int, int]]:
    """Return the (start, stop) pairs one partition of the action axis cuts."""
    bounds = []
    start = 0
    for width in partition:
        bounds.append((start, start + width))
        start += width
    assert start == _N_ACTIONS
    return bounds


def _values() -> FloatND:
    """Return small integral action values, so ties are common and exact."""
    rng = np.random.default_rng(seed=20260907)
    drawn = jnp.asarray(rng.integers(0, 4, size=(_N_STATES, _N_ACTIONS)))
    values = drawn.astype(jnp.zeros(()).dtype)
    assert bool(jnp.isfinite(values).all())
    return values


def _stakeholder_values() -> FloatND:
    """Return per-stakeholder values distinct across actions and stakeholders."""
    rng = np.random.default_rng(seed=20260908)
    drawn = jnp.asarray(
        rng.integers(0, 9, size=(_N_STATES, _N_ACTIONS, _N_STAKEHOLDERS))
    )
    values = drawn.astype(jnp.zeros(()).dtype)
    assert bool(jnp.isfinite(values).all())
    return values


def _feasible() -> BoolND:
    """Return a feasibility mask leaving every state at least one live action."""
    rng = np.random.default_rng(seed=20260909)
    drawn = rng.random((_N_STATES, _N_ACTIONS)) > 0.25
    drawn[:, 0] = True
    mask = jnp.asarray(drawn)
    assert bool(jnp.any(~mask))
    assert bool(jnp.all(jnp.any(mask, axis=-1)))
    return mask


def _action_ids() -> Int1D:
    """Return the canonical global identity of each action position."""
    return jnp.arange(_N_ACTIONS, dtype=jnp.int32)


def _dense_hard_max(
    *, values: FloatND, feasible: BoolND
) -> tuple[np.ndarray, np.ndarray]:
    """Scan the whole axis once, keeping the smallest identity among equals."""
    observed = np.asarray(values)
    live = np.asarray(feasible)
    best_value = np.full(_N_STATES, -np.inf, dtype=observed.dtype)
    best_identity = np.full(_N_STATES, -1, dtype=np.int32)
    for state in range(_N_STATES):
        for position in range(_N_ACTIONS):
            if not live[state, position]:
                continue
            candidate = observed[state, position]
            if best_identity[state] == -1 or candidate > best_value[state]:
                best_value[state] = candidate
                best_identity[state] = position
    return best_value, best_identity


def _folded_hard_max(*, partition: tuple[int, ...]) -> HardMaxResult:
    """Fold the action axis in the blocks one partition cuts."""
    values, feasible, action_ids = _values(), _feasible(), _action_ids()
    accumulator = HARD_MAX_REDUCTION.initialize(value_template=jnp.zeros(_N_STATES))
    for start, stop in _blocks(partition=partition):
        accumulator = HARD_MAX_REDUCTION.add(
            accumulator=accumulator,
            values=values[:, start:stop],
            feasible=feasible[:, start:stop],
            action_ids=action_ids[start:stop],
        )
    return HARD_MAX_REDUCTION.finalize(accumulator=accumulator)


@pytest.mark.parametrize("partition", _PARTITIONS, ids=("width-four", "width-three"))
def test_hard_max_block_fold_matches_the_dense_maximum(
    *, partition: tuple[int, ...]
) -> None:
    """Blockwise hard max publishes the maximum a single pass would."""
    expected, _identity = _dense_hard_max(values=_values(), feasible=_feasible())

    aaae(
        np.asarray(_folded_hard_max(partition=partition).best_value),
        expected,
        decimal=DECIMAL_PRECISION,
    )


@pytest.mark.parametrize("partition", _PARTITIONS, ids=("width-four", "width-three"))
def test_hard_max_block_fold_selects_the_dense_winner_identity(
    *, partition: tuple[int, ...]
) -> None:
    """The winning global identity is the one a single pass selects, exactly."""
    _value, expected = _dense_hard_max(values=_values(), feasible=_feasible())

    np.testing.assert_array_equal(
        np.asarray(_folded_hard_max(partition=partition).best_global_action_id),
        expected,
    )


def _dense_logsumexp(*, values: FloatND) -> np.ndarray:
    """Sum the unshifted exponential mass over the whole axis at once."""
    observed = np.asarray(values)
    scale = observed.dtype.type(_SCALE)
    return scale * np.log(np.exp(observed / scale).sum(axis=-1))


def _folded_logsumexp(*, partition: tuple[int, ...]) -> LogSumExpResult:
    """Fold the branch axis at one scale in the blocks a partition cuts."""
    values = _values()
    scale = jnp.asarray(_SCALE, dtype=values.dtype)
    accumulator = LOGSUMEXP_REDUCTION.initialize(value_template=jnp.zeros(_N_STATES))
    for start, stop in _blocks(partition=partition):
        accumulator = LOGSUMEXP_REDUCTION.add(
            accumulator=accumulator,
            values=values[:, start:stop],
            scale=scale,
        )
    return LOGSUMEXP_REDUCTION.finalize(accumulator=accumulator, scale=scale)


@pytest.mark.parametrize("partition", _PARTITIONS, ids=("width-four", "width-three"))
def test_logsumexp_block_fold_matches_the_dense_smoothed_maximum(
    *, partition: tuple[int, ...]
) -> None:
    """Blockwise log-sum-exp agrees with one unshifted pass to a few ULP."""
    assert_agrees_to_ulp(
        got=np.asarray(_folded_logsumexp(partition=partition).smoothed_value),
        expected=_dense_logsumexp(values=_values()),
        n_ulp=16,
    )


def _dense_collective(
    *, objectives: FloatND, stakeholder_values: FloatND, feasible: BoolND
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Scan the whole axis once for the household winner and its readouts."""
    observed = np.asarray(objectives)
    own = np.asarray(stakeholder_values)
    live = np.asarray(feasible)
    best_objective = np.full(_N_STATES, -np.inf, dtype=observed.dtype)
    best_identity = np.full(_N_STATES, -1, dtype=np.int32)
    best_own = np.zeros((_N_STATES, _N_STAKEHOLDERS), dtype=own.dtype)
    for state in range(_N_STATES):
        for position in range(_N_ACTIONS):
            if not live[state, position]:
                continue
            candidate = observed[state, position]
            if best_identity[state] == -1 or candidate > best_objective[state]:
                best_objective[state] = candidate
                best_identity[state] = position
                best_own[state] = own[state, position]
    return best_objective, best_identity, best_own


def _folded_collective(*, partition: tuple[int, ...]) -> CollectiveHardMaxResult:
    """Fold the collective action axis in the blocks one partition cuts."""
    objectives, own, feasible = _values(), _stakeholder_values(), _feasible()
    action_ids = _action_ids()
    accumulator = COLLECTIVE_HARD_MAX_REDUCTION.initialize(
        stakeholder_template=jnp.zeros((_N_STATES, _N_STAKEHOLDERS))
    )
    for start, stop in _blocks(partition=partition):
        accumulator = COLLECTIVE_HARD_MAX_REDUCTION.add(
            accumulator=accumulator,
            objectives=objectives[:, start:stop],
            stakeholder_values=own[:, start:stop, :],
            feasible=feasible[:, start:stop],
            action_ids=action_ids[start:stop],
        )
    return COLLECTIVE_HARD_MAX_REDUCTION.finalize(accumulator=accumulator)


@pytest.mark.parametrize("partition", _PARTITIONS, ids=("width-four", "width-three"))
def test_collective_block_fold_matches_the_dense_household_objective(
    *, partition: tuple[int, ...]
) -> None:
    """Blockwise collective choice publishes the objective a single pass would."""
    expected, _identity, _own = _dense_collective(
        objectives=_values(),
        stakeholder_values=_stakeholder_values(),
        feasible=_feasible(),
    )

    aaae(
        np.asarray(_folded_collective(partition=partition).best_objective),
        expected,
        decimal=DECIMAL_PRECISION,
    )


@pytest.mark.parametrize("partition", _PARTITIONS, ids=("width-four", "width-three"))
def test_collective_block_fold_selects_the_dense_winner_identity(
    *, partition: tuple[int, ...]
) -> None:
    """The household winner's global identity is the dense one, exactly."""
    _objective, expected, _own = _dense_collective(
        objectives=_values(),
        stakeholder_values=_stakeholder_values(),
        feasible=_feasible(),
    )

    np.testing.assert_array_equal(
        np.asarray(_folded_collective(partition=partition).best_global_action_id),
        expected,
    )


@pytest.mark.parametrize("partition", _PARTITIONS, ids=("width-four", "width-three"))
def test_collective_block_fold_reads_stakeholder_values_at_the_winner(
    *, partition: tuple[int, ...]
) -> None:
    """Every stakeholder's value is read at the one action the household picks."""
    _objective, _identity, expected = _dense_collective(
        objectives=_values(),
        stakeholder_values=_stakeholder_values(),
        feasible=_feasible(),
    )

    aaae(
        np.asarray(_folded_collective(partition=partition).best_stakeholder_values),
        expected,
        decimal=DECIMAL_PRECISION,
    )


def _weights() -> FloatND:
    """Return strictly positive node weights summing to one along the axis."""
    rng = np.random.default_rng(seed=20260910)
    drawn = rng.random(_N_ACTIONS) + 0.1
    normalized = jnp.asarray(drawn / drawn.sum()).astype(jnp.zeros(()).dtype)
    assert bool(jnp.all(normalized > 0))
    assert bool(jnp.isfinite(normalized).all())
    return normalized


def _dense_weighted_expectation(*, values: FloatND, weights: FloatND) -> np.ndarray:
    """Average the whole axis at once, weight by weight."""
    observed = np.asarray(values)
    mass = np.asarray(weights)
    return (observed * mass).sum(axis=-1) / mass.sum(axis=-1)


def _folded_weighted_expectation(
    *, partition: tuple[int, ...], values: FloatND, weights: FloatND
) -> WeightedExpectationResult:
    """Fold the node axis in the blocks one partition cuts."""
    reduction = WEIGHTED_EXPECTATION_REDUCTION.bind(subnormal_is_accounted_for=False)
    accumulator = reduction.initialize(value_template=jnp.zeros(_N_STATES))
    for start, stop in _blocks(partition=partition):
        accumulator = reduction.add(
            accumulator=accumulator,
            values=values[:, start:stop],
            weights=weights[start:stop],
        )
    return reduction.finalize(accumulator=accumulator)


@pytest.mark.parametrize("partition", _PARTITIONS, ids=("width-four", "width-three"))
def test_weighted_expectation_block_fold_matches_the_dense_mean(
    *, partition: tuple[int, ...]
) -> None:
    """Blockwise weighted expectation agrees with one dense pass to a few ULP."""
    assert_agrees_to_ulp(
        got=np.asarray(
            _folded_weighted_expectation(
                partition=partition, values=_values(), weights=_weights()
            ).expectation
        ),
        expected=_dense_weighted_expectation(values=_values(), weights=_weights()),
        n_ulp=16,
    )


@pytest.mark.parametrize("partition", _PARTITIONS, ids=("width-four", "width-three"))
def test_weighted_expectation_block_fold_accumulates_the_whole_weight_mass(
    *, partition: tuple[int, ...]
) -> None:
    """The published mass is the sum of every block's weights, to a few ULP."""
    assert_agrees_to_ulp(
        got=np.asarray(
            _folded_weighted_expectation(
                partition=partition, values=_values(), weights=_weights()
            ).weight_mass
        ),
        expected=np.asarray(_weights()).sum(),
        n_ulp=16,
    )


def _candidate_carry() -> FloatND:
    """Return a per-candidate payload row the winner's carry is read from."""
    rng = np.random.default_rng(seed=20260910)
    drawn = jnp.asarray(rng.integers(0, 50, size=(_N_STATES, _N_ACTIONS, 2)))
    carry = drawn.astype(jnp.zeros(()).dtype)
    assert bool(jnp.isfinite(carry).all())
    return carry


def _dense_hard_max_with_carry(
    *, values: FloatND, feasible: BoolND, carry: FloatND
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Scan the whole axis once, keeping the smallest identity among equals."""
    observed = np.asarray(values)
    live = np.asarray(feasible)
    payload = np.asarray(carry)
    best_value = np.full(_N_STATES, -np.inf, dtype=observed.dtype)
    best_identity = np.full(_N_STATES, -1, dtype=np.int32)
    best_carry = np.zeros((_N_STATES, payload.shape[-1]), dtype=payload.dtype)
    for state in range(_N_STATES):
        for position in range(_N_ACTIONS):
            if not live[state, position]:
                continue
            candidate = observed[state, position]
            if best_identity[state] == -1 or candidate > best_value[state]:
                best_value[state] = candidate
                best_identity[state] = position
                best_carry[state] = payload[state, position]
    return best_value, best_identity, best_carry


def _folded_hard_max_with_carry(
    *, partition: tuple[int, ...]
) -> OuterCandidateAccumulator:
    """Fold the candidate axis in the blocks one partition cuts.

    Each block's value fold runs through `add`; the block's winning carry is
    merged in as a state of its own, which is how a driver that owns a payload
    per candidate drives this reduction.
    """
    values, feasible, action_ids = _values(), _feasible(), _action_ids()
    carry = _candidate_carry()
    accumulator = HARD_MAX_WITH_CARRY_REDUCTION.initialize(
        value_template=jnp.zeros(_N_STATES)
    )
    for start, stop in _blocks(partition=partition):
        block = HARD_MAX_WITH_CARRY_REDUCTION.add(
            accumulator=HARD_MAX_WITH_CARRY_REDUCTION.initialize(
                value_template=jnp.zeros(_N_STATES)
            ),
            values=values[:, start:stop],
            feasible=feasible[:, start:stop],
            action_ids=action_ids[start:stop],
        )
        winner = block.best_candidate_id - start
        block_carry = jnp.take_along_axis(
            carry[:, start:stop, :],
            jnp.clip(winner, 0, stop - start - 1)[:, None, None],
            axis=1,
        )[:, 0, :]
        accumulator = HARD_MAX_WITH_CARRY_REDUCTION.merge(
            left=accumulator,
            right=OuterCandidateAccumulator(
                best_value=block.best_value,
                best_candidate_id=block.best_candidate_id,
                carry=block_carry,
            ),
        )
    return HARD_MAX_WITH_CARRY_REDUCTION.finalize(accumulator=accumulator)


@pytest.mark.parametrize("partition", _PARTITIONS, ids=("width-four", "width-three"))
def test_hard_max_with_carry_block_fold_matches_the_dense_maximum(
    *, partition: tuple[int, ...]
) -> None:
    """Blockwise hard max with carry publishes the maximum a single pass would."""
    expected, _identity, _carry = _dense_hard_max_with_carry(
        values=_values(), feasible=_feasible(), carry=_candidate_carry()
    )

    assert_agrees_to_ulp(
        got=np.asarray(_folded_hard_max_with_carry(partition=partition).best_value),
        expected=expected,
        n_ulp=0,
    )


@pytest.mark.parametrize("partition", _PARTITIONS, ids=("width-four", "width-three"))
def test_weighted_expectation_block_fold_accumulates_the_whole_weight_mass(
    *, partition: tuple[int, ...]
) -> None:
    """The published mass is the sum of every block's weights, to a few ULP."""
    assert_agrees_to_ulp(
        got=np.asarray(
            _folded_weighted_expectation(
                partition=partition, values=_values(), weights=_weights()
            ).weight_mass
        ),
        expected=np.asarray(_weights()).sum(),
        n_ulp=16,
    )


@pytest.mark.parametrize("partition", _PARTITIONS, ids=("width-four", "width-three"))
def test_weighted_expectation_is_invariant_to_rescaling_every_weight(
    *, partition: tuple[int, ...]
) -> None:
    """Scaling every weight by one factor leaves the expectation where it was."""
    values = _values()
    weights = _weights()

    assert_agrees_to_ulp(
        got=np.asarray(
            _folded_weighted_expectation(
                partition=partition, values=values, weights=weights * 4.0
            ).expectation
        ),
        expected=np.asarray(
            _folded_weighted_expectation(
                partition=partition, values=values, weights=weights
            ).expectation
        ),
        n_ulp=16,
    )


def test_weighted_expectation_publishes_nan_for_an_empty_weight_mass() -> None:
    """A lottery of exactly zero mass has no expectation and says so."""
    reduction = WEIGHTED_EXPECTATION_REDUCTION.bind(subnormal_is_accounted_for=False)
    accumulator = reduction.add(
        accumulator=reduction.initialize(value_template=jnp.zeros(_N_STATES)),
        values=_values(),
        weights=jnp.zeros(_N_ACTIONS, dtype=_values().dtype),
    )

    assert bool(
        jnp.all(jnp.isnan(reduction.finalize(accumulator=accumulator).expectation))
    )


def test_weighted_expectation_gives_a_zero_weight_infinity_no_contribution() -> None:
    """A node that cannot occur contributes exactly zero, even at `-inf`."""
    reduction = WEIGHTED_EXPECTATION_REDUCTION.bind(subnormal_is_accounted_for=False)
    values = jnp.asarray([[-jnp.inf, 1.0, 3.0]] * _N_STATES)
    weights = jnp.asarray([0.0, 0.5, 0.5], dtype=values.dtype)
    accumulator = reduction.add(
        accumulator=reduction.initialize(value_template=jnp.zeros(_N_STATES)),
        values=values,
        weights=weights,
    )

    np.testing.assert_array_equal(
        np.asarray(reduction.finalize(accumulator=accumulator).expectation),
        np.full(_N_STATES, 2.0),
    )
def test_hard_max_with_carry_block_fold_selects_the_dense_winner_identity(
    *, partition: tuple[int, ...]
) -> None:
    """The winning global identity is the one a single pass selects, exactly."""
    _value, expected, _carry = _dense_hard_max_with_carry(
        values=_values(), feasible=_feasible(), carry=_candidate_carry()
    )

    np.testing.assert_array_equal(
        np.asarray(_folded_hard_max_with_carry(partition=partition).best_candidate_id),
        expected,
    )


@pytest.mark.parametrize("partition", _PARTITIONS, ids=("width-four", "width-three"))
def test_hard_max_with_carry_block_fold_keeps_the_winners_payload(
    *, partition: tuple[int, ...]
) -> None:
    """The published carry is the winning candidate's own row, bit for bit."""
    _value, _identity, expected = _dense_hard_max_with_carry(
        values=_values(), feasible=_feasible(), carry=_candidate_carry()
    )

    np.testing.assert_array_equal(
        np.asarray(_folded_hard_max_with_carry(partition=partition).carry),
        expected,
    )


def test_hard_max_with_carry_publishes_the_empty_state_of_an_infeasible_cell() -> None:
    """A cell with no feasible candidate keeps value `-inf` at identity `-1`."""
    values = jnp.zeros((1, _N_ACTIONS))
    result = HARD_MAX_WITH_CARRY_REDUCTION.finalize(
        accumulator=HARD_MAX_WITH_CARRY_REDUCTION.add(
            accumulator=HARD_MAX_WITH_CARRY_REDUCTION.initialize(
                value_template=jnp.zeros(1)
            ),
            values=values,
            feasible=jnp.zeros(values.shape, dtype=bool),
            action_ids=_action_ids(),
        )
    )

    assert (float(result.best_value[0]), int(result.best_candidate_id[0])) == (
        -np.inf,
        -1,
    )


def test_hard_max_with_carry_rejects_a_non_int32_identity() -> None:
    """Candidate identities are exactly `int32`; another integer width is refused."""
    with pytest.raises(BeartypeCallHintViolation, match="int32"):
        HARD_MAX_WITH_CARRY_REDUCTION.add(
            accumulator=HARD_MAX_WITH_CARRY_REDUCTION.initialize(
                value_template=jnp.zeros(_N_STATES)
            ),
            values=_values(),
            feasible=_feasible(),
            action_ids=_action_ids().astype(jnp.int16),
        )
