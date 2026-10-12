"""Independent taste draws follow each original row's scalar key.

The expected actions come from a scalar reference that rebuilds the documented
taste-stream address with `hashlib`/`json`/`struct` and draws one centered
Gumbel pair per addressed key, outside any `jax.vmap`. It imports no pylcm
helper, so it shares neither the production address code nor the subject
transform that batches the draws.

The model has two actions with finite utilities `Q = (0, bias)` and no
continuation. For the RBG implementations, `bias` is chosen at the midpoint of a
row's scalar and dependency-batched noise gaps, so a batched draw for that row
selects the opposite action from the scalar one.
"""

import functools
import hashlib
import json
import struct
from fractions import Fraction

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from lcm import (
    AgeGrid,
    DiscreteGrid,
    ExecutionConfig,
    LinSpacedGrid,
    Model,
    Regime,
    categorical,
)
from lcm.taste_shocks import ExtremeValueTasteShocks
from lcm.typing import ContinuousState, DiscreteAction, FloatND, ScalarInt, UserSeed

_N_SUBJECTS = 7
_FOLD_CHILD = 5
_SOURCE_SEEDS = (17, 0, 1, 721)
_WORK_DOMAIN = (("work", (("off", 0), ("on", 1))),)


@categorical(ordered=False)
class _RegimeId:
    alive: ScalarInt
    done: ScalarInt


@categorical(ordered=False)
class _Work:
    off: ScalarInt
    on: ScalarInt


def _utility(*, wealth: ContinuousState, work: DiscreteAction, bias: float) -> FloatND:
    return 0.0 * wealth + bias * work


def _terminal_utility(*, wealth: ContinuousState) -> FloatND:
    return 0.0 * wealth


def _next_wealth(*, wealth: ContinuousState) -> ContinuousState:
    return wealth


def _scalar_address_keys(*, root: jax.Array) -> list[jax.Array]:
    """Return each original row's addressed key at age 0, one scalar fold at a time."""
    age = Fraction(0)
    payload = ("pylcm.taste-stream.v1", (age.numerator, age.denominator), _WORK_DOMAIN)
    encoded = json.dumps(
        payload, ensure_ascii=False, allow_nan=False, separators=(",", ":")
    ).encode("utf-8")
    addressed = root
    for word in struct.unpack(">8I", hashlib.sha256(encoded).digest()):
        addressed = jax.random.fold_in(addressed, np.uint32(word))
    return [
        jax.random.fold_in(
            jax.random.fold_in(addressed, np.uint32(row // 2**32)),
            np.uint32(row % 2**32),
        )
        for row in range(_N_SUBJECTS)
    ]


def _centered_gumbel_pair(key: jax.Array) -> jax.Array:
    return jax.random.gumbel(key, (2,)) - np.euler_gamma


def _scalar_choices(*, root: jax.Array, bias: float) -> np.ndarray:
    """Choose `argmax(Q + noise)` per row with the row's own scalar draw."""
    scores = jnp.asarray([0.0, bias])
    return np.asarray(
        [
            int(np.argmax(np.asarray(_centered_gumbel_pair(key) + scores)))
            for key in _scalar_address_keys(root=root)
        ]
    )


@functools.cache
def _calibration(impl: str) -> tuple[jax.Array, float, int]:
    """Return a folded root, a utility bias and the row it discriminates.

    For Threefry there is nothing to discriminate and the bias is zero; the row
    is reported as `-1`. For the RBG implementations, the bias sits halfway
    between a row's scalar and dependency-batched noise gaps, kept well away
    from a rounded tie.
    """
    if impl == "threefry2x32":
        root = jax.random.fold_in(jax.random.key(17, impl=impl), _FOLD_CHILD)
        return root, 0.0, -1
    for seed in _SOURCE_SEEDS:
        root = jax.random.fold_in(jax.random.key(seed, impl=impl), _FOLD_CHILD)
        keys = _scalar_address_keys(root=root)
        scalar = np.stack([np.asarray(_centered_gumbel_pair(key)) for key in keys])
        batched = np.asarray(jax.jit(jax.vmap(_centered_gumbel_pair))(jnp.stack(keys)))
        scalar_gap = scalar[:, 1] - scalar[:, 0]
        batched_gap = batched[:, 1] - batched[:, 0]
        row = int(np.argmax(np.abs(scalar_gap - batched_gap)))
        bias = np.asarray(-(scalar_gap[row] + batched_gap[row]) / 2, scalar.dtype)
        scores = np.asarray([0.0, bias], dtype=scalar.dtype)
        intended = scalar[row] + scores
        accidental = batched[row] + scores
        margin = min(abs(np.diff(intended)[0]), abs(np.diff(accidental)[0]))
        magnitude = max(1.0, *np.abs(intended), *np.abs(accidental))
        separated = margin > 64 * np.finfo(scalar.dtype).eps * magnitude
        if separated and np.argmax(intended) != np.argmax(accidental):
            return root, float(bias), row
    msg = f"No bounded {impl} root separates scalar and batched choices."
    raise AssertionError(msg)


def _public_choices(
    *,
    root: UserSeed,
    width: int,
    budget: int | None,
    bias: float,
    auto_solve: bool = False,
    enable_jit: bool = True,
) -> np.ndarray:
    """Return period-0 work codes in original subject order."""
    model = Model(
        regimes={
            "alive": Regime(
                functions={"utility": _utility},
                actions={"work": DiscreteGrid(category_class=_Work)},
                taste_shocks=ExtremeValueTasteShocks(),
            ),
            "done": Regime(functions={"utility": _terminal_utility}),
        },
        states={"wealth": LinSpacedGrid(start=1, stop=2, n_points=2)},
        state_transitions={"wealth": _next_wealth},
        ages=AgeGrid(start=0, inclusive_stop=1, step="Y"),
        regime_id_class=_RegimeId,
        execution_config=ExecutionConfig(
            axis_widths={"subject": width}, device_memory_bytes=budget
        ),
        initial_nodes={0: "alive"},
        edges={"alive": {"done": 0}},
        enable_jit=enable_jit,
    )
    params = {
        "alive": {
            "utility": {"bias": bias},
            "koopmans_aggregator": {"discount_factor": 0.0},
            "taste_shocks": {"scale": 1.0},
        }
    }
    solution = None if auto_solve else model.solve(params=params, log_level="off")
    result = model.simulate(
        params=params,
        initial_conditions={
            "wealth": jnp.full(_N_SUBJECTS, 1.5),
            "age": jnp.zeros(_N_SUBJECTS),
            "regime_id": jnp.full(_N_SUBJECTS, _RegimeId.alive, dtype=jnp.int32),
        },
        solution=solution,
        seed=11,
        taste_shock_seed=root,
        log_level="off",
    ).to_dataframe(use_labels=False)
    period_zero = result.query("period == 0").sort_values("subject_id")
    return period_zero["work"].to_numpy()


@pytest.mark.parametrize("impl", ["rbg", "unsafe_rbg"])
def test_calibration_separates_scalar_and_batched_choices(impl: str) -> None:
    """The bias makes a batched draw choose against the scalar draw on one row."""
    _, _, row = _calibration(impl)
    assert 0 <= row < _N_SUBJECTS


@pytest.mark.parametrize("width", [1, 3, 7])
@pytest.mark.parametrize("budget", [None, 2**30])
@pytest.mark.parametrize("impl", ["threefry2x32", "rbg", "unsafe_rbg"])
def test_independent_taste_choices_match_scalar_draws(
    *, impl: str, budget: int | None, width: int
) -> None:
    """Every subject width, padded or not, chooses with each row's scalar draw."""
    root, bias, _ = _calibration(impl)
    actual = _public_choices(root=root, width=width, budget=budget, bias=bias)
    np.testing.assert_array_equal(actual, _scalar_choices(root=root, bias=bias))


def test_independent_rbg_choices_match_scalar_draws_with_automatic_solve() -> None:
    """Simulating without a supplied solution draws the same scalar shocks."""
    root, bias, _ = _calibration("rbg")
    actual = _public_choices(
        root=root, width=7, budget=2**30, bias=bias, auto_solve=True
    )
    np.testing.assert_array_equal(actual, _scalar_choices(root=root, bias=bias))


def test_independent_rbg_choices_match_scalar_draws_without_model_jit() -> None:
    """A model built without JIT draws the same scalar shocks."""
    root, bias, _ = _calibration("rbg")
    actual = _public_choices(
        root=root, width=7, budget=None, bias=bias, enable_jit=False
    )
    np.testing.assert_array_equal(actual, _scalar_choices(root=root, bias=bias))


def test_integer_taste_seed_keeps_threefry_under_rbg_default() -> None:
    """An integer taste seed draws from a Threefry root whatever the ambient default."""
    with jax.default_prng_impl("rbg"):
        actual = _public_choices(root=17, width=3, budget=None, bias=0.0)
    expected_root = jax.random.key(17, impl="threefry2x32")
    np.testing.assert_array_equal(actual, _scalar_choices(root=expected_root, bias=0.0))
