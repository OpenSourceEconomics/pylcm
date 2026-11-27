import functools
from collections.abc import Mapping
from types import SimpleNamespace

import jax
import numpy as np
import pandas as pd
import pytest
from jax import numpy as jnp
from jax.experimental import topologies
from numpy.testing import assert_array_almost_equal as aaae
from pandas.testing import assert_frame_equal

from _lcm.config import TEST_DATA
from lcm.typing import FloatND
from lcm_examples import mortality as mortality_example
from lcm_examples import precautionary_savings as ps_example
from tests.conftest import X64_ENABLED
from tests.data.regression_tests import generate_representative_model_values as archive
from tests.data.regression_tests.generate_representative_model_values import (
    PROVENANCE,
    REPRESENTATIVE_MODELS,
    host_file,
    host_fingerprint,
    representative_outputs,
)
from tests.test_models.deterministic.regression import RegimeId, get_model, get_params

_PRECISION_DIR = TEST_DATA / "regression_tests" / ("f64" if X64_ENABLED else "f32")
_RNG_PRECISION_DIR = _PRECISION_DIR / "random-v1"


def test_regression_test() -> None:
    """Test that the output of lcm does not change."""
    # Load expected output
    expected_simulate = pd.read_pickle(
        TEST_DATA / "regression_tests" / "simulation.pkl"
    )
    expected_solve = pd.read_pickle(TEST_DATA / "regression_tests" / "solution.pkl")

    # Generate current lcm output
    n_periods = 4
    model = get_model(n_periods=n_periods)
    params = get_params(
        n_periods=n_periods,
        discount_factor=0.95,
        disutility_of_work=1.0,
        interest_rate=0.05,
    )

    got_solve: Mapping[int, Mapping[str, FloatND]] = model.solve(
        log_level="debug", params=params
    ).values
    got_simulate = model.simulate(
        log_level="debug",
        params=params,
        initial_conditions={
            "wealth": jnp.array([5.0, 20, 40, 70]),
            "age": jnp.array([18.0, 18.0, 18.0, 18.0]),
            "regime_id": jnp.array([RegimeId.working_life] * 4),
        },
    ).to_dataframe(terminal_rows="all")

    # Compare solution (iterate over expected regimes — got may have additional ones)
    for period in expected_solve:
        for regime in expected_solve[period]:
            aaae(expected_solve[period][regime], got_solve[period][regime], decimal=5)

    # Compare simulation (use tolerance to match solution comparison precision)
    assert_frame_equal(
        got_simulate,
        expected_simulate,
        check_dtype=False,
        atol=1e-5,
        check_column_type=False,
        check_categorical=False,
    )


def test_regression_precautionary_savings() -> None:
    """Test that precautionary savings benchmark model output does not change."""
    expected = pd.read_pickle(
        _RNG_PRECISION_DIR / "precautionary_savings_simulation.pkl"
    )

    model = ps_example.get_model(
        n_periods=5,
        shock_type="rouwenhorst",
        wealth_n_points=10,
        consumption_n_points=10,
    )
    params = ps_example.get_params(shock_type="rouwenhorst", sigma=0.2, rho=0.9)

    n_agents = 4
    got = model.simulate(
        params=params,
        initial_conditions={
            "age": jnp.full(n_agents, 20.0),
            "wealth": jnp.full(n_agents, 5.0),
            "income": jnp.full(n_agents, 0.0),
            "regime_id": jnp.zeros(n_agents, dtype=jnp.int32),
        },
        seed=12345,
        log_level="debug",
    ).to_dataframe(terminal_rows="all")

    assert_frame_equal(
        got,
        expected,
        check_dtype=False,
        atol=1e-5,
        check_column_type=False,
        check_categorical=False,
    )


def test_regression_mortality() -> None:
    """Test that mortality benchmark model output does not change."""
    expected = pd.read_pickle(_RNG_PRECISION_DIR / "mortality_simulation.pkl")

    n_periods = 4
    model = mortality_example.get_model(n_periods=n_periods)
    params = mortality_example.get_params(n_periods=n_periods)

    n_agents = 4
    got = model.simulate(
        params=params,
        initial_conditions={
            "age": jnp.full(n_agents, 40.0),
            "wealth": jnp.full(n_agents, 100.0),
            "regime_id": jnp.zeros(n_agents, dtype=jnp.int32),
        },
        seed=12345,
        log_level="debug",
    ).to_dataframe(terminal_rows="all")

    assert_frame_equal(
        got,
        expected,
        check_dtype=False,
        atol=1e-5,
        check_column_type=False,
        check_categorical=False,
    )


# Complete outputs of four representative models (fixed `params["edges"]` laws,
# gated edges) at a few small configurations: every solved value array and every
# simulated column, with their dtypes and shapes. The archive records the commit
# and precision it was produced at under `__provenance__/`; its module docstring
# is the command that regenerates it.
_STORED_MODEL_VALUES = _RNG_PRECISION_DIR / "representative_model_values.npz"
_TOLERANCE = 1e-10 if X64_ENABLED else 1e-5
_ON_ARCHIVE_HOST = (
    jax.default_backend() == "cpu"
    and host_file(_STORED_MODEL_VALUES).read_text().strip() == host_fingerprint()
)


@pytest.mark.parametrize("changed_input", ["jax", "jaxlib", "target", "math_flags"])
def test_stored_byte_fingerprint_identifies_the_compiler_target(
    *, monkeypatch: pytest.MonkeyPatch, changed_input: str
) -> None:
    """Different compiler versions, CPU targets or math flags cannot share bytes."""
    before = host_fingerprint()
    assert host_fingerprint() == before
    if changed_input == "jax":
        monkeypatch.setattr(jax, "__version__", "different-jax-version")
    elif changed_input == "jaxlib":
        monkeypatch.setattr(archive.jaxlib, "__version__", "different-jaxlib-version")
    elif changed_input == "target":
        monkeypatch.setattr(
            topologies,
            "get_attached_topology",
            lambda _platform: SimpleNamespace(
                serialize=lambda: b"different-cpu-target"
            ),
        )
    else:
        monkeypatch.setenv("XLA_FLAGS", "--xla_cpu_enable_fast_math=true")

    assert host_fingerprint() != before


@pytest.mark.parametrize("name", REPRESENTATIVE_MODELS)
def test_representative_outputs_have_exactly_the_stored_keys(name: str) -> None:
    """Each model produces exactly the stored value arrays and simulated columns."""
    assert set(_outputs(name)) == set(_stored_outputs(name))


@pytest.mark.parametrize("name", REPRESENTATIVE_MODELS)
def test_representative_outputs_have_the_stored_dtypes_and_shapes(name: str) -> None:
    """Each output array has the dtype and shape of its stored array."""
    assert _dtypes_and_shapes(_outputs(name)) == _dtypes_and_shapes(
        _stored_outputs(name)
    )


@pytest.mark.skipif(
    not _ON_ARCHIVE_HOST,
    reason="The stored bytes are the archive host's CPU results; elsewhere the float "
    "outputs are compared within tolerance.",
)
@pytest.mark.parametrize("name", REPRESENTATIVE_MODELS)
def test_representative_outputs_equal_their_stored_bytes(name: str) -> None:
    """Each output equals its stored array byte for byte.

    Signed zeros, NaN payloads and discrete labels included. The last bits of a
    float output depend on the instructions XLA's CPU backend emits, so the bytes
    are compared only on a host with the archive host's fingerprint.
    """
    assert _bytes(_outputs(name)) == _bytes(_stored_outputs(name))


@pytest.mark.parametrize("name", REPRESENTATIVE_MODELS)
def test_representative_float_outputs_match_their_stored_values(name: str) -> None:
    """Each model's value arrays and float columns agree with their stored values."""
    got = _outputs(name)
    expected = {
        key: value
        for key, value in _stored_outputs(name).items()
        if value.dtype.kind == "f"
    }
    np.testing.assert_allclose(
        np.concatenate([np.ravel(got[key]) for key in expected]),
        np.concatenate([np.ravel(value) for value in expected.values()]),
        rtol=_TOLERANCE,
        atol=_TOLERANCE,
    )


@pytest.mark.parametrize("name", REPRESENTATIVE_MODELS)
def test_representative_discrete_outputs_equal_their_stored_values(name: str) -> None:
    """Each model's labels and integer columns equal their stored values."""
    got = _outputs(name)
    expected = {
        key: value
        for key, value in _stored_outputs(name).items()
        if value.dtype.kind != "f"
    }
    assert {key: got[key].tolist() for key in expected} == {
        key: value.tolist() for key, value in expected.items()
    }


def test_stored_outputs_record_the_precision_they_are_compared_at() -> None:
    """The archive of each precision directory was produced at that precision."""
    with np.load(_STORED_MODEL_VALUES) as stored:
        assert str(stored[f"{PROVENANCE}/precision"]) == ("64" if X64_ENABLED else "32")


@functools.cache
def _outputs(name: str) -> dict[str, np.ndarray]:
    return representative_outputs(name)


def _stored_outputs(name: str) -> dict[str, np.ndarray]:
    """Return the stored arrays of one model, keyed as `representative_outputs`."""
    with np.load(_STORED_MODEL_VALUES) as stored:
        return {key: stored[key] for key in stored.files if key.startswith(f"{name}/")}


def _dtypes_and_shapes(
    arrays: Mapping[str, np.ndarray],
) -> dict[str, tuple[str, tuple[int, ...]]]:
    return {key: (value.dtype.str, value.shape) for key, value in arrays.items()}


def _bytes(arrays: Mapping[str, np.ndarray]) -> dict[str, bytes]:
    return {key: value.tobytes(order="C") for key, value in arrays.items()}
