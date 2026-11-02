import json
import math
import os
import shutil
import subprocess
import textwrap
from fractions import Fraction
from functools import partial
from itertools import pairwise
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import pytest
from jax import jacrev, jit, vmap
from jax import numpy as jnp
from numpy.testing import assert_array_almost_equal as aaae
from pandas.testing import assert_frame_equal
from quantecon.markov.approximation import rouwenhorst as qe_rouwenhorst
from quantecon.markov.approximation import tauchen as qe_tauchen
from scipy.integrate import quad
from scipy.special import ndtr
from scipy.stats import norm
from typing_extensions import TypedDict

from _lcm.config import TEST_DATA
from _lcm.processes.ar1 import _AR1Process
from _lcm.processes.state_conditioned import conditioned_row
from _lcm.typing import DataclassInstance
from _lcm.utils.logging import LogLevel
from lcm import (
    AgeGrid,
    InitialNodes,
    LogNormalIIDProcess,
    Model,
    NormalIIDProcess,
    NormalMixtureIIDProcess,
    Regime,
    RouwenhorstAR1Process,
    TauchenAR1Process,
    TauchenNormalMixtureAR1Process,
    UniformIIDProcess,
    categorical,
)
from lcm.exceptions import GridInitializationError
from lcm.execution import ExecutionConfig
from lcm.solver_api import SolutionResult
from lcm.typing import (
    ContinuousState,
    Float1D,
    FloatND,
    Int1D,
    ScalarFloat,
    ScalarInt,
    UserParams,
)
from tests.conftest import DECIMAL_PRECISION, X64_ENABLED
from tests.test_models.processes import (
    MultiRegimeId,
    RegimeId,
    get_model,
    get_multi_regime_model,
    get_multi_regime_params,
    get_params,
)

# Grid centering is float-eps-limited at the active precision.
_CENTERING_DECIMAL = 10 if X64_ENABLED else 5


@pytest.mark.skipif(not X64_ENABLED, reason="Not working with 32-Bit because of RNG")
@pytest.mark.parametrize(
    "distribution_type", ["uniform", "normal", "lognormal", "tauchen", "rouwenhorst"]
)
def test_model_with_process(distribution_type):
    model = get_model(n_periods=4, distribution_type=distribution_type)
    params = get_params(distribution_type)

    got_solve = model.solve(
        log_level="debug",
        params=params,
    )

    got_simulate = model.simulate(
        log_level="debug",
        params=params,
        initial_conditions={
            "health": jnp.asarray([0, 0], dtype=jnp.int32),
            "income": jnp.asarray([0.0, 0.0]),
            "wealth": jnp.asarray([1.0, 1.0]),
            "age": jnp.asarray([0.0, 0.0]),
            "regime_id": jnp.array([RegimeId.alive] * 2),
        },
        solution=got_solve,
        seed=42,
    ).to_dataframe()

    expected_simulate = pd.read_pickle(
        TEST_DATA / "processes" / "random-v1" / f"simulation_{distribution_type}.pkl"
    )
    expected_solve = pd.read_pickle(
        TEST_DATA / "processes" / f"solution_{distribution_type}.pkl"
    )
    # Compare solution (iterate over expected regimes — got may have additional ones)
    for period in expected_solve:
        for regime in expected_solve[period]:
            aaae(
                expected_solve[period][regime],
                got_solve.values[period][regime],
                decimal=5,
            )

    # Compare simulation (use tolerance to match solution comparison precision)
    assert_frame_equal(
        got_simulate,
        expected_simulate,
        check_dtype=False,
        atol=1e-5,
        check_column_type=False,
        check_categorical=False,
    )


@pytest.mark.skipif(not X64_ENABLED, reason="Not working with 32-Bit because of RNG")
@pytest.mark.parametrize(
    "distribution_type", ["uniform", "normal", "lognormal", "tauchen", "rouwenhorst"]
)
def test_model_with_cross_regime_processes(distribution_type: str) -> None:
    """Verify cross-regime transitions work when both regimes have process grids.

    Reproducer for a crash where process-state `lambda: None` stubs leak from the
    source regime into cross-regime transition dicts, causing a `vmap` error in
    `joint_weights_from_marginals` (receives `None` instead of arrays).
    """
    model = get_multi_regime_model(
        n_periods=6,
        distribution_type=distribution_type,  # ty: ignore[invalid-argument-type]
    )
    params = get_multi_regime_params(distribution_type)  # ty: ignore[invalid-argument-type]

    result = model.simulate(
        log_level="debug",
        params=params,
        initial_conditions={
            "health": jnp.zeros(2, dtype=jnp.int32),
            "income": jnp.zeros(2),
            "wealth": jnp.ones(2),
            "age": jnp.zeros(2),
            "regime_id": jnp.full(2, MultiRegimeId.work, dtype=jnp.int32),
        },
        seed=42,
    ).to_dataframe()
    assert set(result["regime_name"]) >= {"work", "retire"}


def test_multi_regime_reached_states_afford_the_lowest_consumption_node() -> None:
    """Income counted as spendable is carried forward, so no living row is stranded."""
    model = get_multi_regime_model(n_periods=6, distribution_type="normal")
    result = model.simulate(
        log_level="off",
        params=get_multi_regime_params("normal"),
        initial_conditions={
            "health": jnp.array([0, 1, 0, 1, 0, 1, 0], dtype=jnp.int32),
            "income": jnp.array([0.0, 0.5, -0.3, 0.2, 0.1, -0.1, 0.4]),
            "wealth": jnp.array([1.0, 2.0, 3.0, 1.5, 2.5, 4.0, 1.2]),
            "age": jnp.zeros(7),
            "regime_id": jnp.full(7, MultiRegimeId.work, dtype=jnp.int32),
        },
        seed=42,
    ).to_dataframe()
    living = result.loc[result["regime_name"].isin(["work", "retire"])]

    assert (living["wealth"] + np.exp(living["income"]) >= 0.1).all()


_GRID_CLASSES_WITH_GH_KWARG = [
    (UniformIIDProcess, {}),
    (NormalIIDProcess, {"gauss_hermite": True}),
    (LogNormalIIDProcess, {"gauss_hermite": True}),
    (NormalMixtureIIDProcess, {}),
    (TauchenAR1Process, {"gauss_hermite": True}),
    (RouwenhorstAR1Process, {}),
    (TauchenNormalMixtureAR1Process, {}),
]


@pytest.mark.parametrize(("grid_cls", "extra_kw"), _GRID_CLASSES_WITH_GH_KWARG)
def test_process_grid_correct_shape_without_params(*, grid_cls, extra_kw):
    """A process grid without params returns correct-shape arrays."""
    grid = grid_cls(n_points=3, **extra_kw)
    assert not grid.is_fully_specified
    assert grid.params_to_pass_at_runtime
    assert grid.get_gridpoints().shape == (3,)
    assert grid.get_transition_probs().shape == (3, 3)
    assert jnp.all(jnp.isnan(grid.get_gridpoints()))
    assert jnp.all(jnp.isnan(grid.get_transition_probs()))


@pytest.mark.parametrize(
    ("grid_cls", "kwargs"),
    [
        (UniformIIDProcess, {"start": 0.0, "stop": 1.0}),
        (
            NormalIIDProcess,
            {"gauss_hermite": True, "mu": 0.0, "sigma": 1.0},
        ),
        (
            NormalIIDProcess,
            {"gauss_hermite": False, "mu": 0.0, "sigma": 1.0, "n_std": 3.0},
        ),
        (
            LogNormalIIDProcess,
            {"gauss_hermite": True, "mu": 0.0, "sigma": 1.0},
        ),
        (
            LogNormalIIDProcess,
            {"gauss_hermite": False, "mu": 0.0, "sigma": 1.0, "n_std": 3.0},
        ),
        (
            TauchenAR1Process,
            {"gauss_hermite": True, "rho": 0.9, "sigma": 1.0, "mu": 0.0},
        ),
        (
            TauchenAR1Process,
            {"gauss_hermite": False, "rho": 0.9, "sigma": 1.0, "mu": 0.0, "n_std": 2},
        ),
        (
            NormalMixtureIIDProcess,
            {
                "n_std": 3.0,
                "p1": 0.9,
                "mu1": 0.0,
                "sigma1": 0.1,
                "mu2": -0.5,
                "sigma2": 0.3,
            },
        ),
        (RouwenhorstAR1Process, {"rho": 0.9, "sigma": 1.0, "mu": 0.0}),
        (
            TauchenNormalMixtureAR1Process,
            {
                "rho": 0.9,
                "mu": 1.0,
                "n_std": 3.0,
                "p1": 0.9,
                "mu1": 0.0,
                "sigma1": 0.1,
                "mu2": -0.5,
                "sigma2": 0.3,
            },
        ),
    ],
)
def test_process_grid_fully_specified_with_all_params(*, grid_cls, kwargs):
    """A process grid with all params provided is fully specified."""
    grid = grid_cls(n_points=3, **kwargs)
    assert grid.is_fully_specified
    result = grid.get_gridpoints()
    assert result.shape == (3,)


type _AR1GridClass = type[
    TauchenAR1Process | RouwenhorstAR1Process | TauchenNormalMixtureAR1Process
]

_AR1_GRID_CLASSES = [TauchenAR1Process, RouwenhorstAR1Process]

_STATIONARY_GRID_CLASSES: list[_AR1GridClass] = [
    TauchenAR1Process,
    RouwenhorstAR1Process,
    TauchenNormalMixtureAR1Process,
]


def _make_ar1_grid(
    *, grid_cls: _AR1GridClass, rho: float
) -> TauchenAR1Process | RouwenhorstAR1Process | TauchenNormalMixtureAR1Process:
    if grid_cls is TauchenAR1Process:
        return TauchenAR1Process(
            n_points=5, rho=rho, sigma=0.5, mu=0.0, gauss_hermite=True
        )
    if grid_cls is RouwenhorstAR1Process:
        return RouwenhorstAR1Process(n_points=5, rho=rho, sigma=0.5, mu=0.0)
    return TauchenNormalMixtureAR1Process(
        n_points=5,
        rho=rho,
        mu=0.0,
        n_std=3.0,
        p1=0.7,
        mu1=0.0,
        sigma1=0.2,
        mu2=0.1,
        sigma2=0.3,
    )


@pytest.mark.parametrize("grid_cls", _STATIONARY_GRID_CLASSES)
@pytest.mark.parametrize("rho", [1.0, 1.2, -1.0, -1.2])
def test_ar1_rejects_nonstationary_persistence(
    *, grid_cls: _AR1GridClass, rho: float
) -> None:
    """A stationary grid cannot represent a unit-root or explosive process."""
    with pytest.raises(
        GridInitializationError,
        match=rf"{grid_cls.__name__}.*rho={rho}.*no stationary distribution",
    ):
        _make_ar1_grid(grid_cls=grid_cls, rho=rho)


@pytest.mark.parametrize("grid_cls", _STATIONARY_GRID_CLASSES)
@pytest.mark.parametrize("rho", [-0.9, 0.0, 0.9])
def test_ar1_stationary_nodes_are_finite(
    *, grid_cls: _AR1GridClass, rho: float
) -> None:
    """Stationary persistence gives finite grid nodes."""
    grid = _make_ar1_grid(grid_cls=grid_cls, rho=rho)
    assert jnp.isfinite(grid.get_gridpoints()).all()


@pytest.mark.parametrize("distribution_type", ["tauchen", "rouwenhorst"])
@pytest.mark.parametrize("rho", [1.0, 1.2, -1.0, -1.2])
def test_ar1_solve_rejects_nonstationary_runtime_persistence(
    *, distribution_type: Literal["tauchen", "rouwenhorst"], rho: float
) -> None:
    """Runtime persistence is checked before a solve places its process grid."""
    model = get_model(n_periods=2, distribution_type=distribution_type)
    params = get_params(distribution_type)
    params["alive"]["income"] = {**params["alive"]["income"], "rho": rho}
    with pytest.raises(GridInitializationError, match="no stationary distribution"):
        model.solve(params=params, log_level="off")


type _Ar1Kind = Literal["tauchen-gh", "tauchen-linspace", "rouwenhorst", "mixture"]
_AR1_KINDS: tuple[_Ar1Kind, ...] = (
    "tauchen-gh",
    "tauchen-linspace",
    "rouwenhorst",
    "mixture",
)


@categorical(ordered=False)
class _ActiveOnlyId:
    active: ScalarInt


@categorical(ordered=False)
class _ActiveDormantId:
    active: ScalarInt
    dormant: ScalarInt


def _unit_utility() -> ScalarFloat:
    return jnp.asarray(1.0)


def _income_utility(income: ContinuousState) -> FloatND:
    return income


def _ar1_grid_without_rho(
    grid_kind: _Ar1Kind,
) -> TauchenAR1Process | RouwenhorstAR1Process | TauchenNormalMixtureAR1Process:
    if grid_kind == "mixture":
        return TauchenNormalMixtureAR1Process(
            n_points=3,
            mu=0.0,
            n_std=3.0,
            p1=0.7,
            mu1=0.0,
            sigma1=0.2,
            mu2=0.1,
            sigma2=0.3,
        )
    if grid_kind == "rouwenhorst":
        return RouwenhorstAR1Process(n_points=3, sigma=0.2, mu=0.0)
    if grid_kind == "tauchen-gh":
        return TauchenAR1Process(n_points=3, gauss_hermite=True, sigma=0.2, mu=0.0)
    return TauchenAR1Process(
        n_points=3, gauss_hermite=False, n_std=3.0, sigma=0.2, mu=0.0
    )


def _terminal_model(
    *,
    regimes: dict[str, Regime],
    regime_id_class: type[DataclassInstance],
    enable_jit: bool,
) -> Model:
    return Model(
        regimes=regimes,
        edges={},
        n_periods=1,
        initial_nodes=InitialNodes(by_period={0: "active"}),
        regime_id_class=regime_id_class,
        enable_jit=enable_jit,
        durable_identity=False,
    )


def _active_subject() -> dict[str, Int1D]:
    return {
        "period": jnp.array([0], dtype=jnp.int32),
        "regime_id": jnp.full(1, _ActiveOnlyId.active, dtype=jnp.int32),
    }


def _float_dtype() -> np.dtype:
    return np.dtype(np.float64 if X64_ENABLED else np.float32)


@pytest.mark.parametrize("grid_kind", _AR1_KINDS)
@pytest.mark.parametrize("enable_jit", [False, True], ids=["eager", "jit"])
@pytest.mark.parametrize("entry", ["solve", "simulate", "simulate-with-solution"])
def test_unreachable_regime_with_runtime_rho_needs_no_parameters(
    *, grid_kind: _Ar1Kind, enable_jit: bool, entry: str
) -> None:
    """A regime no subject can enter imposes no runtime-rho parameter.

    The active terminal cell has constant utility 1, so its value and its single
    simulated row are known exactly, independent of any model machinery.
    """
    model = _terminal_model(
        regimes={
            "active": Regime(functions={"utility": _unit_utility}),
            "dormant": Regime(
                states={"income": _ar1_grid_without_rho(grid_kind)},
                functions={"utility": _income_utility},
            ),
        },
        regime_id_class=_ActiveDormantId,
        enable_jit=enable_jit,
    )
    assert model.get_params_template()["dormant"] == {}
    expected = np.asarray([1.0], dtype=_float_dtype())
    if entry == "solve":
        solution = model.solve(params={}, log_level="off")
        got = np.asarray(solution.values[0]["active"]).reshape(1)
    else:
        frame = model.simulate(
            params={},
            initial_conditions=_active_subject(),
            solution=(
                model.solve(params={}, log_level="off")
                if entry == "simulate-with-solution"
                else None
            ),
            log_level="off",
            seed=0,
        ).to_dataframe()
        assert frame["regime_name"].tolist() == ["active"]
        got = frame["value"].to_numpy()
    assert got.tobytes() == expected.tobytes()


@pytest.mark.parametrize("grid_kind", _AR1_KINDS)
@pytest.mark.parametrize("rho", [1.0, -1.0, 1.2, -1.2])
@pytest.mark.parametrize("entry", ["solve", "simulate", "simulate-with-solution"])
def test_active_runtime_rho_rejected_before_grid_realization(
    *, grid_kind: _Ar1Kind, rho: float, entry: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every public entry rejects nonstationary rho before computing process nodes."""
    grid = _ar1_grid_without_rho(grid_kind)
    model = _terminal_model(
        regimes={
            "active": Regime(
                states={"income": grid}, functions={"utility": _income_utility}
            )
        },
        regime_id_class=_ActiveOnlyId,
        enable_jit=False,
    )
    solution = (
        model.solve(params={"active": {"income": {"rho": 0.5}}}, log_level="off")
        if entry == "simulate-with-solution"
        else None
    )

    monkeypatch.setattr(type(grid), "compute_gridpoints", _fail_if_nodes_realized)
    with pytest.raises(GridInitializationError, match="no stationary distribution"):
        _call_entry(
            model=model,
            entry=entry,
            params={"active": {"income": {"rho": rho}}},
            solution=solution,
        )


def _fail_if_nodes_realized(
    self: _AR1Process,  # noqa: ARG001
    **kwargs: ScalarFloat | ScalarInt,  # noqa: ARG001
) -> Float1D:
    raise AssertionError("process nodes realized before rho was validated")


def _call_entry(
    *, model: Model, entry: str, params: UserParams, solution: SolutionResult | None
) -> None:
    if entry == "solve":
        model.solve(params=params, log_level="off")
    else:
        model.simulate(
            params=params,
            initial_conditions={**_active_subject(), "income": jnp.array([0.25])},
            solution=solution,
            log_level="off",
            seed=0,
        )


@pytest.mark.parametrize("grid_cls", _AR1_GRID_CLASSES)
def test_ar1_grid_centers_on_unconditional_mean(grid_cls):
    """Midpoint of AR(1) gridpoints is approximately mu / (1 - rho)."""
    mu, rho = 2.0, 0.8
    kwargs: dict[str, float] = {"rho": rho, "sigma": 0.5, "mu": mu}
    if grid_cls is TauchenAR1Process:
        kwargs["gauss_hermite"] = True
    grid = grid_cls(n_points=11, **kwargs)
    points = grid.get_gridpoints()
    midpoint = (points[0] + points[-1]) / 2
    expected = mu / (1 - rho)
    aaae(midpoint, expected, decimal=_CENTERING_DECIMAL)


@pytest.mark.parametrize("grid_cls", _AR1_GRID_CLASSES)
def test_ar1_transition_probs_rows_sum_to_one(grid_cls):
    """Each row of the transition matrix sums to 1."""
    kwargs: dict[str, float] = {"rho": 0.9, "sigma": 0.5, "mu": 1.0}
    if grid_cls is TauchenAR1Process:
        kwargs["gauss_hermite"] = True
    grid = grid_cls(n_points=7, **kwargs)
    P = grid.get_transition_probs()
    row_sums = P.sum(axis=1)
    aaae(row_sums, jnp.ones(7), decimal=DECIMAL_PRECISION)


@pytest.mark.parametrize(
    "grid_cls",
    [
        NormalIIDProcess,
        LogNormalIIDProcess,
        TauchenAR1Process,
    ],
    ids=lambda c: c.__name__,
)
@pytest.mark.parametrize("n_points", [2, 4, 10])
@pytest.mark.parametrize("compiled", [False, True])
def test_even_n_points_accepted_for_gauss_hermite(
    *,
    grid_cls: type[NormalIIDProcess | LogNormalIIDProcess | TauchenAR1Process],
    n_points: int,
    compiled: bool,
) -> None:
    """Even Gauss-Hermite grids preserve normal nodes and normalized probabilities."""
    mu, sigma = 0.7, 0.4
    mean, std = mu, sigma
    if grid_cls is TauchenAR1Process:
        rho = 0.6
        grid = TauchenAR1Process(
            n_points=n_points, gauss_hermite=True, mu=mu, sigma=sigma, rho=rho
        )
        mean, std = mu / (1 - rho), sigma / np.sqrt(1 - rho**2)
    else:
        grid = grid_cls(n_points=n_points, gauss_hermite=True, mu=mu, sigma=sigma)
    raw_nodes, raw_weights = np.polynomial.hermite_e.hermegauss(n_points)
    expected = mean + std * raw_nodes
    get_nodes = jit(grid.compute_gridpoints) if compiled else grid.compute_gridpoints
    get_probabilities = (
        jit(grid.compute_transition_probs)
        if compiled
        else grid.compute_transition_probs
    )
    got = get_nodes(**grid.params)
    if grid_cls is LogNormalIIDProcess:
        got = jnp.log(got)
    aaae(got, expected, decimal=DECIMAL_PRECISION)
    assert np.all(np.abs(np.asarray(got) - mean) > std * 0.1)
    probabilities = get_probabilities(**grid.params)
    aaae(probabilities.sum(axis=1), np.ones(n_points), decimal=DECIMAL_PRECISION)
    assert bool((probabilities >= 0).all())
    if grid_cls is not TauchenAR1Process:
        expected_weights = raw_weights / np.sqrt(2 * np.pi)
        aaae(
            probabilities,
            np.tile(expected_weights, (n_points, 1)),
            decimal=DECIMAL_PRECISION,
        )
        aaae(probabilities[0] @ got**2, mu**2 + sigma**2, decimal=DECIMAL_PRECISION)
    else:
        boundaries = np.concatenate(
            ([-np.inf], (expected[:-1] + expected[1:]) / 2, [np.inf])
        )
        conditional_mean = mu + rho * expected
        expected_probabilities = np.diff(
            ndtr((boundaries[None, :] - conditional_mean[:, None]) / sigma), axis=1
        )
        aaae(probabilities, expected_probabilities, decimal=DECIMAL_PRECISION)


@pytest.mark.parametrize(
    "grid_cls", [NormalIIDProcess, LogNormalIIDProcess, TauchenAR1Process]
)
@pytest.mark.parametrize("n_points", [0, -1, -2])
def test_gauss_hermite_rejects_nonpositive_node_counts(
    *,
    grid_cls: type[NormalIIDProcess | LogNormalIIDProcess | TauchenAR1Process],
    n_points: int,
) -> None:
    """A Gauss-Hermite support requires a positive node count at construction."""
    with pytest.raises(GridInitializationError, match="n_points must be positive"):
        grid_cls(n_points=n_points, gauss_hermite=True)


@pytest.mark.parametrize(
    "grid_cls_and_kwargs",
    [
        (NormalIIDProcess, {"gauss_hermite": False, "n_std": 3.0}),
        (LogNormalIIDProcess, {"gauss_hermite": False, "n_std": 3.0}),
        (TauchenAR1Process, {"gauss_hermite": False, "n_std": 3.0}),
        (UniformIIDProcess, {}),
        (NormalMixtureIIDProcess, {}),
        (RouwenhorstAR1Process, {}),
        (TauchenNormalMixtureAR1Process, {}),
    ],
)
def test_even_n_points_accepted_for_non_gauss_hermite(grid_cls_and_kwargs):
    """Non-GH grids accept even n_points (linspace grids work fine without a node at
    the mean)."""
    grid_cls, extra_kw = grid_cls_and_kwargs
    grid_cls(n_points=4, **extra_kw)  # should not raise


@pytest.mark.parametrize(
    "grid_cls",
    [
        NormalIIDProcess,
        LogNormalIIDProcess,
        TauchenAR1Process,
    ],
)
def test_gauss_hermite_and_n_std_mutual_exclusion(grid_cls):
    """gauss_hermite=True and n_std are mutually exclusive."""
    with pytest.raises(GridInitializationError, match="mutually exclusive"):
        grid_cls(n_points=3, gauss_hermite=True, n_std=2.0)


def test_gauss_hermite_required():
    """Normal(n_points=5) without gauss_hermite raises TypeError."""
    with pytest.raises(TypeError):
        NormalIIDProcess(n_points=5)  # ty: ignore[missing-argument]


def test_normal_gauss_hermite_weights_sum_to_one():
    """GH weights for iid Normal sum to 1."""
    grid = NormalIIDProcess(n_points=7, gauss_hermite=True, mu=0.0, sigma=1.0)
    P = grid.get_transition_probs()
    aaae(P[0].sum(), 1.0, decimal=DECIMAL_PRECISION)


def test_normal_linspace_transition_probs_rows_sum_to_one():
    """Non-GH Normal iid transition probability rows sum to 1."""
    grid = NormalIIDProcess(
        n_points=7, gauss_hermite=False, mu=0.0, sigma=1.0, n_std=3.0
    )
    P = grid.get_transition_probs()
    row_sums = P.sum(axis=1)
    aaae(row_sums, jnp.ones(7), decimal=DECIMAL_PRECISION)


def test_lognormal_linspace_transition_probs_rows_sum_to_one():
    """Non-GH LogNormal iid transition probability rows sum to 1."""
    grid = LogNormalIIDProcess(
        n_points=7, gauss_hermite=False, mu=0.0, sigma=1.0, n_std=3.0
    )
    P = grid.get_transition_probs()
    row_sums = P.sum(axis=1)
    aaae(row_sums, jnp.ones(7), decimal=DECIMAL_PRECISION)


def test_normal_gauss_hermite_n_std_not_in_params():
    """n_std is excluded from params_to_pass_at_runtime when gauss_hermite=True."""
    grid = NormalIIDProcess(n_points=5, gauss_hermite=True)
    assert "n_std" not in grid.params_to_pass_at_runtime


@pytest.mark.parametrize(
    "grid",
    [
        NormalIIDProcess(n_points=5, gauss_hermite=True, mu=0.0, sigma=1.0),
        LogNormalIIDProcess(n_points=5, gauss_hermite=True, mu=0.0, sigma=1.0),
        UniformIIDProcess(n_points=5, start=0.0, stop=1.0),
        TauchenAR1Process(n_points=5, gauss_hermite=True, rho=0.9, sigma=0.5, mu=0.0),
    ],
    ids=["normal", "lognormal", "uniform", "tauchen"],
)
def test_process_grid_params_excludes_distributed(grid):
    """`distributed` is a placement flag, not a distribution parameter.

    `ContinuousGrid` exposes `distributed` as a field, but it must not leak
    into `params` (the mapping of distribution-specific parameter names to
    their values used by `compute_gridpoints`).
    """
    assert "distributed" not in grid.params
    assert "distributed" not in grid.params_to_pass_at_runtime


def test_tauchen_gauss_hermite_transition_probs_rows_sum_to_one():
    """Each row of the GH Tauchen transition matrix sums to 1."""
    grid = TauchenAR1Process(n_points=7, gauss_hermite=True, rho=0.9, sigma=0.5, mu=1.0)
    P = grid.get_transition_probs()
    row_sums = P.sum(axis=1)
    aaae(row_sums, jnp.ones(7), decimal=DECIMAL_PRECISION)


def test_tauchen_linspace_transition_probs_rows_sum_to_one():
    """Each row of the non-GH Tauchen transition matrix sums to 1."""
    grid = TauchenAR1Process(
        n_points=7, gauss_hermite=False, rho=0.9, sigma=0.5, mu=1.0, n_std=3.0
    )
    P = grid.get_transition_probs()
    row_sums = P.sum(axis=1)
    aaae(row_sums, jnp.ones(7), decimal=DECIMAL_PRECISION)


def test_tauchen_gauss_hermite_centers_on_unconditional_mean():
    """GH Tauchen gridpoints center on mu / (1 - rho)."""
    mu, rho = 2.0, 0.8
    grid = TauchenAR1Process(n_points=11, gauss_hermite=True, rho=rho, sigma=0.5, mu=mu)
    points = grid.get_gridpoints()
    midpoint = (points[0] + points[-1]) / 2
    expected = mu / (1 - rho)
    aaae(midpoint, expected, decimal=_CENTERING_DECIMAL)


def test_lognormal_correct_shape_without_params():
    """LogNormal without params returns correct-shape NaN arrays."""
    grid = LogNormalIIDProcess(n_points=3, gauss_hermite=True)
    assert not grid.is_fully_specified
    assert grid.get_gridpoints().shape == (3,)
    assert grid.get_transition_probs().shape == (3, 3)


def test_lognormal_fully_specified_gauss_hermite():
    """LogNormal with gauss_hermite=True is fully specified with mu/sigma."""
    grid = LogNormalIIDProcess(n_points=5, gauss_hermite=True, mu=0.0, sigma=1.0)
    assert grid.is_fully_specified
    assert grid.get_gridpoints().shape == (5,)


def test_lognormal_fully_specified_n_std():
    """LogNormal with gauss_hermite=False is fully specified with mu/sigma/n_std."""
    grid = LogNormalIIDProcess(
        n_points=5, gauss_hermite=False, mu=0.0, sigma=1.0, n_std=3.0
    )
    assert grid.is_fully_specified
    assert grid.get_gridpoints().shape == (5,)


def test_lognormal_gridpoints_are_positive():
    """All LogNormal gridpoints are strictly positive."""
    grid = LogNormalIIDProcess(n_points=7, gauss_hermite=True, mu=0.0, sigma=1.0)
    assert jnp.all(grid.get_gridpoints() > 0)


def test_lognormal_gauss_hermite_weights_sum_to_one():
    """GH weights for iid LogNormal sum to 1."""
    grid = LogNormalIIDProcess(n_points=7, gauss_hermite=True, mu=0.0, sigma=1.0)
    P = grid.get_transition_probs()
    aaae(P[0].sum(), 1.0, decimal=DECIMAL_PRECISION)


class _MixtureOptions(TypedDict):
    n_std: float
    p1: float
    mu1: float
    sigma1: float
    mu2: float
    sigma2: float


class _TauchenMixtureOptions(_MixtureOptions):
    rho: float
    mu: float


_NORMAL_MIXTURE_KWARGS: _MixtureOptions = {
    "n_std": 3.0,
    "p1": 0.9,
    "mu1": 0.0,
    "sigma1": 0.1,
    "mu2": -0.5,
    "sigma2": 0.3,
}


def test_normal_mixture_transition_probs_rows_sum_to_one():
    """NormalMixture transition probability rows sum to 1."""
    grid = NormalMixtureIIDProcess(n_points=7, **_NORMAL_MIXTURE_KWARGS)
    P = grid.get_transition_probs()
    row_sums = P.sum(axis=1)
    aaae(row_sums, jnp.ones(7), decimal=DECIMAL_PRECISION)


def test_iid_normal_mixture_stationary_moments():
    """IID NormalMixture stationary mean and std match mixture moments."""
    kwargs = _NORMAL_MIXTURE_KWARGS
    grid = NormalMixtureIIDProcess(n_points=21, **kwargs)
    got_mean, got_std = _stationary_moments(
        gridpoints=grid.get_gridpoints(), P=grid.get_transition_probs()
    )
    p1 = kwargs["p1"]
    mu1, mu2 = kwargs["mu1"], kwargs["mu2"]
    sigma1, sigma2 = kwargs["sigma1"], kwargs["sigma2"]
    expected_mean = p1 * mu1 + (1 - p1) * mu2
    expected_var = (
        p1 * (sigma1**2 + mu1**2) + (1 - p1) * (sigma2**2 + mu2**2) - expected_mean**2
    )
    aaae(got_mean, expected_mean, decimal=1)
    aaae(got_std, float(jnp.sqrt(expected_var)), decimal=1)


_TAUCHEN_NORMAL_MIXTURE_KWARGS: _TauchenMixtureOptions = {
    "rho": 0.8,
    "mu": 1.0,
    "n_std": 3.0,
    "p1": 0.9,
    "mu1": 0.0,
    "sigma1": 0.1,
    "mu2": -0.5,
    "sigma2": 0.3,
}


def test_tauchen_normal_mixture_transition_probs_rows_sum_to_one():
    """TauchenNormalMixture transition probability rows sum to 1."""
    grid = TauchenNormalMixtureAR1Process(n_points=7, **_TAUCHEN_NORMAL_MIXTURE_KWARGS)
    P = grid.get_transition_probs()
    row_sums = P.sum(axis=1)
    aaae(row_sums, jnp.ones(7), decimal=DECIMAL_PRECISION)


def test_tauchen_normal_mixture_centers_on_unconditional_mean():
    """TauchenNormalMixture gridpoints center on (mu + mean_eps) / (1 - rho)."""
    kwargs = _TAUCHEN_NORMAL_MIXTURE_KWARGS
    grid = TauchenNormalMixtureAR1Process(n_points=11, **kwargs)
    points = grid.get_gridpoints()
    midpoint = (points[0] + points[-1]) / 2
    mean_eps = kwargs["p1"] * kwargs["mu1"] + (1 - kwargs["p1"]) * kwargs["mu2"]
    expected = (kwargs["mu"] + mean_eps) / (1 - kwargs["rho"])
    aaae(midpoint, expected, decimal=_CENTERING_DECIMAL)


def test_tauchen_normal_mixture_stationary_moments_and_autocorrelation():
    """TauchenNormalMixture stationary mean, std, and autocorrelation match theory."""
    kwargs = _TAUCHEN_NORMAL_MIXTURE_KWARGS
    grid = TauchenNormalMixtureAR1Process(n_points=21, **kwargs)
    points = grid.get_gridpoints()
    P = grid.get_transition_probs()

    p1, mu1, sigma1 = kwargs["p1"], kwargs["mu1"], kwargs["sigma1"]
    mu2, sigma2 = kwargs["mu2"], kwargs["sigma2"]
    rho = kwargs["rho"]

    mean_eps = p1 * mu1 + (1 - p1) * mu2
    expected_mean = (kwargs["mu"] + mean_eps) / (1 - rho)
    sigma_eps_sq = (
        p1 * (sigma1**2 + mu1**2) + (1 - p1) * (sigma2**2 + mu2**2) - mean_eps**2
    )
    expected_std = float(jnp.sqrt(sigma_eps_sq / (1 - rho**2)))

    got_mean, got_std = _stationary_moments(gridpoints=points, P=P)
    aaae(got_mean, expected_mean, decimal=1)
    aaae(got_std, expected_std, decimal=1)

    got_rho = _lag1_autocorrelation(gridpoints=points, P=P)
    aaae(got_rho, rho, decimal=1)


TAUCHEN_CASES = [
    {"rho": 0.9, "sigma": 0.5, "mu": 0.0, "n_std": 3, "n": 7},
    {"rho": 0.7, "sigma": 1.0, "mu": 2.0, "n_std": 2, "n": 5},
    {"rho": 0.99, "sigma": 0.1, "mu": 0.0, "n_std": 3, "n": 11},
]

ROUWENHORST_CASES = [
    {"rho": 0.9, "sigma": 0.5, "mu": 0.0, "n": 7},
    {"rho": 0.7, "sigma": 1.0, "mu": 2.0, "n": 5},
    {"rho": 0.99, "sigma": 0.1, "mu": 0.0, "n": 11},
]


@pytest.mark.parametrize(
    "case", TAUCHEN_CASES, ids=lambda c: f"rho={c['rho']}_n={c['n']}"
)
def test_tauchen_matches_quantecon(case):
    """Tauchen (non-GH) gridpoints and transition probs match QuantEcon."""
    qe = qe_tauchen(**case)
    grid = TauchenAR1Process(
        n_points=case["n"],
        gauss_hermite=False,
        rho=case["rho"],
        sigma=case["sigma"],
        mu=case["mu"],
        n_std=case["n_std"],
    )
    aaae(grid.get_gridpoints(), qe.state_values, decimal=DECIMAL_PRECISION)
    aaae(grid.get_transition_probs(), qe.P, decimal=DECIMAL_PRECISION)


@pytest.mark.parametrize(
    "case", ROUWENHORST_CASES, ids=lambda c: f"rho={c['rho']}_n={c['n']}"
)
@pytest.mark.filterwarnings("ignore:The API of rouwenhorst has changed:UserWarning")
def test_rouwenhorst_matches_quantecon(case):
    """Rouwenhorst gridpoints and transition probs match QuantEcon."""
    qe = qe_rouwenhorst(**case)
    grid = RouwenhorstAR1Process(
        n_points=case["n"],
        rho=case["rho"],
        sigma=case["sigma"],
        mu=case["mu"],
    )
    aaae(grid.get_gridpoints(), qe.state_values, decimal=DECIMAL_PRECISION)
    aaae(grid.get_transition_probs(), qe.P, decimal=DECIMAL_PRECISION)


def _stationary_moments(*, gridpoints, P):
    """Compute mean and variance from the stationary distribution of a Markov chain."""
    # Stationary distribution: solve pi @ P = pi via eigendecomposition
    eigvals, eigvecs = jnp.linalg.eig(P.T)
    idx = jnp.argmin(jnp.abs(eigvals - 1.0))
    pi = jnp.real(eigvecs[:, idx])
    pi = pi / pi.sum()
    mean = jnp.dot(pi, gridpoints)
    var = jnp.dot(pi, (gridpoints - mean) ** 2)
    return float(mean), float(jnp.sqrt(var))


def _lag1_autocorrelation(*, gridpoints, P):
    """Compute lag-1 autocorrelation from gridpoints and transition matrix."""
    eigvals, eigvecs = jnp.linalg.eig(P.T)
    idx = jnp.argmin(jnp.abs(eigvals - 1.0))
    pi = jnp.real(eigvecs[:, idx])
    pi = pi / pi.sum()
    mean = jnp.dot(pi, gridpoints)
    var = jnp.dot(pi, (gridpoints - mean) ** 2)
    # E[X_t * X_{t+1}] = sum_i sum_j pi_i * P_ij * x_i * x_j
    cross = float(jnp.sum(pi[:, None] * P * gridpoints[:, None] * gridpoints[None, :]))
    return (cross - float(mean) ** 2) / float(var)


@pytest.mark.parametrize("gauss_hermite", [True, False])
def test_iid_normal_stationary_moments(gauss_hermite):
    """IID Normal stationary mean and std match mu and sigma."""
    mu, sigma = 1.5, 0.8
    grid = NormalIIDProcess(
        n_points=21,
        mu=mu,
        sigma=sigma,
        gauss_hermite=gauss_hermite,
        n_std=None if gauss_hermite else 4.0,
    )
    got_mean, got_std = _stationary_moments(
        gridpoints=grid.get_gridpoints(), P=grid.get_transition_probs()
    )
    aaae(got_mean, mu, decimal=1)
    aaae(got_std, sigma, decimal=1)


@pytest.mark.parametrize("gauss_hermite", [True, False])
def test_iid_lognormal_stationary_moments(gauss_hermite):
    """IID LogNormal stationary log-mean and log-std match mu and sigma."""
    mu, sigma = 0.5, 0.3
    grid = LogNormalIIDProcess(
        n_points=21,
        mu=mu,
        sigma=sigma,
        gauss_hermite=gauss_hermite,
        n_std=None if gauss_hermite else 4.0,
    )
    points = grid.get_gridpoints()
    P = grid.get_transition_probs()
    got_mean, got_std = _stationary_moments(gridpoints=jnp.log(points), P=P)
    aaae(got_mean, mu, decimal=1)
    aaae(got_std, sigma, decimal=1)


@pytest.mark.parametrize(
    ("grid_cls", "extra_kw"),
    [
        (TauchenAR1Process, {"gauss_hermite": True}),
        (TauchenAR1Process, {"gauss_hermite": False, "n_std": 3.0}),
        (RouwenhorstAR1Process, {}),
    ],
    ids=["tauchen-gh", "tauchen-linspace", "rouwenhorst"],
)
def test_ar1_stationary_moments_and_autocorrelation(*, grid_cls, extra_kw):
    """AR(1) stationary mean, std, and lag-1 autocorrelation match theory."""
    rho, sigma, mu = 0.8, 0.4, 1.0
    grid = grid_cls(n_points=21, rho=rho, sigma=sigma, mu=mu, **extra_kw)
    points = grid.get_gridpoints()
    P = grid.get_transition_probs()

    expected_mean = mu / (1 - rho)
    expected_std = sigma / jnp.sqrt(1 - rho**2)

    got_mean, got_std = _stationary_moments(gridpoints=points, P=P)
    aaae(got_mean, expected_mean, decimal=1)
    aaae(got_std, expected_std, decimal=1)

    got_rho = _lag1_autocorrelation(gridpoints=points, P=P)
    aaae(got_rho, rho, decimal=1)


@pytest.mark.parametrize("hash_seed", [0, 1, 2])
def test_process_lottery_axes_follow_declaration_order_across_hash_seeds(
    hash_seed: int,
) -> None:
    """Continuation lottery axes retain process declaration order in fresh processes."""
    root = Path(__file__).resolve().parents[1]
    script = textwrap.dedent("""
        import json
        import os
        from typing import Any
        import jax.numpy as jnp
        import numpy as np
        from lcm import (AgeGrid, Model, Regime, UniformIIDProcess,
                         categorical)
        from lcm.typing import ContinuousState, FloatND, ScalarInt
        from _lcm.regime_building import Q_and_F

        @categorical(ordered=False)
        class RegimeId:
            alive: ScalarInt
            dead: ScalarInt

        def utility(*, gamma: ContinuousState, alpha: ContinuousState,
                    beta: ContinuousState) -> FloatND:
            return gamma + 2 * alpha + 3 * beta

        observed = []
        original = Q_and_F._build_target_continuation
        def observe(**kwargs: Any) -> Any:
            result = original(**kwargs)
            if kwargs['target_regime_name'] == 'alive':
                observed.append(list(result.lottery_axis_names))
            return result
        Q_and_F._build_target_continuation = observe
        try:
            model = Model(
                edges={'alive': {'alive': 0, 'dead': 1}},
                regimes={
                    'alive': Regime(
                        states={
                            'gamma': UniformIIDProcess(n_points=4, start=0, stop=3),
                            'alpha': UniformIIDProcess(n_points=2, start=0, stop=1),
                            'beta': UniformIIDProcess(n_points=3, start=0, stop=2),
                        },
                        functions={'utility': utility},
                    ),
                    'dead': Regime(functions={'utility': lambda: 0.0}),
                },
                ages=AgeGrid(start=0, inclusive_stop=2, step='Y'),
                regime_id_class=RegimeId,
                initial_nodes={0: 'alive'},
            )
            solution = model.solve(params={'discount_factor': 1.0}, log_level='off')
            values = solution.values
            grids = model.user_regimes['alive'].states
            gamma = np.asarray(grids['gamma'].get_gridpoints())[:, None, None]
            alpha = np.asarray(grids['alpha'].get_gridpoints())[None, :, None]
            beta = np.asarray(grids['beta'].get_gridpoints())[None, None, :]
            terminal = gamma + 2 * alpha + 3 * beta
            decimal_precision = int(os.environ['TEST_DECIMAL_PRECISION'])
            np.testing.assert_allclose(values[1]['alive'], terminal,
                atol=10**(-decimal_precision), rtol=10**(-decimal_precision))
            np.testing.assert_allclose(values[0]['alive'], terminal + 5.5,
                atol=10**(-decimal_precision), rtol=10**(-decimal_precision))
        finally:
            Q_and_F._build_target_continuation = original
        print('LOTTERY_AXES=' + json.dumps(observed))
    """)
    pixi = shutil.which("pixi")
    assert pixi is not None
    completed = subprocess.run(  # noqa: S603 - fixed, repository-owned test script
        [pixi, "run", "-e", "tests-cpu", "python", "-c", script],
        cwd=root,
        env={
            **os.environ,
            "PYTHONHASHSEED": str(hash_seed),
            "JAX_PLATFORMS": "cpu",
            "JAX_ENABLE_X64": str(X64_ENABLED).lower(),
            "TEST_DECIMAL_PRECISION": str(DECIMAL_PRECISION),
            "PYTHONPATH": os.pathsep.join((str(root / "src"), str(root))),
        },
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    lines = [
        line
        for line in completed.stdout.splitlines()
        if line.startswith("LOTTERY_AXES=")
    ]
    assert len(lines) == 1, completed.stdout
    observed = json.loads(lines[0].removeprefix("LOTTERY_AXES="))
    assert observed, "The public model must construct an alive continuation"
    assert all(
        order == ["next_gamma", "next_alpha", "next_beta"] for order in observed
    ), (hash_seed, observed)


@pytest.mark.parametrize("rho", [-1.2, -1.0, 0.0, 0.8, 1.0, 1.2])
@pytest.mark.parametrize("mu", [-0.3, 0.0, 0.4])
def test_tauchen_on_nodes_matches_integrated_normal(*, rho: float, mu: float) -> None:
    """Physical node bins integrate the conditional normal, including unit roots."""
    nodes = (-1.4, -0.3, 0.2, 1.1)
    sigma = 0.6
    process = TauchenAR1Process.on_nodes(
        nodes=np.array(nodes), rho=rho, sigma=sigma, mu=mu
    )
    boundaries = [
        -np.inf,
        *[(a + b) / 2 for a, b in pairwise(nodes)],
        np.inf,
    ]
    expected = np.asarray(
        [
            [
                quad(
                    partial(norm.pdf, loc=mu + rho * source, scale=sigma),
                    lower,
                    upper,
                    epsabs=1e-13,
                    epsrel=1e-13,
                )[0]
                for lower, upper in pairwise(boundaries)
            ]
            for source in nodes
        ]
    )
    got = process.get_transition_probs()
    aaae(got, expected, decimal=DECIMAL_PRECISION)
    aaae(got.sum(axis=1), np.ones(len(nodes)), decimal=DECIMAL_PRECISION)
    np.testing.assert_array_equal(process.to_jax(), jnp.asarray(nodes))


@pytest.mark.parametrize("gauss_hermite", [False, True])
def test_tauchen_on_default_nodes_matches_stationary_process(
    *,
    gauss_hermite: bool,
) -> None:
    """Supplied stationary nodes preserve Tauchen probabilities with a drift."""
    ordinary = TauchenAR1Process(
        n_points=5,
        gauss_hermite=gauss_hermite,
        n_std=None if gauss_hermite else 3.0,
        rho=0.6,
        sigma=0.4,
        mu=0.3,
    )
    process = TauchenAR1Process.on_nodes(
        nodes=np.array(tuple(float(x) for x in ordinary.to_jax())),
        rho=0.6,
        sigma=0.4,
        mu=0.3,
    )
    aaae(
        process.get_transition_probs(),
        ordinary.get_transition_probs(),
        decimal=DECIMAL_PRECISION,
    )


@pytest.mark.parametrize(
    "nodes",
    [(), (0.0,), (0.0, 0.0), (1.0, -1.0), (0.0, float("nan")), (0.0, float("inf"))],
)
def test_tauchen_on_nodes_rejects_invalid_grid(nodes: tuple[float, ...]) -> None:
    """Supplied nodes require a finite strictly increasing interpolation axis."""
    with pytest.raises(GridInitializationError, match="nodes"):
        TauchenAR1Process.on_nodes(nodes=np.array(nodes), rho=1.0, sigma=0.3)


@pytest.mark.parametrize(
    ("nodes", "type_name"),
    [
        pytest.param([-1.0, 0.0, 1.0], "list", id="list"),
        pytest.param((-1.0, 0.0, 1.0), "tuple", id="tuple"),
    ],
)
def test_tauchen_on_nodes_refuses_a_non_array_container(*, nodes, type_name) -> None:
    """`nodes` is a NumPy or JAX array; the error names the type and the spelling."""
    with pytest.raises(
        GridInitializationError,
        match=rf"got {type_name}.*on_nodes\(nodes=jnp\.array\(",
    ):
        TauchenAR1Process.on_nodes(nodes=nodes, rho=1.0, sigma=0.3)


@pytest.mark.parametrize(
    "nodes",
    [np.array(["-1.0", "0.0", "1.0"]), np.array([-1.0, None, 1.0], dtype=object)],
)
def test_tauchen_on_nodes_refuses_a_non_numeric_array(nodes) -> None:
    """An array of nodes must have an integer or floating dtype."""
    with pytest.raises(GridInitializationError, match="integer or floating dtype"):
        TauchenAR1Process.on_nodes(nodes=nodes, rho=1.0, sigma=0.3)


def test_tauchen_on_nodes_accepts_a_jax_array() -> None:
    """A JAX array of nodes gives the same physical axis as a NumPy array."""
    process = TauchenAR1Process.on_nodes(
        nodes=jnp.array([-1.0, 0.0, 2.0]), rho=1.0, sigma=0.3
    )
    np.testing.assert_array_equal(process.to_jax(), jnp.asarray([-1.0, 0.0, 2.0]))


def test_tauchen_on_nodes_runtime_parameters_keep_fixed_coordinates() -> None:
    """Runtime law parameters leave the supplied physical grid observable."""
    process = TauchenAR1Process.on_nodes(nodes=np.array([-1.0, 0.0, 2.0]))
    assert process.params_to_pass_at_runtime == ("rho", "sigma")
    np.testing.assert_array_equal(process.to_jax(), jnp.asarray([-1.0, 0.0, 2.0]))
    aaae(
        process.get_coordinate(jnp.asarray([-0.5, 1.0])),
        jnp.asarray([0.5, 1.5]),
        decimal=DECIMAL_PRECISION,
    )
    compiled = jit(process.compute_transition_probs)
    params = {
        "rho": jnp.asarray(1.0),
        "sigma": jnp.asarray(0.3),
        "mu": jnp.asarray(0.0),
    }
    aaae(
        compiled(**params),
        process.compute_transition_probs(**params),
        decimal=DECIMAL_PRECISION,
    )


@pytest.mark.parametrize("runtime", [False, True])
@pytest.mark.parametrize("log_level", ["off", "warning"])
@pytest.mark.parametrize("budget", [None, 2**28])
def test_tauchen_on_nodes_unit_root_native_solve_and_simulate(
    *, runtime: bool, log_level: LogLevel, budget: int | None
) -> None:
    """Native checked Bellman integration uses physical nodes and a unit-root law."""

    def income_squared(income: ContinuousState) -> FloatND:
        return income**2

    nodes = (-2.0, -0.5, 0.0, 0.8, 2.0)
    process = TauchenAR1Process.on_nodes(
        nodes=np.array(nodes),
        rho=None if runtime else 1.0,
        sigma=None if runtime else 0.1,
    )
    model = Model(
        regimes={
            "alive": Regime(
                states={"income": process}, functions={"utility": income_squared}
            ),
            "dead": Regime(
                states={"income": process}, functions={"utility": income_squared}
            ),
        },
        ages=AgeGrid(start=0, inclusive_stop=1, step="Y"),
        regime_id_class=RegimeId,
        initial_nodes={0: "alive"},
        edges={"alive": {"dead": 0}},
        execution_config=ExecutionConfig(device_memory_bytes=budget),
    )
    params = {
        "alive": {
            "discount_factor": 1.0,
            **({"income": {"rho": 1.0, "sigma": 0.1}} if runtime else {}),
        },
        "dead": {"income": {"rho": 1.0, "sigma": 0.1}} if runtime else {},
    }
    solution = model.solve(params=params, log_level=log_level, runtime_checks=True)
    expected = jnp.asarray(nodes) ** 2
    weights = process.compute_transition_probs(
        rho=jnp.asarray(1.0), sigma=jnp.asarray(0.1), mu=jnp.asarray(0.0)
    )
    aaae(solution.values[1]["dead"], expected, decimal=DECIMAL_PRECISION)
    aaae(
        solution.values[0]["alive"],
        expected + weights @ expected,
        decimal=DECIMAL_PRECISION,
    )
    result = model.simulate(
        params=params,
        solution=solution,
        log_level=log_level,
        runtime_checks=True,
        initial_conditions={
            "income": jnp.zeros(400),
            "age": jnp.zeros(400),
            "regime_id": jnp.full(400, RegimeId.alive),
        },
        seed=42,
    ).to_dataframe()
    draws = result.loc[result["age"] == 1.0, "income"].to_numpy()
    assert len(draws) == 400
    assert np.isfinite(draws).all()
    assert abs(draws.mean()) < 0.025
    assert 0.075 < draws.std() < 0.125


@pytest.mark.parametrize("rho", [1, 1.0])
def test_tauchen_on_nodes_integer_parameters_match_float_parameters(rho: float) -> None:
    """Canonical integer scalars preserve the physical conditional law."""
    nodes = (-1.0, 0.0, 2.0)
    actual = TauchenAR1Process.on_nodes(nodes=np.array(nodes), rho=rho, sigma=1, mu=0)
    expected = TauchenAR1Process.on_nodes(
        nodes=np.array(nodes), rho=1.0, sigma=1.0, mu=0.0
    )
    aaae(
        actual.get_transition_probs(),
        expected.get_transition_probs(),
        decimal=DECIMAL_PRECISION,
    )


@pytest.mark.parametrize("rho", [float("inf"), float("nan")])
def test_tauchen_on_nodes_rejects_nonfinite_persistence(rho: float) -> None:
    """Given nodes remove the stationary-grid restriction, not a finite-law boundary."""
    with pytest.raises(GridInitializationError, match="rho must be finite"):
        TauchenAR1Process.on_nodes(nodes=np.array([-1.0, 1.0]), rho=rho, sigma=0.1)


def test_tauchen_on_nodes_validates_active_precision() -> None:
    """Nodes that collapse at the working precision cannot be interpolated."""
    epsilon = float(np.finfo(np.float64 if X64_ENABLED else np.float32).eps)
    with pytest.raises(GridInitializationError, match="active precision"):
        TauchenAR1Process.on_nodes(
            nodes=np.array([1.0, 1.0 + epsilon / 4]), rho=1.0, sigma=0.1
        )


def test_tauchen_on_nodes_scale_derivative_matches_normal_bins() -> None:
    """Runtime innovation-scale derivatives follow the analytic normal-bin law."""
    nodes = (-1.4, -0.3, 0.2, 1.1)
    process = TauchenAR1Process.on_nodes(nodes=np.array(nodes))
    rho, mu, sigma = 1.0, 0.3, 0.4

    def matrix(scale: ScalarFloat) -> FloatND:
        return process.compute_transition_probs(
            rho=jnp.asarray(rho), mu=jnp.asarray(mu), sigma=scale
        )

    points = np.asarray(nodes)
    edges = (points[:-1] + points[1:]) / 2
    z = (edges[None, :] - mu - rho * points[:, None]) / sigma
    edge_derivatives = -z * norm.pdf(z) / sigma
    expected = np.concatenate(
        [
            edge_derivatives[:, :1],
            np.diff(edge_derivatives, axis=1),
            -edge_derivatives[:, -1:],
        ],
        axis=1,
    )
    aaae(jit(jacrev(matrix))(jnp.asarray(sigma)), expected, decimal=DECIMAL_PRECISION)


def test_tauchen_on_nodes_batched_runtime_scale_matches_normal_bins() -> None:
    """Batched runtime scales preserve independently integrated physical-bin rows."""
    nodes = (-1.4, -0.3, 0.2, 1.1)
    process = TauchenAR1Process.on_nodes(nodes=np.array(nodes))

    def matrix(scale: ScalarFloat) -> FloatND:
        return process.compute_transition_probs(
            rho=jnp.asarray(1.0), mu=jnp.asarray(0.3), sigma=scale
        )

    scales = (0.2, 0.4, 0.7)
    points = np.asarray(nodes)
    edges = (points[:-1] + points[1:]) / 2
    expected = []
    for sigma in scales:
        cumulative = norm.cdf((edges[None, :] - 0.3 - points[:, None]) / sigma)
        expected.append(
            np.concatenate(
                [
                    cumulative[:, :1],
                    np.diff(cumulative, axis=1),
                    1 - cumulative[:, -1:],
                ],
                axis=1,
            )
        )
    aaae(
        jit(vmap(matrix))(jnp.asarray(scales)),
        np.asarray(expected),
        decimal=DECIMAL_PRECISION,
    )


def _exact_normal_bin_matrix(
    *, nodes: tuple[float, ...], rho: float, mu: float, sigma: float
) -> np.ndarray:
    """Integrate the represented law over exactly centred midpoint bins.

    Rational arithmetic forms every standardized boundary
    $((a+b)/2 - \\mu - \\rho y) / \\sigma$ from the represented inputs before a
    single conversion to float, and a scalar `math.erfc` evaluates the CDF. Neither
    step shares arithmetic with the production row builders.
    """
    return np.asarray(
        [
            np.diff([0.0, *[_scalar_normal_cdf(float(z)) for z in row], 1.0])
            for row in _exact_standardized_edges(
                nodes=nodes, rho=rho, mu=mu, sigma=sigma
            )
        ]
    )


def _exact_standardized_edges(
    *, nodes: tuple[float, ...], rho: float, mu: float, sigma: float
) -> tuple[tuple[Fraction, ...], ...]:
    """Centre exact rational midpoints on each source's conditional mean."""
    exact_nodes = tuple(Fraction(value) for value in nodes)
    return tuple(
        tuple(
            ((left + right) / 2 - Fraction(mu) - Fraction(rho) * source)
            / Fraction(sigma)
            for left, right in pairwise(exact_nodes)
        )
        for source in exact_nodes
    )


def _scalar_normal_cdf(z: float) -> float:
    return 0.5 * math.erfc(-z / math.sqrt(2.0))


def _scalar_normal_pdf(z: float) -> float:
    return math.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)


def _adjacent_nodes(
    *, sign: float, offset_exponent: int
) -> tuple[tuple[float, float], float]:
    """Return two adjacent representable nodes and their exact positive spacing."""
    mantissa_bits = 52 if X64_ENABLED else 23
    offset = math.ldexp(1.0, offset_exponent)
    step = math.ldexp(1.0, offset_exponent - mantissa_bits)
    low, high = sorted((sign * offset, sign * (offset + step)))
    return (low, high), step


type _MatrixRoute = Literal["fixed", "runtime_eager", "runtime_jit"]


def _on_nodes_matrix(
    *,
    nodes: tuple[float, ...],
    rho: float,
    mu: float,
    sigma: float,
    route: _MatrixRoute,
) -> FloatND:
    """Evaluate the supplied-node matrix with a fixed or fully runtime law."""
    if route == "fixed":
        return TauchenAR1Process.on_nodes(
            nodes=np.array(nodes), rho=rho, mu=mu, sigma=sigma
        ).get_transition_probs()
    process = TauchenAR1Process.on_nodes(
        nodes=np.array(nodes), rho=None, mu=None, sigma=None
    )
    evaluate = (
        jit(process.compute_transition_probs)
        if route == "runtime_jit"
        else process.compute_transition_probs
    )
    return evaluate(rho=jnp.asarray(rho), mu=jnp.asarray(mu), sigma=jnp.asarray(sigma))


@pytest.mark.parametrize("route", ["fixed", "runtime_eager", "runtime_jit"])
def test_tauchen_on_adjacent_nodes_keeps_the_half_cell(route: _MatrixRoute) -> None:
    """Rows at adjacent nodes are (Phi(2), 1 - Phi(2)) and (Phi(-2), 1 - Phi(-2))."""
    nodes, step = _adjacent_nodes(sign=1.0, offset_exponent=0)
    expected = _exact_normal_bin_matrix(nodes=nodes, rho=1.0, mu=0.0, sigma=step / 4)
    actual = _on_nodes_matrix(nodes=nodes, rho=1.0, mu=0.0, sigma=step / 4, route=route)
    aaae(actual, expected, decimal=DECIMAL_PRECISION)


@pytest.mark.parametrize(
    ("sign", "offset_exponent"), [(1.0, 0), (-1.0, 0), (1.0, 16), (-1.0, -8)]
)
@pytest.mark.parametrize("sigma_in_steps", [0.25, 1.0])
@pytest.mark.parametrize("route", ["fixed", "runtime_eager", "runtime_jit"])
def test_tauchen_on_adjacent_offset_nodes_match_exact_bins(
    *, sign: float, offset_exponent: int, sigma_in_steps: float, route: _MatrixRoute
) -> None:
    """Both signs and several offsets keep the half-cell of an adjacent node pair."""
    nodes, step = _adjacent_nodes(sign=sign, offset_exponent=offset_exponent)
    sigma = step * sigma_in_steps
    expected = _exact_normal_bin_matrix(nodes=nodes, rho=1.0, mu=0.0, sigma=sigma)
    actual = _on_nodes_matrix(nodes=nodes, rho=1.0, mu=0.0, sigma=sigma, route=route)
    aaae(actual, expected, decimal=DECIMAL_PRECISION)


@pytest.mark.parametrize("sign", [1.0, -1.0])
@pytest.mark.parametrize("route", ["fixed", "runtime_eager", "runtime_jit"])
def test_tauchen_on_nodes_with_overflowing_endpoint_sum_split_at_midpoint(
    *, sign: float, route: _MatrixRoute
) -> None:
    """A finite midpoint whose endpoints' sum overflows still splits mass evenly."""
    scale = math.ldexp(1.0, 1023 if X64_ENABLED else 127)
    low, high = sorted((sign * scale, sign * 1.5 * scale))
    mu, sigma = sign * 1.25 * scale, scale / 4
    actual = _on_nodes_matrix(
        nodes=(low, high), rho=0.0, mu=mu, sigma=sigma, route=route
    )
    aaae(actual, np.full((2, 2), 0.5), decimal=DECIMAL_PRECISION)


def test_tauchen_on_adjacent_nodes_batched_scales_match_exact_bins() -> None:
    """Every vmapped scale integrates over the same exactly centred midpoint."""
    nodes, step = _adjacent_nodes(sign=1.0, offset_exponent=0)
    process = TauchenAR1Process.on_nodes(nodes=np.array(nodes))
    scales = (step / 4, step / 2, step)

    def matrix(scale: ScalarFloat) -> FloatND:
        return process.compute_transition_probs(
            rho=jnp.asarray(1.0), mu=jnp.asarray(0.0), sigma=scale
        )

    expected = np.stack(
        [
            _exact_normal_bin_matrix(nodes=nodes, rho=1.0, mu=0.0, sigma=scale)
            for scale in scales
        ]
    )
    aaae(jit(vmap(matrix))(jnp.asarray(scales)), expected, decimal=DECIMAL_PRECISION)


@pytest.mark.parametrize("sign", [1.0, -1.0])
def test_tauchen_on_adjacent_nodes_scale_derivative_matches_exact_bins(
    sign: float,
) -> None:
    """The derivative in the scale in cell widths uses the centred half-cell."""
    nodes, step = _adjacent_nodes(sign=sign, offset_exponent=0)
    process = TauchenAR1Process.on_nodes(nodes=np.array(nodes))
    ratio = 0.25

    def matrix(scale_in_steps: ScalarFloat) -> FloatND:
        return process.compute_transition_probs(
            rho=jnp.asarray(1.0), mu=jnp.asarray(0.0), sigma=step * scale_in_steps
        )

    edges = _exact_standardized_edges(nodes=nodes, rho=1.0, mu=0.0, sigma=step * ratio)
    expected = np.asarray(
        [
            np.diff(
                [
                    0.0,
                    *[-float(z) * _scalar_normal_pdf(float(z)) / ratio for z in row],
                    0.0,
                ]
            )
            for row in edges
        ]
    )
    aaae(jit(jacrev(matrix))(jnp.asarray(ratio)), expected, decimal=DECIMAL_PRECISION)


def test_tauchen_on_adjacent_nodes_preserve_a_finite_margin_choice() -> None:
    """A certain payoff of 1/4 beats the lottery paying one at the upper node."""
    nodes, step = _adjacent_nodes(sign=1.0, offset_exponent=0)
    upper_mass = TauchenAR1Process.on_nodes(
        nodes=np.array(nodes), rho=1.0, mu=0.0, sigma=step / 4
    ).get_transition_probs()[0, 1]
    assert int(np.argmax([0.25, float(upper_mass)])) == 0


@pytest.mark.parametrize("family", ["tauchen", "iid_normal"])
def test_state_conditioned_rows_on_adjacent_nodes_match_exact_bins(
    family: Literal["tauchen", "iid_normal"],
) -> None:
    """State-conditioned rows centre adjacent-node midpoints exactly, too."""
    nodes, step = _adjacent_nodes(sign=1.0, offset_exponent=0)
    sigma = step / 4
    rho = 1.0 if family == "tauchen" else 0.0
    mu = 0.0 if family == "tauchen" else nodes[0]
    expected = _exact_normal_bin_matrix(nodes=nodes, rho=rho, mu=mu, sigma=sigma)
    rows = [
        conditioned_row(
            family=family,
            nodes=jnp.asarray(nodes),
            sigma=jnp.asarray(sigma),
            from_value=jnp.asarray(source),
            mu=jnp.asarray(mu),
            rho=jnp.asarray(rho),
        )
        for source in nodes
    ]
    aaae(np.stack(rows), expected, decimal=DECIMAL_PRECISION)


@pytest.mark.slow
@pytest.mark.parametrize("runtime", [False, True])
def test_tauchen_on_adjacent_nodes_two_period_value_uses_exact_bins(
    *, runtime: bool
) -> None:
    """The first-period value is the exact-bin expectation of terminal payoffs 0, 1."""
    nodes, step = _adjacent_nodes(sign=1.0, offset_exponent=0)
    sigma = step / 4
    process = TauchenAR1Process.on_nodes(
        nodes=np.array(nodes),
        rho=None if runtime else 1.0,
        sigma=None if runtime else sigma,
    )

    def no_current_payoff(income: ContinuousState) -> FloatND:
        return jnp.zeros_like(income)

    def terminal_payoff(income: ContinuousState) -> FloatND:
        return (income - 1.0) / step

    model = Model(
        regimes={
            "alive": Regime(
                states={"income": process}, functions={"utility": no_current_payoff}
            ),
            "dead": Regime(
                states={"income": process}, functions={"utility": terminal_payoff}
            ),
        },
        ages=AgeGrid(start=0, inclusive_stop=1, step="Y"),
        regime_id_class=RegimeId,
        initial_nodes={0: "alive"},
        edges={"alive": {"dead": 0}},
    )
    law = {"income": {"rho": 1.0, "sigma": sigma}} if runtime else {}
    solution = model.solve(
        params={"alive": {"discount_factor": 1.0, **law}, "dead": law},
        log_level="off",
        runtime_checks=True,
    )
    expected = _exact_normal_bin_matrix(
        nodes=nodes, rho=1.0, mu=0.0, sigma=sigma
    ) @ np.asarray([0.0, 1.0])
    aaae(solution.values[0]["alive"], expected, decimal=DECIMAL_PRECISION)
