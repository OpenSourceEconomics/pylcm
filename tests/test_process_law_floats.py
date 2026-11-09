"""Process-law values are floats, however they are written.

A law coefficient written as a Python `int` describes the same law as its float twin:
the declaration stores a Python `float`, the nodes and transition rows agree, and so
does the model's semantic identity. Integer category codes and callback indices that
feed a law keep their integer kind.
"""

import dataclasses
import itertools
import math
import pickle
from collections.abc import Callable, Mapping
from types import MappingProxyType
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.processes.base import _ContinuousStochasticProcess
from _lcm.processes.state_conditioned import sigma_array_by_code
from _lcm.regime_building.fixed_process_laws import bind_fixed_process_laws
from _lcm.regime_building.next_state import _resolve_conditioned_sigma
from _lcm.regime_building.processing import _get_weights_func_for_process
from lcm import (
    DiscreteGrid,
    InitialNodes,
    JointNormalIIDProcess,
    LogNormalIIDProcess,
    Model,
    NormalIIDProcess,
    NormalMixtureIIDProcess,
    Regime,
    RouwenhorstAR1Process,
    StateConditioned,
    TauchenAR1Process,
    TauchenNormalMixtureAR1Process,
    UniformIIDProcess,
    categorical,
)
from lcm.exceptions import GridInitializationError
from lcm.shocks import GaussHermite, Normal, Product, ShockGroup
from lcm.typing import ContinuousState, FloatND, ScalarInt
from tests.test_state_conditioned_model import _get_model, _params


def code_ordered_scales(
    *,
    categories: tuple[str, ...],
    codes: tuple[int, ...],
    by: Mapping[str, int | float],
) -> tuple[float, ...]:
    """Place each declared scale in its literal category-code slot (stdlib only)."""
    values = [0.0] * len(categories)
    for category, code in zip(categories, codes, strict=True):
        values[code] = float(by[category])
    return tuple(values)


def normal_bin_probabilities(
    *, nodes: tuple[float, ...], location: float, sigma: float
) -> tuple[float, ...]:
    """Integrate a normal law over midpoint bins with open outer tails (stdlib)."""
    cumulative = tuple(
        0.5 * (1.0 + math.erf(((a + b) / 2.0 - location) / (sigma * math.sqrt(2.0))))
        for a, b in itertools.pairwise(nodes)
    )
    interior = tuple(upper - lower for lower, upper in itertools.pairwise(cumulative))
    return (cumulative[0], *interior, 1.0 - cumulative[-1])


@categorical(ordered=True)
class _Risk:
    low: ScalarInt
    high: ScalarInt


@categorical(ordered=False)
class _Employment:
    low: ScalarInt
    high: ScalarInt


@categorical(ordered=False)
class _OnlyRegime:
    r: ScalarInt


@categorical(ordered=False)
class _TerminalRegime:
    terminal: ScalarInt


# Each process family, declared with integer-written law coefficients.
type _Kwargs = dict[str, int | float | bool]

_INTEGER_LAWS: dict[
    str, tuple[Callable[..., _ContinuousStochasticProcess], _Kwargs]
] = {
    "uniform": (UniformIIDProcess, {"n_points": 3, "start": 0, "stop": 2}),
    "normal_gauss_hermite": (
        NormalIIDProcess,
        {"n_points": 3, "gauss_hermite": True, "mu": 1, "sigma": 2},
    ),
    "normal_bins": (
        NormalIIDProcess,
        {"n_points": 5, "gauss_hermite": False, "mu": 1, "sigma": 2, "n_std": 3},
    ),
    "lognormal": (
        LogNormalIIDProcess,
        {"n_points": 5, "gauss_hermite": False, "mu": 0, "sigma": 1, "n_std": 2},
    ),
    "normal_mixture": (
        NormalMixtureIIDProcess,
        {
            "n_points": 5,
            "n_std": 2,
            "p1": 0.5,
            "mu1": -1,
            "sigma1": 1,
            "mu2": 1,
            "sigma2": 2,
        },
    ),
    "tauchen": (
        TauchenAR1Process,
        {
            "n_points": 5,
            "gauss_hermite": False,
            "mu": 1,
            "sigma": 1,
            "rho": 0,
            "n_std": 2,
        },
    ),
    "rouwenhorst": (
        RouwenhorstAR1Process,
        {"n_points": 5, "mu": 1, "sigma": 1, "rho": 0},
    ),
    "tauchen_mixture": (
        TauchenNormalMixtureAR1Process,
        {
            "n_points": 5,
            "mu": 1,
            "n_std": 2,
            "p1": 0.5,
            "mu1": -1,
            "sigma1": 1,
            "mu2": 1,
            "sigma2": 2,
            "rho": 0,
        },
    ),
}

_COUNTS = frozenset({"n_points"})


def _float_twin(kwargs: _Kwargs) -> _Kwargs:
    return {
        name: float(value) if type(value) is int and name not in _COUNTS else value
        for name, value in kwargs.items()
    }


@pytest.mark.parametrize("family", sorted(_INTEGER_LAWS))
def test_integer_written_law_is_its_float_twin(family: str) -> None:
    """Every law coefficient is stored as a float; nodes and rows agree exactly."""
    cls, kwargs = _INTEGER_LAWS[family]
    integer_law = cls(**kwargs)
    float_law = cls(**_float_twin(kwargs))

    for name in integer_law._param_field_names:
        assert type(getattr(integer_law, name)) is float, name
    assert repr(integer_law) == repr(float_law)
    assert all(value.dtype.kind == "f" for value in integer_law.params.values())
    np.testing.assert_array_equal(
        np.asarray(integer_law.get_gridpoints()),
        np.asarray(float_law.get_gridpoints()),
    )
    np.testing.assert_array_equal(
        np.asarray(integer_law.get_transition_probs()),
        np.asarray(float_law.get_transition_probs()),
    )


@pytest.mark.parametrize("value", [True, False])
def test_boolean_law_coefficient_is_refused(*, value: bool) -> None:
    with pytest.raises(GridInitializationError, match="mu"):
        NormalIIDProcess(n_points=3, gauss_hermite=True, mu=value, sigma=1.0)


def test_joint_normal_law_is_stored_as_floats() -> None:
    process = JointNormalIIDProcess(
        states=("a", "b"),
        n_points=3,
        gauss_hermite=False,
        mean=(0, 1),
        cov=((1, 0), (0, 2)),
        n_std=3,
    )
    assert isinstance(process.mean, tuple)
    assert isinstance(process.cov, tuple)
    assert process.mean == (0.0, 1.0)
    assert all(type(value) is float for value in process.mean)
    assert all(type(value) is float for row in process.cov for value in row)
    assert type(process.n_std) is float


def _by(
    *, low: float, high: float, reversed_order: bool, immutable: bool
) -> Mapping[str, float]:
    entries = [("low", low), ("high", high)]
    if reversed_order:
        entries.reverse()
    mapping = dict(entries)
    return MappingProxyType(mapping) if immutable else mapping


def _conditioned_process(
    *, family: Literal["normal_iid", "tauchen"], declaration: StateConditioned
) -> NormalIIDProcess | TauchenAR1Process:
    if family == "normal_iid":
        return NormalIIDProcess(
            n_points=5, gauss_hermite=False, mu=0.0, n_std=3.0, sigma=declaration
        )
    return TauchenAR1Process(
        n_points=5,
        gauss_hermite=False,
        mu=0.0,
        n_std=3.0,
        rho=0.5,
        sigma=declaration,
    )


@pytest.mark.parametrize(
    ("low", "high"), [(1, 2.0), (2.0, 1), (2, 0.5), (1, 2), (1.0, 2.0)]
)
@pytest.mark.parametrize("reversed_order", [False, True])
@pytest.mark.parametrize("immutable", [False, True])
@pytest.mark.parametrize("family", ["normal_iid", "tauchen"])
def test_conditioned_law_is_float_and_code_ordered_in_both_builders(
    *,
    low: float,
    high: float,
    reversed_order: bool,
    immutable: bool,
    family: Literal["normal_iid", "tauchen"],
) -> None:
    """Every stored scale is a float, ordered by category code, eager and jitted."""
    mapping = _by(
        low=low, high=high, reversed_order=reversed_order, immutable=immutable
    )
    declaration = StateConditioned(on="risk", by=mapping)
    process = _conditioned_process(family=family, declaration=declaration)
    risk = DiscreteGrid(category_class=_Risk)
    grids = {"risk": risk, "income": process}

    assert type(declaration.by["low"]) is float
    assert type(declaration.by["high"]) is float
    assert type(process.sigma) is float
    assert process.state_conditioned is declaration

    expected = code_ordered_scales(categories=("low", "high"), codes=(0, 1), by=mapping)
    actual = sigma_array_by_code(cond_grid=risk, by=declaration.by)
    np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))
    assert actual.dtype.kind == "f"

    resolved = _resolve_conditioned_sigma(grid=process, grids=grids)
    assert resolved is not None
    assert resolved[0] is declaration
    np.testing.assert_array_equal(np.asarray(resolved[1]), np.asarray(expected))

    weights = _get_weights_func_for_process(name="income", grid=process, grids=grids)
    nodes = np.asarray(process.get_gridpoints())
    source = 0.25
    location = 0.0 if family == "normal_iid" else 0.5 * source
    tolerance = 32 * np.finfo(nodes.dtype).eps
    for code, scale in ((_Risk.low, float(low)), (_Risk.high, float(high))):
        reference = normal_bin_probabilities(
            nodes=tuple(float(x) for x in nodes), location=location, sigma=scale
        )
        for operation in (weights, jax.jit(weights)):
            row = operation(income=jnp.asarray(source), risk=code)
            np.testing.assert_allclose(
                np.asarray(row), reference, rtol=tolerance, atol=tolerance
            )


def test_conditioned_scales_are_stored_read_only() -> None:
    supplied = {"low": 1, "high": 2}
    declaration = StateConditioned(on="risk", by=supplied)
    supplied["low"] = 5
    assert dict(declaration.by) == {"low": 1.0, "high": 2.0}
    with pytest.raises(TypeError):
        declaration.by["low"] = 3.0  # ty: ignore[invalid-assignment]


@pytest.mark.parametrize(
    "by", [{"low": 1.0, "high": 2.0}, {"low": 1, "high": 2}], ids=["float", "int"]
)
def test_state_conditioned_pickles_and_replaces(by: dict[str, float]) -> None:
    declaration = StateConditioned(on="emp", by=by)
    restored = pickle.loads(pickle.dumps(declaration))
    assert type(restored) is StateConditioned
    assert restored == declaration
    assert all(type(value) is float for value in restored.by.values())
    assert dataclasses.replace(declaration) == declaration


@pytest.mark.parametrize("value", [True, False])
def test_state_conditioned_refuses_a_boolean_scale(*, value: bool) -> None:
    with pytest.raises(GridInitializationError, match="real number"):
        StateConditioned(on="emp", by={"low": 1.0, "high": value})


def test_integer_conditioned_model_is_its_float_twin() -> None:
    """A public model built from integer scales solves to its float twin's values."""
    build = _get_model.__wrapped__
    integer = build(sigma_low=1, sigma_high=2, n_periods=3)
    floating = build(sigma_low=1.0, sigma_high=2.0, n_periods=3)
    assert integer._model_structure_fingerprint == floating._model_structure_fingerprint
    integer_values = integer.solve(params=_params(), log_level="off").values
    float_values = floating.solve(params=_params(), log_level="off").values
    for period, regimes in float_values.items():
        for regime, values in regimes.items():
            np.testing.assert_array_equal(
                np.asarray(integer_values[period][regime]), np.asarray(values)
            )


def _bound_shock_group() -> ShockGroup:
    group = ShockGroup(
        shocks={
            "eps": Normal(
                mu=0.0,
                sigma=StateConditioned(on="emp", by={"low": 1, "high": 2}),
            )
        },
        nodes=Product(eps=GaussHermite(n=3)),
    )
    return group.bind_conditioning(
        state_grids={"emp": DiscreteGrid(category_class=_Employment)}
    )


@pytest.mark.parametrize("jit", [False, True])
@pytest.mark.parametrize(("code", "expected"), [(0, 1.0), (1, 2.0)])
def test_conditioned_shock_reads_integer_category_codes(
    *, jit: bool, code: int, expected: float
) -> None:
    group = _bound_shock_group()

    def transform(category: ScalarInt) -> tuple[FloatND, FloatND]:
        result = group.transform(base_nodes=jnp.ones((1, 1)), inputs={"emp": category})
        return result.values["eps"], result.valid

    evaluate = jax.jit(transform) if jit else transform
    values, valid = evaluate(jnp.asarray(code, dtype=jnp.int32))
    assert bool(valid)
    np.testing.assert_array_equal(np.asarray(values), [expected])


def test_nonintegral_conditioning_code_stays_invalid() -> None:
    result = _bound_shock_group().transform(
        base_nodes=jnp.ones((1, 1)), inputs={"emp": jnp.asarray(0.5)}
    )
    assert not bool(result.valid)


def _shock_utility(*, shock: ContinuousState) -> FloatND:
    return shock * 0.0


@pytest.mark.parametrize("value", [2**31, -(2**31) - 1])
def test_fixed_process_law_is_float_before_any_integer_range_guard(
    value: int,
) -> None:
    process = NormalIIDProcess(n_points=3, gauss_hermite=True, mu=None, sigma=2.0**24)
    regime = Regime(states={"shock": process}, functions={"utility": _shock_utility})

    def bind(leaf: float) -> NormalIIDProcess:
        bound, _, _ = bind_fixed_process_laws(
            user_regimes=MappingProxyType({"r": regime}),
            fixed_params={"r": {"shock": {"mu": leaf}}},
        )
        law = bound["r"].states["shock"]
        assert isinstance(law, NormalIIDProcess)
        return law

    integer_law = bind(value)
    float_law = bind(float(value))
    assert type(integer_law.mu) is float
    assert integer_law.mu == float_law.mu
    np.testing.assert_array_equal(
        np.asarray(integer_law.get_gridpoints()), np.asarray(float_law.get_gridpoints())
    )


def _fixed_law_model(mu: float) -> Model:
    return Model(
        regimes={
            "r": Regime(
                states={
                    "shock": NormalIIDProcess(
                        n_points=3, gauss_hermite=True, sigma=2.0**24
                    )
                },
                functions={"utility": _shock_utility},
            )
        },
        edges={},
        n_periods=1,
        initial_nodes=InitialNodes(by_period={0: "r"}),
        regime_id_class=_OnlyRegime,
        fixed_params={"r": {"shock": {"mu": mu}}},
        enable_jit=False,
    )


@pytest.mark.parametrize("value", [1, 2**31, -(2**31) - 1])
def test_fixed_integer_law_model_is_its_float_twin(value: int) -> None:
    integer = _fixed_law_model(value)
    floating = _fixed_law_model(float(value))
    assert integer._model_structure_fingerprint == floating._model_structure_fingerprint


def _innovation_utility(*, innovation: FloatND) -> FloatND:
    return innovation


def _shock_model(fixed_mu: float) -> Model:
    group = ShockGroup(
        shocks={"innovation": Normal(mu=None, sigma=0.0)},
        nodes=Product(innovation=GaussHermite(1)),
    )
    return Model(
        regimes={
            "terminal": Regime(
                functions={"utility": _innovation_utility}, shocks=("shock_group",)
            )
        },
        n_periods=1,
        edges={},
        regime_id_class=_TerminalRegime,
        initial_nodes=InitialNodes(by_period={0: "terminal"}),
        fixed_params={"shock_group": {"innovation__mu": fixed_mu}},
        shocks={"shock_group": group},
    )


def test_fixed_shock_coefficient_has_one_float_identity() -> None:
    integer = _shock_model(1)
    floating = _shock_model(1.0)
    assert integer._model_structure_fingerprint == floating._model_structure_fingerprint
    np.testing.assert_array_equal(
        np.asarray(integer.solve(params={}, log_level="off").values[0]["terminal"]),
        np.asarray(floating.solve(params={}, log_level="off").values[0]["terminal"]),
    )
