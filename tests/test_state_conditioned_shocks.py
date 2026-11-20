"""Tests for the state-conditioned-shock direct-CDF row builders.

Pins the load-bearing correctness of the kernel: reduction to pylcm's own transition
math at the nodes, rows built directly at the time-t value rather than interpolated,
sigma genuinely moving the kernel on fixed nodes, and analytic conditional moments.
"""

import jax.numpy as jnp
import numpy as np
import pytest
from scipy.stats import norm

from _lcm.processes.state_conditioned import (
    conditioned_row,
    gather_sigma,
    iid_normal_row,
    sigma_array_by_code,
    tauchen_row,
)
from _lcm.regime_building.processing import _get_weights_func_for_process
from lcm import (
    DiscreteGrid,
    NormalIIDProcess,
    TauchenAR1Process,
    TauchenNormalMixtureAR1Process,
    categorical,
)
from lcm.exceptions import GridInitializationError
from lcm.processes import StateConditioned
from lcm.typing import ScalarInt
from tests.conftest import X64_ENABLED

# The `*_reduces_to_pylcm*` tests compare our row against pylcm's OWN CDF-binned
# row, so any gap is floating-point noise in shared arithmetic, not a modelling
# difference. Under `--precision 32` that gap is a few float32 ULPs on O(1)
# probabilities (measured max |diff| ~3e-7, eps = 1.2e-7), which 1e-10 cannot express.
_ROW_ATOL = 1e-10 if X64_ENABLED else 1e-6


@categorical(ordered=True)
class Uncertainty:
    low: ScalarInt
    high: ScalarInt


def test_tauchen_row_reduces_to_pylcm_at_nodes() -> None:
    """At a node with sigma == grid_sigma, tauchen_row == the pylcm Tauchen row."""
    p = TauchenAR1Process(
        n_points=7, gauss_hermite=False, rho=0.9, sigma=0.2, mu=0.0, n_std=3.0
    )
    nodes = p.get_gridpoints()
    P = p.get_transition_probs()
    for i in range(7):
        row = tauchen_row(nodes=nodes, rho=0.9, sigma=0.2, from_value=float(nodes[i]))
        np.testing.assert_allclose(np.asarray(row), np.asarray(P[i]), atol=_ROW_ATOL)


def test_iid_row_reduces_to_pylcm() -> None:
    """iid_normal_row == the pylcm CDF-binned NormalIID row (rows identical)."""
    p = NormalIIDProcess(n_points=7, gauss_hermite=False, mu=0.0, sigma=0.2, n_std=3.0)
    nodes = p.get_gridpoints()
    P = p.get_transition_probs()
    row = iid_normal_row(nodes=nodes, mu=0.0, sigma=0.2)
    np.testing.assert_allclose(np.asarray(row), np.asarray(P[0]), atol=_ROW_ATOL)


@pytest.mark.parametrize("mu", [0.2, -0.5])
def test_tauchen_row_reduces_to_pylcm_at_nodes_with_nonzero_mu(mu: float) -> None:
    """The AR(1) intercept survives into the row at a non-zero mu.

    Stock pylcm builds its Tauchen rows in demeaned coordinates, where the intercept
    drops out. Here the nodes and the time-t value are physical (centred on mu/(1-rho)),
    so the conditional mean is `mu + rho*y_t` and dropping `mu` misplaces every row.
    """
    p = TauchenAR1Process(
        n_points=7, gauss_hermite=False, rho=0.9, sigma=0.2, mu=mu, n_std=3.0
    )
    nodes = p.get_gridpoints()
    P = p.get_transition_probs()
    for i in range(7):
        row = tauchen_row(
            nodes=nodes, rho=0.9, sigma=0.2, from_value=float(nodes[i]), mu=mu
        )
        np.testing.assert_allclose(np.asarray(row), np.asarray(P[i]), atol=_ROW_ATOL)


@pytest.mark.parametrize("mu", [1.0, -0.3])
def test_iid_row_reduces_to_pylcm_with_nonzero_mu(mu: float) -> None:
    """A non-zero IID mean survives into the row."""
    p = NormalIIDProcess(n_points=7, gauss_hermite=False, mu=mu, sigma=0.2, n_std=3.0)
    nodes = p.get_gridpoints()
    P = p.get_transition_probs()
    row = iid_normal_row(nodes=nodes, mu=mu, sigma=0.2)
    np.testing.assert_allclose(np.asarray(row), np.asarray(P[0]), atol=_ROW_ATOL)


def test_dropping_mu_would_misplace_the_row() -> None:
    """Pins WHY F2 mattered: at mu=1, sigma=0.2 the mu=0 row is a different law that
    reverses a continuation-vs-sure choice paying 1 on the middle node against 0.5."""
    nodes = jnp.array([0.0, 1.0, 2.0])
    correct = np.asarray(iid_normal_row(nodes=nodes, mu=1.0, sigma=0.2))
    dropped = np.asarray(iid_normal_row(nodes=nodes, mu=0.0, sigma=0.2))
    tv = 0.5 * np.abs(correct - dropped).sum()
    assert tv > 0.9
    assert dropped[1] < 0.5 < correct[1]


@pytest.mark.parametrize("sigma", [0.05, 0.2, 1.0])
def test_rows_are_distributions(sigma: float) -> None:
    nodes = jnp.linspace(-3.0, 3.0, 9)
    r = np.asarray(tauchen_row(nodes=nodes, rho=0.8, sigma=sigma, from_value=0.3))
    assert r.sum() == pytest.approx(1.0, abs=1e-12)
    assert (r >= -1e-12).all()


def test_state_conditioning_moves_sigma_on_fixed_nodes() -> None:
    """F2-safety: on the SAME fixed nodes, two sigmas give genuinely different variances
    (the failure Rouwenhorst has, which is why it is excluded)."""
    nodes = jnp.linspace(-3.0, 3.0, 41)  # fixed common grid (grid_sigma wide)
    r_lo = np.asarray(iid_normal_row(nodes=nodes, mu=0.0, sigma=0.2))
    r_hi = np.asarray(iid_normal_row(nodes=nodes, mu=0.0, sigma=1.0))
    g = np.asarray(nodes)
    v_lo = (g**2 * r_lo).sum()
    v_hi = (g**2 * r_hi).sum()
    assert v_hi > 4 * v_lo


def test_f1_direct_cdf_is_not_row_interpolation_at_low_sigma() -> None:
    """F1: the direct row differs O(1) from interpolating node rows at low sigma — this
    builder avoids the defect *by construction* (it evaluates the CDF at from_value)."""
    nodes = jnp.array([-1.0, 0.0, 1.0])
    y, rho, sigma = 0.52, 0.9, 0.01
    direct = np.asarray(tauchen_row(nodes=nodes, rho=rho, sigma=sigma, from_value=y))
    K = np.vstack(
        [
            np.asarray(
                tauchen_row(nodes=nodes, rho=rho, sigma=sigma, from_value=float(x))
            )
            for x in nodes
        ]
    )
    interp = 0.48 * K[1] + 0.52 * K[2]
    tv = 0.5 * np.abs(direct - interp).sum()
    assert tv > 0.5
    assert direct[-1] < 0.25 < interp[-1]  # interpolation would reverse a decision


def test_conditional_mean_matches_rho_y_on_fine_grid() -> None:
    """Analytic anchor: on a fine grid the Tauchen conditional mean -> rho * y."""
    nodes = jnp.linspace(-6.0, 6.0, 401)
    y, rho, sigma = 0.5, 0.8, 0.3
    r = np.asarray(tauchen_row(nodes=nodes, rho=rho, sigma=sigma, from_value=y))
    mean = (np.asarray(nodes) * r).sum()
    assert abs(mean - rho * y) < 1e-3


def test_iid_conditional_variance_matches_sigma_on_fine_grid() -> None:
    """Analytic anchor: on a fine grid the IID binned variance -> sigma**2."""
    nodes = jnp.linspace(-8.0, 8.0, 801)
    sigma = 0.7
    r = np.asarray(iid_normal_row(nodes=nodes, mu=0.0, sigma=sigma))
    g = np.asarray(nodes)
    var = (g**2 * r).sum() - (g * r).sum() ** 2
    assert abs(var - sigma**2) < 1e-3


# --- StateConditioned value object + code-ordered sigma resolver ------------- #


def test_sigma_array_ordered_by_code_not_insertion() -> None:
    """Sigma is ordered by the categorical's integer code, not by `by` insertion order.

    Indexing by the conditioning state's code is therefore correct whatever order the
    mapping is written in.
    """
    grid = DiscreteGrid(category_class=Uncertainty)  # codes: low=0, high=1
    by = {"high": 1.0, "low": 0.2}  # deliberately reverse insertion order
    arr = np.asarray(sigma_array_by_code(cond_grid=grid, by=by))
    np.testing.assert_allclose(arr, [0.2, 1.0])  # [code0=low, code1=high]
    assert float(gather_sigma(sigma_by_code=jnp.asarray(arr), code=0)) == pytest.approx(
        0.2
    )
    assert float(gather_sigma(sigma_by_code=jnp.asarray(arr), code=1)) == pytest.approx(
        1.0
    )


def test_sigma_array_missing_category_raises() -> None:
    grid = DiscreteGrid(category_class=Uncertainty)
    with pytest.raises(ValueError, match="missing categories"):
        sigma_array_by_code(cond_grid=grid, by={"low": 0.2})


def test_state_conditioned_dataclass_is_frozen() -> None:
    sc = StateConditioned(on="uncertainty", by={"low": 0.2, "high": 1.0})
    assert sc.on == "uncertainty"
    with pytest.raises((AttributeError, TypeError)):
        sc.on = "other"  # ty: ignore[invalid-assignment]


def test_conditioned_row_dispatch_sums_to_one() -> None:
    nodes = jnp.linspace(-3.0, 3.0, 7)
    r_iid = conditioned_row(
        family="iid_normal", nodes=nodes, sigma=0.5, from_value=0.0, mu=0.0
    )
    r_tau = conditioned_row(
        family="tauchen", nodes=nodes, sigma=0.5, from_value=0.3, mu=0.0, rho=0.8
    )
    assert np.asarray(r_iid).sum() == pytest.approx(1.0)
    assert np.asarray(r_tau).sum() == pytest.approx(1.0)


def test_conditioned_row_tauchen_requires_rho() -> None:
    nodes = jnp.linspace(-3.0, 3.0, 7)
    with pytest.raises(ValueError, match="requires rho"):
        conditioned_row(
            family="tauchen", nodes=nodes, sigma=0.5, from_value=0.0, mu=0.0
        )


def test_conditioned_sigma_places_nodes_from_the_widest_category() -> None:
    """A conditioned `sigma` places the nodes from its widest per-category value."""
    conditioned = NormalIIDProcess(
        n_points=7,
        gauss_hermite=False,
        mu=0.0,
        n_std=3.0,
        sigma=StateConditioned(on="uncertainty", by={"low": 0.2, "high": 0.5}),
    )
    explicit = NormalIIDProcess(
        n_points=7, gauss_hermite=False, mu=0.0, sigma=0.5, n_std=3.0
    )
    assert conditioned.sigma == pytest.approx(0.5)
    np.testing.assert_allclose(
        np.asarray(conditioned.get_gridpoints()), np.asarray(explicit.get_gridpoints())
    )


@pytest.mark.parametrize(
    "process",
    [
        NormalIIDProcess(
            n_points=7,
            gauss_hermite=False,
            mu=0.0,
            n_std=3.0,
            sigma=StateConditioned(on="uncertainty", by={"low": 0.2, "high": 0.5}),
        ),
        TauchenAR1Process(
            n_points=7,
            gauss_hermite=False,
            rho=0.9,
            mu=0.0,
            n_std=3.0,
            sigma=StateConditioned(on="uncertainty", by={"low": 0.2, "high": 0.5}),
        ),
    ],
)
def test_state_conditioned_process_has_no_parameterless_transition_matrix(
    process: NormalIIDProcess | TauchenAR1Process,
) -> None:
    """A conditioned process needs its conditioning state to select a transition law."""
    with pytest.raises(
        GridInitializationError,
        match=r"get_transition_probs.*depends on.*uncertainty",
    ):
        process.get_transition_probs()


def test_conditioned_sigma_is_readable_as_the_conditioning_declaration() -> None:
    """The declaration stays reachable after the scalar node-placing value is set."""
    process = NormalIIDProcess(
        n_points=7,
        gauss_hermite=False,
        mu=0.0,
        n_std=3.0,
        sigma=StateConditioned(on="uncertainty", by={"low": 0.2, "high": 0.5}),
    )
    assert process.state_conditioned == StateConditioned(
        on="uncertainty", by={"low": 0.2, "high": 0.5}
    )


def test_conditioning_declaration_cannot_be_passed_as_its_own_argument() -> None:
    """Conditioning is expressed on the parameter, so there is no separate slot."""
    with pytest.raises(TypeError, match="state_conditioned"):
        NormalIIDProcess(
            n_points=7,
            gauss_hermite=False,
            mu=0.0,
            n_std=3.0,
            state_conditioned=StateConditioned(  # ty: ignore[unknown-argument]
                on="uncertainty", by={"low": 0.2, "high": 0.5}
            ),
        )


@pytest.mark.parametrize(
    "by",
    [
        {"low": 0.2, "high": -0.5},
        {"low": 0.2, "high": 0.0},
        {"low": 0.2, "high": float("inf")},
        {"low": 0.2, "high": float("nan")},
        {},
    ],
)
def test_conditioned_sigma_rejects_an_unusable_category_value(
    by: dict[str, float],
) -> None:
    """Every per-category value must be a finite positive scale."""
    with pytest.raises(GridInitializationError, match="finite positive"):
        NormalIIDProcess(
            n_points=7,
            gauss_hermite=False,
            mu=0.0,
            n_std=3.0,
            sigma=StateConditioned(on="uncertainty", by=by),
        )


def test_conditioned_sigma_places_nodes_for_tauchen_too() -> None:
    """Node placement from the widest value is not specific to the IID family."""
    p = TauchenAR1Process(
        n_points=7,
        gauss_hermite=False,
        rho=0.9,
        mu=0.0,
        n_std=3.0,
        sigma=StateConditioned(on="uncertainty", by={"low": 0.1, "high": 0.4}),
    )
    assert p.sigma == pytest.approx(0.4)


def _on_uncertainty(*, low: float, high: float) -> StateConditioned:
    return StateConditioned(on="uncertainty", by={"low": low, "high": high})


def _tauchen(
    *,
    mu: float | StateConditioned,
    rho: float | StateConditioned,
    sigma: float | StateConditioned,
) -> TauchenAR1Process:
    return TauchenAR1Process(
        n_points=7, gauss_hermite=False, n_std=3.0, mu=mu, rho=rho, sigma=sigma
    )


def test_conditioned_parameters_keep_their_declarations() -> None:
    """Every conditioned parameter is listed by name, in the order of the fields."""
    mu, rho = _on_uncertainty(low=0.1, high=0.3), _on_uncertainty(low=0.9, high=0.5)
    process = _tauchen(mu=mu, rho=rho, sigma=0.2)
    assert process.conditioned_params == (("rho", rho), ("mu", mu))
    assert process.state_conditioned == rho


def test_unconditioned_process_declares_no_conditioned_parameters() -> None:
    """A scalar law carries no conditioning."""
    assert _tauchen(mu=0.1, rho=0.9, sigma=0.2).conditioned_params == ()


def test_only_sigma_conditioned_keeps_the_widest_categorys_nodes_exactly() -> None:
    """With `sigma` alone conditioned, the nodes are the widest category's own."""
    conditioned = _tauchen(mu=0.1, rho=0.9, sigma=_on_uncertainty(low=0.2, high=0.5))
    widest = _tauchen(mu=0.1, rho=0.9, sigma=0.5)
    np.testing.assert_array_equal(
        np.asarray(conditioned.get_gridpoints()), np.asarray(widest.get_gridpoints())
    )


def test_conditioned_nodes_span_every_categorys_own_nodes() -> None:
    """The nodes run evenly from the lowest to the highest category node."""
    conditioned = _tauchen(
        mu=_on_uncertainty(low=0.1, high=1.5),
        rho=_on_uncertainty(low=0.9, high=0.5),
        sigma=0.2,
    )
    low = _tauchen(mu=0.1, rho=0.9, sigma=0.2).get_gridpoints()
    high = _tauchen(mu=1.5, rho=0.5, sigma=0.2).get_gridpoints()
    expected = jnp.linspace(
        jnp.minimum(low[0], high[0]), jnp.maximum(low[-1], high[-1]), 7
    )
    np.testing.assert_array_equal(
        np.asarray(conditioned.get_gridpoints()), np.asarray(expected)
    )


def test_conditioned_parameters_must_share_their_conditioning_state() -> None:
    """One process is conditioned on one state."""
    with pytest.raises(GridInitializationError, match="same state"):
        _tauchen(
            mu=_on_uncertainty(low=0.1, high=0.3),
            rho=StateConditioned(on="health", by={"low": 0.9, "high": 0.5}),
            sigma=0.2,
        )


def test_conditioned_parameters_must_name_the_same_categories() -> None:
    """Every conditioned parameter gives a value for the same categories."""
    with pytest.raises(GridInitializationError, match="same categories"):
        _tauchen(
            mu=_on_uncertainty(low=0.1, high=0.3),
            rho=StateConditioned(on="uncertainty", by={"low": 0.9, "mid": 0.5}),
            sigma=0.2,
        )


def test_each_categorys_law_is_checked_like_a_scalar_law() -> None:
    """A category whose persistence has no stationary law is refused."""
    with pytest.raises(GridInitializationError, match="no stationary distribution"):
        _tauchen(mu=0.0, rho=_on_uncertainty(low=-1.5, high=0.5), sigma=0.2)


def test_processes_conditioned_on_different_values_are_different_laws() -> None:
    """Two declarations that agree only in their widest value are not one law."""
    first = _tauchen(mu=0.0, rho=0.9, sigma=_on_uncertainty(low=0.1, high=0.5))
    second = _tauchen(mu=0.0, rho=0.9, sigma=_on_uncertainty(low=0.3, high=0.5))
    assert first.has_same_law_as(first)
    assert not first.has_same_law_as(second)


def _conditioned_row_at(
    *,
    process: NormalIIDProcess | TauchenAR1Process | TauchenNormalMixtureAR1Process,
    code: int,
    from_value: float,
) -> np.ndarray:
    weights = _get_weights_func_for_process(
        name="income",
        grid=process,
        grids={"uncertainty": DiscreteGrid(category_class=Uncertainty)},
    )
    return np.asarray(
        weights(income=jnp.asarray(from_value), uncertainty=jnp.int32(code))
    )


@pytest.mark.parametrize("code", [0, 1])
def test_iid_row_uses_the_conditioned_mean(code: int) -> None:
    """The row of a category bins a normal centred on that category's `mu`."""
    process = NormalIIDProcess(
        n_points=7,
        gauss_hermite=False,
        n_std=3.0,
        mu=_on_uncertainty(low=-0.2, high=0.4),
        sigma=0.3,
    )
    expected = iid_normal_row(
        nodes=process.get_gridpoints(), mu=(-0.2, 0.4)[code], sigma=0.3
    )
    np.testing.assert_allclose(
        _conditioned_row_at(process=process, code=code, from_value=0.0),
        np.asarray(expected),
        atol=_ROW_ATOL,
    )


@pytest.mark.parametrize("code", [0, 1])
def test_tauchen_row_uses_the_conditioned_intercept_and_persistence(code: int) -> None:
    """The row of a category is the AR(1) row at that category's `mu` and `rho`."""
    process = _tauchen(
        mu=_on_uncertainty(low=0.1, high=0.3),
        rho=_on_uncertainty(low=0.9, high=0.5),
        sigma=0.2,
    )
    expected = tauchen_row(
        nodes=process.get_gridpoints(),
        mu=(0.1, 0.3)[code],
        rho=(0.9, 0.5)[code],
        sigma=0.2,
        from_value=0.7,
    )
    np.testing.assert_allclose(
        _conditioned_row_at(process=process, code=code, from_value=0.7),
        np.asarray(expected),
        atol=_ROW_ATOL,
    )


_MIXTURE_BY_CODE = (
    {
        "mu": 0.10,
        "rho": 0.90,
        "p1": 0.8,
        "mu1": 0.0,
        "sigma1": 0.10,
        "mu2": -0.2,
        "sigma2": 0.40,
    },
    {
        "mu": 0.35,
        "rho": 0.95,
        "p1": 0.9,
        "mu1": 0.0,
        "sigma1": 0.07,
        "mu2": -0.1,
        "sigma2": 0.30,
    },
)


@pytest.mark.parametrize("code", [0, 1])
def test_mixture_row_uses_every_conditioned_parameter(code: int) -> None:
    """The row of a category bins that category's mixture at the node midpoints."""
    by = {
        name: _on_uncertainty(
            low=_MIXTURE_BY_CODE[0][name], high=_MIXTURE_BY_CODE[1][name]
        )
        for name in _MIXTURE_BY_CODE[0]
    }
    process = TauchenNormalMixtureAR1Process(
        n_points=7,
        n_std=2.0,
        mu=by["mu"],
        rho=by["rho"],
        p1=by["p1"],
        mu1=by["mu1"],
        sigma1=by["sigma1"],
        mu2=by["mu2"],
        sigma2=by["sigma2"],
    )
    from_value = 1.5
    law = _MIXTURE_BY_CODE[code]
    nodes = np.asarray(process.get_gridpoints(), dtype=np.float64)
    innovation = (nodes[:-1] + nodes[1:]) / 2 - law["mu"] - law["rho"] * from_value
    cdf = law["p1"] * norm.cdf(innovation, law["mu1"], law["sigma1"]) + (
        1 - law["p1"]
    ) * norm.cdf(innovation, law["mu2"], law["sigma2"])
    expected = np.diff(np.concatenate([[0.0], cdf, [1.0]]))
    np.testing.assert_allclose(
        _conditioned_row_at(process=process, code=code, from_value=from_value),
        expected,
        atol=_ROW_ATOL,
    )
