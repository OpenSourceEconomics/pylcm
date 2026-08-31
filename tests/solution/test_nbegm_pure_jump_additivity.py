"""NBEGM's all-jump route requires an additive (unit-slope) budget.

The pure-jump step recovers the budget from per-interval intercepts assuming
cash-on-hand has unit slope in the liquid (Euler) state — the shape of an additive
subsidy/tax cliff. A budget with a non-unit liquid slope that declares only jump
breakpoints would be silently mis-solved, so it is refused. The
unit-slope check is scoped to exactly this path — liquid-direct, non-ride, all-jump
— so it does not touch the legitimate cases where a non-unit slope is expected:
derived-variable and ride-along schedules (whose asset-space slope is recovered by
the preimage machinery) and floored/clipped budgets (which carry a `continuous_kink`
and route to the mixed step).
"""

import itertools
from types import MappingProxyType
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.solution.nbegm import (
    _fail_if_budget_nonaffine_in_liquid,
    _NBEGMSource,
    _ProbeArguments,
)
from _lcm.solution.preconditions import check_solver_params
from lcm.exceptions import RegimeInitializationError
from lcm.model import Model
from tests.test_models import nbegm_jump_schedule_toy as toy


def _check_probes(*, model: Model, params: dict) -> None:
    """Run the solver's parameter-dependent preconditions, and nothing else."""
    check_solver_params(
        regimes=model._regimes,
        flat_params=model._process_params(params),
    )


def test_all_jump_schedule_with_a_non_unit_liquid_slope_is_refused() -> None:
    """A jump-only schedule over a non-additive budget is refused."""
    model = toy.build_model(
        variant="nbegm", non_additive=True, n_liquid=40, n_savings=40
    )

    with pytest.raises(RegimeInitializationError, match=r"slope|additive"):
        _check_probes(model=model, params=toy.build_params(non_additive=True))


@pytest.mark.parametrize("enable_x64", [False, True], ids=["fp32", "fp64"])
def test_all_jump_schedule_checks_unit_slope_inside_each_declared_interval(
    enable_x64: bool,  # noqa: FBT001
) -> None:
    """A non-unit branch above a parameter-dependent cliff is refused."""
    previous = jax.config.read("jax_enable_x64")
    jax.config.update("jax_enable_x64", enable_x64)
    try:
        model = toy.build_model(
            variant="nbegm",
            non_additive_above_cliff=True,
            n_liquid=5,
            liquid_max=20.0,
            n_savings=20,
        )
        params = toy.build_params(
            non_additive_above_cliff=True,
            cliff=10.0,
        )
        with pytest.raises(RegimeInitializationError, match=r"slope|additive"):
            _check_probes(model=model, params=params)
    finally:
        jax.config.update("jax_enable_x64", previous)


def test_schedule_over_a_nonlinear_budget_is_refused() -> None:
    """A budget that is smooth but not affine in the liquid state is refused.

    The per-interval affine segment is recovered from the budget's slope and value
    at one interior point, exact only for an affine budget; a curved budget is
    mis-tangented everywhere else, so the solve refuses it."""
    model = toy.build_model(variant="nbegm", nonlinear=True, n_liquid=40, n_savings=40)

    with pytest.raises(RegimeInitializationError, match=r"affine|second derivative"):
        _check_probes(model=model, params=toy.build_params(nonlinear=True))


def test_affinity_check_covers_the_budget_s_upper_grid_region() -> None:
    """Curvature confined above ten is still rejected on a grid reaching twenty."""
    model = toy.build_model(
        variant="nbegm",
        nonlinear_above_ten=True,
        n_liquid=40,
        liquid_max=20.0,
        n_savings=40,
    )

    with pytest.raises(RegimeInitializationError, match=r"affine|second derivative"):
        _check_probes(
            model=model,
            params=toy.build_params(nonlinear_above_ten=True, nonlinear=False),
        )


def test_all_jump_schedule_with_a_unit_liquid_slope_builds() -> None:
    """The additive jump-only schedule passes the deferred parameter check."""
    model = toy.build_model(
        variant="nbegm", non_additive=False, n_liquid=40, n_savings=40
    )
    _check_probes(model=model, params=toy.build_params())


def _scalar_interval_representatives(
    *,
    liquid_bounds: tuple[float, float],
    thresholds: tuple[tuple[float, str], ...],
    enable_x64: bool,
) -> tuple[float, ...] | None:
    """Independent represented-float partition oracle for literal thresholds."""
    scalar = np.float64 if enable_x64 else np.float32
    domain_start = scalar(liquid_bounds[0])
    domain_stop = scalar(liquid_bounds[1])
    internal = []
    for value, equality_owner in thresholds:
        boundary = scalar(value)
        if equality_owner == "below":
            boundary = scalar(np.nextafter(boundary, scalar(np.inf)))
        if domain_start < boundary < domain_stop and all(
            boundary != previous for previous in internal
        ):
            internal.append(boundary)
    internal.sort()

    representatives = []
    edges = (domain_start, *internal, domain_stop)
    for lower, upper in itertools.pairwise(edges):
        if not upper > lower:
            continue
        midpoint = scalar(scalar(0.5) * lower + scalar(0.5) * upper)
        if not lower < midpoint < upper:
            midpoint = scalar(np.nextafter(lower, upper))
        if not lower < midpoint < upper:
            return None
        representatives.append(float(midpoint))
    return tuple(representatives)


def _scalar_interval_oracle_accepts(
    *,
    liquid_bounds: tuple[float, float],
    thresholds: tuple[tuple[float, str], ...],
    coefficient_at,
    require_unit_slope: bool,
    enable_x64: bool,
    integer_codes: tuple[int, ...] = (0,),
) -> bool:
    """Decide from literal intervals and analytic branch coefficients only."""
    representatives = _scalar_interval_representatives(
        liquid_bounds=liquid_bounds,
        thresholds=thresholds,
        enable_x64=enable_x64,
    )
    if representatives is None:
        return False
    for code in integer_codes:
        for liquid in representatives:
            slope, second_derivative = coefficient_at(liquid, code)
            if second_derivative != 0.0:
                return False
            if require_unit_slope and slope != 1.0:
                return False
    return True


def _direct_source(
    *,
    threshold_name: str,
    equality_owner: Literal["below", "above"] = "above",
    kind: str = "jump",
) -> _NBEGMSource:
    """Build one direct-liquid source without invoking schedule collection."""
    return _NBEGMSource(
        variable="liquid",
        threshold_param_name=threshold_name,
        kind=kind,
        equality_owner=equality_owner,
        derived_of_liquid_dag=None,
        derived_param_names=(),
    )


def _production_probe_accepts(
    *,
    budget,
    liquid_grid: tuple[float, ...],
    params: dict[str, float],
    sources: tuple[_NBEGMSource, ...],
    require_unit_slope: bool,
    enable_x64: bool,
    integer_codes: dict[str, tuple[int, ...]] | None = None,
) -> bool:
    """Run the repaired production decision and return accept/refuse as a bool."""
    dtype = jnp.float64 if enable_x64 else jnp.float32
    grid = jnp.asarray(liquid_grid, dtype=dtype)
    midpoints = 0.5 * (grid[:-1] + grid[1:])
    probe_arguments = _ProbeArguments(
        int_arg_values=MappingProxyType(integer_codes or {}),
        param_values=MappingProxyType(
            {name: jnp.asarray(value, dtype=dtype) for name, value in params.items()}
        ),
    )
    try:
        _fail_if_budget_nonaffine_in_liquid(
            coh_dag=budget,
            liquid_name="liquid",
            require_unit_slope=require_unit_slope,
            regime_name="generated_interval_oracle",
            probe_arguments=probe_arguments,
            liquid_grid=grid,
            liquid_samples=jnp.sort(jnp.concatenate([grid, midpoints])),
            breakpoint_sources=sources,
        )
    except RegimeInitializationError:
        return False
    return True


def _evaluate_at_precision(
    *,
    enable_x64: bool,
    evaluate,
) -> bool:
    """Run one generated decision under the requested canonical JAX dtype."""
    previous = jax.config.read("jax_enable_x64")
    jax.config.update("jax_enable_x64", enable_x64)
    try:
        return evaluate()
    finally:
        jax.config.update("jax_enable_x64", previous)


@pytest.mark.parametrize("enable_x64", [False, True], ids=["fp32", "fp64"])
@pytest.mark.parametrize(
    "threshold_values",
    [
        (8.875, 9.125),
        (9.125, 8.875),
        (13.625, 14.125),
        (14.125, 13.625),
    ],
    ids=["low", "low-permuted", "high", "high-permuted"],
)
def test_generated_breakpoint_positions_and_permutations_match_scalar_oracle(
    *,
    enable_x64: bool,
    threshold_values: tuple[float, float],
) -> None:
    """A narrow non-unit cell is found regardless of threshold value or order."""
    left, right = threshold_values
    lower, upper = sorted(threshold_values)

    def budget(*, liquid, threshold_left, threshold_right):
        lo = jnp.minimum(threshold_left, threshold_right)
        hi = jnp.maximum(threshold_left, threshold_right)
        return jnp.where(
            liquid < lo,
            liquid + 3.0,
            jnp.where(liquid < hi, 2.0 * liquid - 4.0, liquid + 1.0),
        )

    def coefficient_at(*, liquid: float, _code: int) -> tuple[float, float]:
        slope = 1.0 if liquid < lower or liquid >= upper else 2.0
        return slope, 0.0

    oracle = _scalar_interval_oracle_accepts(
        liquid_bounds=(0.0, 20.0),
        thresholds=((left, "above"), (right, "above")),
        coefficient_at=coefficient_at,
        require_unit_slope=True,
        enable_x64=enable_x64,
    )
    production = _evaluate_at_precision(
        enable_x64=enable_x64,
        evaluate=lambda: _production_probe_accepts(
            budget=budget,
            liquid_grid=(0.0, 5.0, 10.0, 15.0, 20.0),
            params={"threshold_left": left, "threshold_right": right},
            sources=(
                _direct_source(threshold_name="threshold_left"),
                _direct_source(threshold_name="threshold_right"),
            ),
            require_unit_slope=True,
            enable_x64=enable_x64,
        ),
    )

    assert oracle is False
    assert production is oracle


@pytest.mark.parametrize("enable_x64", [False, True], ids=["fp32", "fp64"])
def test_generated_narrow_curved_interval_matches_scalar_oracle(
    enable_x64: bool,  # noqa: FBT001
) -> None:
    """Curvature confined between fixed grid probes is still refused."""
    lower = 8.875
    upper = 9.125

    def budget(*, liquid, lower_threshold, upper_threshold):
        inside = (liquid > lower_threshold) & (liquid < upper_threshold)
        curved = liquid + 0.25 * (liquid - lower_threshold) ** 2
        return jnp.where(inside, curved, liquid)

    def coefficient_at(*, liquid: float, _code: int) -> tuple[float, float]:
        return (1.0, 0.5) if lower < liquid < upper else (1.0, 0.0)

    oracle = _scalar_interval_oracle_accepts(
        liquid_bounds=(0.0, 20.0),
        thresholds=((lower, "above"), (upper, "above")),
        coefficient_at=coefficient_at,
        require_unit_slope=False,
        enable_x64=enable_x64,
    )
    production = _evaluate_at_precision(
        enable_x64=enable_x64,
        evaluate=lambda: _production_probe_accepts(
            budget=budget,
            liquid_grid=(0.0, 5.0, 10.0, 15.0, 20.0),
            params={"lower_threshold": lower, "upper_threshold": upper},
            sources=(
                _direct_source(threshold_name="lower_threshold"),
                _direct_source(threshold_name="upper_threshold"),
            ),
            require_unit_slope=False,
            enable_x64=enable_x64,
        ),
    )

    assert oracle is False
    assert production is oracle


@pytest.mark.parametrize("enable_x64", [False, True], ids=["fp32", "fp64"])
def test_generated_discrete_code_branch_matches_scalar_oracle(
    enable_x64: bool,  # noqa: FBT001
) -> None:
    """A non-unit branch live only at an actual discrete code is refused."""
    cliff = 10.25

    def budget(*, liquid, cliff_threshold, branch_code):
        bad_branch = (branch_code == 7) & (liquid >= cliff_threshold)
        return jnp.where(bad_branch, 2.0 * liquid, liquid)

    def coefficient_at(*, liquid: float, code: int) -> tuple[float, float]:
        return (2.0, 0.0) if code == 7 and liquid >= cliff else (1.0, 0.0)

    oracle = _scalar_interval_oracle_accepts(
        liquid_bounds=(0.0, 20.0),
        thresholds=((cliff, "above"),),
        coefficient_at=coefficient_at,
        require_unit_slope=True,
        enable_x64=enable_x64,
        integer_codes=(0, 7),
    )
    production = _evaluate_at_precision(
        enable_x64=enable_x64,
        evaluate=lambda: _production_probe_accepts(
            budget=budget,
            liquid_grid=(0.0, 5.0, 10.0, 15.0, 20.0),
            params={"cliff_threshold": cliff},
            sources=(_direct_source(threshold_name="cliff_threshold"),),
            require_unit_slope=True,
            enable_x64=enable_x64,
            integer_codes={"branch_code": (0, 7)},
        ),
    )

    assert oracle is False
    assert production is oracle


@pytest.mark.parametrize("enable_x64", [False, True], ids=["fp32", "fp64"])
def test_endpoint_adjacent_interval_matches_scalar_oracle(
    enable_x64: bool,  # noqa: FBT001
) -> None:
    """A one-ULP cell with no strict represented interior is uncertifiable."""
    scalar = np.float64 if enable_x64 else np.float32
    threshold = float(np.nextafter(scalar(1.0), scalar(21.0)))

    def budget(*, liquid, cliff_threshold):
        del cliff_threshold
        return liquid

    oracle = _scalar_interval_oracle_accepts(
        liquid_bounds=(1.0, 21.0),
        thresholds=((threshold, "above"),),
        coefficient_at=lambda _liquid, _code: (1.0, 0.0),
        require_unit_slope=True,
        enable_x64=enable_x64,
    )
    production = _evaluate_at_precision(
        enable_x64=enable_x64,
        evaluate=lambda: _production_probe_accepts(
            budget=budget,
            liquid_grid=(1.0, 6.0, 11.0, 16.0, 21.0),
            params={"cliff_threshold": threshold},
            sources=(_direct_source(threshold_name="cliff_threshold"),),
            require_unit_slope=True,
            enable_x64=enable_x64,
        ),
    )

    assert oracle is False
    assert production is oracle


@pytest.mark.parametrize("enable_x64", [False, True], ids=["fp32", "fp64"])
@pytest.mark.parametrize(
    ("require_unit_slope", "slopes", "kinds"),
    [
        (True, (1.0, 1.0, 1.0), ("jump", "jump")),
        (False, (0.8, 1.2, 1.5), ("jump", "continuous_kink")),
    ],
    ids=["legal-all-jump", "legal-mixed"],
)
def test_legal_piecewise_affine_routes_match_scalar_oracle(
    *,
    enable_x64: bool,
    require_unit_slope: bool,
    slopes: tuple[float, float, float],
    kinds: tuple[str, str],
) -> None:
    """Unit jump and non-unit mixed routes keep their distinct slope contracts."""
    lower = 8.875
    upper = 9.125

    def budget(*, liquid, lower_threshold, upper_threshold):
        return jnp.where(
            liquid < lower_threshold,
            slopes[0] * liquid + 3.0,
            jnp.where(
                liquid < upper_threshold,
                slopes[1] * liquid + 2.0,
                slopes[2] * liquid + 1.0,
            ),
        )

    def coefficient_at(*, liquid: float, _code: int) -> tuple[float, float]:
        position = 0 if liquid < lower else 1 if liquid < upper else 2
        return slopes[position], 0.0

    oracle = _scalar_interval_oracle_accepts(
        liquid_bounds=(0.0, 20.0),
        thresholds=((lower, "above"), (upper, "above")),
        coefficient_at=coefficient_at,
        require_unit_slope=require_unit_slope,
        enable_x64=enable_x64,
    )
    production = _evaluate_at_precision(
        enable_x64=enable_x64,
        evaluate=lambda: _production_probe_accepts(
            budget=budget,
            liquid_grid=(0.0, 5.0, 10.0, 15.0, 20.0),
            params={"lower_threshold": lower, "upper_threshold": upper},
            sources=(
                _direct_source(threshold_name="lower_threshold", kind=kinds[0]),
                _direct_source(threshold_name="upper_threshold", kind=kinds[1]),
            ),
            require_unit_slope=require_unit_slope,
            enable_x64=enable_x64,
        ),
    )

    assert oracle is True
    assert production is oracle


@pytest.mark.parametrize("enable_x64", [False, True], ids=["fp32", "fp64"])
def test_legal_non_unit_derived_ride_route_matches_scalar_oracle(
    enable_x64: bool,  # noqa: FBT001
) -> None:
    """A derived ride-cell schedule may remain affine with a non-unit asset slope."""
    threshold = 10.0
    scale = 2.0
    codes = (0, 1)

    def derived_of_liquid(*, liquid, scale, ride_code):
        return scale * liquid + ride_code

    def budget(*, liquid, threshold, scale, ride_code):
        transfer = jnp.where(
            derived_of_liquid(liquid=liquid, scale=scale, ride_code=ride_code)
            < threshold,
            3.0,
            1.0,
        )
        return 1.4 * liquid + transfer

    oracle = all(
        _scalar_interval_oracle_accepts(
            liquid_bounds=(0.0, 20.0),
            thresholds=(((threshold - code) / scale, "above"),),
            coefficient_at=lambda _liquid, _ignored: (1.4, 0.0),
            require_unit_slope=False,
            enable_x64=enable_x64,
        )
        for code in codes
    )
    source = _NBEGMSource(
        variable="derived",
        threshold_param_name="threshold",
        kind="jump",
        equality_owner="above",
        derived_of_liquid_dag=derived_of_liquid,
        derived_param_names=("scale",),
        derived_state_names=("ride_code",),
    )
    production = _evaluate_at_precision(
        enable_x64=enable_x64,
        evaluate=lambda: _production_probe_accepts(
            budget=budget,
            liquid_grid=(0.0, 5.0, 10.0, 15.0, 20.0),
            params={"threshold": threshold, "scale": scale},
            sources=(source,),
            require_unit_slope=False,
            enable_x64=enable_x64,
            integer_codes={"ride_code": codes},
        ),
    )

    assert oracle is True
    assert production is oracle
