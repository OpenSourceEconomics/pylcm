"""RED regression: exact owner stability must cover the parameter direction."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from _lcm.optimization.implicit_outer_derivative import (
    OwnerProvenance,
    implicit_optimum_diagnostics,
)


def _branches(*, f, theta, center, slope):
    q0 = -((f - (center + slope * theta)) ** 2) + theta
    q1 = -((f - (center - slope * theta)) ** 2) - theta
    return q0, q1


@pytest.mark.parametrize("center", [0.2, 0.3, 0.7])
@pytest.mark.parametrize("slope", [0.25, 1.0, 2.0])
def test_parameter_direction_switch_is_unresolved(
    *, center: float, slope: float
) -> None:
    """A stable-index baseline owner cannot certify a two-sided parameter tangent."""

    def objective(*, f, theta):
        q0, q1 = _branches(f=f, theta=theta, center=center, slope=slope)
        return jnp.maximum(q0, q1)

    def branch_id(*, f, theta):
        q0, q1 = _branches(f=f, theta=theta, center=center, slope=slope)
        return jnp.where(q0 >= q1, 0, 1)

    theta0 = jnp.asarray(0.0)
    f_star = jnp.asarray(center)
    diag = implicit_optimum_diagnostics(
        objective=objective,
        theta=theta0,
        f_star=f_star,
        basin_margin=jnp.asarray(1.0),
        bounds=(jnp.asarray(0.0), jnp.asarray(1.0)),
        branch_id=branch_id,
    )

    # At theta=0 the deterministic stable-index owner remains constant across
    # the existing action probes, while theta+ and theta- select opposite
    # branches.  The one-sided argmax derivatives are +slope and -slope, so no
    # two-sided derivative exists.  A legacy branch label is intentionally
    # incomplete provenance and must not be reported as a certificate.
    h = 1e-6
    assert int(branch_id(f=f_star, theta=jnp.asarray(h))) != int(
        branch_id(f=f_star, theta=jnp.asarray(-h))
    )
    assert not bool(diag.branch_certified)
    assert bool(diag.unresolved), (
        "parameter-direction owner switch was incorrectly certified as smooth",
        center,
        slope,
    )


def _smooth_objective(*, f, theta):
    return -0.5 * (f - 0.4) ** 2 + theta * (f - 0.4)


def _record(
    *,
    signature,
    decided=True,
    strict_primary=True,
    complete=True,
):
    return OwnerProvenance(
        signature=signature,
        decided=jnp.asarray(decided),
        strict_primary=jnp.asarray(strict_primary),
        complete=jnp.asarray(complete),
    )


def _diagnose(*, record, extras=(), require=True):
    return implicit_optimum_diagnostics(
        objective=_smooth_objective,
        theta=jnp.asarray(0.0),
        f_star=jnp.asarray(0.4),
        basin_margin=jnp.asarray(1.0),
        bounds=(jnp.asarray(0.0), jnp.asarray(1.0)),
        owner_provenance=record,
        require_owner_certificate=require,
        reoptimized_owner_points=extras,
    )


def test_strict_complete_composite_record_is_certified() -> None:
    def record(*, f, theta):
        del theta
        zero = jnp.zeros_like(f, dtype=jnp.int32)
        return _record(signature=(zero + 2, zero, zero + 1, zero, zero + 1))

    extras = (
        (jnp.asarray(0.4001), jnp.asarray(1e-3)),
        (jnp.asarray(0.3999), jnp.asarray(-1e-3)),
        (jnp.asarray(0.40005), jnp.asarray(5e-4)),
        (jnp.asarray(0.39995), jnp.asarray(-5e-4)),
    )
    diag = _diagnose(record=record, extras=extras)
    assert bool(diag.branch_certified)
    assert not bool(diag.unresolved)
    assert not bool(diag.owner_missing)


@pytest.mark.parametrize(
    ("flag", "expected_field"),
    [
        ("tie", "owner_primary_tie"),
        ("unresolved", "owner_unresolved"),
        ("incomplete", "owner_incomplete"),
    ],
)
def test_status_tie_and_incomplete_records_fail_closed(
    *, flag: str, expected_field: str
) -> None:
    def record(*, f, theta):
        del f, theta
        return _record(
            signature=(jnp.asarray(3, dtype=jnp.int32),),
            decided=flag != "unresolved",
            strict_primary=flag != "tie",
            complete=flag != "incomplete",
        )

    diag = _diagnose(record=record)
    assert not bool(diag.branch_certified)
    assert bool(diag.unresolved)
    assert bool(getattr(diag, expected_field))


def test_missing_provenance_fails_closed_when_certification_is_required() -> None:
    diag = implicit_optimum_diagnostics(
        objective=_smooth_objective,
        theta=jnp.asarray(0.0),
        f_star=jnp.asarray(0.4),
        basin_margin=jnp.asarray(1.0),
        bounds=(jnp.asarray(0.0), jnp.asarray(1.0)),
        require_owner_certificate=True,
    )
    assert bool(diag.owner_missing)
    assert not bool(diag.branch_certified)
    assert bool(diag.unresolved)


@pytest.mark.parametrize("component", range(5))
def test_any_composite_component_change_fails_closed(component: int) -> None:
    """Segment, inner choice, floor, constraint and branch identities all bind."""

    def record(*, f, theta):
        del theta
        changed = jnp.asarray(f > 0.4, dtype=jnp.int32)
        fields = [jnp.asarray(0, dtype=jnp.int32) for _ in range(5)]
        fields[component] = changed
        return _record(signature=tuple(fields))

    diag = _diagnose(record=record)
    assert bool(diag.owner_changed)
    assert not bool(diag.branch_certified)
    assert bool(diag.unresolved)


def test_parameter_only_signature_change_fails_closed() -> None:
    def record(*, f, theta):
        del f
        return _record(signature=(jnp.asarray(theta > 0.0, dtype=jnp.int32),))

    diag = _diagnose(record=record)
    assert bool(diag.owner_changed)
    assert bool(diag.unresolved)


def test_mixed_corner_only_signature_change_fails_closed() -> None:
    def record(*, f, theta):
        mixed = (f > 0.4) & (theta > 0.0)
        return _record(signature=(jnp.asarray(mixed, dtype=jnp.int32),))

    diag = _diagnose(record=record)
    assert bool(diag.owner_changed)
    assert bool(diag.unresolved)


def test_reoptimized_h_and_h2_points_are_part_of_the_certificate() -> None:
    def record(*, f, theta):
        del f
        return _record(signature=(jnp.asarray(theta > 5e-3, dtype=jnp.int32),))

    # The built-in parameter radius is 1e-5, so only this independently
    # reoptimized Richardson point crosses the signature boundary.
    extras = ((jnp.asarray(0.41), jnp.asarray(1e-2)),)
    diag = _diagnose(record=record, extras=extras)
    assert bool(diag.owner_changed)
    assert bool(diag.unresolved)


def test_empty_signature_is_incomplete_even_when_callback_claims_complete() -> None:
    def empty_record(*, f, theta):
        del f, theta
        return _record(signature=())

    diag = _diagnose(record=empty_record)
    assert bool(diag.owner_incomplete)
    assert not bool(diag.branch_certified)
    assert bool(diag.unresolved)


def test_vectorized_jit_certificate_is_per_cell() -> None:
    def evaluate(theta):
        f_star = jnp.asarray([0.3, 0.4])

        def objective(*, f, t):
            return -0.5 * (f - jnp.asarray([0.3, 0.4])) ** 2 + 0.0 * t

        def record(*, f, t):
            del t
            return OwnerProvenance(
                signature=(jnp.zeros_like(f, dtype=jnp.int32),),
                decided=jnp.ones_like(f, dtype=bool),
                strict_primary=jnp.ones_like(f, dtype=bool),
                complete=jnp.ones_like(f, dtype=bool),
            )

        diag = implicit_optimum_diagnostics(
            objective=objective,
            theta=theta,
            f_star=f_star,
            basin_margin=jnp.ones_like(f_star),
            bounds=(jnp.zeros_like(f_star), jnp.ones_like(f_star)),
            owner_provenance=record,
            require_owner_certificate=True,
        )
        return diag.branch_certified, diag.unresolved

    certified, unresolved = jax.jit(evaluate)(jnp.asarray(0.0))
    assert certified.tolist() == [True, True]
    assert unresolved.tolist() == [False, False]
