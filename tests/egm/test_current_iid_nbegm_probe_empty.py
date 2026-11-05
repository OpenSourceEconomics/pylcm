"""Original empty NB partitions keep their numeric and strict-domain contracts."""

from types import MappingProxyType

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.dtypes import canonical_float_dtype
from _lcm.egm.nbegm_probe_stages import (
    probe_clamped_breakpoints,
    probe_raw_breakpoints,
)
from _lcm.solution import nbegm
from _lcm.solution.nbegm import (
    _LiquidAffinityProbe,
    _NBEGMSource,
    _ProbeArguments,
    _ProbeFillRung,
)
from lcm.typing import FloatND
from tests.egm.test_current_iid_generation_materialization import _owner

pytestmark = pytest.mark.coverage(backends=("cpu",), precisions="both")


def _identity_budget(*, wealth: FloatND) -> FloatND:
    return wealth


def _rung() -> _ProbeFillRung:
    return _ProbeFillRung(
        fill=1.0,
        array_floats=False,
        array_rank=1,
        leaf_rank=1,
        int_overrides=MappingProxyType({}),
    )


def _probe(*, grid: jax.Array) -> _LiquidAffinityProbe:
    return _LiquidAffinityProbe(
        coh_dag=_identity_budget,
        arg_names=("wealth",),
        liquid_name="wealth",
        probe_arguments=_ProbeArguments(),
        liquid_grid=grid,
        liquid_samples=grid,
        breakpoint_sources=(),
        dtype=grid.dtype,
        int_sweeps=(),
    )


@pytest.mark.parametrize("domain", [(1.0, 3.0), (-2.0, 2.0), (2.0, 2.0)])
def test_original_empty_partition_operand_matches_original_clamp(
    *, domain: tuple[float, float]
) -> None:
    dtype = canonical_float_dtype()
    grid = jnp.asarray(domain, dtype=dtype)
    raw = probe_raw_breakpoints(values=(), dtype=dtype)
    expected = probe_clamped_breakpoints(raw=raw, liquid_grid=grid, dtype=dtype)
    assert raw.shape == expected.shape == (0,)
    assert raw.dtype == expected.dtype == grid.dtype
    assert raw.weak_type == expected.weak_type is False
    assert np.asarray(raw).tobytes() == np.asarray(expected).tobytes()


@pytest.mark.parametrize("owned", [False, True])
@pytest.mark.parametrize("domain", ["wide", "signed", "equal", "one_interior", "none"])
def test_actual_empty_partition_keeps_original_strict_domain(
    *, domain: str, owned: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    from dataclasses import replace  # noqa: PLC0415

    dtype = np.dtype(canonical_float_dtype())
    one = dtype.type(1.0)
    adjacent = np.nextafter(one, dtype.type(np.inf))
    after_adjacent = np.nextafter(adjacent, dtype.type(np.inf))
    bounds = {
        "wide": (1.0, 3.0),
        "signed": (-2.0, 2.0),
        "equal": (2.0, 2.0),
        "one_interior": (one, after_adjacent),
        "none": (one, adjacent),
    }
    expected_values = {
        "wide": (2.0,),
        "signed": (0.0,),
        "equal": (),
        "one_interior": (adjacent,),
    }
    grid = jnp.asarray(bounds[domain], dtype=dtype)
    probe = _probe(grid=grid)
    owner = _owner(roots=(grid,)) if owned else None
    if owner is not None:
        probe = replace(probe, materializer=owner)

    def must_not_clamp(**_: FloatND) -> FloatND:
        raise AssertionError("An empty original partition invoked its unused clamp.")

    monkeypatch.setattr(nbegm, "probe_clamped_breakpoints", must_not_clamp)
    try:
        if domain == "none":
            with pytest.raises(ValueError, match="no representable strict interior"):
                probe.strict_interval_samples(rung=_rung())
        else:
            actual = probe.strict_interval_samples(rung=_rung())
            expected = expected_values[domain]
            assert len(actual) == len(expected)
            for value, literal in zip(actual, expected, strict=True):
                reference = jnp.asarray(literal, dtype=dtype).reshape(())
                assert value.dtype == reference.dtype
                assert value.weak_type == reference.weak_type
                assert np.asarray(value).tobytes() == np.asarray(reference).tobytes()
    finally:
        if owner is not None:
            owner.close()


@pytest.mark.parametrize("threshold", [2.0, np.nan, -np.inf, np.inf])
def test_actual_nonempty_partition_retains_original_clamp(
    *, threshold: float, monkeypatch: pytest.MonkeyPatch
) -> None:
    from dataclasses import replace  # noqa: PLC0415

    dtype = canonical_float_dtype()
    grid = jnp.asarray([1.0, 3.0], dtype=dtype)
    cut = jnp.asarray(threshold, dtype=dtype)
    source = _NBEGMSource(
        variable="wealth",
        threshold_param_name="cut",
        kind="continuous_kink",
        equality_owner="above",
        derived_of_liquid_dag=None,
        derived_param_names=(),
    )
    probe = replace(
        _probe(grid=grid),
        breakpoint_sources=(source,),
        probe_arguments=_ProbeArguments(param_values=MappingProxyType({"cut": cut})),
    )
    original = nbegm.probe_clamped_breakpoints
    observed: list[jax.Array] = []

    def same_original_clamp(
        *, raw: FloatND, liquid_grid: FloatND, dtype: jax.typing.DTypeLike
    ) -> FloatND:
        actual = original(raw=raw, liquid_grid=liquid_grid, dtype=dtype)
        expected = probe_clamped_breakpoints(
            raw=raw, liquid_grid=liquid_grid, dtype=dtype
        )
        assert actual.dtype == expected.dtype
        assert actual.weak_type == expected.weak_type
        assert np.asarray(actual).tobytes() == np.asarray(expected).tobytes()
        observed.append(actual)
        return actual

    monkeypatch.setattr(nbegm, "probe_clamped_breakpoints", same_original_clamp)
    actual = probe.strict_interval_samples(rung=_rung())
    assert len(observed) == 1
    expected = (1.5, 2.5) if threshold == 2.0 else (2.0,)
    assert len(actual) == len(expected)
    for value, literal in zip(actual, expected, strict=True):
        reference = jnp.asarray(literal, dtype=dtype)
        assert value.dtype == reference.dtype
        assert value.weak_type == reference.weak_type
        assert np.asarray(value).tobytes() == np.asarray(reference).tobytes()
