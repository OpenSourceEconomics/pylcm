"""Typed current-IID inputs retain independent bases and producing-law support."""

from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import lcm
from _lcm.current_iid import (
    CORRELATION_SUPPORT_ARGUMENT,
    CurrentIIDDrawFrame,
    CurrentIIDDrawPlan,
)
from _lcm.shock_registry import ShockBinding
from lcm import categorical
from lcm import shocks as sh
from lcm.exceptions import ShockDeclarationError
from lcm.typing import FloatND, ScalarInt
from tests.conftest import DECIMAL_PRECISION


@categorical(ordered=False)
class Offer:
    absent: ScalarInt
    present: ScalarInt


def _plan(*, group: sh.ShockGroup) -> CurrentIIDDrawPlan:
    return CurrentIIDDrawPlan(
        bindings={
            "risk": ShockBinding(
                group=group,
                state_inputs=frozenset(),
                state_domains={},
                categorical_domains={},
                functions={},
                params_template={},
                callable_parameters=frozenset(),
            )
        }
    )


def _keys(*, group: sh.ShockGroup) -> Mapping[sh.ShockName, jax.Array]:
    return {name: jax.random.key(index + 31) for index, name in enumerate(group.shocks)}


@pytest.mark.coverage(backends=("cpu",), precisions="both")
def test_typed_payloads_are_dynamic_leaves() -> None:
    """Base and physical inputs remain distinguishable without hiding their arrays."""
    assert (
        Path(lcm.__file__).resolve().parents[2] == Path(__file__).resolve().parents[1]
    )
    assert hasattr(sh, "BaseInnovations"), "BaseInnovations record missing"
    assert hasattr(sh, "PhysicalValues"), "PhysicalValues record missing"
    for record in (sh.BaseInnovations, sh.PhysicalValues):
        payload = jnp.asarray([0.25, 0.75])
        wrapped = record(values=payload)
        leaves, tree = jax.tree_util.tree_flatten(wrapped)
        assert len(leaves) == 1
        assert leaves[0] is payload
        restored = jax.tree_util.tree_unflatten(tree, leaves)
        assert type(restored) is record
        assert restored.values is payload


@pytest.mark.coverage(backends=("cpu",), precisions="both")
@pytest.mark.parametrize("compiled", [False, True])
def test_draw_frame_retains_named_bases_and_existing_action(*, compiled: bool) -> None:
    """An overridden base passes through the current law and preserves other draws."""
    assert hasattr(CurrentIIDDrawPlan, "realise"), "current-IID draw frame missing"
    group = sh.ShockGroup(
        shocks={"income": sh.Normal(mu=10, sigma=3), "taste": sh.Normal(mu=0, sigma=1)},
        nodes=sh.MonteCarlo(1, seed=0),
    )
    plan = _plan(group=group)
    keys = _keys(group=group)
    plain = plan.realise(
        keys=keys,
        group_params={},
        arguments={"period": jnp.asarray(0, dtype=jnp.int32), "age": jnp.asarray(0.0)},
    )

    def draw(value: FloatND) -> CurrentIIDDrawFrame:
        return plan.realise(
            keys=keys,
            group_params={},
            arguments={
                "period": jnp.asarray(0, dtype=jnp.int32),
                "age": jnp.asarray(0.0),
            },
            overrides={"income": sh.BaseInnovations(values=value)},
        )

    frame = (jax.jit(draw) if compiled else draw)(jnp.asarray(2.0))
    assert bool(frame.valid)
    np.testing.assert_array_equal(frame.physical["income"], 16.0)
    np.testing.assert_array_equal(frame.physical["taste"], plain.physical["taste"])
    np.testing.assert_array_equal(
        frame.addressed_bases["income"], plain.held_bases["income"]
    )
    np.testing.assert_array_equal(frame.held_bases["income"], 2.0)
    assert isinstance(frame.physical, MappingProxyType)


@pytest.mark.coverage(backends=("cpu",), precisions="both")
@pytest.mark.parametrize(
    ("law", "physical", "base"),
    [
        (sh.Normal(mu=2, sigma=3), 8.0, 2.0),
        (sh.LogNormal(mu=0, sigma=2), np.exp(1.0), 0.5),
        (sh.Uniform(start=-2, stop=6), 0.0, 0.25),
        (sh.NormalMixture(p1=0.5, mu1=-2, sigma1=1, mu2=2, sigma2=1), 0.0, 0.0),
    ],
)
def test_independent_physical_inverse(
    *, law: sh.Law, physical: float, base: float
) -> None:
    """A supplied physical value retains its bits and recovers its producing base."""
    group = sh.ShockGroup(shocks={"eps": law}, nodes=sh.MonteCarlo(1, seed=0))
    plan = _plan(group=group)
    supplied = jnp.asarray(physical)
    frame = jax.jit(plan.realise)(
        keys=_keys(group=group),
        group_params={},
        arguments={"period": jnp.asarray(0, dtype=jnp.int32), "age": jnp.asarray(0.0)},
        overrides={"eps": sh.PhysicalValues(values=supplied)},
    )
    assert bool(frame.valid)
    np.testing.assert_array_equal(frame.physical["eps"], supplied)
    np.testing.assert_allclose(
        frame.held_bases["eps"], base, atol=10**-DECIMAL_PRECISION
    )
    reconstructed = group.transform(base_nodes=jnp.asarray([[base]])).values["eps"][0]
    np.testing.assert_array_equal(frame.recovery_error["eps"], reconstructed - supplied)


@pytest.mark.coverage(backends=("cpu",), precisions="both")
@pytest.mark.parametrize(
    ("law", "physical"),
    [(sh.Normal(mu=2, sigma=0), 2.0), (sh.Uniform(start=2, stop=2), 2.0)],
)
def test_deterministic_physical_preserves_unresolved_base(
    *, law: sh.Law, physical: float
) -> None:
    """A compatible constant observation cannot replace an unresolved base by zero."""
    group = sh.ShockGroup(shocks={"eps": law}, nodes=sh.MonteCarlo(1, seed=0))
    frame = _plan(group=group).realise(
        keys=_keys(group=group),
        group_params={},
        arguments={"period": jnp.asarray(0, dtype=jnp.int32), "age": jnp.asarray(0.0)},
        overrides={"eps": sh.PhysicalValues(values=jnp.asarray(physical))},
    )
    assert bool(frame.valid)
    np.testing.assert_array_equal(frame.held_bases["eps"], frame.addressed_bases["eps"])


@pytest.mark.coverage(backends=("cpu",), precisions="both")
def test_complete_correlated_physical_frame_and_singular_support() -> None:
    """Complete physical frames recover joint bases and reject contradictory support."""
    for rho, physical, expected in (
        (0.6, (2.0, 2.0), (2.0, 1.0)),
        (1.0, (2.0, 2.0), (2.0, None)),
    ):
        group = sh.ShockGroup(
            shocks={"a": sh.Normal(mu=0, sigma=1), "b": sh.Normal(mu=0, sigma=1)},
            correlation=((1.0, rho), (rho, 1.0)),
            nodes=sh.MonteCarlo(1, seed=0),
        )
        plan = _plan(group=group)
        frame = plan.realise(
            keys=_keys(group=group),
            group_params={},
            arguments={
                "period": jnp.asarray(0, dtype=jnp.int32),
                "age": jnp.asarray(0.0),
            },
            overrides={
                name: sh.PhysicalValues(values=jnp.asarray(value))
                for name, value in zip(("a", "b"), physical, strict=True)
            },
        )
        assert bool(frame.valid)
        np.testing.assert_allclose(
            frame.held_bases["a"], expected[0], atol=10**-DECIMAL_PRECISION
        )
        np.testing.assert_allclose(
            frame.held_bases["b"],
            frame.addressed_bases["b"] if expected[1] is None else expected[1],
            atol=10**-DECIMAL_PRECISION,
        )
        if rho == 1:
            bad = plan.realise(
                keys=_keys(group=group),
                group_params={},
                arguments={
                    "period": jnp.asarray(0, dtype=jnp.int32),
                    "age": jnp.asarray(0.0),
                },
                overrides={
                    "a": sh.PhysicalValues(values=jnp.asarray(2.0)),
                    "b": sh.PhysicalValues(values=jnp.asarray(-2.0)),
                },
            )
            assert not bool(bad.valid)


@pytest.mark.coverage(backends=("cpu",), precisions="both")
def test_partial_and_mixed_correlated_physical_refuse() -> None:
    """Correlated physical overrides require the complete frame alone."""
    group = sh.ShockGroup(
        shocks={"a": sh.Normal(mu=0, sigma=1), "b": sh.Normal(mu=0, sigma=1)},
        correlation=((1.0, 0.5), (0.5, 1.0)),
        nodes=sh.MonteCarlo(1, seed=0),
    )
    for overrides in (
        {"a": sh.PhysicalValues(values=jnp.asarray(1.0))},
        {
            "a": sh.PhysicalValues(values=jnp.asarray(1.0)),
            "b": sh.BaseInnovations(values=jnp.asarray(0.0)),
        },
    ):
        with pytest.raises(ShockDeclarationError, match="complete physical frame"):
            _plan(group=group).realise(
                keys=_keys(group=group),
                group_params={},
                arguments={
                    "period": jnp.asarray(0, dtype=jnp.int32),
                    "age": jnp.asarray(0.0),
                },
                overrides=overrides,
            )


@pytest.mark.coverage(backends=("cpu",), precisions="both")
@pytest.mark.parametrize("value", [-0.1, 1.1, float("nan")])
def test_uniform_base_rejects_outside_support(*, value: float) -> None:
    """Unit-uniform base overrides reject values outside the forward-law support."""
    group = sh.ShockGroup(
        shocks={"eps": sh.Uniform(start=0, stop=1)}, nodes=sh.MonteCarlo(1, seed=0)
    )
    frame = _plan(group=group).realise(
        keys=_keys(group=group),
        group_params={},
        arguments={"period": jnp.asarray(0, dtype=jnp.int32), "age": jnp.asarray(0.0)},
        overrides={"eps": sh.BaseInnovations(values=jnp.asarray(value))},
    )
    assert not bool(frame.valid)


@pytest.mark.coverage(backends=("cpu",), precisions="both")
def test_discrete_physical_uses_addressed_interval_residual() -> None:
    """A category constrains its uniform base to its positive source interval."""
    group = sh.ShockGroup(
        shocks={"offer": sh.Discrete(categories=Offer, probs=(0.8, 0.2))}
    )
    plan = _plan(group=group)
    frame = plan.realise(
        keys=_keys(group=group),
        group_params={},
        arguments={"period": jnp.asarray(0, dtype=jnp.int32), "age": jnp.asarray(0.0)},
        overrides={"offer": sh.PhysicalValues(values=jnp.asarray(0, dtype=jnp.int32))},
    )
    assert bool(frame.valid)
    np.testing.assert_array_equal(frame.physical["offer"], 0)
    np.testing.assert_allclose(
        frame.held_bases["offer"],
        0.8 * frame.addressed_bases["offer"],
        atol=10**-DECIMAL_PRECISION,
    )
    null = sh.ShockGroup(
        shocks={"offer": sh.Discrete(categories=Offer, probs=(0.0, 1.0))}
    )
    rejected = _plan(group=null).realise(
        keys=_keys(group=group),
        group_params={},
        arguments={"period": jnp.asarray(0, dtype=jnp.int32), "age": jnp.asarray(0.0)},
        overrides={"offer": sh.PhysicalValues(values=jnp.asarray(0, dtype=jnp.int32))},
    )
    assert not bool(rejected.valid)


@pytest.mark.coverage(backends=("cpu",), precisions="both")
def test_correlated_noninformative_frame_preserves_or_refuses_latent_custody() -> None:
    """Wholly constant frames reveal nothing; mixed frames need deferred completion."""
    for second_scale, valid in ((0.0, True), (1.0, False)):
        group = sh.ShockGroup(
            shocks={
                "a": sh.Normal(mu=2, sigma=0),
                "b": sh.Normal(mu=3, sigma=second_scale),
            },
            correlation=((1.0, 0.5), (0.5, 1.0)),
            nodes=sh.MonteCarlo(1, seed=0),
        )
        frame = _plan(group=group).realise(
            keys=_keys(group=group),
            group_params={},
            arguments={
                "period": jnp.asarray(0, dtype=jnp.int32),
                "age": jnp.asarray(0.0),
            },
            overrides={
                "a": sh.PhysicalValues(values=jnp.asarray(2.0)),
                "b": sh.PhysicalValues(values=jnp.asarray(3.0)),
            },
        )
        assert bool(frame.valid) is valid
        if valid:
            for name in group.shocks:
                np.testing.assert_array_equal(
                    frame.held_bases[name], frame.addressed_bases[name]
                )


@pytest.mark.coverage(backends=("cpu",), precisions="both")
def test_original_wrapper_keeps_decision_physical_status_contract() -> None:
    """The ordinary wrapper consumes the shared frame with the same output layout."""
    group = sh.ShockGroup(
        shocks={"eps": sh.Normal(mu=2, sigma=3)}, nodes=sh.MonteCarlo(1, seed=0)
    )
    plan = _plan(group=group)

    def choose(*, eps: FloatND) -> tuple[jax.Array, FloatND]:
        return jnp.asarray(eps > 2, dtype=jnp.int32), eps

    arguments = {"period": jnp.asarray(0, dtype=jnp.int32), "age": jnp.asarray(0.0)}
    frame = plan.realise(keys=_keys(group=group), group_params={}, arguments=arguments)
    result = jax.jit(plan.wrap(choose))(
        **arguments, _lcm_current_iid={}, _lcm_current_iid_keys=_keys(group=group)
    )
    assert len(result) == 4
    assert isinstance(result[2], dict), (
        "simulation physical output tree must remain dict"
    )
    np.testing.assert_array_equal(result[0], frame.physical["eps"] > 2)
    np.testing.assert_array_equal(result[1], frame.physical["eps"])
    np.testing.assert_array_equal(result[2]["eps"], frame.physical["eps"])
    assert not bool(result[3])


@pytest.mark.coverage(backends=("cpu",), precisions="both")
def test_uniform_upper_endpoint_retains_forward_support() -> None:
    """Uniform's endpoint remains valid although categorical bases exclude it."""
    group = sh.ShockGroup(
        shocks={"eps": sh.Uniform(start=2, stop=6)}, nodes=sh.MonteCarlo(1, seed=0)
    )
    frame = _plan(group=group).realise(
        keys=_keys(group=group),
        group_params={},
        arguments={"period": jnp.asarray(0, dtype=jnp.int32), "age": jnp.asarray(0.0)},
        overrides={"eps": sh.PhysicalValues(values=jnp.asarray(6.0))},
    )
    assert bool(frame.valid)
    np.testing.assert_array_equal(frame.held_bases["eps"], 1.0)
    direct = _plan(group=group).realise(
        keys=_keys(group=group),
        group_params={},
        arguments={"period": jnp.asarray(0, dtype=jnp.int32), "age": jnp.asarray(0.0)},
        overrides={"eps": sh.BaseInnovations(values=jnp.asarray(1.0))},
    )
    assert bool(direct.valid)
    np.testing.assert_array_equal(direct.physical["eps"], 6.0)
    categorical_group = sh.ShockGroup(
        shocks={"offer": sh.Discrete(categories=Offer, probs=(0.8, 0.2))}
    )
    categorical_frame = _plan(group=categorical_group).realise(
        keys=_keys(group=categorical_group),
        group_params={},
        arguments={"period": jnp.asarray(0, dtype=jnp.int32), "age": jnp.asarray(0.0)},
        overrides={"offer": sh.BaseInnovations(values=jnp.asarray(1.0))},
    )
    assert not bool(categorical_frame.valid)


@pytest.mark.coverage(backends=("cpu",), precisions="both")
@pytest.mark.parametrize("base", [0.25, -0.4])
def test_shifted_singular_physical_frame_requires_exact_recovered_witness(
    *, base: float
) -> None:
    """A rounded frame without an exact recovered witness remains unresolved."""
    group = sh.ShockGroup(
        shocks={
            "x": sh.Normal(mu=0.7, sigma=np.sqrt(2)),
            "y": sh.Normal(mu=-0.3, sigma=np.sqrt(2)),
        },
        correlation=((1.0, 1.0), (1.0, 1.0)),
        nodes=sh.MonteCarlo(1, seed=0),
    )
    source = group.transform(base_nodes=jnp.asarray([[base, -0.7]]))
    source.raise_if_invalid()
    frame = _plan(group=group).realise(
        keys=_keys(group=group),
        group_params={},
        arguments={"period": jnp.asarray(0, dtype=jnp.int32), "age": jnp.asarray(0.0)},
        overrides={
            name: sh.PhysicalValues(values=value[0])
            for name, value in source.values.items()
        },
    )
    reconstructed = group.transform(
        base_nodes=jnp.stack(tuple(frame.held_bases.values()))[None]
    )
    word_type = np.uint64 if jnp.asarray(base).dtype.itemsize == 8 else np.uint32
    matches = all(
        np.array_equal(
            np.asarray(reconstructed.values[name]).view(word_type),
            np.asarray(source.values[name]).view(word_type),
        )
        for name in group.shocks
    )
    assert bool(frame.valid) is matches
    for name in group.shocks:
        np.testing.assert_array_equal(frame.physical[name], source.values[name][0])
    np.testing.assert_allclose(frame.held_bases["x"], base, atol=10**-DECIMAL_PRECISION)


@pytest.mark.coverage(backends=("cpu",), precisions="both")
def test_physical_frame_replays_under_target_law() -> None:
    """A mapped target law transforms the recovered base rather than reclamping x."""
    source = sh.ShockGroup(
        shocks={"eps": sh.Normal(mu=0, sigma=1)}, nodes=sh.MonteCarlo(1, seed=0)
    )
    frame = _plan(group=source).realise(
        keys=_keys(group=source),
        group_params={},
        arguments={"period": jnp.asarray(0, dtype=jnp.int32), "age": jnp.asarray(0.0)},
        overrides={"eps": sh.PhysicalValues(values=jnp.asarray(2.0))},
    )
    target = sh.ShockGroup(
        shocks={"eps": sh.Normal(mu=10, sigma=3)}, nodes=sh.MonteCarlo(1, seed=0)
    )
    replay = target.transform(
        base_nodes=jnp.stack(tuple(frame.held_bases.values()))[None]
    )
    replay.raise_if_invalid()
    np.testing.assert_array_equal(replay.values["eps"], [16.0])


@pytest.mark.coverage(backends=("cpu",), precisions="both")
def test_tiny_positive_categorical_inverse_cannot_invent_interval() -> None:
    """Positive mathematical mass with an unrepresented CDF interval is unresolved."""
    tiny = float(np.finfo(np.float32).tiny)
    group = sh.ShockGroup(
        shocks={"offer": sh.Discrete(categories=Offer, probs=(1.0, tiny))}
    )
    frame = _plan(group=group).realise(
        keys=_keys(group=group),
        group_params={},
        arguments={"period": jnp.asarray(0, dtype=jnp.int32), "age": jnp.asarray(0.0)},
        overrides={"offer": sh.PhysicalValues(values=jnp.asarray(1, dtype=jnp.int32))},
    )
    assert not bool(frame.valid)


@pytest.mark.coverage(backends=("cpu",), precisions="both")
def test_rounded_large_location_singular_inverse_is_unresolved() -> None:
    """Lost affine information cannot be certified by a location-sized tolerance."""
    large = float(2 ** (np.finfo(np.dtype(jnp.result_type(0.0))).nmant + 3))
    group = sh.ShockGroup(
        shocks={"a": sh.Normal(mu=large, sigma=1), "b": sh.Normal(mu=0, sigma=1)},
        correlation=((1.0, 1.0), (1.0, 1.0)),
        nodes=sh.MonteCarlo(1, seed=0),
    )
    values = group.transform(base_nodes=jnp.asarray([[0.25, 0.0]])).values
    frame = _plan(group=group).realise(
        keys=_keys(group=group),
        group_params={},
        arguments={"period": jnp.asarray(0, dtype=jnp.int32), "age": jnp.asarray(0.0)},
        overrides={
            name: sh.PhysicalValues(values=value[0]) for name, value in values.items()
        },
    )
    assert not bool(frame.valid)


@pytest.mark.coverage(backends=("cpu", "gpu-small", "gpu-large"), precisions="both")
def test_batched_native_draw_frame_keeps_payload_and_status_leaves() -> None:
    """Batched AOT draw frames keep per-row bases, contexts and validity visible."""
    group = sh.ShockGroup(
        shocks={"eps": sh.Normal(mu=10, sigma=3)}, nodes=sh.MonteCarlo(1, seed=0)
    )
    plan = _plan(group=group)

    def draw(value: FloatND) -> CurrentIIDDrawFrame:
        return plan.realise(
            keys=_keys(group=group),
            group_params={},
            arguments={
                "period": jnp.asarray(0, dtype=jnp.int32),
                "age": jnp.asarray(0.0),
            },
            overrides={"eps": sh.BaseInnovations(values=value)},
        )

    supplied = jnp.asarray([-1.0, 0.0, 2.0])
    compiled = jax.jit(jax.vmap(draw)).lower(supplied).compile()
    frame = compiled(supplied)
    np.testing.assert_array_equal(frame.physical["eps"], [7.0, 10.0, 16.0])
    np.testing.assert_array_equal(frame.held_bases["eps"], supplied)
    np.testing.assert_array_equal(frame.valid, [True, True, True])


@pytest.mark.coverage(backends=("cpu",), precisions="both")
def test_categorical_physical_rejects_integer_narrowing() -> None:
    """An out-of-range host category cannot wrap into a valid int32 outcome."""
    group = sh.ShockGroup(
        shocks={"offer": sh.Discrete(categories=Offer, probs=(0.8, 0.2))}
    )
    with pytest.raises(ShockDeclarationError, match="narrow"):
        _plan(group=group).realise(
            keys=_keys(group=group),
            group_params={},
            arguments={
                "period": jnp.asarray(0, dtype=jnp.int32),
                "age": jnp.asarray(0.0),
            },
            overrides={
                "offer": sh.PhysicalValues(values=np.asarray(2**32, dtype=np.int64))
            },
        )


@pytest.mark.coverage(backends=("cpu",), precisions="both")
def test_singular_physical_frame_rejects_adjacent_float_contradiction() -> None:
    """Identical producing coordinates cannot emit different represented values."""
    group = sh.ShockGroup(
        shocks={"a": sh.Normal(mu=0, sigma=1), "b": sh.Normal(mu=0, sigma=1)},
        correlation=((1.0, 1.0), (1.0, 1.0)),
        nodes=sh.MonteCarlo(1, seed=0),
    )
    value = jnp.asarray(0.75)
    adjacent = jnp.nextafter(value, jnp.inf)
    frame = _plan(group=group).realise(
        keys=_keys(group=group),
        group_params={},
        arguments={"period": jnp.asarray(0, dtype=jnp.int32), "age": jnp.asarray(0.0)},
        overrides={
            "a": sh.PhysicalValues(values=value),
            "b": sh.PhysicalValues(values=adjacent),
        },
    )
    assert not bool(frame.valid), "singular frame lacks an exact producing witness"


@pytest.mark.coverage(backends=("cpu",), precisions="both")
def test_draw_frame_retains_runtime_correlation_admission_context() -> None:
    """A retained producer context includes the actual admitted correlation support."""
    group = sh.ShockGroup(
        shocks={"a": sh.Normal(mu=0, sigma=1), "b": sh.Normal(mu=0, sigma=1)},
        correlation=None,
        nodes=sh.MonteCarlo(1, seed=0),
    )
    corr = jnp.asarray([[1.0, 0.5], [0.5, 1.0]])
    support = jnp.asarray(group.validate_runtime_correlation(correlation=corr))
    frame = _plan(group=group).realise(
        keys=_keys(group=group),
        group_params={"risk": {"corr": corr, CORRELATION_SUPPORT_ARGUMENT: support}},
        arguments={"period": jnp.asarray(0, dtype=jnp.int32), "age": jnp.asarray(0.0)},
    )
    assert bool(frame.valid)
    np.testing.assert_array_equal(frame.source_params["risk"]["corr"], corr)
    np.testing.assert_array_equal(
        frame.source_params["risk"][CORRELATION_SUPPORT_ARGUMENT], support
    )


@pytest.mark.coverage(backends=("cpu",), precisions="both")
@pytest.mark.parametrize("runtime", [False, True])
def test_exact_identity_allows_partial_physical_and_runtime_change_refuses(
    *, runtime: bool
) -> None:
    """Actual identity needs no correlated completion; changed correlation refuses."""
    identity = ((1.0, 0.0), (0.0, 1.0))
    group = sh.ShockGroup(
        shocks={"a": sh.Normal(mu=2, sigma=3), "b": sh.Normal(mu=0, sigma=1)},
        correlation=None if runtime else identity,
        nodes=sh.MonteCarlo(1, seed=0),
    )
    plan = _plan(group=group)
    arguments = {"period": jnp.asarray(0, dtype=jnp.int32), "age": jnp.asarray(0.0)}
    group_params = (
        {
            "risk": {
                "corr": jnp.asarray(identity),
                CORRELATION_SUPPORT_ARGUMENT: jnp.asarray([False, False]),
            }
        }
        if runtime
        else {}
    )
    overrides = {"a": sh.PhysicalValues(values=jnp.asarray(8.0))}
    frame = jax.jit(plan.realise)(
        keys=_keys(group=group),
        group_params=group_params,
        arguments=arguments,
        overrides=overrides,
    )
    assert bool(frame.valid)
    np.testing.assert_array_equal(frame.physical["a"], 8.0)
    np.testing.assert_array_equal(frame.held_bases["a"], 2.0)
    np.testing.assert_array_equal(frame.held_bases["b"], frame.addressed_bases["b"])
    if runtime:
        changed = {
            "risk": {
                "corr": jnp.asarray([[1.0, 0.5], [0.5, 1.0]]),
                CORRELATION_SUPPORT_ARGUMENT: jnp.asarray([False, False]),
            }
        }
        refused = jax.jit(plan.realise)(
            keys=_keys(group=group),
            group_params=changed,
            arguments=arguments,
            overrides=overrides,
        )
        assert not bool(refused.valid)
