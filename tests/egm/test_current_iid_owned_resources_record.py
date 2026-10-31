"""Original selected resources records satisfy the closed artifact boundary."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.egm.nbegm_resources import NBEGMResourcesBank
from _lcm.execution.output_layout import StateAxesLeading


@pytest.mark.coverage(backends=("cpu",), precisions="both")
def test_nbegm_resource_bank_preserves_original_schema_through_jax() -> None:
    """Abstract templates, declared roles and first-order tangents keep each field."""
    vector = jnp.asarray([0.25, 0.5, 1.0])
    bank = NBEGMResourcesBank(
        endog_grid=vector,
        value=vector + 1,
        policy=vector + 2,
        marginal=vector + 3,
        savings=vector + 4,
        segment_id=vector + 5,
        resources_support=vector + 6,
        borrowing_limit=vector[0],
        upper_savings=vector[-1],
        construction_invalid=jnp.zeros((), dtype=jnp.bool_),
        unsupported_economic_minus_inf=jnp.ones((), dtype=jnp.bool_),
    )
    with_paths, structure = jax.tree_util.tree_flatten_with_path(bank)
    assert tuple(path for path, _leaf in with_paths) == tuple(
        (jax.tree_util.GetAttrKey(name),) for name in NBEGMResourcesBank._fields
    )
    abstract = jax.eval_shape(lambda: bank)
    compiled = jax.jit(lambda original: original)(bank)
    roles = jax.tree.map(
        lambda leaf: StateAxesLeading(
            state_names=(), shape=leaf.shape, dtype=leaf.dtype
        ),
        bank,
    )
    for transported in (abstract, compiled, roles):
        assert type(transported) is NBEGMResourcesBank
        assert jax.tree.structure(transported) == structure
        for name in NBEGMResourcesBank._fields:
            original, actual = getattr(bank, name), getattr(transported, name)
            assert actual.shape == original.shape
            assert actual.dtype == original.dtype
            if transported is compiled:
                np.testing.assert_array_equal(actual, original)
    primal, tangent = jax.jvp(
        lambda scale: bank._replace(value=scale * bank.value),
        (jnp.asarray(2.0),),
        (jnp.asarray(3.0),),
    )
    assert type(primal) is type(tangent) is NBEGMResourcesBank
    np.testing.assert_array_equal(primal.value, 2 * bank.value)
    np.testing.assert_array_equal(tangent.value, 3 * bank.value)
    for name in NBEGMResourcesBank._fields:
        if name == "value":
            continue
        original, actual = getattr(bank, name), getattr(tangent, name)
        if jnp.issubdtype(original.dtype, jnp.bool_):
            assert actual.dtype == jax.dtypes.float0
        else:
            np.testing.assert_array_equal(actual, np.zeros(original.shape))
