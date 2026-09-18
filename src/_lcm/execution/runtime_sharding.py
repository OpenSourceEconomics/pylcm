"""Compare physical runtime layouts without changing declared lowering identity."""

import jax


def runtime_shardings_match(
    *, actual: object, expected: jax.sharding.Sharding, ndim: int
) -> bool:
    """Accept equivalent Named placement, preserving mesh shape and device order.

    Trailing replicated dimensions and Auto/Explicit mesh typing can differ
    while every logical shard still resides on the same device. Neither changes
    an array's physical placement. This predicate is not a compilation key.
    """
    if actual == expected:
        return True
    if not isinstance(actual, jax.NamedSharding) or not isinstance(
        expected, jax.NamedSharding
    ):
        return False
    return (
        actual.mesh.devices.shape == expected.mesh.devices.shape
        and actual.mesh.axis_names == expected.mesh.axis_names
        and tuple(actual.mesh.devices.flat) == tuple(expected.mesh.devices.flat)
        and actual.memory_kind == expected.memory_kind
        and actual.is_equivalent_to(expected, ndim)
    )
