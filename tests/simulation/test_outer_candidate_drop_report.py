"""Why N-NB-EGM replay takes adjuster targets from the declared outer nodes.

N-NB-EGM replay recovers the outer action from the target the solve retained. At
a realized state that action can reach a stock outside the outer state's declared
domain, where no value function exists; the candidate is then dropped from the
affected subjects' choice sets rather than published.
"""

import jax.numpy as jnp

from _lcm.simulation.simulate import map_coordinates


def test_interpolating_a_constant_target_surface_does_not_reproduce_its_node() -> None:
    """A constant surface read off the candidate bank misses its own node.

    This is why an adjuster candidate's target comes from the declared outer
    search nodes rather than from the interpolated bank: the bank's adjuster
    rows are constant along the state axes, so reading them "should" return the
    node, yet linear interpolation forms `(1 - w) * c + w * c`, which rounds. A
    target a rounding away from a domain endpoint is not that endpoint, and the
    candidate would then be admitted on containment and reach a stock outside
    the domain.
    """
    node = jnp.float32(20.0)
    surface = jnp.full((10,), node, dtype=jnp.float32)
    coordinates = [jnp.linspace(0.0, 9.0, 100_001, dtype=jnp.float32)]

    read = map_coordinates(input=surface, coordinates=coordinates)

    assert read.dtype == jnp.float32
    assert bool(jnp.isfinite(read).all())
    assert int(jnp.sum(read != node)) > 0
