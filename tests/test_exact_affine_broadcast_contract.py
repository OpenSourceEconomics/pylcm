"""The exact-affine FFI publishes its broadcast operands as a tuple.

`jnp.broadcast_arrays` does not promise a container type, and it has differed
between jax patch releases. The helper's declared return type is a runtime
contract under the beartype claw, so the container has to be built rather than
borrowed from whatever the library happens to hand back.
"""

import jax.numpy as jnp

from _lcm.egm.upper_envelope._exact_affine import ffi


def test_broadcast_publishes_a_tuple_when_jax_returns_a_list(monkeypatch):
    """The helper's published container does not depend on jax's."""
    monkeypatch.setattr(ffi.jnp, "broadcast_arrays", lambda *operands: list(operands))

    result = ffi._broadcast(jnp.asarray(1.0), jnp.asarray([1.0, 2.0]))

    assert isinstance(result, tuple)
