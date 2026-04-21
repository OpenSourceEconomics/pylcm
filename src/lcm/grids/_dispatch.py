"""Per-dimension dispatch strategies for `DiscreteGrid` states.

The enum below controls how a discrete state dimension is compiled into
the Bellman kernel — where in the XLA program it lives (inside the
state-action space vs lifted to the top-level partition axis), and
which JAX primitive sweeps it (`jax.vmap`, `jax.lax.map`, or
`jax.lax.scan`). The trade-offs around memory, parallelism, compile
time, and multi-GPU readiness are documented at
`docs/user_guide/dispatch.md`.
"""

from enum import StrEnum


class DispatchStrategy(StrEnum):
    """How a `DiscreteGrid` dim is compiled into the Bellman kernel.

    See `docs/user_guide/dispatch.md` for the full trade-off table
    (memory / parallelism / compile cost / multi-GPU readiness) and
    a decision tree for picking a strategy per dim.
    """

    FUSED_VMAP = "fused_vmap"
    """Default. The dim lives in the state-action space; `jax.vmap`
    fuses it with the other state / action axes into one XLA kernel.
    Maximum single-GPU parallelism; memory scales with the full
    Cartesian product of all state / action sizes."""

    CHUNKED_LAX_MAP = "chunked_lax_map"
    """The dim stays in the state-action space but is swept by
    `jax.lax.map(batch_size=k)` with a user-chosen `batch_size`.
    Memory-bounded alternative to `FUSED_VMAP`. A `batch_size` of 1
    gives a fully serial scan; higher values give chunked parallelism.
    """

    PARTITION_SCAN = "partition_scan"
    """The dim is lifted out of the state-action space and swept by
    `jax.lax.scan` at the kernel's top level. Requires this state's
    transition to be the identity in every regime
    (`state_transitions[name] = None`). Minimal per-device memory;
    the axis stays JAX-visible so a future `shard_map` multi-device
    swap is a drop-in replacement at the wrap site."""

    PARTITION_VMAP = "partition_vmap"
    """Same as `PARTITION_SCAN` except the top-level sweep is
    `jax.vmap`. On a single GPU this is equivalent to `FUSED_VMAP`
    with partition bookkeeping on top, so it is mainly useful once
    `shard_map` multi-device dispatch is wired up — at that point
    `shard_map` shards the partition axis across devices and the
    inner `jax.vmap` keeps per-device parallelism."""

    @property
    def is_partition_lifted(self) -> bool:
        """True when the dim is lifted out of the state-action space."""
        return self in (
            DispatchStrategy.PARTITION_SCAN,
            DispatchStrategy.PARTITION_VMAP,
        )

    @property
    def requires_batch_size(self) -> bool:
        """True when the strategy requires a companion `batch_size` kwarg."""
        return self is DispatchStrategy.CHUNKED_LAX_MAP
