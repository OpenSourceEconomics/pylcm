"""Observe what a `BufferRegistry` declares, for tests of its lifetime rule."""

from _lcm.execution.scheduler import BufferRegistry, ShardIdentity


def declared_shards(*, registry: BufferRegistry) -> frozenset[ShardIdentity]:
    """Return every shard identity a live declaring array still declares.

    Reads the registry's private state after the pruning every declaration runs.
    The registry's own query, `is_not_produced`, would add the queried array as a
    weak owner and so extend the lifetime a test is measuring.
    """
    registry._prune_dead_declarations()
    return frozenset(registry._unproduced_shards)
