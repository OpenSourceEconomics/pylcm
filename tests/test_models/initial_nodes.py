"""Initial-node declarations for test-model reconstruction."""

from lcm import InitialNodes, Model


def initial_nodes_of(*, model: Model) -> InitialNodes:
    """Return the model's immutable, explicitly labelled admissible starts."""
    return model.initial_nodes
