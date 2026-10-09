"""Root manifests shared by tests that rebuild a model from another one."""

from lcm import Model


def initial_nodes_of(*, model: Model) -> dict[object, tuple[str, ...]]:
    """Return the `initial_nodes` mapping admitting exactly `model`'s starts."""
    names_by_age: dict[object, list[str]] = {}
    for age, name in sorted(model.graph.initial_nodes, key=repr):
        names_by_age.setdefault(age, []).append(name)
    return {age: tuple(names) for age, names in names_by_age.items()}
