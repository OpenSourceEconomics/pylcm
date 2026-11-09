"""Tiny enumeration oracles for the singleton-signature formulation.

Two independent enumerations of a component's Pareto frontier — set partitions and
cell-to-union-signature assignments — that the tests compare with the binary model.
Never the production path.
"""
from collections import defaultdict
from itertools import product
import math

from _lcm_regime_finder.signature_proof import (
    join, literal_partition_cost, pareto, rows, signature, union_closure, union_signatures,
    validate,
)


def all_partitions(items):
    """Tiny-only independent set-partition oracle; never the production path."""
    if not items:
        yield []
        return
    first, *rest = items
    for partition in all_partitions(rest):
        yield [[first], *[b[:] for b in partition]]
        for i in range(len(partition)):
            candidate = [b[:] for b in partition]
            candidate[i].append(first)
            yield candidate


def partition_frontier(instance, *, max_cells=7):
    if len(instance['cells']) > max_cells:
        raise ValueError('Tiny oracle cell limit')
    ids = sorted(c['id'] for c in instance['cells'])
    return pareto(cost for blocks in all_partitions(ids)
                  if (cost := literal_partition_cost(instance, blocks)) is not None)


def signature_assignment_frontier(instance, *, max_cells=7):
    """Different enumeration: cell-to-union-signature assignments, not partitions."""
    validate(instance)
    if len(instance['cells']) > max_cells:
        raise ValueError('Tiny oracle cell limit')
    cells = sorted(instance['cells'], key=lambda c: c['id'])
    bins = union_closure(instance)
    choices = [[k for k, sig in enumerate(bins) if join(signature(c), sig) == sig] for c in cells]
    values = []
    for labels in product(*choices):
        groups = defaultdict(list)
        for c, k in zip(cells, labels):
            groups[k].append(c)
        if any(union_signatures(signature(c) for c in group) != bins[k] for k, group in groups.items()):
            continue
        owner = {c['id']: k for c, k in zip(cells, labels)}
        w, b = instance['clarity_weights'], instance['boundary_unit']
        work = sum(rows(instance, bins[owner[c['id']]])[0] * rows(instance, bins[owner[c['id']]])[1] for c in cells)
        work += sum(b * math.prod(rows(instance, bins[owner[s]])) for s, t in set(map(tuple, instance['pairs']))
                    if t not in owner or owner[t] != owner[s])
        storage = sum(rows(instance, bins[owner[c['id']]])[0] for c in cells)
        clarity = w['dispatch'] * len(cells) + sum(
            w['block'] - w['dispatch'] + w['characteristic'] * len(bins[k][2]) for k in groups)
        values.append((work, storage, clarity))
    return pareto(values)
