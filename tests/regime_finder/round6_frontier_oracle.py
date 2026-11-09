"""Literal finite-family oracle: no producer or checker imports.

Enumerate every original-label set partition and every exact date cut mask.
This is intentionally exponential and is restricted to tiny metadata fixtures.
No callback, solver, numerical model row, or production Cartesian grid is used.
"""
from collections import defaultdict
from itertools import combinations, product
import json
import math
from typing import TypedDict


class Block(TypedDict):
    cells: list[str]
    option_by_cell: dict[str, str]
    cost: dict[str, int]


class Candidate(TypedDict):
    blocks: list[Block]
    projection_policy: str
    cost: dict[str, int]

METRICS = ('work', 'storage', 'clarity')


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def context(cell):
    role = cell.get('role')
    if role is None:
        role = {'unresolved_original_role': cell['regime']}
    return canonical([role, cell.get('ownership'), cell.get('stage'), bool(cell.get('terminal')),
                      cell.get('declaration_type'), cell.get('structural_roles', {}), cell.get('solver_identity')])


def block(analysis, members, policy):
    cells = {cell['id']: cell for cell in analysis['declarations']['cells']}
    selected = [cells[cell] for cell in members]
    if not selected or len(set(members)) != len(members) or len({context(cell) for cell in selected}) != 1:
        return None
    options = [next(option for option in analysis['interfaces'][cell]['options'] if option['id'] == policy[cell]) for cell in members]
    allocation = analysis.get('allocation_scenario', analysis['declarations']['allocation'])
    domains = {**analysis['declarations']['domains'], **analysis.get('derived_domains', {})}
    axes = []
    for kind in ('states', 'actions'):
        union = {}
        if allocation['kind'] == 'global_union' and allocation.get(kind) is not None:
            union = allocation[kind]
        else:
            source = cells.values() if allocation['kind'] == 'global_union' else options
            for obj in source:
                for name, domain in obj[kind].items():
                    if name in union and union[name] != domain:
                        return None
                    union[name] = domain
        axes.append(union)
    state_rows, action_rows = [math.prod(domains[domain]['size'] for domain in union.values()
        if domains[domain].get('solve_axis') is not False) for union in axes]
    pairs = {(relation['source'], target) for relation in analysis['declarations']['relations']
        if relation.get('support') != 'zero' for target in relation.get('targets', [])}
    outgoing = len({(source, target) for source, target in pairs if source in members and target not in members})
    characteristics = {canonical(value) for option in options for value in option['state_projection'].values()
                       if value['kind'] == 'characteristic'}
    weights = {'block': 4, 'characteristic': 2, 'dispatch': 1, **analysis.get('clarity_weights', {})}
    cost = {'work': (len(members) + outgoing * allocation.get('boundary_unit', 1)) * state_rows * action_rows,
            'storage': len(members) * state_rows,
            'clarity': weights['block'] + weights['characteristic'] * len(characteristics)
                       + weights['dispatch'] * (len({cell['regime'] for cell in selected}) - 1)}
    return {'cells': sorted(members), 'option_by_cell': {c: policy[c] for c in sorted(members)}, 'cost': cost}


def partitions(labels):
    if not labels:
        yield []
        return
    first, rest = labels[0], labels[1:]
    for count in range(len(rest) + 1):
        for subset in combinations(rest, count):
            for tail in partitions([label for label in rest if label not in subset]):
                yield [[first, *subset], *tail]


def interval_layouts(analysis, ids, policy):
    cells = {cell['id']: cell for cell in analysis['declarations']['cells']}
    buckets = defaultdict(list)
    for c in ids:
        buckets[cells[c]['period']].append(c)
    dates = [sorted(values) for _, values in sorted(buckets.items())]
    for mask in range(1 << max(0, len(dates) - 1)):
        current, layout = [], []
        for i, bucket in enumerate(dates):
            current.extend(bucket)
            if i == len(dates) - 1 or mask & (1 << i):
                candidate = block(analysis, current, policy)
                if candidate is None:
                    break
                layout.append(candidate)
                current = []
        else:
            yield layout


def brute_family(analysis):
    components = defaultdict(lambda: defaultdict(list))
    for cell in analysis['declarations']['cells']:
        components[context(cell)][cell['regime']].append(cell['id'])
    all_candidates: list[Candidate] = []
    for policy_record in analysis['projection_policies']:
        policy = policy_record['option_by_cell']
        choices = []
        for _, component in sorted(components.items()):
            layouts = []
            for partition in partitions(sorted(component)):
                by_group = [list(interval_layouts(analysis, [c for label in group for c in component[label]], policy)) for group in partition]
                layouts.extend([b for layout in selected for b in layout] for selected in product(*by_group))
            choices.append(layouts)
        for selected in product(*choices):
            blocks = [b for layout in selected for b in layout]
            all_candidates.append({'blocks': blocks, 'projection_policy': policy_record['id'],
                'cost': {metric: sum(b['cost'][metric] for b in blocks) for metric in METRICS}})
    kept = {}
    for candidate in all_candidates:
        cost = tuple(candidate['cost'][metric] for metric in METRICS)
        key = canonical([[b['cells'], b['option_by_cell']] for b in candidate['blocks']])
        if cost not in kept or key < kept[cost][0]:
            kept[cost] = (key, candidate)
    return [kept[cost][1] for cost in sorted(kept) if not any(
        other != cost and all(a <= b for a, b in zip(other, cost)) for other in kept)]
