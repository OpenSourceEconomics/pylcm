"""Exact cost-family search, separate from native/source and display-tie claims.

The input is the existing metadata analysis or its lossless cost packet. These
certificates have their own version; saved frontier v2-v5 meanings are unchanged.
"""
from __future__ import annotations

from itertools import product
import time

from .frontier import Exhausted, Search, canonical, digest, search_frontier
from .cost_limits import count_limits

SCHEMA = 'regime-cost-family/v1'
SCOPE = 'Exact integer Pareto costs with feasible representatives; source semantics and legacy endpoint/layout preferences are separate'


def analysis_from_packet(packet):
    if 'policy_encoding' not in packet:
        return packet
    encoding = packet['policy_encoding']
    cells = encoding['cells']
    if len(cells) != len(set(cells)) or len(cells) != len(encoding['basis']):
        raise ValueError('Invalid original policy basis')
    policies = []
    for row in encoding['policies']:
        choices = list(encoding['basis'])
        for index, value in row['changes']:
            if type(index) is not int or not 0 <= index < len(cells):
                raise ValueError('Invalid policy coordinate')
            choices[index] = value
        if any(type(index) is not int or index < 0 or index >= len(packet['interfaces'][cell]['options'])
               for cell, index in zip(cells, choices)):
            raise ValueError('Invalid selected interface index')
        policies.append({'id': row['id'], 'option_by_cell': {
            cell: packet['interfaces'][cell]['options'][choices[i]]['id'] for i, cell in enumerate(cells)}})
    if digest([[p['id'], p['option_by_cell']] for p in policies]) != packet['original_policy_selection_sha256']:
        raise ValueError('Original policy selections changed')
    return {key: packet[key] for key in ('declarations', 'interfaces', 'derived_domains',
        'discovery_complete', 'clarity_weights', 'allocation_scenario', 'characteristic_candidates') if key in packet} | {
            'projection_policies': policies}


def encode_analysis(analysis):
    """Keep the existing analysis exact while encoding offered policy selections."""
    cells = sorted(analysis['interfaces'])
    indices = {cell: {o['id']: i for i, o in enumerate(analysis['interfaces'][cell]['options'])} for cell in cells}
    policies = analysis['projection_policies']
    basis = [indices[cell][policies[0]['option_by_cell'][cell]] for cell in cells]
    return {key: value for key, value in analysis.items() if key != 'projection_policies'} | {
        'schema': 'exact-graph-cost-instance/v1',
        'policy_encoding': {'cells': cells, 'basis': basis, 'policies': [
            {'id': p['id'], 'changes': [[i, indices[cell][p['option_by_cell'][cell]]]
                for i, cell in enumerate(cells) if indices[cell][p['option_by_cell'][cell]] != basis[i]]}
            for p in policies]},
        'original_policy_selection_sha256': digest([[p['id'], p['option_by_cell']] for p in policies])}


def chain_cover(candidate, cells=()):
    """An exact DAG path cover; cross-chain implications stay in the optimizer."""
    nodes = sorted(candidate['cells'])
    edges = {node: sorted(set(candidate['requires'].get(node, ())) - {node}) for node in nodes}
    metadata = {cell['id']: cell for cell in cells}
    def edge_order(source, target):
        left, right = metadata.get(source, {}), metadata.get(target, {})
        same_regime = left.get('regime') == right.get('regime')
        difference = sum(len(set(left.get(kind, {}).items()) ^ set(right.get(kind, {}).items()))
                         for kind in ('states', 'actions'))
        return (not same_regime, difference, right.get('period', 0), target)
    for node in edges:
        edges[node].sort(key=lambda target: edge_order(node, target))
    if any(target not in edges for targets in edges.values() for target in targets):
        raise ValueError('Implication endpoint outside the original candidate')
    incoming = dict.fromkeys(nodes, 0)
    for targets in edges.values():
        for node in targets:
            incoming[node] += 1
    ready = sorted(node for node in nodes if not incoming[node])
    order = []
    while ready:
        node = ready.pop()
        order.append(node)
        for target in edges[node]:
            incoming[target] -= 1
            if not incoming[target]:
                ready.append(target)
    if len(order) != len(nodes):
        raise ValueError('Cyclic implications need an SCC-aware search route')
    matched = {}
    def augment(node, seen):
        for target in edges[node]:
            if target in seen:
                continue
            seen.add(target)
            if target not in matched or augment(matched[target], seen):
                matched[target] = node
                return True
        return False
    for node in order:
        augment(node, set())
    successor = {node: target for target, node in matched.items()}
    chains = []
    for node in nodes:
        if node in matched:
            continue
        chain = [node]
        while chain[-1] in successor:
            chain.append(successor[chain[-1]])
        chains.append(chain)
    # This orders symbolic coordinates only; every original implication remains.
    owner = {cell: i for i, chain in enumerate(chains) for cell in chain}
    chains.sort(key=lambda chain: (-sum(owner[source] != owner[target]
        for source in chain for target in edges[source]), -len(chain), chain))
    return chains


def separator_problem(analysis):
    """Find the ordered incompatible-domain guard without naming a paper/axis."""
    from .separator_frontier import Problem
    axes = sorted({axis for record in analysis['interfaces'].values() for option in record['options']
                   for axis in option['states']})
    labels = sorted({cell['regime'] for cell in analysis['declarations']['cells']})
    policy = analysis['projection_policies'][0]['option_by_cell']
    selected = {cell: next(o for o in record['options'] if o['id'] == policy[cell])
                for cell, record in analysis['interfaces'].items()}
    for axis in axes:
        domains = {label: {selected[cell['id']]['states'].get(axis)
            for cell in analysis['declarations']['cells'] if cell['regime'] == label} for label in labels}
        if any(len(values) != 1 for values in domains.values()):
            continue
        bridges = [label for label in labels if domains[label] == {None}]
        if len(bridges) != 1 or len(set().union(*domains.values()) - {None}) != 2:
            continue
        for bridge in bridges:
            try:
                return Problem(analysis, bridge=bridge, axis=axis)
            except ValueError:
                continue
    return None


def _piece(analysis, cells, policy):
    selected = set(cells)
    declaration = analysis['declarations']
    return analysis | {'declarations': declaration | {
        'cells': [cell for cell in declaration['cells'] if cell['id'] in selected],
        'relations': [row for row in declaration['relations'] if row['source'] in selected]},
        'interfaces': {cell: analysis['interfaces'][cell] for cell in selected},
        'projection_policies': [{'id': 'selected-component', 'option_by_cell': {cell: policy[cell] for cell in selected}}]}


def _pareto(candidates):
    costs = []
    for vector in candidates:
        if any(all(a <= b for a, b in zip(old, vector)) for old in costs):
            continue
        costs = [old for old in costs if not all(a <= b for a, b in zip(vector, old))]
        costs.append(tuple(vector))
    return sorted(costs)


def _original_representation(search, policy):
    groups, costs = {}, dict.fromkeys(('work', 'storage', 'clarity'), 0)
    for component in search.components():
        for label, ids in component.items():
            dates = sorted({search.cells[cell]['period'] for cell in ids})
            blocks = [[cell for cell in sorted(ids) if search.cells[cell]['period'] == date] for date in dates]
            for cells in blocks:
                block = search.block(cells, policy)
                if block is None:
                    raise ValueError('Original date bucket has incompatible cost domains')
                for name in costs:
                    costs[name] += block['cost'][name]
            groups[canonical([search.roles[ids[0]], label])] = blocks
    return {'groups': groups, 'cost': costs, 'option_by_cell': policy,
            'meaning': 'Valid original labels with all original date buckets; an incumbent, not an optimum'}


def find_cost_frontier(problem, *, seconds=900, limits=None, checkpoint=None):
    """Return an independently replayable cost proof, never a native certificate."""
    seconds = float('inf') if seconds is None else seconds
    if seconds < 0:
        raise ValueError('Metadata budget must be nonnegative')
    counts = count_limits(limits)
    start = time.perf_counter()
    remaining = lambda: max(0, seconds - (time.perf_counter() - start))
    analysis = analysis_from_packet(problem)
    result = {'schema': SCHEMA, 'input_sha256': digest(problem), 'analysis_sha256': digest(analysis),
              'scope': SCOPE, 'source_semantics': 'NOT_ESTABLISHED_BY_COST_SEARCH',
              'legacy_endpoint_preference': 'NOT_APPLIED', 'status': 'INCOMPLETE',
              'requested_bookkeeping_limits': {key: None if value == float('inf') else value for key, value in counts.items()}}
    if analysis['declarations'].get('analysis_scope') != 'pylcm-structural-graph/v1':
        return result | {'reason': 'Constructed-Model structural scope is required'}
    if analysis.get('allocation_scenario', analysis['declarations']['allocation']).get('kind') != 'per_block':
        return result | {'reason': 'Per-block storage is required'}
    search_limits = {key: counts[key] for key in ('segments', 'groupings', 'frontier_points')}
    search = Search(analysis, limits=search_limits | {'seconds': remaining()})
    policy = analysis['projection_policies'][0]['option_by_cell']
    result['original_representation'] = _original_representation(search, policy)
    if checkpoint:
        checkpoint(result)
    if remaining() <= 0:
        return result | {'reason': 'Metadata time budget before exact cost search'}
    if analysis.get('characteristic_candidates'):
        if len(analysis['characteristic_candidates']) != 1:
            return result | {'reason': 'The symbolic route requires one complete categorical candidate'}
        from .symbolic_frontier import solve
        packet = problem if 'policy_encoding' in problem else encode_analysis(analysis)
        try:
            chains = chain_cover(packet['characteristic_candidates'][0], analysis['declarations']['cells'])
        except ValueError as exc:
            return result | {'reason': str(exc)}
        proof = solve(packet=packet, chains=chains, seconds=remaining(),
                      max_nodes=counts['symbolic_nodes'], max_operations=counts['symbolic_operations'],
                      max_frontier_points=counts['frontier_points'])
        return result | {'backend': 'symbolic-policy', 'proof': proof,
            'status': 'COMPLETE_COST_FAMILY' if proof['status'] == 'COMPLETE_COST_FAMILY' else 'INCOMPLETE'}
    if len(analysis['projection_policies']) != 1 or not analysis['discovery_complete']:
        return result | {'reason': 'Fixed-policy route requires a complete single policy family'}
    separator = separator_problem(analysis)
    if separator is not None:
        from .separator_frontier import produce
        proof = produce(separator, seconds=remaining(), max_points=counts['separator_points'])
        return result | {'backend': 'ordered-domain-separator', 'proof': proof,
            'status': 'COMPLETE_COST_FAMILY' if proof['status'] == 'COMPLETE' else 'INCOMPLETE'}
    try:
        components = [part for component in search.components() for part in search.domain_components(component, policy)]
    except Exhausted as exc:
        return result | {'reason': str(exc)}
    records, combined = [], [(0, 0, 0)]
    for component in components:
        ids = sorted(cell for group in component.values() for cell in group)
        instance = None
        if all(len(group) == 1 for group in component.values()):
            from .forest_frontier import _prepare, optimize
            instance = search.signature_instance(component, policy)
            try:
                _prepare(instance)
            except ValueError:
                instance = None
        if instance is not None:
            proof = optimize(instance, seconds=remaining(), max_states=counts['forest_states'],
                             max_points=counts['forest_points'], checkpoint=checkpoint)
            record = {'backend': 'singleton-forest', 'cells': ids, 'proof': proof}
            costs = [point['cost'] for point in proof['frontier']]
        else:
            proof = search_frontier(_piece(analysis, ids, policy), limits=search_limits | {'seconds': remaining()})
            record = {'backend': 'legacy-exact-component', 'cells': ids, 'proof': proof}
            costs = [[point['cost'][name] for name in ('work', 'storage', 'clarity')] for point in proof['frontier']]
        records.append(record)
        if checkpoint:
            checkpoint(result | {'backend': 'protected-components', 'components': records,
                                 'completed_component_count': sum(r['proof']['status'] == 'COMPLETE' for r in records)})
        if proof['status'] != 'COMPLETE':
            return result | {'backend': 'protected-components', 'components': records,
                             'reason': proof.get('exhaustion_reason')}
        def sums():
            for left, right in product(combined, costs):
                if remaining() <= 0:
                    raise Exhausted('Root Pareto addition time budget')
                yield tuple(a + b for a, b in zip(left, right))
        try:
            combined = _pareto(sums())
        except Exhausted as exc:
            return result | {'backend': 'protected-components', 'components': records, 'reason': str(exc)}
    return result | {'status': 'COMPLETE_COST_FAMILY', 'backend': 'protected-components',
                     'components': records, 'frontier': combined}
