"""Original-input reconstruction and coverage checking for cost-family/v1.

No import of the cost-family producer or its component/signature constructors.
The original source semantics and legacy endpoint preference stay unproved.
"""
from __future__ import annotations

import hashlib
import json
import time
from .frontier_check import _Bounded
from .cost_limits import count_limits


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def _analysis(packet):
    if 'policy_encoding' not in packet:
        return packet
    encoding = packet['policy_encoding']
    names = encoding['cells']
    if len(names) != len(set(names)) or len(names) != len(encoding['basis']):
        raise ValueError('Invalid original cost policy basis')
    policies = []
    for row in encoding['policies']:
        chosen = dict(zip(names, encoding['basis']))
        for index, value in row['changes']:
            if type(index) is not int or not 0 <= index < len(names):
                raise ValueError('Invalid original cost policy coordinate')
            chosen[names[index]] = value
        mapping = {}
        for cell, index in chosen.items():
            offered = packet['interfaces'][cell]['options']
            if type(index) is not int or index < 0 or index >= len(offered):
                raise ValueError('Invalid original cost interface')
            mapping[cell] = offered[index]['id']
        policies.append({'id': row['id'], 'option_by_cell': mapping})
    if _hash([[p['id'], p['option_by_cell']] for p in policies]) != packet['original_policy_selection_sha256']:
        raise ValueError('Original cost policy identity mismatch')
    return {k: packet[k] for k in ('declarations', 'interfaces', 'derived_domains', 'discovery_complete',
        'clarity_weights', 'allocation_scenario', 'characteristic_candidates') if k in packet} | {
            'projection_policies': policies}


def _symbolic_packet(original, analysis):
    if 'policy_encoding' in original:
        return original
    names = sorted(analysis['interfaces'])
    policies = analysis['projection_policies']
    lookup = {c: {o['id']: i for i, o in enumerate(analysis['interfaces'][c]['options'])} for c in names}
    basis = [lookup[c][policies[0]['option_by_cell'][c]] for c in names]
    return {k: v for k, v in analysis.items() if k != 'projection_policies'} | {
        'schema': 'exact-graph-cost-instance/v1',
        'policy_encoding': {'cells': names, 'basis': basis, 'policies': [
            {'id': p['id'], 'changes': [[i, lookup[c][p['option_by_cell'][c]]]
                for i, c in enumerate(names) if lookup[c][p['option_by_cell'][c]] != basis[i]]} for p in policies]},
        'original_policy_selection_sha256': _hash([[p['id'], p['option_by_cell']] for p in policies])}


def check_cost_frontier(original, result, *, seconds=900, limits=None):
    seconds = float('inf') if seconds is None else seconds
    start = time.perf_counter()
    counts = count_limits(limits)
    family_limits = {'seconds': seconds, 'max_segments': counts['segments'],
                     'max_groupings': counts['groupings'], 'max_frontier_points': counts['frontier_points']}
    try:
        if result.get('schema') != 'regime-cost-family/v1' or result.get('input_sha256') != _hash(original):
            raise ValueError('Cost family input/schema identity differs')
        analysis = _analysis(original)
        if result.get('analysis_sha256') != _hash(analysis):
            raise ValueError('Cost family analysis identity differs')
        if (result.get('source_semantics') != 'NOT_ESTABLISHED_BY_COST_SEARCH'
                or result.get('legacy_endpoint_preference') != 'NOT_APPLIED'):
            raise ValueError('Cost proof was promoted to a source or legacy recommendation guarantee')
        from .frontier_check import IndependentFamily
        original_family = IndependentFamily(analysis, **family_limits)
        original_policy = analysis['projection_policies'][0]['option_by_cell']
        incumbent = result['original_representation']
        if incumbent['option_by_cell'] != original_policy:
            raise ValueError('Original incumbent policy changed')
        seen, total = set(), dict.fromkeys(('work', 'storage', 'clarity'), 0)
        groups = {}
        for cell in original_family.cells.values():
            key = json.dumps([original_family.role(cell), cell['regime']], sort_keys=True, separators=(',', ':'))
            groups.setdefault(key, {}).setdefault(cell['period'], []).append(cell['id'])
        expected = {key: [sorted(dates[date]) for date in sorted(dates)] for key, dates in groups.items()}
        if incumbent['groups'] != expected:
            raise ValueError('Original incumbent cell/date/group coverage differs')
        for blocks in expected.values():
            for ids in blocks:
                if seen.intersection(ids):
                    raise ValueError('Original incumbent repeats cells')
                seen.update(ids)
                block = original_family.block(ids, original_policy)
                if block is None:
                    raise ValueError('Original incumbent has incompatible domains')
                for name in total:
                    total[name] += block['cost'][name]
        if seen != set(original_family.cells) or total != incumbent['cost']:
            raise ValueError('Original incumbent costs or coverage changed')
        if result['status'] != 'COMPLETE_COST_FAMILY':
            return {'status': 'UNVERIFIED', 'cost_family_complete_verified': False,
                    'original_representation_verified': True,
                    'reason': 'A complete root coverage certificate is absent'}
        backend = result['backend']
        if backend == 'symbolic-policy':
            from .symbolic_frontier import check
            packet = _symbolic_packet(original, analysis)
            checked = check(packet=packet, certificate=result['proof'], max_seconds=seconds,
                            max_equations=counts['symbolic_operations'],
                            max_leaf_candidates=counts['symbolic_leaf_candidates'],
                            max_frontier_points=counts['frontier_points'])
        elif backend == 'ordered-domain-separator':
            from .separator_frontier import Problem, check
            guard = result['proof']['separator']
            problem = Problem(analysis, bridge=guard['bridge'], axis=guard['axis'])
            checked = check(problem, result['proof'], seconds=seconds, max_points=counts['separator_points'])
        elif backend == 'protected-components':
            from .frontier_check import IndependentFamily, check_frontier
            from .forest_check import verify
            family = IndependentFamily(analysis, **family_limits)
            if len(analysis['projection_policies']) != 1 or not analysis['discovery_complete']:
                raise ValueError('Fixed-policy certificate omits policy choices')
            policy = analysis['projection_policies'][0]['option_by_cell']
            components = [part for component in family.components() for part in family.separate_domains(component, policy)]
            records = result['components']
            if len(records) != len(components):
                raise ValueError('Original component omitted or added')
            combined = {(0, 0, 0)}
            for component, record in zip(components, records):
                ids = sorted(cell for group in component.values() for cell in group)
                if record['cells'] != ids:
                    raise ValueError('Original component cell coverage changed')
                if record['backend'] == 'singleton-forest':
                    checked = verify(family.signature_instance(component, policy), record['proof'])
                    vectors = [tuple(point['cost']) for point in record['proof']['frontier']]
                elif record['backend'] == 'legacy-exact-component':
                    selected = set(ids)
                    declaration = analysis['declarations']
                    subproblem = analysis | {'declarations': declaration | {
                        'cells': [c for c in declaration['cells'] if c['id'] in selected],
                        'relations': [r for r in declaration['relations'] if r['source'] in selected]},
                        'interfaces': {c: analysis['interfaces'][c] for c in selected},
                        'projection_policies': [{'id': 'selected-component',
                                                'option_by_cell': {c: policy[c] for c in selected}}]}
                    checked = check_frontier(subproblem, record['proof'], limits={
                        key: counts[key] for key in ('segments', 'groupings', 'frontier_points')} | {
                        'seconds': max(0, seconds - (time.perf_counter() - start))})
                    vectors = [tuple(p['cost'][k] for k in ('work', 'storage', 'clarity')) for p in record['proof']['frontier']]
                else:
                    raise ValueError('Unknown component certificate')
                if checked['status'] != 'PASS' or not checked.get('frontier_complete_verified'):
                    return {'status': checked['status'] if checked['status'] != 'PASS' else 'UNVERIFIED',
                            'cost_family_complete_verified': False, 'component_check': checked}
                sums = {tuple(x + y for x, y in zip(a, b)) for a in combined for b in vectors}
                combined = {a for a in sums if not any(b != a and all(x <= y for x, y in zip(b, a)) for b in sums)}
            if [list(v) for v in sorted(combined)] != [list(v) for v in result['frontier']]:
                raise ValueError('Root Pareto addition is not complete')
            checked = {'status': 'PASS', 'frontier_complete_verified': True}
        else:
            raise ValueError('Unknown cost family certificate backend')
        complete = checked['status'] in {'PASS', 'PASS_COST_FAMILY'} and bool(checked.get('frontier_complete_verified',
                                                              checked.get('cost_frontier_complete')))
        return {'status': 'PASS' if complete else checked['status'] if checked['status'] != 'PASS' else 'UNVERIFIED',
                'cost_family_complete_verified': complete, 'backend_check': checked,
                'source_semantics': 'NOT_CHECKED_HERE', 'legacy_endpoint_preference': 'NOT_APPLIED'}
    except _Bounded as exc:
        return {'status': 'UNVERIFIED', 'cost_family_complete_verified': False, 'reason': str(exc)}
    except (KeyError, IndexError, ValueError, TypeError, RuntimeError, RecursionError) as exc:
        return {'status': 'FAIL', 'cost_family_complete_verified': False, 'reason': str(exc)}
