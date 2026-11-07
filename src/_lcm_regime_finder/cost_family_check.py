"""Original-input reconstruction and coverage checking for cost-family/v1.

No import of the cost-family producer or its component/signature constructors.
The original source semantics and legacy endpoint preference stay unproved.
"""
from __future__ import annotations

import hashlib
import json
import time
from fractions import Fraction
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


def _vectors(result):
    if result['backend'] == 'protected-components':
        return [tuple(v) for v in result['frontier']]
    if result['backend'] == 'ordered-domain-separator':
        return [tuple(p['cost'][k] for k in ('work', 'storage', 'clarity')) for p in result['proof']['frontier']]
    return [tuple(v) for v in result['proof']['frontier']]


def cost_query(original, result, *, upper_bounds=None, dominated_vector=None, weights=None, seconds=900, limits=None):
    """Independently verified threshold/dominance/recommendation with one witness."""
    seconds = float('inf') if seconds is None else seconds
    counts = count_limits(limits)
    checked = check_cost_frontier(original, result, seconds=seconds, limits=counts)
    if not checked.get('cost_family_complete_verified'):
        return checked
    points = _vectors(result)
    for value in (upper_bounds, dominated_vector):
        if value is not None and (len(value) != 3 or any(type(v) is not int for v in value)):
            raise ValueError('Cost bounds require an integer triple')
    if upper_bounds is not None:
        points = [p for p in points if all(a <= b for a, b in zip(p, upper_bounds))]
    if dominated_vector is not None:
        points = [p for p in points if p != tuple(dominated_vector)
                  and all(a <= b for a, b in zip(p, dominated_vector))]
    if weights is not None:
        if len(weights) != 3:
            raise ValueError('Cost weights require a positive rational triple')
        coefficients = [Fraction(str(w)) for w in weights]
        if any(w <= 0 for w in coefficients):
            raise ValueError('Cost weights must be strictly positive')
        points.sort(key=lambda p: (sum(w * v for w, v in zip(coefficients, p)), p))
    else:
        points.sort()
    response = {'status': 'PASS', 'answer': bool(points), 'input_sha256': _hash(original),
                'certificate_sha256': _hash(result), 'legacy_endpoint_preference': 'NOT_APPLIED',
                'source_semantics': 'NOT_CHECKED_HERE'}
    if not points:
        return response
    selected = points[0]
    analysis = _analysis(original)
    backend = result['backend']
    if backend == 'symbolic-policy':
        from .symbolic_frontier import recover_witness
        witness = recover_witness(packet=_symbolic_packet(original, analysis), certificate=result['proof'], cost=selected,
                                  max_seconds=seconds, max_equations=counts['symbolic_operations'],
                                  max_leaf_candidates=counts['symbolic_leaf_candidates'],
                                  max_frontier_points=counts['frontier_points'])
    elif backend == 'ordered-domain-separator':
        witness = next(p for p in result['proof']['frontier']
                       if tuple(p['cost'][k] for k in ('work', 'storage', 'clarity')) == selected)
    else:
        from .frontier_check import IndependentFamily
        from .forest_check import materialize_verified_plan
        # Preserve one feasible predecessor per Pareto cost, not assignments for
        # every recurrence point. All component proofs were already replayed.
        partial = {(0, 0, 0): []}
        for record in result['components']:
            choices = ([tuple(p['cost']) for p in record['proof']['frontier']] if record['backend'] == 'singleton-forest'
                       else [tuple(p['cost'][k] for k in ('work', 'storage', 'clarity')) for p in record['proof']['frontier']])
            candidates = {}
            for old, path in partial.items():
                for index, vector in enumerate(choices):
                    total = tuple(a + b for a, b in zip(old, vector))
                    if all(a <= b for a, b in zip(total, selected)):
                        candidates.setdefault(total, path + [index])
            partial = {p: path for p, path in candidates.items()
                       if not any(q != p and all(a <= b for a, b in zip(q, p)) for q in candidates)}
        if selected not in partial:
            raise ValueError('Verified root cost has no component witness')
        family = IndependentFamily(analysis, seconds=seconds, max_segments=counts['segments'],
            max_groupings=counts['groupings'], max_frontier_points=counts['frontier_points'])
        policy = analysis['projection_policies'][0]['option_by_cell']
        components = [part for c in family.components() for part in family.separate_domains(c, policy)]
        witness = []
        for component, record, index in zip(components, result['components'], partial[selected]):
            if record['backend'] == 'singleton-forest':
                witness.append(materialize_verified_plan(family.signature_instance(component, policy),
                                                         record['proof'], frontier_index=index))
            else:
                witness.append(record['proof']['frontier'][index])
    return response | {'witness_cost': selected, 'witness': witness}
