"""Raw evaluation curves on the offered original-cell layout family.

No scalar score, improvement threshold, or layout recommendation is used. A
partial or heuristic curve describes retained feasible witnesses, not an optimum.
"""

from collections import defaultdict
import hashlib
import json

from .frontier_check import IndependentFamily

SCOPE = 'existing family, no dense start'


def _key(plan):
    return json.dumps([(b['cells'], b['option_by_cell']) for b in plan['blocks']],
                      sort_keys=True, separators=(',', ':'), allow_nan=False)


def _original_layout(family):
    """Find an offered identity policy; never synthesize a new family member."""
    for policy_record in family.analysis['projection_policies']:
        policy = policy_record['option_by_cell']
        for cid, cell in family.cells.items():
            option = family.options[cid][policy[cid]]
            if (any(option[kind] != cell[kind] for kind in ('states', 'actions'))
                    or any(value.get('kind') != 'identity'
                           for value in option['state_projection'].values())
                    or any(value.get('kind') != 'identity'
                           for value in option['action_reconstruction'].values())):
                break
        else:
            groups = defaultdict(list)
            for cid, cell in family.cells.items():
                groups[cell['regime']].append(cid)
            blocks = []
            total = dict.fromkeys(('work', 'storage', 'clarity'), 0)
            for _, members in sorted(groups.items()):
                ids = sorted(members)
                block = family.block(ids, policy)
                if block is None:
                    break
                for metric in total:
                    total[metric] += block['cost'][metric]
                blocks.append({**block, 'cells': ids,
                               'option_by_cell': {cid: policy[cid] for cid in ids}})
            else:
                return {'id': 'our-layout', 'projection_policy': policy_record['id'],
                        'blocks': blocks,
                        'cost': total}
    return None


def _bound_capture(family):
    original = family.declaration.get('original')
    if not isinstance(original, dict):
        return False
    source = original.get('graph_snapshot', original)
    if not isinstance(source, dict) or source.get('schema') != 'constructed-pylcm-model-graph-1':
        return False
    payload = json.dumps({'original': original,
                          'contract': family.declaration.get('original_contract', {})},
                         sort_keys=True, separators=(',', ':'), default=str)
    return family.declaration.get('source_identity') == hashlib.sha256(payload.encode()).hexdigest()


def _metrics(family, plan, *, capture_bound):
    errors = family.admitted_plan(plan)
    if errors:
        raise ValueError('Invalid layout witness: ' + '; '.join(errors))
    grid, links, link_count, link_units, maps, dispatch = 0, 0, 0, 0, 0, 0
    rectangle = 0
    arrays = []
    block_reports = []
    for block in plan['blocks']:
        ids = block['cells']
        ns, na = (family.sizes(block[kind]) for kind in ('state_axes', 'action_axes'))
        grid += len(ids) * ns * na
        links += block['cost']['work'] - len(ids) * ns * na
        members = set(ids)
        crossings = sum(source in members and target not in members for source, target in family.pairs)
        link_count += crossings
        link_units += crossings * ns * na
        selected = [family.options[cid][block['option_by_cell'][cid]] for cid in ids]
        labels = sorted({family.cells[cid]['regime'] for cid in ids})
        periods = sorted({family.cells[cid]['period'] for cid in ids})
        rectangular_work = len(labels) * len(periods) * ns * na
        rectangle += rectangular_work
        terminal_uniform = all(
            len(values := {family.cells[cid].get('terminal') for cid in ids
                           if family.cells[cid]['period'] == period}) == 1 and None not in values
            for period in periods)
        identity = all(
            all(option[kind] == family.cells[cid][kind] for kind in ('states', 'actions'))
            and all(p['kind'] == 'identity' for kind in ('state_projection', 'action_reconstruction')
                    for p in option[kind].values())
            for cid, option in zip(ids, selected))
        one_source = len(labels) == 1
        complete_original = one_source and members == {
            cid for cid, cell in family.cells.items() if cell['regime'] == labels[0]}
        criteria = {'terminal_uniform_by_period': terminal_uniform,
                    'source_identity_dispatch_available': one_source,
                    'unchanged_original_interfaces': identity,
                    'complete_captured_original_regime': complete_original,
                    'bound_constructed_model_capture': capture_bound}
        reasons = []
        if not terminal_uniform:
            reasons.append('Original terminal classifications differ within a period or are missing.')
        if not one_source:
            reasons.append('No emitted source-identity state and dispatcher witness exists for this pooled block.')
        if not identity:
            reasons.append('Projected function/interface implementations have not been emitted and constructed.')
        if not complete_original:
            reasons.append('The block is not one complete captured original regime with its constructor witness.')
        if not capture_bound:
            reasons.append('No original constructed-Model capture is bound to this cost analysis.')
        block_reports.append({'cells': ids, 'source_regimes': labels, 'periods': periods,
            'source_cell_count': len(ids), 'source_regime_count': len(labels),
            'period_count': len(periods), 'state_action_rows': ns * na,
            'source_cell_evaluations': len(ids) * ns * na,
            'emitted_rectangle_evaluations': rectangular_work,
            'rectangle_extra_evaluations': rectangular_work - len(ids) * ns * na,
            **criteria, 'emittable_as_one_regime': all(criteria.values()),
            'criteria': criteria, 'reasons': reasons,
            'emittability_scope': 'conservative captured-constructor witness; false means not established, not impossible'})
        maps += len({json.dumps(projection, sort_keys=True, separators=(',', ':'))
                     for option in selected for projection in option['state_projection'].values()
                     if projection['kind'] == 'characteristic'})
        dispatch += max(0, len({family.cells[cid]['regime'] for cid in ids}) - 1)
        arrays.extend((family.cells[cid]['period'], ns * na, ns) for cid in ids)
    largest = max((row[1] for row in arrays), default=0)
    largest_values = max((row[2] for row in arrays), default=0)
    return {'graph_id': plan.get('id'), 'evaluations': grid + links,
            'grid_evaluations': grid, 'link_charge_evaluations': links,
            'source_cell_evaluations': grid, 'emitted_rectangle_evaluations': rectangle,
            'rectangle_extra_evaluations': rectangle - grid, 'block_reports': block_reports,
            'directed_link_count': link_count, 'unpriced_link_row_units': link_units,
            'storage': plan['cost']['storage'], 'regimes': len(plan['blocks']),
            'largest_single_array': largest,
            'periods_at_largest_array': len({p for p, rows, _ in arrays if rows == largest}),
            'regime_period_arrays_at_largest_size': sum(rows == largest for _, rows, _ in arrays),
            'largest_stored_value_array': largest_values,
            'periods_at_largest_value_array': len({p for p, _, rows in arrays if rows == largest_values}),
            'characteristic_maps': maps, 'dispatch_clauses': dispatch,
            'witness': plan}


def build_curve_report(analysis, result, *, our_layout=None, membership=None, checking_limits=None):
    """Check retained witnesses and derive the at-most-k evaluation envelope.

    Completeness requires a completed producer and its independent full-family
    check, as well as complete offered-policy discovery. Source semantics remain
    a separate obligation. An explicitly supplied membership record is evidence
    to retain; it never overrides the independent layout-membership check.
    """
    family = IndependentFamily(analysis)
    capture_bound = _bound_capture(family)
    if result.get('schema') not in ('regime-frontier/v6', 'regime-heuristic-frontier/v1'):
        raise ValueError('A current-family curve cannot reinterpret a legacy admission certificate')
    if family.weights != {'block': 1, 'characteristic': 0, 'dispatch': 0}:
        raise ValueError('A regime-count curve requires block/characteristic/dispatch prices 1/0/0')
    if family.boundary_unit != 0:
        raise ValueError('A grid-evaluation curve requires boundary_unit=0 before searching')
    points = result.get('frontier', [])
    frontier = [_metrics(family, point, capture_bound=capture_bound) for point in points]
    if result.get('schema') == 'regime-heuristic-frontier/v1':
        from .heuristic_check import check_heuristic_frontier
        checked = check_heuristic_frontier(analysis, result, limits=checking_limits)
    else:
        from .frontier_check import check_frontier
        checked = check_frontier(analysis, result, limits=checking_limits)
    if checked.get('status') == 'FAIL':
        raise ValueError('Invalid frontier: ' + '; '.join(checked.get('errors', [])))
    complete = (result.get('status') == 'COMPLETE'
                and checked.get('status') == 'PASS'
                and checked.get('frontier_complete_verified') is True
                and analysis.get('discovery_complete') is True)
    curve = []
    if frontier:
        for k in range(min(point['regimes'] for point in frontier), len(family.cells) + 1):
            eligible = [point for point in frontier if point['regimes'] <= k]
            best = min(eligible, key=lambda point: (point['evaluations'], point['regimes'],
                                                    point['storage'], _key(point['witness'])))
            curve.append({'at_most_regimes': k,
                          **{key: value for key, value in best.items()
                             if key not in ('witness', 'block_reports')}})
    original = our_layout if our_layout is not None else _original_layout(family)
    if original is None:
        our = {'status': 'not_admitted', 'curve_attained': None,
               'reason': 'No offered coherent identity policy admits the unchanged original-regime blocks.'}
    else:
        errors = family.admitted_plan(original)
        if errors:
            our = {'status': 'not_admitted', 'curve_attained': None, 'reasons': errors}
        else:
            our = _metrics(family, original, capture_bound=capture_bound)
            cost = (our['evaluations'], our['storage'], our['regimes'])
            dominating = [point for point in frontier
                          if (vector := (point['evaluations'], point['storage'], point['regimes'])) != cost
                          and all(a <= b for a, b in zip(vector, cost))]
            our['status'] = ('dominated' if dominating else
                             'on_frontier' if complete else 'not_dominated_by_retained_candidates')
            our['dominating_layout'] = min(dominating, key=lambda p: (
                p['evaluations'], p['regimes'], p['storage'], _key(p['witness']))) if dominating else None
            eligible = [point for point in frontier if point['regimes'] <= our['regimes']]
            best = min((point['evaluations'] for point in eligible), default=None)
            our['curve_attained'] = best == our['evaluations'] if best is not None else None
    if membership is not None:
        our['supplied_membership_evidence'] = membership
    return {'schema': 'regime-evaluation-curve/v1', 'scope': SCOPE,
            'status': 'COMPLETE' if complete else 'PARTIAL',
            'search_status': result.get('status'), 'independent_check': checked,
            'frontier': frontier, 'curve': curve, 'our_layout': our,
            'cost_contract': {'evaluations': 'grid evaluations only; directed links are reported, not priced',
                'storage': 'sum of per-source-cell stored value entries under each block domain',
                'largest_single_array': 'state-action entries in one source regime-period array',
                'periods_at_largest_array': 'distinct original periods, not summed regime-period arrays',
                'characteristic_maps': 'distinct declared maps per block, summed across blocks',
                'dispatch_clauses': 'extra original regime labels per block, summed across blocks',
                'tie_rule': 'fewest evaluations, then fewest regimes, then least storage, then canonical witness',
                'units': 'structural entry counts; no measured runtime, bytes or peak live memory'},
            'emission_contract': {'search_objective_changed': False,
                'rectangle': 'distinct original source regimes times block state-action rows times distinct active period buckets',
                'emittable_as_one_regime': 'true only for a complete captured single-regime identity witness; false means not established',
                'capture_authority': 'recorded original constructed-Model capture with matching source binding; not a certificate of a new emission or original economics',
                'new_regime_emission': 'NOT_IMPLEMENTED_OR_CERTIFIED'},
            'source_semantics': 'NOT_ESTABLISHED_BY_COST_SEARCH',
            'discovery_complete': analysis.get('discovery_complete') is True,
            'exhaustion_reason': result.get('exhaustion_reason')}
