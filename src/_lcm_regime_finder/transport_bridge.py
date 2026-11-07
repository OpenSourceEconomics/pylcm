"""Materialize a generated finite quotient and run the inherited transport checker.

This is an execution bridge for the scalar finite-grid IR, not an adapter for
opaque native callbacks.  The *reported* per-atom interfaces determine the
projection.  Reduced primitives are evaluated at one fixed original state in
each fiber; the finite transport checker then compares every original state/action.
No desired cuts, hand-authored quotient, solved values or oracle answers enter
the construction.  Dense allocation padding is deliberately not decision state.
"""
from __future__ import annotations

from collections import defaultdict
from fractions import Fraction
import hashlib
import itertools
import json
from pathlib import Path

from .core import canonical, digest, prepare, values
from .expressions import AdmissionError, evaluate, literal


class _BridgeStop(Exception):
    def __init__(self, status, code, reason, **witness):
        self.status, self.code, self.reason, self.witness = status, code, reason, witness
        super().__init__(reason)


def _stop(status, code, reason, **witness):
    raise _BridgeStop(status, code, reason, **witness)


def _wire(value):
    """Use the IR's decimal-literal interpretation, never binary float arithmetic."""
    if isinstance(value, float):
        value = literal(value)
    if isinstance(value, Fraction):
        return value.numerator if value.denominator == 1 else str(value)
    if isinstance(value, dict):
        return {str(k): _wire(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_wire(v) for v in value]
    if value is None or type(value) in (str, int, bool):
        return value
    _stop('NOT_RUN', 'unsupported_scalar', 'A decoded value is not a finite scalar or JSON value.',
          value_type=type(value).__name__)


def _number(value, label):
    value = literal(value)
    if isinstance(value, bool) or not isinstance(value, (int, Fraction)):
        _stop('FAIL', 'nonnumeric_primitive', f'{label} must evaluate to a finite rational scalar.',
              value=_wire(value))
    return Fraction(value)


def _scalar_key(value):
    value = literal(value)
    if value is not None and type(value) not in (str, int, bool, Fraction):
        _stop('NOT_RUN', 'nonscalar_grid', 'Only scalar finite decoded grid values are executable here.',
              value_type=type(value).__name__)
    tag = 'number' if type(value) in (int, Fraction) else type(value).__name__
    return tag, canonical(_wire(value))


def _assignment_key(assignment, names):
    return tuple((name, _scalar_key(assignment[name])) for name in names)


def _opaque(value):
    if isinstance(value, dict):
        return 'opaque' in value or any(_opaque(v) for v in value.values())
    if isinstance(value, list):
        return any(_opaque(v) for v in value)
    return False


def _assignments(names, domains):
    for row in itertools.product(*(domains[name] for name in names)):
        yield dict(zip(names, row))


def _resolve_report_value(value, grid, atom, axis):
    matches = [x for x in grid if _wire(x) == value]
    if len(matches) != 1:
        _stop('FAIL', 'invalid_fixed_coordinate',
              'A reported fixed value must decode uniquely to an original grid value.',
              atom=atom, axis=axis, value=value, matching_values=len(matches))
    return matches[0]


def _interfaces(model, report, state_domains, action_domains):
    regimes = report.get('recommended', {}).get('regimes')
    if not isinstance(regimes, list) or not regimes:
        _stop('FAIL', 'missing_generated_interfaces', 'The report has no recommended regimes.')
    interfaces, owners = {}, {}
    block_ids = []
    for block in regimes:
        block_id = block.get('id')
        if not isinstance(block_id, str) or not block_id:
            _stop('FAIL', 'invalid_generated_regime', 'Every generated regime needs a nonempty id.')
        block_ids.append(block_id)
        ids, declared = block.get('atoms'), block.get('semantic_interfaces')
        if not isinstance(ids, list) or not isinstance(declared, dict) or set(ids) != set(declared):
            _stop('FAIL', 'interface_domain', 'Regime atom membership and semantic interface keys differ.',
                  regime=block_id)
        for atom_id in ids:
            if atom_id not in model['by_id'] or atom_id in interfaces:
                _stop('FAIL', 'atom_coverage', 'Each original atom must occur in exactly one generated regime.',
                      atom=atom_id)
            atom, interface = model['by_id'][atom_id], declared[atom_id]
            if block.get('actor_information_role') != atom['role'] or block.get('routing_owner') != atom['routing_owner']:
                _stop('FAIL', 'economic_role_changed', 'A generated regime changes the actor/order or routing owner.',
                      atom=atom_id, regime=block_id)
            required = {'states', 'actions', 'forced_actions', 'constant_states'}
            if not isinstance(interface, dict) or not required <= set(interface):
                _stop('FAIL', 'malformed_interface', 'A generated semantic interface is incomplete.', atom=atom_id)
            sn, an = interface['states'], interface['actions']
            if not isinstance(sn, list) or len(set(sn)) != len(sn) or not set(sn) <= set(atom['declared_states']):
                _stop('FAIL', 'invalid_state_projection', 'Retained state axes must be distinct original state axes.', atom=atom_id)
            if not isinstance(an, list) or len(set(an)) != len(an) or not set(an) <= set(atom['declared_actions']):
                _stop('FAIL', 'invalid_action_projection', 'Free action axes must be distinct original action axes.', atom=atom_id)
            fixed, constants = interface['forced_actions'], interface['constant_states']
            if not isinstance(fixed, dict) or set(an) & set(fixed) or set(an) | set(fixed) != set(atom['declared_actions']):
                _stop('FAIL', 'action_coordinate_partition', 'Every original action coordinate must be free or explicitly fixed exactly once.', atom=atom_id)
            if not isinstance(constants, dict) or not set(constants) <= set(atom['declared_states']):
                _stop('FAIL', 'invalid_constant_states', 'Constant state declarations must refer to original axes.', atom=atom_id)
            decoded_fixed = {x: _resolve_report_value(v, action_domains[x], atom_id, x) for x, v in fixed.items()}
            for x, v in constants.items():
                if len(state_domains[x]) != 1:
                    _stop('FAIL', 'nonconstant_state_claim', 'A reported constant state axis has multiple original values.', atom=atom_id, axis=x)
                _resolve_report_value(v, state_domains[x], atom_id, x)
            interfaces[atom_id] = {'states': list(sn), 'actions': list(an), 'forced_actions': decoded_fixed}
            owners[atom_id] = block_id
    if len(block_ids) != len(set(block_ids)) or set(interfaces) != set(model['by_id']):
        _stop('FAIL', 'atom_coverage', 'Generated regimes must have unique ids and cover every original atom once.')
    return interfaces, owners


def _stages(problem, model):
    stages = []
    for ai, age in enumerate(problem['ages']):
        roles = {canonical(a['role']): a['role'] for a in model['atoms'] if a['age_index'] == ai}
        if len(roles) != 1:
            _stop('UNRESOLVED', 'stage_role_encoding',
                  'The inherited checker needs exactly one actor/information/order role per date; a role-indexed extension is required.',
                  age=age, roles=list(roles.values()))
        role = next(iter(roles.values()))
        stages.append({'age': age, 'actor': role.get('actor', 'agent'),
                       'observations': role.get('observations', []),
                       'commitment': role.get('commitments', role.get('commitment', [])),
                       'discount_boundary': role.get('discount_boundary', 'between_dates')})
    return stages


def _load_checker():
    from . import transport_checker

    path = Path(transport_checker.__file__)
    return transport_checker.check, hashlib.sha256(path.read_bytes()).hexdigest()


def validate_generated_transport(problem, report, max_cells=20000):
    """Return actual checker evidence for the generated semantic projection.

    ``max_cells`` bounds enumerated original-plus-reduced state/action candidate
    rows (a scalar terminal counts as one row per state).  It is not a reachable
    state mask.  Every declared status/state is admitted as an incoming start.
    The returned ``checker_input`` contains the generated rows and maps so that
    the inherited checker can be rerun without importing the finder.

    A PASS concerns the listed scalar IR instance and lifted compatible policies.
    It does not execute opaque callbacks, native backend layouts, distinct
    simulation reoptimization preferences, or a same-period value-read extension.
    """
    result = {'status': 'NOT_RUN', 'checker_executed': False, 'checker_result': None,
              'checker_input': None, 'generated': {'original_cells': 0, 'reduced_cells': 0,
                  'original_feasible_action_rows': 0, 'reduced_feasible_action_rows': 0,
                  'original_terminal_rows': 0, 'reduced_terminal_rows': 0},
              'layers': {'generation': {'status': 'NOT_RUN'}, 'finite_transport': {'status': 'NOT_RUN'}},
              'guarantee': 'none', 'diagnostics': [],
              'scope': {'representation': 'reported per-atom semantic interfaces; dense layout padding is not state',
                        'support': 'all declared finite states/statuses and all feasible actions, each state admitted as an incoming start',
                        'numeric_interpretation': 'exact rational arithmetic under scalar IR decimal-literal semantics; no renormalization',
                        'probability_support': 'every strictly positive solve and simulation edge retained; only exact zeros omitted',
                        'policy': 'values, argmax correspondence and projected laws for lifted compatible policies',
                        'not_claimed': ['native source/backend execution', 'allocation or runtime improvement',
                                        'arbitrary history-dependent policies or same-seed ties',
                                        'distinct simulation reoptimization primitives', 'parameter continua']}}
    try:
        if type(max_cells) is not int or max_cells < 1:
            _stop('FAIL', 'invalid_budget', 'max_cells must be a positive integer.')
        if not isinstance(problem, dict) or not isinstance(report, dict):
            _stop('FAIL', 'invalid_input', 'The original problem and generated report must be objects.')
        if report.get('problem_sha256') != digest(problem):
            _stop('FAIL', 'wrong_original_problem', 'The generated report is not bound to this original problem.')
        model = prepare(problem)
        domains = {}
        for kind in ('states', 'actions'):
            domains[kind] = {}
            for name, domain in problem.get(kind, {}).items():
                grid = values(domain)
                if grid is None:
                    _stop('NOT_RUN', 'nonenumerable_grid', 'A size-only grid has no decoded values to execute.',
                          coordinate_kind=kind, coordinate=name, grid_id=domain.get('grid_id'))
                normalized = [literal(x) for x in grid]
                keys = [_scalar_key(x) for x in normalized]
                if len(set(keys)) != len(keys):
                    _stop('FAIL', 'ambiguous_decoded_grid', 'Decoded grid values coincide under the IR numeric interpretation.', coordinate=name)
                domains[kind][name] = normalized
        for atom in model['atoms']:
            p = atom['_program']
            if _opaque(p):
                _stop('NOT_RUN', 'opaque_primitive', 'An original primitive is a declared opaque callback; no substitute execution is performed.', atom=atom['id'])
            if p.get('reads'):
                _stop('UNRESOLVED', 'same_period_or_external_read', 'Generated value-read transport requires an explicit ranked read adapter; this bridge does not execute it.', atom=atom['id'])
            if p.get('taste_group') is not None or p.get('taste_law') is not None or p.get('taste', {}).get('kind', 'deterministic') != 'deterministic':
                _stop('NOT_RUN', 'taste_extension', 'This generated bridge currently instantiates deterministic finite action problems only.', atom=atom['id'])
            if atom['terminal'] and atom['declared_actions']:
                _stop('UNRESOLVED', 'terminal_action_problem', 'An optimizing terminal action problem must be evaluated separately before it becomes a scalar terminal payoff.', atom=atom['id'])
            if atom['terminal'] and p.get('transitions'):
                _stop('UNRESOLVED', 'terminal_continuation', 'A scalar terminal payoff cannot also carry continuation transitions.', atom=atom['id'])
        if problem.get('simulation_choice', 'same_declared_primitives') != 'same_declared_primitives':
            _stop('UNRESOLVED', 'simulation_reoptimization', 'Distinct simulation decision primitives need a separately generated decision-problem certificate.')
        interfaces, owners = _interfaces(model, report, domains['states'], domains['actions'])
        stages = _stages(problem, model)
        estimates = {'original_cells': 0, 'reduced_cells': 0, 'candidate_rows': 0}
        for atom in model['atoms']:
            interface = interfaces[atom['id']]
            for side, sn, an in [('original', atom['declared_states'], atom['declared_actions']),
                                 ('reduced', interface['states'], interface['actions'])]:
                ns = 1
                for x in sn:
                    ns *= len(domains['states'][x])
                na = 1
                if not atom['terminal']:
                    for x in an:
                        na *= len(domains['actions'][x])
                estimates[side + '_cells'] += ns
                estimates['candidate_rows'] += ns * na
        result['estimated'] = {**estimates, 'max_cells': max_cells,
                               'budget_unit': 'original plus reduced state/action candidate rows; scalar terminal rows count once'}
        if estimates['candidate_rows'] > max_cells:
            _stop('NOT_RUN', 'finite_generation_budget', 'Full declared finite enumeration exceeds max_cells; no sampled or reachable-only quotient is substituted.',
                  required=estimates['candidate_rows'], max_cells=max_cells)
        old_cells, new_cells, projection, representatives = {}, {}, {}, {}
        old_records, new_records, old_lookup = {}, {}, {}
        for atom in model['atoms']:
            atom_id, sn = atom['id'], atom['declared_states']
            kept = interfaces[atom_id]['states']
            fibers = {}
            for i, state in enumerate(_assignments(sn, domains['states'])):
                oid = f'o:{atom_id}:{i}'
                key = _assignment_key(state, kept)
                if key not in fibers:
                    nid = f'n:{atom_id}:{len(fibers)}'
                    fibers[key] = nid
                    representatives[nid] = oid
                    new_records[nid] = (atom, state)
                projection[oid] = fibers[key]
                old_records[oid] = (atom, state)
                old_lookup[(atom_id, _assignment_key(state, sn))] = oid
        by_age = defaultdict(list)
        for atom in model['atoms']:
            by_age[atom['age_index']].append(atom)

        def state_attributes(state, axes):
            out = {}
            for name in axes:
                value = state[name]
                numeric = type(value) in (int, Fraction)
                code = next(i for i, v in enumerate(domains['states'][name]) if _scalar_key(v) == _scalar_key(value))
                out[name] = {'type': 'rational' if numeric else 'categorical', 'code': code, 'value': _wire(value)}
            return out

        def kernel(atom, env, phase, reduced):
            weights = defaultdict(Fraction)
            for edge in atom['_program'].get('transitions', []):
                if edge.get('phase', 'both') not in (phase, 'both'):
                    continue
                if not bool(evaluate(edge.get('when', True), env)):
                    continue
                probability = _number(evaluate(edge.get('probability', 1), env), 'transition probability')
                if probability < 0:
                    _stop('FAIL', 'negative_probability', 'A generated finite transition has negative probability.', atom=atom['id'], phase=phase)
                if probability == 0:
                    continue
                status = {x: evaluate(edge.get('status_updates', {}).get(x, x), env) for x in problem.get('status_axes', {})}
                candidates = [a for a in by_age[atom['age_index'] + 1]
                              if _assignment_key(a['status'], sorted(status)) == _assignment_key(status, sorted(status))
                              and (not edge.get('target') or a['original_regime'] == edge['target'])
                              and (edge.get('target') or a['channel'] == atom['channel'])]
                if len(candidates) != 1:
                    _stop('UNRESOLVED', 'ambiguous_or_missing_target', 'A positive transition must identify exactly one next-date economic atom.',
                          atom=atom['id'], phase=phase, matching_atoms=[a['id'] for a in candidates])
                target = candidates[0]
                target_state = {x: evaluate(edge.get('updates', {}).get(x, x), env) for x in target['declared_states']}
                key = (target['id'], _assignment_key(target_state, target['declared_states']))
                if key not in old_lookup:
                    _stop('FAIL', 'out_of_domain_transition', 'A positive transition leaves the original decoded state grid.', atom=atom['id'], target=target['id'], state=_wire(target_state))
                destination = old_lookup[key]
                weights[projection[destination] if reduced else destination] += probability
            if not weights:
                _stop('UNRESOLVED', 'missing_probability_row', 'A nonterminal action lacks a positive normalized continuation row in one declared phase.', atom=atom['id'], phase=phase)
            return {d: _wire(w) for d, w in sorted(weights.items())}

        action_lookup, new_lifts = {}, {}
        for reduced, records, cells in [(False, old_records, old_cells), (True, new_records, new_cells)]:
            for node, (atom, state) in records.items():
                p, interface = atom['_program'], interfaces[atom['id']]
                # Remove prepare()'s proven fixed actions before enumerating the
                # original problem.  Every original action tuple is still tried.
                base = {**problem.get('parameters', {}), 'age': atom['age'], **atom['status'], **state}
                observables = {'observed_status': _wire(atom['status']), 'economic_role': _wire(atom['role']),
                               'routing_owner': _wire(atom['routing_owner'])}
                cell = {'stage': atom['age_index'], 'regime': owners[atom['id']] if reduced else atom['original_regime'],
                        'channel': atom['channel'], 'rank': 0,
                        'state': state_attributes(state, interface['states'] if reduced else atom['declared_states']),
                        'observables': observables, 'terminal': None, 'actions': {}, 'taste': {'kind': 'deterministic'}}
                if atom['terminal']:
                    cell['terminal'] = {'value': _wire(_number(evaluate(p.get('terminal_payoff', 0), base), 'terminal payoff'))}
                    if p.get('outputs'):
                        observables['outputs'] = {name: _wire(evaluate(expr, base)) for name, expr in p['outputs'].items()}
                else:
                    action_axes = interface['actions'] if reduced else atom['declared_actions']
                    output_rows = []
                    for i, decoded in enumerate(_assignments(action_axes, domains['actions'])):
                        lifted = {**decoded, **interface['forced_actions']} if reduced else decoded
                        env = {**base, **lifted}
                        if not bool(evaluate(p.get('feasible', True), env)):
                            continue
                        action_id = f'a{i}'
                        discount = _number(evaluate(p.get('discount', 1), env), 'discount')
                        if discount < 0:
                            _stop('FAIL', 'negative_discount', 'The expected-return finite core requires nonnegative discounts.', atom=atom['id'])
                        cell['actions'][action_id] = {
                            'code': i, 'decoded': _wire(decoded),
                            'reward': {'value': _wire(_number(evaluate(p.get('reward', 0), env), 'reward'))},
                            'discount': _wire(discount),
                            'kernels': {phase: kernel(atom, env, phase, reduced) for phase in ('solve', 'simulation')},
                            'references': {}, 'taste_group': None}
                        action_key = _assignment_key(lifted, atom['declared_actions'])
                        if reduced:
                            new_lifts[(node, action_id)] = action_key
                        else:
                            action_lookup[(node, action_key)] = action_id
                        if p.get('outputs'):
                            output_rows.append({'action': _wire(lifted), 'values': {name: _wire(evaluate(expr, env)) for name, expr in p['outputs'].items()}})
                    if p.get('outputs'):
                        observables['outputs_by_decoded_action'] = sorted(output_rows, key=lambda row: canonical(row['action']))
                    if not cell['actions']:
                        cell['empty_semantics'] = 'negative_infinity'
                cells[node] = cell
        embeddings, fixed, tastes = {}, {}, {}
        for oid, (atom, _) in old_records.items():
            if atom['terminal']:
                continue
            nid = projection[oid]
            embeddings[oid] = {action_id: action_lookup[(oid, new_lifts[(nid, action_id)])]
                               for action_id in new_cells[nid]['actions']
                               if (oid, new_lifts[(nid, action_id)]) in action_lookup}
            fixed[oid] = _wire(interfaces[atom['id']]['forced_actions'])
            tastes[oid] = {}
        numeric_parameters = {}
        for name, value in problem.get('parameters', {}).items():
            if isinstance(value, bool):
                continue
            try:
                numeric_parameters[name] = _wire(Fraction(str(value)))
            except (ValueError, ZeroDivisionError, TypeError):
                continue  # Complete bindings remain in the result, including tables/categories.
        data = {'format_version': '1.0', 'id': str(problem.get('name', 'generated')) + ':generated-transport',
                'claims': ['finite_model'], 'parameter_scope': 'listed_cases',
                'policy_claim': 'values_argmax_lifted_law', 'simulation_choice': 'same_declared_primitives',
                'cases': [{'id': 'generated-scalar-instance', 'parameters': numeric_parameters,
                    'original': {'objective': 'value', 'stages': stages, 'cells': old_cells,
                                 'information_partition': [[x] for x in old_cells]},
                    'reduced': {'objective': 'value', 'stages': stages, 'cells': new_cells,
                                'information_partition': [[x] for x in new_cells]},
                    'starts': [{'id': f'incoming:{x}', 'kind': 'incoming', 'original': {x: 1}, 'reduced': {projection[x]: 1}} for x in old_cells],
                    'certificate': {'kind': 'quotient', 'support': {'original': list(old_cells), 'reduced': list(new_cells)},
                                    'state_map': projection, 'action_embedding': embeddings,
                                    'fixed_action_attributes': fixed, 'taste_group_map': tastes},
                    'subjects': {'original': {}, 'reduced': {}}}], 'obligations': []}
        generated = result['generated']
        for side, cells in [('original', old_cells), ('reduced', new_cells)]:
            generated[side + '_cells'] = len(cells)
            generated[side + '_feasible_action_rows'] = sum(len(c['actions']) for c in cells.values())
            generated[side + '_terminal_rows'] = sum(c['terminal'] is not None for c in cells.values())
            generated[side + '_positive_edges'] = {phase: sum(len(a['kernels'][phase]) for c in cells.values() for a in c['actions'].values()) for phase in ('solve', 'simulation')}
        generated['incoming_starts'] = len(data['cases'][0]['starts'])
        generated['state_map_entries'] = len(projection)
        generated['action_embedding_entries'] = sum(map(len, embeddings.values()))
        result.update(checker_input=data, checker_input_sha256=digest(data),
                      problem_sha256=digest(problem), bound_parameters=_wire(problem.get('parameters', {})),
                      representatives=representatives)
        result['layers']['generation'] = {'status': 'PASS', 'candidate_rows': estimates['candidate_rows']}
        check, checker_hash = _load_checker()
        numerical = problem.get('numerical', {})
        tolerance = Fraction(0) if numerical.get('strict', False) else Fraction(str(numerical.get('probability_mass_absolute', '1e-12')))
        checker_result = check(data, probability_tolerance=tolerance)
        result.update(status=checker_result['status'], checker_executed=True,
                      checker_result=checker_result, checker_source_sha256=checker_hash)
        result['layers']['finite_transport'] = {'status': checker_result['status'],
            'checks': checker_result['layers']['finite_model']['checks']}
        exact_normalization = not checker_result.get('probability_normalization')
        result['exact_markov_normalization'] = exact_normalization if checker_result['status'] == 'PASS' else False
        if checker_result['status'] == 'PASS':
            result['guarantee'] = 'exact_finite_markov_transport' if exact_normalization else 'exact_raw_operator_transport_with_mass_error'
            result['probability_mass_absolute'] = str(tolerance)
    except _BridgeStop as exc:
        result['status'] = exc.status
        result['diagnostics'].append({'code': exc.code, 'reason': exc.reason, **_wire(exc.witness)})
        if exc.status == 'FAIL':
            result['layers']['generation']['status'] = 'FAIL'
    except AdmissionError as exc:
        unexecuted = 'not numerically executable' in str(exc)
        result['status'] = 'NOT_RUN' if unexecuted else 'FAIL'
        result['diagnostics'].append({'code': 'unexecuted_expression' if unexecuted else 'original_ir_admission', 'reason': str(exc)})
        if not unexecuted:
            result['layers']['generation']['status'] = 'FAIL'
    except (KeyError, TypeError, ValueError, ArithmeticError) as exc:
        result['status'] = 'FAIL'
        result['layers']['generation']['status'] = 'FAIL'
        result['diagnostics'].append({'code': 'malformed_generated_input', 'reason': str(exc)})
    return result
