"""Metadata-only semantic discovery and Pareto recommendations.

This entry point never calls a model function, compiles a solver, or enumerates
the state/action Cartesian product. Categorical characteristic tables concern
one explicitly finite original coordinate, under a small pure language.
"""
from __future__ import annotations
from copy import deepcopy
from itertools import product, combinations
import time

from .records import Scenario
from .expressions import jsonable
from .facts import (FACTS_BACKENDS, FactsBudgetError, MetadataFacts, characteristic_cells,
                    identity_stops, lane_needs, lane_rows, live_states, make_facts, status_axes)
from .frontier import canonical, digest, search_frontier
from .block_admission import admission_record, context_key


class MetadataAnalysis:
    def __init__(self, declarations, limits=None, *, search_mode, facts_backend):
        if search_mode not in {'heuristic', 'exact'}:
            raise ValueError('search_mode must be heuristic or exact')
        if facts_backend not in FACTS_BACKENDS:
            raise ValueError('facts_backend must be one of ' + ', '.join(FACTS_BACKENDS))
        self.search_mode = search_mode
        self.decl = declarations
        self.limits = {'seconds': 900, 'metadata_categories': 100000,
                       'projection_policies': 4096} | (limits or {})
        self.started = time.perf_counter()
        self.obligations = deepcopy(declarations.get('obligations', []))
        self._interface_cache = {}
        # The scalar interpreter also renders the reduced function signatures,
        # whichever backend supplies the dependency facts.
        self.facts = make_facts(facts_backend, declarations, self.limits, self.guard)
        self.interpreter = (self.facts if isinstance(self.facts, MetadataFacts)
                            else MetadataFacts(declarations, self.limits, self.guard))
        self.cells = self.facts.cells
        self.callbacks = self.facts.callbacks
        self.domains = self.facts.domains
        self.constants = self.facts.constants
        self.forced = self.facts.forced
        self.live = {}
        self.reasons = {}

    def guard(self):
        if time.perf_counter() - self.started >= self.limits['seconds']:
            raise TimeoutError('Metadata dependency analysis budget exhausted')

    def dependencies(self):
        self.live, self.reasons = live_states(self.facts, self.guard)

    def characteristic_candidates(self):
        result = []
        for name, spec in sorted(self.decl.get('characteristics', {}).items()):
            self.guard()
            mapping = self.facts.characteristic_image(spec)
            if mapping is None:
                continue
            unique = {canonical(v): v for v in mapping}
            if len(unique) >= len(mapping):
                continue
            eligible, requires = characteristic_cells(self.facts, self.live, spec)
            if eligible:
                cb = self.callbacks[spec['callback']]
                domain_key = 'characteristic:' + digest([cb.get('identity'), spec['domain'], mapping])[:24]
                result.append({'id': name, 'state': spec['state'], 'domain': spec['domain'],
                               'callback': spec['callback'], 'target': spec.get('name', name),
                               'values': mapping, 'target_values': [unique[k] for k in sorted(unique)],
                               'target_domain': domain_key, 'cells': sorted(eligible),
                               'requires': {k: sorted(v) for k, v in requires.items()}})
        return result

    def closed_regions(self, candidate):
        """Every forward-closed coarsening region; identity can precede coarsening."""
        def closure(nodes):
            found = set(nodes);todo = list(nodes)
            while todo:
                for target in candidate['requires'].get(todo.pop(), []):
                    if target not in found:
                        found.add(target);todo.append(target)
            return frozenset(found)
        regions = {frozenset()}
        complete = True
        for cid in candidate['cells']:
            self.guard()
            regions.update(closure(region | {cid}) for region in tuple(regions))
            if len(regions) > self.limits['projection_policies']:
                # Preserve valid full and suffix regions, explicitly incomplete.
                regions = set(sorted(regions, key=lambda s: (len(s), sorted(s)))[:self.limits['projection_policies']])
                complete = False;break
        return [dict(candidate, active_cells=sorted(r)) for r in sorted(regions, key=lambda s: (len(s), sorted(s))) if r], complete

    def heuristic_regions(self, candidate):
        """Propose full and original-label regions, closed under carry laws."""
        by_label = {}
        for cid in candidate['cells']:
            by_label.setdefault(self.cells[cid]['regime'], set()).add(cid)
        regions = []
        for seed in [candidate['cells'], *(by_label[key] for key in sorted(by_label))]:
            found, todo = set(seed), list(seed)
            while todo:
                self.guard()
                for target in candidate['requires'].get(todo.pop(), []):
                    if target not in found:
                        found.add(target)
                        todo.append(target)
            active = sorted(found)
            if not any(row['active_cells'] == active for row in regions):
                regions.append(dict(candidate, active_cells=active))
        return regions, False

    def status_splits(self, interfaces, policies, derived):
        """Offer each cell split by the status axis whose lanes need the fewest rows.

        A lane is the set of codes of one status axis whose points need the same
        states. Splitting pays when some lane drops a state the whole cell keeps.
        One added policy replaces every splittable cell of the identity policy
        by its split. The metadata backend has no per-status facts.
        """
        if self.facts.name == 'metadata':
            return {'status': 'NOT_OFFERED', 'reason': 'The metadata backend has no per-status facts; '
                    'use facts_backend="dense" or "implicit" to consider status splits.', 'cells': []}
        base = next((p for p in policies if not p['characteristics']), None)
        if base is None:
            return {'status': 'NOT_OFFERED', 'reason': 'No identity projection policy', 'cells': []}
        index = {cid: {o['id']: o for o in record['options']} for cid, record in interfaces.items()}
        chosen = {}
        records = []
        for cid, cell in self.cells.items():
            self.guard()
            option = index[cid][base['option_by_cell'][cid]]
            unsplit = 1
            for domain_id in option['states'].values():
                if self.domains[domain_id].get('solve_axis') is not False:
                    unsplit *= self.domains[domain_id]['size']
            best = None
            for axis, codes in status_axes(self.facts, cell):
                if axis not in option['states']:
                    continue
                kept = set(option['states']) - {axis}
                groups = {}
                for code in codes:
                    needs = frozenset(lane_needs(self.facts, cell, self.live, {axis: code}) & kept)
                    groups.setdefault(needs, []).append(code)
                if len(groups) < 2:
                    continue
                lanes = [{'codes': {axis: sorted(group)}, 'states': {s: option['states'][s] for s in sorted(needs)}}
                         for needs, group in sorted(groups.items(), key=lambda item: sorted(item[1]))]
                rows = lane_rows(lanes, self.domains[cell['states'][axis]], self.domains)
                if rows < unsplit and (best is None or rows < best[0]):
                    best = (rows, axis, lanes)
            if best is None:
                continue
            rows, axis, lanes = best
            domain_id = 'lanes:' + digest([axis, lanes])[:24]
            derived[domain_id] = {'kind': 'lanes', 'size': rows, 'status_axis': axis, 'lanes': lanes,
                                  'identity': domain_id, 'source_path': f'/cells/{cid}'}
            split = deepcopy(option)
            kept = {axis} | {name for lane in lanes for name in lane['states']}
            for name, projection in split['state_projection'].items():
                if projection['kind'] == 'identity' and name not in kept:
                    split['state_projection'][name] = {'kind': 'drop'}
            split['states'] = {f'lanes[{axis}]': domain_id}
            split['lanes'] = deepcopy(lanes)
            split['witnesses'] = [*split['witnesses'], {'kind': 'status_split', 'axis': axis,
                                                        'unsplit_state_rows': unsplit, 'state_rows': rows}]
            split['id'] = 'interface:' + digest(split)[:20]
            interfaces[cid]['options'].append(split)
            chosen[cid] = split['id']
            records.append({'cell': cid, 'axis': axis, 'lanes': lanes, 'unsplit_state_rows': unsplit, 'state_rows': rows})
        if chosen:
            policy = {**base['option_by_cell'], **chosen}
            policies.append({'id': 'projection:' + digest([base['id'], sorted(chosen.items())])[:16],
                             'option_by_cell': policy, 'characteristics': [], 'status_splits': sorted(chosen)})
        return {'status': 'OFFERED' if chosen else 'NONE_USEFUL',
                'reason': ('Split cells are offered in one added projection policy.' if chosen else
                           'No status lane drops a state its whole cell keeps.'), 'cells': records}

    def function_interfaces(self, cell, projection, constants):
        """A source-bound DAG rewrite plan, with actual reduced signatures.

        It is deliberately an inspectable plan, not a claim that a native solver
        was compiled. Existing pure numerical bodies and joint laws stay pinned.
        """
        key = (cell['id'], canonical(projection), canonical(constants))
        if key in self._interface_cache:
            return deepcopy(self._interface_cache[key])
        rewritten = {}
        stops = identity_stops(self.callbacks, [v['callback'] for v in projection.values() if v['kind'] == 'characteristic'])
        for phase in ('solve', 'simulation'):
            providers = cell.get('phase_functions', {}).get(phase, cell.get('functions', {}))
            records = {}
            for name, cbid in providers.items():
                cb = self.callbacks[cbid]
                environment = self.interpreter.environment(cell, cb, phase) | constants
                replacement = next((v for v in projection.values()
                                    if v['kind'] == 'characteristic'
                                    and self.callbacks[v['callback']].get('identity') == cb.get('identity')), None)
                if replacement:
                    target = replacement['target']
                    records[name] = {
                        'original_callback': cbid, 'source_identity': cb.get('identity'),
                        'original_arguments': cb.get('arguments', []),
                        'arguments': [] if target in constants else [target],
                        'implementation': {'kind': 'bound_constant', 'value': constants[target]}
                            if target in constants else {'kind': 'read_reduced_coordinate', 'coordinate': target},
                        'witness': 'existing characteristic map, checked over its complete original categorical domain'}
                    continue
                reads = self.interpreter.direct_reads(cbid, cell, overrides=constants, phase=phase)
                state_reads = self.facts.callback_reads(cbid, cell, phase, stops=stops)
                unavailable = {axis for axis in state_reads if axis in projection
                               and projection[axis]['kind'] != 'identity'
                               and axis not in constants}
                # A raw provider which is no longer used by any retained root is
                # omitted from the reduced DAG, not kept with dummy arguments.
                if unavailable:
                    records[name] = {'original_callback': cbid,
                                     'source_identity': cb.get('identity'),
                                     'implementation': {'kind': 'omit_unconsumed_provider'},
                                     'original_arguments': cb.get('arguments', []), 'arguments': [],
                                     'unavailable_original_axes': sorted(unavailable)}
                    continue
                original_args = cb.get('arguments', [])
                bound = {arg: jsonable(environment[arg]) for arg in original_args if arg in environment}
                records[name] = {'original_callback': cbid, 'source_identity': cb.get('identity'),
                                 'original_arguments': original_args,
                                 'arguments': [arg for arg in original_args if arg in reads and arg not in bound],
                                 'bound_arguments': bound,
                                 'unused_arguments': [arg for arg in original_args if arg not in reads and arg not in bound],
                                 'implementation': {'kind': 'source_bound_partial_application',
                                                    'native_execution': 'NOT_RUN'}}
            rewritten[phase] = records
        # A policy can differ only at other cells; its local rewrite is unchanged.
        self._interface_cache[key] = deepcopy(rewritten)
        return rewritten

    def family_boundary(self, interfaces, policies):
        """Report witnessed structural admission separately from economic adoption."""
        by_regime = {}
        for cell in self.cells.values():
            by_regime.setdefault(cell['regime'], []).append(cell)
        options = {cid: {option['id']: option for option in record['options']}
                   for cid, record in interfaces.items()}
        candidates = []
        omitted = 0
        names = sorted(by_regime)
        limit = self.limits.get('conditional_merge_candidates', 2048)
        pairs_visited = 0
        cells_checked = 0
        pair_budget = self.limits.get('conditional_merge_pair_checks', 8192)
        for left_name, right_name in combinations(names, 2):
            self.guard()
            if pairs_visited >= pair_budget:
                break
            pairs_visited += 1
            if len(candidates) >= limit:
                omitted += 1
                continue
            witness = None
            exhausted = False
            for left, right in product(by_regime[left_name], by_regime[right_name]):
                self.guard()
                if witness is not None or exhausted:
                    break
                if cells_checked >= pair_budget:
                    exhausted = True
                    break
                cells_checked += 1
                if context_key(left) != context_key(right):
                    continue
                for record in policies:
                    if cells_checked >= pair_budget:
                        exhausted = True
                        break
                    cells_checked += 1
                    policy = record['option_by_cell']
                    selected = [options[cell['id']][policy[cell['id']]] for cell in (left, right)]
                    compatible = True
                    for kind in ('states', 'actions'):
                        union = {}
                        for option in selected:
                            for name, domain in option[kind].items():
                                if name in union and union[name] != domain:
                                    compatible = False
                                union[name] = domain
                    if compatible:
                        witness = {'cells': [left['id'], right['id']],
                                   'projection_policy': record['id'],
                                   'admission': admission_record([left, right])}
                        break
            candidates.append({
                'regimes': [left_name, right_name],
                'status': ('ADMITTED_STRUCTURAL_BLOCK' if witness is not None else
                           'UNEXAMINED' if exhausted else 'NOT_ADMITTED'),
                'witness': witness,
                'witness_scope': 'local block admission only; complete grouping/date-interval membership is checked separately',
                'reason': ('Compatible original-cell dispatch and axis domains under a coherent policy.'
                           if witness is not None else
                           'Diagnostic cell/policy budget exhausted; no negative admission claim.'
                           if exhausted else 'No compatible original-cell interfaces under the offered coherent policies.'),
                'required_obligations': [
                    {'kind': 'merge_semantic_and_incidence_check',
                     'component': 'semantic and frontier checkers',
                     'required_fact': 'Preserve each original actor, information stage, terminal payoff, feasible actions, joint/phased laws, requested outputs and owned query route; structural admission does not discharge source semantics.'}],
                'automatic_graph_change': False})
        opaque = []
        for identifier, callback in sorted(self.callbacks.items()):
            if callback.get('status') == 'proved':
                continue
            opaque.append({'callback': identifier, 'source_path': callback.get('source_path'),
                           'captured_inputs': callback.get('arguments', []),
                           'status': callback.get('status'),
                           'component': 'source-bound callback/declaration interpreter',
                           'required_fact': 'Interpret the original body and actual bindings over these captured inputs, or supply a separately checked declaration contract. Preserve the current conservative dependency boundary until then.'})
        return {
            'schema': 'regime-discovery-family/v3',
            'complete_claim_scope': 'coherent source-bound projection policies and the explicitly admitted context/grouping interval family',
            'unrestricted_model_representation_optimum_claimed': False,
            'original_regime_labels_are_semantic_restrictions': False,
            'unknown_context_policy': 'retain every original source-cell context by dispatch; structural admission and semantic adoption are separate',
            'endpoint_preference_is_admissibility_rule': False,
            'conditional_merge_candidates': candidates,
            'conditional_merge_candidates_omitted_by_display_budget': omitted,
            'conditional_merge_pairs_unexamined': len(names) * (len(names) - 1) // 2 - pairs_visited,
            'callback_obligations': opaque,
            'parameter_family_sharing': 'NOT_ESTABLISHED: each source/fixed-input identity requires its own checked feasibility, liveness and support closure',
        }

    def run(self):
        try:
            return self._run()
        except TimeoutError:
            if self.facts.name == 'metadata':
                raise
            raise FactsBudgetError(backend=self.facts.name, limit='time budget',
                                   unfinished=sorted(set(self.cells) - set(self.live)) or sorted(self.cells)) from None

    def _run(self):
        self.dependencies()
        candidates = self.characteristic_candidates()
        by_axis = {}
        local_choice_keys = {}
        region_complete = True
        for c in candidates:
            regions, complete = (self.heuristic_regions(c) if self.search_mode == 'heuristic'
                                 else self.closed_regions(c))
            key = digest(c)
            local_choice_keys.update((id(region), key) for region in regions)
            region_complete = region_complete and complete
            by_axis.setdefault(c['state'], []).extend(regions)
        choices = [[None] + values for _, values in sorted(by_axis.items())]
        if self.search_mode == 'heuristic':
            # No categorical powerset or cross-axis Cartesian family is needed
            # for a heuristic: offer single reductions and one combined seed.
            combinations_to_try = [()] + [(region,) for _, regions in sorted(by_axis.items())
                                           for region in regions]
            combined = tuple(regions[0] for _, regions in sorted(by_axis.items()))
            if len(combined) > 1:
                combinations_to_try.append(combined)
        else:
            combinations_to_try = product(*choices)
        interfaces = {c: {'options': []} for c in self.cells}
        derived = {}
        derived_records = {}
        policies = []
        local_options = {}
        policy_complete = region_complete
        for combination in combinations_to_try:
            self.guard()
            if len(policies) >= self.limits['projection_policies']:
                policy_complete = False;break
            selected = [x for x in combination if x is not None]
            policy_id = 'projection:' + digest([(c['id'], c['active_cells']) for c in selected])[:16]
            policy = {}
            for cid, cell in self.cells.items():
                self.guard()
                local = [c for c in selected if cid in c['active_cells']]
                # Keep original active-cell registration order for shared domains.
                for c in local:
                    if id(c) not in derived_records:
                        derived_records[id(c)] = {'kind': 'categorical', 'size': len(c['target_values']),
                            'values': c['target_values'], 'codes': list(range(len(c['target_values']))),
                            'identity': digest(c), 'source_path': f'/characteristics/{c["id"]}'}
                    derived[c['target_domain']] = derived_records[id(c)]
                key = (cid, tuple(local_choice_keys[id(c)] for c in local))
                if key in local_options:
                    policy[cid] = local_options[key]['id']
                    continue
                states = {s: d for s, d in cell['states'].items() if s in self.live[cid]}
                actions = {a: d for a, d in cell['actions'].items() if a not in self.forced[cid]}
                projection = {s: {'kind': 'identity', 'target': s} if s in states else {'kind': 'drop'}
                              for s in cell['states']}
                constants = dict(self.constants[cid])
                for c in local:
                    states.pop(c['state'])
                    projection[c['state']] = {'kind': 'characteristic', 'callback': c['callback'],
                        'target': c['target'], 'domain': c['target_domain'], 'values': c['values']}
                    if len(c['target_values']) == 1:
                        constants[c['target']] = c['target_values'][0]
                    else:
                        states[c['target']] = c['target_domain']
                reconstruction = {a: {'kind': 'constant', 'value': self.forced[cid][a]}
                                  if a in self.forced[cid] else {'kind': 'identity', 'source': a}
                                  for a in cell['actions']}
                option = {'states': states, 'actions': actions, 'state_projection': projection,
                          'action_reconstruction': reconstruction, 'constants': constants,
                          'function_interfaces': self.function_interfaces(cell, projection, constants),
                          'derived_code_bindings': {alias: {'provider': name,
                              'original_domain': cell['derived_categoricals'][name],
                              'mapping': deepcopy(self.domains[cell['derived_categoricals'][name]])}
                              for alias, name in cell.get('derived_code_providers', {}).items()},
                          'witnesses': [{'kind': 'dependency_closure', 'retained': self.reasons[cid]},
                                        {'kind': 'symbolic_characteristic_factorization',
                                         'characteristics': [c['id'] for c in selected if cid in c['active_cells']]}]}
                option['id'] = 'interface:' + digest(option)[:20]
                local_options[key] = option
                if all(o['id'] != option['id'] for o in interfaces[cid]['options']):
                    interfaces[cid]['options'].append(option)
                policy[cid] = option['id']
            policies.append({'id': policy_id, 'option_by_cell': policy,
                             'characteristics': [{'id': c['id'], 'cells': c['active_cells']} for c in selected]})
        status_splits = self.status_splits(interfaces, policies, derived)
        return {'schema': 'regime-semantic-analysis/v1', 'declarations': self.decl,
                'status_splits': status_splits,
                'facts_backend': self.facts.name,
                'interfaces': interfaces, 'derived_domains': derived,
                'projection_policies': policies, 'characteristic_candidates': candidates,
                'dependency_closure': {c: sorted(v) for c, v in self.live.items()},
                'discovery_complete': policy_complete,
                'family_boundary': self.family_boundary(interfaces, policies),
                'obligations': self.obligations,
                'production_execution': {'callbacks_called': 0, 'bellman_rows_evaluated': 0,
                                         'dense_cartesian_cells_enumerated': 0,
                                         'characteristic_metadata_values': sum(len(c['values']) for c in candidates)},
                'elapsed_seconds': time.perf_counter() - self.started}


def analyze_production(original, *, facts_backend, search_mode, limits=None, recommendation=None,
                       production_contract=None, checkpoint=None, analysis_scope='graph',
                       heuristic_effort=None):
    if search_mode not in {'heuristic', 'exact'}:
        raise ValueError('search_mode must be heuristic or exact')
    if facts_backend not in FACTS_BACKENDS:
        raise ValueError('facts_backend must be one of ' + ', '.join(FACTS_BACKENDS))
    started = time.perf_counter()
    phase_timings = {}
    active_phase = 'normalization'
    phase_started = started
    from .declarations import normalize_declarations
    try:
        if not isinstance(original, dict):
            from .model_snapshot import read_model
            original = read_model(original)
        if isinstance(original, dict) and original.get('schema') == 'regime-declarations/v1':
            declarations = deepcopy(original)
        else:
            declarations = normalize_declarations(original, production_contract=production_contract)
        from .graph_scope import apply_scope
        if analysis_scope not in {'graph', 'full_semantics'}:
            raise ValueError('Unknown analysis scope: '+str(analysis_scope))
        if analysis_scope == 'graph':
            declarations = apply_scope(original, declarations)
        phase_timings['normalization_seconds'] = time.perf_counter() - phase_started
        if checkpoint:
            checkpoint({'stage': 'normalized', 'declarations': declarations,
                        'phase_timings': dict(phase_timings)})
        active_phase = 'metadata_analysis'
        phase_started = time.perf_counter()
        metadata = MetadataAnalysis(declarations, limits, search_mode=search_mode, facts_backend=facts_backend)
        analysis = metadata.run()
        phase_timings['metadata_analysis_seconds'] = time.perf_counter() - phase_started
        if checkpoint:
            checkpoint({'stage': 'metadata_analysis', 'analysis': analysis,
                        'phase_timings': dict(phase_timings)})
    except (TimeoutError, ValueError, KeyError, TypeError, ArithmeticError) as exc:
        phase_timings.setdefault(active_phase + '_seconds', time.perf_counter() - phase_started)
        incomplete = isinstance(exc, TimeoutError)
        return {'schema': 'regime-production-result/v1',
                'status': 'INCOMPLETE' if incomplete else 'UNRESOLVED',
                'search_mode': search_mode, 'facts_backend': facts_backend,
                'recommendations': ([{'scenario_index': None, 'graph_id': 'retain-original',
                    'operation': 'retain_original', 'authority': 'retain_original',
                    'reason': 'Normalization or metadata analysis did not establish a checked proposal'}]
                    if search_mode == 'heuristic' else []),
                'error': type(exc).__name__ + ': ' + str(exc),
                'scenarios': [], 'analysis': {'declarations': locals().get('declarations', {}),
                                            'obligations': [{'kind': 'metadata_admission', 'detail': str(exc)}]},
                'semantic_check': {'status': 'NOT_RUN'},
                'established_lower_bound': dict.fromkeys(('work', 'storage', 'clarity'), 0),
                'layers': {'facts': facts_backend, 'source_semantics': 'NOT_RUN', 'discovery': 'INCOMPLETE',
                           'allocation': 'UNRESOLVED', 'frontier': 'NOT_RUN',
                           'native_numerical_execution': 'NOT_RUN'},
                'elapsed_seconds': time.perf_counter() - started,
                'phase_timings': phase_timings,
                'scope': 'No completed semantic projection or complete frontier is claimed; original input was not changed.'}
    from .proposal_decision import prepare_decision_context, build_decisions
    captured_original = original if isinstance(original, dict) else declarations.get('original', original)
    phase_started = time.perf_counter()
    decision_context = prepare_decision_context(captured_original, analysis, check_coverage=search_mode == 'exact',
                                                facts=metadata.facts)
    semantic = decision_context.semantic
    phase_timings['source_check_seconds'] = time.perf_counter() - phase_started
    if checkpoint:
        checkpoint({'stage': 'analysis', 'analysis': analysis, 'semantic_check': semantic,
                    'phase_timings': dict(phase_timings)})
    scenarios: list[Scenario] = []
    phase_timings['frontier_search_seconds'] = 0.0
    phase_timings['frontier_check_seconds'] = 0.0
    allocation = declarations.get('allocation', {})
    allocation_known = allocation.get('status') in ('known', 'declared') and allocation.get('kind') in ('per_block', 'global_union')
    if allocation_known:
        allocations = [allocation]
    else:
        allocations = [{'status': 'conditional_scenario', 'kind': name, 'boundary_unit': 1,
                        'assumption': 'representation scenario; actual backend allocation is missing'}
                       for name in ('per_block', 'global_union')]
    for contract in allocations:
        scoped = dict(analysis, allocation_scenario=contract)
        remaining = max(0, (limits or {}).get('seconds', 900) - (time.perf_counter() - started))
        try:
            hook = (lambda partial: checkpoint({'stage': 'frontier', 'allocation_contract': contract,
                                                 'partial': partial})) if checkpoint else None
            phase_started = time.perf_counter()
            if search_mode == 'heuristic':
                from .heuristic import search_heuristic
                result = search_heuristic(scoped, limits=dict(limits or {}, seconds=remaining),
                    recommendation=recommendation, effort=heuristic_effort, checkpoint=hook)
            else:
                result = search_frontier(scoped, limits=dict(limits or {}, seconds=remaining),
                                         recommendation=recommendation, checkpoint=hook)
        except (ValueError, KeyError) as exc:
            phase_timings['frontier_search_seconds'] += time.perf_counter() - phase_started
            scenarios.append({'allocation_contract': contract, 'status': 'UNRESOLVED', 'error': str(exc), 'frontier': []})
            continue
        phase_timings['frontier_search_seconds'] += time.perf_counter() - phase_started
        if search_mode == 'heuristic':
            from .heuristic_check import check_heuristic_frontier as check_frontier
        else:
            from .frontier_check import check_frontier
        checking_seconds = max(0, (limits or {}).get('seconds', 900) - (time.perf_counter() - started))
        phase_started = time.perf_counter()
        result['independent_check'] = check_frontier(scoped, result, limits=dict(limits or {}, seconds=checking_seconds))
        phase_timings['frontier_check_seconds'] += time.perf_counter() - phase_started
        scenarios.append(result)
    complete = analysis['discovery_complete'] and all(s['status'] == 'COMPLETE' for s in scenarios)
    failed = semantic.get('status') == 'FAIL' or any(s.get('independent_check', {}).get('status') == 'FAIL' for s in scenarios)
    unresolved = semantic.get('status') != 'PASS' or bool(analysis['obligations']) or not allocation_known or any(s.get('independent_check', {}).get('status') != 'PASS' for s in scenarios)
    exhausted = not analysis['discovery_complete'] or any(s['status'] == 'INCOMPLETE' or
                    s.get('independent_check', {}).get('status') == 'INCOMPLETE' or
                    s.get('independent_check', {}).get('checking_complete') is False for s in scenarios)
    phase_started = time.perf_counter()
    decisions = build_decisions(captured_original, analysis, scenarios,
                                seconds=max(0, (limits or {}).get('seconds', 900) - (time.perf_counter() - started)),
                                _context=decision_context)
    phase_timings['proposal_check_seconds'] = time.perf_counter() - phase_started
    proposal_exhausted = any(any(item.get('kind') == 'proposal checking budget exhausted'
                                  for item in record.get('residual_obligations', []))
                             for record in decisions)
    exhausted = exhausted or proposal_exhausted
    status = 'INVALID' if failed else 'INCOMPLETE' if exhausted else 'UNRESOLVED' if unresolved or not complete else 'COMPLETE'
    recommendations = []
    if search_mode == 'heuristic':
        from .heuristic_check import choose_candidate
        for index, scenario in enumerate(scenarios):
            if scenario.get('schema') != 'regime-heuristic-frontier/v1':
                recommendations.append({'scenario_index': index, 'graph_id': 'retain-original',
                    'operation': 'retain_original', 'authority': 'retain_original',
                    'reason': 'No independently checked heuristic candidate set is available'})
                continue
            allowed = {record['graph_id'] for record in decisions
                       if record.get('scenario_index') == index and record.get('disposition') == 'adopt'}
            if failed or scenario.get('independent_check', {}).get('status') != 'PASS':
                allowed = set()
            chosen = choose_candidate(dict(analysis, allocation_scenario=scenario['allocation_contract']),
                                      scenario, allowed_ids=allowed)
            chosen.update(scenario_index=index,
                          authority='source_checked' if chosen['operation'] == 'representation_specification'
                                    else 'retain_original')
            recommendations.append(chosen)
        # HEURISTIC describes this completed effort, never complete coverage.
        # Unresolved candidate premises select retain-original, not a bypass.
        status = 'INVALID' if failed else 'HEURISTIC'
    return {'schema': 'regime-production-result/v1',
            'status': status, 'search_mode': search_mode, 'facts_backend': facts_backend,
            'recommendations': recommendations,
            'analysis_scope': declarations.get('analysis_scope', 'full_semantics'),
            'source_identity': declarations.get('source_identity'), 'analysis': analysis,
            'semantic_check': semantic, 'scenarios': scenarios,
            'proposal_decisions': decisions,
            'production_cost_claim': 'conditional' if not allocation_known else 'declared allocation contract',
            'layers': {'facts': facts_backend, 'source_semantics': semantic.get('status'),
                       'discovery': 'COMPLETE' if analysis['discovery_complete'] else 'INCOMPLETE',
                       'allocation': 'DECLARED' if allocation_known else 'UNRESOLVED',
                       'frontier': ('RETAINED_CANDIDATES_CHECKED' if search_mode == 'heuristic' else 'VERIFIED')
                           if all(s.get('independent_check', {}).get('status') == 'PASS' for s in scenarios) else 'UNRESOLVED',
                       'proposal_checking': 'INCOMPLETE' if proposal_exhausted else 'COMPLETE',
                       'native_numerical_execution': 'NOT_RUN'},
            'elapsed_seconds': time.perf_counter() - started,
            'phase_timings': phase_timings,
            'scope': ('heuristic candidate recommendations; only independently adopted local specifications may be selected; '
                      'retained costs are checked without global-family replay; unknown optimality gap; native numerical execution not performed'
                      if search_mode == 'heuristic' else
                      'metadata-only source-bound semantic projections and complete finite grouping/interval frontiers when all obligations and checks pass; native numerical execution not performed')}
