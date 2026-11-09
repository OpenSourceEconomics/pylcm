"""Metadata-only semantic discovery and Pareto recommendations.

This entry point never calls a model function, compiles a solver, or enumerates
the state/action Cartesian product. Categorical characteristic tables concern
one explicitly finite original coordinate, under a small pure language.
"""
from __future__ import annotations
from copy import deepcopy
from itertools import product, combinations
import ast
import time

from .records import Scenario
from .expressions import partial, jsonable, AdmissionError
from .frontier import canonical, digest, search_frontier


class MetadataAnalysis:
    def __init__(self, declarations, limits=None, *, search_mode='exact'):
        if search_mode not in {'heuristic', 'exact'}:
            raise ValueError('search_mode must be heuristic or exact')
        self.search_mode = search_mode
        self.decl = declarations
        self.cells = {c['id']: c for c in declarations['cells']}
        if len(self.cells) != len(declarations['cells']):
            raise ValueError('Duplicate exact admission cell')
        self.callbacks = declarations['callbacks']
        self.domains = declarations['domains']
        self.limits = {'seconds': 900, 'metadata_categories': 100000,
                       'projection_policies': 4096} | (limits or {})
        self.started = time.perf_counter()
        self.obligations = deepcopy(declarations.get('obligations', []))
        self._cache = {}
        self._env_cache = {}
        self._interface_cache = {}
        self.forced = {}
        self.constants = {}
        self.live = {}
        self.reasons = {}
        for c in self.cells.values():
            for kind in ('states', 'actions'):
                if not isinstance(c[kind], dict):
                    raise ValueError(f'{c["id"]}: {kind} must map axis names to domains')
                for name, domain in c[kind].items():
                    if domain not in self.domains:
                        raise ValueError(f'Unknown domain {name}/{domain}')
            self.constants[c['id']] = {
                name: self.domains[domain]['values'][0] for name, domain in c['states'].items()
                if self.domains[domain].get('size') == 1
                and len(self.domains[domain].get('values', [])) == 1}
            self.forced[c['id']] = self.forced_actions(c)
            self._env_cache.clear()
        for relation in declarations['relations']:
            if relation['source'] not in self.cells or any(t not in self.cells for t in relation.get('targets', [])):
                raise ValueError('Relation names an undeclared exact admission')

    def guard(self):
        if time.perf_counter() - self.started >= self.limits['seconds']:
            raise TimeoutError('Metadata dependency analysis budget exhausted')

    @staticmethod
    def scalar_bindings(callback):
        return {k: v for k, v in callback.get('bindings', {}).items()
                if isinstance(v, (bool, int, float, str, list)) or v is None}

    def environment(self, cell, callback=None, phase='solve', overrides=None):
        """Bind provably known date/parameter DAG values, never native functions.

        A declaration may represent age as ``age(period)``. Treating that value
        as unknown would miss a sustained feasibility restriction. This finite
        fixed point evaluates only the admitted scalar expression language.
        """
        key = (cell['id'], phase)
        cached = self._env_cache.get(key) if not overrides else None
        if cached is not None:
            env = dict(cached)
        else:
            env = {k: cell[k] for k in ('age', 'period') if cell.get(k) is not None}
            env.update(self.decl.get('parameters', {}))
            env.update(self.constants.get(cell['id'], {}))
            env.update(self.forced.get(cell['id'], {}))
            env.update(overrides or {})
            env = self.bind_codes(cell, env)
            providers = cell.get('phase_functions', {}).get(phase, cell.get('functions', {}))
            pending = dict(providers)
            for _ in range(len(pending) + 1):
                changed = False
                for name, cbid in list(pending.items()):
                    cb = self.callbacks[cbid]
                    if name in env:
                        pending.pop(name)
                        continue
                    if cb.get('status') != 'proved' or cb.get('expression') is None:
                        continue
                    try:
                        value = partial(cb['expression'], self.scalar_bindings(cb) | env)
                    except (AdmissionError, TypeError, ValueError, ArithmeticError):
                        continue
                    if value.known:
                        env[name] = value.value
                        pending.pop(name)
                        changed = True
                for alias, name in cell.get('derived_code_providers', {}).items():
                    if name in env and alias not in env:
                        domain = self.domains[cell['derived_categoricals'][name]]
                        values, codes = domain.get('values', []), domain.get('codes', [])
                        matches = [i for i, v in enumerate(values) if v == env[name]]
                        if len(matches) == 1 and len(values) == len(codes):
                            env[alias] = codes[matches[0]]
                            changed = True
                if not changed:
                    break
            if not overrides and cell['id'] in self.forced:
                self._env_cache[key] = dict(env)
        return self.bind_codes(cell, (self.scalar_bindings(callback) if callback else {}) | env)

    @staticmethod
    def bind_codes(cell, env):
        env = dict(env)
        for axis, record in cell.get('category_codes', {}).items():
            if axis not in env:
                continue
            values, codes = record.get('values', []), record.get('codes', [])
            matches = [i for i, value in enumerate(values) if canonical(value) == canonical(env[axis])]
            if len(matches) == 1 and len(codes) == len(values):
                env[record['argument']] = codes[matches[0]]
        return env

    def direct_reads(self, callback_id, cell, overrides=None, phase='solve'):
        cb = self.callbacks[callback_id]
        env = self.environment(cell, cb, phase, overrides)
        expression = cb.get('expression')
        if expression is not None:
            try:
                return set(partial(expression, env).reads)
            except (AdmissionError, TypeError, ValueError):
                pass
        return set(cb.get('reads', cb.get('arguments', []))) - set(env)

    def expanded_reads(self, callback_id, cell, phase='solve', stops=frozenset(), trail=frozenset()):
        key = (callback_id, cell['id'], phase, tuple(sorted(stops)))
        if key in self._cache:
            return set(self._cache[key])
        if callback_id in stops:
            return set()
        if callback_id in trail:
            raise ValueError(f'Cyclic derived function DAG at {callback_id}')
        cb = self.callbacks[callback_id]
        if cb.get('status') == 'unresolved':
            return set(cell['states']) | set(cell['actions'])
        providers = cell.get('phase_functions', {}).get(phase, cell.get('functions', {}))
        result = set()
        code_axes = {v['argument']: k for k, v in cell.get('category_codes', {}).items()}
        for name in self.direct_reads(callback_id, cell, phase=phase):
            name = code_axes.get(name, name)
            name = cell.get('derived_code_providers', {}).get(name, name)
            if name in cell['states'] or name in cell['actions']:
                result.add(name)
            elif name in providers:
                result.update(self.expanded_reads(providers[name], cell, phase, stops, trail | {callback_id}))
        self._cache[key] = frozenset(result)
        return result

    @staticmethod
    def phases(record):
        phase = record.get('phase', 'both')
        return ('solve', 'simulation') if phase == 'both' else (phase,)

    def consumer_reads(self, cell, stops=frozenset()):
        reads = set()
        for consumer in cell.get('consumers', []):
            cb = consumer.get('callback')
            if cb:
                for phase in self.phases(consumer):
                    reads.update(self.expanded_reads(cb, cell, phase, stops))
        # Probability, gate, exit and read weights are always consumers. Updates
        # are propagated separately only when the destination coordinate is live.
        for relation in self.decl['relations']:
            if relation['source'] != cell['id'] or relation.get('support') == 'zero':
                continue
            for cb in relation.get('callbacks', []):
                for phase in self.phases(relation):
                    reads.update(self.expanded_reads(cb, cell, phase, stops))
        return reads

    def update_reads(self, relation, target_axis, stops=frozenset()):
        source = self.cells[relation['source']]
        update = relation.get('state_updates', {}).get(target_axis)
        if update is None:
            # Unknown reconstruction cannot justify removal of any source input.
            return set(source['states']) | set(source['actions'])
        if isinstance(update, str) and update in source['states']:
            return {update}
        if isinstance(update, str) and update in self.callbacks:
            result = set()
            for phase in self.phases(relation):
                result.update(self.expanded_reads(update, source, phase, stops))
            return result
        if isinstance(update, dict):
            if update.get('kind') == 'identity':
                return {update.get('source', target_axis)}
            if update.get('kind') == 'constant':
                return set()
            if update.get('callback') in self.callbacks:
                result = set()
                for phase in self.phases(relation):
                    result.update(self.expanded_reads(update['callback'], source, phase, stops))
                return result
        return set(source['states']) | set(source['actions'])

    def identity_update(self, update, axis, target_axis):
        if update == axis:
            return True
        if isinstance(update, dict):
            if update.get('kind') == 'identity':
                return update.get('source', target_axis) == axis
            update = update.get('callback')
        if isinstance(update, str) and update in self.callbacks:
            cb = self.callbacks[update]
            if cb.get('status') == 'proved' and isinstance(cb.get('expression'), str):
                try:
                    node = ast.parse(cb['expression'], mode='eval').body
                    return isinstance(node, ast.Name) and node.id == axis
                except SyntaxError:
                    pass
        return False

    def forced_actions(self, cell):
        result = {}
        constraints = [x for x in cell.get('consumers', [])
                       if x.get('kind') in ('constraint', 'feasibility', 'feasible') and x.get('callback')]
        decision_phases = cell.get('decision_phases', ('solve', 'simulation'))
        for name, domain_id in cell['actions'].items():
            self.guard()
            domain = self.domains[domain_id]
            values = domain.get('values')
            if not isinstance(values, list) or len(values) != domain.get('size') or len(values) > self.limits['metadata_categories']:
                continue
            possible = []
            for value in values:
                self.guard()
                # A shared action interface must cover the UNION of the
                # phase-specific feasible sets. Rejection by the simulation
                # constraints does not remove a choice available to solve.
                phase_allowed = []
                for decision_phase in decision_phases:
                    allowed = True
                    for constraint in constraints:
                        applies = constraint.get('decision_phases')
                        if applies is None:
                            evaluation_phases = (decision_phase,) if decision_phase in self.phases(constraint) else ()
                        else:
                            evaluation_phases = self.phases(constraint) if decision_phase in applies else ()
                        if not evaluation_phases:
                            continue
                        cb = self.callbacks[constraint['callback']]
                        if cb.get('expression') is None or cb.get('status') != 'proved':
                            continue
                        for phase in evaluation_phases:
                            try:
                                v = partial(cb['expression'], self.environment(cell, cb, phase, {name: value}))
                            except (AdmissionError, TypeError, ValueError, ArithmeticError):
                                continue
                            if v.known and v.value is False:
                                allowed = False
                                break
                        if not allowed:
                            break
                    phase_allowed.append(allowed)
                if any(phase_allowed):
                    possible.append(value)
            if not possible:
                raise ValueError(f'{cell["id"]}: no admissible value of action {name}')
            if len(possible) == 1:
                result[name] = possible[0]
        return result

    def dependencies(self):
        for cell in self.cells.values():
            reads = self.consumer_reads(cell)
            self.live[cell['id']] = set(cell['states']) & reads - set(self.constants[cell['id']])
            self.reasons[cell['id']] = {name: ['current/phase/required-output consumer'] for name in self.live[cell['id']]}
        changed = True
        while changed:
            self.guard()
            changed = False
            for relation in self.decl['relations']:
                if relation.get('support') == 'zero':
                    continue
                sid = relation['source'];source = self.cells[sid]
                for tid in relation.get('targets', []):
                    for target_axis in tuple(self.live[tid]):
                        if 'produces' in relation and target_axis not in relation['produces']:
                            continue
                        needed = self.update_reads(relation, target_axis) & set(source['states']) - set(self.constants[sid])
                        new = needed - self.live[sid]
                        if new:
                            changed = True;self.live[sid].update(new)
                            for name in new:
                                self.reasons[sid][name] = [f'{relation["id"]} -> {tid}/{target_axis}']

    def characteristic_candidates(self):
        result = []
        for name, spec in sorted(self.decl.get('characteristics', {}).items()):
            self.guard()
            axis, domain_id, cb_id = spec['state'], spec['domain'], spec['callback']
            domain = self.domains[domain_id];cb = self.callbacks[cb_id]
            values = domain.get('values')
            if domain.get('kind') != 'categorical' or cb.get('status') != 'proved' or cb.get('expression') is None:
                continue
            if not isinstance(values, list) or len(values) != domain.get('size') or len(values) > self.limits['metadata_categories']:
                continue
            mapping = []
            try:
                for index, value in enumerate(values):
                    self.guard()
                    env = {k: v for k, v in cb.get('bindings', {}).items()
                           if isinstance(v, (bool, int, float, str, list)) or v is None}
                    env[axis] = value
                    code_arguments = {c.get('category_codes', {}).get(axis, {}).get('argument')
                                      for c in self.cells.values() if c['states'].get(axis) == domain_id}
                    code_arguments.discard(None)
                    if spec.get('code_argument'):
                        code_arguments.add(spec['code_argument'])
                    for argument in code_arguments:
                        if len(domain.get('codes', [])) == len(values):
                            env[argument] = domain['codes'][index]
                    observed = partial(cb['expression'], env)
                    if not observed.known:
                        raise ValueError('Characteristic has unresolved original arguments')
                    mapping.append(jsonable(observed.value))
            except (AdmissionError, TypeError, ValueError):
                continue
            unique = {canonical(v): v for v in mapping}
            if len(unique) >= len(values):
                continue
            # Aliases of the exact original characteristic retain their identity.
            stops = frozenset(k for k, value in self.callbacks.items()
                              if value.get('identity') == cb.get('identity'))
            eligible = {cid for cid, cell in self.cells.items()
                        if cell['states'].get(axis) == domain_id and axis in self.live[cid]
                        and axis not in self.consumer_reads(cell, stops)}
            changed = True
            while changed:
                changed = False
                for relation in self.decl['relations']:
                    sid = relation['source']
                    if sid not in eligible or relation.get('support') == 'zero':
                        continue
                    for tid in relation.get('targets', []):
                        for target_axis in self.live[tid]:
                            if 'produces' in relation and target_axis not in relation['produces']:
                                continue
                            update = relation.get('state_updates', {}).get(target_axis)
                            identity = self.identity_update(update, axis, target_axis)
                            if target_axis == axis and identity:
                                safe = tid in eligible
                            else:
                                safe = axis not in self.update_reads(relation, target_axis, stops)
                            if not safe:
                                eligible.discard(sid);changed = True;break
                        if sid not in eligible:
                            break
            if eligible:
                target_name = spec.get('name', name)
                domain_key = 'characteristic:' + digest([cb.get('identity'), domain_id, mapping])[:24]
                requires = {cid: set() for cid in eligible}
                for relation in self.decl['relations']:
                    sid = relation['source']
                    if sid not in eligible or relation.get('support') == 'zero':
                        continue
                    if 'produces' in relation and axis not in relation['produces']:
                        continue
                    update = relation.get('state_updates', {}).get(axis)
                    identity = self.identity_update(update, axis, axis)
                    if identity:
                        requires[sid].update(tid for tid in relation.get('targets', []) if tid in eligible and axis in self.live[tid])
                result.append({'id': name, 'state': axis, 'domain': domain_id, 'callback': cb_id,
                               'target': target_name, 'values': mapping,
                               'target_values': [unique[k] for k in sorted(unique)],
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

    def function_interfaces(self, cell, projection, constants):
        """A source-bound DAG rewrite plan, with actual reduced signatures.

        It is deliberately an inspectable plan, not a claim that a native solver
        was compiled. Existing pure numerical bodies and joint laws stay pinned.
        """
        key = (cell['id'], canonical(projection), canonical(constants))
        if key in self._interface_cache:
            return deepcopy(self._interface_cache[key])
        rewritten = {}
        identities = {self.callbacks[v['callback']].get('identity') for v in projection.values()
                      if v['kind'] == 'characteristic'}
        stops = frozenset(k for k, cb in self.callbacks.items() if cb.get('identity') in identities)
        for phase in ('solve', 'simulation'):
            providers = cell.get('phase_functions', {}).get(phase, cell.get('functions', {}))
            records = {}
            for name, cbid in providers.items():
                cb = self.callbacks[cbid]
                environment = self.environment(cell, cb, phase) | constants
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
                reads = self.direct_reads(cbid, cell, overrides=constants, phase=phase)
                state_reads = self.expanded_reads(cbid, cell, phase, stops)
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

    def family_boundary(self):
        """Make conditional cross-regime questions visible without adopting them.

        Missing actor/information authority cannot be filled from equal shapes.
        These candidates are reporting records, not additions to the optimizer's
        admitted family. There is one record per original-regime pair, not per
        date pair, and enumeration is bounded independently of grid sizes.
        """
        representatives = {}
        for cell in self.cells.values():
            representatives.setdefault(cell['regime'], cell)
        candidates = []
        omitted = 0
        names = sorted(representatives)
        limit = self.limits.get('conditional_merge_candidates', 2048)
        protected = ('terminal', 'stage', 'ownership', 'declaration_type',
                     'structural_roles', 'solver_identity', 'report_tags')
        pairs_visited = 0
        pair_budget = self.limits.get('conditional_merge_pair_checks', 8192)
        for left_name, right_name in combinations(names, 2):
            self.guard()
            if pairs_visited >= pair_budget:
                break
            pairs_visited += 1
            left, right = representatives[left_name], representatives[right_name]
            if left is None or right is None:
                raise ValueError('Missing representative declaration')
            if left.get('role') is not None and right.get('role') is not None:
                continue
            # A known conflicting protected fact is not a useful
            # conditional merge suggestion. Missing facts stay missing.
            conflict = [key for key in protected if left.get(key) is not None
                        and right.get(key) is not None
                        and canonical(left.get(key)) != canonical(right.get(key))]
            if conflict:
                continue
            if len(candidates) >= limit:
                omitted += 1
                continue
            candidates.append({
                'regimes': [left_name, right_name],
                'source_paths': [left.get('source_path'), right.get('source_path')],
                'status': 'CONDITIONAL_NOT_ADMITTED',
                'known_matching_fields': [key for key in protected
                    if left.get(key) is not None and right.get(key) is not None],
                'required_obligations': [
                    {'kind': 'actor_information_timing_equivalence',
                     'component': 'native declaration context adapter',
                     'required_fact': 'Source-bound equality of actor/stakeholder, information, commitment and discount timing for both original regimes.'},
                    {'kind': 'merge_semantic_and_incidence_check',
                     'component': 'semantic and frontier checkers',
                     'required_fact': 'After context authority is supplied, check feasible actions, rewards, joint/phased laws, requested outputs and every owned current-policy/read route for the proposed group.'}],
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
            'schema': 'regime-discovery-family/v2',
            'complete_claim_scope': 'coherent source-bound projection policies and the explicitly admitted context/grouping interval family',
            'unrestricted_model_representation_optimum_claimed': False,
            'original_regime_labels_are_semantic_restrictions': False,
            'unknown_context_policy': 'retain original context in the admitted family; expose conditional merge obligations separately',
            'endpoint_preference_is_admissibility_rule': False,
            'conditional_merge_candidates': candidates,
            'conditional_merge_candidates_omitted_by_display_budget': omitted,
            'conditional_merge_pairs_unexamined': len(names) * (len(names) - 1) // 2 - pairs_visited,
            'callback_obligations': opaque,
            'parameter_family_sharing': 'NOT_ESTABLISHED: each source/fixed-input identity requires its own checked feasibility, liveness and support closure',
        }

    def run(self):
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
        return {'schema': 'regime-semantic-analysis/v1', 'declarations': self.decl,
                'interfaces': interfaces, 'derived_domains': derived,
                'projection_policies': policies, 'characteristic_candidates': candidates,
                'dependency_closure': {c: sorted(v) for c, v in self.live.items()},
                'discovery_complete': policy_complete,
                'family_boundary': self.family_boundary(),
                'obligations': self.obligations,
                'production_execution': {'callbacks_called': 0, 'bellman_rows_evaluated': 0,
                                         'dense_cartesian_cells_enumerated': 0,
                                         'characteristic_metadata_values': sum(len(c['values']) for c in candidates)},
                'elapsed_seconds': time.perf_counter() - self.started}


def analyze_production(original, *, limits=None, recommendation=None, production_contract=None,
                       checkpoint=None, analysis_scope='graph', search_mode='heuristic',
                       heuristic_effort=None):
    if search_mode not in {'heuristic', 'exact'}:
        raise ValueError('search_mode must be heuristic or exact')
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
        analysis = MetadataAnalysis(declarations, limits, search_mode=search_mode).run()
        phase_timings['metadata_analysis_seconds'] = time.perf_counter() - phase_started
        if checkpoint:
            checkpoint({'stage': 'metadata_analysis', 'analysis': analysis,
                        'phase_timings': dict(phase_timings)})
    except (TimeoutError, ValueError, KeyError, TypeError, ArithmeticError) as exc:
        phase_timings.setdefault(active_phase + '_seconds', time.perf_counter() - phase_started)
        incomplete = isinstance(exc, TimeoutError)
        return {'schema': 'regime-production-result/v1',
                'status': 'INCOMPLETE' if incomplete else 'UNRESOLVED',
                'search_mode': search_mode,
                'recommendations': ([{'scenario_index': None, 'graph_id': 'retain-original',
                    'operation': 'retain_original', 'authority': 'retain_original',
                    'reason': 'Normalization or metadata analysis did not establish a checked proposal'}]
                    if search_mode == 'heuristic' else []),
                'error': type(exc).__name__ + ': ' + str(exc),
                'scenarios': [], 'analysis': {'declarations': locals().get('declarations', {}),
                                            'obligations': [{'kind': 'metadata_admission', 'detail': str(exc)}]},
                'semantic_check': {'status': 'NOT_RUN'},
                'established_lower_bound': dict.fromkeys(('work', 'storage', 'clarity'), 0),
                'layers': {'source_semantics': 'NOT_RUN', 'discovery': 'INCOMPLETE',
                           'allocation': 'UNRESOLVED', 'frontier': 'NOT_RUN',
                           'native_numerical_execution': 'NOT_RUN'},
                'elapsed_seconds': time.perf_counter() - started,
                'phase_timings': phase_timings,
                'scope': 'No completed semantic projection or complete frontier is claimed; original input was not changed.'}
    from .proposal_decision import prepare_decision_context, build_decisions
    captured_original = original if isinstance(original, dict) else declarations.get('original', original)
    phase_started = time.perf_counter()
    decision_context = prepare_decision_context(captured_original, analysis, check_coverage=search_mode == 'exact')
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
            'status': status, 'search_mode': search_mode,
            'recommendations': recommendations,
            'analysis_scope': declarations.get('analysis_scope', 'full_semantics'),
            'source_identity': declarations.get('source_identity'), 'analysis': analysis,
            'semantic_check': semantic, 'scenarios': scenarios,
            'proposal_decisions': decisions,
            'production_cost_claim': 'conditional' if not allocation_known else 'declared allocation contract',
            'layers': {'source_semantics': semantic.get('status'),
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
