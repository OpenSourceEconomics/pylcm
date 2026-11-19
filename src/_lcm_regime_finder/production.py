"""Regime analysis: facts, their source check, and the Supplement's layout curve.

This entry point never calls a model function, compiles a solver, or enumerates
the state/action Cartesian product.
"""
from __future__ import annotations
from copy import deepcopy
import time
from math import prod

from .expressions import jsonable
from .facts import FACTS_BACKENDS, FactsBudgetError, MetadataFacts, live_states, make_facts
from .records import canonical, digest


class MetadataAnalysis:
    def __init__(self, declarations, limits=None, *, facts_backend):
        if facts_backend not in FACTS_BACKENDS:
            raise ValueError('facts_backend must be one of ' + ', '.join(FACTS_BACKENDS))
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

    def function_interfaces(self, cell, projection, constants):
        """A source-bound DAG rewrite plan, with actual reduced signatures.

        It is deliberately an inspectable plan, not a claim that a native solver
        was compiled. Existing pure numerical bodies and joint laws stay pinned.
        """
        key = (cell['id'], canonical(projection), canonical(constants))
        if key in self._interface_cache:
            return deepcopy(self._interface_cache[key])
        rewritten = {}
        for phase in ('solve', 'simulation'):
            providers = cell.get('phase_functions', {}).get(phase, cell.get('functions', {}))
            records = {}
            for name, cbid in providers.items():
                cb = self.callbacks[cbid]
                environment = self.interpreter.environment(cell, cb, phase) | constants
                reads = self.interpreter.direct_reads(cbid, cell, overrides=constants, phase=phase)
                state_reads = self.facts.callback_reads(cbid, cell, phase)
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

    def run(self):
        try:
            return self._run()
        except TimeoutError:
            if self.facts.name == 'metadata':
                raise
            raise FactsBudgetError(backend=self.facts.name, limit='time budget',
                                   unfinished=sorted(set(self.cells) - set(self.live)) or sorted(self.cells)) from None

    def _run(self):
        """One identity interface per cell: its live states and its unforced actions.

        The semantic check verifies these interfaces against the source; the
        layout search reads the same facts at every fibre.
        """
        self.dependencies()
        interfaces = {}
        policy = {}
        for cid, cell in self.cells.items():
            self.guard()
            states = {s: d for s, d in cell['states'].items() if s in self.live[cid]}
            actions = {a: d for a, d in cell['actions'].items() if a not in self.forced[cid]}
            projection = {s: {'kind': 'identity', 'target': s} if s in states else {'kind': 'drop'}
                          for s in cell['states']}
            constants = dict(self.constants[cid])
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
                      'witnesses': [{'kind': 'dependency_closure', 'retained': self.reasons[cid]}]}
            option['id'] = 'interface:' + digest(option)[:20]
            interfaces[cid] = {'options': [option]}
            policy[cid] = option['id']
        return {'schema': 'regime-semantic-analysis/v1', 'declarations': self.decl,
                'facts_backend': self.facts.name,
                'interfaces': interfaces, 'derived_domains': {},
                'projection_policies': [{'id': 'projection:identity', 'option_by_cell': policy,
                                         'characteristics': []}],
                'dependency_closure': {c: sorted(v) for c, v in self.live.items()},
                'discovery_complete': True,
                'obligations': self.obligations,
                'production_execution': {'callbacks_called': 0, 'bellman_rows_evaluated': 0,
                                         'dense_cartesian_cells_enumerated': 0},
                'elapsed_seconds': time.perf_counter() - self.started}


def _score(cost, reference, weights):
    return sum(weights[k] * cost[k] / reference[k] for k in ('work', 'storage', 'clarity'))


def recommend(layouts, original_cost, recommendation=None):
    """Return the curve point that minimizes the linear cost model, or retain the original.

    Work is evaluations, storage is stored values, and complexity is the number of
    blocks; each is scaled by the starting model's value unless reference scales
    are given. A point must improve the original's score, and by at least `minimum_improvement`.
    """
    recommendation = recommendation or {}
    weights = recommendation.get('weights', {'work': 1.0, 'storage': 1.0, 'clarity': 1.0})
    reference = recommendation.get('reference') or {k: max(1, v) for k, v in original_cost.items()}
    original = _score(original_cost, reference, weights)
    scored = [(_score({'work': p['evaluations'], 'storage': p['storage'], 'clarity': p['blocks']},
                      reference, weights), p['blocks'], index)
              for index, p in enumerate(layouts.get('curve', []))]
    if not scored:
        return {'operation': 'retain_original', 'reason': 'The layout curve is empty', 'score': original}
    score, _, index = min(scored)
    if score >= original * (1 - recommendation.get('minimum_improvement', 0.0)):
        return {'operation': 'retain_original', 'score': original,
                'reason': 'No curve point improves the original by the required margin'}
    return {'operation': 'layout', 'curve_index': index,
            'max_blocks': layouts['curve'][index]['max_blocks'], 'score': score, 'original_score': original}


def original_cost(declarations, domains):
    """The starting model's evaluations, stored values and regimes (ev^0, Supplement :851-853)."""
    def size(did):
        return 1 if domains[did].get('solve_axis') is False else domains[did]['size']
    cells = declarations['cells']
    # A collective regime stores one value per stakeholder.
    states = [prod(size(d) for d in c['states'].values()) * (len(c.get('stakeholders') or ()) or 1) for c in cells]
    actions = [prod(size(d) for d in c['actions'].values()) for c in cells]
    return {'work': sum(s * a for s, a in zip(states, actions, strict=True)), 'storage': sum(states),
            'clarity': len({c['regime'] for c in cells})}


def analyze_production(original, *, facts_backend, limits=None, recommendation=None,
                       production_contract=None, checkpoint=None, analysis_scope='graph'):
    """Analyze a model: certify its facts against the source and search its layout curve."""
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
        metadata = MetadataAnalysis(declarations, limits, facts_backend=facts_backend)
        analysis = metadata.run()
        phase_timings['metadata_analysis_seconds'] = time.perf_counter() - phase_started
    except (TimeoutError, ValueError, KeyError, TypeError, ArithmeticError) as exc:
        phase_timings.setdefault(active_phase + '_seconds', time.perf_counter() - phase_started)
        return {'schema': 'regime-production-result/v2',
                'status': 'INCOMPLETE' if isinstance(exc, TimeoutError) else 'UNRESOLVED',
                'facts_backend': facts_backend,
                'recommendation': {'operation': 'retain_original',
                                   'reason': 'Normalization or fact analysis did not complete'},
                'error': type(exc).__name__ + ': ' + str(exc),
                'layouts': None, 'layout_check': {'status': 'NOT_RUN'},
                'analysis': {'declarations': locals().get('declarations', {}),
                             'obligations': [{'kind': 'metadata_admission', 'detail': str(exc)}]},
                'semantic_check': {'status': 'NOT_RUN'},
                'layers': {'facts': facts_backend, 'source_semantics': 'NOT_RUN', 'layouts': 'NOT_RUN',
                           'native_numerical_execution': 'NOT_RUN'},
                'elapsed_seconds': time.perf_counter() - started, 'phase_timings': phase_timings,
                'scope': 'No checked layout curve is claimed; the original input was not changed.'}
    from .semantic_check import check_semantics
    captured_original = original if isinstance(original, dict) else declarations.get('original', original)
    phase_started = time.perf_counter()
    semantic = check_semantics(captured_original, declarations, analysis,
                               facts=metadata.facts)
    phase_timings['source_check_seconds'] = time.perf_counter() - phase_started
    if checkpoint:
        checkpoint({'stage': 'analysis', 'analysis': analysis, 'semantic_check': semantic,
                    'phase_timings': dict(phase_timings)})
    from .layout_check import check_layouts
    from .layout_search import search_layouts
    phase_started = time.perf_counter()
    try:
        layouts = search_layouts(declarations, metadata.facts, metadata.limits, metadata.guard)
    except NotImplementedError as exc:
        # A layout the search cannot price; its message says why to the user.
        layouts, error = None, str(exc)
    except (TimeoutError, ValueError) as exc:
        layouts, error = None, type(exc).__name__ + ': ' + str(exc)
    else:
        error = None
    phase_timings['layout_search_seconds'] = time.perf_counter() - phase_started
    phase_started = time.perf_counter()
    check = (check_layouts(declarations, metadata.facts, layouts) if layouts is not None
             else {'status': 'NOT_RUN', 'errors': []})
    phase_timings['layout_check_seconds'] = time.perf_counter() - phase_started
    cost = original_cost(declarations, metadata.domains)
    failed = semantic.get('status') == 'FAIL' or check['status'] == 'FAIL'
    if failed:
        status = 'INVALID'
    elif layouts is None:
        status = 'INCOMPLETE' if error and error.startswith('TimeoutError') else 'UNRESOLVED'
    elif semantic.get('status') != 'PASS' or analysis['obligations'] or layouts['status'] != 'COMPLETE':
        status = 'UNRESOLVED'
    else:
        status = 'COMPLETE'
    chosen = (recommend(layouts, cost, recommendation) if layouts is not None and not failed
              else {'operation': 'retain_original', 'reason': 'No checked layout curve'})
    return {'schema': 'regime-production-result/v2',
            'status': status, 'facts_backend': facts_backend,
            'recommendation': chosen, 'recommendation_preferences': recommendation, 'original_cost': cost,
            'analysis_scope': declarations.get('analysis_scope', 'full_semantics'),
            'source_identity': declarations.get('source_identity'), 'analysis': analysis,
            'semantic_check': semantic, 'layouts': layouts, 'layout_check': check,
            **({'error': error} if error else {}),
            'layers': {'facts': facts_backend, 'source_semantics': semantic.get('status'),
                       'layouts': 'NOT_RUN' if layouts is None else layouts['status'],
                       'layout_check': check['status'], 'native_numerical_execution': 'NOT_RUN'},
            'elapsed_seconds': time.perf_counter() - started,
            'phase_timings': phase_timings,
            'scope': ('The layout curve of the Supplement\'s family over the certified facts; '
                      'RESTRICTED counts are upper bounds; native numerical execution not performed.')}
