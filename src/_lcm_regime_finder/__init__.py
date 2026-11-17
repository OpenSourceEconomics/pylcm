"""Original-declaration discovery and independently checkable layout curves."""


def is_production_input(problem):
    """Only the explicit legacy scalar schema uses the finite regression engine."""
    if not isinstance(problem, dict):
        return True
    if any(k in problem for k in ('regimes', 'user_regimes', 'graph_snapshot', 'declaration_spec')):
        return True
    if problem.get('schema') in ('regime-declarations/v1', 'constructed-pylcm-model-graph-1'):
        return True
    return not (problem.get('schema_version') == 1 and isinstance(problem.get('programs'), list))


def find_regimes(problem, budget=None, *, facts_backend=None, recommendation=None,
                 production_contract=None, analysis_scope='graph'):
    """Analyze a problem; a production analysis needs `facts_backend`."""
    if is_production_input(problem):
        from .production import analyze_production
        if facts_backend is None:
            raise TypeError("find_regimes() needs the keyword argument 'facts_backend' for a production analysis")
        return analyze_production(problem, limits=dict(budget or {}), recommendation=recommendation,
                                  production_contract=production_contract, analysis_scope=analysis_scope,
                                  facts_backend=facts_backend)
    from .core import find_regimes as finite_find
    return finite_find(problem, budget)


def analyze_production(problem, **kwargs):
    from .production import analyze_production as analyze
    return analyze(problem, **kwargs)


def verify_certificate(problem, report):
    if report.get('schema') == 'regime-production-result/v2':
        return verify_production_result(problem, report)
    from .certificate import verify_certificate as verify
    return verify(problem, report)


def verify_production_result(original, result, *, limits=None):
    """Replay the source check, re-check the layout curve and recompute the recommendation."""
    import time

    from .facts import make_facts
    from .layout_check import check_layouts
    from .production import original_cost, recommend
    from .semantic_check import check_semantics
    started = time.perf_counter()
    analysis = result.get('analysis', {})
    declarations = analysis.get('declarations', {})
    if result.get('source_identity') != declarations.get('source_identity') and declarations:
        return {'status': 'FAIL', 'error': 'Result source identity differs from its checked declarations'}
    if not isinstance(original, dict):
        from .model_snapshot import read_model
        original = read_model(original)
    semantic = check_semantics(original, declarations, analysis)
    layouts = result.get('layouts')
    errors = []
    if layouts is None:
        check = {'status': 'NOT_RUN', 'errors': []}
    else:
        limits = {'seconds': 900, 'metadata_categories': 100_000} | dict(limits or {})

        def guard():
            if time.perf_counter() - started > limits['seconds']:
                raise TimeoutError('Result verification budget exhausted')

        try:
            facts = make_facts(result['facts_backend'], declarations, limits, guard)
            check = check_layouts(declarations, facts, layouts, guard)
        except TimeoutError as exc:
            check = {'status': 'UNRESOLVED', 'errors': [], 'reason': str(exc)}
        else:
            expected = recommend(layouts, original_cost(declarations, facts.domains),
                                 result.get('recommendation_preferences'))
            if expected != result.get('recommendation'):
                errors.append('The saved recommendation differs from the recomputed one')
    statuses = [semantic.get('status'), check['status']]
    status = ('FAIL' if 'FAIL' in statuses or errors else
              'PASS' if all(s == 'PASS' for s in statuses) and not analysis.get('obligations') else 'UNRESOLVED')
    return {'status': status, 'source_semantics': semantic, 'layout_check': check, 'errors': errors,
            'elapsed_seconds': time.perf_counter() - started,
            'scope': 'independent original-source replay and layout-curve check; native numerical execution NOT_RUN'}


__all__ = ['find_regimes', 'analyze_production', 'verify_certificate', 'verify_production_result']
