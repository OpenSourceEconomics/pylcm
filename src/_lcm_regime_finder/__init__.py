"""Original-declaration discovery and independently checkable recommendations."""


def is_production_input(problem):
    """Only the explicit legacy scalar schema uses the finite regression engine."""
    if not isinstance(problem, dict):
        return True
    if any(k in problem for k in ('regimes', 'user_regimes', 'graph_snapshot', 'declaration_spec')):
        return True
    if problem.get('schema') in ('regime-declarations/v1', 'constructed-pylcm-model-graph-1'):
        return True
    return not (problem.get('schema_version') == 1 and isinstance(problem.get('programs'), list))


def find_regimes(problem, budget=None, *, recommendation=None, production_contract=None, analysis_scope='graph',
                 search_mode='heuristic', heuristic_effort=None):
    if isinstance(problem, dict) and problem.get('schema') == 'exact-graph-cost-instance/v1':
        return find_cost_frontier(problem, seconds=(budget or {}).get('seconds', 900),
                                  limits=(budget or {}).get('cost_limits'))
    if is_production_input(problem):
        from .production import analyze_production
        limits = dict(budget or {})
        if 'candidate_segments' in limits:
            limits['segments'] = limits.pop('candidate_segments')
        return analyze_production(problem, limits=limits, recommendation=recommendation,
                                  production_contract=production_contract, analysis_scope=analysis_scope,
                                  search_mode=search_mode, heuristic_effort=heuristic_effort)
    from .core import find_regimes as finite_find
    return finite_find(problem, budget)


def analyze_production(problem, **kwargs):
    from .production import analyze_production as analyze
    return analyze(problem, **kwargs)

def verify_certificate(problem, report):
    if report.get('schema') == 'regime-cost-family/v1':
        return check_cost_frontier(problem, report)
    if report.get('schema') == 'regime-production-result/v1':
        return verify_production_result(problem, report)
    from .certificate import verify_certificate as verify
    return verify(problem, report)


def find_cost_frontier(analysis, **kwargs):
    """Optimize the exact metadata cost family; native source proof stays separate."""
    from .cost_family import find_cost_frontier as find
    return find(analysis, **kwargs)


def check_cost_frontier(original, result, **kwargs):
    from .cost_family_check import check_cost_frontier as check
    return check(original, result, **kwargs)




def verify_production_result(original, result, *, limits=None):
    """Replay original sources and the result's exact or heuristic contract."""
    from .proposal_decision import prepare_decision_context, verify_decisions
    from .frontier_check import check_frontier
    import time
    started = time.perf_counter()
    heuristic = result.get('search_mode') == 'heuristic'
    if heuristic and (result.get('status') in {'COMPLETE', 'OPTIMAL'}
                      or result.get('layers', {}).get('frontier') == 'VERIFIED'):
        return {'status': 'FAIL', 'error': 'Heuristic result claimed exact complete-family authority'}
    analysis = result.get('analysis', {})
    declarations = analysis.get('declarations', {})
    allocation = declarations.get('allocation', {})
    known = allocation.get('status') in ('known', 'declared') and allocation.get('kind') in ('per_block', 'global_union')
    scenarios = result.get('scenarios', [])
    if known and (len(scenarios) != 1 or scenarios[0].get('allocation_contract') != allocation):
        return {'status': 'FAIL', 'error': 'Frontier allocation differs from the original declared contract'}
    if not known and any(s.get('allocation_contract', {}).get('status') != 'conditional_scenario' for s in scenarios):
        return {'status': 'FAIL', 'error': 'Unknown backend allocation was promoted to an actual production contract'}
    if result.get('source_identity') != declarations.get('source_identity') and declarations:
        return {'status': 'FAIL', 'error': 'Result source identity differs from its checked declarations'}
    if not isinstance(original, dict):
        from .model_snapshot import read_model
        original = read_model(original)
    try:
        decision_context = prepare_decision_context(original, analysis, check_coverage=not heuristic)
        semantic = decision_context.semantic
    except (KeyError, TypeError, ValueError) as exc:
        return {'status': 'FAIL', 'error': f'{type(exc).__name__}: {exc}'}
    checks = []
    for scenario in result.get('scenarios', []):
        if heuristic:
            if scenario.get('schema') != 'regime-heuristic-frontier/v1':
                checks.append({'status': 'UNRESOLVED', 'checking_complete': False,
                               'error': 'No heuristic candidate set was supplied'})
                continue
            from .heuristic_check import check_heuristic_frontier
            remaining = max(0, (limits or {}).get('seconds', 900) - (time.perf_counter() - started))
            checks.append(check_heuristic_frontier(dict(analysis, allocation_scenario=scenario['allocation_contract']),
                scenario, limits=dict(limits or {}, seconds=remaining)))
            continue
        if scenario.get('schema') == 'regime-heuristic-frontier/v1':
            return {'status': 'FAIL', 'error': 'Heuristic scenario cannot claim exact result semantics'}
        if 'certificate' not in scenario:
            checks.append({'status': 'UNRESOLVED', 'error': scenario.get('error', 'no frontier certificate')})
            continue
        remaining = max(0, (limits or {}).get('seconds', 900) - (time.perf_counter() - started))
        checks.append(check_frontier(dict(analysis, allocation_scenario=scenario['allocation_contract']),
                                     scenario, limits=dict(limits or {}, seconds=remaining)))
    decision_check = None
    if 'proposal_decisions' in result:
        remaining = max(0, (limits or {}).get('seconds', 900) - (time.perf_counter() - started))
        decision_check = verify_decisions(original, analysis, scenarios, result['proposal_decisions'],
                                         seconds=remaining, _context=decision_context,
                                         include_replayed=heuristic)
    if heuristic:
        from .heuristic_check import choose_candidate
        replayed_decisions = (decision_check or {}).pop('replayed_decisions', [])
        if any(check.get('status') == 'FAIL' for check in checks):
            return {'status': 'FAIL', 'source_semantics': semantic, 'candidates': checks,
                    'proposal_decisions': decision_check, 'frontier_complete_verified': False,
                    'scope': 'An independently checked candidate contract failed; no new graph is authorized'}
        selected, selection_errors = [], []
        if 'proposal_decisions' not in result:
            selection_errors.append('Missing independently replayable proposal decisions')
        for index, (scenario, check) in enumerate(zip(scenarios, checks)):
            if scenario.get('schema') != 'regime-heuristic-frontier/v1':
                selected.append({'scenario_index': index, 'graph_id': 'retain-original',
                    'operation': 'retain_original', 'authority': 'retain_original',
                    'reason': 'No independently checked heuristic candidate set is available'})
                continue
            saved_adopt = {record.get('graph_id') for record in result.get('proposal_decisions', [])
                           if record.get('scenario_index') == index and record.get('disposition') == 'adopt'}
            allowed = {record['graph_id'] for record in replayed_decisions
                       if record.get('scenario_index') == index and record.get('disposition') == 'adopt'
                       and record['graph_id'] in saved_adopt}
            if semantic.get('status') == 'FAIL' or check.get('status') != 'PASS':
                allowed = set()
            chosen = choose_candidate(dict(analysis, allocation_scenario=scenario['allocation_contract']),
                                      scenario, allowed_ids=allowed)
            chosen.update(scenario_index=index,
                          authority='source_checked' if chosen['operation'] == 'representation_specification'
                                    else 'retain_original')
            selected.append(chosen)
        selection_complete = (decision_check is not None and decision_check.get('status') == 'PASS'
                              and all(check.get('status') == 'PASS' for check in checks))
        if selection_complete and result.get('recommendations') != selected:
            selection_errors.append('Saved recommendation differs from freshly source-checked selection')
        statuses = [check.get('status') for check in checks]
        if decision_check is not None:
            statuses.append(decision_check.get('status'))
        failed = semantic.get('status') == 'FAIL' or 'FAIL' in statuses or bool(selection_errors)
        status = 'FAIL' if failed else 'PASS' if checks and all(value == 'PASS' for value in statuses) else 'UNRESOLVED'
        return {'status': status, 'source_semantics': semantic, 'candidates': checks,
                'proposal_decisions': decision_check, 'recommendations': selected,
                'errors': selection_errors, 'frontier_complete_verified': False,
                'elapsed_seconds': time.perf_counter() - started,
                'scope': 'independent original-source and retained-candidate replay; no exhaustive search, global completeness or optimality; native numerical execution NOT_RUN'}
    statuses = [semantic.get('status')] + [c.get('status') for c in checks]
    if decision_check is not None:
        statuses.append(decision_check['status'])
    if any(c.get('checking_complete') is False for c in checks):
        statuses.append('INCOMPLETE')
    status = ('FAIL' if 'FAIL' in statuses else 'INCOMPLETE' if 'INCOMPLETE' in statuses else
              'PASS' if checks and all(s == 'PASS' for s in statuses) else 'UNRESOLVED')
    if status == 'PASS' and (analysis.get('obligations') or not analysis.get('discovery_complete') or not known):
        status = 'UNRESOLVED'
    return {'status': status, 'source_semantics': semantic, 'frontiers': checks,
            'proposal_decisions': decision_check,
            'elapsed_seconds': time.perf_counter() - started,
            'scope': 'independent original-source and finite-family replay; native numerical execution NOT_RUN'}


__all__ = ['find_regimes', 'analyze_production', 'verify_certificate', 'verify_production_result']
