"""The constructed pylcm graph is the input contract, with symbolic runtime slots."""

SCOPE = 'pylcm-structural-graph/v1'
SIGNATURE = 'pylcm-declared-inputs/v1'
ALLOCATION = {'kind': 'per_block', 'status': 'declared',
              'identity': 'pylcm-regime-domains/v1', 'boundary_unit': 1,
              'meaning': 'Separate value domains for each declared regime; structural entries, not peak device bytes'}

IID_TYPES = {'lcm.processes.' + name for name in (
    'NormalIIDProcess', 'UniformIIDProcess', 'LogNormalIIDProcess', 'NormalMixtureIIDProcess')}
IID_TYPES.add('_lcm.processes.joint_normal_lowering._JointNormalCoordinate')
AR1_TYPES = {'lcm.processes.' + name for name in (
    'TauchenAR1Process', 'RouwenhorstAR1Process', 'TauchenNormalMixtureAR1Process')}
AR1_TYPES.add('_lcm.processes.ar1_on_nodes._TauchenOnNodesProcess')


def unconditioned_iid(value):
    return (isinstance(value, dict) and value.get('type') in IID_TYPES
            and value.get('fields', {}).get('state_conditioned') is None)


def native_graph(original):
    source = original.get('graph_snapshot', original)
    return source.get('schema') == 'constructed-pylcm-model-graph-1'


def identity_state(metadata):
    """Canonical fixed_transition advertises the state it carries, through wrappers."""
    if not isinstance(metadata, dict):
        return None
    bound = metadata.get('bound_self', {})
    if bound.get('type') != '_lcm.identity_transition._IdentityTransition':
        return None
    name = bound.get('fields', {}).get('_state_name')
    return name if isinstance(name, str) and metadata.get('arguments') == [name] else None


def _missing_signature(callback):
    return any(callback.get(key) for key in ('unresolved_specialization_signature',
        'unresolved_graph_signature', 'unresolved_input_signature'))


def apply_scope(original, declarations):
    if not native_graph(original):
        return declarations
    declarations['analysis_scope'] = SCOPE
    declarations['allocation'] = dict(ALLOCATION)
    source = original.get('graph_snapshot', original)
    regimes = source.get('user_regimes', source.get('regimes', {}))
    for cell in declarations['cells']:
        declaration = regimes[cell['regime']]
        fields = declaration.get('fields', declaration)
        cell['decision_objective'] = fields.get('pareto_objective', source.get('pareto_objective'))
    declarations['parameter_scope'] = 'All runtime values admitted by this constructed Model; no calibration-specific pruning'
    for callback in declarations['callbacks'].values():
        if _missing_signature(callback):
            # Missing effective inputs cannot justify a smaller local domain.
            callback.pop('dependency_contract', None)
            callback.update(status='unresolved', expression=None)
            continue
        # pylcm's dependency DAG consumes the effective declared signature. An
        # opaque body needs no numerical interpretation to retain every input.
        callback['dependency_contract'] = SIGNATURE
        state = identity_state(callback.get('original', {}))
        if state is not None:
            callback.update(arguments=[state], reads=[state], expression=state, status='proved')
        elif callback.get('status') != 'proved':
            callback['reads'] = list(callback['arguments'])
            callback['status'] = 'structural'
            callback['expression'] = None
    outside = {'allocation_contract', 'role_timing_contract', 'callback_interpretation',
               'variadic_dependency', 'capture_identity', 'structural_storage_semantics',
               'composite_semantic_transport', 'composite_reachable_domain',
               'structural_domain_evaluation', 'policy_query_evaluation', 'production_provenance'}
    canonical_selectors = {'lcm.collective.ParetoObjective'}
    known_paths = {cb.get('source_path') for cb in declarations['callbacks'].values()
                   if isinstance(cb.get('original'), dict) and cb['original'].get('type') in canonical_selectors}
    for callback in declarations['callbacks'].values():
        if _missing_signature(callback):
            continue
        raw = callback.get('original', {})
        if not isinstance(raw,dict):continue
        if (raw.get('type') in IID_TYPES | AR1_TYPES
                and raw.get('fields', {}).get('state_conditioned') is None):
            path = callback.get('source_path', '')
            if '/states/' in path:
                axis = path.split('/states/',1)[1].split('/',1)[0].replace('~1', '/').replace('~0', '~')
                reads = [] if raw['type'] in IID_TYPES else [axis]
                callback.update(arguments=reads, reads=reads, expression=None, status='structural')
                known_paths.add(path)
    declarations['obligations'] = [item for item in declarations['obligations']
        if not (item.get('kind') == 'declaration_callback' and item.get('source_path') in known_paths)]
    declarations['additional_semantic_obligations'] = [item for item in declarations['obligations']
                                                     if item.get('kind') in outside]
    declarations['obligations'] = [item for item in declarations['obligations']
                                   if item.get('kind') not in outside]
    declarations['coverage']['unresolved_obligations'] = len(declarations['obligations'])
    return declarations
