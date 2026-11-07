"""Automatic conservative interfaces and globally optimal interval layouts.

The finite class is explicit: full observed-status lanes, fixed semantic
projections derived from original primitives, all admitted age intervals, plus
the original grouping as an incumbent. Dense union axes are evaluation layout,
not newly revived latent states. Costs are integer enumeration/dispatch units.
"""
from __future__ import annotations

from collections import defaultdict, deque
from copy import deepcopy
from fractions import Fraction
import hashlib
import itertools
import json
import math
import time

from .expressions import AdmissionError, evaluate, jsonable, literal, partial


LAYERS = ('admission', 'semantic', 'numerical', 'source_adapter', 'implementation', 'efficiency')
DEFAULT_ROLE = {'actor': 'agent', 'observations': [], 'commitments': [], 'discount_boundary': 'between_dates'}


def canonical(obj):
    return json.dumps(jsonable(obj), sort_keys=True, separators=(',', ':'), ensure_ascii=False)


def digest(obj):
    return hashlib.sha256(canonical(obj).encode()).hexdigest()


def size(domain):
    if isinstance(domain, list):
        return len(domain)
    return len(domain['values']) if 'values' in domain else domain['size']


def values(domain):
    return domain if isinstance(domain, list) else domain.get('values')


def solve_axis(domain):
    return isinstance(domain, list) or domain.get('solve_axis', True)


def product(domains, names, *, solve_only=False):
    return math.prod(size(domains[name]) for name in names if not solve_only or solve_axis(domains[name]))


def _domain_check(domains, label):
    if not isinstance(domains, dict):
        raise AdmissionError(f'{label} must be an axis-to-domain object')
    for name, domain in domains.items():
        if not isinstance(name, str) or not name.isidentifier():
            raise AdmissionError('Coordinate names must be Python identifiers')
        if not isinstance(domain, (dict, list)):
            raise AdmissionError(f'{label}.{name}: domain must have values or positive size')
        n = size(domain)
        if isinstance(n, bool) or not isinstance(n, int) or n < 1:
            raise AdmissionError(f'{label}.{name}: empty or invalid domain')
        vals = values(domain)
        if vals is not None:
            if len({canonical(v) for v in vals}) != n:
                raise AdmissionError(f'{label}.{name}: duplicate decoded grid values')
            for val in vals:
                if isinstance(val, float) and not math.isfinite(val):
                    raise AdmissionError('Nonfinite grid value')
        elif not domain.get('grid_id'):
            raise AdmissionError(f'{label}.{name}: symbolic grid requires a stable grid_id')


def _condition(expr, env):
    v = partial(expr, env)
    if not v.known:
        raise AdmissionError('Program admission must depend only on age, parameters and declared observed status: ' + ','.join(sorted(v.reads)))
    return bool(v.value)


def _opaque(expr):
    if isinstance(expr, dict):
        return 'opaque' in expr or any(_opaque(v) for v in expr.values())
    if isinstance(expr, list):
        return any(_opaque(v) for v in expr)
    return False


def _forcing(problem, program, env, action_names):
    """Overapproximate feasible categorical tuples; one possible value proves constancy."""
    enumerable = [a for a in action_names if values(problem['actions'][a]) is not None]
    cap = int(problem.get('limits', {}).get('action_combinations', 4096))
    if product(problem['actions'], enumerable) > cap:
        return {}, {'complete': False, 'reason': 'action enumeration cap; all action axes retained'}
    possible = []
    for assignment in itertools.product(*(values(problem['actions'][a]) for a in enumerable)):
        local = dict(env, **dict(zip(enumerable, assignment)))
        v = partial(program.get('feasible', True), local)
        if not v.known or bool(v.value):
            possible.append(assignment)
    if not possible:
        raise AdmissionError('Declared decision domain has no possibly feasible action')
    fixed = {}
    for index, name in enumerate(enumerable):
        choices = {canonical(row[index]): row[index] for row in possible}
        if len(choices) == 1:
            fixed[name] = next(iter(choices.values()))
    return fixed, {'complete': True, 'possible_tuples': len(possible), 'enumerated_axes': enumerable}


def _program_expressions(program):
    for key in ['reward', 'discount', 'feasible', 'terminal_payoff', 'taste_group', 'taste_law']:
        if key in program:
            yield key, program[key]
    for name, expr in program.get('outputs', {}).items():
        yield 'output:' + name, expr


def _finite_admission(problem, atoms, lookup):
    """Validate represented finite rows when all primitives are executable.

    Symbolic source grids/callbacks retain explicit numerical obligations; they
    are never replaced with stubs or assumed to have passed a numerical test.
    """
    checked=0; skipped=[]; max_mass=Fraction()
    limit=int(problem.get('limits',{}).get('finite_rows',100000))
    tolerance=Fraction(str(problem.get('numerical',{}).get('probability_mass_absolute','1e-12')))
    if tolerance<0:raise AdmissionError('Negative normalization budget')
    for atom in atoms:
        p=atom['_program']; sn=atom['declared_states'];an=atom['declared_actions']
        expressions=[v for _,v in _program_expressions(p)]
        for e in p.get('transitions',[])+p.get('reads',[]):
            expressions.extend([e.get('when',True),e.get('probability',e.get('weight',1))])
            expressions.extend(e.get('updates',{}).values());expressions.extend(e.get('status_updates',{}).values())
        if any(values(problem['states'][x]) is None for x in sn) or any(values(problem['actions'][x]) is None for x in an) or any(_opaque(e) for e in expressions):
            skipped.append({'atom':atom['id'],'reason':'symbolic grid or pure opaque primitive'});continue
        if any(partial(e,atom['_env']).reads & set(problem.get('external_parameters',[])) for e in expressions):
            skipped.append({'atom':atom['id'],'reason':'declared symbolic external parameter'});continue
        rows=product(problem['states'],sn)*product(problem['actions'],an)
        if checked+rows>limit:
            skipped.append({'atom':atom['id'],'reason':'finite admission row cap'});continue
        for sv in itertools.product(*(values(problem['states'][x]) for x in sn)):
            feasible_count=0
            for av in itertools.product(*(values(problem['actions'][x]) for x in an)):
                env=dict(atom['_env'],**dict(zip(sn,sv)),**dict(zip(an,av)))
                checked+=1
                if not bool(evaluate(p.get('feasible',True),env)):continue
                feasible_count+=1
                for name in ['reward','discount','terminal_payoff']:
                    if name not in p:continue
                    val=evaluate(p[name],env)
                    if not isinstance(val,(int,float,Fraction)) or isinstance(val,float) and not math.isfinite(val):
                        raise AdmissionError('Nonfinite or nonnumeric payoff/discount')
                    if name=='discount' and val<0:raise AdmissionError('Negative discount')
                if atom['age_index']==len(problem['ages'])-1:continue
                for phase in ['solve','simulation']:
                    raw=[e for e in p.get('transitions',[]) if e.get('phase','both') in [phase,'both']]
                    if not raw:continue
                    mass=Fraction()
                    for edge in raw:
                        if not bool(evaluate(edge.get('when',True),env)):continue
                        w=Fraction(evaluate(edge.get('probability',1),env))
                        if w<0:raise AdmissionError('Negative probability on a represented finite row')
                        mass+=w
                        if w==0:continue
                        status={name:evaluate(edge.get('status_updates',{}).get(name,name),env) for name in problem.get('status_axes',{})}
                        targets=[t for t in lookup[atom['age_index']+1] if literal(t['status'])==status
                                 and (not edge.get('target') or t['original_regime']==edge['target'])
                                 and (edge.get('target') or t['channel']==atom['channel'])]
                        if not targets:raise AdmissionError('Finite positive transition has no admitted target')
                        for target in targets:
                            for x in target['declared_states']:
                                expr=edge.get('updates',{}).get(x,x)
                                value=evaluate(expr,env)
                                grid=values(problem['states'][x])
                                if grid is not None and value not in [literal(v) for v in grid]:
                                    raise AdmissionError('Transition update leaves declared target grid: '+x)
                    err=abs(mass-1);max_mass=max(max_mass,err)
                    if err>tolerance:raise AdmissionError(f'Probability row mass error {err} exceeds {tolerance}')
            if feasible_count==0:raise AdmissionError('Empty feasible action row at represented state '+canonical(dict(zip(sn,sv))))
    return {'rows_checked':checked,'unexecuted_atoms':skipped,'max_mass_error':str(max_mass),
            'weights_changed':False,'mode':'exact represented rows' if max_mass==0 else 'bounded raw probability operator; no renormalization'}


def prepare(problem):
    """Derive interfaces from primitive dependency facts; never read proposed cuts."""
    if problem.get('schema_version') != 1:
        raise AdmissionError('schema_version must be 1')
    forbidden = {'reduced', 'certificate', 'winning_cuts', 'state_map', 'action_embedding', 'necessary_signature'}
    if forbidden & set(problem):
        raise AdmissionError('Finder input must be one original problem, without an answer or transformation certificate')
    ages = problem.get('ages')
    if not isinstance(ages, list) or not ages or any(type(x) is not int for x in ages) or ages != sorted(set(ages)):
        raise AdmissionError('ages must be a strictly increasing nonempty integer tick list')
    states, actions = problem.get('states', {}), problem.get('actions', {})
    _domain_check(states, 'states'); _domain_check(actions, 'actions')
    if set(states) & set(actions):
        raise AdmissionError('State and action coordinate names overlap')
    programs = problem.get('programs', [])
    if not programs or len({p.get('id') for p in programs}) != len(programs):
        raise AdmissionError('Original programs need unique nonempty ids')
    if any(not isinstance(p.get('id'), str) or not p['id'] for p in programs):
        raise AdmissionError('Invalid original program id')
    status_axes = problem.get('status_axes', {})
    if set(status_axes) & (set(states) | set(actions) | {'age'}):
        raise AdmissionError('Observed status axes must have distinct names')
    _domain_check(status_axes, 'status_axes')
    if any(values(d) is None for d in status_axes.values()):
        raise AdmissionError('Observed statuses must be explicitly finite')
    if product(status_axes, status_axes) > int(problem.get('limits', {}).get('status_lanes', 128)):
        raise AdmissionError('Observed-status product exceeds declared support limit')
    status_names = sorted(status_axes)
    statuses = [dict(zip(status_names, vals)) for vals in itertools.product(*(values(status_axes[x]) for x in status_names))]
    atoms = []
    by_id = {}
    # Each original program is a symbolic decision primitive family, not a desired regime partition.
    for ai, age in enumerate(ages):
        for si, status in enumerate(statuses):
            occupied = set()
            for program in programs:
                env = dict(problem.get('parameters', {}), age=age, **status)
                if not _condition(program.get('when', True), env):
                    continue
                channel = program.get('channel', 'main')
                if channel in occupied:
                    raise AdmissionError('Overlapping original programs in one observed-status/channel atom; provide the original economic channel or status distinction')
                occupied.add(channel)
                sn = list(program.get('declared_states', states))
                an = list(program.get('declared_actions', actions))
                if len(sn) != len(set(sn)) or not set(sn) <= set(states) or len(an) != len(set(an)) or not set(an) <= set(actions):
                    raise AdmissionError('Program declares an unknown or duplicate coordinate')
                constants = {x: values(states[x])[0] for x in sn if values(states[x]) is not None and size(states[x]) == 1}
                env.update(constants)
                fixed, forcing = _forcing(problem, program, env, an) if an else ({}, {'complete': True, 'possible_tuples': 1})
                env.update(fixed)
                primitive_reads = {}
                direct = set()
                for name, expr in _program_expressions(program):
                    val = partial(expr, env)
                    primitive_reads[name] = sorted(val.reads)
                    permitted=set(sn)|set(an)|set(problem.get('external_parameters',[]))
                    unknown_names=val.reads-permitted
                    if unknown_names:
                        raise AdmissionError('Undeclared primitive variable: '+','.join(sorted(unknown_names)))
                    bad = val.reads & (set(states) - set(sn))
                    if bad:
                        raise AdmissionError('A primitive reads a state missing from its original domain: ' + ','.join(sorted(bad)))
                    direct.update(val.reads & set(sn))
                identifier = f'a{ai}s{si}c{len(occupied)-1}'
                atom = {'id': identifier, 'age': age, 'age_index': ai, 'status': status,
                        'channel': channel, 'original_regime': program['id'],
                        'declared_states': sorted(sn), 'declared_actions': sorted(an),
                        'constant_states': constants, 'forced_actions': fixed,
                        'live_states': set(direct) - set(constants),
                        'free_actions': sorted(set(an) - set(fixed)), 'primitive_reads': primitive_reads,
                        'forcing': forcing, 'role': deepcopy(program.get('role', DEFAULT_ROLE)),
                        'routing_owner': program.get('routing_owner'), 'terminal': bool(program.get('terminal', False)),
                        'allocation': program.get('allocation', 'per_block'),
                        'source_anchor': program.get('source_anchor'),
                        '_env': env, '_program': program, '_direct': set(direct)}
                if atom['allocation'] not in ['per_block', 'global_union']:
                    raise AdmissionError('Unsupported allocation mode')
                atoms.append(atom); by_id[identifier] = atom
    if not atoms:
        raise AdmissionError('No original admitted decision atoms')
    lookup = defaultdict(list)
    for atom in atoms:
        lookup[atom['age_index']].append(atom)
    numerical_admission=_finite_admission(problem,atoms,lookup)
    edges = []
    probability_checks = []
    for atom in atoms:
        program, env = atom['_program'], atom['_env']
        raw_edges = [('transition', e) for e in program.get('transitions', [])]
        raw_edges += [('read', e) for e in program.get('reads', [])]
        for kind, edge in raw_edges:
            gate = partial(edge.get('when', True), env)
            if gate.known and not gate.value:
                continue
            weight_expr = edge.get('weight', 1) if kind == 'read' else edge.get('probability', 1)
            weight = partial(weight_expr, env)
            for expr in [edge.get('when',True),weight_expr,*edge.get('updates',{}).values(),*edge.get('status_updates',{}).values()]:
                unknown_names=partial(expr,env).reads-set(atom['declared_states'])-set(atom['declared_actions'])-set(problem.get('external_parameters',[]))
                if unknown_names:raise AdmissionError('Undeclared transition variable: '+','.join(sorted(unknown_names)))
            atom['live_states'].update((gate.reads | weight.reads) & set(atom['declared_states']))
            if weight.known:
                if not isinstance(weight.value, (int, float, Fraction)) or weight.value < 0:
                    raise AdmissionError('A declared transition/read weight is negative or nonnumeric')
                if weight.value == 0:
                    continue  # Exactly zero only. No probability tolerance is used here.
            offset = edge.get('offset', 0 if kind == 'read' else 1)
            if offset not in [0, 1] or (kind == 'transition' and offset != 1):
                raise AdmissionError('Only same-period reads and next-date transitions/reads are supported')
            ti = atom['age_index'] + offset
            if ti >= len(ages):
                continue  # Finite-horizon endpoint; terminal continuation lies outside this instance.
            target_status = {}
            for name in status_names:
                v = partial(edge.get('status_updates', {}).get(name, name), env)
                atom['live_states'].update(v.reads & set(atom['declared_states']))
                if v.known:
                    if v.value not in [literal(x) for x in values(status_axes[name])]:
                        raise AdmissionError('Transition leaves the declared observed-status domain')
                    target_status[name] = v.value
            destinations = [t for t in lookup[ti]
                            if all(literal(t['status'][k]) == v for k, v in target_status.items())
                            and (not edge.get('target') or t['original_regime'] == edge['target'])
                            and (edge.get('target') or t['channel'] == atom['channel'])]
            if not destinations:
                raise AdmissionError('Positive or possibly positive dependency has no declared target atom')
            phase = edge.get('phase', 'both')
            if phase not in ['solve', 'simulation', 'both']:
                raise AdmissionError('Unknown transition phase')
            for target in destinations:
                edges.append({'source': atom['id'], 'target': target['id'], 'kind': kind,
                              'phase': phase, 'offset': offset, 'updates': edge.get('updates', {}),
                              'probability': jsonable(weight.value) if weight.known else 'symbolic_possible_positive',
                              'weight_expression': weight_expr,
                              'source_anchor': edge.get('source_anchor', atom['source_anchor'])})
        # Check constant rows separately, without renormalization or deleting support.
        for phase in ['solve', 'simulation']:
            relevant = [e for e in program.get('transitions', []) if e.get('phase', 'both') in [phase, 'both']]
            row = []
            complete = True
            for e in relevant:
                gate = partial(e.get('when', True), env)
                if gate.known and not gate.value:
                    continue
                w = partial(e.get('probability', 1), env)
                if not gate.known or not w.known:
                    complete = False; break
                row.append(Fraction(w.value))
            if row and complete and atom['age_index'] < len(ages)-1:
                err = abs(sum(row)-1)
                tolerance = Fraction(str(problem.get('numerical', {}).get('probability_mass_absolute', '1e-12')))
                if tolerance < 0:
                    raise AdmissionError('Negative normalization budget')
                if err > tolerance:
                    raise AdmissionError(f'Probability row mass error {err} exceeds {tolerance}')
                probability_checks.append({'atom':atom['id'], 'phase':phase, 'mass_error':str(err), 'weights_changed':False})
    # Topological check for same-date reads. Future-date edges cannot form a finite-horizon cycle.
    indegree = {a['id']: 0 for a in atoms}; successors = defaultdict(set)
    for e in edges:
        if e['offset'] == 0 and e['target'] not in successors[e['source']]:
            successors[e['source']].add(e['target']); indegree[e['target']] += 1
    queue = [x for x, n in indegree.items() if n == 0]; visited = 0
    while queue:
        x = queue.pop(); visited += 1
        for y in successors[x]:
            indegree[y] -= 1
            if indegree[y] == 0: queue.append(y)
    if visited != len(atoms):
        raise AdmissionError('Same-period dependency cycle requires a fixed-point extension')
    # Least fixed point: keep every predecessor of every actual current/future consumer.
    reasons = {a['id']: {x: ['current primitive consumer'] for x in a['live_states']} for a in atoms}
    incoming = defaultdict(list)
    for edge in edges:
        incoming[edge['target']].append(edge)
    pending = deque((atom['id'], state) for atom in atoms for state in sorted(atom['live_states']))
    while pending:
        target_id, coordinate = pending.popleft()
        for edge in incoming[target_id]:
            source = by_id[edge['source']]
            expr = edge['updates'].get(coordinate, coordinate)
            val = partial(expr, source['_env'])
            bad = val.reads & (set(states) - set(source['declared_states']))
            if bad:
                raise AdmissionError('Incoming history/update reads absent source state: ' + ','.join(sorted(bad)))
            for dep in sorted(val.reads & set(source['declared_states']) - set(source['constant_states'])):
                if dep not in source['live_states']:
                    source['live_states'].add(dep)
                    pending.append((source['id'], dep))
                reason = f"{edge['phase']} {edge['kind']} into {target_id}.{coordinate}"
                if reason not in reasons[source['id']].setdefault(dep, []):
                    reasons[source['id']][dep].append(reason)
    lanes = defaultdict(list)
    for atom in atoms:
        atom['live_states'] = sorted(atom['live_states'])
        atom['retention_reasons'] = reasons[atom['id']]
        atom['dropped_states'] = sorted(set(atom['declared_states']) - set(atom['live_states']))
        atom['specialized_interfaces'] = {name: sorted(set(reads) - set(atom['constant_states']) - set(atom['forced_actions']))
                                          for name, reads in atom['primitive_reads'].items()}
        lane = canonical([atom['status'], atom['channel']])
        lanes[lane].append(atom['id'])
    for lane in lanes.values():
        lane.sort(key=lambda x: by_id[x]['age_index'])
    return {'atoms':atoms,'by_id':by_id,'edges':edges,'lanes':dict(sorted(lanes.items())),
            'probability_checks':probability_checks,
            'numerical_admission':numerical_admission,
            'support_policy':'Full declared domains including every status lane, incoming/later starts and possible solve/simulation/read dependency; no sampling or tolerance-zeroing.'}


def segment(problem, model, lane, start, stop, *, original=False):
    """A complete segment admission and cost calculation; independent of search history."""
    ids = lane[start:stop]
    atoms = [model['by_id'][x] for x in ids]
    if not atoms: return None, 'empty'
    roles = {(canonical(a['role']), a['routing_owner'], a['terminal']) for a in atoms}
    if len(roles) != 1: return None, 'actor_information_commitment_or_routing_boundary'
    if not original and not problem.get('implementation', {}).get('allow_sparse_admission', False) and any(b['age_index'] != a['age_index']+1 for a,b in zip(atoms,atoms[1:])):
        return None, 'nonconsecutive_admission'
    if len({a['allocation'] for a in atoms}) != 1: return None, 'allocation_backend_boundary'
    if not original and len(atoms) == 1:
        a = atoms[0]
        saving = set(problem.get('clarity', {}).get('endpoint_saving_actions', ['saving','savings','assets_next','assets_end']))
        earlier = [model['by_id'][x] for x in lane[:start] if not model['by_id'][x]['terminal']]
        later = [model['by_id'][x] for x in lane[stop:] if not model['by_id'][x]['terminal']]
        if not later and earlier:
            prev = earlier[-1]
            newly_fixed = saving & set(a['forced_actions']) & set(prev['free_actions'])
            if newly_fixed and a['role'] == prev['role'] and a['routing_owner']==prev['routing_owner']:
                return None, 'endpoint_only_saving_specialization'
    global_union = atoms[0]['allocation'] == 'global_union'
    if original or global_union:
        state_axes = sorted(set().union(*(set(a['declared_states']) for a in atoms)))
        action_axes = sorted(set().union(*(set(a['declared_actions']) for a in atoms)))
        if global_union:
            # One backend-wide union is allocated at every covered age regardless of chosen labels.
            state_axes = sorted(problem.get('implementation', {}).get('global_state_axes', problem['states']))
            action_axes = sorted(problem.get('implementation', {}).get('global_action_axes', problem['actions']))
    else:
        state_axes = sorted(set().union(*(set(a['live_states']) for a in atoms)))
        action_axes = sorted(set().union(*(set(a['free_actions']) for a in atoms)))
    ns = product(problem['states'],state_axes,solve_only=True)
    na = product(problem['actions'],action_axes)
    impl = problem.get('implementation', {})
    if not original and ns > impl.get('max_block_state_entries', math.inf): return None, 'state_layout_limit'
    if not original and na > impl.get('max_block_action_entries', math.inf): return None, 'action_layout_limit'
    if not original and ns*na > impl.get('max_block_entries', math.inf): return None, 'combined_layout_limit'
    members = set(ids)
    crossings = sorted({(e['source'],e['target']) for e in model['edges'] if e['source'] in members and e['target'] not in members})
    unit = impl.get('boundary_unit', 1)
    if type(unit) is not int or unit < 0: raise AdmissionError('boundary_unit must be a nonnegative integer')
    enumeration = len(ids)*ns*na
    boundary_rows = len(crossings)*ns*na
    status_values=list({canonical(a['status']):a['status'] for a in atoms}.values())
    solve_states=[x for x in state_axes if solve_axis(problem['states'][x])]
    simulation_states=[x for x in state_axes if not solve_axis(problem['states'][x])]
    return {'atoms':ids,'ages':sorted({a['age'] for a in atoms}), 'status':status_values[0] if len(status_values)==1 else None,
            'status_admission':status_values,'admission':[{'age':a['age'],'status':a['status']} for a in atoms],
            'actor_information_role':atoms[0]['role'],'routing_owner':atoms[0]['routing_owner'],
            'allocated_state_axes':solve_states,'simulation_state_axes':simulation_states,'allocated_action_axes':action_axes,
            'semantic_interfaces':{a['id']:{'states':a['live_states'],'actions':a['free_actions'],
                                          'forced_actions':jsonable(a['forced_actions']),
                                          'constant_states':jsonable(a['constant_states'])} for a in atoms},
            'allocation':atoms[0]['allocation'],'layout_semantics':'dense broadcast of fixed per-atom semantic interfaces; continuation reads project to those interfaces',
            'cost':{'enumeration':enumeration,'boundary':unit*boundary_rows,'boundary_rows':boundary_rows,
                    'boundary_edges':len(crossings),'work':enumeration+unit*boundary_rows,'blocks':1,
                    'storage':len(ids)*ns}, 'boundary_edges':[list(edge) for edge in crossings],
            'original_labels':sorted({a['original_regime'] for a in atoms})}, None


def total_cost(segments):
    keys = ['enumeration','boundary','boundary_rows','boundary_edges','work','blocks','storage']
    return {k:sum(s['cost'][k] for s in segments) for k in keys}


def original_partition(problem, model):
    groups = defaultdict(list)
    for a in model['atoms']:
        groups[(a['original_regime'],canonical(a['role']),a['routing_owner'],a['terminal'],a['allocation'])].append(a['id'])
    result = []
    for ids in groups.values():
        s,reason=segment(problem,model,ids,0,len(ids),original=True)
        if s is None: raise AdmissionError('Original grouping is not admissible: '+reason)
        result.append(s)
    return result


def original_admissibility(problem,model,original):
    reasons=[];impl=problem.get('implementation',{})
    for s in original:
        ns=product(problem['states'],s['allocated_state_axes'],solve_only=True)
        na=product(problem['actions'],s['allocated_action_axes'])
        if ns>impl.get('max_block_state_entries',math.inf) or na>impl.get('max_block_action_entries',math.inf) or ns*na>impl.get('max_block_entries',math.inf):
            reasons.append('original allocation exceeds declared new-layout limit')
    membership={x:i for i,s in enumerate(original) for x in s['atoms']}
    saving=set(problem.get('clarity',{}).get('endpoint_saving_actions',['saving','savings','assets_next','assets_end']))
    for lane in model['lanes'].values():
        alive=[model['by_id'][x] for x in lane if not model['by_id'][x]['terminal']]
        if len(alive)<2:continue
        prev,last=alive[-2:]
        if membership[prev['id']]!=membership[last['id']] and saving&set(prev['free_actions'])&set(last['forced_actions']) and prev['role']==last['role'] and prev['routing_owner']==last['routing_owner']:
            reasons.append('original grouping contains an endpoint-only saving specialization')
    return {'admissible':not reasons,'reasons':sorted(set(reasons))}


def _public_model(model):
    return {'atoms':[{k:jsonable(v) for k,v in a.items() if not k.startswith('_')} for a in model['atoms']],
            'edges':jsonable(model['edges']),'lanes':model['lanes'],
            'probability_checks':model['probability_checks'],'numerical_admission':model['numerical_admission'],'support_policy':model['support_policy']}


def reported_transitions(model,selected):
    membership={atom:s['id'] for s in selected for atom in s['atoms']}
    result=[]
    for edge in model['edges']:
        source=model['by_id'][edge['source']];target=model['by_id'][edge['target']]
        result.append({'source':membership[edge['source']],'target':membership[edge['target']],
            'source_atom':edge['source'],'target_atom':edge['target'],'kind':edge['kind'],'phase':edge['phase'],
            'projection':target['live_states'],
            'state_update_expressions':{x:edge['updates'].get(x,x) for x in target['live_states']},
            'source_bindings':jsonable({'age':source['age'],**source['status'],**source['constant_states'],**source['forced_actions']}),
            'target_status':target['status'],'target_constant_states':jsonable(target['constant_states'])})
    return result


def _diagnostics(problem, model, chosen):
    result=[]
    atom_to_block={a:i for i,s in enumerate(chosen) for a in s['atoms']}
    for lane in model['lanes'].values():
        for left,right in zip(lane,lane[1:]):
            a,b=model['by_id'][left],model['by_id'][right]
            before=a['original_regime']!=b['original_regime']
            after=atom_to_block[left]!=atom_to_block[right]
            if not before and after:
                kind='MISSING_SPLIT'
            elif before and not after:
                kind='REDUNDANT_SPLIT'
            elif before or after:
                kind='RETAIN'
            else:
                continue
            lost_states=sorted(set(a['live_states'])-set(b['live_states']))
            lost_actions=sorted(set(a['free_actions'])-set(b['free_actions']))
            role_change=a['role']!=b['role'] or a['routing_owner']!=b['routing_owner'] or a['terminal']!=b['terminal']
            reason='required economic/information/routing boundary' if role_change else ('allocated domain/cost change' if after else 'same admissible interface family; age-dependent coefficients remain schedules')
            result.append({'kind':kind,'at_age':b['age'],'status':b['status'],'witness':{'left':left,'right':right,'states_expiring':lost_states,'actions_forced':lost_actions,'retained_histories':b['retention_reasons']},
                           'source_anchor':b['source_anchor'],'reason':reason,
                           'proposed_change':'split' if kind=='MISSING_SPLIT' else ('merge' if kind=='REDUNDANT_SPLIT' else 'retain'),
                           'validation_status':'structural_projection_and_complete_segment_check'})
    for a in model['atoms']:
        if a['dropped_states'] or a['forced_actions']:
            result.append({'kind':'RETAIN','at_age':a['age'],'status':a['status'],
                           'witness':{'dropped_states':a['dropped_states'],'forced_actions':jsonable(a['forced_actions']),'retained_states':a['live_states']},
                           'source_anchor':a['source_anchor'],'reason':'Automatically derived per-atom interface; named labels are not discovery inputs',
                           'proposed_change':'specialize function signatures and project boundary state','validation_status':'structural_exact_relative_to_primitive_contract'})
    for item in problem.get('source_adapter', {}).get('unsupported', []):
        result.append({'kind':'UNRESOLVED','reason':item,'source_anchor':problem.get('source_adapter',{}).get('source'),'proposed_change':'separate extension','validation_status':'not_claimed'})
    return result


def find_regimes(problem, budget=None):
    if not isinstance(problem, dict):
        from .model_adapter import adapt_model
        problem = adapt_model(problem)
    if problem.get('implementation', {}).get('compact_search', False):
        from .compact import find_compact
        return find_compact(problem, budget)
    started=time.perf_counter()
    report={'schema_version':1,'status':'INVALID','problem_sha256':digest(problem),
            'layers':{name:{'status':'NOT_RUN','checks':0,'notes':[]} for name in LAYERS},'diagnostics':[]}
    active='admission'
    try:
        model=prepare(problem)
        report['layers']['admission']={'status':'PASS','checks':len(model['atoms']),'notes':['Full declared support; no model import or native source execution.']}
        active='semantic'
        original=original_partition(problem,model)
        original_ok=original_admissibility(problem,model,original)
        report['layers']['semantic']={'status':'PASS','checks':len(model['atoms'])+len(model['edges']),
            'notes':['Fixed semantic projections are derived by primitive partial evaluation, constant-action enumeration, and global predecessor liveness. No approximate equivalence-class discovery.',
                     'Opaque primitives are trusted pure total functions of the declared inputs; source-adapter provenance is separate.',
                     'External immutable subject/report tags remain external and associated with original subjects.']}
        pc=model['probability_checks']
        finite=model['numerical_admission']
        report['layers']['numerical']={'status':'PASS' if not finite['unexecuted_atoms'] else 'NOT_RUN','checks':finite['rows_checked'],
            'notes':['Finite admission record lists checked and unexecuted atoms. Symbolic/action/state-dependent source rows not executed retain the original primitive normalization obligation. Weights are never renormalized and positive edges are never tolerance-zeroed.']}
        report['layers']['source_adapter']={'status':'PASS' if problem.get('source_adapter',{}).get('status')=='SUPPORTED_STATIC' else 'NOT_RUN',
            'checks':len(problem.get('source_adapter',{}).get('sources',[])),
            'notes':['Static source extraction only; native pylcm/JAX execution and backend code generation are not claimed.']}
        active='implementation'
        report['layers']['implementation']={'status':'PASS','checks':len(model['atoms']),
            'notes':['Declared actor/order/routing ownership and allocation restrictions enforced. Dense layouts do not revive omitted latent states. Native backend broadcast/continuation-read bridge remains separate.']}
        active='efficiency'
        budget=budget or {}
        max_segments=int(budget.get('candidate_segments',problem.get('limits',{}).get('candidate_segments',200000)))
        max_seconds=float(budget.get('seconds',problem.get('limits',{}).get('seconds',60)))
        if max_segments<0 or max_seconds<0: raise AdmissionError('Invalid negative search budget')
        all_candidates={}; rejected=[]; count=0; complete=True
        for key,lane in model['lanes'].items():
            entries=[]
            for i in range(len(lane)):
                for j in range(i+1,len(lane)+1):
                    if count>=max_segments or time.perf_counter()-started>=max_seconds:
                        complete=False;break
                    count+=1
                    s,why=segment(problem,model,lane,i,j)
                    if s is not None: entries.append({'start':i,'stop':j,'segment':s})
                    else: rejected.append({'lane':key,'start':i,'stop':j,'reason':why})
                if not complete:break
            all_candidates[key]=entries
            if not complete:break
        chosen=original
        potentials={}; refined=[]; chosen_indices={}; feasible=True
        if complete:
            for key,lane in model['lanes'].items():
                n=len(lane); best=[None]*(n+1); best[n]=(0,0,())
                successors=defaultdict(list)
                for entry in all_candidates[key]: successors[entry['start']].append(entry)
                choice={}
                for i in range(n-1,-1,-1):
                    options=[]
                    for e in successors[i]:
                        tail=best[e['stop']]
                        if tail is not None:
                            c=e['segment']['cost']
                            options.append(((c['work']+tail[0],1+tail[1],(e['stop'],)+tail[2]),e))
                    if options: best[i],choice[i]=min(options,key=lambda v:v[0])
                potentials[key]=[list(v[:2]) if v is not None else None for v in best]
                if best[0] is None: feasible=False;continue
                i=0; cuts=[]
                while i<n:
                    e=choice[i]; refined.append(e['segment']);cuts.append([i,e['stop']]);i=e['stop']
                chosen_indices[key]=cuts
            if feasible:
                oc,rc=total_cost(original),total_cost(refined)
                # Refined branch wins exact score ties: it emits the automatically specialized interfaces.
                if not original_ok['admissible'] or (rc['work'],rc['blocks']) <= (oc['work'],oc['blocks']):chosen=refined
            elif not original_ok['admissible']:
                raise AdmissionError('No admissible partition exists within the complete declared family')
        branch='refined' if chosen is refined else 'original'
        if branch=='original':chosen_indices={}
        for i,s in enumerate(chosen): s['id']=f'R{i:03d}'
        cost=total_cost(chosen)
        public=_public_model(model)
        report.update({'status':'OPTIMAL' if complete else 'INCOMPLETE','analysis':public,
            'original':{'cost':total_cost(original),'regimes':original,**original_ok},
            'recommended':{'cost':cost,'regimes':chosen,'external_report_tags':problem.get('external_report_tags',[]),
               'incumbent_admissible':branch=='refined' or original_ok['admissible'],
               'transitions':reported_transitions(model,chosen)},
            'search':{'complete':complete,'segments_evaluated':count,'admitted_segments':sum(map(len,all_candidates.values())),
                'optimality_scope':'Original grouping OR all admitted interval layouts independently in fully enumerated observed-status/channel lanes, with globally fixed conservative semantic interfaces. No coalescence of distinct status lanes or search for arbitrary sufficient statistics.',
                'objective':['allocated enumeration plus weighted source-row boundary operations','number of structural blocks','lexicographically earliest cut tuple; refined branch on exact branch-score tie'],
                'lower_bound':[cost['work'],cost['blocks']] if complete else [0,0],
                'gap':0 if complete else None,'incomplete_reason':None if complete else 'candidate or wall-time budget exhausted; original feasible incumbent retained',
                'counterfactual_segments':[{**{k:e[k] for k in ['start','stop']},'lane':key,'cost':e['segment']['cost'],
                    'allocated_state_axes':e['segment']['allocated_state_axes'],'allocated_action_axes':e['segment']['allocated_action_axes']} for key,entries in all_candidates.items() for e in entries],
                'rejected_segments':rejected},
            'certificate':{'version':1,'problem_sha256':digest(problem),'analysis_sha256':digest(public),
                'complete':complete,'branch':branch,'potentials':potentials,'chosen_segments':chosen_indices,
                'score':[cost['work'],cost['blocks']],'family_sha256':digest({key:[{'start':e['start'],'stop':e['stop'],'cost':e['segment']['cost']} for e in entries] for key,entries in all_candidates.items()})}})
        report['diagnostics']=_diagnostics(problem,model,chosen)
        if not complete: report['diagnostics'].append({'kind':'UNRESOLVED','reason':'Search budget exhausted','witness':{'segments_evaluated':count},'proposed_change':'rerun with larger explicit budget','validation_status':'incumbent_only'})
        report['layers']['efficiency']={'status':'PASS' if complete else 'UNRESOLVED','checks':count,'notes':['Cost is an explicit enumeration/dispatch proxy, not measured hardware speed or peak live memory.']}
    except (AdmissionError, KeyError, TypeError, ValueError) as exc:
        report['layers'][active]={'status':'FAIL','checks':1,'notes':[str(exc)]}
        report['diagnostics'].append({'kind':'UNRESOLVED','reason':str(exc),'proposed_change':'correct unsupported or malformed original specification','validation_status':'admission_failed'})
        report['error']=str(exc)
    report['measured_search_seconds']=time.perf_counter()-started
    return report
