"""Independent singleton-signature formulation and proof kernel.

This file imports no production module.
The binary model is a scalable representation, not a claim of solved optimization.
The caller must first establish source semantics and protected-context identity.
"""
from __future__ import annotations
from collections import Counter, defaultdict
import hashlib
import json
import math
from .records import LinearObjective

METRICS = ('work', 'storage', 'clarity')


def canonical_model(model):
    def polynomial(terms):
        values=Counter()
        for variable,coefficient in terms:values[variable]+=coefficient
        return sorted((name,value) for name,value in values.items() if value!=0)
    rows=[dict(row,terms=polynomial(row['terms'])) for row in model['constraints']]
    objectives={metric:dict(value,terms=polynomial(value['terms'])) for metric,value in model['objectives'].items()}
    return dict(model,schema='regime-signature-binary-model/v2',constraints=sorted(rows,key=canonical),objectives=objectives)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def validate(instance):
    cells = instance['cells']
    ids = [c['id'] for c in cells]
    if not cells or len(ids) != len(set(ids)):
        raise ValueError('Nonempty component with unique cells required')
    if len({c['label'] for c in cells}) != len(cells):
        raise ValueError('Singleton original labels required; use general recurrence otherwise')
    if len({c['context'] for c in cells}) != 1:
        raise ValueError('Protected contexts must coincide')
    if instance.get('allocation') != 'per_block':
        raise ValueError('This formulation requires per_block allocation')
    weights = instance['clarity_weights']
    if any(type(weights[k]) is not int or weights[k] < 0 for k in ('block', 'dispatch', 'characteristic')):
        raise ValueError('Nonnegative integer clarity weights required')
    if weights['block'] <= weights['dispatch']:
        raise ValueError('Strict block-minus-dispatch dominance guard required')
    b = instance['boundary_unit']
    if type(b) is not int or b < 0:
        raise ValueError('Nonnegative integer boundary cost required')
    for c in cells:
        for kind in ('states', 'actions'):
            for domain in c[kind].values():
                d = instance['domains'][domain]
                if type(d['size']) is not int or d['size'] <= 0:
                    raise ValueError('Positive captured cardinalities required')
        if len(c['characteristics']) != len(set(c['characteristics'])):
            raise ValueError('Canonical characteristic identifiers must be unique')
    if any(s not in set(ids) for s, _ in instance['pairs']):
        raise ValueError('Only source-owned pairs belong in a component instance')


def signature(cell):
    return (tuple(sorted(cell['states'].items())),
            tuple(sorted(cell['actions'].items())),
            tuple(sorted(set(cell['characteristics']))))


def join(left, right):
    axes = []
    for a, b in zip(left[:2], right[:2]):
        merged = dict(a)
        for name, domain in b:
            if name in merged and merged[name] != domain:
                return None
            merged[name] = domain
        axes.append(tuple(sorted(merged.items())))
    return (*axes, tuple(sorted(set(left[2]) | set(right[2]))))


def union_signatures(items):
    result = ((), (), ())
    for item in items:
        result = join(result, item)
        if result is None:
            return None
    return result


def union_closure(instance, *, max_signatures=4096):
    """Exactly close compatible signatures under union, or explicitly exhaust."""
    validate(instance)
    seen = {signature(c) for c in instance['cells']}
    if len(seen) > max_signatures:
        raise RuntimeError('INCOMPLETE: initial signature budget')
    todo = list(sorted(seen))
    while todo:
        left = todo.pop()
        for right in tuple(seen):
            candidate = join(left, right)
            if candidate is not None and candidate not in seen:
                if len(seen) >= max_signatures:
                    raise RuntimeError('INCOMPLETE: union-signature budget')
                seen.add(candidate)
                todo.append(candidate)
    return sorted(seen)


def rows(instance, sig):
    return tuple(math.prod(instance['domains'][domain]['size']
                           for _, domain in axes
                           if instance['domains'][domain].get('solve_axis') is not False)
                 for axes in sig[:2])


def pareto(vectors):
    values = set(vectors)
    return sorted(v for v in values if not any(
        u != v and all(a <= b for a, b in zip(u, v)) for u in values))


def literal_partition_cost(instance, blocks):
    """Independent literal block cost, using only original metadata pairs."""
    validate(instance)
    by_id = {c['id']: c for c in instance['cells']}
    if Counter(c for block in blocks for c in block) != Counter(by_id.keys()):
        raise ValueError('Each original cell must occur exactly once')
    total = [0, 0, 0]
    w = instance['clarity_weights']
    pairs = set(map(tuple, instance['pairs']))
    for block in blocks:
        members = set(block)
        if not members:
            raise ValueError('Empty block')
        sig = union_signatures(signature(by_id[c]) for c in block)
        if sig is None:
            return None
        ns, na = rows(instance, sig)
        crossing = sum(s in members and t not in members for s, t in pairs)
        total[0] += (len(block) + instance['boundary_unit'] * crossing) * ns * na
        total[1] += len(block) * ns
        total[2] += w['block'] + w['dispatch'] * (len(block) - 1) + w['characteristic'] * len(sig[2])
    return tuple(total)








def binary_model(instance):
    """Exact 0-1 formulation. Export to a proof-capable integer/PB backend.

    This constructs a Pareto-complete reduced family by strict-dominance elimination.
    Producing this object alone NEVER sets frontier_complete_verified=true.
    """
    validate(instance)
    cells = sorted(instance['cells'], key=lambda c: c['id'])
    bins = union_closure(instance)
    index = {c['id']: i for i, c in enumerate(cells)}
    choices = {i: [k for k, sig in enumerate(bins) if join(signature(c), sig) == sig]
               for i, c in enumerate(cells)}
    variables, constraints = set(), []
    objectives: dict[str, LinearObjective] = {metric: {'constant': 0, 'terms': []} for metric in METRICS}
    x = lambda i, k: f'x_{i}_{k}'
    y = lambda k: f'y_{k}'
    def constraint(terms, relation, rhs):
        terms = [(v, a) for v, a in terms if a]
        variables.update(v for v, _ in terms)
        constraints.append({'terms': terms, 'relation': relation, 'rhs': rhs})
    def term(metric, var, coefficient):
        if coefficient:
            objectives[metric]['terms'].append((var, coefficient))
    for i in range(len(cells)):
        constraint([(x(i, k), 1) for k in choices[i]], '==', 1)
        for k in choices[i]:
            constraint([(x(i, k), 1), (y(k), -1)], '<=', 0)
            ns, na = rows(instance, bins[k])
            term('work', x(i, k), ns * na)
            term('storage', x(i, k), ns)
    w, b = instance['clarity_weights'], instance['boundary_unit']
    objectives['clarity']['constant'] = w['dispatch'] * len(cells)
    for k, sig in enumerate(bins):
        eligible = [i for i in range(len(cells)) if k in choices[i]]
        constraint([(y(k), 1), *[(x(i, k), -1) for i in eligible]], '<=', 0)
        term('clarity', y(k), w['block'] - w['dispatch'] + w['characteristic'] * len(sig[2]))
        # Every declared token must actually occur: no invented union axes/chars.
        tokens = [(kind, token) for kind, items in enumerate(sig) for token in items]
        for kind, token in tokens:
            providers = [i for i in eligible if token in signature(cells[i])[kind]]
            constraint([*[(x(i, k), 1) for i in providers], (y(k), -1)], '>=', 0)
    for edge, (source, target) in enumerate(sorted(set(map(tuple, instance['pairs'])))):
        i = index[source]
        if source == target:
            continue
        for k in choices[i]:
            coefficient = b * math.prod(rows(instance, bins[k]))
            if target not in index or k not in choices[index[target]]:
                term('work', x(i, k), coefficient)
                continue
            j, z = index[target], f'z_{edge}_{k}'
            # z = x_source * (1 - x_target), even when b == 0.
            constraint([(z, 1), (x(i, k), -1), (x(j, k), 1)], '>=', 0)
            constraint([(z, 1), (x(i, k), -1)], '<=', 0)
            constraint([(z, 1), (x(j, k), 1)], '<=', 1)
            term('work', z, coefficient)
    return {'schema': 'regime-signature-binary-model/v1', 'instance_sha256': digest(instance),
            'cell_order': [c['id'] for c in cells], 'bins': bins,
            'variables': sorted(variables), 'constraints': constraints, 'objectives': objectives,
            'pareto_complete_representation': True, 'full_plan_set_preserved': False,
            'frontier_complete_verified': False,
            'guard': 'one cell per original label; one protected context; per_block; block > dispatch'}


def verify_binary_assignment(model, assignment):
    """Check integer constraints and objectives; no producer cost routine."""
    if set(assignment) != set(model['variables']) or any(type(v) is not int or v not in (0, 1) for v in assignment.values()):
        raise ValueError('Full binary assignment required')
    for row in model['constraints']:
        value = sum(a * assignment[v] for v, a in row['terms'])
        good = {'==': value == row['rhs'], '<=': value <= row['rhs'], '>=': value >= row['rhs']}[row['relation']]
        if not good:
            raise ValueError('Infeasible binary witness')
    return tuple(obj['constant'] + sum(a * assignment[v] for v, a in obj['terms'])
                 for obj in (model['objectives'][m] for m in METRICS))


def impossible(row, partial):
    """Sound interval contradiction over remaining independent binary values."""
    low = high = 0
    coefficients = defaultdict(int)
    for variable, coefficient in row['terms']:
        coefficients[variable] += coefficient
    for variable, coefficient in coefficients.items():
        if variable in partial:
            low += coefficient * partial[variable]
            high += coefficient * partial[variable]
        else:
            low += min(0, coefficient)
            high += max(0, coefficient)
    if row['relation'] == '<=':
        return low > row['rhs']
    if row['relation'] == '>=':
        return high < row['rhs']
    if row['relation'] == '==':
        return high < row['rhs'] or low > row['rhs']
    raise ValueError('Unknown relation')


def check_unsat_tree(model, clauses, proof, *, max_visits=1000000, check=lambda:None):
    """Independent, deliberately small query-proof kernel.

    clauses is a conjunction of disjunctions of exact integer inequalities.
    A branch covers both values of an unassigned binary variable. A leaf names
    one model constraint or query clause that is impossible on that subcube.
    No unproved solver status, floating LP bound, or omitted child is accepted.
    This reference tree can be large; proof DAG memoization is receiver work.
    """
    variables = set(model['variables'])
    visits = 0
    def visit(node, partial):
        check()
        nonlocal visits
        visits += 1
        if visits > max_visits:
            raise RuntimeError('UNRESOLVED: proof verification budget')
        if node['kind'] == 'split':
            v = node['variable']
            if v not in variables or v in partial:
                raise ValueError('Invalid split variable')
            visit(node['zero'], {**partial, v: 0})
            visit(node['one'], {**partial, v: 1})
        elif node['kind'] == 'constraint':
            at = node['index']
            if type(at) is not int or not 0 <= at < len(model['constraints']) or not impossible(model['constraints'][at], partial):
                raise ValueError('Unproved constraint contradiction')
        elif node['kind'] == 'clause':
            at = node['index']
            if type(at) is not int or not 0 <= at < len(clauses) or not all(impossible(row, partial) for row in clauses[at]):
                raise ValueError('Unproved clause contradiction')
        else:
            raise ValueError('Unknown proof node')
    visit(proof, {})
    return {'status': 'PASS', 'visited_nodes': visits, 'claim': 'specified integer query is infeasible'}


def frontier_cover_clauses(model, frontier):
    """For every claimed p require C < p in at least one coordinate.

    Infeasibility means every feasible cost is weakly dominated by a reported
    feasible p. Feasible reported vectors + antichain + this cover proof suffice
    for the full Pareto frontier, including unsupported weighted optima.
    """
    return [[{'terms': model['objectives'][metric]['terms'], 'relation': '<=',
              'rhs': point[j] - 1 - model['objectives'][metric]['constant']}
             for j, metric in enumerate(METRICS)] for point in frontier]


def check_frontier_certificate(instance, certificate):
    model = binary_model(instance)
    if certificate['model_sha256'] != digest(model):
        raise ValueError('Certificate model identity mismatch')
    vectors = [verify_binary_assignment(model, assignment) for assignment in certificate['assignments']]
    if len(vectors) != len(set(vectors)) or sorted(vectors) != pareto(vectors):
        raise ValueError('Claimed frontier is not a unique antichain')
    check_unsat_tree(model, frontier_cover_clauses(model, vectors), certificate['coverage_proof'])
    return {'status': 'PASS', 'frontier_complete_verified': True, 'vectors': sorted(vectors)}
