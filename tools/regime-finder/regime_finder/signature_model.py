"""Exact signature assignment producer; independent reconstruction is elsewhere."""
from collections import defaultdict
import json
import hashlib
import math


def encode(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def fingerprint(value):
    return hashlib.sha256(encode(value).encode()).hexdigest()


def normalize_model(model):
    """Canonical integer-polynomial serialization, preserving multiplicities."""
    def terms(values):
        coefficients=defaultdict(int)
        for name,value in values:coefficients[name]+=value
        return sorted((name,value) for name,value in coefficients.items() if value)
    result=dict(model, schema='regime-signature-binary-model/v2')
    result['constraints']=sorted((dict(row,terms=terms(row['terms'])) for row in model['constraints']),key=encode)
    result['objectives']={name:dict(obj,terms=terms(obj['terms'])) for name,obj in model['objectives'].items()}
    return result


def cell_signature(cell):
    return (tuple(sorted(cell['states'].items())),tuple(sorted(cell['actions'].items())),
            tuple(sorted(set(cell['characteristics']))))


def merged(left,right):
    axes=[]
    for a,b in zip(left[:2],right[:2]):
        values=dict(a)
        for name,domain in b:
            if name in values and values[name]!=domain:return None
            values[name]=domain
        axes.append(tuple(sorted(values.items())))
    return (*axes,tuple(sorted(set(left[2])|set(right[2]))))


def build(instance, *, check=lambda:None, max_signatures=4096, max_variables=1000000):
    """Construct the exact binary family, never assert a solved frontier."""
    cells=sorted(instance['cells'],key=lambda cell:cell['id'])
    weights=instance['clarity_weights']
    if (not cells or len({c['id'] for c in cells})!=len(cells)
            or len({c['label'] for c in cells})!=len(cells)
            or len({c['context'] for c in cells})!=1
            or instance['allocation']!='per_block'
            or weights['block']<=weights['dispatch']):
        raise ValueError('Singleton signature dominance guards are not established')
    if any(type(v) is not int or v<0 for v in [*weights.values(),instance['boundary_unit']]):
        raise ValueError('Invalid integer cost rules')
    for cell in cells:
        for kind in ('states','actions'):
            for domain in cell[kind].values():
                size=instance['domains'][domain]['size']
                if type(size) is not int or size<=0:raise ValueError('Invalid captured domain size')
    bases={cell_signature(c) for c in cells}
    closure=set(bases)
    if len(closure)>max_signatures:raise RuntimeError('signature union budget')
    queue=list(closure)
    while queue:
        current=queue.pop()
        for base in bases:
            check()
            value=merged(current,base)
            if value is not None and value not in closure:
                if len(closure)>=max_signatures:raise RuntimeError('signature union budget')
                closure.add(value);queue.append(value)
    bins=sorted(closure)
    eligible=[[k for k,bin in enumerate(bins) if merged(cell_signature(cell),bin)==bin] for cell in cells]
    pairs=sorted(set(map(tuple,instance['pairs'])))
    index={cell['id']:i for i,cell in enumerate(cells)}
    if any(source not in index for source,_ in pairs):raise ValueError('Unknown source in component pair')
    bound=sum(map(len,eligible))+len(bins)+len(pairs)*len(bins)
    if bound>max_variables:raise RuntimeError('signature model variable budget')
    rows=[]
    for states,actions,_ in bins:
        rows.append(tuple(math.prod(instance['domains'][d]['size'] for _,d in axes
                         if instance['domains'][d].get('solve_axis') is not False) for axes in (states,actions)))
    variables=set();constraints=[]
    objectives={name:{'constant':0,'terms':[]} for name in ('work','storage','clarity')}
    def rule(coefficients,relation,rhs):
        terms=[(name,value) for name,value in coefficients if value]
        variables.update(name for name,_ in terms)
        constraints.append({'terms':terms,'relation':relation,'rhs':rhs})
    def cost(kind,name,value):
        if value:objectives[kind]['terms'].append((name,value))
    for i,choices in enumerate(eligible):
        check();rule([(f'x_{i}_{k}',1) for k in choices],'==',1)
        for k in choices:
            rule([(f'x_{i}_{k}',1),(f'y_{k}',-1)],'<=',0)
            cost('work',f'x_{i}_{k}',math.prod(rows[k]));cost('storage',f'x_{i}_{k}',rows[k][0])
    objectives['clarity']['constant']=weights['dispatch']*len(cells)
    for k,bin in enumerate(bins):
        check();providers=[i for i,choices in enumerate(eligible) if k in choices]
        rule([(f'y_{k}',1),*[(f'x_{i}_{k}',-1) for i in providers]],'<=',0)
        cost('clarity',f'y_{k}',weights['block']-weights['dispatch']+weights['characteristic']*len(bin[2]))
        for kind,tokens in enumerate(bin):
            for token in tokens:
                rule([*[(f'x_{i}_{k}',1) for i in providers if token in cell_signature(cells[i])[kind]],
                      (f'y_{k}',-1)],'>=',0)
    for edge,(source,target) in enumerate(pairs):
        check();i=index[source]
        if source==target:continue
        for k in eligible[i]:
            amount=instance['boundary_unit']*math.prod(rows[k])
            if target not in index or k not in eligible[index[target]]:
                cost('work',f'x_{i}_{k}',amount);continue
            j=index[target];z=f'z_{edge}_{k}'
            rule([(z,1),(f'x_{i}_{k}',-1),(f'x_{j}_{k}',1)],'>=',0)
            rule([(z,1),(f'x_{i}_{k}',-1)],'<=',0)
            rule([(z,1),(f'x_{j}_{k}',1)],'<=',1)
            cost('work',z,amount)
    return normalize_model({'schema':'regime-signature-binary-model/v1','instance_sha256':fingerprint(instance),
        'cell_order':[c['id'] for c in cells],'bins':bins,'variables':sorted(variables),
        'constraints':constraints,'objectives':objectives,'pareto_complete_representation':True,
        'full_plan_set_preserved':False,'frontier_complete_verified':False,
        'guard':'one cell per original label; one protected context; per_block; block > dispatch'})
