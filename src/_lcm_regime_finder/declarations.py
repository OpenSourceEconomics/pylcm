"""Normalize original declaration metadata without numerical model-row execution.

Inputs are captured public declarations/specifications, independent of Python
builder syntax. Every record retains an original JSON pointer and identity.
Unknown capture data creates an obligation, never a fictitious domain or role.
"""
from __future__ import annotations

import ast
import copy
from fractions import Fraction
import hashlib
import json
import math
import operator

from .callable_identity import primitive_operation, resolve_binding, resolve_callable


class DeclarationError(ValueError):
    pass


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def pointer(base, key):
    return base + "/" + str(key).replace("~", "~0").replace("/", "~1")


def fields(value):
    return value.get("fields", {}) if isinstance(value, dict) and value.get("kind") in {"dataclass", "object"} else value


def short(value):
    label = value.get("type", "") if isinstance(value, dict) else ""
    return label.rsplit(".", 1)[-1] if isinstance(label, str) else ""


def plain(value):
    if isinstance(value, list): return [plain(x) for x in value]
    if not isinstance(value, dict): return value
    if value.get("kind") == "array": return plain(value["values"]) if "values" in value else value
    if "fraction" in value: return str(Fraction(value["fraction"]))
    if value.get("kind") == "categorical-class": return {k:plain(v) for k,v in value["members"].items()}
    if value.get("kind") == "mapping": return [(plain(k),plain(v)) for k,v in value.get("items",[])]
    return {k:plain(v) for k,v in fields(value).items()} if value.get("kind") in {"dataclass","object"} else value


def _items(value):
    if isinstance(value,dict) and value.get("kind")=="mapping": return value.get("items",[])
    return list(value.items()) if isinstance(value,dict) else list(value or [])


def _select(value, phase, period, age):
    if short(value)=="Phased": return _select(fields(value).get("solve" if phase=="solve" else "simulate"),phase,period,age)
    if short(value).endswith("SpecializedFunction"):
        for row in value.get("resolved_by_period",[]):
            if row.get("period")==period: return _select(row["value"],phase,period,age)
    if short(value)=="_PeriodDispatch":
        f=fields(value);indices=plain(f.get("case_by_period",[]));cases=f.get("cases",[])
        if period<len(indices) and isinstance(indices[period],int) and 0<=indices[period]<len(cases):
            return _select(cases[indices[period]],phase,period,age)
    return value


def _unresolved_effective_callable(value):
    """A factory declaration does not expose the returned economic signature."""
    if isinstance(value, list):
        return any(_unresolved_effective_callable(item) for item in value)
    if not isinstance(value, dict):
        return False
    if value.get('kind') in {'callable', 'primitive', 'callable-reference'}:
        return False
    if short(value) in {'AgeSpecializedFunction', 'PeriodSpecializedFunction'}:
        return True
    children = fields(value) if value.get('kind') in {'dataclass', 'object'} else value
    return any(_unresolved_effective_callable(item) for item in children.values())


class _DateMetadata(list):
    """Run-local caches for immutable original date/selector declarations."""
    def __init__(self, values):
        super().__init__(values)
        self.rationals = tuple(Fraction(str(plain(v))) for v in values)
        self.selector_cache = {}
        self.law_cache = {}


def _graph_signature_omitted(value):
    if isinstance(value, list): return any(_graph_signature_omitted(v) for v in value)
    if not isinstance(value, dict): return False
    if value.get('kind') in {'callable','primitive','callable-reference'}:
        return 'Original graph checkpoint omits optional callback expansion' in value.get('obligations',[])
    children = fields(value) if value.get('kind') in {'object','dataclass'} else value
    return (any(_graph_signature_omitted(v) for v in children.values())
            or _graph_signature_omitted(value.get('callable')))


def _unresolved_input_signature(value, records=None):
    """Inspect declaration slots, not helper bodies, for missing effective inputs.

    A finite advertised signature remains the Model DAG input contract even
    when the raw implementation is variadic. A physical variadic signature or
    an unsuccessful signature inspection does not establish an empty input set.
    Legacy captures without parameter kinds retain their explicit finite
    argument-list contract unless they advertise unresolved variadic inputs.
    """
    if isinstance(value, list):
        return any(_unresolved_input_signature(item, records) for item in value)
    if not isinstance(value, dict):
        return False
    records = value.get('callable_records', records)
    if value.get('kind') == 'callable-reference':
        value = resolve_callable(value, records)
        if value.get('kind') not in {'callable', 'primitive'}:
            return True
    if value.get('kind') in {'object', 'dataclass'} and 'callable' in value:
        primary = value['callable']
        if not isinstance(primary, dict) or primary.get('kind') not in {'callable', 'primitive', 'callable-reference'}:
            return True
        # Fields are captured bindings, not additional Model DAG entry points.
        return _unresolved_input_signature(primary, records)
    callable_envelope = (value.get('kind') in {'callable', 'primitive'}
                         or ('kind' not in value and 'arguments' in value
                             and ('source' in value or 'bindings' in value)))
    if callable_envelope:
        arguments = value.get('arguments')
        if (not isinstance(arguments, list) or not all(isinstance(name, str) for name in arguments)
                or len(arguments) != len(set(arguments))):
            return True
        kinds = value.get('parameter_kinds')
        if kinds is not None:
            finite = {'POSITIONAL_ONLY', 'POSITIONAL_OR_KEYWORD', 'KEYWORD_ONLY'}
            return (not isinstance(kinds, dict) or set(kinds) != set(arguments)
                    or any(kind not in finite for kind in kinds.values()))
        return bool(value.get('variadic', False))
    children = fields(value) if value.get('kind') in {'object', 'dataclass'} else value
    return (any(_unresolved_input_signature(item, records) for item in children.values())
            or _unresolved_input_signature(value.get('callable'), records))


def selector_periods(selector, times, coordinate_kind):
    if not isinstance(times, _DateMetadata):
        return _selector_periods(selector, times, coordinate_kind)
    key = (id(selector), coordinate_kind)
    cached = times.selector_cache.get(key)
    if cached is None:
        cached = (selector, frozenset(_selector_periods(selector, times, coordinate_kind)))
        times.selector_cache[key] = cached
    return cached[1]


def _selector_periods(selector, times, coordinate_kind):
    """Resolve the public date-selector vocabulary using metadata alone."""
    typ=short(selector);f=fields(selector) if isinstance(selector,dict) else {}
    values=times.rationals if isinstance(times,_DateMetadata) else [Fraction(str(plain(v))) for v in times]
    if typ in {'AgeRange','PeriodRange'}:
        if (typ=='PeriodRange')!=(coordinate_kind=='period'):
            raise DeclarationError('Date selector and model clock disagree')
        lo=plain(f.get('start'));hi=plain(f.get('exclusive_stop'))
        return {i for i,v in enumerate(values) if (lo is None or v>=Fraction(str(lo))) and (hi is None or v<Fraction(str(hi)))}
    if typ=='Periods':
        if coordinate_kind!='period':raise DeclarationError('Periods selector requires a period clock')
        selector=plain(f.get('values',[]))
    elif typ=='range':
        start,stop,step=(plain(f.get(k,d)) for k,d in [('start',0),('stop',None),('step',1)])
        if stop is None or not isinstance(step,int) or step==0:raise DeclarationError('Uncaptured range selector')
        # Test membership arithmetically; never expand a potentially huge range.
        return {i for i,v in enumerate(values) if v.denominator==1 and
                ((start<=v<stop) if step>0 else (stop<v<=start)) and (int(v)-start)%step==0}
    elif isinstance(selector,dict) and selector.get('kind') in {'object','dataclass'}:
        raise DeclarationError('Unresolved date selector '+typ)
    selected=selector if isinstance(selector,(list,tuple)) else [selector]
    wanted={Fraction(str(plain(v))) for v in selected}
    if coordinate_kind=='period' and any(v.denominator!=1 for v in wanted):raise DeclarationError('Period coordinates must be integers')
    absent=wanted-set(values)
    if absent:raise DeclarationError('Exact date selector names a coordinate outside the supplied horizon')
    return {i for i,v in enumerate(values) if v in wanted}


def schedule_law(value,phase,period,times,coordinate_kind):
    if not isinstance(times,_DateMetadata):
        return _schedule_law(value,phase,period,times,coordinate_kind)
    key=(id(value),phase,period,coordinate_kind)
    if key not in times.law_cache:
        times.law_cache[key]=(value,_schedule_law(value,phase,period,times,coordinate_kind))
    return times.law_cache[key][1]


def _schedule_law(value,phase,period,times,coordinate_kind):
    """Select declared laws; do not invoke routing callbacks or factories."""
    if short(value)=='Phased':return schedule_law(fields(value).get('solve' if phase=='solve' else 'simulate'),phase,period,times,coordinate_kind)
    if short(value) not in {'ByAge','ByPeriod'}:return _select(value,phase,period,times[period])
    if (short(value)=='ByPeriod')!=(coordinate_kind=='period'):raise DeclarationError('Schedule and model clock disagree')
    f=fields(value)
    if not any(k in f for k in ('cases','until','default')):raise DeclarationError('Old schedule capture omitted its cases/default/until; supply current declaration metadata or its effective graph')
    until=fields(f.get('until')) if f.get('until') is not None else None
    if until:
        stop=Fraction(str(plain(until['stop_age_exclusive'])));start=plain(until.get('start_age_inclusive'))
        vals=times.rationals if isinstance(times,_DateMetadata) else [Fraction(str(plain(v))) for v in times]
        if stop not in vals or (start is not None and Fraction(str(start)) not in vals):raise DeclarationError('Until bounds must be exact supplied coordinates')
        selected=[i for i,v in enumerate(vals) if v<stop and (start is None or v>=Fraction(str(start)))]
        if period not in selected:return None
        return schedule_law(until['then' if period==selected[-1] else 'law'],phase,period,times,coordinate_kind)
    matched=[]
    for selector,law in _items(f.get('cases',[])):
        if period in selector_periods(selector,times,coordinate_kind):matched.append(law)
    if len(matched)>1:raise DeclarationError('Overlapping original schedule selectors')
    selected=matched[0] if matched else f.get('default')
    return schedule_law(selected,phase,period,times,coordinate_kind) if selected is not None else None


def _transition_gates(data,*names):
    from .current_api_contract import gate_maps
    try:return gate_maps(data,*names)
    except ValueError as exc:raise DeclarationError(str(exc)) from exc


def declaration_graph(source,regimes,times,starts,coordinate_kind):
    """Build a finite declaration graph, stopping before all numerical setup.

    The only loops range over dates, declaration labels and actual graph edges.
    No state/action grid, transition probability or specialization is evaluated.
    """
    if 'edges' not in source:raise DeclarationError('Pre-program spec needs original edges or an explicitly captured effective graph')
    phases={};laws={};gates={};n=len(times)
    for phase in ('solve','simulation'):
        edge_source=_select(source['edges'],phase,0,times[0])
        if not isinstance(edge_source,dict) or 'kind' in edge_source:raise DeclarationError('Phase edge declaration is not a mapping')
        rows=[{} for _ in times]
        for name,edge in edge_source.items():
            if name not in regimes:raise DeclarationError('Edge source is absent from declarations')
            if short(edge)=='Transition':
                ef=fields(edge);declared=ef.get('targets');gate_map=_transition_gates(ef,'gates','arrivals','participation')
                for period in range(n-1):
                    law=schedule_law(ef.get('law'),phase,period,times,coordinate_kind)
                    named=[law] if isinstance(law,str) else list(law) if isinstance(law,dict) and 'kind' not in law else None
                    targets=([t for t,sel in (declared or {}).items() if period in selector_periods(sel,times,coordinate_kind)] if declared is not None else named)
                    if targets is None:
                        if law is None:targets=[]
                        else:raise DeclarationError('Vector/deterministic transition requires explicit target selectors')
                    if named is not None and set(targets)!=set(named):raise DeclarationError('Named transition law disagrees with explicit scheduled targets')
                    if law is None and len(targets)>1:raise DeclarationError('Multiple targets have no selected original law')
                    rows[period][name]=list(targets);laws[(phase,period,name)]=law;gates[(phase,period,name)]=gate_map
            elif isinstance(edge,dict) and 'kind' not in edge:
                for period in range(n-1):
                    targets=[t for t,sel in edge.items() if period in selector_periods(sel,times,coordinate_kind)]
                    if len(targets)>1:raise DeclarationError('Multiple plain edges require an explicit original Transition law')
                    rows[period][name]=targets;laws[(phase,period,name)]=None
            else:raise DeclarationError('Unsupported edge declaration; supply its captured effective graph')
            for row in rows:
                if any(target not in regimes for target in row.get(name,[])):raise DeclarationError('Edge target is absent from declarations')
        phases[phase]=rows
    pos={str(Fraction(str(v))):i for i,v in enumerate(times)}
    pending=[(pos[str(Fraction(str(plain(date))))],name) for date,name in starts]
    admitted=set()
    while pending:
        period,name=pending.pop()
        if (period,name) in admitted:continue
        if name not in regimes:raise DeclarationError('Initial or referenced regime is absent')
        admitted.add((period,name))
        # A current policy read creates demand for an owner, not a physical
        # visit. The separate query registry below closes those demands at the
        # reader's date and phase. Only initial nodes, scheduled transitions
        # and explicitly settling fallback routes admit physical cells here.
        if period>=n-1:continue
        for phase,rows in phases.items():
            for target in rows[period].get(name,[]):
                pending.append((period+1,target))
                gate=gates.get((phase,period,name),{}).get(target)
                selected_gate=_select(gate,phase,period+1,times[period+1])
                gate_fields=fields(selected_gate) or {}
                # Gate references price other owners at the landing date;
                # only a route's fallback declares a physical settlement.
                for route in gate_fields.get('legs',gate_fields.get('routes',{})).values():
                    selected_route=_select(route,phase,period+1,times[period+1])
                    fallback=_select((fields(selected_route) or {}).get('fallback'),phase,period+1,times[period+1])
                    if fallback is not None:
                        # The supplied engine contract makes this particular
                        # legacy leg payload a settlement as well as a value
                        # read. Other current-read types do not inherit it.
                        if short(fallback)!='ProjectedRegimeValue':
                            raise DeclarationError('Physical fallback settlement contract unavailable for '+short(fallback))
                        pending.append((period+1,fields(fallback).get('regime')))
    return [[times[p],name] for p,name in sorted(admitted)],phases,laws,gates


def _literal(value, budget=None, depth=0):
    """Render a finite scalar/table without expanding a shared data DAG forever.

    The budget counts the emitted tree, not just distinct captured objects. A
    compact binding can describe an exponentially large repeated list; capture
    completeness is not permission to materialize that list as an expression.
    """
    budget={'nodes':8192,'bytes':65536} if budget is None else budget
    budget['nodes']-=1
    if budget['nodes']<0 or depth>64:
        raise DeclarationError('Captured literal exceeds the scalar proof budget')
    def emit(text):
        budget['bytes']-=len(text.encode('utf-8'))
        if budget['bytes']<0:raise DeclarationError('Captured literal exceeds the scalar proof budget')
        return text
    child=lambda item:_literal(item,budget,depth+1)
    # Normalize one node at a time. Calling plain() on the whole value here
    # would duplicate shared children before the literal budget can observe it.
    if isinstance(value,dict):
        kind=value.get('kind')
        if kind=='array' and 'values' in value:return child(value['values'])
        if 'fraction' in value:value=str(Fraction(value['fraction']))
        elif kind=='categorical-class':value=value.get('members',{})
        elif kind=='mapping':value=value.get('items',[])
        elif kind in {'dataclass','object'}:value=fields(value)
    if isinstance(value,(str,bool,int,float)) or value is None:return emit(repr(value))
    if isinstance(value,(list,tuple)) and len(value)<=8192:
        opening,closing=('(',')') if isinstance(value,tuple) else ('[',']')
        suffix=',' if isinstance(value,tuple) and len(value)==1 else ''
        emit(opening+closing+suffix+', '*max(0,len(value)-1))
        return opening+', '.join(child(x) for x in value)+suffix+closing
    if isinstance(value,dict) and not any(k in value for k in ('kind','source','fields')) and len(value)<=8192:
        emit('{'+ '}'+', '*max(0,len(value)-1))
        return '{'+', '.join(emit(repr(k)+': ')+child(v) for k,v in value.items())+'}'
    raise DeclarationError('Captured value is not a supported immutable scalar/table')


class ScalarSource:
    """Small identity-bound scalar proof language, with ordered local bindings."""
    def __init__(self,metadata,depth=0,callable_records=None,binding_memo=None,literal_memo=None):
        self.metadata=metadata;self.depth=depth
        self.callable_records=callable_records or metadata.get('callable_records',{})
        self.binding_memo={} if binding_memo is None else binding_memo
        self.literal_memo={} if literal_memo is None else literal_memo
        self.unresolved_calls=[]
        self.arguments=list(metadata.get("arguments",[]))
        self.env={name:name for name in self.arguments}
        # A keyword of functools.partial remains an overridable argument in
        # its effective signature. Only eliminated positional formals are
        # immutable here; a helper call applies effective defaults below.
        closed_partial={name:value for name,value in metadata.get("partial_bindings",{}).items()
                        if name not in self.arguments}
        for name,value in {**metadata.get("bindings",{}),**closed_partial}.items():
            try:self.env[name]=self.binding_literal(value)
            except DeclarationError:pass

    def binding_literal(self,value):
        key=value.get('capture_ref') if isinstance(value,dict) and value.get('kind')=='callable-reference' else None
        if key in self.literal_memo:return self.literal_memo[key]
        try:result=_literal(resolve_binding(value,self.callable_records,memo=self.binding_memo))
        except (ValueError,TypeError,RecursionError) as error:
            raise DeclarationError('Captured binding lacks a bounded literal interpretation: '+str(error)) from error
        if key is not None:self.literal_memo[key]=result
        return result

    def expression(self):
        if self.depth>6:raise DeclarationError("Helper proof depth exceeded")
        if self.metadata.get('capture_status','COMPLETE')!='COMPLETE':
            raise DeclarationError('Callable capture is incomplete or changed during capture')
        tree=ast.parse(self.metadata.get("source",""))
        fn=next((n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))),None)
        if fn is None or isinstance(fn,ast.AsyncFunctionDef):raise DeclarationError("No synchronous source function")
        if fn.decorator_list and self.metadata.get('capture_contract') not in {'inspection-only-frozen-bindings-1','inspection-only-frozen-bindings-2'}:
            raise DeclarationError('Legacy decorated source is not evidence of the callable body that actually runs')
        original=[a.arg for a in fn.args.posonlyargs+fn.args.args+fn.args.kwonlyargs]
        if set(self.arguments)-set(original):raise DeclarationError('Evaluated signature has arguments absent from captured source; wrapper interpretation is unresolved')
        if any(a not in self.env for a in original):raise DeclarationError("Evaluated/source signature or captured receiver is unresolved")
        for stmt in fn.body:
            if isinstance(stmt,ast.Expr) and isinstance(stmt.value,ast.Constant) and isinstance(stmt.value.value,str):continue
            if isinstance(stmt,(ast.Assign,ast.AnnAssign)):
                targets=stmt.targets if isinstance(stmt,ast.Assign) else [stmt.target]
                if any(not isinstance(t,ast.Name) for t in targets):raise DeclarationError("Mutation is outside scalar proof language")
                if stmt.value is None:continue
                result=self.node(stmt.value)
                for target in targets:self.env[target.id]=copy.deepcopy(result)
            elif isinstance(stmt,ast.Delete):
                for target in stmt.targets:
                    if isinstance(target,ast.Name):self.env.pop(target.id,None)
            elif isinstance(stmt,ast.Return):return self.node(stmt.value)
            else:raise DeclarationError("Control flow is outside scalar proof language")
        raise DeclarationError("Source has no return")

    def node(self,node):
        if isinstance(node,ast.Constant):return repr(node.value)
        if isinstance(node,ast.Name):
            if node.id not in self.env:raise DeclarationError("Unresolved captured/global name "+node.id)
            return copy.deepcopy(self.env[node.id])
        if isinstance(node,ast.Attribute):
            parent=self.node(node.value)
            if isinstance(parent,str):return '('+parent+')['+repr(node.attr)+']'
            raise DeclarationError("Dynamic attribute")
        if isinstance(node,(ast.List,ast.Tuple)):
            parts=[self.node(x) for x in node.elts]
            if not all(isinstance(x,str) for x in parts):raise DeclarationError("Nonliteral tuple component")
            if isinstance(node,ast.Tuple):return '('+', '.join(parts)+(',' if len(parts)==1 else '')+')'
            return '['+', '.join(parts)+']'
        if isinstance(node,ast.Subscript):
            left,right=self.node(node.value),self.node(node.slice)
            if isinstance(left,str) and isinstance(right,str):return '('+left+')['+right+']'
            raise DeclarationError("Dynamic subscript")
        if isinstance(node,ast.IfExp):return {"if":self.node(node.test),"then":self.node(node.body),"else":self.node(node.orelse)}
        if isinstance(node,ast.BoolOp):
            if not all(self.boolean_node(x) for x in node.values):raise DeclarationError('Value-returning Python Boolean expression lacks Boolean operand proofs')
            return {"and" if isinstance(node.op,ast.And) else "or":[self.node(x) for x in node.values]}
        if isinstance(node,ast.BinOp) and isinstance(node.op,(ast.BitAnd,ast.BitOr)):
            if not self.boolean_node(node.left) or not self.boolean_node(node.right):raise DeclarationError('Integer bitwise operation is not a Boolean primitive')
            return {"and" if isinstance(node.op,ast.BitAnd) else "or":[self.node(node.left),self.node(node.right)]}
        if isinstance(node,ast.UnaryOp):
            val=self.node(node.operand)
            if isinstance(node.op,ast.Not):return {"not":val}
            if isinstance(node.op,ast.Invert):raise DeclarationError('Bitwise inversion requires an original array-type proof; no logical-not substitution')
            if isinstance(val,str) and isinstance(node.op,(ast.UAdd,ast.USub)):return '('+('+' if isinstance(node.op,ast.UAdd) else '-')+val+')'
        if isinstance(node,ast.BinOp):
            symbols={ast.Add:'+',ast.Sub:'-',ast.Mult:'*',ast.Div:'/',ast.FloorDiv:'//',ast.Mod:'%',ast.Pow:'**'}
            if type(node.op) in symbols:
                left,right=self.node(node.left),self.node(node.right)
                if isinstance(left,str) and isinstance(right,str):return '('+left+' '+symbols[type(node.op)]+' '+right+')'
                return {"opaque":digest(ast.dump(node)),"reads":[],"arguments":[left,right],"source_operator":symbols[type(node.op)]}
        if isinstance(node,ast.Compare):
            symbols={ast.Lt:'<',ast.LtE:'<=',ast.Gt:'>',ast.GtE:'>=',ast.Eq:'==',ast.NotEq:'!=',ast.In:'in',ast.NotIn:'not in'}  # codespell:ignore
            values=[self.node(node.left)]+[self.node(x) for x in node.comparators]
            if all(isinstance(x,str) for x in values):
                return '('+values[0]+' '+ ' '.join(symbols[type(op)]+' '+v for op,v in zip(node.ops,values[1:]))+')'
        if isinstance(node,ast.Call):
            op=primitive_operation(self.metadata,node.func,callable_records=self.callable_records)
            if op=='select' and len(node.args)==3 and not node.keywords:
                return dict(zip(('if','then','else'),[self.node(x) for x in node.args]))
            if op in {'logical_and','logical_or'} and len(node.args)==2 and not node.keywords:
                return {'and' if op=='logical_and' else 'or':[self.node(x) for x in node.args]}
            target=self.metadata.get('call_targets',{}).get(ast.unparse(node.func),{})
            if target.get('kind')=='callable-reference':
                resolved=self.callable_records.get(target.get('capture_ref'),{})
                if not resolved or resolved.get('identity_sha256')!=target.get('identity_sha256'):
                    raise DeclarationError('Shared callable reference is absent or its captured identity changed')
                target=resolved
            if target.get('kind')=='callable':
                if target.get('variadic'):raise DeclarationError('Variadic helper needs an explicit binding proof')
                helper=ScalarSource(target,self.depth+1,self.callable_records,self.binding_memo,self.literal_memo)
                names=target.get('arguments',[])
                kinds=target.get('parameter_kinds',{})
                if ('partial_args' in target or 'partial_keywords' in target) and not kinds:
                    raise DeclarationError('Partial helper lacks captured effective parameter kinds')
                positional=[name for name in names if not kinds or kinds.get(name) in {'POSITIONAL_ONLY','POSITIONAL_OR_KEYWORD'}]
                if len(node.args)>len(positional) or any(isinstance(a,ast.Starred) for a in node.args) or any(kw.arg is None for kw in node.keywords):raise DeclarationError('Unsupported helper argument binding')
                actual={name:self.binding_literal(value) for name,value in target.get('defaults',{}).items()}
                actual.update({name:self.node(arg) for name,arg in zip(positional,node.args)})
                supplied=set(positional[:len(node.args)])
                for kw in node.keywords:
                    if kw.arg not in names or kw.arg in supplied or kinds.get(kw.arg)=='POSITIONAL_ONLY':raise DeclarationError('Ambiguous helper keyword binding')
                    actual[kw.arg]=self.node(kw.value);supplied.add(kw.arg)
                if any(name not in actual for name in names):raise DeclarationError('Missing required helper argument')
                helper.env.update(actual)
                try:
                    if not target.get('source'):raise DeclarationError('Captured helper source unavailable')
                    result=helper.expression()
                except DeclarationError as error:
                    if target.get('capture_status','COMPLETE')!='COMPLETE' or not target.get('identity_sha256'):
                        raise
                    # An unproved helper body still receives only its bound
                    # actual inputs, not every formal of the enclosing caller.
                    self.unresolved_calls.append('Opaque captured helper '+ast.unparse(node.func)+': '+str(error))
                    return {'opaque':digest({'target':target['identity_sha256'],'call':ast.dump(node)}),
                            'reads':[], 'arguments':list(helper.env.values())+list(actual.values())}
                self.unresolved_calls.extend(helper.unresolved_calls)
                return result
            raise DeclarationError("Call target lacks a proved identity-bound interpretation: "+ast.unparse(node.func))
        raise DeclarationError("Unsupported scalar source node "+type(node).__name__)

    def boolean_node(self,node):
        if isinstance(node,ast.Compare):return True
        if isinstance(node,ast.Constant):return isinstance(node.value,bool)
        if isinstance(node,ast.UnaryOp) and isinstance(node.op,ast.Not):return True
        if isinstance(node,ast.BoolOp):return all(self.boolean_node(x) for x in node.values)
        if isinstance(node,ast.BinOp) and isinstance(node.op,(ast.BitAnd,ast.BitOr)):
            return self.boolean_node(node.left) and self.boolean_node(node.right)
        if isinstance(node,ast.Call):
            op=primitive_operation(self.metadata,node.func,callable_records=self.callable_records)
            return op in {'logical_and','logical_or'} or (op=='select' and len(node.args)==3 and self.boolean_node(node.args[1]) and self.boolean_node(node.args[2]))
        return False


def expression_names(expr):
    if isinstance(expr,str):
        try:return {n.id for n in ast.walk(ast.parse(expr,mode='eval')) if isinstance(n,ast.Name)}
        except SyntaxError:return set()
    if isinstance(expr,list):return set().union(*(expression_names(x) for x in expr))
    if isinstance(expr,dict):return set(expr.get('reads',[]))|set().union(*(expression_names(v) for k,v in expr.items() if k not in {'opaque','source_operator'}))
    return set()


def metadata_scalar(text,env):
    """Evaluate one categorical construction characteristic, never model rows."""
    binary={ast.Add:operator.add,ast.Sub:operator.sub,ast.Mult:operator.mul,ast.Div:operator.truediv,
            ast.FloorDiv:operator.floordiv,ast.Mod:operator.mod,ast.Pow:operator.pow}
    compare={ast.Eq:operator.eq,ast.NotEq:operator.ne,ast.Lt:operator.lt,ast.LtE:operator.le,ast.Gt:operator.gt,ast.GtE:operator.ge}
    def visit(n):
        if isinstance(n,ast.Constant):return n.value
        if isinstance(n,ast.Name):return env[n.id]
        if isinstance(n,(ast.List,ast.Tuple)):return [visit(x) for x in n.elts]
        if isinstance(n,ast.Dict):return {visit(k):visit(v) for k,v in zip(n.keys,n.values)}
        if isinstance(n,ast.Subscript):return visit(n.value)[visit(n.slice)]
        if isinstance(n,ast.BinOp) and type(n.op) in binary:return binary[type(n.op)](visit(n.left),visit(n.right))
        if isinstance(n,ast.UnaryOp) and isinstance(n.op,(ast.UAdd,ast.USub)):return visit(n.operand)*(1 if isinstance(n.op,ast.UAdd) else -1)
        if isinstance(n,ast.Compare):
            vals=[visit(n.left)]+[visit(x) for x in n.comparators]
            return all(compare[type(op)](a,b) for op,a,b in zip(n.ops,vals,vals[1:]))
        raise DeclarationError("Characteristic outside bounded metadata language")
    return visit(ast.parse(text,mode='eval').body)


class Normalizer:
    def __init__(self,original,contract):
        self.original=copy.deepcopy(original);self.contract=contract or {}
        self.input_original=original;self.source_object_paths=None
        capture_source=original.get('graph_snapshot',original)
        self.callable_records=capture_source.get('callable_records',{})
        self.binding_memo={};self.literal_memo={}
        self.domains={};self.callbacks={};self.cells=[];self.relations=[];self.characteristics={};self.obligations=[]
        self.obligation_keys=set();self.callback_keys={};self.domain_keys={}
        self.original_paths={}
        self.encoded_values={}

    def encoded(self,value):
        key=id(value)
        cached=self.encoded_values.get(key)
        if cached is None:
            payload=json.dumps(value,sort_keys=True,separators=(',',':'),default=str).encode()
            cached=(value,payload,hashlib.sha256(payload).hexdigest())
            self.encoded_values[key]=cached
        return cached

    def value_digest(self,value):
        return self.encoded(value)[2]

    def declaration_key(self,value,field,setting):
        # Preserve the exact canonical JSON identity; cache only immutable
        # original payload bytes, never conclusions or numerical values.
        payload=b'{"declaration":'+self.encoded(value)[1]+b',"'+field.encode()+b'":'+json.dumps(setting,default=str).encode()+b'}'
        return hashlib.sha256(payload).hexdigest()

    def source_path(self,value,preferred):
        """Resolve exact captured records through wrappers; synthetic records fail."""
        current=self.original
        try:
            for token in preferred.split('/')[1:]:
                token=token.replace('~1','/').replace('~0','~')
                current=current[int(token)] if isinstance(current,list) else current[token]
            if current==value:return preferred
        except (KeyError,IndexError,ValueError,TypeError):pass
        if isinstance(value,(dict,list)):
            if self.source_object_paths is None:
                self.source_object_paths={}
                def index(node,path):
                    if id(node) in self.source_object_paths:return
                    self.source_object_paths[id(node)]=path
                    entries=node.items() if isinstance(node,dict) else enumerate(node)
                    for k,v in entries:
                        if isinstance(v,(dict,list)):index(v,pointer(path,k))
                index(self.input_original,'')
            actual=self.source_object_paths.get(id(value))
            if actual is not None:
                current=self.original
                for token in actual.split('/')[1:]:
                    token=token.replace('~1','/').replace('~0','~')
                    current=current[int(token)] if isinstance(current,list) else current[token]
                if current==value:return actual
        key=self.value_digest(value)
        if key not in self.original_paths:
            found=None
            def visit(node,path):
                nonlocal found
                if found is not None:return
                if isinstance(node,type(value)) and node==value:found=path;return
                if isinstance(node,dict):
                    for k,v in node.items():visit(v,pointer(path,k))
                elif isinstance(node,list):
                    for i,v in enumerate(node):visit(v,pointer(path,i))
            visit(self.original,'')
            self.original_paths[key]=found
        return self.original_paths[key] or preferred

    def obligation(self,kind,path,detail):
        key=(kind,path,detail)
        if key not in self.obligation_keys:
            self.obligation_keys.add(key);self.obligations.append({'id':'O'+str(len(self.obligations)),'kind':kind,'source_path':path,'detail':detail,'status':'unresolved'})

    def callback(self,value,path,phase=None):
        scalar_callable=isinstance(value,dict) and value.get('kind') in {'callable','primitive','callable-reference'}
        source_key=self.value_digest(value)
        key=source_key if scalar_callable or phase is None else self.declaration_key(value,'selected_phase',phase)
        if '/states/' in path and short(value).endswith('Process') and 'IID' not in short(value):
            key=self.declaration_key(value,'persistent_state_owner',path.split('/states/',1)[1].split('/',1)[0])
        if key in self.callback_keys:return self.callback_keys[key]
        path=self.source_path(value,path)
        cid='F'+key[:24];self.callback_keys[key]=cid
        if scalar_callable:
            metadata = resolve_callable(value, self.callable_records)
            if not isinstance(metadata, dict) or metadata.get('kind') not in {'callable', 'primitive'}:
                raise DeclarationError('Original callable reference is missing or changed at '+path)
            args=list(metadata.get('arguments',[]));expr=None;status='proved';reason=None
            try:
                scalar=ScalarSource(value,callable_records=self.callable_records,
                                    binding_memo=self.binding_memo,literal_memo=self.literal_memo)
                expr=scalar.expression()
                if scalar.unresolved_calls:
                    status='opaque';reason='; '.join(scalar.unresolved_calls)
            except (DeclarationError,SyntaxError,ValueError,KeyError,TypeError,StopIteration) as exc:status='opaque';reason=str(exc)
            if expr is not None:reads=sorted(expression_names(expr)&set(args))
            else:
                # Retain every actual source input on unresolved interpretation;
                # unknown identity is never allowed to delete an argument.
                reads=sorted(args)
            if status!='proved':self.obligation('callback_interpretation',path,reason or 'Unresolved callable')
            if metadata.get('variadic'):
                status='unresolved';self.obligation('variadic_dependency',path,'Actual callback may consume undeclared keyword inputs')
            for reason in metadata.get('obligations',[]):self.obligation('capture_identity',path,reason)
            record={'arguments':args,'reads':reads,'source':metadata.get('source'), 'identity':metadata.get('identity_sha256',key),
                    'bindings':copy.deepcopy(metadata.get('bindings',{})),'expression':expr,'status':status,'source_path':path,
                    'original':copy.deepcopy(value),'provenance':{'source_path':path,'sha256':source_key}}
        else:
            child=[]
            def walk(item,p):
                if phase is not None and short(item)=='Phased':
                    side='solve' if phase=='solve' else 'simulate'
                    walk(fields(item).get(side),pointer(pointer(p,'fields'),side));return
                if isinstance(item,dict) and item.get('kind') in {'object','dataclass'} and 'callable' in item:
                    walk(item['callable'],pointer(p,'callable'));return
                if isinstance(item,dict) and item.get('kind') in {'callable','primitive','callable-reference'}:child.append(self.callback(item,p));return
                if isinstance(item,dict):
                    for k,v in item.items():walk(v,pointer(p,k))
                elif isinstance(item,list):
                    for i,v in enumerate(item):walk(v,pointer(p,i))
            walk(value,path)
            reads=sorted({a for c in child for a in self.callbacks[c]['reads']})
            if isinstance(value,dict) and value.get('type') in {'lcm.decoupled.Decoupled', 'lcm.shocks.ShockGroup'}:
                reads=sorted({a for c in child for a in self.callbacks[c]['arguments']})
            constant=isinstance(value,(int,float,bool,str)) or value is None
            record={'arguments':reads,'reads':reads,'source':None,'identity':key,'bindings':{},
                    'expression':repr(value) if constant else None,'status':'proved' if constant else 'unresolved' if any(self.callbacks[c].get('status')=='unresolved' for c in child) else 'opaque',
                    'source_path':path,'original':copy.deepcopy(value),'members':child,'declaration_type':short(value),
                    'selected_phase':phase,'provenance':{'source_path':path,'sha256':source_key}}
            if not constant and not child:
                builtin_reads={'lcm.koopmans_aggregation.LinearAggregator':['utility','CE','discount_factor'],
                               'lcm.koopmans_aggregation.CESAggregator':['utility','CE','discount_factor','intertemporal_elasticity_of_substitution'],
                               'lcm.shocks.ShockGroup':[],
                               '_lcm.certainty_equivalent.LinearExpectation':[]}
                typ=value.get('type') if isinstance(value,dict) else None
                if typ in builtin_reads:
                    record.update(arguments=builtin_reads[typ],reads=builtin_reads[typ],declaration_contract='pinned built-in callable signature')
                elif typ=='_lcm.identity_transition._IdentityTransition' and isinstance(fields(value).get('state_name'),str):
                    name=fields(value)['state_name'];record.update(arguments=[name],reads=[name],expression=name,status='proved',declaration_contract='pinned fixed_transition identity')
                else:
                    record['status']='unresolved'
                    self.obligation('declaration_callback',path,'No exposed callback inputs or scalar interpretation for '+short(value))
        if _unresolved_effective_callable(value):
            record.update(status='unresolved', expression=None,
                          unresolved_specialization_signature=True)
            self.obligation('specialized_effective_signature', path,
                'Factory inputs are not economic callable inputs; capture the already constructed per-period callable and its signature before reducing this interface')
        if _graph_signature_omitted(value):
            record.update(status='unresolved', expression=None, unresolved_graph_signature=True)
            self.obligation('graph_callback_signature',path,
                'This is a topology checkpoint, not a complete callback capture; re-export the constructed Model before reducing its interfaces')
        if _unresolved_input_signature(value, self.callable_records):
            record.update(status='unresolved', expression=None, unresolved_input_signature=True)
            self.obligation('effective_input_signature', path,
                'The captured effective signature is variadic or unavailable; retain the full local domain until its finite Model inputs are captured')
        self.callbacks[cid]=record
        return cid

    def domain(self,value,path,period=None):
        specialized='SpecializedGrid' in short(value)
        shared_program=(specialized and type(value.get('resolved_cardinality')) is int
                        and value['resolved_cardinality']>0 and bool(value.get('resolved_cardinality_source')))
        key=self.declaration_key(value,'period',period if specialized and not shared_program else None)
        if key in self.domain_keys:return self.domain_keys[key]
        path=self.source_path(value,path)
        did='D'+key[:24];self.domain_keys[key]=did
        attrs=fields(value) if isinstance(value,dict) else {};attrs=attrs if isinstance(attrs,dict) else {}
        typ=short(value);record={'kind':'categorical' if typ=='DiscreteGrid' else 'continuous','size':None,'identity':key,
                              'source_path':path,'original':copy.deepcopy(value),'provenance':{'source_path':path,'sha256':self.value_digest(value)}}
        if typ=='Phased':
            solve,sim=attrs.get('solve'),attrs.get('simulate');imputed=isinstance(solve,dict) and solve.get('kind')=='callable'
            sd=None if imputed else self.domain(solve,pointer(pointer(path,'fields'),'solve'),period)
            td=self.domain(sim,pointer(pointer(path,'fields'),'simulate'),period)
            base=self.domains[sd or td];record.update({k:copy.deepcopy(v) for k,v in base.items() if k in {'kind','size','values','codes','category_labels','structural_values','structural_codes'}})
            record.update(solve_axis=not imputed,simulation_axis=True,phase_domains={'solve':sd,'simulation':td})
            if imputed:record['solve_imputation']=self.callback(solve,pointer(pointer(path,'fields'),'solve'))
        elif typ=='ConditionalAction':
            base=self.domain(attrs.get('grid'),pointer(pointer(path,'fields'),'grid'),period)
            record.update({k:copy.deepcopy(v) for k,v in self.domains[base].items() if k in {'kind','size','values','codes','category_labels','structural_values','structural_codes'}})
            record['conditional_declaration']=copy.deepcopy(attrs)
        elif 'SpecializedGrid' in typ:
            record['specialization']={'period':None if shared_program else period,'factory':attrs.get('build'),
                'identity_scope':'original-shape-invariant-program' if shared_program else 'resolved-period',
                'resolved_at_period':value.get('resolved_at_period')}
            count=value.get('resolved_cardinality',attrs.get('n_points'))
            if isinstance(count,int) and not isinstance(count,bool) and count>0:record['size']=count
            else:self.obligation('production_grid_metadata',path,'Specialized grid lacks a supplied shape-invariant cardinality; factory was not executed')
        else:
            codes=plain(attrs.get('codes'));values=plain(attrs.get('values'));points=plain(attrs.get('points'))
            if typ=='DiscreteGrid':
                from .structural_context import reference_info
                code_ref = codes.get('structural_ref', codes) if isinstance(codes, dict) else codes
                value_ref = values.get('structural_ref', values) if isinstance(values, dict) else values
                code_info, value_info = reference_info(code_ref), reference_info(value_ref)
                for info in (code_info, value_info):
                    if info is not None and info['type'] == 'array':
                        if len(info['shape']) != 1: raise DeclarationError('Categorical array values must be one-dimensional')
                        info['length'] = info['shape'][0]
                if isinstance(codes,list):record['codes']=codes
                elif code_info is not None: record['structural_codes']=code_ref
                if isinstance(values,list):record['values']=values
                elif value_info is not None: record['structural_values']=value_ref
                elif values is None and isinstance(codes,list):record.update(values=codes,value_semantics='declared_categorical_codes')
                record['category_labels']=attrs.get('categories')
                record['size']=len(values) if isinstance(values,list) else len(codes) if isinstance(codes,list) else None
                if value_info is not None or code_info is not None:
                    record['size']=(value_info or code_info)['length']
                    if value_info is not None and code_info is not None and value_info['length'] != code_info['length']:
                        raise DeclarationError('Structural category code/value length mismatch')
                    self.obligation('structural_domain_evaluation',path,'Exact categorical payload is retained in the checked store; unsupported scalar characteristic enumeration remains unresolved')
                if isinstance(values,list) and isinstance(codes,list) and len(values)!=len(codes):raise DeclarationError('Category code/value length mismatch')
            else:
                count=plain(attrs.get('n_points',attrs.get('size')))
                if count is None and isinstance(points,list):count=len(points)
                counts=plain(attrs.get('points_per_segment'))
                if count is None and isinstance(counts,list) and all(isinstance(x,int) and not isinstance(x,bool) and x>0 for x in counts):count=sum(counts)
                if isinstance(count,int) and not isinstance(count,bool) and count>0:record['size']=count
                if typ in {'CompositeState','History','CompositeGrid'}:record['kind']='composite'
            if record['size'] is None:self.obligation('production_grid_metadata',path,'Domain cardinality not supplied; no numerical grid was materialized')
        self.domains[did]=record
        return did

    def relation(self,kind,source,targets,path,phase='both',**extra):
        record={'id':'L'+str(len(self.relations)),'kind':kind,'source':source,'targets':targets,'phase':phase,
                'callbacks':[],'state_updates':{},'joint_outputs':{},'ownership':None,'source_path':path,
                'provenance':{'source_path':path},**extra}
        self.relations.append(record);return record


def normalize_declarations(snapshot,production_contract=None):
    """Normalize an original captured graph or declaration spec; no model rows.

    A paired input may contain declaration_spec plus graph_snapshot when an old
    serialized spec omitted a graph schedule. Both identities remain in original.
    """
    if not isinstance(snapshot,dict):raise DeclarationError('A declaration/spec mapping is required before program installation')
    from .structural_context import validate_references
    validate_references(snapshot)
    if snapshot.get('schema')=='regime-declarations/v1':return copy.deepcopy(snapshot)
    from .current_api_contract import validate_original_types
    try:validate_original_types(snapshot)
    except ValueError as exc:raise DeclarationError(str(exc)) from exc
    normal=Normalizer(snapshot,production_contract)
    paired='graph_snapshot' in snapshot
    source=snapshot['graph_snapshot'] if paired else snapshot
    base='/graph_snapshot' if paired else ''
    graph=fields(source.get('graph',{})) or {};reach=fields(graph.get('reachability',source.get('reachability',{}))) or {}
    regimes=source.get('user_regimes',source.get('regimes',{}));regime_key='user_regimes' if 'user_regimes' in source else 'regimes'
    if not isinstance(regimes,dict) or not regimes:raise DeclarationError('No original regime declarations')
    global_fields={k:source.get(k,{}) for k in ('states','actions','functions','constraints','state_transitions','derived_categoricals')}
    regimes={name:{**(fields(value) or {}),**{k:{**global_fields[k],**((fields(value) or {}).get(k,{}) or {})} for k in global_fields}}
             for name,value in regimes.items()}
    for r in regimes.values():
        for k in global_fields:r[k]={name:value for name,value in r[k].items() if value is not None}
        for k in ('koopmans_aggregator','certainty_equivalent','pareto_objective'):
            if r.get(k) is None and source.get(k) is not None:r[k]=source[k]
        r['output_only_functions']=list(dict.fromkeys([*source.get('output_only_functions',[]),*r.get('output_only_functions',[])]))
    time=plain((fields(source.get('ages',{})) or {}).get('exact_values'))
    n=source.get('n_periods')
    if time is None and isinstance(n,int):time=list(range(n))
    if not isinstance(time,list) or not time:raise DeclarationError('Exact finite date/period metadata is missing')
    if len(time)>100000:raise DeclarationError('Date metadata exceeds bounded normalizer limit; input not shortened')
    time=_DateMetadata(time)
    coordinate_kind=graph.get('coordinate_kind','age' if source.get('ages') is not None else 'period')
    positions={str(Fraction(str(v))):i for i,v in enumerate(time)}
    def period_of(value):return positions[str(Fraction(str(plain(value))))]
    pairs=list(reach.get('nodes',graph.get('nodes',[])))
    for pair in reach.get('visited_nodes',[]):
        if pair not in pairs:pairs.append(pair)
    starts=graph.get('initial_nodes',source.get('initial_nodes',[]))
    if isinstance(starts,dict):
        f=fields(starts);start_pairs=[]
        slots=('by_age','by_period') if any(k in f for k in ('by_age','by_period')) else (None,)
        for slot in slots:
            for selector,names in _items(f if slot is None else f.get(slot)):
                positions_selected=selector_periods(selector,time,'period' if slot=='by_period' else coordinate_kind)
                names=[names] if isinstance(names,str) else names
                start_pairs.extend([(time[p],name) for p in sorted(positions_selected) for name in names])
        starts=start_pairs
    if not isinstance(starts,list):starts=list(starts or [])
    phase_targets={phase:(fields(reach.get('solution' if phase=='solve' else 'simulation',{})) or {}).get('targets_by_period',[])
                   for phase in ('solve','simulation')}
    declaration_laws={};declaration_gates={};graph_origin='captured_effective_graph'
    if not pairs:
        # A raw spec can supply explicit complete admissions/phase target lists.
        pairs=source.get('admissions',[])
        supplied=source.get('phase_targets',{})
        for phase in phase_targets:phase_targets[phase]=supplied.get(phase,phase_targets[phase])
        if not pairs:
            if not starts:raise DeclarationError('Original initial nodes must be supplied; no artificial start is inferred')
            pairs,phase_targets,declaration_laws,declaration_gates=declaration_graph(source,regimes,time,starts,coordinate_kind)
            graph_origin='resolved_original_declarations'
    start_set={(period_of(a),r) for a,r in starts if r in regimes}
    by_pair={};by_id={};functions_by_cell={}
    contract=production_contract or source.get('semantic_contract',{})
    for date,name in pairs:
        if name not in regimes:raise DeclarationError('Admitted regime is absent: '+str(name))
        period=period_of(date);key=(period,name)
        if key in by_pair:continue
        declaration=regimes[name];path=pointer(pointer(base,regime_key),name)
        source_fields=source.get(regime_key,{}).get(name,{})
        fp=pointer(path,'fields') if isinstance(source_fields,dict) and source_fields.get('kind') in {'dataclass','object'} else path
        cell={'id':'C'+digest([period,name])[:20],'regime':name,'period':period,
              'age':plain(date) if coordinate_kind=='age' else None,'stage':contract.get('stages',{}).get(name,declaration.get('stage')),
              'states':{},'actions':{},'functions':{},'phase_functions':{'solve':{},'simulation':{}},'consumers':[],'category_codes':{},
              'derived_categoricals':{},'derived_code_providers':{},
              'declaration_type':source_fields.get('type','original-declaration-mapping'),
              'structural_roles':{k:copy.deepcopy(declaration[k]) for k in ('liquid','outer_continuous') if declaration.get(k) is not None},
              'solver_identity':normal.value_digest(declaration['solver']) if declaration.get('solver') is not None else None,
              'role':contract.get('roles',{}).get(name,declaration.get('semantic_role')),
              'ownership':None,'terminal':True,'admissible_start':key in start_set,
              'report_tags':list(contract.get('report_tags',{}).get(name,[])),
              'source_path':path,'provenance':{'source_path':path,'sha256':normal.value_digest(source_fields),'date':plain(date),'coordinate_kind':coordinate_kind}}
        if cell['role'] is None:normal.obligation('role_timing_contract',path,'Actor/information/commitment/discount equivalence is not supplied; preserve original identity')
        for kind in ('states','actions'):
            for axis,value in declaration.get(kind,{}).items():
                cell[kind][axis]=normal.domain(value,pointer(pointer(fp,kind),axis),period)
                domain=normal.domains[cell[kind][axis]]
                if isinstance(domain.get('codes'),list) and isinstance(domain.get('values'),list):
                    cell['category_codes'][axis]={'argument':axis+'_code','values':domain['values'],'codes':domain['codes'],'domain':cell[kind][axis]}
        for output,value in declaration.get('derived_categoricals',{}).items():
            cell['derived_categoricals'][output]=normal.domain(value,pointer(pointer(fp,'derived_categoricals'),output),period)
            cell['derived_code_providers'][output+'_code']=output
        for phase in ('solve','simulation'):
            for output,value in declaration.get('functions',{}).items():
                cell['phase_functions'][phase][output]=normal.callback(_select(value,phase,period,date),pointer(pointer(fp,'functions'),output),phase)
            # Current primitives may read deterministic next-state providers.
            # Those providers use the truth/simulation variants at the decision;
            # continuation kernels below use the respective phase variants.
            for axis,value in declaration.get('state_transitions',{}).items():
                if isinstance(value,dict) and not value:continue
                output='next_'+axis
                if output in cell['phase_functions'][phase]:raise DeclarationError('Ambiguous original next-state DAG provider '+output)
                cell['phase_functions'][phase][output]=normal.callback(_select(value,phase,period,date),pointer(pointer(fp,'state_transitions'),axis),phase)
        cell['functions']=dict(cell['phase_functions']['solve'])
        # Conditional current shocks affect the integrated value even when no
        # payoff or ordinary next-state function reads their conditioning state.
        for phase in ('solve', 'simulation'):
            groups = _select(declaration.get('shocks', []), phase, period, date)
            if not isinstance(groups, list) or any(not isinstance(group, str) for group in groups):
                raise DeclarationError('Original shock-group registrations are not captured at '+fp)
            for group in groups:
                registry = source.get('shocks', {})
                if group not in registry:
                    raise DeclarationError('Original shock group is missing: '+group)
                selected = _select(registry[group], phase, period, date)
                if short(selected) != 'ShockGroup':
                    raise DeclarationError('Original shock group has no supported declaration: '+group)
                cid = normal.callback(selected, pointer(pointer(base, 'shocks'), group), phase)
                cell['consumers'].append({'kind':'shock_law', 'callback':cid, 'phase':phase, 'group':group})
        # The pinned Phased contract supplies current utility/feasibility and
        # the decision aggregator from the truth variant. Perceived variants
        # enter only continuation kernels (and solve imputations).
        for phase in ('simulation',):
            for kind,mapping in [('reward',{'utility':declaration.get('functions',{}).get('utility')}),('feasible',declaration.get('constraints',{})),
                                 ('value_constraint',declaration.get('value_constraints',{}))]:
                for output,value in mapping.items():
                    if value is not None:
                        selected=_select(value,phase,period,date)
                        consumer_kind='value_constraint' if short(selected)=='ValueDependentConstraint' else kind
                        if consumer_kind=='value_constraint' and short(selected)=='ValueDependentConstraint':selected=fields(selected).get('predicate')
                        cell['consumers'].append({'kind':consumer_kind,'callback':normal.callback(selected,pointer(pointer(fp,'functions' if kind=='reward' else 'constraints' if kind=='feasible' else 'value_constraints'),output),phase), 'phase':phase})
                        if consumer_kind=='feasible':
                            # The native current-feasibility contract uses the
                            # truth callback in BOTH solve and simulation. Keep
                            # applicability separate from provider selection.
                            cell['consumers'][-1]['decision_phases']=['solve','simulation']
            for output in declaration.get('output_only_functions',[]):
                if output in declaration.get('functions',{}):cell['consumers'].append({'kind':'reporting','callback':normal.callback(_select(declaration['functions'][output],phase,period,date),pointer(pointer(fp,'functions'),output)),'phase':phase})
            for kind in ('koopmans_aggregator','pareto_objective'):
                value=declaration.get(kind)
                if value is not None:
                    selected=_select(value,phase,period,date)
                    # Operator records are retained even when their numerical
                    # law is symbolic; DAG discount providers remain consumers.
                    cell['consumers'].append({'kind':kind,'callback':normal.callback(selected,pointer(fp,kind),phase),'phase':phase})
            if 'discount_factor' in declaration.get('functions',{}):
                cell['consumers'].append({'kind':'discount','callback':normal.callback(_select(declaration['functions']['discount_factor'],phase,period,date),pointer(pointer(fp,'functions'),'discount_factor')),'phase':phase})
        if declaration.get('certainty_equivalent') is not None:
            cell['consumers'].append({'kind':'certainty_equivalent','callback':normal.callback(_select(declaration['certainty_equivalent'],'solve',period,date),pointer(fp,'certainty_equivalent')),'phase':'solve'})
        for slot,value in cell['structural_roles'].items():
            cid=normal.callback(value,pointer(fp,slot));f=fields(value) or {};names=[]
            for key in ('state','action','resources','post_decision_state','no_adjustment'):
                entry=f.get(key)
                if isinstance(entry,str) and entry!='__outer_unchanged__':names.append(entry)
                elif short(entry)=='NetOfAdjustmentCost':names.extend(fields(entry).get(k) for k in ('before_cost','cost'))
            record=normal.callbacks[cid];names=sorted({n for n in names if isinstance(n,str)}|set(record['reads']))
            record.update(arguments=names,reads=names,status='opaque',declaration_contract='pinned typed margin names')
            cell['consumers'].append({'kind':'structural_margin','callback':cid,'phase':'solve'})
        known_slots={'states','actions','functions','constraints','state_transitions','joint_transitions','derived_categoricals',
                     'output_only_functions','koopmans_aggregator','certainty_equivalent','pareto_objective','value_constraints',
                     'same_period_refs','exits','description','solver','taste_shocks','shocks','stakeholders','liquid','outer_continuous',
                     'semantic_role','stage'}
        for slot,value in declaration.items():
            if slot not in known_slots and value is not None and value!={} and value!=[]:
                cid=normal.callback(value,pointer(fp,slot));normal.callbacks[cid]['status']='unresolved'
                cell['consumers'].append({'kind':'unresolved_declaration_slot','callback':cid,'phase':'both'})
                normal.obligation('declaration_slot',pointer(fp,slot),'Original slot has no established consumer interpretation; all local coordinates retained')
        by_pair[key]=cell['id'];by_id[cell['id']]=cell;normal.cells.append(cell);functions_by_cell[cell['id']]=declaration
        for axis,did in cell['states'].items():
            domain=normal.domains[did]
            if domain.get('solve_imputation'):
                normal.relation('imputation',cell['id'],[cell['id']],domain['source_path'],'solve',state_updates={axis:domain['solve_imputation']},produces=[axis])
        for output,cid in cell['functions'].items():
            callback=normal.callbacks[cid]
            if output in {'utility','discount_factor'} or callback['status']!='proved' or not isinstance(callback['expression'],str):continue
            args=callback['reads']
            if len(args)==1 and args[0] in cell['states']:
                state=args[0];domain=normal.domains[cell['states'][state]]
                if domain['kind']=='categorical' and isinstance(domain.get('values'),list) and len(domain['values'])<=4096:
                    try:values=[metadata_scalar(callback['expression'],{state:v}) for v in domain['values']]
                    except (ValueError,KeyError,TypeError,IndexError,ZeroDivisionError,OverflowError):continue
                    if len({digest(v) for v in values})<len(values):
                        normal.characteristics[cid]={'state':state,'domain':cell['states'][state],'callback':cid,'output':output,'map':values,
                                                     'evaluation':'bounded single-axis original characteristic','source_path':callback['source_path']}
    # Policy-query owners are declarations evaluated at a reader's current date,
    # not physical visits or dense value banks. Keep their demand registry separate
    # from admitted decision cells; their evaluation remains a local obligation.
    query_owners={}
    query_declarations={}
    def query_references(value,path,phase):
        if short(value)=='Phased':
            side='solve' if phase=='solve' else 'simulate'
            yield from query_references(fields(value).get(side),pointer(pointer(path,'fields'),side),phase)
        elif short(value) in {'ProjectedRegimeValue','ArrivalRead','ParticipationRead','ExitTarget','Counterparty'}:
            yield value,path
        elif isinstance(value,dict):
            for key,child in value.items():yield from query_references(child,pointer(path,key),phase)
        elif isinstance(value,list):
            for index,child in enumerate(value):yield from query_references(child,pointer(path,index),phase)
    def query_owner(target,period,phase):
        if target not in regimes or not 0<=period<len(time):return None
        if (period,target) in by_pair:return None
        qid='Q'+digest([period,target,phase])[:20]
        if qid in query_owners:return qid
        raw=source.get(regime_key,{}).get(target,{})
        path=pointer(pointer(base,regime_key),target)
        declaration=regimes[target]
        if target not in query_declarations:query_declarations[target]=normal.value_digest(raw)
        record={'id':qid,'regime':target,'period':period,'phase':phase,
                'coordinate':str(Fraction(str(plain(time[period])))),'coordinate_kind':coordinate_kind,
                'source_path':path,'declaration_sha256':query_declarations[target],
                'states':sorted(declaration.get('states',{})),'actions':sorted(declaration.get('actions',{})),
                'role':copy.deepcopy(contract.get('roles',{}).get(target,declaration.get('semantic_role'))),
                'stage':contract.get('stages',{}).get(target,declaration.get('stage')),
                'admission':'policy_query_only','storage':'not_asserted','dependencies':[]}
        query_owners[qid]=record
        fp=pointer(path,'fields') if raw.get('kind') in {'dataclass','object'} else path
        keys=['same_period_refs','value_constraints','exits']
        if not declaration.get('same_period_refs'):keys.append('constraints')
        for key in keys:
            for reference,reference_path in query_references(declaration.get(key,{}),pointer(fp,key),phase):
                name=fields(reference).get('regime')
                if (period,name) in by_pair:
                    destination={'cell':by_pair[(period,name)]}
                else:
                    nested=query_owner(name,period,phase)
                    destination={'query_owner':nested} if nested is not None else {'unavailable_owner':name}
                record['dependencies'].append({**destination,'source_path':normal.source_path(reference,reference_path),
                                               'reference_sha256':normal.value_digest(reference)})
        normal.obligation('policy_query_evaluation',path,
            'Declared current policy owner is available; its source-bound query factorization and evaluation/storage cost remain unproved')
        return qid
    laws=graph.get('laws',{})
    for cell in normal.cells:
        name,period=cell['regime'],cell['period'];date=time[period];declaration=regimes[name]
        fp=pointer(cell['source_path'],'fields') if source.get(regime_key,{}).get(name,{}).get('kind') in {'dataclass','object'} else cell['source_path']
        for phase,rows in phase_targets.items():
            targets=rows[period].get(name,[]) if period<len(rows) else []
            for target in targets:
                target_id=by_pair.get((period+1,target))
                if target_id is None:
                    normal.obligation('phase_target_admission',cell['source_path'],'Declared target lacks admitted destination cell: '+str(target));continue
                cell['terminal']=False
                law=declaration_laws.get((phase,period,name),_select((fields(laws.get(name,{})) or {}).get('transition'),phase,period,date))
                lp=pointer(pointer(base,'edges'),name) if graph_origin=='resolved_original_declarations' else pointer(pointer(pointer(pointer(base,'graph'),'fields'),'laws'),name)
                relation=normal.relation('transition',cell['id'],[target_id],lp,phase,
                                         callbacks=[] if law is None else [normal.callback(law,pointer(lp,'transition'))])
                joint=declaration.get('joint_transitions',{}).get(target,{})
                produced=set()
                for joint_name,value in joint.items():
                    jp=pointer(pointer(pointer(fp,'joint_transitions'),target),joint_name);selected=_select(value,phase,period,date);jf=fields(selected) or {}
                    outputs={axis:normal.callback(_select(cb,phase,period,date),pointer(pointer(jp,'outputs'),axis)) for axis,cb in jf.get('outputs',{}).items()}
                    callbacks=[normal.callback(jf[k],pointer(jp,k)) for k in ('probabilities','support') if isinstance(jf.get(k),dict) and jf[k].get('kind')=='callable']
                    normal.relation('joint',cell['id'],[target_id],jp,phase,callbacks=callbacks,joint_outputs=outputs,state_updates=outputs,
                                    joint_name=joint_name,support=copy.deepcopy(jf.get('support')),support_size=jf.get('support_size'),original=copy.deepcopy(selected),produces=sorted(outputs))
                    produced.update(outputs)
                target_axes=[axis for axis,did in by_id[target_id]['states'].items() if phase!='solve' or normal.domains[did].get('solve_axis',True)]
                for axis in target_axes:
                    if axis in produced:continue
                    value=declaration.get('state_transitions',{}).get(axis)
                    if isinstance(value,dict) and 'kind' not in value:value=value.get(target)
                    if value is not None:
                        relation['state_updates'][axis]=normal.callback(_select(value,phase,period,date),pointer(pointer(fp,'state_transitions'),axis))
                    else:
                        from .graph_scope import unconditioned_iid
                        domain_value=_select(declaration.get('states',{}).get(axis),phase,period,date)
                        domain_path=pointer(pointer(fp,'states'),axis)
                        if domain_value is None:
                            from .graph_scope import unconditioned_iid
                            arriving=regimes[target].get('states',{}).get(axis)
                            arriving=_select(arriving,phase,period+1,time[period+1])
                            if unconditioned_iid(arriving):
                                domain_value=arriving
                                raw_target=source.get(regime_key,{}).get(target,{})
                                target_path=pointer(pointer(base,regime_key),target)
                                if raw_target.get('kind') in {'dataclass','object'}:target_path=pointer(target_path,'fields')
                                domain_path=pointer(pointer(target_path,'states'),axis)
                        if short(domain_value).endswith('Process') or unconditioned_iid(domain_value):
                            cid=normal.callback(domain_value,domain_path)
                            reads=sorted(set(normal.callbacks[cid].get('reads',[])) | (set() if 'IID' in short(domain_value) or unconditioned_iid(domain_value) else {axis}))
                            normal.callbacks[cid].update(arguments=reads,reads=reads,expression=None,status='opaque',
                                                         intrinsic_process_contract=short(domain_value))
                            relation['state_updates'][axis]=cid
                        else:normal.obligation('target_state_producer',pointer(pointer(fp,'state_transitions'),axis),'No individual/joint/process producer is exposed for '+target)
                relation['produces']=sorted(set(target_axes)-produced)
        # Every projected same-period read is a distinct typed relation. Gate
        # routes retain leg ownership rather than merging equal payload names.
        def references(value,path,kind='value_read',ownership=None,phase='both',offset=0,origin=None,incoming_source=None):
            if short(value)=='Phased':
                for side,p in [('solve','solve'),('simulate','simulation')]:
                    if phase in {'both',p}:references(fields(value).get(side),pointer(pointer(path,'fields'),side),kind,ownership,p,offset,origin,incoming_source)
                return
            if short(value) in {'ProjectedRegimeValue','ArrivalRead','ParticipationRead','ExitTarget','Counterparty'}:
                f=fields(value);target=f.get('regime');tid=by_pair.get((period+offset,target))
                if tid is None:
                    # The reference can price a separately declared owner's
                    # current policy without admitting a physical transition.
                    for p in ('solve','simulation'):
                        if phase not in {'both',p}:continue
                        qid=query_owner(target,period+offset,p)
                        if qid is None:
                            normal.obligation('same_period_reference_target',path,'Referenced target has no declared current owner: '+str(target));continue
                        cb=normal.callback(value,path,p)
                        # Until the query bridge is proved, every current input
                        # is retained. Presence of metadata is not factorization.
                        normal.callbacks[cb]['status']='unresolved'
                        normal.relation('policy_query',origin or cell['id'],[],normal.source_path(value,path),p,
                                        query_target=qid,reference_kind=kind,reference_sha256=normal.value_digest(value),
                                        callbacks=[cb],produces=[],ownership=copy.deepcopy(ownership),
                                        incoming_source=incoming_source,stakeholder=f.get('stakeholder',f.get('role')),
                                        read_type=f.get('kind','Participating' if short(value)=='ExitTarget' else 'Raw' if short(value)=='Counterparty' else 'value'),
                                        read_field=f.get('field','value'),observes=copy.deepcopy(f.get('observes',[])))
                    return
                projection_value=f.get('projection',{})
                variants=[(p,_select(projection_value,p,period+offset,time[period+offset])) for p in ('solve','simulation') if phase in {'both',p}]
                for p,mapping in variants:
                    projection={axis:normal.callback(cb,pointer(pointer(path,'projection'),axis)) for axis,cb in (mapping or {}).items()}
                    extra_callbacks=[normal.callback(f['utility_adjustment'],pointer(path,'utility_adjustment'))] if f.get('utility_adjustment') is not None else []
                    normal.relation(kind,origin or cell['id'],[tid],normal.source_path(value,path),p,state_updates=projection,produces=sorted(projection),ownership=ownership,
                                    callbacks=extra_callbacks,incoming_source=incoming_source,stakeholder=f.get('stakeholder',f.get('role')),
                                    read_type=f.get('kind','Participating' if short(value)=='ExitTarget' else 'Raw' if short(value)=='Counterparty' else 'value'),
                                    read_field=f.get('field','value'),observes=f.get('observes',[]),original=copy.deepcopy(value))
                if ownership is not None:
                    owners=by_id[tid].setdefault('ownership',[])
                    if owners is None:owners=[];by_id[tid]['ownership']=owners
                    if ownership not in owners:owners.append(ownership)
                return
            if isinstance(value,dict):
                for k,v in value.items():references(v,pointer(path,k),kind,ownership,phase,offset,origin,incoming_source)
            elif isinstance(value,list):
                for i,v in enumerate(value):references(v,pointer(path,i),kind,ownership,phase,offset,origin,incoming_source)
        reference_keys=['same_period_refs','value_constraints','exits']
        if not declaration.get('same_period_refs'):reference_keys.append('constraints')
        for key in reference_keys:
            references(declaration.get(key,{}),pointer(fp,key),'exit' if key=='exits' else 'value_read')
            if key=='exits':
                for label,value in declaration.get(key,{}).items():
                    normal.relation('exit',cell['id'],[],pointer(pointer(fp,key),label),callbacks=[normal.callback(value,pointer(pointer(fp,key),label))],original=copy.deepcopy(value))
        declared_source_law=fields(laws.get(name,{})) or {}
        gate_rows=[(phase,_transition_gates(declared_source_law,'gated_edges','arrivals','participation')) for phase in ('solve','simulation')] if not declaration_gates else [(phase,declaration_gates.get((phase,period,name),{})) for phase in ('solve','simulation')]
        for gate_phase,target,gate in ((phase,target,gate) for phase,mapping in gate_rows for target,gate in mapping.items()):
            # A declaration may carry a gate for a route used only at later
            # dates. Ownership is an incidence of an admitted incoming edge,
            # not of the Cartesian product of all dates and static gate names.
            rows=phase_targets[gate_phase]
            if period>=len(rows) or target not in rows[period].get(name,[]):continue
            gate=_select(gate,gate_phase,period+1,time[period+1])
            gf=fields(gate) or {};gp=pointer(pointer(pointer(pointer(pointer(base,'graph'),'fields'),'laws'),name),'gated_edges')
            tid=by_pair.get((period+1,target))
            if tid is None:continue
            predicate_slot='predicate' if gf.get('predicate') is not None else 'gate'
            callbacks=[normal.callback(gf[predicate_slot],pointer(gp,predicate_slot))] if gf.get(predicate_slot) is not None else []
            # Gate/arrival predicates and settlement projections see the landing
            # target grid. Their authority remains the particular incoming leg.
            normal.relation('arrival' if short(gate) in {'ArrivalDecision','ParticipationCheck'} else 'gate',tid,[tid],normal.source_path(gate,gp),gate_phase,
                            callbacks=callbacks,produces=[],incoming_source=cell['id'],ownership={'source_regime':name,'target_regime':target},
                            observes=gf.get('observes',[]),original=copy.deepcopy(gate))
            references(gf.get('references',gf.get('reads',gf.get('gate_refs',{}))),pointer(gp,'references'),'value_read',
                       {'source_regime':name,'target_regime':target},phase=gate_phase,offset=1,origin=tid,incoming_source=cell['id'])
            for leg,route in gf.get('legs',gf.get('routes',{})).items():
                owner={'source_regime':name,'target_regime':target,'leg':leg}
                references((fields(route) or {}).get('fallback'),pointer(pointer(gp,'legs'),leg),'fallback',owner,phase=gate_phase,offset=1,origin=tid,incoming_source=cell['id'])
    allocation=contract.get('allocation',source.get('allocation_contract',{'status':'unknown'}))
    composites=copy.deepcopy(source.get('composite_states',{}));composite_metadata=copy.deepcopy(source.get('composite_metadata',{}))
    store_header = source.get('structural_store')
    if store_header is not None:
        manifest = store_header.get('manifest', {})
        if manifest.get('source_consistency') != 'CALLER_QUIESCENCE_CONTRACT':
            normal.obligation('structural_capture_consistency',base,'Exact endpoint transport does not prove a coherent concurrent snapshot; caller quiescence authority is unresolved')
        for detail in manifest.get('storage_obligations', []):
            normal.obligation('structural_storage_semantics',base,detail)
        for label in composite_metadata:
            wire_contract = source.get('composite_transport_contract', {})
            if wire_contract.get('source_sha256') and type(wire_contract.get('missing_code')) is int:
                from .structural_composite import check_composite
                check_composite(composite_metadata[label], missing_code=wire_contract['missing_code'])
            normal.obligation('composite_semantic_transport',pointer(pointer(base,'composite_metadata'),label),
                              'Full structural support is retained and source-bound; economic reachability and changed member/code factorization need their separate proofs')
    for label,declaration in (composites or {}).items():
        if label not in composite_metadata:
            normal.obligation('composite_reachable_domain',pointer(pointer(base,'composite_states'),label),
                              'Supply the original lowered reachable tuple domain and membership/reconstruction metadata; no Cartesian product is substituted or enumerated')
    if not isinstance(allocation,dict) or allocation.get('kind',allocation.get('mode')) not in {'per_block','global_union'}:
        allocation={'status':'unknown'};normal.obligation('allocation_contract',base,'Backend allocation contract absent; only conditional representation costs are supported')
    elif not allocation.get('identity'):
        allocation={**allocation,'identity':digest(allocation),'status':allocation.get('status','declared')}
    metadata=copy.deepcopy(contract.get('production_metadata',source.get('production_metadata',{})))
    if not metadata:
        metadata={'status':'fixture' if source.get('capture_test') else 'unverified','capture_recipe':source.get('capture_recipe'),
                  'capture_test':source.get('capture_test'),'grid_substitution_performed':False}
        normal.obligation('production_provenance',base,'Actual production parameter/grid provenance must be established separately from a constructor snapshot')
    return {'schema':'regime-declarations/v1','source_identity':digest({'original':snapshot,'contract':contract}),'production_metadata':metadata,
            'callable_records':copy.deepcopy(normal.callable_records),
            'domains':normal.domains,'callbacks':normal.callbacks,'cells':normal.cells,'relations':normal.relations,'query_owners':query_owners,
            'composite_states':composites,'composite_metadata':composite_metadata,
            'characteristics':normal.characteristics,'allocation':allocation,'obligations':normal.obligations,
            'original':copy.deepcopy(snapshot),'original_contract':copy.deepcopy(contract),'execution':{'mode':'metadata_only','numerical_model_rows':0,'solver_compilation':False,
                                                        'native_model_execution':'NOT_RUN','specialization_factories_executed':0},
            'graph_derivation':{'kind':graph_origin,'clock':coordinate_kind,'dates':list(time),'phase_targets':phase_targets},
            'coverage':{'cells':len(normal.cells),'relations':len(normal.relations),'callbacks':len(normal.callbacks),
                        'unresolved_obligations':len(normal.obligations)}}
