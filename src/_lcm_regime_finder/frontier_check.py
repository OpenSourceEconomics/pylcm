"""Independent integer certificate replay for the declared graph frontier.

Imports only the standard library. Admission, costs, grouping enumeration,
interval recurrences and recommendation arithmetic are written here directly;
no producer translator, preparation, segment builder or scorer is called.
The independent tiny regression oracle additionally enumerates every cut mask.
"""
from __future__ import annotations

from collections import defaultdict
from bisect import bisect_left, bisect_right
from fractions import Fraction
import hashlib
from itertools import chain, combinations, product
import json
import math
import time


METRICS = ("work", "storage", "clarity")
ENDPOINT_CLARITY_POLICY = (
    "All semantically admissible final-date intervals are searched. By default the recommendation defers extra final-decision-date regimes used only to omit newly inactive coordinates; allow_final_only_split explicitly overrides this preference. The admissible family and Pareto frontier are unchanged."
)
FAMILY_ENCODING = "canonical JSON group/date buckets once; newline [group,start,stop,cost], reverse start order; unique groups"
SUBSET_ENCODING = 'canonical unique date buckets; interval records sorted by group_id; memoized label-subset recurrence records and singleton signature models sorted by canonical JSON'
BUCKET_ENCODING = SUBSET_ENCODING + '; independently checked constant date-bucket union dominance/v1'
PREFIX_ENCODING = BUCKET_ENCODING + '; independently checked constant-prefix single-cut dominance/v1'


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _vector(cost):
    return tuple(cost[name] for name in METRICS)


def _antichain(points):
    points = sorted(set(points))
    # Independent staircase envelope, not the producer's Fenwick routine.
    # Among earlier work coordinates, retain exactly the decreasing minima
    # of clarity as storage rises. A predecessor witnesses domination.
    storage, clarity, kept = [], [], []
    for point in points:
        _, s, c = point
        previous = bisect_right(storage, s) - 1
        if previous >= 0 and clarity[previous] <= c:
            continue
        kept.append(point)
        first = bisect_left(storage, s)
        stop = first
        while stop < len(storage) and clarity[stop] >= c:
            stop += 1
        storage[first:stop] = [s]
        clarity[first:stop] = [c]
    return kept


def _sum(a, b):
    return tuple(x + y for x, y in zip(a, b))


def _partitions(labels):
    """Pivot/subset enumeration, independent of producer restricted-growth DFS."""
    labels = tuple(labels)
    def unordered(remaining):
        if not remaining:
            yield ()
            return
        pivot, rest = remaining[0], remaining[1:]
        for size in range(len(rest) + 1):
            for subset in combinations(rest, size):
                chosen = set(subset)
                for tail in unordered(tuple(value for value in rest if value not in chosen)):
                    yield ((pivot, *subset), *tail)
    def growth_key(partition):
        owner = {label: index for index, block in enumerate(partition) for label in block}
        return tuple(owner[label] for label in labels)
    # The ordering is a certificate serialization convention, not an admission
    # restriction. Bounding occurs before materialization in the caller.
    return sorted(unordered(labels), key=growth_key)


class _Bounded(Exception):
    pass


class _IntervalTable:
    """Independent source-pair/prefix derivation, not producer accumulators.

    An internal directed pair has both endpoints in [i,j) exactly when its
    minimum bucket is >= i and its maximum bucket is < j. Activate rows by
    their minimum endpoint while i descends; a prefix by maximum endpoint
    gives internal counts. Subtract from the original-source outdegree prefix.
    External targets contribute to outdegree only. Storage is O(V+E), no dense
    date-by-date matrix or production Cartesian grid is allocated.
    """
    def __init__(self, family, buckets, policy):
        self.family, self.buckets = family, buckets
        self.n = len(buckets)
        where = {cell: i for i, bucket in enumerate(buckets) for cell in bucket}
        degrees = [0] * self.n
        self.activate = [[] for _ in buckets]
        for source, target in family.pairs:
            family.bound()
            if source not in where:
                continue
            a = where[source]
            degrees[a] += 1
            if target in where:
                b = where[target]
                self.activate[min(a, b)].append(max(a, b))
        self.outgoing_prefix = [0]
        self.member_prefix = [0]
        for i, bucket in enumerate(buckets):
            self.outgoing_prefix.append(self.outgoing_prefix[-1] + degrees[i])
            self.member_prefix.append(self.member_prefix[-1] + len(bucket))
        self.axis_positions = {kind: defaultdict(lambda: defaultdict(set)) for kind in ('states', 'actions')}
        self.characteristics, self.regimes, self.roles = defaultdict(set), defaultdict(set), defaultdict(set)
        for cell, i in where.items():
            option = family.options[cell][policy[cell]]
            for kind in ('states', 'actions'):
                for name, domain in option[kind].items():
                    self.axis_positions[kind][name][domain].add(i)
            for projection in option['state_projection'].values():
                if projection['kind'] == 'characteristic':
                    self.characteristics[_json(projection)].add(i)
            self.regimes[family.cells[cell]['regime']].add(i)
            self.roles[family.role(family.cells[cell])].add(i)
        self.axis_positions = {kind: {name: {domain: sorted(pos) for domain, pos in domains.items()}
            for name, domains in axes.items()} for kind, axes in self.axis_positions.items()}
        self.characteristics = [sorted(pos) for pos in self.characteristics.values()]
        self.regimes = [sorted(pos) for pos in self.regimes.values()]
        self.roles = [sorted(pos) for pos in self.roles.values()]

    @staticmethod
    def occurs(positions, start, stop):
        at = bisect_left(positions, start)
        return at < len(positions) and positions[at] < stop

    def axes(self, kind, start, stop):
        result = {}
        for name, domains in self.axis_positions[kind].items():
            found = [domain for domain, pos in domains.items() if self.occurs(pos, start, stop)]
            if len(found) > 1:
                return None
            if found:
                result[name] = found[0]
        return result

    def cost(self, start, stop, internal):
        family = self.family
        if sum(self.occurs(p, start, stop) for p in self.roles) != 1:
            return None
        if family.allocation['kind'] == 'global_union':
            states, actions = family.global_axes
        else:
            states, actions = (self.axes(kind, start, stop) for kind in ('states', 'actions'))
        if states is None or actions is None:
            return None
        state_rows, action_rows = family.sizes(states), family.sizes(actions)
        boundary = self.outgoing_prefix[stop] - self.outgoing_prefix[start] - internal
        members = self.member_prefix[stop] - self.member_prefix[start]
        characteristics = sum(self.occurs(p, start, stop) for p in self.characteristics)
        regimes = sum(self.occurs(p, start, stop) for p in self.regimes)
        return {'work': (members + boundary * family.boundary_unit) * state_rows * action_rows,
                'storage': members * state_rows,
                'clarity': family.weights['block'] + family.weights['characteristic'] * characteristics
                           + family.weights['dispatch'] * (regimes - 1)}


class IndependentFamily:
    """Independent data model/cost interpretation shared only by the checker.

    Public for tiny test-oracle use; the producer never imports this class.
    """

    def __init__(self, analysis, *, seconds=900, max_groupings=100_000, max_segments=2_000_000,
                 max_frontier_points=5000, subset_recurrence=True, bucket_dominance=False, prefix_dominance=False):
        self.started = time.perf_counter()
        self.analysis = analysis
        self.declaration = analysis["declarations"]
        self.cells = {cell["id"]: cell for cell in self.declaration["cells"]}
        self.domains = {**self.declaration["domains"], **analysis.get("derived_domains", {})}
        self.options = {cell: {option["id"]: option for option in record["options"]}
                        for cell, record in analysis["interfaces"].items()}
        self.allocation = analysis.get("allocation_scenario", self.declaration["allocation"])
        self.weights = {"block": 4, "characteristic": 2, "dispatch": 1, **analysis.get("clarity_weights", {})}
        self.seconds = seconds
        self.max_groupings = max_groupings
        self.max_segments = max_segments
        self.max_frontier_points = max_frontier_points
        self.counts = {"groupings": 0, "segments": 0}
        self.stream = hashlib.sha256()
        self.records = []
        self.grouping_records = []
        self.signature_records = []
        self.signature_models = {}
        self.signature_components = {}
        self.analysis_digest = hashlib.sha256(_json(analysis).encode()).hexdigest()
        self.subset_recurrence = subset_recurrence
        self.bucket_dominance = bucket_dominance
        self.prefix_dominance = prefix_dominance
        self.component_reductions = []
        self.interval_cache = {}
        self.roles = {name: self.role(cell) for name, cell in self.cells.items()}
        self.pairs = set()
        for relation in self.declaration['relations']:
            self.bound()
            if relation.get('support') != 'zero':
                self.pairs.update((relation['source'], target) for target in relation.get('targets', []))
        self.boundary_unit = self.allocation.get('boundary_unit', 1)
        if type(self.boundary_unit) is not int or self.boundary_unit < 0:
            raise ValueError('Invalid boundary work unit')
        kind = self.allocation.get('kind')
        if kind not in ('per_block', 'global_union'):
            raise ValueError('No explicit allocation contract or scenario')
        self.global_axes = None
        if kind == 'global_union':
            self.global_axes = tuple(self.allocation.get(axis) if self.allocation.get(axis) is not None
                else self.union(cell[axis] for cell in self.cells.values()) for axis in ('states', 'actions'))
            if any(axes is None for axes in self.global_axes):
                raise ValueError('Incompatible original global allocation domain identities')
        for value in self.weights.values():
            if type(value) is not int or value < 0: raise ValueError("Invalid clarity coefficient")
        if self.weights["block"] == 0: raise ValueError("Block clarity coefficient must be positive")

    def bound(self):
        if time.perf_counter() - self.started > self.seconds:
            raise _Bounded("independent checker elapsed-time budget")
        if self.counts["groupings"] > self.max_groupings:
            raise _Bounded("independent checker grouping budget")
        if self.counts["segments"] > self.max_segments:
            raise _Bounded("independent checker interval budget")

    def prune(self, candidates):
        seen = set()
        for point in candidates:
            seen.add(point)
            if len(seen) > max(1, 2 * min(self.max_frontier_points, 5000)):
                self.bound()
                seen = set(_antichain(seen))
                if len(seen) > self.max_frontier_points:
                    raise _Bounded('independent checker frontier-point budget')
        result = _antichain(seen)
        if len(result) > self.max_frontier_points:
            raise _Bounded('independent checker frontier-point budget')
        return result

    def role(self, cell):
        if hasattr(self, 'roles') and cell['id'] in self.roles:
            return self.roles[cell['id']]
        observed = cell.get("role")
        if observed is None:
            observed = ({'original_contexts_preserved_by_dispatch': True, 'decision_objective': cell.get('decision_objective')}
                if self.declaration.get('analysis_scope') == 'pylcm-structural-graph/v1'
                else {"unresolved_original_role": cell["regime"]})
        terminal_context=None if self.declaration.get('analysis_scope')=='pylcm-structural-graph/v1' else bool(cell.get('terminal'))
        return _json([observed, cell.get("ownership"), cell.get("stage"), terminal_context,
                      cell.get("declaration_type"), cell.get("structural_roles", {}), cell.get("solver_identity")])

    def components(self):
        records = defaultdict(lambda: defaultdict(list))
        for cell in self.cells.values(): records[self.role(cell)][cell["regime"]].append(cell["id"])
        return [dict(sorted(value.items())) for _, value in sorted(records.items())]

    def all_partitions(self, labels):
        # Bell numbers are computed without enumerating candidates; a large
        # complete claim cannot force an unbounded checker allocation.
        bell = [1]
        for n in range(1, len(labels) + 1):
            bell.append(sum(math.comb(n - 1, k) * bell[k] for k in range(n)))
        if bell[-1] > self.max_groupings - self.counts["groupings"]:
            raise _Bounded("independent checker grouping budget")
        return _partitions(labels)

    def sizes(self, axes):
        result = 1
        for domain_id in axes.values():
            domain = self.domains[domain_id]
            if domain.get("solve_axis") is False: continue
            size = domain.get("size")
            if type(size) is not int or size <= 0: raise ValueError("Original domain cardinality is missing")
            result *= size
        return result

    @staticmethod
    def union(axis_maps):
        merged = {}
        for axis_map in axis_maps:
            for name, domain in axis_map.items():
                if name in merged and merged[name] != domain: return None
                merged[name] = domain
        return merged

    def block(self, ids, policy):
        if not ids or len(ids) != len(set(ids)): return None
        cells = [self.cells[cell] for cell in ids]
        if len({self.role(cell) for cell in cells}) != 1: return None
        options = [self.options[cell][policy[cell]] for cell in ids]
        kind = self.allocation.get("kind")
        if kind == "per_block":
            states, actions = (self.union(option[key] for option in options) for key in ("states", "actions"))
        elif kind == "global_union":
            states, actions = self.global_axes
        else:
            raise ValueError("No explicit allocation contract or scenario")
        if states is None or actions is None: return None
        state_rows, action_rows = self.sizes(states), self.sizes(actions)
        boundary_unit = self.allocation.get("boundary_unit", 1)
        if type(boundary_unit) is not int or boundary_unit < 0: raise ValueError("Invalid boundary work unit")
        members = set(ids)
        directed = set()
        for source, target in self.pairs:
            if source in members and target not in members:
                directed.add((source, target))
        distinct_characteristics = set()
        for option in options:
            for projection in option["state_projection"].values():
                if projection["kind"] == "characteristic": distinct_characteristics.add(_json(projection))
        dispatch = len({cell["regime"] for cell in cells}) - 1
        return {"cost": {"work": (len(ids) + len(directed) * boundary_unit) * state_rows * action_rows,
                         "storage": len(ids) * state_rows,
                         "clarity": self.weights["block"] + self.weights["characteristic"] * len(distinct_characteristics) + self.weights["dispatch"] * dispatch},
                "state_axes": states, "action_axes": actions,
                "periods": sorted({cell["period"] for cell in cells}),
                "ages": [self.cells[cell]["age"] for cell in sorted(ids, key=lambda item: (self.cells[item]["period"], item))]}

    def interval_points(self, ids, policy):
        ids = sorted(ids)
        cache_key = (tuple(ids), tuple(policy[cell] for cell in ids))
        if cache_key in self.interval_cache:
            return self.interval_cache[cache_key]
        buckets = defaultdict(list)
        for cell in ids: buckets[self.cells[cell]["period"]].append(cell)
        buckets = [sorted(bucket) for _, bucket in sorted(buckets.items())]
        group_record = {'buckets': buckets, 'option_by_cell': {cell: policy[cell] for cell in ids}}
        group_id = hashlib.sha256(_json(group_record).encode()).hexdigest()
        n = len(buckets)
        table = _IntervalTable(self, buckets, policy)
        if (self.bucket_dominance and n>1
                and self.declaration.get('analysis_scope')=='pylcm-structural-graph/v1'
                and self.allocation['kind']=='per_block' and self.weights['block']>self.weights['dispatch']):
            signatures=[]
            for index,bucket in enumerate(buckets):
                self.bound()
                states,actions=(table.axes(kind,index,index+1) for kind in ('states','actions'))
                roles=frozenset(self.roles[c] for c in bucket)
                if states is None or actions is None or len(roles)!=1:break
                characteristics=tuple(i for i,positions in enumerate(table.characteristics)
                    if table.occurs(positions,index,index+1))
                signatures.append((states,actions,roles,characteristics))
            else:
                if all(signature==signatures[0] for signature in signatures):
                    whole=self.block(ids,policy)
                    if whole is not None:
                        self.counts['segments']+=1;self.bound()
                        self.records.append({'group_id':group_id,**group_record,'reduction':{
                            'rule':'constant-date-bucket-union/v1','cost':whole['cost']}})
                        points=[_vector(whole['cost'])]
                        self.interval_cache[cache_key]=points
                        return points
                if (self.prefix_dominance
                        and all(signature==signatures[0] for signature in signatures[:-1])
                        and signatures[-1][2]==signatures[0][2]):
                    # Independent pair endpoint counts give every prefix/suffix
                    # cost; no producer growth state or reduction flag is used.
                    by_maximum=[0]*n
                    for row in table.activate:
                        for maximum in row:by_maximum[maximum]+=1
                    prefixes={};internal=0
                    for stop in range(1,n):
                        internal+=by_maximum[stop-1]
                        self.counts['segments']+=1;self.bound()
                        cost=table.cost(0,stop,internal)
                        if cost is not None:prefixes[stop]=_vector(cost)
                    candidates=[];internal=0
                    for start in reversed(range(n)):
                        internal+=len(table.activate[start])
                        self.counts['segments']+=1;self.bound()
                        cost=table.cost(start,n,internal)
                        if cost is None:continue
                        if start==0:candidates.append(_vector(cost))
                        elif start in prefixes:candidates.append(_sum(prefixes[start],_vector(cost)))
                    points=self.prune(candidates)
                    self.records.append({'group_id':group_id,**group_record,'reduction':{
                        'rule':'constant-prefix-single-cut/v1',
                        'frontier':[dict(zip(METRICS,point)) for point in points]}})
                    self.interval_cache[cache_key]=points
                    return points
        active_maximum = [0] * n
        suffix = {n: [(0, 0, 0)]}
        for start in reversed(range(n)):
            for maximum in table.activate[start]:
                active_maximum[maximum] += 1
            possibilities = []
            internal = 0
            for stop in range(start + 1, n + 1):
                self.counts["segments"] += 1
                self.bound()
                internal += active_maximum[stop - 1]
                cost = table.cost(start, stop, internal)
                if not self.subset_recurrence:
                    self.stream.update((_json([group_id, start, stop, cost]) + "\n").encode())
                if cost is None and self.subset_recurrence:
                    # An interval with conflicting domains/protected contexts
                    # keeps that conflict at every larger endpoint.
                    self.counts['segments']+=n-stop;self.bound()
                    break
                if cost is not None:
                    possibilities.extend(_sum(_vector(cost), rest) for rest in suffix[stop])
                    if len(possibilities) > max(1, 2 * min(self.max_frontier_points, 5000)):
                        possibilities = self.prune(possibilities)
            suffix[start] = self.prune(possibilities)
        self.records.append({"group_id": group_id, **group_record, "suffix_frontiers":
                             [[dict(zip(METRICS, point)) for point in suffix[index]] for index in range(n + 1)]})
        self.interval_cache[cache_key] = suffix[0]
        return suffix[0]

    def contracted_layout_points(self, component, policy):
        nodes = sorted(node for members in component.values() for node in members)
        selected = {node: self.options[node][policy[node]] for node in nodes}
        inside = set(nodes)
        if (self.declaration.get('analysis_scope') != 'pylcm-structural-graph/v1'
                or self.weights['block'] <= self.weights['dispatch']
                or any(len(members) != 1 for members in component.values())
                or any(left in inside and right in inside for left, right in self.pairs)):
            return None
        if any(item['actions'] != selected[nodes[0]]['actions'] for item in selected.values()):
            return None
        if any(projection.get('kind') == 'characteristic'
               for item in selected.values() for projection in item['state_projection'].values()):
            return None
        classes = defaultdict(list)
        for node in nodes: classes[_json(selected[node]['states'])].append(node)
        frontier = []
        for partition in self.all_partitions(tuple(sorted(classes))):
            self.counts['groupings'] += 1
            self.bound()
            value = (0, 0, 0)
            for group in partition:
                block = self.block(sorted(node for layout in group for node in classes[layout]), policy)
                if block is None: break
                value = _sum(value, _vector(block['cost']))
            else:
                frontier.append(value)
            if len(frontier) > max(1, 2*min(self.max_frontier_points, 5000)):
                frontier = self.prune(frontier)
        self.component_reductions.append({'rule': 'single-date-layout-contraction/v1',
            'layouts': [classes[layout] for layout in sorted(classes)],
            'option_by_cell': {node: policy[node] for node in nodes}})
        return self.prune(frontier)

    def separate_domains(self, component, policy):
        if (self.declaration.get('analysis_scope')!='pylcm-structural-graph/v1'
                or self.allocation.get('kind')!='per_block'):
            return [component]
        by_label={}
        def signature(node):
            selected=self.options[node][policy[node]]
            return tuple(tuple(sorted(selected[kind].items())) for kind in ('states','actions'))
        for label,nodes in component.items(): by_label[label]={signature(node) for node in nodes}
        keys=sorted(set().union(*by_label.values()))
        adjacency={key:set() for key in keys}
        for variants in by_label.values():
            first=min(variants)
            for item in variants:
                adjacency[first].add(item);adjacency[item].add(first)
        for i,left in enumerate(keys):
            self.bound()
            for right in keys[i+1:]:
                if all(all(name not in dict(a) or dict(a)[name]==domain for name,domain in b)
                       for a,b in zip(left,right)):
                    adjacency[left].add(right);adjacency[right].add(left)
        remaining=set(keys);owners={}
        while remaining:
            root=min(remaining);todo=[root];seen=set()
            while todo:
                current=todo.pop()
                if current in seen:continue
                seen.add(current);todo.extend(adjacency[current]-seen)
            remaining-=seen
            for item in seen: owners[item]=root
        grouped=defaultdict(dict)
        for label in sorted(component):grouped[owners[min(by_label[label])]][label]=component[label]
        result=sorted(grouped.values(),key=lambda part:tuple(part))
        if len(result)>1:
            self.component_reductions.append({'rule':'incompatible-domain-components/v1',
                'components':[list(part) for part in result],
                'option_by_cell':{node:policy[node] for node in sorted(node for group in component.values() for node in group)}})
        return result

    def signature_instance(self,component,policy):
        nodes=sorted(node for group in component.values() for node in group)
        cells=[];domains={}
        for node in nodes:
            source=self.cells[node];option=self.options[node][policy[node]]
            cells.append({'id':node,'label':source['regime'],'period':source['period'],'age':source.get('age'),
                          'context':self.role(source),'states':option['states'],'actions':option['actions'],
                          'characteristics':sorted({_json(p) for p in option['state_projection'].values() if p['kind']=='characteristic'})})
            for kind in ('states','actions'):
                for name in option[kind].values():
                    d=self.domains[name]
                    domains[name]={'size':d['size'],'solve_axis':d.get('solve_axis',True),
                                   'source_domain_sha256':hashlib.sha256(_json(d).encode()).hexdigest()}
        return {'cells':cells,'domains':dict(sorted(domains.items())),
                'pairs':sorted((source,target) for source,target in self.pairs if source in set(nodes)),
                'allocation':'per_block','boundary_unit':self.boundary_unit,'clarity_weights':self.weights,
                'bindings':{'analysis_sha256':self.analysis_digest,
                            'policy_sha256':hashlib.sha256(_json(policy).encode()).hexdigest(),
                            'option_by_cell':{node:policy[node] for node in nodes}}}

    def signature_record(self,component,policy):
        from .signature_proof import binary_model, canonical_model, digest, union_closure, signature, join
        instance=self.signature_instance(component,policy)
        identity=digest(instance)
        if identity in self.signature_models:return self.signature_models[identity]
        self.bound()
        try:
            bins=union_closure(instance)
            offered=[[k for k,bin in enumerate(bins) if join(signature(cell),bin)==bin] for cell in instance['cells']]
            upper=sum(map(len,offered))+len(bins)+len(instance['pairs'])*len(bins)
            if upper>1000000:raise _Bounded('independent signature model variable budget')
            model=canonical_model(binary_model(instance))
        except RuntimeError as exc:
            if isinstance(exc,_Bounded):raise
            raise _Bounded(str(exc)) from exc
        self.bound()
        record={'schema':'regime-signature-component/v1','status':'PARETO_FAMILY_REPRESENTED',
                'instance':instance,'model':model,'model_sha256':digest(model),
                'assignment_count':math.prod(map(len,offered))}
        self.signature_models[identity]=record
        return record

    def check_signature_record(self,record):
        """Derive the full protected component from the analysis, not the record."""
        policy_hash=record['instance']['bindings']['policy_sha256']
        found=next((p['option_by_cell'] for p in self.analysis['projection_policies']
                    if hashlib.sha256(_json(p['option_by_cell']).encode()).hexdigest()==policy_hash),None)
        if found is None:raise ValueError('Signature record has an unknown projection policy')
        if policy_hash not in self.signature_components:
            old=len(self.component_reductions)
            pieces=[piece for component in self.components() for piece in self.separate_domains(component,found)]
            del self.component_reductions[old:]
            self.signature_components[policy_hash]={tuple(sorted(c for ids in piece.values() for c in ids)):piece for piece in pieces}
        ids=tuple(cell['id'] for cell in record['instance']['cells'])
        component=self.signature_components[policy_hash].get(ids)
        if component is None:raise ValueError('Signature model does not cover an exact protected/domain component')
        expected=self.signature_record(component,found)
        if _json(record)!=_json(expected):raise ValueError('Signature model/source binding or integer constraints changed')
        return expected['model']

    def signature_points(self,component,policy):
        from .signature_proof import signature, join, union_signatures, literal_partition_cost
        record=self.signature_record(component,policy)
        self.signature_records.append(record)
        if record['assignment_count']>self.max_groupings-self.counts['groupings']:
            raise _Bounded('independent signature assignment cover budget')
        instance=record['instance'];bins=record['model']['bins']
        choices=[[index for index,bin in enumerate(bins) if join(signature(cell),bin)==bin] for cell in instance['cells']]
        def costs():
            for assignment in product(*choices):
                self.counts['groupings']+=1;self.bound()
                members=defaultdict(list)
                for cell,bin in zip(instance['cells'],assignment):members[bin].append(cell)
                if any(union_signatures(signature(cell) for cell in group)!=bins[bin] for bin,group in members.items()):continue
                value=literal_partition_cost(instance,[[cell['id'] for cell in group] for group in members.values()])
                if value is None:raise ValueError('Independently admitted signature assignment is infeasible')
                yield value
        return self.prune(costs())

    def subset_points(self, component, policy):
        """Independent bottom-up replay of every reachable remaining-label set."""
        labels = sorted(component)
        if 2**(len(labels)-1) > self.max_groupings-self.counts['groupings']:
            raise _Bounded('independent checker grouping budget')
        full = 2**len(labels)-1
        table = {0:[(0,0,0)]}
        for mask in range(1,full+1):
            if mask & 1 and mask != full:
                continue
            remaining = [i for i in range(len(labels)) if mask & (1 << i)]
            pivot, rest = remaining[0], remaining[1:]
            points = []
            for count in range(len(rest)+1):
                for selected in combinations(rest,count):
                    self.counts['groupings'] += 1
                    self.bound()
                    group = (pivot,*selected)
                    group_mask = sum(1 << i for i in group)
                    ids = [cell for i in group for cell in component[labels[i]]]
                    intervals = self.interval_points(ids,policy)
                    points = self.prune(chain(points,(_sum(a,b) for a in intervals for b in table[mask ^ group_mask])))
            table[mask] = points
        self.grouping_records.append({'rule':'label-subset-recurrence/v1',
            'cells_by_label':{label:sorted(component[label]) for label in labels},
            'option_by_cell':{cell:policy[cell] for cell in sorted(c for cells in component.values() for c in cells)},
            'states':[{'remaining':[label for i,label in enumerate(labels) if mask & (1 << i)],
                       'frontier':[dict(zip(METRICS,p)) for p in table[mask]]} for mask in sorted(table)]})
        return table[full]

    def complete_frontier(self):
        all_points = []
        components = self.components()
        for policy_record in self.analysis["projection_policies"]:
            policy = policy_record["option_by_cell"]
            if set(policy) != set(self.cells): raise ValueError("Projection policy does not cover all original cells")
            across_components = [(0, 0, 0)]
            def separated():
                for original_component in components:
                    yield from self.separate_domains(original_component,policy)
            for component in separated():
                identifiers = sorted(cell for members in component.values() for cell in members)
                layouts = {(tuple(sorted(self.options[cell][policy[cell]]['states'].items())),
                            tuple(sorted(self.options[cell][policy[cell]]['actions'].items()))) for cell in identifiers}
                if (self.declaration.get('analysis_scope') == 'pylcm-structural-graph/v1'
                        and self.weights['block'] > self.weights['dispatch'] and len(layouts) == 1):
                    combined = self.block(identifiers, policy)
                    if combined is not None:
                        self.counts['groupings'] += 1
                        self.bound()
                        self.component_reductions.append({'rule': 'homogeneous-layout-dominance/v1',
                            'cells': identifiers, 'option_by_cell': {cell: policy[cell] for cell in identifiers}})
                        across_components = self.prune(_sum(point, _vector(combined['cost'])) for point in across_components)
                        continue
                contracted = self.contracted_layout_points(component, policy)
                if contracted is not None:
                    across_components = self.prune(_sum(left, right) for left in across_components for right in contracted)
                    continue
                if (self.subset_recurrence and len(component)>1 and all(len(nodes)==1 for nodes in component.values())
                        and self.allocation['kind']=='per_block' and self.weights['block']>self.weights['dispatch']):
                    component_points=self.signature_points(component,policy)
                    across_components=self.prune(_sum(left,right) for left in across_components for right in component_points)
                    continue
                if self.subset_recurrence:
                    component_points = self.subset_points(component,policy)
                    across_components = self.prune(_sum(left,right) for left in across_components for right in component_points)
                    continue
                component_candidates = []
                for partition in self.all_partitions(tuple(component)):
                    self.counts["groupings"] += 1
                    self.bound()
                    across_groups = [(0, 0, 0)]
                    for group in partition:
                        group_cells = [cell for regime in group for cell in component[regime]]
                        intervals = self.interval_points(group_cells, policy)
                        across_groups = self.prune(_sum(left, right) for left in across_groups for right in intervals)
                    component_candidates.extend(across_groups)
                    if len(component_candidates) > max(1, 2 * min(self.max_frontier_points, 5000)):
                        component_candidates = self.prune(component_candidates)
                component_points = self.prune(component_candidates)
                across_components = self.prune(_sum(left, right) for left in across_components for right in component_points)
            all_points.extend(across_components)
            if len(all_points) > max(1, 2 * min(self.max_frontier_points, 5000)):
                all_points = self.prune(all_points)
        return self.prune(all_points)

    def admitted_plan(self, plan):
        errors = []
        policies = {record["id"]: record["option_by_cell"] for record in self.analysis["projection_policies"]}
        policy = policies.get(plan.get("projection_policy"))
        if policy is None: return ["unknown global projection policy"]
        covered, total = [], (0, 0, 0)
        parent = {(self.role(cell), cell["regime"]): (self.role(cell), cell["regime"]) for cell in self.cells.values()}
        def find(item):
            while parent[item] != item: item = parent[item]
            return item
        for block in plan.get("blocks", []):
            ids = block.get("cells", [])
            if any(cell not in self.cells for cell in ids):
                errors.append("unknown original cell in block"); continue
            covered.extend(ids)
            if block.get("option_by_cell") != {cell: policy[cell] for cell in ids}: errors.append("block option/global policy mismatch")
            expected = self.block(ids, policy)
            if expected is None:
                errors.append("inadmissible block"); continue
            for key in ("cost", "state_axes", "action_axes", "periods", "ages"):
                if block.get(key) != expected[key]: errors.append("incorrect block " + key)
            total = _sum(total, _vector(expected["cost"]))
            labels = [(self.role(self.cells[cell]), self.cells[cell]["regime"]) for cell in ids]
            for label in labels[1:]: parent[find(label)] = find(labels[0])
        if sorted(covered) != sorted(self.cells): errors.append("plan does not cover each exact original admission once")
        if plan.get("cost") != dict(zip(METRICS, total)): errors.append("incorrect plan total")
        groups = defaultdict(list)
        for cell in self.cells.values(): groups[find((self.role(cell), cell["regime"]))].append(cell["id"])
        for group_ids in groups.values():
            bucket = defaultdict(set)
            for cell in group_ids: bucket[self.cells[cell]["period"]].add(cell)
            dates = sorted(bucket)
            for block in plan.get("blocks", []):
                members = set(block.get("cells", []))
                overlap = members & set(group_ids)
                if not overlap: continue
                indexes = [index for index, date in enumerate(dates) if bucket[date] & overlap]
                expected = set().union(*(bucket[dates[index]] for index in range(min(indexes), max(indexes) + 1)))
                if overlap != expected: errors.append("block is not an interval of complete exact date buckets")
        return errors


def _independent_endpoint_preference(family: IndependentFamily, point: dict) -> bool:
    """Derive the preference from original dated cells and selected interfaces.

    Traverse each original regime's timeline, rather than consuming producer
    flags or importing its endpoint classifier or block-construction code.
    """
    timelines = defaultdict(lambda: defaultdict(set))
    native=family.declaration.get('analysis_scope')=='pylcm-structural-graph/v1'
    decision_labels={cell['regime'] for cell in family.cells.values() if not cell.get('terminal')}
    for cid, cell in family.cells.items():
        if (cell['regime'] in decision_labels if native else not cell.get('terminal')):
            timelines[cell['regime']][cell['period']].add(cid)
    if not timelines:
        return False
    last = max(date for timeline in timelines.values() for date in timeline)
    owner = {cid: index for index, block in enumerate(point['blocks']) for cid in block['cells']}
    option = {cid: family.options[cid][point['blocks'][owner[cid]]['option_by_cell'][cid]] for cid in owner}
    for label, dates in timelines.items():
        if last not in dates or len(dates) < 2:
            continue
        previous_date = max(date for date in dates if date < last)
        for final_index in {owner[cid] for cid in dates[last]}:
            end_ids = point['blocks'][final_index]['cells']
            if any((family.cells[cid].get('terminal') and not native) or family.cells[cid]['period'] != last
                   or family.cells[cid]['regime'] != label for cid in end_ids):
                continue
            for prior_index in {owner[cid] for cid in dates[previous_date]}:
                before_ids = point['blocks'][prior_index]['cells']
                if any(family.cells[cid]['regime'] != label or family.cells[cid]['period'] > previous_date for cid in before_ids):
                    continue
                if len({family.roles[cid] for cid in end_ids + before_ids}) != 1:
                    continue
                actual_previous = dates[previous_date].intersection(before_ids)
                reduced = False
                matches = True
                for axis_kind, reconstruction, expected_kind in (
                        ('states', 'state_projection', 'drop'),
                        ('actions', 'action_reconstruction', 'constant')):
                    old_axes = family.union(option[cid][axis_kind] for cid in before_ids)
                    new_axes = family.union(option[cid][axis_kind] for cid in end_ids)
                    if old_axes is None or new_axes is None or not new_axes.items() <= old_axes.items():
                        matches = False
                        break
                    missing = old_axes.keys() - new_axes.keys()
                    reduced |= bool(missing)
                    for cid in end_ids:
                        for name in missing:
                            if name not in family.cells[cid][axis_kind] or option[cid][reconstruction].get(name, {}).get('kind') != expected_kind:
                                matches = False
                        for prior_id in actual_previous:
                            if not missing <= option[prior_id][axis_kind].keys():
                                matches = False
                            if any(option[cid][reconstruction].get(name) != option[prior_id][reconstruction].get(name)
                                   for name in new_axes):
                                matches = False
                if matches and reduced:
                    return True
    return False


def check_frontier(analysis, result, *, limits=None):
    """Check complete frontier or certify only supplied incomplete incumbents."""
    errors = []
    unresolved = []
    checked = 0
    limits = limits or {}
    try:
        family = IndependentFamily(analysis, seconds=limits.get("seconds", 900),
                                   max_groupings=limits.get("groupings", 100_000),
                                   max_segments=limits.get("segments", 2_000_000),
                                   max_frontier_points=limits.get('frontier_points', 5000),
                                   subset_recurrence=result.get('schema') in ('regime-frontier/v3','regime-frontier/v4','regime-frontier/v5'),
                                   bucket_dominance=result.get('schema') in ('regime-frontier/v4','regime-frontier/v5'),
                                   prefix_dominance=result.get('schema')=='regime-frontier/v5')
        certificate = result.get("certificate", {})
        if result.get('schema') not in ('regime-frontier/v2','regime-frontier/v3','regime-frontier/v4','regime-frontier/v5'): errors.append('frontier format version')
        expected_encoding = PREFIX_ENCODING if family.prefix_dominance else BUCKET_ENCODING if family.bucket_dominance else SUBSET_ENCODING if family.subset_recurrence else FAMILY_ENCODING
        if certificate.get('family_encoding') != expected_encoding: errors.append('candidate-family encoding')
        expected_digest = hashlib.sha256(_json(analysis).encode()).hexdigest()
        if certificate.get("analysis_sha256") != expected_digest: errors.append("analysis identity digest")
        if result.get("allocation_contract") != family.allocation: errors.append("allocation contract/scenario changed")
        if result.get("clarity_weights") != family.weights: errors.append("clarity metric changed")
        if result.get("endpoint_clarity_policy") != ENDPOINT_CLARITY_POLICY: errors.append("fixed endpoint clarity policy changed")
        complete = result.get("status") == "COMPLETE"
        if certificate.get("complete") is not complete: errors.append("completeness flag/status contradiction")
        if result.get("status") not in {"COMPLETE", "INCOMPLETE"}: errors.append("unknown frontier status")
        points = result.get("frontier", [])
        if len({point.get("id") for point in points}) != len(points): errors.append("duplicate frontier graph IDs")
        for point in points:
            errors.extend(point.get("id", "?") + ": " + issue for issue in family.admitted_plan(point))
            checked += 1
        actual_vectors = [_vector(point["cost"]) for point in points]
        if len(set(actual_vectors)) != len(actual_vectors): errors.append("duplicate cost vectors")
        if sorted(actual_vectors) != _antichain(actual_vectors): errors.append("frontier contains dominated points")
        if certificate.get("projection_policies") != [record["id"] for record in analysis["projection_policies"]]: errors.append("projection policy family changed")
        if certificate.get("component_regimes") != [list(component) for component in family.components()]: errors.append("protected-role grouping family changed")
        if family.subset_recurrence:
            for record in certificate.get('signature_records',[]):family.check_signature_record(record)
        if complete:
            expected_vectors = family.complete_frontier()
            if sorted(actual_vectors) != expected_vectors: errors.append("frontier is not the complete nondominated cost set")
            if certificate.get('component_reductions', []) != family.component_reductions:
                errors.append('component dominance reduction proof changed')
            records = sorted(family.records,key=lambda row:row['group_id']) if family.subset_recurrence else family.records
            groups = sorted(family.grouping_records,key=_json)
            signatures=sorted(family.signature_records,key=_json)
            expected_hash = hashlib.sha256(_json([records,groups,signatures]).encode()).hexdigest() if family.subset_recurrence else family.stream.hexdigest()
            if certificate.get("family_sha256") != expected_hash: errors.append("candidate-family stream digest")
            if certificate.get("interval_records") != records: errors.append("interval-frontier recurrence certificate")
            if family.subset_recurrence and certificate.get('grouping_records') != groups:
                errors.append('label-subset recurrence coverage certificate')
            if family.subset_recurrence and _json(certificate.get('signature_records')) != _json(sorted(family.signature_records,key=_json)):
                errors.append('singleton signature family coverage certificate')
            if certificate.get("search_counts") != family.counts: errors.append("enumerated family counts")
            if certificate.get("lower_bound") is not None: errors.append("unexpected complete-search lower bound")
            if result.get("exhaustion_reason") is not None: errors.append("complete search reports exhaustion")
        else:
            if certificate.get("lower_bound") != dict.fromkeys(METRICS, 0): errors.append("unestablished incomplete-search lower bound")
            if not result.get("exhaustion_reason"): errors.append("incomplete search lacks exhaustion reason")
            unresolved.append("Global frontier completeness is not claimed: only supplied complete plans, Pareto antichain and the zero lower bound were checked")
        recommendation = result.get("recommendation")
        if not points:
            if recommendation is not None: errors.append("recommendation without an incumbent")
        elif not isinstance(recommendation, dict):
            errors.append("missing configurable recommendation")
        else:
            weights = {key: Fraction(str(recommendation["weights"][key])) for key in METRICS}
            scales = recommendation["reference_scales"]
            preference = recommendation['endpoint_preference']
            override = preference['allow_final_only_split']
            if type(override) is not bool:
                raise ValueError('allow_final_only_split must be a boolean')
            deferred = sorted(point['id'] for point in points if _independent_endpoint_preference(family, point))
            if preference.get('deferred_graph_ids') != deferred or preference.get('reason') != ENDPOINT_CLARITY_POLICY:
                errors.append('endpoint recommendation preference changed')
            eligible = points if override else [point for point in points if point['id'] not in deferred]
            if any(value <= 0 for value in weights.values()) or any(type(scales[key]) is not int or scales[key] <= 0 for key in METRICS):
                errors.append("invalid recommendation weights/scales")
            else:
                scores = {point["id"]: sum(weights[key] * Fraction(point["cost"][key], scales[key]) for key in METRICS) for point in points}
                winner = min(eligible, key=lambda point: (scores[point["id"]], _vector(point["cost"]),
                    _json([(block["cells"], block["option_by_cell"]) for block in point["blocks"]]))) if eligible else None
                if recommendation.get("graph_id") != (winner["id"] if winner else None): errors.append("recommendation is not the configured minimizer")
                if winner is None:
                    if recommendation.get('score') is not None: errors.append('score without a preference-eligible recommendation')
                elif Fraction(str(recommendation.get("score"))) != scores[winner["id"]]: errors.append("recommendation rational score")
        return {"status": "FAIL" if errors else "PASS", "errors": errors, "unresolved": unresolved,
                "checking_complete": True, "exhaustion_reason": None,
                "frontier_complete_verified": complete and not errors,
                "projection_family_discovery_complete": analysis.get("discovery_complete") is True,
                "frontier_scope": "the explicitly offered coherent projection-policy family; source/projection discovery is checked separately",
                "checked_complete_plans": checked, "independent_counts": family.counts,
                "dense_cartesian_rows_evaluated": 0,
                "guarantee": "integer metrics, admitted global projection policy/grouping/date-interval family, frontier recurrence and configured recommendation"}
    except _Bounded as exc:
        return {"status": "UNRESOLVED", "errors": errors, "unresolved": [str(exc)],
                "checking_complete": False, "exhaustion_reason": str(exc),
                "frontier_complete_verified": False, "checked_complete_plans": checked,
                "dense_cartesian_rows_evaluated": 0}
    except (KeyError, TypeError, ValueError, ZeroDivisionError, OverflowError) as exc:
        return {"status": "FAIL", "errors": errors + ["malformed frontier certificate: " + str(exc)],
                "unresolved": unresolved, "frontier_complete_verified": False,
                "checked_complete_plans": checked, "dense_cartesian_rows_evaluated": 0}


def verify_signature_cover(analysis,component_record,certificate,*,seconds=900,max_visits=1000000):
    """Check a component's exact integer Pareto cover against the analysis.

    This is not a whole-Model or source-semantics certificate. No solver status
    or producer constraint model is trusted; the integer model is rebuilt here.
    """
    from .signature_proof import digest, verify_binary_assignment, frontier_cover_clauses, check_unsat_tree
    try:
        family=IndependentFamily(analysis,seconds=seconds)
        model=family.check_signature_record(component_record)
        if certificate['model_sha256']!=digest(model):raise ValueError('Cover proof model identity changed')
        vectors=[verify_binary_assignment(model,assignment) for assignment in certificate['assignments']]
        if len(vectors)!=len(set(vectors)) or sorted(vectors)!=_antichain(vectors):
            raise ValueError('Cover witnesses are not a unique Pareto antichain')
        receipt=check_unsat_tree(model,frontier_cover_clauses(model,vectors),certificate['coverage_proof'],
                                 max_visits=max_visits,check=family.bound)
        return {'status':'PASS','component_frontier_complete_verified':True,
                'global_frontier_complete_verified':False,'source_semantics':'NOT_CHECKED_HERE',
                'analysis_sha256':family.analysis_digest,'model_sha256':digest(model),
                'vectors':sorted(vectors),'proof':receipt}
    except RuntimeError as exc:
        return {'status':'UNRESOLVED','component_frontier_complete_verified':False,'reason':str(exc)}
    except (KeyError,TypeError,ValueError,OverflowError,RecursionError) as exc:
        return {'status':'FAIL','component_frontier_complete_verified':False,'errors':[str(exc)]}
