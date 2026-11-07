"""Finite grouping/interval Pareto search over checked declaration metadata.

No callback, numerical law, or Cartesian state/action tensor is evaluated here.
Regime grouping is exhaustively searched; no graph coloring defines the family.
"""
from __future__ import annotations
from collections import defaultdict
from bisect import bisect_left
from fractions import Fraction
from itertools import chain, product
import hashlib
import json
import time

METRICS = ('work', 'storage', 'clarity')
FAMILY_ENCODING = 'canonical JSON group/date buckets once; newline [group,start,stop,cost], reverse start order; unique groups'
SUBSET_ENCODING = 'canonical unique date buckets; interval records sorted by group_id; memoized label-subset recurrence records and singleton signature models sorted by canonical JSON'
BUCKET_ENCODING = SUBSET_ENCODING + '; independently checked constant date-bucket union dominance/v1'
PREFIX_ENCODING = BUCKET_ENCODING + '; independently checked constant-prefix single-cut dominance/v1'
ENDPOINT_CLARITY_POLICY = 'All semantically admissible final-date intervals are searched. By default the recommendation defers extra final-decision-date regimes used only to omit newly inactive coordinates; allow_final_only_split explicitly overrides this preference. The admissible family and Pareto frontier are unchanged.'


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def vector(cost):
    return tuple(cost[k] for k in METRICS)


def dominates(a, b):
    return all(x <= y for x, y in zip(a, b)) and a != b


def plan_key(plan):
    # Preserve the previous byte-for-byte layout tie convention. Long member
    # lists are materialized only for a genuine equal-cost tie or final output.
    return '[' + ','.join(
        b.tie_key() if isinstance(b, _BlockRef) else canonical([b['cells'], b['option_by_cell']])
        for b in plan['blocks']) + ']'


def pareto(plans):
    unique = {}
    for p in plans:
        v = vector(p['cost'])
        if v not in unique or _layout_compare(p, unique[v]) < 0:
            unique[v] = p
    ordered = sorted(unique)
    # Sweep work, then storage, then clarity. A prefix-minimum Fenwick tree
    # answers whether an earlier vector has storage <= this one and clarity
    # <= this one. Exact equal costs have already chosen their layout winner.
    storage = sorted({v[1] for v in ordered})
    prefix_min = [None] * (len(storage) + 1)
    result = []
    for v in ordered:
        index = bisect_left(storage, v[1]) + 1
        best, at = None, index
        while at:
            known = prefix_min[at]
            if known is not None and (best is None or known < best):
                best = known
            at -= at & -at
        if best is None or best > v[2]:
            result.append(unique[v])
        at = index
        while at < len(prefix_min):
            known = prefix_min[at]
            if known is None or v[2] < known:
                prefix_min[at] = v[2]
            at += at & -at
    return result


def add(left, right):
    return {'cost': {k: left['cost'][k] + right['cost'][k] for k in METRICS},
            'blocks': _JoinedBlocks(left['blocks'], right['blocks'])}


def zero():
    return {'cost': dict.fromkeys(METRICS, 0), 'blocks': []}


class Exhausted(RuntimeError):
    pass


class _JoinedBlocks:
    """A persistent concatenation/backpointer; no copied prefix membership."""
    __slots__ = ('left', 'right', 'first')

    def __init__(self, left, right):
        self.left, self.right = left, right
        # Date-DP heads are immutable single-block tuples. Other iterables
        # retain the general traversal, including externally mutable lists.
        self.first = left[0] if isinstance(left,tuple) and left else left.first if isinstance(left,_JoinedBlocks) else None

    def __iter__(self):
        stack = [self]
        while stack:
            item = stack.pop()
            if isinstance(item, _JoinedBlocks):
                stack.extend((item.right, item.left))
            else:
                yield from item


class _BlockRef:
    __slots__ = ('owner', 'buckets', 'policy', 'start', 'stop', 'cost', 'mask', '_materialized')

    def __init__(self, owner, buckets, policy, start, stop, cost, mask):
        self.owner, self.buckets, self.policy = owner, buckets, policy
        self.start, self.stop, self.cost, self.mask = start, stop, cost, mask
        self._materialized = None

    @classmethod
    def from_materialized(cls, owner, block, policy):
        mask = sum(1 << owner.cell_index[c] for c in block['cells'])
        obj = cls(owner, (), policy, 0, 0, block['cost'], mask)
        obj._materialized = block
        return obj

    def members(self):
        result, bits = [], self.mask
        while bits:
            low = bits & -bits
            result.append(self.owner.cell_order[low.bit_length() - 1])
            bits ^= low
        return result

    def tie_key(self):
        ids = self.members()
        return canonical([ids, {c: self.policy[c] for c in ids}])

    def materialize(self):
        if self._materialized is not None:
            return self._materialized
        block = self.owner.block(self.members(), self.policy)
        if block is None or block['cost'] != self.cost:
            raise ValueError('Indexed interval and materialized block disagree')
        return block


def _block_compare(left, right):
    if isinstance(left, _BlockRef) and isinstance(right, _BlockRef) and left.owner is right.owner:
        different = left.mask ^ right.mask
        if different:
            first = (different & -different).bit_length() - 1
            left_first = bool(left.mask & (1 << first))
            other = (right.mask if left_first else left.mask) >> first
            if not other:
                # A longer JSON list with the same prefix sorts before its
                # shorter prefix: comma/quote sort before closing bracket.
                return -1 if left_first else 1
            following = first + (other & -other).bit_length() - 1
            a, b = left.owner.encoded_cell_ids[first], left.owner.encoded_cell_ids[following]
            order = -1 if a < b else 1
            return order if left_first else -order
        if left.policy is right.policy:
            return 0
        bits = left.mask
        while bits:
            low = bits & -bits
            cell = left.owner.cell_order[low.bit_length() - 1]
            a, b = canonical(left.policy[cell]), canonical(right.policy[cell])
            if a != b:
                return -1 if a < b else 1
            bits ^= low
        return 0
    a = left.tie_key() if isinstance(left, _BlockRef) else canonical([left['cells'], left['option_by_cell']])
    b = right.tie_key() if isinstance(right, _BlockRef) else canonical([right['cells'], right['option_by_cell']])
    return (a > b) - (a < b)


def _layout_compare(left, right):
    """Canonical JSON layout ordering without expanding interval membership.

    The first differing membership bit gives the first unequal sorted cell;
    encoded IDs retain JSON escaping semantics. Equal membership compares the
    selected option values in canonical key order. Persistent block chains are
    traversed lazily, so this does not allocate long interval strings/lists.
    """
    left_blocks,right_blocks=left['blocks'],right['blocks']
    if (isinstance(left_blocks,_JoinedBlocks) and isinstance(right_blocks,_JoinedBlocks)
            and left_blocks.first is not None and right_blocks.first is not None):
        order=_block_compare(left_blocks.first,right_blocks.first)
        if order:return order
    a, b, done = iter(left_blocks), iter(right_blocks), object()
    while True:
        x, y = next(a, done), next(b, done)
        if x is done or y is done:
            return 0 if x is done and y is done else (1 if x is done else -1)
        order = _block_compare(x, y)
        if order:
            return order


class _GrowingInterval:
    """Exact sparse directed-boundary update as a date bucket is appended.

    For old membership M and a new cell x, remove incoming pairs M -> x and
    add outgoing pairs x -> outside (M union {x}). Each distinct supported pair
    is indexed once. Self-pairs never cross; unknown support remains present.
    """
    def __init__(self, owner, policy):
        self.owner, self.policy = owner, policy
        self.members, self.regimes, self.characteristics = set(), set(), set()
        self.mask = 0
        self.states, self.actions = {}, {}
        self.crossings = 0
        self.compatible = True
        self.role = None

    def extend(self, bucket):
        for c in bucket:
            self.owner.tick()
            role = self.owner.roles[c]
            if self.role is None:
                self.role = role
            self.compatible = self.compatible and role == self.role
            self.crossings -= len(self.owner.incoming.get(c, set()) & self.members)
            self.crossings += sum(t != c and t not in self.members for t in self.owner.outgoing.get(c, ()))
            self.members.add(c)
            self.mask |= 1 << self.owner.cell_index[c]
            self.regimes.add(self.owner.cells[c]['regime'])
            option = self.owner.options[c][self.policy[c]]
            self.characteristics.update(self.owner.characteristics[(c, self.policy[c])])
            for key, axes in (('states', self.states), ('actions', self.actions)):
                for name, domain in option[key].items():
                    if self.owner.allocation['kind'] == 'per_block':
                        if name in axes and axes[name] != domain:
                            self.compatible = False
                    axes[name] = domain

    def cost(self):
        if not self.compatible:
            return None
        owner = self.owner
        states, actions = (owner.global_axes if owner.allocation['kind'] == 'global_union'
                           else (self.states, self.actions))
        ns, na = owner.product(states), owner.product(actions)
        return {'work': (len(self.members) + self.crossings * owner.boundary_unit) * ns * na,
                'storage': len(self.members) * ns,
                'clarity': owner.rules['block'] + owner.rules['characteristic'] * len(self.characteristics)
                           + owner.rules['dispatch'] * (len(self.regimes) - 1)}


class Search:
    def __init__(self, analysis, limits=None, checkpoint=None):
        self.started = time.perf_counter()
        self.analysis = analysis
        self.decl = analysis['declarations']
        self.cells = {c['id']: c for c in self.decl['cells']}
        self.cell_order = tuple(sorted(self.cells))
        self.cell_index = {c: i for i, c in enumerate(self.cell_order)}
        self.encoded_cell_ids = tuple(canonical(c) for c in self.cell_order)
        self.domains = dict(self.decl['domains']) | analysis.get('derived_domains', {})
        self.options = {c: {o['id']: o for o in v['options']}
                        for c, v in analysis['interfaces'].items()}
        self.limits = {'seconds': 900, 'segments': 2_000_000,
                       'groupings': 100_000, 'frontier_points': 5000} | (limits or {})
        for name in ('seconds', 'segments', 'groupings', 'frontier_points'):
            if self.limits[name] < 0:
                raise ValueError(f'{name} budget must be nonnegative')
        self.counts = {'segments': 0, 'groupings': 0}
        self.records = []
        self.grouping_records = []
        self.signature_records = []
        self.analysis_digest = digest(analysis)
        self.component_reductions = []
        self.interval_cache = {}
        self.checkpoint = checkpoint
        self.allocation = analysis.get('allocation_scenario', self.decl['allocation'])
        if self.allocation.get('kind') not in ('per_block', 'global_union'):
            raise ValueError('An explicit allocation scenario is required to compute costs')
        self.boundary_unit = self.allocation.get('boundary_unit', 1)
        if type(self.boundary_unit) is not int or self.boundary_unit < 0:
            raise ValueError('boundary_unit must be a nonnegative integer')
        self.roles = {c: self.protected_key(cell) for c, cell in self.cells.items()}
        self.outgoing, self.incoming = defaultdict(set), defaultdict(set)
        for relation in self.decl['relations']:
            if relation.get('support') == 'zero':
                continue
            source = relation['source']
            for target in relation.get('targets', []):
                self.outgoing[source].add(target)
                self.incoming[target].add(source)
        self.characteristics = {
            (c, o): frozenset(canonical(v) for v in option['state_projection'].values()
                             if v['kind'] == 'characteristic')
            for c, options in self.options.items() for o, option in options.items()}
        self.global_axes = None
        if self.allocation.get('kind') == 'global_union':
            for axis_kind in ('states', 'actions'):
                if self.allocation.get(axis_kind) is None and self.union_axes(c[axis_kind] for c in self.cells.values()) is None:
                    raise ValueError('Global allocation has incompatible original ' + axis_kind +
                                     ' domain identities; supply the actual backend-wide axis/domain mapping. No grid union was invented.')
            self.global_axes = tuple(self.allocation.get(axis_kind) if self.allocation.get(axis_kind) is not None
                else self.union_axes(c[axis_kind] for c in self.cells.values())
                for axis_kind in ('states', 'actions'))
        self.rules = {'block': 4, 'characteristic': 2, 'dispatch': 1} | analysis.get('clarity_weights', {})
        if any(type(v) is not int or v < 0 for v in self.rules.values()) or self.rules['block'] == 0:
            raise ValueError('Clarity costs must be nonnegative integers with positive block cost')

    def tick(self, kind=None, amount=1):
        if time.perf_counter() - self.started >= self.limits['seconds']:
            raise Exhausted('seconds')
        if kind:
            self.counts[kind] += amount
            if self.counts[kind] > self.limits[kind]:
                raise Exhausted(kind)

    def prune(self, plans):
        buffered = []
        cap = self.limits['frontier_points']
        for plan in plans:
            buffered.append(plan)
            if len(buffered) > max(1, 2 * min(cap, 5000)):
                self.tick()
                buffered = pareto(buffered)
                if len(buffered) > cap:
                    raise Exhausted('frontier_points')
        result = pareto(buffered)
        if len(result) > self.limits['frontier_points']:
            raise Exhausted('frontier_points')
        return result

    def protected_key(self, cell):
        if hasattr(self, 'roles') and cell['id'] in self.roles:
            return self.roles[cell['id']]
        # Unknown information contracts cannot become guessed common labels.
        role = cell.get('role')
        role_key = role if role is not None else ({'original_contexts_preserved_by_dispatch': True, 'decision_objective': cell.get('decision_objective')}
            if self.decl.get('analysis_scope') == 'pylcm-structural-graph/v1'
            else {'unresolved_original_role': cell['regime']})
        terminal_context = None if self.decl.get('analysis_scope') == 'pylcm-structural-graph/v1' else bool(cell.get('terminal'))
        return canonical([role_key, cell.get('ownership'), cell.get('stage'), terminal_context,
                          cell.get('declaration_type'), cell.get('structural_roles', {}), cell.get('solver_identity')])

    def components(self):
        groups = defaultdict(lambda: defaultdict(list))
        for cell in self.cells.values():
            groups[self.protected_key(cell)][cell['regime']].append(cell['id'])
        return [dict(sorted(v.items())) for _, v in sorted(groups.items())]

    def partitions(self, labels):
        # Canonical restricted-growth enumeration gives every set partition once.
        blocks = []
        def visit(i):
            if i == len(labels):
                self.tick('groupings')
                yield [b[:] for b in blocks]
                return
            label = labels[i]
            for block in blocks:
                block.append(label)
                yield from visit(i + 1)
                block.pop()
            blocks.append([label])
            yield from visit(i + 1)
            blocks.pop()
        yield from visit(0)

    def product(self, axes, *, solve=True):
        size = 1
        for name, domain_id in sorted(axes.items()):
            domain = self.domains[domain_id]
            if solve and domain.get('solve_axis') is False:
                continue
            n = domain.get('size')
            if type(n) is not int or n <= 0:
                raise ValueError(f'Missing original cardinality for {name}/{domain_id}')
            size *= n
        return size

    @staticmethod
    def union_axes(items):
        result = {}
        for axes in items:
            for name, domain_id in axes.items():
                if name in result and result[name] != domain_id:
                    return None
                result[name] = domain_id
        return result

    def block(self, ids, policy, *, enforce_clarity=True):
        # `enforce_clarity` is retained as a compatibility keyword. Clarity is a
        # cost/recommendation preference, never a semantic admission predicate.
        if not ids or len(ids) != len(set(ids)):
            return None
        cells = [self.cells[c] for c in ids]
        options = [self.options[c][policy[c]] for c in ids]
        if len({self.protected_key(c) for c in cells}) != 1:
            return None
        allocation_kind = self.allocation.get('kind')
        if allocation_kind not in ('per_block', 'global_union'):
            raise ValueError('An explicit allocation scenario is required to compute costs')
        if allocation_kind == 'global_union':
            states, actions = self.global_axes
        else:
            states = self.union_axes(o['states'] for o in options)
            actions = self.union_axes(o['actions'] for o in options)
        if states is None or actions is None:
            return None
        ns, na = self.product(states), self.product(actions)
        row_count = ns * na
        members = set(ids)
        crossings = {(c, t) for c in members for t in self.outgoing.get(c, ()) if t not in members}
        boundary_unit = self.allocation.get('boundary_unit', 1)
        if type(boundary_unit) is not int or boundary_unit < 0:
            raise ValueError('boundary_unit must be a nonnegative integer')
        enumeration = len(ids) * row_count
        boundary = len(crossings) * row_count * boundary_unit
        characteristics = set().union(*(self.characteristics[(c, policy[c])] for c in ids))
        dispatch = max(0, len({c['regime'] for c in cells}) - 1)
        clarity = self.rules['block'] + self.rules['characteristic'] * len(characteristics) + self.rules['dispatch'] * dispatch
        return {'cells': sorted(ids), 'option_by_cell': {c: policy[c] for c in sorted(ids)},
                'periods': sorted({c['period'] for c in cells}),
                'ages': [self.cells[c]['age'] for c in sorted(ids, key=lambda c: (self.cells[c]['period'], c))],
                'original_regimes': sorted({c['regime'] for c in cells}),
                'state_axes': states, 'action_axes': actions,
                'cost': {'work': enumeration + boundary, 'storage': len(ids) * ns, 'clarity': clarity},
                'cost_detail': {'enumeration': enumeration, 'boundary': boundary,
                                'boundary_incidences': len(crossings), 'state_rows_per_atom': ns,
                                'action_rows_per_atom': na, 'dispatch_clauses': dispatch,
                                'characteristic_count': len(characteristics)},
                'allocation_kind': allocation_kind, 'peak_live_storage': None}

    def interval_frontier(self, ids, policy):
        ids = sorted(ids)
        cache_key = (tuple(ids), tuple(policy[c] for c in ids))
        if cache_key in self.interval_cache:
            return self.interval_cache[cache_key]
        dates = defaultdict(list)
        for c in ids:
            dates[self.cells[c]['period']].append(c)
        buckets = [sorted(v) for _, v in sorted(dates.items())]
        group_record = {'buckets': buckets, 'option_by_cell': {c: policy[c] for c in ids}}
        group_id = digest(group_record)
        n = len(buckets)
        if (n>1 and self.decl.get('analysis_scope')=='pylcm-structural-graph/v1'
                and self.allocation['kind']=='per_block' and self.rules['block']>self.rules['dispatch']):
            signatures=[]
            for bucket in buckets:
                self.tick()
                options=[self.options[c][policy[c]] for c in bucket]
                states=self.union_axes(o['states'] for o in options)
                actions=self.union_axes(o['actions'] for o in options)
                roles={self.roles[c] for c in bucket}
                if states is None or actions is None or len(roles)!=1:break
                characteristics=frozenset(v for c in bucket for v in self.characteristics[c,policy[c]])
                signatures.append((states,actions,roles,characteristics))
            else:
                if all(signature==signatures[0] for signature in signatures):
                    # Every interval has the same axes/characteristics. Joining
                    # cuts cannot add crossings and strictly reduces clarity.
                    block=self.block(ids,policy)
                    if block is not None:
                        self.tick('segments')
                        points=[{'cost':block['cost'],'blocks':(_BlockRef.from_materialized(self,block,policy),)}]
                        self.records.append({'group_id':group_id,**group_record,'reduction':{
                            'rule':'constant-date-bucket-union/v1','cost':block['cost']}})
                        self.interval_cache[cache_key]=points
                        return points
                if (all(signature==signatures[0] for signature in signatures[:-1])
                        and signatures[-1][2]==signatures[0][2]):
                    # Every block before the last lies in the constant prefix.
                    # Joining those blocks dominates all plans with >1 cut.
                    prefixes={}
                    growing=_GrowingInterval(self,policy)
                    for stop in range(1,n):
                        growing.extend(buckets[stop-1]);self.tick('segments')
                        cost=growing.cost()
                        if cost is not None:
                            prefixes[stop]={'cost':cost,'blocks':(_BlockRef(self,buckets,policy,0,stop,cost,growing.mask),)}
                    growing=_GrowingInterval(self,policy)
                    candidates=[]
                    for start in reversed(range(n)):
                        growing.extend(buckets[start]);self.tick('segments')
                        cost=growing.cost()
                        if cost is None:continue
                        tail={'cost':cost,'blocks':(_BlockRef(self,buckets,policy,start,n,cost,growing.mask),)}
                        if start==0:candidates.append(tail)
                        elif start in prefixes:candidates.append(add(prefixes[start],tail))
                    points=self.prune(candidates)
                    self.records.append({'group_id':group_id,**group_record,'reduction':{
                        'rule':'constant-prefix-single-cut/v1','frontier':[point['cost'] for point in points]}})
                    self.interval_cache[cache_key]=points
                    return points
        suffix = [[] for _ in range(n + 1)]
        suffix[n] = [zero()]
        for i in range(n - 1, -1, -1):
            possibilities = []
            growing = _GrowingInterval(self, policy)
            for j in range(i + 1, n + 1):
                self.tick('segments')
                growing.extend(buckets[j - 1])
                cost = growing.cost()
                if cost is None:
                    # An owner/domain conflict cannot be repaired by extending
                    # the interval. Cover all remaining rejected endpoints.
                    self.tick('segments',n-j)
                    break
                if cost is not None:
                    head = {'cost': cost, 'blocks': (_BlockRef(self, buckets, policy, i, j, cost, growing.mask),)}
                    possibilities.extend(add(head, tail) for tail in suffix[j])
                    if len(possibilities) > max(1, 2 * min(self.limits['frontier_points'], 5000)):
                        possibilities = self.prune(possibilities)
            suffix[i] = self.prune(possibilities)
        self.records.append({'group_id': group_id, **group_record, 'suffix_frontiers':
                             [[p['cost'] for p in points] for points in suffix]})
        self.interval_cache[cache_key] = suffix[0]
        return suffix[0]

    @staticmethod
    def materialize(plans):
        return [{**{k: v for k, v in p.items() if k not in ('blocks', '_tie_key')},
                 'blocks': [b.materialize() if isinstance(b, _BlockRef) else b for b in p['blocks']]}
                for p in plans]

    def layout_partition_frontier(self, component, policy):
        """Contract interchangeable single-date labels, retaining every cost vector.

        With no internal edges, common action axes and no characteristics, each
        cell's work/storage both increase with its block's state-row count. Move
        every equal-layout cell to its cheapest already containing block: axes
        do not grow, no cost increases and empty blocks can disappear. Thus an
        optimum exists with each layout unsplit, even for arbitrary date order.
        """
        ids = sorted(c for cells in component.values() for c in cells)
        members = set(ids)
        if (self.decl.get('analysis_scope') != 'pylcm-structural-graph/v1'
                or self.rules['block'] <= self.rules['dispatch']
                or any(len(cells) != 1 for cells in component.values())
                or any(self.outgoing[c] & members for c in ids)):
            return None
        options = {c: self.options[c][policy[c]] for c in ids}
        if (any(o['actions'] != options[ids[0]]['actions'] for o in options.values())
                or any(self.characteristics[(c, policy[c])] for c in ids)):
            return None
        layouts = defaultdict(list)
        for c in ids: layouts[canonical(options[c]['states'])].append(c)
        candidates = []
        for partition in self.partitions(sorted(layouts)):
            plan = zero()
            for group in partition:
                members = sorted(c for key in group for c in layouts[key])
                block = self.block(members, policy)
                if block is None:
                    plan = None; break
                plan = add(plan, {'cost': block['cost'],
                    'blocks': [_BlockRef.from_materialized(self, block, policy)]})
            if plan is not None: candidates.append(plan)
            if len(candidates) > max(1, 2*min(self.limits['frontier_points'], 5000)):
                candidates = self.prune(candidates)
        self.component_reductions.append({'rule': 'single-date-layout-contraction/v1',
            'layouts': [layouts[key] for key in sorted(layouts)],
            'option_by_cell': {c: policy[c] for c in ids}})
        return self.prune(candidates)

    def domain_components(self, component, policy):
        """A valid block cannot span disconnected layout-compatibility classes."""
        if (self.decl.get('analysis_scope') != 'pylcm-structural-graph/v1'
                or self.allocation.get('kind') != 'per_block'):
            return [component]
        layouts, labels = {}, {}
        for label, ids in component.items():
            labels[label] = []
            for cid in ids:
                option = self.options[cid][policy[cid]]
                key = canonical([option['states'], option['actions']])
                layouts[key] = option
                labels[label].append(key)
        parent = {key: key for key in layouts}
        def find(key):
            while parent[key] != key:
                parent[key] = parent[parent[key]]; key = parent[key]
            return key
        def join(a, b): parent[find(a)] = find(b)
        for keys in labels.values():
            for key in keys[1:]: join(keys[0], key)
        keys = sorted(layouts)
        for i, left in enumerate(keys):
            self.tick()
            for right in keys[i+1:]:
                if all(self.union_axes((layouts[left][kind], layouts[right][kind])) is not None
                       for kind in ('states', 'actions')):
                    join(left, right)
        groups = defaultdict(dict)
        for label in sorted(component): groups[find(labels[label][0])][label] = component[label]
        parts = sorted(groups.values(), key=lambda part: tuple(part))
        if len(parts)>1:
            self.component_reductions.append({'rule':'incompatible-domain-components/v1',
                'components':[list(part) for part in parts],
                'option_by_cell':{c:policy[c] for c in sorted(c for ids in component.values() for c in ids)}})
        return parts

    def component_frontier(self, component, policy):
        parts = self.domain_components(component, policy)
        if len(parts)>1:
            combined=[zero()]
            for part in parts:
                points=self.component_frontier(part,policy)
                combined=self.prune(add(a,b) for a in combined for b in points)
            return combined
        ids = sorted(c for cells in component.values() for c in cells)
        options = [self.options[c][policy[c]] for c in ids]
        if (self.decl.get('analysis_scope') == 'pylcm-structural-graph/v1'
                and self.rules['block'] > self.rules['dispatch']
                and all((o['states'], o['actions']) == (options[0]['states'], options[0]['actions']) for o in options)):
            # Equal layouts make enumeration/storage additive. Coalescing removes
            # crossings and saves strictly positive block-minus-dispatch clarity.
            block = self.block(ids, policy)
            if block is not None:
                self.tick('groupings')
                self.component_reductions.append({'rule': 'homogeneous-layout-dominance/v1',
                    'cells': ids, 'option_by_cell': {c: policy[c] for c in ids}})
                return [add(zero(), {'cost': block['cost'],
                    'blocks': [_BlockRef.from_materialized(self, block, policy)]})]
        contracted = self.layout_partition_frontier(component, policy)
        if contracted is not None:
            return contracted
        if (len(component)>1 and all(len(ids)==1 for ids in component.values())
                and self.allocation['kind']=='per_block' and self.rules['block']>self.rules['dispatch']):
            return self.signature_frontier(component,policy)
        return self.subset_frontier(component, policy)

    def signature_instance(self,component,policy):
        ids=sorted(c for members in component.values() for c in members)
        domain_ids={domain for c in ids for kind in ('states','actions') for domain in self.options[c][policy[c]][kind].values()}
        instance={'cells':[{'id':c,'label':self.cells[c]['regime'],'period':self.cells[c]['period'],
                           'age':self.cells[c].get('age'),'context':self.roles[c],
                           'states':self.options[c][policy[c]]['states'],'actions':self.options[c][policy[c]]['actions'],
                           'characteristics':sorted(self.characteristics[c,policy[c]])} for c in ids],
                  'domains':{d:{'size':self.domains[d]['size'],'solve_axis':self.domains[d].get('solve_axis',True),
                                'source_domain_sha256':digest(self.domains[d])} for d in sorted(domain_ids)},
                  'pairs':sorted((c,t) for c in ids for t in self.outgoing[c]),
                  'allocation':'per_block','boundary_unit':self.boundary_unit,'clarity_weights':self.rules,
                  'bindings':{'analysis_sha256':self.analysis_digest,'policy_sha256':digest(policy),
                              'option_by_cell':{c:policy[c] for c in ids}}}
        return instance

    def signature_frontier(self,component,policy):
        from .signature_model import build, cell_signature, merged
        instance = self.signature_instance(component, policy)
        try:
            model=build(instance,check=self.tick)
        except RuntimeError as exc:
            if isinstance(exc,Exhausted):raise
            raise Exhausted(str(exc)) from exc
        choices=[[k for k,bin in enumerate(model['bins']) if merged(cell_signature(cell),bin)==bin]
                 for cell in instance['cells']]
        combinations=1
        for offered in choices:combinations*=len(offered)
        self.signature_records.append({'schema':'regime-signature-component/v1',
            'status':'PARETO_FAMILY_REPRESENTED','instance':instance,'model':model,
            'model_sha256':digest(model),'assignment_count':combinations})
        if combinations>self.limits['groupings']-self.counts['groupings']:
            raise Exhausted('signature assignment cover budget')
        candidates=[]
        for assignment in product(*choices):
            self.tick('groupings')
            groups=defaultdict(list)
            for cell,bin in zip(instance['cells'],assignment):groups[bin].append(cell)
            for bin,members in groups.items():
                union=((),(),())
                for cell in members:union=merged(union,cell_signature(cell))
                if union!=model['bins'][bin]:break
            else:
                plan=zero()
                for members in sorted(groups.values(),key=lambda members:min(c['label'] for c in members)):
                    block=self.block([c['id'] for c in members],policy)
                    if block is None:raise ValueError('Signature assignment produced an inadmissible block')
                    plan=add(plan,{'cost':block['cost'],'blocks':[_BlockRef.from_materialized(self,block,policy)]})
                candidates=self.prune(chain(candidates,[plan]))
        return candidates

    def subset_frontier(self, component, policy):
        """Every grouping has one group containing the least remaining label."""
        labels = tuple(sorted(component))
        if (1 << (len(labels)-1)) > self.limits['groupings']-self.counts['groupings']:
            raise Exhausted('groupings')
        memo = {0: [zero()]}
        def visit(mask):
            if mask in memo:
                return memo[mask]
            pivot = mask & -mask
            rest = mask ^ pivot
            subset = rest
            candidates = []
            while True:
                self.tick('groupings')
                group = subset | pivot
                ids = [cell for index,label in enumerate(labels) if group & (1 << index)
                       for cell in component[label]]
                head = self.interval_frontier(ids, policy)
                tail = visit(mask ^ group)
                candidates = self.prune(chain(candidates,(add(a,b) for a in head for b in tail)))
                if subset == 0:
                    break
                subset = (subset-1) & rest
            memo[mask] = candidates
            return candidates
        result = visit((1 << len(labels))-1)
        self.grouping_records.append({'rule':'label-subset-recurrence/v1',
            'cells_by_label':{label:sorted(component[label]) for label in labels},
            'option_by_cell':{cell:policy[cell] for cell in sorted(c for cells in component.values() for c in cells)},
            'states':[{'remaining':[label for i,label in enumerate(labels) if mask & (1 << i)],
                       'frontier':[p['cost'] for p in memo[mask]]} for mask in sorted(memo)]})
        return result

    def run(self):
        components = self.components()
        policies = self.analysis['projection_policies']
        complete_candidates = []
        # An admitted all-original-label reference is available before the search.
        for policy_record in policies:
            policy = policy_record['option_by_cell']
            seed = zero()
            for component in components:
                for ids in component.values():
                    block = self.block(ids, policy)
                    if block is None:
                        seed = None
                        break
                    seed = add(seed, {'cost': block['cost'], 'blocks': [_BlockRef.from_materialized(self, block, policy)]})
                if seed is None:
                    break
            if seed is not None:
                seed['projection_policy'] = policy_record['id']
                complete_candidates.append(seed)
        complete = True
        reason = None
        if self.checkpoint:
            self.checkpoint({'status': 'INCOMPLETE', 'frontier': self.materialize(pareto(complete_candidates)),
                             'analysis_sha256': digest(self.analysis),
                             'lower_bound': dict.fromkeys(METRICS, 0),
                             'scope': 'Valid complete candidate plans; global completeness not checked'})
        try:
            for policy_record in policies:
                self.tick()
                policy = policy_record['option_by_cell']
                combined = [zero()]
                for component in components:
                    cf = self.component_frontier(component, policy)
                    combined = self.prune(add(a, b) for a in combined for b in cf)
                for p in combined:
                    p['projection_policy'] = policy_record['id']
                complete_candidates.extend(combined)
                complete_candidates = self.prune(complete_candidates)
                if self.checkpoint:
                    self.checkpoint({'status': 'INCOMPLETE', 'frontier': self.materialize(complete_candidates),
                                     'analysis_sha256': digest(self.analysis),
                                     'lower_bound': dict.fromkeys(METRICS, 0),
                                     'scope': 'Valid complete candidate plans; global completeness not checked'})
        except Exhausted as exc:
            complete = False
            reason = str(exc)
        frontier = self.materialize(pareto(complete_candidates))
        for i, point in enumerate(frontier):
            point['id'] = f'G{i:04d}'
        records = sorted(self.records, key=lambda row:row['group_id'])
        grouping_records = sorted(self.grouping_records, key=canonical)
        signature_records = sorted(self.signature_records,key=canonical)
        return {'schema': 'regime-frontier/v5', 'status': 'COMPLETE' if complete else 'INCOMPLETE',
                'frontier': frontier,
                'certificate': {'analysis_sha256': digest(self.analysis),
                    'family_sha256': digest([records,grouping_records,signature_records]), 'family_encoding': PREFIX_ENCODING, 'complete': complete,
                    'family': 'all protected-role-compatible original-regime groupings; every interval of exact date buckets; every offered coherent categorical policy; source certification checked separately',
                    'component_regimes': [list(c) for c in components],
                    'component_reductions': self.component_reductions,
                    'projection_policies': [p['id'] for p in policies],
                    'search_counts': self.counts, 'interval_records': records, 'grouping_records':grouping_records,
                    'signature_records':signature_records,
                    'lower_bound': dict.fromkeys(METRICS, 0) if not complete else None,
                    'frontier_scope': 'all candidates in the declared finite family' if complete else 'nondominated among examined complete candidates'},
                'elapsed_seconds': time.perf_counter() - self.started,
                'exhaustion_reason': reason, 'allocation_contract': self.allocation,
                'clarity_weights': self.rules,
                'endpoint_clarity_policy': ENDPOINT_CLARITY_POLICY}


def _final_only_cleanup(search: Search, point: dict) -> bool:
    """Recognize an optional last-date omission within one existing regime.

    This conservative structural preference establishes no economic equivalence.
    Existing nodes, cross-regime groupings and changed retained interfaces are
    outside its scope; semantic checks remain responsible for every proposal.
    """
    native = search.decl.get('analysis_scope') == 'pylcm-structural-graph/v1'
    decision_labels = {cell['regime'] for cell in search.cells.values() if not cell.get('terminal')}
    decision_cells = [cell for cell in search.cells.values()
                      if cell['regime'] in decision_labels] if native else [cell for cell in search.cells.values() if not cell.get('terminal')]
    if not decision_cells:
        return False
    final = max(cell['period'] for cell in decision_cells)
    for block in point['blocks']:
        cells = [search.cells[c] for c in block['cells']]
        labels = {cell['regime'] for cell in cells}
        if len(labels) != 1 or any((cell.get('terminal') and not native) or cell['period'] != final for cell in cells):
            continue
        label = next(iter(labels))
        earlier = [cell['period'] for cell in decision_cells if cell['regime'] == label and cell['period'] < final]
        if not earlier:
            continue  # The original node already covers only the last date.
        previous_date = max(earlier)
        for previous in point['blocks']:
            prior = [search.cells[c] for c in previous['cells']]
            if ({cell['regime'] for cell in prior} != labels
                    or max(cell['period'] for cell in prior) != previous_date
                    or len({search.roles[cell['id']] for cell in cells + prior}) != 1):
                continue
            selected = [search.options[cell['id']][block['option_by_cell'][cell['id']]] for cell in cells]
            preceding = [search.options[cell['id']][previous['option_by_cell'][cell['id']]]
                         for cell in prior if cell['period'] == previous_date]
            removed = {}
            valid = True
            for kind, projection, removal in (('states', 'state_projection', 'drop'),
                                              ('actions', 'action_reconstruction', 'constant')):
                before = search.union_axes(search.options[cell['id']][previous['option_by_cell'][cell['id']]][kind] for cell in prior)
                after = search.union_axes(option[kind] for option in selected)
                if before is None or after is None or any(before.get(name) != domain for name, domain in after.items()):
                    valid = False
                    break
                removed[kind] = set(before) - set(after)
                for cell, option in zip(cells, selected):
                    if any(name not in cell[kind] or option[projection].get(name, {}).get('kind') != removal
                           for name in removed[kind]):
                        valid = False
                    for old in preceding:
                        if any(name not in old[kind] for name in removed[kind]):
                            valid = False
                        if any(option[projection].get(name) != old[projection].get(name) for name in after):
                            valid = False
            if valid and any(removed.values()):
                return True
    return False


def recommend(result, reference=None, weights=None, *, deferred_graph_ids=(), allow_final_only_split=False):
    if type(allow_final_only_split) is not bool:
        raise ValueError('allow_final_only_split must be a boolean')
    weights = dict.fromkeys(METRICS, 1) | (weights or {})
    weights = {k: Fraction(str(weights[k])) for k in METRICS}
    if any(v <= 0 for v in weights.values()):
        raise ValueError('Recommendation weights must be strictly positive')
    points = result['frontier']
    if not points:
        return None
    if reference is None:
        reference = {k: max(p['cost'][k] for p in points) or 1 for k in METRICS}
    scales = {k: max(1, int(reference[k])) for k in METRICS}
    score = lambda p: sum(weights[k] * Fraction(p['cost'][k], scales[k]) for k in METRICS)
    deferred = sorted(deferred_graph_ids)
    eligible = points if allow_final_only_split else [p for p in points if p['id'] not in deferred]
    winner = min(eligible, key=lambda p: (score(p), vector(p['cost']), plan_key(p))) if eligible else None
    return {'graph_id': winner['id'] if winner else None, 'score': str(score(winner)) if winner else None,
            'weights': {k: str(v) for k, v in weights.items()}, 'reference_scales': scales,
            'endpoint_preference': {'allow_final_only_split': allow_final_only_split,
                'deferred_graph_ids': deferred,
                'reason': ENDPOINT_CLARITY_POLICY},
            'rule': 'minimum positive weighted reference-normalized cost among preference-eligible frontier points; exact rational score; deterministic cost/layout ties; retain the original if none is eligible'}


def search_frontier(analysis, *, limits=None, recommendation=None, checkpoint=None):
    search = Search(analysis, limits, checkpoint)
    result = search.run()
    recommendation = recommendation or {}
    deferred = [point['id'] for point in result['frontier'] if _final_only_cleanup(search, point)]
    result['recommendation'] = recommend(result, recommendation.get('reference'), recommendation.get('weights'),
        deferred_graph_ids=deferred, allow_final_only_split=recommendation.get('allow_final_only_split', False))
    return result
