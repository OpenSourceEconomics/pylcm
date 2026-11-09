"""Checked-candidate layout search; no complete-family or optimality claim.

The scalar interval subproblem is exact for one fixed grouping and one offered
coherent policy. Sparse grouping moves and a finite number of sweeps are a
heuristic. Source semantics are established separately by proposal_decision.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from fractions import Fraction
from math import lcm
import time

from .frontier import (METRICS, Search, Exhausted, _BlockRef, _GrowingInterval,
                       _final_only_cleanup, canonical, digest, dominates,
                       pareto, plan_key, vector)

SCHEMA = 'regime-heuristic-frontier/v1'
ENDPOINT_POLICY = ('Heuristic search may omit admissible layouts. By default the recommendation '
    'defers extra final-decision-date regimes used only to omit newly inactive coordinates; '
    'allow_final_only_split explicitly overrides this preference. Retained cosmetic candidates '
    'remain visible and receive the same local checks.')
RECOMMENDATION_RULE = ('fixed-reference positive weighted scalar cost; strict improvement meeting '
    'the relative threshold; deterministic cost/layout/policy ties; retain the original otherwise')


def _original_cost(search):
    """The unchanged graph is an identity action, not an invented offered policy."""
    groups = defaultdict(list)
    for cid, cell in search.cells.items():
        groups[cell['regime']].append(cid)
    total = dict.fromkeys(METRICS, 0)
    for ids in groups.values():
        if len({search.roles[c] for c in ids}) != 1:
            return None
        if search.allocation['kind'] == 'global_union':
            states, actions = search.global_axes
        else:
            states = search.union_axes(search.cells[c]['states'] for c in ids)
            actions = search.union_axes(search.cells[c]['actions'] for c in ids)
        if states is None or actions is None:
            return None
        ns, na = search.product(states), search.product(actions)
        members = set(ids)
        crossings = sum(t not in members for c in ids for t in search.outgoing[c])
        total['work'] += (len(ids) + crossings * search.boundary_unit) * ns * na
        total['storage'] += len(ids) * ns
        total['clarity'] += search.rules['block']
    return total


def _objective(original, recommendation):
    weights = {key: Fraction(str((recommendation.get('weights') or {}).get(key, 1)))
               for key in METRICS}
    if any(weight <= 0 for weight in weights.values()):
        raise ValueError('Heuristic weights must be strictly positive')
    supplied = recommendation.get('reference')
    if supplied is not None:
        scales = {key: supplied[key] for key in METRICS}
        kind = 'explicit'
    elif original is not None:
        scales = {key: max(1, original[key]) for key in METRICS}
        kind = 'original'
    else:
        scales = dict.fromkeys(METRICS, 1)
        kind = 'unit_fallback'
    if any(type(value) is not int or value <= 0 for value in scales.values()):
        raise ValueError('Fixed reference scales must be positive integers')
    improvement = Fraction(str(recommendation.get('minimum_improvement', 0)))
    if not 0 <= improvement < 1:
        raise ValueError('minimum_improvement must be a fraction in [0, 1)')
    return {'weights': {key: str(value) for key, value in weights.items()},
            'reference_scales': scales, 'reference_kind': kind,
            'minimum_improvement': str(improvement)}


def _score(cost, objective):
    return sum(Fraction(objective['weights'][key]) * Fraction(cost[key], objective['reference_scales'][key])
               for key in METRICS)


def _recommend(search, result, *, allow_final_only_split):
    """Producer ranking; heuristic_check independently reconstructs this choice."""
    objective = result['objective']
    points = result['candidates']
    deferred = sorted(point['id'] for point in points if _final_only_cleanup(search, point))
    original = result['original']['cost']
    original_score = _score(original, objective) if original is not None else None
    eligible = [point for point in points
                if allow_final_only_split or point['id'] not in deferred]
    rank = lambda point: (_score(point['cost'], objective), vector(point['cost']),
                          plan_key(point), point['projection_policy'], point['id'])
    best = min(eligible, key=rank) if eligible else None
    gain = Fraction(objective['minimum_improvement'])
    if (best is None or original_score is None or rank(best)[0] >= original_score
            or rank(best)[0] > original_score * (1 - gain)):
        best = None
    return {'graph_id': best['id'] if best is not None else 'retain-original',
            'operation': 'representation_specification' if best is not None else 'retain_original',
            'score': str(_score(best['cost'], objective)) if best is not None else
                     (str(original_score) if original_score is not None else None),
            'original_score': str(original_score) if original_score is not None else None,
            'weights': objective['weights'], 'reference_scales': objective['reference_scales'],
            'minimum_improvement': objective['minimum_improvement'],
            'endpoint_preference': {'allow_final_only_split': allow_final_only_split,
                'deferred_graph_ids': deferred, 'reason': ENDPOINT_POLICY},
            'rule': RECOMMENDATION_RULE, 'authority': 'cost_only'}


class HeuristicSearch(Search):
    """One scalar objective, sparse local moves, streaming examined Pareto set."""

    def __init__(self, analysis, *, limits=None, recommendation=None, effort=None):
        # Existing exact group/frontier/segment caps do not define this method.
        # A caller may retain its existing cooperative wall-clock boundary.
        seconds = (limits or {}).get('seconds')
        super().__init__(analysis, limits={'seconds': float('inf') if seconds is None else seconds,
            'segments': float('inf'), 'groupings': float('inf'), 'frontier_points': float('inf')})
        self.effort = {'rounds': 2, 'neighbor_degree': 2} | (effort or {})
        rounds = self.effort['rounds']
        if rounds is not None and (type(rounds) is not int or rounds < 0):
            raise ValueError('rounds must be a nonnegative integer or None')
        degree = self.effort['neighbor_degree']
        if type(degree) is not int or degree < 0:
            raise ValueError('neighbor_degree must be a nonnegative integer')
        self.recommendation = dict(recommendation or {})
        self.allow_final = self.recommendation.get('allow_final_only_split', False)
        if type(self.allow_final) is not bool:
            raise ValueError('allow_final_only_split must be a boolean')
        self.original = _original_cost(self)
        self.objective = _objective(self.original, self.recommendation)
        fractions = [Fraction(self.objective['weights'][key]) /
                     self.objective['reference_scales'][key] for key in METRICS]
        denominator = lcm(*(value.denominator for value in fractions))
        self.scalar_coefficients = tuple(int(value * denominator) for value in fractions)
        self.components_by_id = self.components()
        self.archive = []
        self.eligible_by_policy = {}
        self.evaluated = self.accepted_moves = self.group_evaluations = 0
        self.completed_policy_ids = []

    def scalar(self, cost):
        # One exact integer dot product per interval. Fractions are formed only
        # when fixing the objective and formatting the final recommendation.
        return sum(coefficient * cost[key] for coefficient, key in
                   zip(self.scalar_coefficients, METRICS))

    def members(self, group):
        component, labels = group
        return sorted(cid for label in labels for cid in self.components_by_id[component][label])

    def signature(self, group, policy):
        """Whole-label signature for sparse neighbors and a homogeneous seed."""
        ids = self.members(group)
        states = self.union_axes(self.options[c][policy[c]]['states'] for c in ids)
        actions = self.union_axes(self.options[c][policy[c]]['actions'] for c in ids)
        if states is None or actions is None:
            return ('incompatible-whole-label', canonical(group))
        characteristics = sorted(set().union(*(self.characteristics[c, policy[c]] for c in ids)))
        return ('layout', canonical([states, actions, characteristics]))

    def scalar_intervals(self, group, policy):
        """Exact min-plus interval DP; one backpointer per date, no Pareto sum.

        T(T+1)/2 sparse interval scores, O(T) DP state. The growing interval
        counts every supported directed pair, including same-period links.
        Only selected intervals are materialized into candidate output.
        """
        self.group_evaluations += 1
        self.tick()
        ids = self.members(group)
        dates = defaultdict(list)
        for cid in ids:
            dates[self.cells[cid]['period']].append(cid)
        buckets = [tuple(sorted(members)) for _, members in sorted(dates.items())]
        options = [self.options[c][policy[c]] for c in ids]
        if (self.rules['block'] > self.rules['dispatch']
                and all((option['states'], option['actions'], self.characteristics[c, policy[c]]) ==
                        (options[0]['states'], options[0]['actions'], self.characteristics[ids[0], policy[ids[0]]])
                        for c, option in zip(ids, options))):
            # Exact homogeneous dominance: no added axes, fewer crossings and
            # positive block-minus-dispatch saving. No union closure is built.
            block = self.block(ids, policy)
            if block is not None:
                self.tick('segments')
                return {'cost': block['cost'],
                        'blocks': (_BlockRef.from_materialized(self, block, policy),)}
        n = len(buckets)
        costs = [None] * (n + 1)
        following = [-1] * n
        segments = [None] * n
        costs[n] = dict.fromkeys(METRICS, 0)
        for start in range(n - 1, -1, -1):
            growing = _GrowingInterval(self, policy)
            best_rank = None
            for stop in range(start + 1, n + 1):
                self.tick('segments')
                growing.extend(buckets[stop - 1])
                cost = growing.cost()
                if cost is None:
                    break  # Conflicting identities cannot disappear on extension.
                suffix_cost = costs[stop]
                if suffix_cost is None:
                    continue
                total = {key: cost[key] + suffix_cost[key] for key in METRICS}
                # The stop index is a stable local tie, never an optimality claim
                # about ties across other groupings or omitted policies.
                rank = (self.scalar(total), vector(total), stop)
                if best_rank is None or rank < best_rank:
                    best_rank, costs[start], following[start] = rank, total, stop
                    segments[start] = _BlockRef(self, buckets, policy, start, stop, cost, growing.mask)
        if costs[0] is None:
            return None
        blocks, start = [], 0
        while start < n:
            segment = segments[start]
            if segment is None or following[start] <= start:
                raise ValueError('Incomplete interval backpointer')
            blocks.append(segment)
            start = following[start]
        return {'cost': costs[0], 'blocks': tuple(blocks)}

    @staticmethod
    def total(group_plans):
        return {key: sum(plan['cost'][key] for plan in group_plans.values()) for key in METRICS}

    def point(self, group_plans, policy_id, cost=None):
        return {'projection_policy': policy_id, 'cost': cost or self.total(group_plans),
                'blocks': tuple(block for group in sorted(group_plans) for block in group_plans[group]['blocks'])}

    def consider(self, point, *, cleanup=True):
        """Exact online pruning; checked scope is only the retained candidates.

        Dominance is transitive, so dropping a dominated examined plan cannot
        remove an eventual member of the examined Pareto set. The separately
        retained per-policy preference winners protect useful alternatives
        from cosmetic or semantically unresolved cross-policy dominators.
        """
        self.evaluated += 1
        score = self.scalar(point['cost'])
        previous = self.eligible_by_policy.get(point['projection_policy'])
        if (previous is None or
                (score, vector(point['cost']), plan_key(point), point['projection_policy']) <
                (self.scalar(previous['cost']), vector(previous['cost']),
                 plan_key(previous), previous['projection_policy'])):
            materialized = self.materialize([point])[0]
            if self.allow_final or not _final_only_cleanup(self, materialized):
                self.eligible_by_policy[point['projection_policy']] = point
        self.archive = pareto([*self.archive, point])
        if cleanup and not self.allow_final:
            # Keep a useful noncosmetic alternative even if a terminal-only
            # split dominates it in the unconstrained surrogate frontier.
            cleaned = self.without_cosmetic_splits(point)
            if cleaned is not None:
                self.consider(cleaned, cleanup=False)

    def without_cosmetic_splits(self, point):
        materialized = self.materialize([point])[0]
        blocks = materialized['blocks']
        if not _final_only_cleanup(self, materialized):
            return None
        policy = {cid: oid for block in blocks for cid, oid in block['option_by_cell'].items()}
        changed, index = False, 0
        while index + 1 < len(blocks):
            left, right = blocks[index:index + 2]
            # Only adjacent intervals within the same single original label
            # can be coalesced by this recommendation preference.
            if (left['original_regimes'] == right['original_regimes']
                    and len(left['original_regimes']) == 1
                    and _final_only_cleanup(self, {'blocks': [left, right]})):
                block = self.block([*left['cells'], *right['cells']], policy)
                if block is not None:
                    blocks[index:index + 2] = [block]
                    changed = True
                    continue
            index += 1
        if not changed:
            return None
        return {'projection_policy': point['projection_policy'], 'blocks': tuple(blocks),
                'cost': {key: sum(block['cost'][key] for block in blocks) for key in METRICS}}

    def initial_groups(self):
        return tuple((index, (label,)) for index, component in enumerate(self.components_by_id)
                     for label in sorted(component))

    def coalesced_groups(self, groups, policy):
        by_signature = defaultdict(list)
        for group in groups:
            by_signature[(group[0], self.signature(group, policy))].extend(group[1])
        return tuple(sorted((key[0], tuple(sorted(labels))) for key, labels in by_signature.items()))

    def moves(self, group_plans, policy):
        """All one-label ejections plus bounded-degree merge proposals.

        Merges use neighbors in signature order and strongest typed incidence.
        At most O(R*d) merge proposals from O(E) incidence input, with
        deterministic signature/neighbor sorting; no partition enumeration.
        Neighborhood omissions are intentional and explicitly heuristic.
        """
        groups = sorted(group_plans)
        for group in groups:
            if len(group[1]) > 1:
                for label in group[1]:
                    self.tick()
                    remainder = tuple(value for value in group[1] if value != label)
                    yield (group,), ((group[0], (label,)), (group[0], remainder))
        degree = self.effort['neighbor_degree']
        if not degree:
            return
        pairs = set()
        by_component = defaultdict(list)
        for group in groups:
            by_component[group[0]].append(group)
        for component_groups in by_component.values():
            ordered = sorted(component_groups, key=lambda group: (self.signature(group, policy), group))
            for index, left in enumerate(ordered):
                for right in ordered[index + 1:index + 1 + degree]:
                    pairs.add(tuple(sorted((left, right))))
        owner = {cid: group for group in groups for cid in self.members(group)}
        neighbors = defaultdict(Counter)
        for source in self.cell_order:
            self.tick()
            left = owner[source]
            for target in self.outgoing[source]:
                right = owner[target]
                if left != right and left[0] == right[0]:
                    neighbors[left][right] += 1
                    neighbors[right][left] += 1
        for left, counts in neighbors.items():
            for right in sorted(counts, key=lambda right: (-counts[right], right))[:degree]:
                pairs.add(tuple(sorted((left, right))))
        for left, right in sorted(pairs):
            self.tick()
            yield (left, right), ((left[0], tuple(sorted((*left[1], *right[1])))),)

    def run_heuristic(self, checkpoint=None):
        reason = 'effort_completed'
        try:
            for policy_record in sorted(self.analysis['projection_policies'], key=lambda row: row['id']):
                self.tick()
                policy, policy_id = policy_record['option_by_cell'], policy_record['id']
                if set(policy) != set(self.cells):
                    raise ValueError('Offered policy must cover every original cell')
                initial = self.initial_groups()
                group_plans = {}
                for group in initial:
                    plan = self.scalar_intervals(group, policy)
                    if plan is None:
                        break
                    group_plans[group] = plan
                else:
                    current = self.point(group_plans, policy_id)
                    self.consider(current)
                    coarse = self.coalesced_groups(initial, policy)
                    if coarse != initial:
                        coarse_plans = {}
                        for group in coarse:
                            plan = group_plans.get(group) or self.scalar_intervals(group, policy)
                            if plan is None:
                                break
                            coarse_plans[group] = plan
                        else:
                            proposal = self.point(coarse_plans, policy_id)
                            self.consider(proposal)
                            if self.scalar(proposal['cost']) < self.scalar(current['cost']):
                                group_plans, current = coarse_plans, proposal
                    rounds = 0
                    if group_plans is None:
                        raise ValueError('Missing incumbent group plans')
                    while self.effort['rounds'] is None or rounds < self.effort['rounds']:
                        winner, winning_plans = None, None
                        for removed, added in self.moves(group_plans, policy):
                            new_plans = {}
                            for group in added:
                                plan = self.scalar_intervals(group, policy)
                                if plan is None:
                                    break
                                new_plans[group] = plan
                            else:
                                # Directed boundary is charged entirely to its
                                # source block, so unaffected groups cancel exactly.
                                cost = {key: current['cost'][key]
                                        - sum(group_plans[group]['cost'][key] for group in removed)
                                        + sum(added_plan['cost'][key] for added_plan in new_plans.values()) for key in METRICS}
                                proposed_plans = {group: retained_plan for group, retained_plan in group_plans.items()
                                                  if group not in removed} | new_plans
                                proposal = self.point(proposed_plans, policy_id, cost)
                                self.consider(proposal)
                                if self.scalar(cost) >= self.scalar(current['cost']):
                                    continue
                                rank = (self.scalar(cost), vector(cost), plan_key(proposal))
                                if winner is None or rank < (self.scalar(winner['cost']),
                                                              vector(winner['cost']), plan_key(winner)):
                                    winner, winning_plans = proposal, proposed_plans
                        rounds += 1
                        if winner is None:
                            break
                        if winning_plans is None:
                            raise ValueError('A winning move requires group plans')
                        current, group_plans = winner, winning_plans
                        self.accepted_moves += 1
                    self.completed_policy_ids.append(policy_id)
                    if checkpoint is not None:
                        checkpoint(self.result('policy_checkpoint'))
        except Exhausted as exc:
            reason = 'cooperative_budget:' + str(exc)
        return self.result(reason)

    def result(self, reason):
        def identify(point):
            point = self.materialize([point])[0]
            point['id'] = 'H' + digest([point['projection_policy'],
                [(block['cells'], block['option_by_cell']) for block in point['blocks']]])[:20]
            return point
        frontier = [identify(point) for point in self.archive]
        candidates = {point['id']: point for point in frontier}
        # Policy-local preference winners survive dominance by an unresolved
        # different policy. Source decisions are made after cost optimization.
        for eligible in self.eligible_by_policy.values():
            point = identify(eligible)
            candidates[point['id']] = point
        candidates = sorted(candidates.values(), key=lambda point: point['id'])
        result = {'schema': SCHEMA, 'status': 'HEURISTIC', 'analysis_sha256': self.analysis_digest,
                  'frontier': frontier, 'candidates': candidates,
                  'original': {'operation': 'retain_original', 'cost': self.original,
                      'reason': ('raw original-regime allocation' if self.original is not None else
                          'original domains or protected context do not admit one comparable block per original regime')},
                  'objective': self.objective, 'allocation_contract': self.allocation,
                  'clarity_weights': self.rules, 'endpoint_clarity_policy': ENDPOINT_POLICY,
                  'search': {'algorithm': 'scalar-interval-sparse-split-merge/v1', 'complete': False,
                      'optimality_gap': 'unknown', 'effort': self.effort,
                      'stopping_reason': reason, 'evaluated_candidates': self.evaluated,
                      'accepted_grouping_moves': self.accepted_moves,
                      'group_evaluations': self.group_evaluations,
                      'interval_segments': self.counts['segments'],
                      'completed_policy_ids': self.completed_policy_ids,
                      'offered_policies': len(self.analysis['projection_policies']),
                      'frontier_scope': 'nondominated among examined admitted cost candidates; maintained by exact online pruning',
                      'checked_scope': 'returned candidates only; no independent cover of omitted search history'},
                  'elapsed_seconds': time.perf_counter() - self.started}
        result['recommendation'] = _recommend(self, result, allow_final_only_split=self.allow_final)
        return result


def search_heuristic(analysis, *, limits=None, recommendation=None, effort=None, checkpoint=None):
    """Search offered metadata policies; source capture/semantics remain separate.

    The unchanged original is always returned as an identity fallback, including
    when no comparable original cost or completed candidate is available.
    """
    return HeuristicSearch(analysis, limits=limits, recommendation=recommendation,
                           effort=effort).run_heuristic(checkpoint)
