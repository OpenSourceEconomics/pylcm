"""Independent finite truth-table and all-cut-mask oracle for the test families.

This module imports no production finder code. It derives coordinate dependence
by comparing finite primitive tables, propagates target-coordinate dependence,
then enumerates every interval partition. It never uses the production segment
builder, dynamic program, residual-expression machinery, or certificate verifier.
"""
from __future__ import annotations

import ast
from collections.abc import Callable, Mapping
from collections import defaultdict
from dataclasses import dataclass
from fractions import Fraction
from itertools import product
import json
import math
import operator
from typing import Any, NotRequired, TypedDict

type StateName = str
type ActionName = str


class OracleSegment(TypedDict):
    start: int
    stop: int
    ages: list[int]
    lane: tuple[int, ...]
    states: list[str]
    actions: list[str]
    rows: int
    enumeration: int
    boundary: int
    storage: int
    work: int
    external_incidences: int


class OracleDependency(TypedDict):
    age: int
    status: dict[str, int]
    live_states: list[str]
    nonconstant_actions: list[str]
    forced_actions: dict[str, int]


class OracleResult(TypedDict):
    score: tuple[int, int, tuple[int, ...]]
    segments: list[OracleSegment]
    cost: dict[str, int]
    cuts_by_lane: dict[str, list[int]]
    cut_masks_examined: NotRequired[int]
    atom_dependencies: NotRequired[list[OracleDependency]]


class OutsideOracleFamily(ValueError):
    pass


def scalar(expression, environment):
    """Small exact evaluator, used only on these test-owned scalar primitives."""
    if isinstance(expression, dict):
        if set(expression) == {"and"}:
            return all(scalar(x, environment) for x in expression["and"])
        raise OutsideOracleFamily("Opaque or unsupported expression in independent oracle")
    if isinstance(expression, list):
        return all(scalar(x, environment) for x in expression)
    if not isinstance(expression, str):
        return expression

    def visit(node):
        if isinstance(node, ast.Constant):
            return Fraction(str(node.value)) if isinstance(node.value, float) else node.value
        if isinstance(node, ast.Name):
            return environment[node.id]
        if isinstance(node, ast.IfExp):
            return visit(node.body if visit(node.test) else node.orelse)
        if isinstance(node, ast.BinOp):
            left, right = visit(node.left), visit(node.right)
            if isinstance(node.op, ast.Div):
                return Fraction(left) / Fraction(right)
            operations = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
                          ast.Pow: operator.pow, ast.Mod: operator.mod, ast.FloorDiv: operator.floordiv}
            return operations[type(node.op)](left, right)
        if isinstance(node, ast.UnaryOp):
            return {ast.Not: operator.not_, ast.USub: operator.neg,
                    ast.UAdd: operator.pos}[type(node.op)](visit(node.operand))
        if isinstance(node, ast.BoolOp):
            if isinstance(node.op, ast.And):
                return all(visit(value) for value in node.values)
            return any(visit(value) for value in node.values)
        if isinstance(node, ast.Compare):
            left = visit(node.left)
            operations: Mapping[type[ast.cmpop], Callable[..., bool]] = {ast.Eq: operator.eq, ast.NotEq: operator.ne, ast.Lt: operator.lt,
                          ast.LtE: operator.le, ast.Gt: operator.gt, ast.GtE: operator.ge,
                          ast.In: lambda x, y: x in y, ast.NotIn: lambda x, y: x not in y}  # codespell:ignore
            for op, comparator in zip(node.ops, node.comparators):
                right = visit(comparator)
                if not operations[type(op)](left, right):
                    return False
                left = right
            return True
        if isinstance(node, (ast.List, ast.Tuple)):
            return tuple(visit(value) for value in node.elts)
        if isinstance(node, ast.Subscript):
            return visit(node.value)[visit(node.slice)]
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            return {"min": min, "max": max, "abs": abs}[node.func.id](
                *(visit(value) for value in node.args))
        raise OutsideOracleFamily(f"Unsupported AST node {type(node).__name__}")

    return visit(ast.parse(expression, mode="eval").body)


def values(axis):
    return tuple(axis["values"]) if "values" in axis else tuple(range(axis["size"]))


def assignments(names, domains):
    for row in product(*(values(domains[name]) for name in names)):
        yield dict(zip(names, row))


@dataclass(frozen=True)
class Atom:
    key: tuple[int, tuple[int, ...]]
    age: int
    lane: tuple[int, ...]
    status: dict[str, int]
    program: dict[str, Any]  # noqa: PAN002 - Test oracle reads independent heterogeneous primitive-program JSON records.
    state_names: tuple[StateName, ...]
    action_names: tuple[ActionName, ...]
    rows: list[tuple[dict[str, int], dict[str, int], dict[str, int], bool]]
    forced: dict[str, int]


class FiniteOracle:
    def __init__(self, problem):
        self.problem = problem
        self.ages = tuple(problem["ages"])
        self.status_names = tuple(sorted(problem.get("status_axes", {})))
        status_domains = problem.get("status_axes", {})
        self.lanes = list(product(*(status_domains[name] for name in self.status_names)))
        self.atoms = {}
        self.edges = set()
        self.state_domains = problem.get("states", {})
        self.action_domains = problem.get("actions", {})
        for age in self.ages:
            for lane in self.lanes:
                status = dict(zip(self.status_names, lane))
                environment = {**problem.get("parameters", {}), **status, "age": age}
                selected = [p for p in problem["programs"] if scalar(p.get("when", "True"), environment)]
                if len(selected) != 1:
                    raise OutsideOracleFamily("Oracle expects exactly one original program per atom")
                p = selected[0]
                states = tuple(p.get("declared_states", tuple(self.state_domains)))
                actions = tuple(p.get("declared_actions", () if p.get("terminal") else tuple(self.action_domains)))
                rows = []
                feasible_values = {name: set() for name in actions}
                for state in assignments(states, self.state_domains):
                    state_has_action = False
                    for action in assignments(actions, self.action_domains):
                        env = {**environment, **state, **action}
                        feasible = bool(scalar(p.get("feasible", "True"), env))
                        rows.append((state, action, env, feasible))
                        if feasible:
                            state_has_action = True
                            for name in actions:
                                feasible_values[name].add(action[name])
                    if not state_has_action:
                        raise OutsideOracleFamily("No feasible action in an admitted state")
                forced = {name: next(iter(found)) for name, found in feasible_values.items()
                          if len(found) == 1}
                key = (age, lane)
                self.atoms[key] = Atom(key, age, lane, status, p, states, actions, rows, forced)
        self.live = {key: set() for key in self.atoms}
        self._derive_liveness()

    def _target(self, atom, edge, env, is_read):
        offset = edge.get("offset", 0 if is_read else 1)
        position = self.ages.index(atom.age) + offset
        if position < 0 or position >= len(self.ages):
            return None
        status = dict(atom.status)
        status.update({name: scalar(expr, env) for name, expr in edge.get("status_updates", {}).items()})
        lane = tuple(status[name] for name in self.status_names)
        key = (self.ages[position], lane)
        if key not in self.atoms:
            raise OutsideOracleFamily("Transition exits the represented status family")
        target = self.atoms[key]
        if edge.get("target") and edge["target"] != target.program["id"]:
            raise OutsideOracleFamily("Explicit target does not own the target atom")
        return target

    def _signature(self, atom, row, live):
        _, _, env, feasible = row
        if not feasible:
            return (False,)
        p = atom.program
        payoff = p.get("terminal_payoff", "0") if p.get("terminal") else p.get("reward", "0")
        direct = (True, scalar(payoff, env), scalar(p.get("discount", "1"), env),
                  tuple((name, scalar(expr, env)) for name, expr in sorted(p.get("outputs", {}).items())))
        laws = {phase: defaultdict(Fraction) for phase in ("solve", "simulation", "read")}
        for is_read, source_edges in ((False, p.get("transitions", [])), (True, p.get("reads", []))):
            for edge in source_edges:
                weight = Fraction(scalar(edge.get("weight" if is_read else "probability", "1"), env))
                if weight == 0:
                    continue
                if weight < 0:
                    raise OutsideOracleFamily("Negative weight in independent finite fixture")
                target = self._target(atom, edge, env, is_read)
                if target is None:
                    continue
                self.edges.add((atom.key, target.key))
                mapped = tuple((name, scalar(edge.get("updates", {}).get(name, name), env))
                               for name in sorted(live[target.key]))
                destination = (target.key, mapped)
                phase = edge.get("phase", "both")
                phases = ("read",) if is_read else (("solve", "simulation") if phase == "both" else (phase,))
                for name in phases:
                    laws[name][destination] += weight
        return direct + tuple((name, tuple(sorted(table.items(), key=repr))) for name, table in laws.items())

    def _derive_liveness(self):
        # This is an independent semantic dependency fixed point over truth tables.
        # It does not simplify or inspect production expression dependency trees.
        while True:
            next_live = {key: set(found) for key, found in self.live.items()}
            for key, atom in self.atoms.items():
                signatures = [self._signature(atom, row, self.live) for row in atom.rows]
                for name in atom.state_names:
                    buckets = {}
                    for row, signature in zip(atom.rows, signatures):
                        state, action = row[:2]
                        others = tuple((other, state[other]) for other in atom.state_names if other != name)
                        fixed_action = tuple((other, action[other]) for other in atom.action_names)
                        bucket = (others, fixed_action)
                        if bucket in buckets and buckets[bucket] != signature:
                            next_live[key].add(name)
                            break
                        buckets[bucket] = signature
            if next_live == self.live:
                return
            self.live = next_live

    def role(self, atom):
        return json.dumps({"role": atom.program.get("role", {}),
                           "routing_owner": atom.program.get("routing_owner"),
                           "terminal": bool(atom.program.get("terminal"))}, sort_keys=True)

    def segment(self, lane, start, stop, clarity=True):
        """Compute one interval from independent finite primitive tables."""
        atoms = [self.atoms[(age, lane)] for age in self.ages[start:stop]]
        if len({self.role(atom) for atom in atoms}) != 1:
            return None
        states = set().union(*(self.live[atom.key] for atom in atoms))
        actions = set().union(*(set(atom.action_names) - set(atom.forced) for atom in atoms))
        if any(atom.program.get("allocation") == "global_union" for atom in atoms):
            states.update(self.state_domains)
            actions.update(self.action_domains)
        imp = self.problem.get("implementation", {})
        if clarity and len(atoms) == 1 and not atoms[0].program.get("terminal"):
            decision = [self.atoms[(age, lane)] for age in self.ages
                        if not self.atoms[(age, lane)].program.get("terminal")]
            endpoint = decision[-1]
            required_boundary = len(decision) > 1 and self.role(decision[-2]) != self.role(endpoint)
            if atoms[0].key == endpoint.key and not required_boundary:
                protected = imp.get("endpoint_saving_actions", ["saving", "savings", "assets_end", "assets_next"])
                actions.update(set(endpoint.forced) & set(protected))
        state_count = math.prod(len(values(self.state_domains[name])) for name in states
                                if self.state_domains[name].get("solve_axis", True))
        action_count = math.prod(len(values(self.action_domains[name])) for name in actions)
        if state_count > imp.get("max_block_state_entries", math.inf):
            return None
        if action_count > imp.get("max_block_action_entries", math.inf):
            return None
        rows = state_count * action_count
        keys = {atom.key for atom in atoms}
        external = {(source, target) for source, target in self.edges if source in keys and target not in keys}
        enumeration = len(atoms) * rows
        boundary = rows * len(external) * imp.get("boundary_unit", 1)
        return {"start": start, "stop": stop, "ages": list(self.ages[start:stop]),
                "lane": lane, "states": sorted(states), "actions": sorted(actions),
                "rows": rows, "enumeration": enumeration, "boundary": boundary,
                "storage": len(atoms) * state_count, "work": enumeration + boundary,
                "external_incidences": len(external)}

    def exhaustive(self, clarity=True):
        """Enumerate every cut mask, without a shortest-path or DP recurrence."""
        lane_options = []
        examined = 0
        for lane in self.lanes:
            choices = []
            for mask in range(1 << (len(self.ages) - 1)):
                cuts = tuple(i + 1 for i in range(len(self.ages) - 1) if mask & (1 << i))
                endpoints = (0,) + cuts + (len(self.ages),)
                segments = [self.segment(lane, left, right, clarity) for left, right in zip(endpoints, endpoints[1:])]
                examined += 1
                if all(segment is not None for segment in segments):
                    choices.append((cuts, segments))
            lane_options.append(choices)
        best: OracleResult | None = None
        for combination in product(*lane_options):
            segments = [segment for _, lane_segments in combination for segment in lane_segments]
            cuts = tuple(cut for lane_cuts, _ in combination for cut in lane_cuts)
            work = sum(segment["work"] for segment in segments)
            score = (work, len(segments), cuts)
            if best is None or score < best["score"]:
                best = {"score": score, "segments": segments,
                        "cost": {key: sum(segment[key] for segment in segments)
                                 for key in ("enumeration", "boundary", "storage", "work")},
                        "cuts_by_lane": {repr(lane): list(cut) for lane, (cut, _) in zip(self.lanes, combination)}}
        if best is None:
            raise OutsideOracleFamily("No admissible partition in independently enumerated family")
        best["cost"]["blocks"] = len(best["segments"])
        best["cut_masks_examined"] = examined
        best["atom_dependencies"] = [{"age": atom.age, "status": atom.status,
                                       "live_states": sorted(self.live[key]),
                                       "nonconstant_actions": sorted(set(atom.action_names) - set(atom.forced)),
                                       "forced_actions": atom.forced}
                                      for key, atom in self.atoms.items()]
        return best

    def economic_values(self, interfaces=None):
        """Evaluate the economic decisions, independently of partition costs.

        `interfaces` maps atom keys to state-axis names in a proposed quotient.
        Omitted original state coordinates use their first decoded grid value;
        comparing every original cell to this quotient detects a false removal.
        The fixture family has ordinary forward solve edges and no same-age reads.
        Returns all feasible Q values and maxima, using exact rational arithmetic.
        """
        chosen = {key: tuple(sorted(interfaces[key])) if interfaces is not None else atom.state_names
                  for key, atom in self.atoms.items()}
        stored = {}
        q_stored = {}
        for age in reversed(self.ages):
            for lane in self.lanes:
                atom = self.atoms[(age, lane)]
                if atom.program.get("reads"):
                    raise OutsideOracleFamily("Value evaluator deliberately excludes value-read extensions")
                for reduced_state in assignments(chosen[atom.key], self.state_domains):
                    state = {name: values(self.state_domains[name])[0] for name in atom.state_names}
                    state.update(reduced_state)
                    base = {**self.problem.get("parameters", {}), **atom.status, "age": age, **state}
                    state_key = (atom.key, tuple((name, reduced_state[name]) for name in chosen[atom.key]))
                    candidates = {}
                    actions = tuple(name for name in atom.action_names
                                    if interfaces is None or name not in atom.forced)
                    for reduced_action in assignments(actions, self.action_domains):
                        action = {} if interfaces is None else dict(atom.forced)
                        action.update(reduced_action)
                        env = {**base, **action}
                        if not scalar(atom.program.get("feasible", "True"), env):
                            continue
                        is_terminal = atom.program.get("terminal", False)
                        payoff_expr = atom.program.get("terminal_payoff", "0") if is_terminal else atom.program.get("reward", "0")
                        payoff = Fraction(scalar(payoff_expr, env))
                        continuation = Fraction()
                        if not is_terminal:
                            for edge in atom.program.get("transitions", []):
                                if edge.get("phase", "both") not in ("solve", "both"):
                                    continue
                                probability = Fraction(scalar(edge.get("probability", "1"), env))
                                if probability == 0:
                                    continue
                                target = self._target(atom, edge, env, False)
                                if target is None:
                                    continue
                                if target.age <= age:
                                    raise OutsideOracleFamily("Value evaluator requires forward time edges")
                                target_state = tuple((name, scalar(edge.get("updates", {}).get(name, name), env))
                                                     for name in chosen[target.key])
                                continuation += probability * stored[(target.key, target_state)]
                        q = payoff + Fraction(scalar(atom.program.get("discount", "1"), env)) * continuation
                        candidates[tuple((name, action[name]) for name in atom.action_names)] = q
                    if not candidates:
                        raise OutsideOracleFamily("No feasible action while evaluating economic values")
                    stored[state_key] = max(candidates.values())
                    q_stored[state_key] = candidates
        return {"values": stored, "q_values": q_stored, "state_axes": chosen}

    def check_economic_pullback(self, interfaces):
        original = self.economic_values()
        quotient = self.economic_values(interfaces)
        comparisons = 0
        action_comparisons = 0
        for (atom_key, state_values), value in original["values"].items():
            state = dict(state_values)
            projected = (atom_key, tuple((name, state[name]) for name in quotient["state_axes"][atom_key]))
            if quotient["values"][projected] != value:
                raise AssertionError(f"Value pullback failed at {atom_key}: {state}")
            old_q = original["q_values"][(atom_key, state_values)]
            new_q = quotient["q_values"][projected]
            if old_q != new_q:
                raise AssertionError(f"Feasible-action Q pullback failed at {atom_key}: {state}")
            comparisons += 1
            action_comparisons += len(old_q)
        return {"state_value_comparisons": comparisons, "feasible_q_comparisons": action_comparisons}
