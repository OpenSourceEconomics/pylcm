#!/usr/bin/env python3
"""Explicit finite checker with exact transport and configurable row-mass admission."""
from __future__ import annotations

import argparse
from collections import defaultdict
from fractions import Fraction
import json
from pathlib import Path
import re
import sys
from .records import CheckLayer

LAYERS = ("finite_model", "finite_operator", "source_alignment", "allocation", "runtime")
KINDS = {"relabel", "age_split", "support", "forced_coordinate", "quotient", "merge", "dominance"}


class Invalid(ValueError):
    pass


def need(condition, message):
    if not condition:
        raise Invalid(message)


def rational(value):
    """Integers and integer/fraction strings only. No implicit float conversion."""
    if type(value) is int:
        return Fraction(value)
    if isinstance(value, str) and re.fullmatch(r"-?\d+(?:/[1-9]\d*)?", value):
        return Fraction(value)
    raise Invalid(f"not an exact integer or rational literal: {value!r}")


def encoded(value):
    if isinstance(value, Fraction):
        return str(value)
    if isinstance(value, dict):
        return {str(k): encoded(v) for k, v in value.items()}
    if isinstance(value, (set, tuple, list)):
        return [encoded(v) for v in value]
    return value


def keys(obj, required, optional=(), where="object"):
    need(isinstance(obj, dict), f"{where} must be an object")
    need(set(required) <= set(obj), f"{where}: missing {sorted(set(required) - set(obj))}")
    need(set(obj) <= set(required) | set(optional),
         f"{where}: unknown fields {sorted(set(obj) - set(required) - set(optional))}")


def identifier(value, where):
    need(isinstance(value, str) and bool(value), f"{where}: expected nonempty string")


def exact_json(value):
    need(not isinstance(value, float), "binary floating-point JSON numbers are not admitted")
    if isinstance(value, dict):
        for v in value.values():
            exact_json(v)
    elif isinstance(value, list):
        for v in value:
            exact_json(v)


def scalar_map(obj, where):
    need(isinstance(obj, dict), f"{where}: expected object of exact scalars")
    return {str(k): rational(v) for k, v in obj.items()}


def push(weights, mapping):
    out = defaultdict(Fraction)
    for source, weight in weights.items():
        if weight:
            need(source in mapping, f"nonzero source {source!r} has no projection")
            out[mapping[source]] += weight
    return {k: v for k, v in out.items() if v}


def nonzero(weights):
    return {k: v for k, v in weights.items() if v}


class Report:
    def __init__(self, name, claims, probability_tolerance):
        self.name = name
        self.layers: dict[str, CheckLayer] = {layer: {"status": "NOT_RUN", "checks": 0, "witnesses": [], "notes": []}
                       for layer in claims}
        self.case_results = []
        self.probability_tolerance = probability_tolerance
        self.accepted_nonunit_rows = 0
        self.max_probability_mass_error = Fraction()

    def probability_row(self, weights, message):
        weights = tuple(weights)
        total = sum(weights, Fraction())
        error = abs(total - 1)
        need(all(p >= 0 for p in weights) and total > 0
             and error <= self.probability_tolerance, message)
        if error:
            self.accepted_nonunit_rows += 1
            self.max_probability_mass_error = max(self.max_probability_mass_error, error)

    def test(self, condition, code, *, layer="finite_model", **witness):
        row = self.layers[layer]
        row["checks"] += 1
        if row["status"] == "NOT_RUN":
            row["status"] = "PASS"
        if not condition:
            row["status"] = "FAIL"
            row["witnesses"].append(encoded({"condition": code, **witness}))
        return bool(condition)

    def unresolved(self, code, *, layer="finite_model", **witness):
        row = self.layers[layer]
        if row["status"] != "FAIL":
            row["status"] = "UNRESOLVED"
        row["witnesses"].append(encoded({"condition": code, **witness}))

    def note(self, layer, text):
        if text not in self.layers[layer]["notes"]:
            self.layers[layer]["notes"].append(text)

    def admission_aborted(self):
        """An earlier successful check does not certify unvisited later cases."""
        for row in self.layers.values():
            if row["status"] == "PASS":
                row["status"] = "UNRESOLVED"
                row["notes"].append("Admission aborted before all requested obligations completed.")

    def finish(self):
        statuses = {x["status"] for x in self.layers.values()}
        status = "FAIL" if "FAIL" in statuses else "UNRESOLVED" if statuses & {"UNRESOLVED", "NOT_RUN"} else "PASS"
        result = {"format_version": "1.0", "id": self.name, "status": status,
                "layers": self.layers, "cases": self.case_results,
                "trust_boundary": "Explicit finite data only; extraction, scientific-source faithfulness, "
                "parameter continua, actual allocation, compiler behavior and machine arithmetic are not inferred."}
        if self.accepted_nonunit_rows:
            result["probability_normalization"] = {
                "accepted_nonunit_rows": self.accepted_nonunit_rows,
                "max_absolute_mass_error": str(self.max_probability_mass_error),
                "absolute_tolerance": str(self.probability_tolerance),
                "weights_changed": False,
                "exact_markov_normalization": False,
            }
            self.note("finite_model", "Nonunit rows are admitted under the reported mass budget and remain unchanged. "
                      "Exact transport comparisons concern those raw finite operators; the exact-normalization "
                      "hypothesis of the finite Markov theorem has not been established for them.")
        return result


def parse_model(model, label, report):
    keys(model, ("objective", "stages", "cells", "information_partition"), where=label)
    identifier(model["objective"], f"{label}.objective")
    need(isinstance(model["stages"], list) and model["stages"], f"{label}.stages empty")
    stages = []
    for i, stage in enumerate(model["stages"]):
        keys(stage, ("age", "actor", "observations", "commitment", "discount_boundary"), where=f"{label}.stages[{i}]")
        stage = {**stage, "age": rational(stage["age"])}
        stages.append(stage)
        identifier(stage["actor"], "stage.actor")
        identifier(stage["discount_boundary"], "stage.discount_boundary")
        need(isinstance(stage["observations"], list) and isinstance(stage["commitment"], list),
             "observations and commitment must be ordered lists")
        for item in stage["observations"] + stage["commitment"]:
            identifier(item, "stage sequence item")
    need(isinstance(model["cells"], dict), "cells must be an object")
    result = {}
    for name, raw in model["cells"].items():
        identifier(name, "cell id")
        keys(raw, ("stage", "regime", "channel", "rank", "state", "observables", "terminal", "actions", "taste"),
             ("empty_semantics",), where=f"{label}.{name}")
        need(type(raw["stage"]) is int and 0 <= raw["stage"] < len(model["stages"]), "invalid cell stage")
        need(type(raw["rank"]) is int and raw["rank"] >= 0, "rank must be a nonnegative integer")
        identifier(raw["regime"], "cell.regime")
        identifier(raw["channel"], "cell.channel")
        need(isinstance(raw["state"], dict) and isinstance(raw["observables"], dict), "state/observables must be objects")
        for attr, value in raw["state"].items():
            keys(value, ("type", "code", "value"), where=f"state attribute {attr}")
            need(value["type"] in ("categorical", "rational", "label"), "unknown state attribute type")
            need(type(value["code"]) in (str, int), "state code must be integer or string")
            if value["type"] == "rational":
                rational(value["value"])
        cell = dict(raw)
        terminal = raw["terminal"]
        cell["terminal"] = None if terminal is None else scalar_map(terminal, "terminal")
        need(isinstance(raw["actions"], dict), "actions must be an object")
        if terminal is not None:
            need(model["objective"] in terminal and not raw["actions"], "terminal cells need objective and no actions")
        elif not raw["actions"]:
            need(raw.get("empty_semantics") == "negative_infinity", "empty row must explicitly declare negative_infinity")
        actions = {}
        for action_id, action in raw["actions"].items():
            identifier(action_id, "action id")
            keys(action, ("code", "decoded", "reward", "discount", "kernels", "references", "taste_group"), where="action")
            need(type(action["code"]) in (str, int), "action code must be integer or string")
            need(isinstance(action["decoded"], dict), "decoded action attributes must be an object")
            parsed = dict(action)
            parsed["reward"] = scalar_map(action["reward"], "reward")
            need(model["objective"] in parsed["reward"], "action reward lacks objective")
            parsed["discount"] = rational(action["discount"])
            need(parsed["discount"] >= 0, "economic expected-return core requires nonnegative discount")
            keys(action["kernels"], ("solve", "simulation"), where="kernels")
            parsed["kernels"] = {phase: scalar_map(action["kernels"][phase], "kernel") for phase in ("solve", "simulation")}
            for phase, row in parsed["kernels"].items():
                report.probability_row(row.values(),
                    f"{label}/{name}/{action_id}/{phase}: probabilities must be nonnegative and sum to one")
            parsed["references"] = scalar_map(action["references"], "references")
            actions[action_id] = parsed
        cell["actions"] = actions
        cell["taste"] = parse_taste(raw["taste"], actions, report)
        result[name] = cell
    for name, cell in result.items():
        for action_id, action in cell["actions"].items():
            for phase, row in action["kernels"].items():
                for destination, weight in row.items():
                    need(destination in result, f"{label}: unknown {phase} destination {destination}")
                    if weight:
                        need(result[destination]["stage"] == cell["stage"] + 1,
                             f"{label}/{name}: chronological edges must go to the next decision stage")
            for destination, weight in action["references"].items():
                need(destination in result, f"unknown outside-value destination {destination}")
                if weight:
                    need(result[destination]["stage"] == cell["stage"], "outside-value read must be same-stage")
    partition = model["information_partition"]
    need(isinstance(partition, list) and all(isinstance(block, list) and block for block in partition),
         "information_partition must list nonempty blocks")
    flat = [node for block in partition for node in block]
    need(len(flat) == len(set(flat)) and set(flat) == set(result), "information partition must cover every cell exactly once")
    for block in partition:
        need(len({result[node]["stage"] for node in block}) == 1, "information block crosses stages")
    return {**model, "cells": result, "stages": stages}


def parse_taste(raw, actions, report):
    need(isinstance(raw, dict) and "kind" in raw, "taste must specify kind")
    if raw["kind"] == "deterministic":
        keys(raw, ("kind",), where="deterministic taste")
        need(all(a["taste_group"] is None for a in actions.values()), "deterministic actions have null taste_group")
        return dict(raw)
    if raw["kind"] == "finite":
        keys(raw, ("kind", "groups", "atoms"), where="finite taste")
    elif raw["kind"] == "ev1":
        keys(raw, ("kind", "groups", "scale", "normalization", "dependence"), where="EV1 taste")
        need(rational(raw["scale"]) > 0 and raw["dependence"] == "independent", "EV1 metadata supports positive-scale iid groups only")
    else:
        raise Invalid("unsupported taste encoding; use an explicit unresolved obligation")
    groups = raw["groups"]
    need(isinstance(groups, list) and len(groups) == len(set(groups)), "taste groups must be distinct")
    for group in groups:
        identifier(group, "taste group")
    need(set(a["taste_group"] for a in actions.values()) == set(groups), "every taste group must index a feasible alternative")
    parsed = dict(raw)
    if raw["kind"] == "ev1":
        parsed["scale"] = rational(raw["scale"])
    if raw["kind"] == "finite":
        need(isinstance(raw["atoms"], list) and raw["atoms"], "finite taste needs atoms")
        parsed["atoms"] = []
        for atom in raw["atoms"]:
            keys(atom, ("probability", "values"), where="taste atom")
            prob = rational(atom["probability"])
            values = scalar_map(atom["values"], "taste vector")
            need(prob >= 0 and set(values) == set(groups), "invalid taste atom")
            parsed["atoms"].append({"probability": prob, "values": values})
        report.probability_row((a["probability"] for a in parsed["atoms"]),
                               "taste probabilities must sum to one")
    return parsed


def taste_law(taste, group_map):
    law = defaultdict(Fraction)
    for atom in taste["atoms"]:
        vector = tuple(sorted((group_map[g], value) for g, value in atom["values"].items()))
        law[vector] += atom["probability"]
    return {vector: prob for vector, prob in law.items() if prob}


def check_support(report, model, support, seeds, case, side):
    cells = model["cells"]
    need(isinstance(support, list) and len(support) == len(set(support)), "support must list distinct cells")
    need(set(support) <= set(cells), "support contains unknown cell")
    active = set(support)
    seen = set(seeds)
    frontier = list(seeds)
    while frontier:
        node = frontier.pop()
        for action in cells[node]["actions"].values():
            destinations = {v for row in action["kernels"].values() for v, p in row.items() if p}
            destinations |= {v for v, p in action["references"].items() if p}
            for destination in destinations - seen:
                seen.add(destination)
                frontier.append(destination)
    report.test(seen <= active, "complete_admitted_support", case=case, side=side,
                missing=sorted(seen - active))
    for node in support:
        cell = cells[node]
        if cell["terminal"] is None and not cell["actions"]:
            report.unresolved("reachable_empty_row_outside_finite_real_core", case=case, side=side, cell=node,
                              declared_semantics="negative_infinity")
        for action_id, action in cell["actions"].items():
            for phase, row in action["kernels"].items():
                for destination, prob in row.items():
                    if prob:
                        report.test(destination in active, "positive_kernel_support", case=case, side=side,
                                    cell=node, action=action_id, phase=phase, destination=destination, weight=prob)
            for destination, weight in action["references"].items():
                if weight:
                    report.test(destination in active, "outside_value_support", case=case, side=side,
                                cell=node, action=action_id, destination=destination, weight=weight)
                    report.test(cells[destination]["rank"] < cell["rank"], "acyclic_reference_rank", case=case,
                                side=side, cell=node, destination=destination, rank=cell["rank"],
                                destination_rank=cells[destination]["rank"])
    if any(len(set(block) & active) > 1 for block in model["information_partition"]):
        report.unresolved("partial_observation_not_in_core", case=case, side=side,
                          reason="Bellman maximization here assumes fully observed Markov cells.")
    return seen


def check_tastes(report, old, new, mapping, embed, case, node):
    original_taste, reduced_taste = old["taste"], new["taste"]
    context = {"case": case, "cell": node}
    if not report.test(original_taste["kind"] == reduced_taste["kind"], "taste_operator_kind", **context):
        return
    if original_taste["kind"] == "deterministic":
        report.test(mapping == {}, "deterministic_taste_map_empty", **context)
        return
    if not report.test(set(mapping) == set(original_taste["groups"]) and set(mapping.values()) == set(reduced_taste["groups"])
                       and len(mapping) == len(set(mapping.values())), "taste_group_bijection", **context,
                       original=original_taste["groups"], reduced=reduced_taste["groups"], mapping=mapping):
        return
    for reduced_action, original_action in embed.items():
        report.test(mapping[old["actions"][original_action]["taste_group"]] == new["actions"][reduced_action]["taste_group"],
                    "shared_discrete_taste_group", **context, original_action=original_action, reduced_action=reduced_action)
    if original_taste["kind"] == "finite":
        before = taste_law(original_taste, mapping)
        after = taste_law(reduced_taste, {g: g for g in reduced_taste["groups"]})
        report.test(before == after, "conditional_joint_taste_law", **context,
                    original_law=str(before), reduced_law=str(after))
    else:
        report.test(all(original_taste[k] == reduced_taste[k] for k in ("scale", "normalization", "dependence")),
                    "ev1_law_metadata", **context)
        report.unresolved("continuous_EV1_extension_not_formally_instantiated", **context,
                          reason="Necessary group/law metadata checked; finite-rational evaluator does not integrate EV1.")


def action_pair(report, old, new, original_action, reduced_action, projection, fixed, context, inequality=False, objective="value"):
    a, b = old["actions"][original_action], new["actions"][reduced_action]
    here = {**context, "original_action": original_action, "reduced_action": reduced_action}
    if not inequality:
        report.test(not set(fixed) & set(b["decoded"]), "fixed_coordinate_not_overwritten", **here)
        report.test(a["decoded"] == {**b["decoded"], **fixed}, "decoded_action_embedding", **here,
                    original=a["decoded"], embedded={**b["decoded"], **fixed})
        report.test(a["reward"] == b["reward"], "reward_transport", **here,
                    original=a["reward"], reduced=b["reward"])
    else:
        report.test(set(a["reward"]) == set(b["reward"]) and a["reward"][objective] <= b["reward"][objective]
                    and all(a["reward"][k] == b["reward"][k] for k in a["reward"] if k != objective),
                    "dominance_reward_cover", **here, omitted=a["reward"], retained=b["reward"])
    report.test(a["discount"] == b["discount"], "discount_transport", **here,
                original=a["discount"], reduced=b["discount"])
    for phase in ("solve", "simulation"):
        try:
            transformed = push(a["kernels"][phase], projection)
        except Invalid as exc:
            report.test(False, "kernel_projection_missing", **here, phase=phase, error=str(exc))
            continue
        report.test(transformed == nonzero(b["kernels"][phase]), "kernel_pushforward", **here,
                    phase=phase, original_projected=transformed, reduced=b["kernels"][phase])
    try:
        refs = push(a["references"], projection)
        report.test(refs == nonzero(b["references"]), "outside_value_transport", **here,
                    original_projected=refs, reduced=b["references"])
    except Invalid as exc:
        report.test(False, "outside_value_projection_missing", **here, error=str(exc))


def evaluate(model, support):
    """Independent recurrence on each model; it does not read the certificate."""
    values, argmax = {}, {}
    cells, objective = model["cells"], model["objective"]
    order = sorted(support, key=lambda n: (-cells[n]["stage"], cells[n]["rank"], n))
    for node in order:
        cell = cells[node]
        if cell["terminal"] is not None:
            values[node] = cell["terminal"][objective]
            continue
        if not cell["actions"] or cell["taste"]["kind"] == "ev1":
            raise Invalid("backward evaluator unsupported for empty/EV1 row")
        qs = {}
        for action_id, action in cell["actions"].items():
            try:
                continuation = sum((p * values[d] for d, p in action["kernels"]["solve"].items() if p), Fraction())
                outside = sum((p * values[d] for d, p in action["references"].items() if p), Fraction())
            except KeyError as exc:
                raise Invalid(f"unavailable evaluation dependency {exc.args[0]}") from exc
            qs[action_id] = action["reward"][objective] + action["discount"] * continuation + outside
        if cell["taste"]["kind"] == "deterministic":
            best = max(qs.values())
            values[node] = best
            argmax[node] = sorted(a for a, value in qs.items() if value == best)
        else:
            values[node] = sum((atom["probability"] * max(q + atom["values"][cell["actions"][a]["taste_group"]]
                                 for a, q in qs.items()) for atom in cell["taste"]["atoms"]), Fraction())
    return values, argmax


def check_case(report, raw):
    keys(raw, ("id", "parameters", "original", "reduced", "starts", "certificate", "subjects"),
         ("operators", "allocation"), where="case")
    case = raw["id"]
    identifier(case, "case id")
    need(isinstance(raw["parameters"], dict), "parameters must be an explicit object")
    for value in raw["parameters"].values():
        rational(value)
    old, new = parse_model(raw["original"], "original", report), parse_model(raw["reduced"], "reduced", report)
    cert = raw["certificate"]
    keys(cert, ("kind", "support", "state_map", "action_embedding", "fixed_action_attributes", "taste_group_map"),
         ("dominance_cover", "action_specific_fixed"), where="certificate")
    need(cert["kind"] in KINDS, "unknown transformation kind")
    keys(cert["support"], ("original", "reduced"), where="support")
    for field in ("state_map", "action_embedding", "fixed_action_attributes", "taste_group_map"):
        need(isinstance(cert[field], dict), f"{field} must be an object")
    projection = cert["state_map"]
    os, ns = cert["support"]["original"], cert["support"]["reduced"]
    report.test(old["objective"] == new["objective"], "objective_identity", case=case)
    report.test(old["stages"] == new["stages"], "clock_actor_information_commitment", case=case,
                original=old["stages"], reduced=new["stages"])
    map_ok = report.test(set(projection) == set(os) and set(projection.values()) == set(ns),
                         "surjective_state_projection", case=case, domain=sorted(projection), image=sorted(set(projection.values())),
                         original_support=os, reduced_support=ns)
    need(isinstance(raw["starts"], list) and raw["starts"], "at least one initial/incoming/evaluation start required")
    seeds = {"original": set(), "reduced": set()}
    for start in raw["starts"]:
        keys(start, ("id", "kind", "original", "reduced"), where="start")
        need(start["kind"] in ("initial", "incoming", "value_read"), "unknown start kind")
        distributions = {}
        for side, model in (("original", old), ("reduced", new)):
            distribution = scalar_map(start[side], "start distribution")
            report.probability_row(distribution.values(), "start distribution not normalized")
            need(set(distribution) <= set(model["cells"]), "start refers to unknown cell")
            seeds[side] |= {n for n, p in distribution.items() if p}
            distributions[side] = distribution
        try:
            report.test(push(distributions["original"], projection) == nonzero(distributions["reduced"]),
                        "initial_or_incoming_law", case=case, start=start["id"], kind=start["kind"],
                        original=distributions["original"], reduced=distributions["reduced"])
        except Invalid as exc:
            report.test(False, "initial_or_incoming_projection_missing", case=case, start=start["id"], error=str(exc))
    reached_old = check_support(report, old, os, seeds["original"], case, "original")
    reached_new = check_support(report, new, ns, seeds["reduced"], case, "reduced")
    decision_nodes = {n for n in os if old["cells"][n]["terminal"] is None}
    report.test(set(cert["action_embedding"]) == decision_nodes, "action_certificate_domain", case=case)
    report.test(set(cert["fixed_action_attributes"]) == decision_nodes, "fixed_coordinate_certificate_domain", case=case)
    report.test(set(cert["taste_group_map"]) == decision_nodes, "taste_certificate_domain", case=case)
    specific_fixed = cert.get("action_specific_fixed", {})
    need(isinstance(specific_fixed, dict) and set(specific_fixed) <= decision_nodes, "invalid action_specific_fixed domain")
    for node in os:
        if node not in projection or projection[node] not in new["cells"]:
            continue
        reduced_node = projection[node]
        x, y = old["cells"][node], new["cells"][reduced_node]
        context = {"case": case, "original_cell": node, "reduced_cell": reduced_node}
        report.test(x["stage"] == y["stage"] and x["channel"] == y["channel"], "stage_value_channel", **context)
        for attribute in sorted(set(x["state"]) & set(y["state"])):
            xa, ya = x["state"][attribute], y["state"][attribute]
            xv = rational(xa["value"]) if xa["type"] == "rational" else xa["value"]
            yv = rational(ya["value"]) if ya["type"] == "rational" else ya["value"]
            report.test(xa["type"] == ya["type"] and xv == yv, "retained_state_decoded_attribute",
                        **context, attribute=attribute, original=xa, reduced=ya)
        report.test(x["observables"] == y["observables"], "output_observables", **context,
                    original=x["observables"], reduced=y["observables"])
        if x["terminal"] is not None or y["terminal"] is not None:
            report.test(x["terminal"] == y["terminal"], "terminal_payoff_transport", **context,
                        original=x["terminal"], reduced=y["terminal"])
            continue
        embed = cert["action_embedding"].get(node, {})
        fixed = cert["fixed_action_attributes"].get(node, {})
        groupmap = cert["taste_group_map"].get(node, {})
        need(isinstance(embed, dict) and isinstance(fixed, dict) and isinstance(groupmap, dict), "invalid action certificate")
        embed_ok = report.test(set(embed) == set(y["actions"]) and set(embed.values()) <= set(x["actions"])
                               and len(embed.values()) == len(set(embed.values())), "feasible_action_injection", **context,
                               embedding=embed, original_actions=list(x["actions"]), reduced_actions=list(y["actions"]))
        if cert["kind"] != "dominance":
            report.test(set(embed.values()) == set(x["actions"]), "feasible_action_surjection", **context,
                        embedding=embed, original_actions=list(x["actions"]))
        if not embed_ok:
            continue
        check_tastes(report, x, y, groupmap, embed, case, node)
        action_fixed = specific_fixed.get(node)
        if action_fixed is not None:
            need(isinstance(action_fixed, dict) and set(action_fixed) == set(embed) and fixed == {},
                 "action_specific_fixed must cover each reduced action and use empty constant fixed attributes")
            need(all(isinstance(v, dict) for v in action_fixed.values()), "fixed action attributes must be objects")
        for b, a in embed.items():
            embedded_fixed = fixed if action_fixed is None else action_fixed[b]
            action_pair(report, x, y, a, b, projection, embedded_fixed, context, objective=old["objective"])
        if cert["kind"] == "dominance":
            cover = cert.get("dominance_cover", {}).get(node, {})
            cover_ok = report.test(set(cover) == set(x["actions"]) and set(cover.values()) <= set(y["actions"]),
                                   "total_dominance_cover", **context, cover=cover)
            if cover_ok:
                for a, b in cover.items():
                    action_pair(report, x, y, a, b, projection, {}, context, inequality=True, objective=old["objective"])
                    if x["taste"]["kind"] != "deterministic":
                        report.test(groupmap.get(x["actions"][a]["taste_group"]) == y["actions"][b]["taste_group"],
                                    "dominance_requires_shared_taste", **context, original_action=a, reduced_action=b)
    keys(raw["subjects"], ("original", "reduced"), where="subjects")
    subjects_old, subjects_new = raw["subjects"]["original"], raw["subjects"]["reduced"]
    need(isinstance(subjects_old, dict) and isinstance(subjects_new, dict), "subjects must be objects keyed by immutable subject id")
    report.test(set(subjects_old) == set(subjects_new), "subject_identity", case=case)
    for subject in sorted(set(subjects_old) & set(subjects_new)):
        a, b = subjects_old[subject], subjects_new[subject]
        keys(a, ("node", "tags", "outputs"), where="original subject")
        keys(b, ("node", "tags", "outputs"), where="reduced subject")
        report.test(a["node"] in projection and projection.get(a["node"]) == b["node"]
                    and a["tags"] == b["tags"] and a["outputs"] == b["outputs"], "immutable_report_tag_join", case=case,
                    subject=subject, original=a, reduced=b)
    # Corroboration is independent from local transport checks and never rescues a failed certificate.
    evaluation = {"status": "not_run", "reason": "primitive certificate or admission unresolved"}
    relevant = [w for w in report.layers["finite_model"]["witnesses"] if w.get("case") == case]
    if not relevant and map_ok:
        try:
            ov, oa = evaluate(old, os)
            nv, na = evaluate(new, ns)
            witness_count = len(report.layers["finite_model"]["witnesses"])
            for node in os:
                report.test(ov[node] == nv[projection[node]], "independent_backward_value", case=case, cell=node,
                            original_value=ov[node], reduced_value=nv[projection[node]])
                if node in oa and projection[node] in na:
                    lifted = sorted(cert["action_embedding"][node][b] for b in na[projection[node]])
                    report.test(lifted == oa[node] if cert["kind"] != "dominance" else set(lifted) <= set(oa[node]),
                                "argmax_transport" if cert["kind"] != "dominance" else "retained_optimum", case=case,
                                cell=node, original_argmax=oa[node], lifted_reduced_argmax=lifted)
            evaluation = {"status": "passed" if len(report.layers["finite_model"]["witnesses"]) == witness_count else "failed",
                          "original_values": encoded(ov), "reduced_values": encoded(nv)}
        except Invalid as exc:
            report.unresolved("backward_evaluator_not_applicable", case=case, reason=str(exc))
            evaluation = {"status": "not_run", "reason": str(exc)}
    report.case_results.append({"id": case, "transformation": cert["kind"],
                                "reachable_original": sorted(reached_old), "reachable_reduced": sorted(reached_new),
                                "backward_evaluation": evaluation})
    if "finite_operator" in report.layers:
        check_operators(report, raw.get("operators"), projection, old, new, case)
    if "allocation" in report.layers:
        check_allocation(report, raw.get("allocation"), case)


def check_operators(report, operators, projection, old, new, case):
    layer = "finite_operator"
    if operators is None:
        report.unresolved("missing_operator_certificate", layer=layer, case=case)
        return
    keys(operators, ("kind", "original", "reduced", "query_map", "conventions"), where="operators")
    if operators["kind"] != "finite_linear":
        report.unresolved("unsupported_operator_class", layer=layer, case=case, kind=operators["kind"])
        return
    keys(operators["conventions"], ("original", "reduced"), where="operator conventions")
    report.test(operators["conventions"]["original"] == operators["conventions"]["reduced"],
                "operator_conventions", layer=layer, case=case)
    oq, nq, qm = operators["original"], operators["reduced"], operators["query_map"]
    need(isinstance(oq, dict) and isinstance(nq, dict) and isinstance(qm, dict), "operator rows/maps must be objects")
    ok = report.test(set(qm) == set(oq) and set(qm.values()) == set(nq) and len(qm) == len(set(qm.values())),
                     "query_bijection", layer=layer, case=case)
    if ok:
        for query, mapped in qm.items():
            a, b = scalar_map(oq[query], "operator row"), scalar_map(nq[mapped], "operator row")
            need(set(a) <= set(old["cells"]) and set(b) <= set(new["cells"]), "operator references unknown knot")
            try:
                projected = push(a, projection)
                report.test(projected == nonzero(b), "finite_operator_commutation", layer=layer, case=case,
                            original_query=query, reduced_query=mapped, original_projected=projected, reduced=b)
            except Invalid as exc:
                report.test(False, "operator_knot_not_in_evaluation_support", layer=layer, case=case, query=query, error=str(exc))
    report.note(layer, "Only the listed exact finite linear rows are checked for all continuation vectors; "
                "unsupplied queries, nonlinear interpolants and floating-point evaluations are not certified.")


def allocation_counts(spec):
    keys(spec, ("arrays", "candidate_products"), where="allocation side")
    need(isinstance(spec["arrays"], list) and isinstance(spec["candidate_products"], list), "allocation lists required")
    total_bytes = 0
    array_names = [array["name"] for array in spec["arrays"]]
    need(len(array_names) == len(set(array_names)), "allocated array names must be unique")
    for array in spec["arrays"]:
        keys(array, ("name", "axes", "bytes_per_entry"), where="allocated array")
        need(type(array["bytes_per_entry"]) is int and array["bytes_per_entry"] > 0, "invalid entry byte size")
        count = 1
        for axis in array["axes"]:
            need(isinstance(axis, list) and len(axis) == len(set(axis)), "allocated axis must list distinct ids")
            count *= len(axis)
        total_bytes += count * array["bytes_per_entry"]
    candidates = 0
    for product in spec["candidate_products"]:
        keys(product, ("states", "actions"), where="candidate product")
        need(isinstance(product["states"], list) and isinstance(product["actions"], list), "candidate axes must be lists")
        need(len(product["states"]) == len(set(product["states"])) and len(product["actions"]) == len(set(product["actions"])),
             "candidate axes have duplicates")
        candidates += len(product["states"]) * len(product["actions"])
    return {"allocated_bytes": total_bytes, "enumerated_candidates": candidates}


def check_allocation(report, allocation, case):
    layer = "allocation"
    if allocation is None:
        report.unresolved("missing_allocation_model", layer=layer, case=case)
        return
    keys(allocation, ("original", "reduced", "claim", "bridge"), where="allocation")
    need(allocation["claim"] in ("counts_only", "strictly_less_bytes", "strictly_less_candidates"), "invalid allocation claim")
    before, after = allocation_counts(allocation["original"]), allocation_counts(allocation["reduced"])
    report.test(True, "declared_allocation_counts", layer=layer, case=case,
                original=before, reduced=after)
    report.note(layer, f"Case {case}: declared allocation counts {before} -> {after}.")
    if allocation["claim"] != "counts_only":
        field = "allocated_bytes" if allocation["claim"] == "strictly_less_bytes" else "enumerated_candidates"
        report.test(after[field] < before[field], "declared_allocation_saving", layer=layer, case=case,
                    field=field, original=before[field], reduced=after[field])
    if allocation["bridge"] != "declared_model_only":
        report.unresolved("actual_implementation_allocation_bridge_unverified", layer=layer, case=case,
                          bridge=allocation["bridge"])
    report.note(layer, "Counts describe supplied allocated products, including unsupported cells; they are neither peak live memory nor runtime.")


def check(data, *, probability_tolerance=Fraction(1, 10**12)):
    name = data.get("id", "invalid") if isinstance(data, dict) else "invalid"
    report = Report(name, ["finite_model"], probability_tolerance)
    try:
        need(type(probability_tolerance) is int or isinstance(probability_tolerance, Fraction),
             "probability_tolerance must be a nonnegative Fraction or integer")
        need(probability_tolerance >= 0, "probability_tolerance must be nonnegative")
        exact_json(data)
        keys(data, ("format_version", "id", "claims", "parameter_scope", "policy_claim", "simulation_choice", "cases", "obligations"), where="input")
        need(data["format_version"] == "1.0", "unsupported format_version")
        identifier(data["id"], "input.id")
        claims = data["claims"]
        need(isinstance(claims, list) and "finite_model" in claims and len(claims) == len(set(claims))
             and set(claims) <= set(LAYERS), "claims must be distinct supported layers including finite_model")
        report = Report(data["id"], claims, probability_tolerance)
        need(data["parameter_scope"] in ("listed_cases", "continuum"), "invalid parameter_scope")
        if data["parameter_scope"] == "continuum":
            report.unresolved("unverified_parameter_continuum", reason="Finite case enumeration proves only the listed parameter vectors.")
        need(data["policy_claim"] in ("values_argmax_lifted_law", "value_retained_optimum", "fixed_selector", "same_seed", "arbitrary_original_policy"), "unknown policy claim")
        if data["policy_claim"] in ("fixed_selector", "same_seed", "arbitrary_original_policy"):
            report.unresolved("policy_claim_needs_additional_certificate", policy_claim=data["policy_claim"])
        if data["simulation_choice"] != "same_declared_primitives":
            report.unresolved("separate_simulation_choice_primitives_not_certified", value=data["simulation_choice"],
                              reason="Both kernels are checked, but reoptimization under other preferences/feasibility needs another complete certificate.")
        need(isinstance(data["cases"], list) and data["cases"], "cases must be a nonempty list")
        need(len([x["id"] for x in data["cases"]]) == len({x["id"] for x in data["cases"]}), "duplicate case id")
        for case in data["cases"]:
            if case["certificate"]["kind"] == "dominance":
                report.test(data["policy_claim"] == "value_retained_optimum", "dominance_claim_scope", case=case["id"])
            check_case(report, case)
        need(isinstance(data["obligations"], list), "obligations must be a list")
        for obligation in data["obligations"]:
            keys(obligation, ("id", "layer", "statement", "evidence"), where="obligation")
            need(obligation["layer"] in claims, "obligation layer not claimed")
            report.unresolved("external_or_symbolic_obligation", layer=obligation["layer"], obligation=obligation)
        if "source_alignment" in claims:
            report.unresolved("scientific_source_bridge_not_mechanically_checked", layer="source_alignment")
        if "runtime" in claims:
            report.unresolved("runtime_not_implied_by_finite_certificate", layer="runtime")
        report.note("finite_model", "PASS is limited to the explicitly enumerated cases, declared fully observed decision primitives, "
                    "closed evaluation support and stated action/state transports. Projected laws apply to lifted reduced policies "
                    "with corresponding realized kernels; not arbitrary discarded-history policies, tie breakers or seeds.")
        if any(a["references"] for c in data["cases"] for m in (c["original"], c["reduced"]) for x in m["cells"].values() for a in x["actions"].values()):
            report.note("finite_model", "Same-period linear value reads use an explicitly checked acyclic extension; "
                        "they are not an automatic instance of the finite chronological Lean core.")
    except (Invalid, KeyError, TypeError) as exc:
        report.test(False, "schema_or_admission", error=str(exc))
        report.admission_aborted()
    return report.finish()


def probability_tolerance_argument(value):
    try:
        tolerance = Fraction(value)
    except (ValueError, ZeroDivisionError) as exc:
        raise argparse.ArgumentTypeError("expected a nonnegative exact fraction or decimal tolerance") from exc
    if tolerance < 0:
        raise argparse.ArgumentTypeError("probability tolerance must be nonnegative")
    return tolerance


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--probability-tolerance", type=probability_tolerance_argument,
                        default=Fraction(1, 10**12), metavar="TOLERANCE",
                        help="absolute row-mass tolerance (default: 1e-12; 0 is strict); weights are unchanged")
    args = parser.parse_args(argv)
    try:
        data = json.loads(args.input.read_text(encoding="utf-8"), parse_constant=lambda x: (_ for _ in ()).throw(Invalid(f"nonfinite literal {x}")))
        report = check(data, probability_tolerance=args.probability_tolerance)
    except (OSError, json.JSONDecodeError, Invalid) as exc:
        report = {"format_version": "1.0", "id": args.input.name, "status": "FAIL", "error": str(exc)}
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(text, encoding="utf-8")
    else:
        print(text, end="")
    return {"PASS": 0, "FAIL": 1, "UNRESOLVED": 2}[report["status"]]


if __name__ == "__main__":
    sys.exit(main())
