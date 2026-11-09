"""Original-only finite finder inputs; expected partitions never enter the IR."""
from copy import deepcopy


ROLE = {"actor": "household", "observations": [], "commitments": [],
        "discount_boundary": "annual"}


def problem(name, periods, states=None, actions=None, reward="0", feasible="True",
            updates=None, programs=None, implementation=None):
    states = states or {}
    actions = actions or {}
    if programs is None:
        programs = [{
            "id": "original", "when": f"age < {periods}", "role": deepcopy(ROLE),
            "reward": reward, "discount": "1", "feasible": feasible,
            "terminal": False,
            "transitions": [{"probability": "1", "updates": updates or {
                key: key for key in states}}],
        }]
    programs = deepcopy(programs)
    programs.append({
        "id": "terminal", "when": f"age == {periods}",
        "role": deepcopy(ROLE), "terminal": True, "terminal_payoff": "0",
        "declared_states": [], "declared_actions": [], "transitions": [],
    })
    result = {"schema_version": 1, "name": name, "ages": list(range(periods + 1)),
              "parameters": {}, "states": states, "actions": actions,
              "programs": programs}
    if implementation:
        result["implementation"] = implementation
    return result


def fertility():
    return problem(
        "missing-fertility-and-last-child-consumer", 8,
        states={"child": {"values": [0, 1]}},
        actions={"birth": {"values": [0, 1]}},
        reward="(child if age < 5 else 0) + (birth if age < 2 else 0)",
        feasible="birth == 0 if age >= 2 else True",
        updates={"child": "child"},
    )


def retirement():
    primitive = {
        "role": deepcopy(ROLE), "reward": "(wage * work if age < 2 else 0) - (coverage_lag if age < 5 else 0)",
        "discount": "1", "feasible": "work == 0 if age >= 2 else True",
        "terminal": False,
        "transitions": [{"probability": "1", "updates": {
            "wage": "wage", "coverage_lag": "coverage_lag"}}],
    }
    return problem(
        "retirement-lagged-bill-and-redundant-late-cut", 8,
        states={"wage": {"values": [1, 2]}, "coverage_lag": {"values": [0, 1]}},
        actions={"work": {"values": [0, 1]}},
        programs=[dict(deepcopy(primitive), id="working_block", when="age < 6"),
                  dict(deepcopy(primitive), id="unnecessary_late_block", when="6 <= age < 8")],
    )


def duplicate_coefficients():
    primitive = {
        "role": deepcopy(ROLE), "reward": "(age + 1) * x + effort",
        "discount": "1", "feasible": "True", "terminal": False,
        "transitions": [{"probability": "1", "updates": {"x": "x"}}],
    }
    return problem(
        "redundant-coefficient-and-label-split", 4,
        states={"x": {"values": [0, 1]}}, actions={"effort": {"values": [0, 1]}},
        programs=[dict(deepcopy(primitive), id="arbitrary_first_name", when="age < 2"),
                  dict(deepcopy(primitive), id="arbitrary_second_name", when="2 <= age < 4")],
    )


def nontransitive():
    return problem(
        "nontransitive-segment-size-global-optimum", 3,
        states={"x": {"values": [0, 1, 2]}, "z": {"values": [0, 1]}},
        reward="x if age == 0 else (z if age == 2 else 0)",
        updates={"x": "0", "z": "0"},
        implementation={"max_block_state_entries": 3},
    )


def endpoint_saving():
    return problem(
        "endpoint-saving-clarity", 3,
        actions={"saving": {"values": [0, 1, 2]}},
        reward="saving", feasible="saving == 0 if age == 2 else True",
    )


def endpoint_with_redundant_label():
    p = endpoint_saving()
    first = p["programs"][0]
    first["when"] = "age < 2"
    last = deepcopy(first)
    last.update(id="arbitrary_last_label", when="age == 2")
    p["programs"].insert(1, last)
    p["name"] = "endpoint-saving-redundant-original-label"
    return p


def invalid_cases():
    cases = {
        "empty_feasible_state": problem("empty-feasible-row", 2,
            states={"x": {"values": [0, 1]}}, actions={"a": {"values": [0, 1]}},
            reward="x", feasible="x == 0", updates={"x": "x"}),
        "unbound_reward": problem("unbound-reward", 2, reward="unknown_payoff"),
        "out_of_grid_update": problem("out-of-grid-update", 2,
            states={"x": {"values": [0, 1]}}, reward="x", updates={"x": "2"}),
    }
    for name, expression in (("negative_conditional_weight", "1 if x == 0 else -1"),
                             ("nonnormalized_conditional_row", "1 if x == 0 else 1/2")):
        p = problem(name, 2, states={"x": {"values": [0, 1]}}, reward="x", updates={"x": "x"})
        p["programs"][0]["transitions"][0]["probability"] = expression
        cases[name] = p
    malformed = problem("empty-ages", 2)
    malformed["ages"] = []
    cases["empty_ages"] = malformed
    forbidden = problem("answer-bearing-input", 2)
    forbidden["winning_cuts"] = [1]
    cases["answer_bearing_input"] = forbidden
    return cases


def allocation():
    return problem(
        "fewer-nodes-union-padding-loses", 6,
        states={"x": {"values": [0, 1, 2]}, "z": {"values": [0, 1, 2]}},
        reward="x if age < 3 else z", updates={"x": "0", "z": "0"},
    )


def routing():
    zero = {"role": deepcopy(ROLE), "terminal": True, "terminal_payoff": "0",
            "declared_states": [], "declared_actions": [], "transitions": []}
    return {
        "schema_version": 1, "name": "equal-zero-payoff-distinct-routing-owner",
        "ages": [0, 1], "parameters": {}, "states": {}, "actions": {},
        "programs": [dict(deepcopy(zero), id="fallback_left", when="age == 0",
                          routing_owner="offering_leg"),
                     dict(deepcopy(zero), id="fallback_right", when="age == 1",
                          routing_owner="response_leg")],
    }


def observed_status():
    p = problem(
        "absorbing-claimed-smaller-domain", 4,
        states={"earnings": {"values": [0, 1]}},
        actions={"work": {"values": [0, 1]}},
        reward="earnings + work if status == 'active' else 0",
        feasible="work == 0 if status == 'claimed' else True",
        updates={"earnings": "earnings"},
    )
    p["status_axes"] = {"status": ["active", "claimed"]}
    p["programs"][0]["transitions"][0]["status_updates"] = {"status": "status"}
    return p


def external_read(weight="1", cyclic=False):
    reader = {
        "id": "chooser", "when": "mode == 'chooser'", "role": deepcopy(ROLE),
        "reward": "0", "discount": "1", "feasible": "True", "terminal": False,
        "transitions": [], "reads": [{"target": "outside", "offset": 0, "weight": weight,
            "status_updates": {"mode": "'outside'"}, "updates": {"history": "history"}}],
    }
    consumer = {
        "id": "outside", "when": "mode == 'outside'", "role": dict(deepcopy(ROLE), actor="settlement"),
        "terminal": True, "terminal_payoff": "history", "declared_actions": [], "transitions": [],
    }
    if cyclic:
        consumer["reads"] = [{"target": "chooser", "offset": 0, "weight": "1",
            "status_updates": {"mode": "'chooser'"}, "updates": {"history": "history"}}]
    return {"schema_version": 1, "name": "same-age-exogenous-value-consumer",
            "ages": [0, 1, 2], "states": {"history": {"values": [0, 1]}},
            "actions": {}, "parameters": {}, "status_axes": {"mode": ["chooser", "outside"]},
            "programs": [reader, consumer]}


def rare_incoming(positive=True, phase="both"):
    p = problem(
        "rare-incoming-history-" + ("positive" if positive else "zero") + "-" + phase,
        6, states={"history": {"values": [0, 1]}},
        reward="history if status == 'charged' else 0", updates={"history": "0"},
    )
    p["status_axes"] = {"status": ["active", "charged"]}
    p["parameters"] = {"tiny": "1/100000000000000000000000000000000000000000000000000" if positive else 0}
    # Literal exact rational expressions make support independent of a binary-float cutoff.
    q = "1/100000000000000000000000000000000000000000000000000" if positive else "0"
    rare = f"({q}) if status == 'active' and age < 3 else 0"
    p["programs"][0]["transitions"] = [
        {"phase": phase, "probability": rare,
         "status_updates": {"status": "'charged'"}, "updates": {"history": "history"}},
        {"phase": phase, "probability": f"1 - ({rare})",
         "status_updates": {"status": "status"}, "updates": {"history": "0"}},
    ]
    if phase != "both":
        p["programs"][0]["transitions"].append({
            "phase": "simulation" if phase == "solve" else "solve",
            "probability": "1", "status_updates": {"status": "status"},
            "updates": {"history": "0"},
        })
    return p


def generated(seed):
    """Families with reset laws, so dependency truth tables have a small exact oracle."""
    import random
    rng = random.Random(seed)
    periods = rng.randint(3, 6)
    states = {f"s{k}": {"values": list(range(rng.randint(2, 3)))} for k in range(2)}
    active = []
    largest_atom = 1
    for age in range(periods):
        axes = [name for name in states if rng.randrange(3) == 0]
        active.append(" + ".join(f"{age + 1} * {name}" for name in axes) or "0")
        size = 1
        for name in axes:
            size *= len(states[name]["values"])
        largest_atom = max(largest_atom, size)
    reward = active[-1]
    for age in range(periods - 2, -1, -1):
        reward = f"({active[age]}) if age == {age} else ({reward})"
    return problem(
        f"independent-generated-{seed:02}", periods, states=states,
        reward=reward, updates={name: "0" for name in states},
        implementation={"max_block_state_entries": rng.choice([size for size in [3, 6, 9] if size >= largest_atom])},
    )


def named_cases():
    return {
        "fertility": fertility(), "retirement": retirement(),
        "duplicate_coefficients": duplicate_coefficients(),
        "nontransitive": nontransitive(), "allocation": allocation(),
        "endpoint_saving": endpoint_saving(), "routing": routing(),
        "observed_status": observed_status(),
        "rare_incoming": rare_incoming(), "zero_incoming": rare_incoming(False),
        "rare_simulation_incoming": rare_incoming(True, "simulation"),
    }
