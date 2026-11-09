"""Independent finite decision oracle for specialization-signature loss.

This file imports no regime-finder component. It is an offline acceptance oracle,
not production work and not an execution claimed by the reviewer.
"""
from fractions import Fraction
from itertools import product


def payoff(*, choice, history, scale=1, shift=0):
    return Fraction(shift) - Fraction(scale) * (Fraction(choice) - Fraction(history)) ** 2


def exact_policy(*, histories=(0, 1), choices=(0, 1), scale=1, shift=0):
    answer = {}
    for history in histories:
        rows = [(choice, payoff(choice=choice, history=history, scale=scale, shift=shift))
                for choice in choices]
        best = max(value for _, value in rows)
        answer[history] = {choice for choice, value in rows if value == best}
    return answer


def constant_representation_is_policy_sufficient(*, histories=(0, 1), choices=(0, 1), scale=1):
    policies = exact_policy(histories=histories, choices=choices, scale=scale)
    return bool(set.intersection(*(set(actions) for actions in policies.values())))


def mutation_oracle():
    cases = 0
    for domain, scale, shift in product(((0, 1), (1, 0), (-3, 2), (0, 2, 7)),
                                      (Fraction(1, 8), Fraction(1), Fraction(8)),
                                      (Fraction(-17), Fraction(0), Fraction(29))):
        policies = exact_policy(histories=domain, choices=domain, scale=scale, shift=shift)
        assert all(policies[history] == {history} for history in domain)
        assert not constant_representation_is_policy_sufficient(
            histories=domain, choices=domain, scale=scale)
        cases += 1
    return {"oracle_type": "structural", "cases": cases,
            "binary_unique_argmax": {"history_0": 0, "history_1": 1},
            "binary_best_constant_policy_disagreement_under_uniform_states": "1/2",
            "binary_worst_regret_of_any_constant_deterministic_policy": 1}


if __name__ == "__main__":
    import json
    print(json.dumps(mutation_oracle(), indent=2))
