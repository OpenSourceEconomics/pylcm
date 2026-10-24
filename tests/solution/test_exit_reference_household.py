"""A literal-loop reference household with declared same-period information.

Adapted from lcm-zoo tests/test_ecksteinCareerFamilyDecisions2019_reference_household.py
at commit ad2658dae4df00aaf53934f9011adc4c60dcc959. Economic primitives and loops
are preserved; signatures and annotations follow this repository. The reference
imports no production transition, mask or reducer helpers.

Plain Python with exact `Fraction` arithmetic and no pylcm import, so it shares no code
with the engine it checks. Three regimes — `single_f`, `single_m` and `couple` — over a
few periods, with one experience state per person.

Shocks, all discrete and drawn at the start of each period:

- `z`, a wage shock read by every regime at the same node;
- `offer`, a marriage offer read by singles: "marry" is available only on an offer;
- `partner`, the experience of the partner on offer, read by singles; its law depends
  on the single's own experience;
- `job_offer`, read only by singles: a single may work only on a job offer. It is
  hidden from a spouse deciding whether to divorce, who values their single target by
  integrating over it, in solve and in simulation alike;
- `theta`, a home-production shock read by couples and observed by a single deciding
  whether to marry. A formation exit must observe every shock the couple's
  participation reads, so the oracle refuses a hidden `theta`.

Per period and node, following the declared policy queries:

1. **Single stay.** Each single's max over its own alternatives, V^j.
2. **Couple participating policy.** The Pareto argmax over the couple's feasible
   alternatives with V^{jM}(a) >= V^j - Delta_j for both members, where V^j is the
   member's single stay value integrated over the hidden `job_offer`; -inf for both if
   no alternative passes.
3. **Couple decision.** With no participating alternative the couple divorces, and
   each member gets V^j - Delta_j.
4. **Single decision.** On an offer, marry iff both partners' values at the couple's
   participating policy strictly exceed their single stay values: the single's own at
   its observed `job_offer`, the partner's integrated over his. Delta_j enters only the
   participation thresholds. An empty own feasible set gives the single -inf and never
   forces a marriage.

A row that moves in period t acts in its target at the target's participating policy
(newlyweds) or stay choice at the realised `job_offer` (divorcees), and does not move
again until t + 1. The stored V of a period sums the per-node values weighted by the
node probabilities; the terminal period has no continuation.

Alternatives are indexed from 0; `MARRY` and `DIVORCE` follow the stay alternatives.

The tests check it against hand-computed values and its own rules. Hand values use
the default `Params` in the terminal period (t = 2), where a value is a flow. The
defaults make `offer`, `partner`, `job_offer` and `theta` degenerate: a marriage offer
and a job offer always arrive, the partner has the single's experience and `theta` is
0. At z = -1 and zero experience, a
single woman values not working at 1 + 5/2 = 7/2 and working at 2; a single man values
them at 7/2 and 3. A couple at `(0, 0)` has, for joint alternatives
`(0,0), (0,1), (1,0), (1,1)`:

- Q_f = (4, 5, 2, 3)
- Q_m = (4, 5/2, 9/2, 3)

so its Pareto sums are (4, 15/4, 13/4, 3).
"""

import itertools
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, replace
from fractions import Fraction as Frac
from typing import NamedTuple, TypedDict

import numpy as np
import pytest
from numpy.typing import NDArray

type ExactArray = NDArray[np.object_]
type BoolArray = NDArray[np.bool_]
type RegimeName = str


class Record(TypedDict):
    t: int
    regime: RegimeName
    e_own: int
    e_partner: int | None
    shocks: dict[str, int]
    choice: int
    acting_regime: RegimeName
    acting_choice: int


SEXES = ("f", "m")
SINGLE_ALTERNATIVES = (0, 1)
COUPLE_ALTERNATIVES = tuple(itertools.product((0, 1), (0, 1)))
MARRY = len(SINGLE_ALTERNATIVES)
DIVORCE = len(COUPLE_ALTERNATIVES)
NO_POLICY = -1
NEG_INF = float("-inf")


def _halves() -> tuple[Frac, Frac]:
    return (Frac(1, 2), Frac(1, 2))


@dataclass(frozen=True)
class Params:
    """Primitives of the reference household."""

    n_periods: int = 3
    "Periods `0 .. n_periods - 1`; the last one is terminal."
    max_experience: int = 2
    "Experience runs over `0 .. max_experience` and is capped there."
    wage_shock_nodes: tuple[Frac, ...] = (Frac(-1), Frac(1))
    wage_shock_probs: tuple[Frac, ...] = field(default_factory=_halves)
    offer_probability: Frac = Frac(1)
    "Probability that a single receives a marriage offer."
    same_experience_partner_probability: Frac = Frac(1)
    """Probability that the partner on offer has the single's experience.

    The rest is spread equally over the other experience levels.
    """
    job_offer_probability: Frac = Frac(1)
    "Probability that a single may work this period."
    home_shock_nodes: tuple[Frac, ...] = (Frac(0),)
    "Nodes of `theta`, added to each member's flow when neither spouse works."
    home_shock_probs: tuple[Frac, ...] = (Frac(1),)
    formation_observes_home_shock: bool = True
    "Whether a single sees `theta` when deciding to marry; `False` is refused."
    wage_level: dict[str, Frac] = field(
        default_factory=lambda: {"f": Frac(2), "m": Frac(3)}
    )
    "Wage at zero experience before the shock."
    wage_return: dict[str, Frac] = field(
        default_factory=lambda: {"f": Frac(1), "m": Frac(1)}
    )
    "Wage gain per year of experience."
    nonlabor_income_single: Frac = Frac(1)
    nonlabor_income_couple: Frac = Frac(1)
    leisure: dict[str, Frac] = field(
        default_factory=lambda: {"f": Frac(5, 2), "m": Frac(5, 2)}
    )
    "Utility of not working."
    match_quality: dict[str, Frac] = field(
        default_factory=lambda: {"f": Frac(1), "m": Frac(1)}
    )
    "Flow utility of being married."
    pareto_weight: dict[str, Frac] = field(
        default_factory=lambda: {"f": Frac(1, 2), "m": Frac(1, 2)}
    )
    divorce_cost: dict[str, Frac] = field(
        default_factory=lambda: {"f": Frac(0), "m": Frac(0)}
    )
    """Delta_j, subtracted from a member's single value on divorce.

    Tests may make it negative to move outside options up.
    """
    consumption_floor: Frac = Frac(0)
    "An alternative is feasible only if household consumption reaches it."
    discount_factor: Frac = Frac(9, 10)


class Policy(NamedTuple):
    """A selected policy and each member's value at it."""

    choice: int
    "The selected alternative, or `NO_POLICY`."
    values: dict[str, Frac | float]
    "Each member's value at `choice`; `-inf` under `NO_POLICY`."


class Values(NamedTuple):
    """Stored, node-integrated values by `(t, e)` and `(t, e_f, e_m)`."""

    single: dict[str, dict[tuple[int, int], Frac | float]]
    couple: dict[str, dict[tuple[int, int, int], Frac | float]]


@dataclass(frozen=True)
class Solution:
    """Stored values and per-node policies.

    - `V.single[s][t, e]` and `V.couple[s][t, e_f, e_m]` are the stored values.
    - `policy_single[s][t, e, z, offer, partner, job_offer, theta]` is a single's
      choice, up to `MARRY`.
    - `policy_couple[t, e_f, e_m, z, theta]` is a couple's choice, up to `DIVORCE`.

    Shock entries are node indices; `partner` is the partner's experience.
    """

    V: Values
    policy_single: dict[str, dict[tuple[int, ...], int]]
    policy_couple: dict[tuple[int, ...], int]


def unilateral(
    *,
    q: dict[str, ExactArray],
    feasible: BoolArray,
    v_exit: dict[str, Frac | float],
    weight: dict[str, Frac],
) -> int:
    """Choose under limited commitment: any member can force the exit."""
    choice = participating(q=q, feasible=feasible, outside=v_exit, weight=weight).choice
    return len(feasible) if choice == NO_POLICY else choice


def mutual(
    *,
    q: dict[str, ExactArray],
    feasible: BoolArray,
    v_exit: dict[str, Frac | float],
    weight: dict[str, Frac],
) -> int:
    """Choose under the dissolution-type `mutual` rule: the exit needs every member."""
    wanted = np.zeros(len(feasible), dtype=bool)
    for s, q_s in q.items():
        wanted |= np.array([q_s[a] >= v_exit[s] for a in range(len(feasible))])
    choice = _pareto_argmax(q=q, kept=feasible & wanted, weight=weight)
    return len(feasible) if choice == NO_POLICY else choice


def participating(
    *,
    q: dict[str, ExactArray],
    feasible: BoolArray,
    outside: Mapping[str, Frac | float],
    weight: dict[str, Frac],
) -> Policy:
    """Return the Pareto argmax over feasible alternatives every member accepts."""
    kept = feasible.copy()
    for s, q_s in q.items():
        kept &= np.array([q_s[a] >= outside[s] for a in range(len(feasible))])
    choice = _pareto_argmax(q=q, kept=kept, weight=weight)
    return _policy(q=q, choice=choice)


def consent(
    *, at_policy: Mapping[str, Frac | float], stay: Mapping[str, Frac | float]
) -> bool:
    """Return whether every party strictly gains at the policy over staying."""
    return all(at_policy[s] > stay[s] for s in at_policy)


def formation(
    *,
    q: dict[str, ExactArray],
    feasible: BoolArray,
    outside: Mapping[str, Frac | float],
    stay: Mapping[str, Frac | float],
    weight: dict[str, Frac],
) -> tuple[bool, dict[str, Frac | float]]:
    """Return whether a match forms and each party's value at the couple's policy.

    The couple runs its participating policy against its members' `outside` values;
    the parties then compare their values there with their own `stay` values.
    """
    policy = participating(q=q, feasible=feasible, outside=outside, weight=weight)
    return (consent(at_policy=policy.values, stay=stay), policy.values)


def solve(p: Params) -> Solution:
    """Run backward induction by explicit loops over periods, states and nodes."""
    _fail_if_formation_hides_home_shock(p)
    V = Values({s: {} for s in SEXES}, {s: {} for s in SEXES})
    policy_single = {s: {} for s in SEXES}
    policy_couple = {}
    for t in reversed(range(p.n_periods)):
        for s, e in itertools.product(SEXES, _experience(p)):
            node_values = []
            for z, offer, partner, job_offer, theta, prob in _single_nodes(p=p, e=e):
                choice, value = single_decision(
                    p=p,
                    V=V,
                    t=t,
                    sex=s,
                    e=e,
                    z=z,
                    offer=offer,
                    partner=partner,
                    job_offer=job_offer,
                    theta=theta,
                )
                policy_single[s][t, e, z, offer, partner, job_offer, theta] = choice
                node_values.append((prob, value))
            V.single[s][t, e] = _expect(node_values)
        for ef, em in itertools.product(_experience(p), repeat=2):
            node_values = {s: [] for s in SEXES}
            for z, theta, prob in _couple_nodes(p):
                choice, values = couple_decision(
                    p=p, V=V, t=t, ef=ef, em=em, z=z, theta=theta
                )
                policy_couple[t, ef, em, z, theta] = choice
                for s in SEXES:
                    node_values[s].append((prob, values[s]))
            for s in SEXES:
                V.couple[s][t, ef, em] = _expect(node_values[s])
    return Solution(V, policy_single, policy_couple)


def single_stay(
    *, p: Params, V: Values, t: int, sex: str, e: int, z: int, job_offer: int
) -> Policy:
    """Return a single's max over their own alternatives at `z` and `job_offer`."""
    wage = _wage(p=p, sex=sex, e=e, z=z)
    q = np.empty(len(SINGLE_ALTERNATIVES), dtype=object)
    feasible = np.empty(len(SINGLE_ALTERNATIVES), dtype=bool)
    for a, work in enumerate(SINGLE_ALTERNATIVES):
        consumption = work * wage + p.nonlabor_income_single
        feasible[a] = consumption >= p.consumption_floor and (job_offer or not work)
        q[a] = consumption + p.leisure[sex] * (1 - work)
        if t + 1 < p.n_periods:
            q[a] += (
                p.discount_factor * V.single[sex][t + 1, _next_e(p=p, e=e, work=work)]
            )
    return _policy(
        q={sex: q},
        choice=_pareto_argmax(q={sex: q}, kept=feasible, weight={sex: Frac(1)}),
    )


def single_outside(
    *, p: Params, V: Values, t: int, sex: str, e: int, z: int
) -> Frac | float:
    """Return a single's stay value to someone who does not see their `job_offer`."""
    return _expect(
        (
            (
                prob,
                single_stay(p=p, V=V, t=t, sex=sex, e=e, z=z, job_offer=job).values[
                    sex
                ],
            )
            for job, prob in enumerate(_job_offer_probs(p))
        )
    )


def couple_values(
    *, p: Params, V: Values, t: int, ef: int, em: int, z: int, theta: int
) -> tuple[dict[str, ExactArray], BoolArray]:
    """Return each member's value and the feasibility of each joint alternative."""
    e = {"f": ef, "m": em}
    q = {s: np.empty(len(COUPLE_ALTERNATIVES), dtype=object) for s in SEXES}
    feasible = np.empty(len(COUPLE_ALTERNATIVES), dtype=bool)
    for a, (work_f, work_m) in enumerate(COUPLE_ALTERNATIVES):
        work = {"f": work_f, "m": work_m}
        consumption = p.nonlabor_income_couple + sum(
            work[s] * _wage(p=p, sex=s, e=e[s], z=z) for s in SEXES
        )
        feasible[a] = consumption >= p.consumption_floor
        home = p.home_shock_nodes[theta] * (1 - work_f) * (1 - work_m)
        for s in SEXES:
            q[s][a] = (
                consumption / 2
                + p.leisure[s] * (1 - work[s])
                + p.match_quality[s]
                + home
            )
            if t + 1 < p.n_periods:
                next_state = (
                    _next_e(p=p, e=ef, work=work_f),
                    _next_e(p=p, e=em, work=work_m),
                )
                q[s][a] += p.discount_factor * V.couple[s][t + 1, *next_state]
    return (q, feasible)


def couple_participating(
    *, p: Params, V: Values, t: int, ef: int, em: int, z: int, theta: int
) -> Policy:
    """Return the couple's participating policy against its divorce values."""
    q, feasible = couple_values(p=p, V=V, t=t, ef=ef, em=em, z=z, theta=theta)
    return participating(
        q=q,
        feasible=feasible,
        outside=divorce_values(p=p, V=V, t=t, ef=ef, em=em, z=z),
        weight=p.pareto_weight,
    )


def divorce_values(
    *, p: Params, V: Values, t: int, ef: int, em: int, z: int
) -> dict[str, Frac | float]:
    """Return each member's value on divorce: single outside value minus Delta_j."""
    e = {"f": ef, "m": em}
    return {
        s: single_outside(p=p, V=V, t=t, sex=s, e=e[s], z=z) - p.divorce_cost[s]
        for s in SEXES
    }


def couple_decision(
    *, p: Params, V: Values, t: int, ef: int, em: int, z: int, theta: int
) -> tuple[int, dict[str, Frac | float]]:
    """Return a couple's choice and each member's value; either member can divorce."""
    policy = couple_participating(p=p, V=V, t=t, ef=ef, em=em, z=z, theta=theta)
    if policy.choice != NO_POLICY:
        return (policy.choice, policy.values)
    return (DIVORCE, divorce_values(p=p, V=V, t=t, ef=ef, em=em, z=z))


def single_decision(
    *,
    p: Params,
    V: Values,
    t: int,
    sex: str,
    e: int,
    z: int,
    offer: int,
    partner: int,
    job_offer: int,
    theta: int,
) -> tuple[int, Frac | float]:
    """Return a single's choice and value at its observed shocks."""
    own = single_stay(p=p, V=V, t=t, sex=sex, e=e, z=z, job_offer=job_offer)
    if not offer:
        return (own.choice, own.values[sex])
    other = _other(sex)
    stay = {
        sex: own.values[sex],
        other: single_outside(p=p, V=V, t=t, sex=other, e=partner, z=z),
    }
    ef, em = _couple_state(sex=sex, e_own=e, e_partner=partner)
    q, feasible = couple_values(p=p, V=V, t=t, ef=ef, em=em, z=z, theta=theta)
    outside = divorce_values(p=p, V=V, t=t, ef=ef, em=em, z=z)
    marries, at_policy = formation(
        q=q, feasible=feasible, outside=outside, stay=stay, weight=p.pareto_weight
    )
    if marries:
        return (MARRY, at_policy[sex])
    return (own.choice, own.values[sex])


def partner_probs(*, p: Params, e: int) -> tuple[Frac, ...]:
    """Return the law of the partner's experience for a single with experience `e`."""
    same = p.same_experience_partner_probability
    rest = (1 - same) / p.max_experience
    return tuple(same if k == e else rest for k in _experience(p))


def draw(*, probs: tuple[Frac, ...], u: Frac) -> int:
    """Return the smallest category whose cumulative probability exceeds `u`."""
    cumulative = Frac(0)
    for k, prob in enumerate(probs[:-1]):
        cumulative += prob
        if u < cumulative:
            return k
    return len(probs) - 1


def simulate(
    *,
    p: Params,
    solution: Solution,
    initial: list[tuple[str, str, int, int]],
    uniforms: list[list[dict[str, Frac]]],
) -> list[list[Record]]:
    """Simulate rows over a fixed table of uniforms, re-choosing at each draw.

    Each row of `initial` is `(regime, sex, e_own, e_partner)` with `regime` either
    `"single"` or `"couple"`; a single's `e_partner` is ignored. `uniforms[i][t]` maps
    each shock name to the uniform that row `i` uses for it in period `t`; each shock's
    law turns it into a node through `draw`.

    Each record holds the period's starting `regime`, `e_own` and `e_partner`, the drawn
    `shocks`, the `choice` there, and `acting_regime` and `acting_choice`, where the row
    acts and the alternative it takes there.
    """
    return [
        _simulate_row(p=p, V=solution.V, start=start, uniforms=uniforms[i])
        for i, start in enumerate(initial)
    ]


def _simulate_row(
    *,
    p: Params,
    V: Values,
    start: tuple[str, str, int, int],
    uniforms: list[dict[str, Frac]],
) -> list[Record]:
    regime, sex, e_own, e_partner = start
    rows: list[Record] = []
    for t in range(p.n_periods):
        u = uniforms[t]
        shocks = {
            "z": draw(probs=p.wage_shock_probs, u=u["z"]),
            "offer": draw(
                probs=(1 - p.offer_probability, p.offer_probability), u=u["offer"]
            ),
            "partner": draw(probs=partner_probs(p=p, e=e_own), u=u["partner"]),
            "job_offer": draw(probs=_job_offer_probs(p), u=u["job_offer"]),
            "theta": draw(probs=p.home_shock_probs, u=u["theta"]),
        }
        record: Record = {
            "t": t,
            "regime": regime,
            "e_own": e_own,
            "e_partner": e_partner if regime == "couple" else None,
            "shocks": shocks,
            "choice": NO_POLICY,
            "acting_regime": regime,
            "acting_choice": NO_POLICY,
        }
        if regime == "single":
            choice, _ = single_decision(
                p=p,
                V=V,
                t=t,
                sex=sex,
                e=e_own,
                z=shocks["z"],
                offer=shocks["offer"],
                partner=shocks["partner"],
                job_offer=shocks["job_offer"],
                theta=shocks["theta"],
            )
            moves = choice == MARRY
            e_partner = shocks["partner"] if moves else e_partner
        else:
            ef, em = _couple_state(sex=sex, e_own=e_own, e_partner=e_partner)
            choice, _ = couple_decision(
                p=p, V=V, t=t, ef=ef, em=em, z=shocks["z"], theta=shocks["theta"]
            )
            moves = choice == DIVORCE
        acting = {"single": "couple", "couple": "single"}[regime] if moves else regime
        if acting == "single":
            policy = single_stay(
                p=p,
                V=V,
                t=t,
                sex=sex,
                e=e_own,
                z=shocks["z"],
                job_offer=shocks["job_offer"],
            )
            acting_choice = policy.choice if moves else choice
            _fail_if_no_policy(choice=acting_choice, t=t, start=start)
            e_own = _next_e(p=p, e=e_own, work=SINGLE_ALTERNATIVES[acting_choice])
        else:
            ef, em = _couple_state(sex=sex, e_own=e_own, e_partner=e_partner)
            policy = couple_participating(
                p=p, V=V, t=t, ef=ef, em=em, z=shocks["z"], theta=shocks["theta"]
            )
            acting_choice = policy.choice if moves else choice
            _fail_if_no_policy(choice=acting_choice, t=t, start=start)
            e_own, e_partner = _advance_couple(
                p=p, sex=sex, e_own=e_own, e_partner=e_partner, joint=acting_choice
            )
        record |= {
            "choice": choice,
            "acting_regime": acting,
            "acting_choice": acting_choice,
        }
        rows.append(record)
        regime = acting
    return rows


def _fail_if_formation_hides_home_shock(p: Params) -> None:
    if not p.formation_observes_home_shock:
        msg = (
            "A formation exit must observe `theta`: the couple's same-period "
            "participation reads it. To keep it hidden, decide the match against "
            "next-period values instead."
        )
        raise ValueError(msg)


def _fail_if_no_policy(
    *, choice: int, t: int, start: tuple[str, str, int, int]
) -> None:
    if choice == NO_POLICY:
        msg = f"A row starting at {start} has no admissible action in period {t}."
        raise ValueError(msg)


def _single_nodes(*, p: Params, e: int) -> list[tuple[int, int, int, int, int, Frac]]:
    offer_probs = (1 - p.offer_probability, p.offer_probability)
    return [
        (
            z,
            offer,
            partner,
            job,
            theta,
            p.wage_shock_probs[z]
            * offer_probs[offer]
            * partner_prob
            * job_prob
            * p.home_shock_probs[theta],
        )
        for z in range(len(p.wage_shock_nodes))
        for offer in (0, 1)
        for partner, partner_prob in enumerate(partner_probs(p=p, e=e))
        for job, job_prob in enumerate(_job_offer_probs(p))
        for theta in range(len(p.home_shock_nodes))
    ]


def _job_offer_probs(p: Params) -> tuple[Frac, Frac]:
    return (1 - p.job_offer_probability, p.job_offer_probability)


def _couple_nodes(p: Params) -> list[tuple[int, int, Frac]]:
    return [
        (z, theta, p.wage_shock_probs[z] * p.home_shock_probs[theta])
        for z in range(len(p.wage_shock_nodes))
        for theta in range(len(p.home_shock_nodes))
    ]


def _expect(pairs: Iterable[tuple[Frac, Frac | float]]) -> Frac | float:
    """Return the probability-weighted sum, skipping zero-probability nodes."""
    total = Frac(0)
    for prob, value in pairs:
        if prob != 0:
            total += prob * value
    return total


def _policy(*, q: dict[str, ExactArray], choice: int) -> Policy:
    if choice == NO_POLICY:
        return Policy(choice, dict.fromkeys(q, NEG_INF))
    return Policy(choice, {s: q[s][choice] for s in q})


def _pareto_argmax(
    *, q: dict[str, ExactArray], kept: BoolArray, weight: dict[str, Frac]
) -> int:
    """Return the kept alternative with the largest weighted sum; ties go lowest."""
    best, best_value = (NO_POLICY, None)
    for a in range(len(kept)):
        if not kept[a]:
            continue
        value = sum(weight[s] * q[s][a] for s in q if weight[s] != 0)
        if best == NO_POLICY or value > best_value:
            best, best_value = (a, value)
    return best


def _experience(p: Params) -> range:
    return range(p.max_experience + 1)


def _wage(*, p: Params, sex: str, e: int, z: int) -> Frac:
    return p.wage_level[sex] + p.wage_return[sex] * e + p.wage_shock_nodes[z]


def _next_e(*, p: Params, e: int, work: int) -> int:
    return min(e + work, p.max_experience)


def _other(sex: str) -> str:
    return "m" if sex == "f" else "f"


def _couple_state(*, sex: str, e_own: int, e_partner: int) -> tuple[int, int]:
    return (e_own, e_partner) if sex == "f" else (e_partner, e_own)


def _advance_couple(
    *, p: Params, sex: str, e_own: int, e_partner: int, joint: int
) -> tuple[int, int]:
    work_f, work_m = COUPLE_ALTERNATIVES[joint]
    work_own, work_partner = (work_f, work_m) if sex == "f" else (work_m, work_f)
    return (
        _next_e(p=p, e=e_own, work=work_own),
        _next_e(p=p, e=e_partner, work=work_partner),
    )


TERMINAL = 2
DISSOLUTION_RULES = {"unilateral": unilateral, "mutual": mutual}
HALF = {"f": Frac(1, 2), "m": Frac(1, 2)}
HOME_THETA = {
    "home_shock_nodes": (Frac(0), Frac(2)),
    "home_shock_probs": (Frac(1, 2), Frac(1, 2)),
}


def _q(*, f: list[Frac | float], m: list[Frac | float]) -> dict[str, ExactArray]:
    return {"f": np.array(f, dtype=object), "m": np.array(m, dtype=object)}


def _all(n: int) -> BoolArray:
    return np.ones(n, dtype=bool)


def _couple_at_terminal_origin(p: Params) -> tuple[int, dict[str, Frac | float]]:
    """Return the couple's choice and values at `(0, 0)`, z = -1, theta = 0."""
    solution = solve(p)
    choice = solution.policy_couple[TERMINAL, 0, 0, 0, 0]
    return (
        choice,
        couple_decision(p=p, V=solution.V, t=TERMINAL, ef=0, em=0, z=0, theta=0)[1],
    )


def _uniform_table(n_periods: int, **overrides: Frac) -> list[dict[str, Frac]]:
    """Return per-period uniforms, all 1/4 unless named as `<shock>_<t>`."""
    return [
        {
            shock: overrides.get(f"{shock}_{t}", Frac(1, 4))
            for shock in ("z", "offer", "partner", "job_offer", "theta")
        }
        for t in range(n_periods)
    ]


def test_unilateral_drops_the_best_alternative_a_member_vetoes() -> None:
    """Q_f = (7, 4, 3), Q_m = (1, 3, 2), exits (5/2, 3/2): m vetoes 0, so 1 wins."""
    q = _q(f=[7, 4, 3], m=[1, 3, 2])
    assert (
        unilateral(
            q=q,
            feasible=_all(3),
            v_exit={"f": Frac(5, 2), "m": Frac(3, 2)},
            weight=HALF,
        )
        == 1
    )


def test_mutual_keeps_the_best_alternative_one_member_wants() -> None:
    """The same cell under `mutual`: f still wants alternative 0, so it is chosen."""
    q = _q(f=[7, 4, 3], m=[1, 3, 2])
    assert (
        mutual(
            q=q,
            feasible=_all(3),
            v_exit={"f": Frac(5, 2), "m": Frac(3, 2)},
            weight=HALF,
        )
        == 0
    )


def test_a_couple_divorces_when_every_alternative_violates_a_participation() -> None:
    """m's divorce value 7/2 + 3/5 = 41/10 beats every Q_m but 9/2, which f vetoes."""
    p = replace(Params(), divorce_cost={"f": Frac(0), "m": Frac(-3, 5)})
    assert _couple_at_terminal_origin(p) == (
        DIVORCE,
        {"f": Frac(7, 2), "m": Frac(41, 10)},
    )


def test_a_couple_moves_to_the_best_alternative_both_members_accept() -> None:
    """Exits (41/10, 2) remove alternative 0 (f) and keep 1 at values (5, 5/2)."""
    p = replace(Params(), divorce_cost={"f": Frac(-3, 5), "m": Frac(3, 2)})
    assert _couple_at_terminal_origin(p) == (1, {"f": Frac(5), "m": Frac(5, 2)})


def test_a_single_marries_when_both_gain_at_the_participating_policy() -> None:
    """Outside (7/2, 7/2) keeps only (0, 0) at (4, 4); both strictly gain."""
    assert solve(Params()).policy_single["f"][TERMINAL, 0, 0, 1, 0, 1, 0] == MARRY


def test_a_tie_with_the_partners_single_value_keeps_her_single() -> None:
    """At z = 1 the policy is (1, 1) at (5, 5), and his single value is 5.

    She marries at z = -1 for 4 and stays single at z = 1 for 4, so she is worth 4.
    """
    assert solve(Params()).V.single["f"][TERMINAL, 0] == 4


def test_a_single_stays_single_when_no_couple_policy_participates() -> None:
    """m's match quality 2/5 leaves no alternative above both outside values 7/2."""
    p = replace(Params(), match_quality={"f": Frac(1), "m": Frac(2, 5)})
    assert solve(p).policy_single["f"][TERMINAL, 0, 0, 1, 0, 1, 0] == 0


def test_stored_values_average_the_nodes_and_discount_the_next_period() -> None:
    """Without marriage, a single woman with no experience is worth 147/20 at t = 1.

    Terminal: V(0) = (7/2 + 4) / 2 = 15/4 and V(1) = (7/2 + 5) / 2 = 17/4. At t = 1,
    z = -1 gives max(7/2 + 9/10 * 15/4, 2 + 9/10 * 17/4) = 55/8 and z = 1 gives
    max(55/8, 4 + 9/10 * 17/4) = 313/40; their mean is 147/20.
    """
    p = replace(Params(), match_quality={"f": Frac(-100), "m": Frac(-100)})
    assert solve(p).V.single["f"][1, 0] == Frac(147, 20)


def test_a_single_weights_the_nodes_by_their_probabilities() -> None:
    """Without marriage and weights (1/4, 3/4): 1/4 * 7/2 + 3/4 * 4 = 31/8."""
    p = replace(
        Params(),
        wage_shock_probs=(Frac(1, 4), Frac(3, 4)),
        match_quality={"f": Frac(-100), "m": Frac(-100)},
    )
    assert solve(p).V.single["f"][TERMINAL, 0] == Frac(31, 8)


def test_a_couple_weights_the_nodes_by_their_probabilities() -> None:
    """At `(0, 0)` f gets 4 at z = -1 and 5 at z = 1: 1/4 * 4 + 3/4 * 5 = 19/4."""
    p = replace(Params(), wage_shock_probs=(Frac(1, 4), Frac(3, 4)))
    assert solve(p).V.couple["f"][TERMINAL, 0, 0] == Frac(19, 4)


@pytest.mark.parametrize("rule", DISSOLUTION_RULES)
def test_equal_pareto_sums_go_to_the_lowest_alternative(rule: str) -> None:
    """Q_f = (1, 2) and Q_m = (2, 1) give both alternatives the sum 3/2."""
    q = _q(f=[1, 2], m=[2, 1])
    assert (
        DISSOLUTION_RULES[rule](
            q=q, feasible=_all(2), v_exit={"f": 0, "m": 0}, weight=HALF
        )
        == 0
    )


@pytest.mark.parametrize("rule", DISSOLUTION_RULES)
def test_an_exact_tie_keeps_staying(rule: str) -> None:
    q = _q(f=[1], m=[1])
    assert (
        DISSOLUTION_RULES[rule](
            q=q, feasible=_all(1), v_exit={"f": 1, "m": 1}, weight=HALF
        )
        == 0
    )


@pytest.mark.parametrize("rule", DISSOLUTION_RULES)
def test_an_exit_one_ulp_above_the_tie_is_taken(rule: str) -> None:
    above = Frac(np.nextafter(1.0, np.inf))
    q = _q(f=[1], m=[1])
    assert (
        DISSOLUTION_RULES[rule](
            q=q, feasible=_all(1), v_exit={"f": above, "m": above}, weight=HALF
        )
        == 1
    )


def test_a_couple_stays_when_a_divorce_value_ties_its_value() -> None:
    """f's divorce value 7/2 + 1/2 is exactly her 4 at alternative 0."""
    p = replace(Params(), divorce_cost={"f": Frac(-1, 2), "m": Frac(0)})
    assert _couple_at_terminal_origin(p)[0] == 0


def test_a_couple_divorces_when_a_divorce_value_is_one_ulp_above_its_value() -> None:
    above = Frac(np.nextafter(4.0, np.inf))
    p = replace(Params(), divorce_cost={"f": Frac(7, 2) - above, "m": Frac(0)})
    assert _couple_at_terminal_origin(p)[0] == DIVORCE


@pytest.mark.parametrize("rule", DISSOLUTION_RULES)
def test_an_empty_feasible_set_dissolves(rule: str) -> None:
    q = _q(f=[9, 9], m=[9, 9])
    choice = DISSOLUTION_RULES[rule](
        q=q, feasible=np.zeros(2, dtype=bool), v_exit={"f": 0, "m": 0}, weight=HALF
    )
    assert choice == 2


def test_a_couple_with_no_feasible_alternative_takes_its_divorce_values() -> None:
    """Couple income -10 + wages stays below the floor 0 at `(0, 0)`, z = -1."""
    p = replace(Params(), nonlabor_income_couple=Frac(-10))
    assert _couple_at_terminal_origin(p) == (
        DIVORCE,
        {"f": Frac(7, 2), "m": Frac(7, 2)},
    )


def test_unilateral_ignores_a_member_whose_exit_is_minus_infinity() -> None:
    q = _q(f=[1], m=[-(10**300)])
    assert (
        unilateral(q=q, feasible=_all(1), v_exit={"f": 0, "m": NEG_INF}, weight=HALF)
        == 0
    )


def test_mutual_never_exits_to_a_target_at_minus_infinity() -> None:
    q = _q(f=[-(10**300)], m=[-(10**300)])
    assert (
        mutual(q=q, feasible=_all(1), v_exit={"f": NEG_INF, "m": NEG_INF}, weight=HALF)
        == 0
    )


def test_no_consent_is_given_to_a_target_at_minus_infinity() -> None:
    assert not consent(
        at_policy={"f": NEG_INF, "m": Frac(9)},
        stay={"f": Frac(-(10**300)), "m": Frac(0)},
    )


def test_an_empty_own_feasible_set_does_not_force_a_marriage() -> None:
    """Her stay value is -inf, but he keeps his single 5 over the couple's 4."""
    assert not consent(
        at_policy={"f": Frac(4), "m": Frac(4)}, stay={"f": NEG_INF, "m": Frac(5)}
    )


def test_an_empty_own_feasible_set_still_marries_with_consent() -> None:
    assert consent(
        at_policy={"f": Frac(4), "m": Frac(6)}, stay={"f": NEG_INF, "m": Frac(5)}
    )


def test_a_single_with_nothing_feasible_stays_without_a_policy_if_refused() -> None:
    """A floor of 5/2 leaves her nothing at z = -1 (consumption 1 or 2), while he can
    work for 3; his match quality -100 makes him refuse."""
    p = replace(
        Params(),
        consumption_floor=Frac(5, 2),
        match_quality={"f": Frac(1), "m": Frac(-100)},
    )
    assert solve(p).policy_single["f"][TERMINAL, 0, 0, 1, 0, 1, 0] == NO_POLICY


def test_the_zero_cost_witness_marries_at_the_participating_policy() -> None:
    """Actions (20, 0) and (6, 6), stay values (5, 5): (20, 0) fails his participation
    and (6, 6) passes, where both strictly gain."""
    stay = {"f": Frac(5), "m": Frac(5)}
    choice = formation(
        q=_q(f=[20, 6], m=[0, 6]),
        feasible=_all(2),
        outside=stay,
        stay=stay,
        weight=HALF,
    )
    assert choice == (True, {"f": Frac(6), "m": Frac(6)})


def test_the_positive_cost_witness_does_not_marry() -> None:
    """Actions (20, 4) and (6, 6), costs (0, 2): his threshold 3 keeps (20, 4), which
    has the larger Pareto sum, and he loses there (4 < 5).

    A fresh argmax over the actions both parties strictly prefer would pick (6, 6).
    """
    stay = {"f": Frac(5), "m": Frac(5)}
    outside = {"f": Frac(5), "m": Frac(3)}
    choice = formation(
        q=_q(f=[20, 6], m=[4, 6]),
        feasible=_all(2),
        outside=outside,
        stay=stay,
        weight=HALF,
    )
    assert choice == (False, {"f": Frac(20), "m": Frac(4)})


def _unequal_leisure() -> Params:
    """Leisure (3/2, 4), one period: at z = 1, f with experience 1 and m with 0 are
    both worth 5 single. The couple has Q_f = (3, 5, 7/2, 11/2), Q_m = (11/2, 7/2,
    15/2, 11/2): alternatives 2 and 3 tie at the Pareto sum 11/2, so the
    unconstrained choice is 2, where she gets 7/2; the participating policy is 3."""
    return replace(
        Params(),
        n_periods=1,
        leisure={"f": Frac(3, 2), "m": Frac(4)},
        same_experience_partner_probability=Frac(1, 3),
    )


def test_a_single_marries_where_the_unconstrained_couple_choice_fails_her() -> None:
    assert solve(_unequal_leisure()).policy_single["f"][0, 1, 1, 1, 0, 1, 0] == MARRY


def test_newlyweds_act_at_the_participating_policy_not_the_unconstrained_one() -> None:
    p = _unequal_leisure()
    row = simulate(
        p=p,
        solution=solve(p),
        initial=[("single", "f", 1, 0)],
        uniforms=[_uniform_table(1, z_0=Frac(3, 4))],
    )
    assert (row[0][0]["choice"], row[0][0]["acting_choice"]) == (MARRY, 3)


@pytest.mark.parametrize("member", SEXES)
def test_a_higher_divorce_value_never_makes_staying_more_likely(member: str) -> None:
    """With the continuation fixed, the set of dissolving cells grows as cost falls.

    The detector fires: the lowest cost dissolves strictly more cells than the highest.
    """
    base = Params()
    V = solve(base).V
    costs = (Frac(2), Frac(1, 2), Frac(0), Frac(-1, 4), Frac(-1, 2), Frac(-1), Frac(-3))
    dissolved = []
    for cost in costs:
        p = replace(base, divorce_cost={**base.divorce_cost, member: cost})
        cells = {
            (t, ef, em, z)
            for t in range(p.n_periods)
            for z in range(len(p.wage_shock_nodes))
            for ef, em in itertools.product(_experience(p), repeat=2)
            if couple_decision(p=p, V=V, t=t, ef=ef, em=em, z=z, theta=0)[0] == DIVORCE
        }
        dissolved.append(cells)
    nested = all((lo <= hi for lo, hi in itertools.pairwise(dissolved)))
    assert (nested, dissolved[0] < dissolved[-1]) == (True, True)


def test_marry_is_never_chosen_without_an_offer() -> None:
    p = replace(Params(), offer_probability=Frac(1, 2))
    policy = solve(p).policy_single
    no_offer = {
        choice
        for s in SEXES
        for (_, _, _, offer, *_), choice in policy[s].items()
        if not offer
    }
    assert MARRY not in no_offer


def test_a_single_weights_marriage_by_the_offer_probability() -> None:
    """At z = -1 she marries on an offer for 4 and is worth 7/2 without one; at z = 1
    she is worth 4 either way: (1/2 * 4 + 1/2 * 7/2) / 2 + 4 / 2 = 31/8."""
    p = replace(Params(), offer_probability=Frac(1, 2))
    assert solve(p).V.single["f"][TERMINAL, 0] == Frac(31, 8)


def test_a_partner_with_more_experience_refuses_her() -> None:
    """At z = -1 a partner with experience 1 is worth 4 single, and the couple's
    participating policy (0, 0) gives him exactly 4. With experience 2 he is worth 5,
    and no alternative keeps both at their outside values (7/2, 5)."""
    solution = solve(replace(Params(), same_experience_partner_probability=Frac(1, 3)))
    choices = [
        solution.policy_single["f"][TERMINAL, 0, 0, 1, k, 1, 0] for k in range(3)
    ]
    assert choices == [MARRY, 0, 0]


def test_the_partner_law_puts_the_rest_on_other_experience_levels() -> None:
    p = replace(Params(), same_experience_partner_probability=Frac(1, 2))
    assert partner_probs(p=p, e=1) == (Frac(1, 4), Frac(1, 2), Frac(1, 4))


def test_a_married_rows_partner_has_the_drawn_experience() -> None:
    """With experience 1 she draws a partner with experience 0 and marries him; his
    experience next period follows the couple's choice."""
    p = replace(Params(), same_experience_partner_probability=Frac(1, 3))
    solution = solve(p)
    rows = simulate(
        p=p,
        solution=solution,
        initial=[("single", "f", 1, 0)],
        uniforms=[_uniform_table(3)],
    )[0]
    assert (rows[0]["choice"], rows[0]["shocks"]["partner"], rows[1]["e_partner"]) == (
        MARRY,
        0,
        COUPLE_ALTERNATIVES[rows[0]["acting_choice"]][1],
    )


def test_a_partners_experience_is_never_redrawn() -> None:
    """Partner uniforms after her marriage leave her path unchanged.

    The detector is the period-0 draw: it decides whom she marries.
    """
    p = replace(Params(), same_experience_partner_probability=Frac(1, 3))
    solution = solve(p)
    paths = [
        [
            (r["regime"], r["e_own"], r["e_partner"], r["choice"])
            for r in simulate(
                p=p,
                solution=solution,
                initial=[("single", "f", 1, 0)],
                uniforms=[_uniform_table(3, partner_1=u, partner_2=u)],
            )[0]
        ]
        for u in (Frac(0), Frac(99, 100))
    ]
    assert paths[0] == paths[1]


def test_draw_never_returns_a_zero_probability_category() -> None:
    probs = (Frac(1, 2), Frac(0), Frac(1, 2))
    assert [draw(probs=probs, u=u) for u in (Frac(0), Frac(1, 2), Frac(99, 100))] == [
        0,
        2,
        2,
    ]


def test_a_formation_that_hides_the_home_shock_is_refused() -> None:
    p = replace(Params(), **HOME_THETA, formation_observes_home_shock=False)
    with pytest.raises(ValueError, match="must observe `theta`"):
        solve(p)


def test_a_single_conditions_marriage_on_the_observed_home_shock() -> None:
    """She marries for 4 and 6 at z = -1 (theta 0 and 2), and at z = 1 she stays single
    for 4 at theta = 0 (his single 5 ties his 5) and marries for 6 at theta = 2."""
    p = replace(Params(), **HOME_THETA)
    assert solve(p).V.single["f"][TERMINAL, 0] == 5


def test_a_marriage_decision_reads_the_observed_home_shock() -> None:
    """At z = 1 she stays single and works at theta = 0, and at theta = 2 she marries
    and the newlyweds take (0, 0) at (6, 6)."""
    p = replace(Params(), **HOME_THETA, n_periods=1)
    solution = solve(p)
    rows = [
        simulate(
            p=p,
            solution=solution,
            initial=[("single", "f", 0, 0)],
            uniforms=[_uniform_table(1, z_0=Frac(3, 4), theta_0=u)],
        )[0][0]
        for u in (Frac(1, 4), Frac(3, 4))
    ]
    assert [(r["choice"], r["acting_choice"]) for r in rows] == [(1, 1), (MARRY, 0)]


def test_a_single_works_only_on_a_job_offer() -> None:
    """At z = 1 working pays her 4 over 7/2, but only with an offer."""
    policy = solve(replace(Params(), job_offer_probability=Frac(1, 2))).policy_single
    assert [policy["f"][TERMINAL, 0, 1, 0, 0, job, 0] for job in (0, 1)] == [0, 1]


def test_a_divorce_threshold_integrates_the_hidden_job_offer() -> None:
    """With job-offer odds 1/2 at z = 1, f's single value is (7/2 + 4) / 2 = 15/4 and
    m's is (7/2 + 5) / 2 = 17/4; Delta_m = -1/2 puts his threshold at 19/4, so (1, 1)
    at (5, 5) participates. Seeing an offer would put it at 11/2 and dissolve them."""
    p = replace(
        Params(),
        job_offer_probability=Frac(1, 2),
        divorce_cost={"f": Frac(0), "m": Frac(-1, 2)},
    )
    assert solve(p).policy_couple[TERMINAL, 0, 0, 1, 0] == 3


def test_a_divorcee_gets_the_single_value_integrated_over_job_offers() -> None:
    """Delta_m = -2 puts his threshold at 25/4, above every Q_m, so they divorce."""
    p = replace(
        Params(),
        job_offer_probability=Frac(1, 2),
        divorce_cost={"f": Frac(0), "m": Frac(-2)},
    )
    V = solve(p).V
    assert couple_decision(p=p, V=V, t=TERMINAL, ef=0, em=0, z=1, theta=0) == (
        DIVORCE,
        {"f": Frac(15, 4), "m": Frac(25, 4)},
    )


def test_a_divorce_ignores_the_realised_job_offer_but_the_divorcee_acts_on_it() -> None:
    p = replace(
        Params(),
        n_periods=1,
        job_offer_probability=Frac(1, 2),
        divorce_cost={"f": Frac(0), "m": Frac(-2)},
    )
    solution = solve(p)
    rows = [
        simulate(
            p=p,
            solution=solution,
            initial=[("couple", "m", 0, 0)],
            uniforms=[_uniform_table(1, z_0=Frac(3, 4), job_offer_0=u)],
        )[0][0]
        for u in (Frac(1, 4), Frac(3, 4))
    ]
    assert [(r["choice"], r["acting_choice"]) for r in rows] == [
        (DIVORCE, 0),
        (DIVORCE, 1),
    ]


def test_consent_integrates_the_partners_job_offer() -> None:
    """At z = 1 and job-offer odds 1/2 the couple's policy is (1, 1) at (5, 5); his
    single value 17/4 integrates his offer, so he consents (with an offer he would be
    worth 5 and the tie would keep him single)."""
    p = replace(Params(), job_offer_probability=Frac(1, 2))
    assert solve(p).policy_single["f"][TERMINAL, 0, 1, 1, 0, 1, 0] == MARRY


def test_her_consent_reads_her_realised_job_offer() -> None:
    """With experience 2 at z = -1 she is worth 7/2 single without a job offer and 4
    with one; the couple's policy with a partner of experience 0 is (0, 0) at (4, 4)
    and his integrated single value is 7/2, so she marries only without an offer."""
    p = replace(Params(), n_periods=1, job_offer_probability=Frac(1, 2))
    policy = solve(p).policy_single["f"]
    assert [policy[0, 2, 0, 1, 0, job, 0] for job in (0, 1)] == [MARRY, 1]


def test_a_simulation_reproduces_the_solved_choices() -> None:
    """Every row's choice at its realised shocks equals the solved policy there.

    The table includes a marriage and a divorce, so both moves are exercised.
    """
    p = replace(
        Params(),
        divorce_cost={"f": Frac(-3, 10), "m": Frac(-3, 10)},
        offer_probability=Frac(1, 2),
        same_experience_partner_probability=Frac(1, 2),
        **HOME_THETA,
    )
    solution = solve(p)
    initial = [
        ("single", "f", 0, 0),
        ("single", "m", 0, 0),
        ("couple", "f", 0, 0),
        ("couple", "m", 2, 1),
        ("single", "f", 2, 0),
    ]
    rng = np.random.default_rng(seed=20261023)
    uniforms = [
        [
            {
                s: Frac(int(rng.integers(0, 100)), 100)
                for s in ("z", "offer", "partner", "job_offer", "theta")
            }
            for _ in range(p.n_periods)
        ]
        for _ in initial
    ]
    mismatches, moves = ([], set())
    for (_, sex, _, _), rows in zip(
        initial,
        simulate(p=p, solution=solution, initial=initial, uniforms=uniforms),
        strict=True,
    ):
        for r in rows:
            k = r["shocks"]
            if r["regime"] == "single":
                key = (
                    r["t"],
                    r["e_own"],
                    k["z"],
                    k["offer"],
                    k["partner"],
                    k["job_offer"],
                    k["theta"],
                )
                solved = solution.policy_single[sex][key]
                moves |= {"marry"} if r["choice"] == MARRY else set()
            else:
                assert r["e_partner"] is not None
                ef, em = _couple_state(
                    sex=sex, e_own=r["e_own"], e_partner=r["e_partner"]
                )
                solved = solution.policy_couple[r["t"], ef, em, k["z"], k["theta"]]
                moves |= {"divorce"} if r["choice"] == DIVORCE else set()
            if solved != r["choice"]:
                mismatches.append((sex, r, solved))
    assert (mismatches, moves) == ([], {"marry", "divorce"})


def test_simulation_divorce_threshold_hides_job_offer() -> None:
    """A realised offer cannot change the already integrated divorce threshold."""
    p = replace(
        Params(),
        n_periods=1,
        job_offer_probability=Frac(1, 2),
        divorce_cost={"f": Frac(0), "m": Frac(-1, 2)},
    )
    solution = solve(p)
    rows = [
        simulate(
            p=p,
            solution=solution,
            initial=[("couple", "m", 0, 0)],
            uniforms=[_uniform_table(n_periods=1, z_0=Frac(3, 4), job_offer_0=uniform)],
        )[0][0]
        for uniform in (Frac(1, 4), Frac(3, 4))
    ]
    assert [(row["choice"], row["acting_choice"]) for row in rows] == [(3, 3), (3, 3)]


def test_empty_own_set_marries_a_willing_partner_end_to_end() -> None:
    """Her single consumption is below the floor; joint work gives both six."""
    p = replace(
        Params(),
        n_periods=1,
        consumption_floor=Frac(5),
        match_quality={"f": Frac(2), "m": Frac(2)},
    )
    solution = solve(p)
    rows = simulate(
        p=p,
        solution=solution,
        initial=[("single", "f", 0, 0)],
        uniforms=[_uniform_table(n_periods=1, z_0=Frac(3, 4))],
    )
    assert (
        single_stay(p=p, V=solution.V, t=0, sex="f", e=0, z=1, job_offer=1).choice,
        solution.policy_single["f"][0, 0, 1, 1, 0, 1, 0],
        rows[0][0]["choice"],
        rows[0][0]["acting_choice"],
    ) == (NO_POLICY, MARRY, MARRY, 3)


def test_opposing_movers_act_once_in_final_regime() -> None:
    p = replace(Params(), n_periods=1)
    solution = solve(p)
    rows = simulate(
        p=p,
        solution=solution,
        initial=[("single", "f", 0, 0), ("couple", "m", 2, 1)],
        uniforms=[
            _uniform_table(n_periods=1),
            _uniform_table(n_periods=1, z_0=Frac(3, 4)),
        ],
    )
    assert [
        [(row["choice"], row["acting_regime"], row["acting_choice"]) for row in subject]
        for subject in rows
    ] == [[(MARRY, "couple", 0)], [(DIVORCE, "single", 1)]]


def test_nonterminal_collective_value_with_formation_and_divorce_active() -> None:
    """At t=1, (0,0), z=-1: action 0 gives 4 + 9/10*9/2 = 161/20.

    At z=1 the competing joint-work action gives 5 + 9/10*5 = 19/2.
    Terminal couple V is (4+5)/2=9/2 at (0,0), and (4+6)/2=5 at (1,1).
    Singles can marry at low z; a high-experience male couple divorces at high z.
    Their mean is 351/40, including the full terminal routing decision.
    """
    p = replace(Params(), n_periods=2)
    solution = solve(p)
    assert (
        solution.V.couple["f"][0, 0, 0],
        solution.V.couple["m"][0, 0, 0],
        solution.policy_single["f"][1, 0, 0, 1, 0, 1, 0],
        solution.policy_couple[1, 1, 2, 1, 0],
    ) == (Frac(351, 40), Frac(351, 40), MARRY, DIVORCE)
