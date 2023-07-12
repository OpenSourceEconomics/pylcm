"""Define example model specifications."""
import jax.numpy as jnp

RETIREMENT_AGE = 65
N_CHOICE_GRID_POINTS = 500
N_STATE_GRID_POINTS = 100


def phelps_deaton_utility_with_shock(
    consumption,
    working,
    delta,
    additive_utility_shock,
):
    return jnp.log(consumption) + additive_utility_shock - delta * working


def phelps_deaton_utility(consumption, working, delta):
    return jnp.log(consumption) - delta * working


def phelps_deaton_utility_with_filter(
    consumption,
    working,
    delta,
    lagged_retirement,  # noqa: ARG001
):
    return jnp.log(consumption) - delta * working


def working(retirement):
    return 1 - retirement


@lcm.mark.stochastic
def next_wage_category(wage_category, wage_category_transition):
    pass


@lcm.mark.stochastic
def wage_category(wage_category, wage_category_transition):
    pass


def wage(wage_category, wage_low, wage_high):
    if wage_category == "low":
        out = wage_low
    elif wage_category == "high":
        out = wage_high
    return out


# Question: dict as parameter?
def wage_with_dict(wage_category, wage_by_category):
    return wage_by_category[wage_category]


def next_wealth(wealth, consumption, working, wage, interest_rate):
    return (1 + interest_rate) * (wealth - consumption) + wage * working


def consumption_constraint(consumption, wealth):
    return consumption <= wealth


def age(period):
    return period + 18


def mandatory_retirement_filter(retirement, age):
    return jnp.logical_or(retirement == 1, age < RETIREMENT_AGE)


def absorbing_retirement_filter(retirement, lagged_retirement):
    return jnp.logical_or(retirement == 1, lagged_retirement == 0)


PHELPS_DEATON = {
    "functions": {
        "utility": phelps_deaton_utility,
        "next_wealth": next_wealth,
        "consumption_constraint": consumption_constraint,
        "working": working,
    },
    "choices": {
        "retirement": {"options": [0, 1]},
        "consumption": {
            "grid_type": "linspace",
            "start": 0,
            "stop": 100,
            "n_points": N_CHOICE_GRID_POINTS,
        },
    },
    "states": {
        "wealth": {
            "grid_type": "linspace",
            "start": 0,
            "stop": 100,
            "n_points": N_STATE_GRID_POINTS,
        },
    },
    "n_periods": 20,
}

PHELPS_DEATON_WITH_PERSISTENT_DISCRETE_TRANSITION_SHOCKS = {
    **PHELPS_DEATON,
    "functions": {
        "utility": phelps_deaton_utility,
        "next_wealth": next_wealth,
        "wage": wage,
        "next_wage_category": next_wage_category,
        "consumption_constraint": consumption_constraint,
        "working": working,
    },
    "states": {
        "wealth": {
            "grid_type": "linspace",
            "start": 0,
            "stop": 100,
            "n_points": N_STATE_GRID_POINTS,
        },
        "wage_category": {"options": ["low", "high"]},
        # Question: Possibility to specify number of values in params file?
    },
}

PHELPS_DEATON_WITH_TRANSITORY_DISCRETE_TRANSITION_SHOCKS = {
    **PHELPS_DEATON,
    "functions": {
        "utility": phelps_deaton_utility,
        "next_wealth": next_wealth,
        "wage": wage,
        "wage_category": wage_category,
        "consumption_constraint": consumption_constraint,
        "working": working,
    },
    "states": {
        "wealth": {
            "grid_type": "linspace",
            "start": 0,
            "stop": 100,
            "n_points": N_STATE_GRID_POINTS,
        },
        # Question: Where are options of wage_category specified?
    },
}

PHELPS_DEATON_WITH_FILTERS = {
    "functions": {
        "utility": phelps_deaton_utility_with_filter,
        "next_wealth": next_wealth,
        "consumption_constraint": consumption_constraint,
        "working": working,
        "absorbing_retirement_filter": absorbing_retirement_filter,
        "next_lagged_retirement": lambda retirement: retirement,
    },
    "choices": {
        "retirement": {"options": [0, 1]},
        "consumption": {
            "grid_type": "linspace",
            "start": 1,
            "stop": 100,
            "n_points": N_CHOICE_GRID_POINTS,
        },
    },
    "states": {
        "wealth": {
            "grid_type": "linspace",
            "start": 0,
            "stop": 100,
            "n_points": N_STATE_GRID_POINTS,
        },
        "lagged_retirement": {"options": [0, 1]},
    },
    "n_periods": 20,
}
