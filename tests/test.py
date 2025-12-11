from tests.test_models.deterministic_regimes import model

working_params = {
    "beta": 0.95,
    "utility": {"disutility_of_work": 0.5},
    "next_wealth": {"interest_rate": 0.05},
    "borrowing_constraint": {},
    "labor_income": {"wage": 10.0},
    "working": {},
}


retired_params = {
    "beta": 0.95,
    "utility": {"disutility_of_work": 0.5},
    "next_wealth": {"interest_rate": 0.05, "labor_income": 0.0},
    "borrowing_constraint": {},
}

params = {
    "working": working_params,
    "retired": retired_params,
    "dead": {"beta": 0.0},
}


def test_solve_deterministic_regimes():
    model.solve(params=params)
