from evaluate import dominates, pareto


def test_pareto_excludes_ineligible_and_dominated_points():
    rows = [
        {"rmse_y_m": .08, "rmse_heading_deg": .9, "eligible": True},
        {"rmse_y_m": .09, "rmse_heading_deg": 1.0, "eligible": True},
        {"rmse_y_m": .07, "rmse_heading_deg": 1.1, "eligible": True},
        {"rmse_y_m": .06, "rmse_heading_deg": .7, "eligible": False},
    ]
    assert dominates(rows[0], rows[1])
    assert pareto(rows) == [rows[0], rows[2]]
