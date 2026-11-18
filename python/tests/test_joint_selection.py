from evaluate import rank_candidate


def test_dual_improvement_requires_both_errors_and_completion():
    baseline = {"rmse_y_m": 0.2, "rmse_heading_deg": 2.0}
    row = {"terminated": True, "truncated": False, "failed": False,
           "qp_failures": 0, "rmse_y_m": 0.18, "rmse_heading_deg": 1.8}
    assert rank_candidate(row, baseline)[:2] == (True, True)
    assert rank_candidate({**row, "rmse_heading_deg": 2.1}, baseline)[1] is False
    assert rank_candidate({**row, "qp_failures": 1}, baseline)[:2] == (False, False)
    assert rank_candidate({**row, "truncated": True}, baseline)[:2] == (False, False)
