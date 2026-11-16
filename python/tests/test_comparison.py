import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from compare_four import comparison_limits
from report import control_trace
from scenario import controller_policy_signature


@pytest.mark.parametrize("end", ["nan", "inf", "0", "-1"])
def test_comparison_rejects_invalid_end_without_segment_cuts(tmp_path, end):
    pars = tmp_path / "case.par"
    pars.write_text("LTARG_TABLE SPLINE_FLAT\n0,0\n60,1\n120,0\nENDTABLE\n")
    config = tmp_path / "config.ini"
    config.write_text(f"[simulation]\ndistance_m={end}\n")
    with pytest.raises(ValueError):
        comparison_limits({"parsfile": str(pars)}, config, config, ())


def test_empty_failed_episode_has_actionable_error(tmp_path):
    path = tmp_path / "bridge_trace.csv"
    path.write_text("x_m,ey_m\n")
    with pytest.raises(ValueError, match="solver episode log"):
        control_trace(path)


def test_policy_contract_uses_inherited_values_and_detects_changed_action_mapping(tmp_path):
    (tmp_path / "base.ini").write_text("[mpc]\nnc=13\n[weights]\nq_y=11500\n")
    old = tmp_path / "old.ini"
    old.write_text("[inherit]\nbase=base.ini\n[rl]\nq_y_values=8050, 9775, 11500, 13225, 14950\n")
    current = tmp_path / "current.ini"
    current.write_text("[mpc]\nnc=13\n[weights]\nq_y=11500.0\n"
                       "[rl]\nq_y_values=8050.0,9775,11500,13225,14950\n")
    assert controller_policy_signature(old) == controller_policy_signature(current)
    current.write_text(current.read_text().replace("8050.0,9775,11500,13225,14950", "8050,11500,14950"))
    assert controller_policy_signature(old) != controller_policy_signature(current)
