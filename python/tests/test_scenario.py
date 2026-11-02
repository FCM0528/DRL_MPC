import pytest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scenario import read_scenario

def test_relative_macro_and_fractional_speed(tmp_path):
    pars = tmp_path / "case.par"
    pars.write_text("SPEED_TARGET_CONSTANT 73.5\n", encoding="utf-8")
    sim = tmp_path / "case.sim"
    sim.write_text("SET_MACRO $(CASE)$ case.par\nINPUT $(CASE)$\n", encoding="utf-8")
    info = read_scenario(sim)
    assert info["speed_token"] == "v73p5kmh"
    assert info["parsfile"] == str(pars)
    assert len(info["parsfile_sha256"]) == 64

def test_variable_speed_label(tmp_path):
    (tmp_path / "case.par").write_text("SPEED_TARGET_CONSTANT 60\nSPEED_TARGET_CONSTANT 80\n")
    sim = tmp_path / "case.sim"
    sim.write_text("INPUT case.par\n")
    assert read_scenario(sim)["speed_token"] == "vVariable"

def test_resolve_inherited_config_and_freeze(tmp_path):
    from scenario import resolve_simfile, snapshot_scenario
    (tmp_path / "case.par").write_text("SPEED_TARGET_CONSTANT 80\n", encoding="utf-8")
    sim = tmp_path / "case.sim"
    sim.write_text("INPUT case.par\n", encoding="utf-8")
    (tmp_path / "base.ini").write_text("[carsim]\nsimfile = case.sim\n", encoding="utf-8")
    config = tmp_path / "child.ini"
    config.write_text("[inherit]\nbase = base.ini\n", encoding="utf-8")
    assert resolve_simfile(config) == sim
    info = read_scenario(sim)
    frozen = snapshot_scenario(info, tmp_path / "frozen")
    (tmp_path / "case.par").write_text("SPEED_TARGET_CONSTANT 90\n", encoding="utf-8")
    assert read_scenario(frozen)["target_speed_kmh"] == 80
