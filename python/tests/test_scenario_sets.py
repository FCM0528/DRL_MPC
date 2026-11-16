import json
import numpy as np
import pytest
from scenario_sets import freeze_sets
from envs.carsim_mpc_env import CarSimMPCEnv


def test_physical_duplicate_rejected_across_splits(tmp_path):
    for index in (1,2):
        (tmp_path/f'{index}.par').write_text(f'# different label {index}\nSPEED_TARGET_CONSTANT 70\nMU_ROAD_CONSTANT 0.8\n')
        (tmp_path/f'{index}.sim').write_text(f'INPUT {index}.par\n')
    manifest=tmp_path/'sets.json'
    manifest.write_text(json.dumps(dict(train=['1.sim'],validation=['2.sim'],test=[])))
    with pytest.raises(ValueError,match='Physically identical'):
        freeze_sets(manifest,tmp_path/'frozen')


def test_reset_seed_reproduces_case_sequence(tmp_path,monkeypatch):
    import envs.carsim_mpc_env as module
    class Bridge:
        def __init__(self,*args): pass
        def exchange(self,*args): return [0]*20
        def close(self): pass
    monkeypatch.setattr(module,'BridgeClient',Bridge)
    exe=tmp_path/'bridge';exe.write_text('test')
    config=tmp_path/'config.ini';config.write_text('[rl]\nobservation_profile=road_preview_joint\n')
    cases=[]
    for i in range(3):
        case=tmp_path/f'{i}.sim';case.write_text('test');cases.append(case)
    env=CarSimMPCEnv(exe,config,cases[0],tmp_path/'out');env.scenario_pool=cases
    def sequence():
        env.reset(seed=42)
        values=[env.simfile]
        for _ in range(9):
            env.reset();values.append(env.simfile)
        return values
    assert sequence()==sequence()
    env.close()
