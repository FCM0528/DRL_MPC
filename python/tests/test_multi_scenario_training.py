import json
import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scenario_sets import training_source, freeze_sets, SAMPLING_RULE
from envs.carsim_mpc_env import CarSimMPCEnv


def sources(tmp_path):
    for speed in (60,65,70):
        (tmp_path/f'{speed}.par').write_text(f'SPEED_TARGET_CONSTANT {speed}\nMU_ROAD_CONSTANT 0.8\n')
        (tmp_path/f'{speed}.sim').write_text(f'INPUT {speed}.par\n')
    sets=tmp_path/'sets.json';sets.write_text(json.dumps(dict(train=['60.sim','70.sim'],validation=['65.sim'],test=[])))
    config=tmp_path/'config.ini';config.write_text('[carsim]\nsimfile=65.sim\n[rl]\nobservation_profile=road_preview_joint\n')
    return config,sets


def test_train_source_ignores_mutable_validation_default(tmp_path):
    config,sets=sources(tmp_path)
    assert training_source(config,scenario_set=sets,joint_selection=True)==tmp_path/'60.sim'
    assert training_source(config,tmp_path/'70.sim',scenario_set=sets,joint_selection=True)==tmp_path/'70.sim'
    with pytest.raises(ValueError,match='outside the train'):
        training_source(config,tmp_path/'65.sim',scenario_set=sets,joint_selection=True)
    with pytest.raises(ValueError,match='joint-selection'):training_source(config,scenario_set=sets)
    with pytest.raises(ValueError,match='validation entries'):
        training_source(config,scenario_set=sets,joint_selection=True,validation_simfile=tmp_path/'65.sim')
    assert training_source(config)==tmp_path/'65.sim'  # Legacy no-set interface unchanged.


def test_duplicate_training_files_cannot_bias_sampling(tmp_path):
    _,sets=sources(tmp_path)
    data=json.loads(sets.read_text());data['train'].append('60.sim');sets.write_text(json.dumps(data))
    with pytest.raises(ValueError,match='bias sampling'):freeze_sets(sets,tmp_path/'freeze')


def test_episode_sampler_is_local_uniform_and_closes_old_solver(tmp_path,monkeypatch):
    import envs.carsim_mpc_env as module
    config,_=sources(tmp_path);exe=tmp_path/'bridge';exe.write_text('unit fixture')
    instances=[]
    class Bridge:
        def __init__(self,executable,config,simfile,*args):
            if instances:assert instances[-1].closed
            self.simfile=simfile;self.closed=False;instances.append(self)
        def exchange(self,command,*args):
            if command=='RESET':return [0.]*20
            return [0.]*20+[0.,1,0,120.,0,0,0,0,0,2,0,36]
        def close(self):self.closed=True
    monkeypatch.setattr(module,'BridgeClient',Bridge)
    env=CarSimMPCEnv(exe,config,tmp_path/'60.sim',tmp_path/'probe')
    env.scenario_pool=[tmp_path/'60.sim',tmp_path/'70.sim']
    def sequence():
        result=[]
        for i in range(12):
            _,info=env.reset(seed=42 if i==0 else None)
            result.append(info['case_index'])
            # Algorithm-specific randomness must not consume the environment sampler RNG.
            np.random.seed(i);np.random.normal(size=100+i)
            _,_,done,_,step=env.step(np.zeros(4))
            assert done and step['simfile']==str(instances[-1].simfile)
        return result
    try:
        actual=sequence();assert actual==sequence()
        assert actual==np.random.default_rng(42).integers(2,size=12).tolist()
        assert set(actual)=={0,1}
        assert sum(env.scenario_decision_counts.values())==24
        log=[json.loads(s) for s in (tmp_path/'probe/episode_scenarios.jsonl').read_text().splitlines()]
        assert all(r['sampling_rule']==SAMPLING_RULE and r['pool_size']==2 for r in log)
    finally:env.close()


def test_preflight_solver_failure_is_retained_without_zero_qp_claim(tmp_path,monkeypatch):
    import scenario_preflight
    def fail(*args,**kwargs):raise RuntimeError('native solver rejected case; inspect fixed_run.log')
    monkeypatch.setattr(scenario_preflight,'fixed_episode',fail)
    row=scenario_preflight.preflight_baseline('exe','case',tmp_path/'failed','config')
    assert row['failed'] and not row['stable'] and not row['terminated']
    assert row['qp_failures'] is None and 'native solver' in row['error']
    monkeypatch.setattr(scenario_preflight,'fixed_episode',lambda *a,**k:dict(terminated=False,truncated=True,failed=False,
        qp_failures=3,rmse_y_m=.5,peak_measured_ay_mps2=9.,peak_measured_beta_deg=6.))
    row=scenario_preflight.preflight_baseline('exe','case',tmp_path/'partial','config')
    assert row['rmse_y_m']==.5 and row['qp_failures']==3 and not row['stable']
