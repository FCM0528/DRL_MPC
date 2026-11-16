import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import json
import numpy as np
import pytest
from action_mapping import action_contract, bridge_action, check_model_contract


@pytest.mark.parametrize('names',[('q_y',),('q_y','q_psi'),('np','q_y'),('np','q_y','q_psi'),('np','q_y','q_psi','r_delta')])
def test_named_mapping_preserves_neutral_coordinates(tmp_path,names):
    ini=tmp_path/'config.ini';ini.write_text(f'[rl]\npolicy_action_dims={len(names)}\npolicy_action_names={",".join(names)}\n')
    actual,version=action_contract(ini)
    assert actual==names and version=='named_actions_v1'
    values=np.linspace(.2,.8,len(names))
    result=bridge_action(values,names)
    for index,name in enumerate(('np','q_y','q_psi','r_delta')):
        assert result[index]==(values[names.index(name)] if name in names else 0)
    np.testing.assert_array_equal(bridge_action(np.zeros(len(names)),names),np.zeros(4))


def test_same_dimension_different_semantics_rejected(tmp_path):
    model=tmp_path/'model.zip';model.write_bytes(b'test')
    (tmp_path/'manifest.json').write_text(json.dumps(dict(policy_action_dims='2')))
    config=tmp_path/'config.ini';config.write_text('[rl]\npolicy_action_dims=2\npolicy_action_names=q_y,q_psi\n')
    with pytest.raises(ValueError,match='action names'):check_model_contract(model,config)
    config.write_text('[rl]\npolicy_action_dims=2\n')
    check_model_contract(model,config)
    assert action_contract(config)==(('np','q_y'),'legacy_prefix_v1')


def test_duplicate_or_wrong_dimension_rejected(tmp_path):
    config=tmp_path/'config.ini'
    for value in ('q_y,q_y','np,q_psi','q_y'):
        config.write_text(f'[rl]\npolicy_action_dims=2\npolicy_action_names={value}\n')
        with pytest.raises(ValueError):action_contract(config)


@pytest.mark.parametrize('names,expected',[('q_y',[0,.4,0,0]),('q_y,q_psi',[0,.4,-.3,0])])
def test_named_environment_sends_neutral_unlearned_actions(tmp_path,names,expected):
    from envs.carsim_mpc_env import CarSimMPCEnv
    dims=len(names.split(','))
    config=tmp_path/'config.ini';config.write_text(f'[rl]\npolicy_action_dims={dims}\npolicy_action_names={names}\n')
    executable=tmp_path/'bridge.exe';executable.write_bytes(b'fixture')
    simfile=tmp_path/'case.sim';simfile.write_text('fixture')
    env=CarSimMPCEnv(executable,config,simfile,tmp_path/'out')
    class Bridge:
        def exchange(self,command,tag):
            np.testing.assert_allclose([float(v) for v in command.split()[1:]],expected)
            return [0.]*14 + [0.,0,0,1.,0.,0.,0.,0,0,1,0,36]
        def close(self):pass
    env._bridge=Bridge()
    env.step(np.array([.4,-.3])[:dims]);env.close()
