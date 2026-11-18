"""Exact compatibility recovery, and equivalent current profile/override materialization."""
import hashlib
import json
from pathlib import Path
import pytest
from scenario import restore_config, snapshot_config, write_config_overrides, controller_policy_signature
from train import load_training_settings

ROOT=Path(__file__).resolve().parents[2]
CAT=json.loads((ROOT/'configs/legacy.json').read_text(encoding='utf-8'))


@pytest.mark.parametrize('name',[k for k,v in CAT['files'].items() if v['retired']])
def test_every_retired_ini_recovers_exact_bytes_and_complete_chain(tmp_path,name):
    path=restore_config(name,tmp_path/'recovered')
    assert hashlib.sha256(path.read_bytes()).hexdigest()==CAT['files'][name]['sha256']
    for file in path.parent.iterdir():
        assert hashlib.sha256(file.read_bytes()).hexdigest()==CAT['files'][file.name]['sha256']
    # Parse inherited effective values, not just the leaf's text.
    if name.startswith('td3_qy_'):assert load_training_settings(path)
    else:assert controller_policy_signature(path)


@pytest.mark.parametrize('profile,old',[('qy1','strong_qy_only.ini'),('weights2','strong_qy_qpsi.ini'),
                                     ('weights3','strong_weights3.ini'),('full4','strong_full4.ini')])
def test_current_generated_profile_equals_historical_control_contract(tmp_path,profile,old):
    historical=restore_config(old,tmp_path/'historical')
    base=snapshot_config(ROOT/'configs/strong_weights3.ini',tmp_path/'current')
    generated=write_config_overrides(base,base.parent/'profile.ini',profile=profile)
    assert controller_policy_signature(historical)==controller_policy_signature(generated)


def test_recovery_refuses_overwrite_and_failed_snapshot_leaves_no_partial_chain(tmp_path):
    output=tmp_path/'existing';output.mkdir();(output/'keep').write_text('important')
    with pytest.raises(FileExistsError):restore_config('strong_qy_only.ini',output)
    assert (output/'keep').read_text()=='important'
    bad=tmp_path/'bad';bad.mkdir();(bad/'strong_qy_only.ini').write_text('invalid')
    with pytest.raises(ValueError,match='SHA256'):restore_config('strong_qy_only.ini',tmp_path/'failed',snapshot=bad)
    assert not (tmp_path/'failed').exists()


def test_overrides_reject_unknown_fields_and_use_existing_validation():
    with pytest.raises(ValueError):load_training_settings(ROOT/'configs/rl_training_pilot.ini',['td3.unknown=1'])
    with pytest.raises(ValueError):load_training_settings(ROOT/'configs/rl_training_pilot.ini',['td3.actor_learning_rate=-1'])


def test_model_manifest_restores_its_own_older_version_and_detects_tampering(tmp_path):
    from scenario import restore_model_config
    run=tmp_path/'old_run';run.mkdir()
    config=run/'model.ini';config.write_text('[rl]\npolicy_action_dims=1\npolicy_action_names=q_y\n')
    model=run/'best_legal_model.zip';model.write_bytes(b'unit fixture, not a trained model')
    (run/'manifest.json').write_text(json.dumps(dict(frozen_config=str(config),controller_config_sha256={config.name:hashlib.sha256(config.read_bytes()).hexdigest()})))
    result=restore_model_config(model,tmp_path/'new')
    assert result.read_bytes()==config.read_bytes()
    config.write_text('tampered')
    with pytest.raises(ValueError,match='SHA256'):restore_model_config(model,tmp_path/'rejected')
