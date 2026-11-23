"""Explicit policy coordinates; the C++ wire protocol always has four coordinates."""
from pathlib import Path
import json
import numpy as np
from scenario import resolve_config_value, controller_policy_signature

BRIDGE_NAMES = ('np', 'q_y', 'q_psi', 'r_delta')
SUPPORTED = {('q_y',), ('q_y','q_psi'), ('q_y','q_psi','r_delta'), BRIDGE_NAMES[:2], BRIDGE_NAMES[:3], BRIDGE_NAMES}


def action_contract(config):
    raw = resolve_config_value(config, 'rl', 'policy_action_names', '')
    dims = int(resolve_config_value(config, 'rl', 'policy_action_dims', '4'))
    if raw.strip():
        names = tuple(v.strip().strip('\"\'').lower() for v in raw.strip().strip('[]').split(','))
        if names not in SUPPORTED or len(names) != dims:
            raise ValueError('Unsupported policy_action_names or dimension mismatch')
        return names, 'named_actions_v1'
    if dims not in (2,3,4):
        raise ValueError('Legacy policy_action_dims must be 2, 3 or 4')
    return BRIDGE_NAMES[:dims], 'legacy_prefix_v1'


def bridge_action(action, names):
    values = np.asarray(action, dtype=np.float64)
    if values.shape != (len(names),) or not np.all(np.isfinite(values)) or np.any(np.abs(values)>1):
        raise ValueError('Policy action has invalid shape, finite values or bounds')
    result = np.zeros(4, dtype=np.float64)
    for i, name in enumerate(names): result[BRIDGE_NAMES.index(name)] = values[i]
    return result


def check_model_contract(model_path, config, *, simfile=None):
    names, version = action_contract(config)
    path=Path(model_path).resolve()
    source=next((p/'manifest.json' for p in path.parents if (p/'manifest.json').is_file()),None)
    if source is None:
        if version=='named_actions_v1': raise ValueError('Named action policy requires provenance manifest')
        return
    manifest=json.loads(source.read_text(encoding='utf-8'))
    prior=tuple(manifest.get('policy_action_names',BRIDGE_NAMES[:int(manifest.get('policy_action_dims',4))]))
    prior_version=manifest.get('action_mapping_version','legacy_prefix_v1')
    if prior_version=='case_conditioned_named_actions_v1':
        from scenario_sets import describe_case
        from train import controller_config_hashes

        if simfile is None or prior!=names or version!='named_actions_v1':
            raise ValueError('Conditioned policy requires an explicit frozen physical case and four named actions')
        signature=describe_case(simfile)['experiment_signature']
        case=manifest['case_contracts'].get(signature)
        if case is None or controller_config_hashes(Path(config))!=case['config_chain_sha256']:
            raise ValueError('Conditioned policy config is not the frozen contract for this physical case')
        return
    if prior!=names or prior_version!=version:
        raise ValueError('Model action names/mapping version differ from evaluation config')
    frozen=manifest.get('frozen_config')
    if frozen and Path(frozen).is_file():
        if controller_policy_signature(frozen)!=controller_policy_signature(config):
            raise ValueError('Model controller/reward/observation contract differs from evaluation config (including track_course)')
    elif version=='named_actions_v1':
        raise ValueError('Named policy requires its frozen controller configuration')


def inverse_qy(weight,center=18400.,low=11500.,high=23920.):
    if not low-1e-8<=weight<=high+1e-8:raise ValueError('Requested Qy outside unchanged action range')
    weight=float(np.clip(weight,low,high))
    if weight==low:return -1.
    if weight==high:return 1.
    return np.log(weight/center)/(np.log(high/center) if weight>=center else np.log(center/low))
