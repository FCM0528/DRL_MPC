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


def check_model_contract(model_path, config):
    names, version = action_contract(config)
    path=Path(model_path).resolve()
    source=next((p/'manifest.json' for p in path.parents if (p/'manifest.json').is_file()),None)
    if source is None:
        if version=='named_actions_v1': raise ValueError('Named action policy requires provenance manifest')
        return
    manifest=json.loads(source.read_text(encoding='utf-8'))
    prior=tuple(manifest.get('policy_action_names',BRIDGE_NAMES[:int(manifest.get('policy_action_dims',4))]))
    prior_version=manifest.get('action_mapping_version','legacy_prefix_v1')
    if prior!=names or prior_version!=version:
        raise ValueError('Model action names/mapping version differ from evaluation config')
    frozen=manifest.get('frozen_config')
    if frozen and Path(frozen).is_file():
        if controller_policy_signature(frozen)!=controller_policy_signature(config):
            raise ValueError('Model controller/reward/observation contract differs from evaluation config (including track_course)')
    elif version=='named_actions_v1':
        raise ValueError('Named policy requires its frozen controller configuration')
