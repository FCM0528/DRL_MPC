"""Frozen source collections with physical signatures and reproducible sampling."""
import hashlib
import json
import re
from pathlib import Path

from scenario import read_scenario, snapshot_scenario

SAMPLING_RULE = 'uniform_episode_v1'
BALANCED_SAMPLING_RULE = 'shuffled_complete_case_blocks_v1'


class EpisodeBlockSchedule:
    """A fixed sequence of case blocks, with the same case repeated after an incomplete run."""
    def __init__(self, cases, per_case, seed):
        import numpy as np

        if cases<1 or per_case<1:
            raise ValueError('Expected positive case count and completed-episode quota')
        rng=np.random.default_rng(seed)
        self.sequence=np.concatenate([rng.permutation(cases) for _ in range(per_case)]).tolist()
        self.completed=0

    @property
    def finished(self):
        return self.completed==len(self.sequence)

    def case(self):
        return self.sequence[self.completed]

    def advance(self, completed):
        self.completed+=int(completed)


def training_source(config, simfile=None, *, scenario_set=None, validation_simfile=None, joint_selection=False,
                    include_test=True):
    """Anchor metadata/parity to the real train collection, never the mutable Browser default."""
    from scenario import resolve_simfile
    if not scenario_set:
        return resolve_simfile(config,simfile)
    if validation_simfile:
        raise ValueError('Use scenario-set validation entries instead of --validation-simfile')
    if not joint_selection:
        raise ValueError('Scenario-set training requires --joint-selection for all validation cases')
    sets=load_case_sets(scenario_set,include_test=include_test)
    if not sets.get('train') or not sets.get('validation'):
        raise ValueError('Scenario-set training requires nonempty real train and validation collections')
    paths=[Path(c['simfile']) for c in sets['train']]
    if simfile is not None:
        exact=next((c for c in sets['train'] if Path(c['simfile']).resolve()==Path(simfile).resolve()),None)
        signature=(exact or describe_case(simfile))['experiment_signature']
        for case in sets['train']:
            if case['experiment_signature']==signature:return Path(case['simfile'])
        raise ValueError('Explicit --simfile is outside the train physical collection')
    return paths[0]


def describe_case(simfile, *, parsfile_override=None):
    info = read_scenario(simfile, parsfile_override=parsfile_override)
    lines = Path(info['parsfile']).read_text(encoding='utf-8-sig').splitlines()
    physical, reference, grip, vehicle, io = [], [], [], [], []
    inside = False
    grip_inside=False
    vehicle_root=None
    for line in lines:
        text = line.split('!',1)[0].strip()
        if text.startswith('ENTER_PARSFILE Vehicles\\Assembly\\'):
            vehicle_root=text.split(maxsplit=1)[1]
        if vehicle_root and text.startswith('EXIT_PARSFILE '+vehicle_root):
            vehicle_root=None
        if not text or text.startswith(('#', '!', 'LOG_ENTRY', 'ENTER_PARSFILE', 'EXIT_PARSFILE')):
            continue
        physical.append(' '.join(text.split()))
        if vehicle_root:vehicle.append(' '.join(text.split()))
        if text.split()[0] in ('IMPORT','EXPORT','IMP_STEER_L1','IMP_STEER_R1'):
            io.append(' '.join(text.split()))
        if text.startswith('LTARG_TABLE'):
            inside = True
        if inside:
            reference.append(text)
        if text.startswith('ENDTABLE'):
            inside = False
        if text.startswith('MU_ROAD'):
            grip.append(text)
        if text.startswith('MU_ROAD_TABLE'):grip_inside=True
        elif grip_inside:
            grip.append(text)
            if text.startswith('ENDTABLE'):grip_inside=False
    info.update(physical_sha256=hashlib.sha256('\n'.join(physical).encode()).hexdigest(),
                reference_sha256=hashlib.sha256('\n'.join(reference).encode()).hexdigest(),
                reference_table=reference, grip_description=grip)
    def digest(items):
        def token(value):
            try:return format(float(value),'.16g')
            except ValueError:return value
        canonical=[' '.join(token(t) for t in re.split(r'[\s,]+',s)) for s in items]
        return hashlib.sha256('\n'.join(canonical).encode()).hexdigest()
    info['vehicle_sha256']=digest(vehicle) if vehicle else None
    info['io_sha256']=digest(io)
    info['reference_numeric_sha256']=digest(reference)
    info['grip_sha256']=digest(grip)
    # Include the actual longitudinal command law, not only its constant offset.
    # Preserve v2 signatures for the existing constant-speed sources.
    speed_law=[]; speed_table=False
    for line in physical:
        key=line.split()[0]
        if key.startswith('SPEED_TARGET'):
            speed_law.append(line)
            speed_table=key.endswith('TABLE')
        elif speed_table:
            speed_law.append(line)
            if key=='ENDTABLE':speed_table=False
    variable_speed=any('TABLE' in s.split()[0] for s in speed_law)
    for s in speed_law:
        p=s.split()
        if p[0].startswith('SPEED_TARGET') and p[0] not in ('SPEED_TARGET_CONSTANT','SPEED_TARGET_S_CONSTANT','SPEED_TARGET_COMBINE'):
            variable_speed=True
        if p[0]=='SPEED_TARGET_COMBINE' and p[1:]!=['ADD']:variable_speed=True
        if p[0]=='SPEED_TARGET_S_CONSTANT' and len(p)>1:
            try:variable_speed |= float(p[1])!=0
            except ValueError:variable_speed=True
    signature=[str(info['target_speed_kmh']),str(info['vehicle_sha256']),info['io_sha256'],info['reference_numeric_sha256'],info['grip_sha256']]
    if variable_speed:signature.append(digest(speed_law))
    info['experiment_signature']=digest(signature) if vehicle and reference and grip else info['physical_sha256']
    info['scenario_signature_version']='semantic_numeric_v3_speed_law' if variable_speed else 'semantic_numeric_v2'
    points=[]
    for s in reference[1:-1]:
        try:points.append([float(v) for v in re.split(r'[\s,]+',s)][:2])
        except ValueError:pass
    if len(points)>2 and all(len(p)==2 for p in points):
        import numpy as np
        xy=np.asarray(points);slope=np.gradient(xy[:,1],xy[:,0]);curvature=np.gradient(slope,xy[:,0])/(1+slope*slope)**1.5
        info['reference_station_bounds_m']=[float(xy[0,0]),float(xy[-1,0])]
        info['reference_max_abs_curvature_inv_m']=float(np.max(np.abs(curvature)))
    return info


def load_case_sets(manifest, *, include_test=True):
    """Read legacy path collections or a verified immutable dataset manifest."""
    source=Path(manifest).resolve(strict=True)
    value=json.loads(source.read_text(encoding='utf-8-sig'))
    if 'dataset_version' in value:
        from dataset_manager import load_dataset
        sets = load_dataset(source,verify_test_payload=include_test)[1]
        if not include_test:
            sets['test'] = []
        return sets
    return {s:[describe_case(Path(p) if Path(p).is_absolute() else source.parent/p)
               for p in value.get(s,[]) if s != 'test' or include_test] for s in ('train','validation','test')}


def freeze_sets(manifest, output, *, include_test=True):
    """Input JSON has train/validation/test lists of existing Simfile paths."""
    sets = load_case_sets(manifest,include_test=include_test)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    result, seen = {}, {}
    vehicle_contract=None
    for split in ('train', 'validation', 'test'):
        result[split] = []
        for index, info in enumerate(sets.get(split, [])):
            signature = info['experiment_signature']
            if info['vehicle_sha256']:
                contract=(info['vehicle_sha256'],info['io_sha256'])
                if vehicle_contract is not None and contract!=vehicle_contract:
                    raise ValueError('Vehicle or I/O content differs across scenario sets')
                vehicle_contract=contract
            if signature in seen:
                raise ValueError(f'Physically identical case appears more than once in {seen[signature]} and {split}; duplicate entries bias sampling')
            seen[signature] = split
            frozen = snapshot_scenario(info, output / f'{split}_{index:03d}')
            result[split].append(dict(info, split=split, frozen_simfile=str(frozen)))
    if not result['train']:
        raise ValueError('At least one real training case is required')
    (output / 'scenario_sets.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    return result
