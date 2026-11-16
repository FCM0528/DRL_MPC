"""Frozen source collections with physical signatures and reproducible sampling."""
import hashlib
import json
import re
from pathlib import Path

from scenario import read_scenario, snapshot_scenario


def describe_case(simfile):
    info = read_scenario(simfile)
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
    signature=[str(info['target_speed_kmh']),str(info['vehicle_sha256']),info['io_sha256'],info['reference_numeric_sha256'],info['grip_sha256']]
    info['experiment_signature']=digest(signature) if vehicle and reference and grip else info['physical_sha256']
    info['scenario_signature_version']='semantic_numeric_v2'
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


def freeze_sets(manifest, output):
    """Input JSON has train/validation/test lists of existing Simfile paths."""
    source = Path(manifest).resolve()
    sets = json.loads(source.read_text(encoding='utf-8-sig'))
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    result, seen = {}, {}
    vehicle_contract=None
    for split in ('train', 'validation', 'test'):
        result[split] = []
        for index, value in enumerate(sets.get(split, [])):
            path = Path(value)
            info = describe_case(path if path.is_absolute() else source.parent / path)
            signature = info['experiment_signature']
            if info['vehicle_sha256']:
                contract=(info['vehicle_sha256'],info['io_sha256'])
                if vehicle_contract is not None and contract!=vehicle_contract:
                    raise ValueError('Vehicle or I/O content differs across scenario sets')
                vehicle_contract=contract
            if signature in seen and seen[signature] != split:
                raise ValueError(f'Physically identical case appears in {seen[signature]} and {split}')
            seen[signature] = split
            frozen = snapshot_scenario(info, output / f'{split}_{index:03d}')
            result[split].append(dict(info, split=split, frozen_simfile=str(frozen)))
    if not result['train']:
        raise ValueError('At least one real training case is required')
    (output / 'scenario_sets.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    return result
