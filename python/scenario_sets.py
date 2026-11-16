"""Frozen source collections with physical signatures and reproducible sampling."""
import hashlib
import json
from pathlib import Path

from scenario import read_scenario, snapshot_scenario


def describe_case(simfile):
    info = read_scenario(simfile)
    lines = Path(info['parsfile']).read_text(encoding='utf-8-sig').splitlines()
    physical, reference, grip = [], [], []
    inside = False
    for line in lines:
        text = line.strip()
        if not text or text.startswith(('#', '!', 'LOG_ENTRY', 'ENTER_PARSFILE', 'EXIT_PARSFILE')):
            continue
        physical.append(' '.join(text.split()))
        if text.startswith('LTARG_TABLE'):
            inside = True
        if inside:
            reference.append(text)
        if text.startswith('ENDTABLE'):
            inside = False
        if text.startswith('MU_ROAD'):
            grip.append(text)
    info.update(physical_sha256=hashlib.sha256('\n'.join(physical).encode()).hexdigest(),
                reference_sha256=hashlib.sha256('\n'.join(reference).encode()).hexdigest(),
                reference_table=reference, grip_description=grip)
    return info


def freeze_sets(manifest, output):
    """Input JSON has train/validation/test lists of existing Simfile paths."""
    source = Path(manifest).resolve()
    sets = json.loads(source.read_text(encoding='utf-8-sig'))
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    result, seen = {}, {}
    for split in ('train', 'validation', 'test'):
        result[split] = []
        for index, value in enumerate(sets.get(split, [])):
            path = Path(value)
            info = describe_case(path if path.is_absolute() else source.parent / path)
            signature = info['physical_sha256']
            if signature in seen and seen[signature] != split:
                raise ValueError(f'Physically identical case appears in {seen[signature]} and {split}')
            seen[signature] = split
            frozen = snapshot_scenario(info, output / f'{split}_{index:03d}')
            result[split].append(dict(info, split=split, frozen_simfile=str(frozen)))
    if not result['train']:
        raise ValueError('At least one real training case is required')
    (output / 'scenario_sets.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    return result
