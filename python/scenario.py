"""Read the target speed and source identity from a CarSim Simfile."""
import hashlib
import configparser
import math
import shutil
import json
import subprocess
from pathlib import Path

ACTION_PROFILES = {'qy1': ('q_y',), 'weights2': ('q_y','q_psi'),
                   'weights3': ('q_y','q_psi','r_delta'), 'full4': ('np','q_y','q_psi','r_delta')}


def restore_model_config(model, output):
    """Prefer a model's own version, even if its filename occurs in later studies."""
    model=Path(model).resolve(strict=True)
    manifest=next((p/'manifest.json' for p in model.parents if (p/'manifest.json').exists()),None)
    if manifest is None:raise ValueError('Model has no source manifest')
    info=json.loads(manifest.read_text(encoding='utf-8-sig'))
    source=Path(info['frozen_config']);hashes=info['controller_config_sha256']
    for name,digest in hashes.items():
        if Path(name).name!=name:raise ValueError('Invalid source hash filename')
        if hashlib.sha256((source.parent/name).read_bytes()).hexdigest()!=digest:
            raise ValueError('Frozen model config SHA256 mismatch: '+name)
    restored=snapshot_config(source,output)
    for file in restored.parent.iterdir():
        if hashlib.sha256(file.read_bytes()).hexdigest()!=hashes.get(file.name):
            raise ValueError('Restored model configuration differs: '+file.name)
    return restored


def restore_config(name, output, *, snapshot=None):
    """Restore exact historical bytes and the entire local inheritance chain; never overwrite."""
    root=Path(__file__).resolve().parents[1]
    catalog=json.loads((root/'configs/legacy.json').read_text(encoding='utf-8'))
    if name not in catalog['files']:raise ValueError('Unknown historical config')
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    installed=[]
    def restore(filename):
        expected=catalog['files'][filename]['sha256']
        payload=(Path(snapshot)/filename).read_bytes() if snapshot else subprocess.check_output(
            ['git','show',catalog['commit']+':configs/'+filename],cwd=root)
        candidates=[payload,payload.replace(b'\r\n',b'\n').replace(b'\n',b'\r\n')]
        if 'lf_lines' in catalog['files'][filename]:
            lf=set(catalog['files'][filename]['lf_lines'])
            lines=payload.replace(b'\r\n',b'\n').splitlines(keepends=True)
            candidates.append(b''.join(line[:-1]+b'\r\n' if i not in lf and line.endswith(b'\n') else line for i,line in enumerate(lines)))
        payload=next((p for p in candidates if hashlib.sha256(p).hexdigest()==expected),None)
        if payload is None:raise ValueError('Historical SHA256 mismatch: '+filename)
        target=output/filename;target.write_bytes(payload);installed.append(target)
        if filename.endswith('.ini'):
            parser=configparser.ConfigParser(interpolation=None);parser.read(target,encoding='utf-8-sig')
            base=parser.get('inherit','base',fallback=None)
            if base and not (output/base).exists():restore(base)
    try:
        restore(name)
        if not (output/'magic_tire.csv').exists():restore('magic_tire.csv')
    except Exception:
        for path in installed:path.unlink()
        output.rmdir()
        raise
    return output/name


def write_config_overrides(base, target, *, profile=None, overrides=()):
    """Run-private INI delta; keep the selected source chain byte-identical."""
    base=Path(base).resolve();target=Path(target).resolve()
    if base.parent!=target.parent or base==target:raise ValueError('Delta must share its frozen base directory')
    values={}
    if profile:
        names=ACTION_PROFILES[profile]
        values.update({'rl.policy_action_names':', '.join(names),'rl.policy_action_dims':str(len(names)),
                       'rl.max_np_change_per_decision':'2' if profile=='full4' else '0'})
    for item in overrides:
        key,value=item.split('=',1)
        section,field=key.split('.',1)
        if section not in ('rl','rl_reward') or resolve_config_value(base,section,field) is None:
            raise ValueError('Only existing rl/rl_reward fields may be overridden: '+key)
        values[key]=value
    parser=configparser.ConfigParser(interpolation=None);parser['inherit']={'base':base.name}
    for key,value in values.items():
        section,field=key.split('.',1)
        if not parser.has_section(section):parser.add_section(section)
        parser.set(section,field,value)
    with target.open('x',encoding='utf-8') as file:parser.write(file)
    return target


def reference_bounds(parsfile):
    """Read the station limits of the staged CarSim LTARG_TABLE."""
    entries = []
    inside = False
    found = False
    for line in Path(parsfile).read_text(encoding='utf-8-sig', errors='replace').splitlines():
        parts = line.split()
        if not parts:
            continue
        if parts[0] == 'LTARG_TABLE':
            if found or len(parts) < 2 or parts[1] != 'SPLINE_FLAT':
                raise ValueError('Expected one SPLINE_FLAT LTARG_TABLE')
            inside = found = True
        elif inside and parts[0] == 'ENDTABLE':
            inside = False
        elif inside and not parts[0].startswith(('#', '!')):
            x = float(line.split(',', 1)[0])
            if not math.isfinite(x) or (entries and x <= entries[-1]):
                raise ValueError('Invalid CarSim LTARG_TABLE station')
            entries.append(x)
    if inside or len(entries) < 3:
        raise ValueError('Missing or incomplete CarSim LTARG_TABLE')
    return entries[0], entries[-1]


def read_scenario(simfile, *, parsfile_override=None):
    simfile = Path(simfile).resolve()
    macros = {}
    input_value = None
    data_dir = None
    simfile_bytes = simfile.read_bytes()
    for line in simfile_bytes.decode("utf-8-sig", errors="replace").splitlines():
        parts = line.split(maxsplit=2)
        if not parts:
            continue
        if parts[0] == "SET_MACRO" and len(parts) == 3:
            macros[parts[1]] = parts[2].strip()
        elif parts[0] == "DATADIR":
            data_dir = line[len(parts[0]):].strip()
        elif parts[0] == "INPUT":
            if input_value is not None:
                raise ValueError("CarSim Simfile has multiple INPUT entries")
            input_value = line[len(parts[0]):].strip()
    if not input_value:
        raise ValueError("CarSim Simfile has no INPUT Parsfile")
    for _ in range(8):
        expanded = input_value
        for key, value in sorted(macros.items()):
            expanded = expanded.replace(key, value)
        if expanded == input_value:
            break
        input_value = expanded
    if "$(" in input_value:
        raise ValueError("Unresolved macro in CarSim Simfile INPUT")
    parsfile = Path(input_value)
    if not parsfile.is_absolute():
        parsfile = Path(data_dir) / parsfile if data_dir else simfile.parent / parsfile
    # Dataset records carry an explicitly hashed, byte-identical local Parsfile.
    # The archived native Simfile is left untouched; runtime snapshots redirect INPUT.
    parsfile = Path(parsfile_override).resolve(strict=True) if parsfile_override else parsfile.resolve()
    payload = parsfile.read_bytes()
    speeds = []
    for line in payload.decode("utf-8-sig", errors="replace").splitlines():
        parts = line.split()
        if parts and parts[0] == "SPEED_TARGET_CONSTANT":
            if len(parts) < 2:
                raise ValueError("SPEED_TARGET_CONSTANT has no value")
            speeds.append(float(parts[1]))
    speed = speeds[0] if len(speeds) == 1 and math.isfinite(speeds[0]) and speeds[0] > 0 else None
    speed_token = "vVariable" if speed is None else f"v{format(speed, '.12g').replace('.', 'p')}kmh"
    return {"target_speed_kmh": speed, "speed_token": speed_token,
            "simfile": str(simfile), "simfile_sha256": hashlib.sha256(simfile_bytes).hexdigest(), "parsfile": str(parsfile),
            "parsfile_sha256": hashlib.sha256(payload).hexdigest()}


def same_scenario_content(left, right):
    """Conservatively reject splits sharing either full source file content."""
    return (left["simfile_sha256"] == right["simfile_sha256"] or
            left["parsfile_sha256"] == right["parsfile_sha256"])


def resolve_config_value(config, section, key, default=None):
    config = Path(config).resolve()
    active = set()
    def read_file(path):
        path = path.resolve()
        if path in active:
            raise ValueError("Cyclic config inheritance")
        active.add(path)
        parser = configparser.ConfigParser(interpolation=None)
        with path.open(encoding="utf-8-sig") as file:
            parser.read_file(file)
        inherited = parser.get("inherit", "base", fallback=None)
        value = read_file(path.parent / inherited) if inherited else default
        value = parser.get(section, key, fallback=value)
        active.remove(path)
        return value
    return read_file(config)


def controller_policy_signature(config):
    """Compare the control/observation/action contract, independently of paths and comments."""
    active = set()
    def read(path):
        path = Path(path).resolve(strict=True)
        if path in active:
            raise ValueError("Cyclic config inheritance")
        active.add(path)
        parser = configparser.ConfigParser(interpolation=None)
        parser.read(path, encoding="utf-8-sig")
        base = parser.get("inherit", "base", fallback=None)
        result = read(path.parent / base) if base else {}
        for section in ("mpc", "weights", "constraints", "solver", "rl", "rl_reward"):
            if parser.has_section(section):
                result.update({f"{section}.{key}": value for key, value in parser[section].items()})
        active.remove(path)
        return result
    def canonical(value):
        try:
            return tuple(float(part.strip()) for part in value.split(","))
        except ValueError:
            return value.strip().lower()
    return {key: canonical(value) for key, value in read(config).items()}


def resolve_simfile(config, simfile=None):
    if simfile:
        return Path(simfile).resolve()
    config = Path(config).resolve()
    value = resolve_config_value(config, "carsim", "simfile")
    if not value:
        raise ValueError("No carsim.simfile in config; pass --simfile")
    path = Path(value)
    return (path if path.is_absolute() else config.parent / path).resolve()


def snapshot_config(config, output):
    """Copy a controller INI, its local inheritance chain and data tables."""
    source = Path(config).resolve(strict=True)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    path = source
    seen = set()
    while path not in seen:
        seen.add(path)
        shutil.copy2(path, output / path.name)
        parser = configparser.ConfigParser(interpolation=None)
        with path.open(encoding="utf-8-sig") as file:
            parser.read_file(file)
        base = parser.get("inherit", "base", fallback=None)
        if not base:
            break
        path = (path.parent / base).resolve(strict=True)
        if path.parent != source.parent:
            raise ValueError("Inherited config must be in the same directory")
    else:
        raise ValueError("Cyclic config inheritance")
    for table in source.parent.glob("*.csv"):
        shutil.copy2(table, output / table.name)
    return output / source.name


def snapshot_scenario(scenario, output):
    """Freeze the selected Parsfile and redirect a private Simfile to it."""
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    source = Path(scenario["parsfile"])
    payload = source.read_bytes()
    if hashlib.sha256(payload).hexdigest() != scenario["parsfile_sha256"]:
        raise RuntimeError("CarSim Parsfile changed while preparing the run; retry")
    pars = output / "Run_all.par"
    pars.write_bytes(payload)
    simfile = Path(scenario["simfile"])
    simfile_bytes = simfile.read_bytes()
    if hashlib.sha256(simfile_bytes).hexdigest() != scenario["simfile_sha256"]:
        raise RuntimeError("CarSim Simfile changed while preparing the run; retry")
    lines = simfile_bytes.decode("utf-8-sig", errors="strict").splitlines()
    replaced = 0
    for index, line in enumerate(lines):
        if line.split(maxsplit=1)[:1] == ["INPUT"]:
            lines[index] = "INPUT " + str(pars)
            replaced += 1
    if replaced != 1:
        raise ValueError("Expected exactly one INPUT in CarSim Simfile")
    target = output / "simfile.sim"
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return target
