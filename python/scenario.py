"""Read the target speed and source identity from a CarSim Simfile."""
import hashlib
import configparser
import math
from pathlib import Path


def read_scenario(simfile):
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
    parsfile = parsfile.resolve()
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


def resolve_simfile(config, simfile=None):
    if simfile:
        return Path(simfile).resolve()
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
        value = read_file(path.parent / inherited) if inherited else None
        value = parser.get("carsim", "simfile", fallback=value)
        active.remove(path)
        return value
    value = read_file(config)
    if not value:
        raise ValueError("No carsim.simfile in config; pass --simfile")
    path = Path(value)
    return (path if path.is_absolute() else config.parent / path).resolve()


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
