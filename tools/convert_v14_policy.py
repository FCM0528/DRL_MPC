"""Convert a V14 multiplier actor to an equivalent V15 absolute-action actor.

The source training effective_config.ini supplies the six weights that V14
multiplied. Conversion is exact only for reciprocal multiplier bounds and a
V15 configuration whose absolute bounds equal the resulting value ranges.
"""

from __future__ import annotations

import argparse
import configparser
import hashlib
from math import isclose, log
from pathlib import Path
import shutil


WEIGHTS = ("q_y", "q_psi", "q_vy", "q_r", "r_steer", "r_delta")


def load_ini(path: Path) -> configparser.ConfigParser:
    ini = configparser.ConfigParser(interpolation=None)
    if not ini.read(path, encoding="utf-8-sig"):
        raise ValueError(f"Missing configuration: {path}")
    return ini


def near(actual: float, expected: float, label: str) -> None:
    if not isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-14):
        raise ValueError(f"{label}: source {actual:.17g}, target {expected:.17g}")


def convert(policy: Path, target_config: Path, output: Path) -> None:
    source_config = policy.parent / "effective_config.ini"
    source = load_ini(source_config)
    target = load_ini(target_config)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite policy: {output}")
    with policy.open("r", encoding="utf-8") as stream:
        header = stream.readline().split()
        actor_weights = stream.read()
    if len(header) != 10 or header[0] != "DDPG_MPC_V14" or not actor_weights.strip():
        raise ValueError("Expected a complete V14 multiplier policy")
    nominal, np_min, np_max = map(int, header[1:4])
    q_min, q_max, r_min, r_max, mu_min, mu_max = map(float, header[4:])
    if not all(value > 0 for value in (q_min, q_max, r_min, r_max)):
        raise ValueError("V14 multiplier bounds must be positive")
    # V14 uses a piecewise log map around multiplier 1. V15 uses a straight
    # log map between absolute bounds. They coincide for reciprocal bounds.
    if abs(log(q_min) + log(q_max)) > 1e-12 or abs(log(r_min) + log(r_max)) > 1e-12:
        raise ValueError("V14 multiplier bounds are not reciprocal; conversion would change actor actions")
    rl = target["rl"]
    if (int(rl["nominal_np"]), int(rl["min_np"]), int(rl["max_np"])) != (
        nominal, np_min, np_max
    ):
        raise ValueError("V15 Np action mapping differs from the source policy")
    near(mu_min, float(rl["train_mu_min"]), "train_mu_min")
    near(mu_max, float(rl["train_mu_max"]), "train_mu_max")
    near(float(source["weights"]["terminal_scale"]),
         float(rl["terminal_scale"]), "terminal_scale")
    for index, key in enumerate(WEIGHTS):
        factor_min, factor_max = (q_min, q_max) if index < 4 else (r_min, r_max)
        source_base = float(source["weights"][key])
        if source_base <= 0:
            raise ValueError(f"Source training weight {key} must be positive")
        near(float(source["rl"]["q_scale_min" if index < 4 else "r_scale_min"]),
             factor_min, f"source {key} scale minimum")
        near(float(source["rl"]["q_scale_max" if index < 4 else "r_scale_max"]),
             factor_max, f"source {key} scale maximum")
        near(source_base * factor_min, float(rl[f"{key}_min"]), f"{key}_min")
        near(source_base * factor_max, float(rl[f"{key}_max"]), f"{key}_max")

    # Print floats from the target configuration to round-trip to the same
    # binary values that the C++ V15 loader expects.
    bounds = [str(float(rl[f"{key}_{side}"]))
              for key in WEIGHTS for side in ("min", "max")]
    new_header = " ".join((
        "DDPG_MPC_V15_ABSOLUTE", str(nominal), str(np_min), str(np_max),
        *bounds, repr(float(rl["terminal_scale"])),
        repr(float(rl["train_mu_min"])), repr(float(rl["train_mu_max"]))
    ))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(new_header + "\n" + actor_weights, encoding="utf-8")
    shutil.copy2(source_config, output.parent / "v14_training_config.ini")
    (output.parent / "conversion_manifest.txt").write_text(
        "Converted V14 multiplier actor to V15 absolute actions.\n"
        f"Source policy: {policy.resolve()}\n"
        f"Source SHA256: {hashlib.sha256(policy.read_bytes()).hexdigest()}\n"
        f"Source training configuration: {source_config.resolve()}\n"
        f"Target RL action configuration: {target_config.resolve()}\n"
        "Actor weights were copied unchanged; the equivalent absolute bounds\n"
        "were verified against the source training weights and V14 multipliers.\n"
        "This conversion does not retrain or establish performance on new CarSim scenarios.\n",
        encoding="utf-8",
    )
    print(output.resolve())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("policy", type=Path, help="V14 ddpg_policy.txt")
    parser.add_argument("--config", type=Path, default=Path("configs/default.ini"))
    parser.add_argument("--out", type=Path, required=True, help="New V15 policy path")
    args = parser.parse_args()
    convert(args.policy, args.config, args.out)


if __name__ == "__main__":
    main()
