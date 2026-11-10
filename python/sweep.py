"""Tune fixed MPC on one frozen CarSim case, then write matching DRL configs."""
import argparse
import configparser
import csv
import json
import math
from itertools import product
from pathlib import Path

from evaluate import DEFAULT_CONTROLLER, DEFAULT_FIXED_CONFIG, fixed_episode
from scenario import read_scenario, resolve_config_value, resolve_simfile, snapshot_config, snapshot_scenario


WEIGHTS = ("q_y", "q_psi", "r_delta")
NP_GROUP_SIZE = 5
RL_TEMPLATE = DEFAULT_FIXED_CONFIG.with_name("carsim_python_rl.ini")


def write_ini(path, sections):
    config = configparser.ConfigParser(interpolation=None)
    config.read_dict(sections)
    with path.open("x", encoding="utf-8") as file:
        config.write(file)


def write_training_configs(base, simfile, distance, best):
    fixed = base.with_name("sweep_fixed.ini")
    write_ini(fixed, {
        "inherit": {"base": base.name},
        "simulation": {"distance_m": distance},
        "carsim": {"simfile": simfile, "sync_browser_result": "false"},
        "horizon_schedule": {"short_np": 0},
        "mpc": {key: best[key] for key in ("np", "nc")},
        "weights": {key: best[key] for key in WEIGHTS},
        "rl": {"align_neutral_to_mpc": "false"},
    })
    rl = configparser.ConfigParser(interpolation=None)
    with RL_TEMPLATE.open(encoding="utf-8-sig") as file:
        rl.read_file(file)
    rl["inherit"]["base"] = fixed.name
    rl["rl"].update({
        "min_np": str(max(2, best["nc"], best["np"] - 12)),
        "max_np": str(min(150, best["np"] + 12)),
    })
    for key in WEIGHTS:
        rl["rl"][f"{key}_values"] = ", ".join(
            format(best[key] * factor, ".17g") for factor in (0.70, 0.85, 1, 1.15, 1.30)
        )
    learned = base.with_name("sweep_rl.ini")
    with learned.open("x", encoding="utf-8") as file:
        rl.write(file)
    return fixed, learned


def horizon_candidates(parser, name, values, bounds, default_bounds, nominal):
    minimum, maximum = (3, 149) if name == "np" else (1, 148)
    if values is not None:
        if min(values) < minimum or max(values) > maximum:
            parser.error(f"--{name} values must be in {minimum}..{maximum}")
        return sorted(set(values))
    if bounds is None:
        start, end = default_bounds
        bounds = min(start, nominal), max(end, nominal)
    start, end = bounds
    if not minimum <= start <= end <= maximum:
        parser.error(f"--{name}-range requires {minimum} <= MIN <= MAX <= {maximum}")
    return list(range(start, end + 1))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_FIXED_CONFIG)
    parser.add_argument("--controller", default=DEFAULT_CONTROLLER)
    parser.add_argument("--simfile", help="Override the Simfile in the selected INI")
    parser.add_argument("--out", type=Path, required=True, help="New output directory")
    parser.add_argument("--distance", default="reference_end", help="End X in meters or reference_end")
    for name, label in (("np", "prediction"), ("nc", "control")):
        group = parser.add_mutually_exclusive_group()
        group.add_argument(f"--{name}", nargs="+", type=int, help=f"Explicit {label} horizons")
        group.add_argument(f"--{name}-range", nargs=2, type=int, metavar=("MIN", "MAX"),
                           help=f"Inclusive integer range of {label} horizons")
    parser.add_argument("--weight-scales", nargs="+", type=float, default=(0.8, 1, 1.25),
                        help="Multipliers applied independently to Qy, Qpsi and Rdelta")
    args = parser.parse_args()

    def value(section, key):
        return resolve_config_value(args.config, section, key)

    horizons = horizon_candidates(parser, "np", args.np, args.np_range, (20, 60),
                                  int(value("mpc", "np")))
    controls = horizon_candidates(parser, "nc", args.nc, args.nc_range, (5, 16),
                                  int(value("mpc", "nc")))
    scales = sorted({1.0, *args.weight_scales})
    horizons = [np_ for np_ in horizons if np_ > controls[0]]
    if not horizons or any(not math.isfinite(scale) or scale <= 0 for scale in scales):
        parser.error("Require at least one Nc < Np pair and positive finite weight scales")
    weights = tuple(float(value("weights", key)) for key in WEIGHTS)
    controller = Path(args.controller).resolve(strict=True)
    scenario = read_scenario(resolve_simfile(args.config, args.simfile))
    root = args.out.resolve()
    root.mkdir(parents=True, exist_ok=False)
    frozen = snapshot_scenario(scenario, root / "source_case")
    base = snapshot_config(args.config, root / "configs")

    def rank(row):
        return row["rmse_y_m"], row["peak_y_m"]

    with (root / "sweep_metrics.csv").open("w", newline="", encoding="utf-8") as file:
        fields = ["stage", "np", "nc", *WEIGHTS, "eligible", "distance_m", "mae_y_m",
                  "rmse_y_m", "peak_y_m", "qp_failures", "trace", "error"]
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        index = 0

        def evaluate(stage, np_, nc, values):
            nonlocal index
            index += 1
            folder = root / "cases" / f"{index:04d}_{stage}"
            row = dict(stage=stage, np=np_, nc=nc, **dict(zip(WEIGHTS, values)))
            overrides = {"simulation.distance_m": args.distance, "mpc.np": np_,
                         "mpc.nc": nc, "horizon_schedule.short_np": 0,
                         "rl.align_neutral_to_mpc": "false",
                         **{f"weights.{key}": row[key] for key in WEIGHTS}}
            try:
                result = fixed_episode(controller, frozen, folder, base,
                                       overrides=overrides, require_complete=False)
            except RuntimeError as exc:
                row.update(eligible=False, error=str(exc))
            else:
                row.update({key: result[key] for key in fields if key in result and key != "trace"})
                row["trace"] = result["trace"].relative_to(root).as_posix()
                row["eligible"] = result["terminated"] and result["qp_failures"] == 0
            writer.writerow(row)
            file.flush()
            print(f"{index:04d} {stage} Np={np_} Nc={nc} "
                  f"RMSE={row.get('rmse_y_m', 'NA')} eligible={row['eligible']}", flush=True)
            return row

        def sweep_horizons(stage, candidates):
            return [evaluate(stage, np_, nc, weights)
                    for np_, nc in product(candidates, controls) if nc < np_]

        def best_eligible(rows):
            return min((r for r in rows if r["eligible"]), key=rank, default=None)

        results = []
        coarse_best = None
        winning_group = None
        fine_horizons = horizons
        if args.np is None:
            groups = [horizons[i:i + NP_GROUP_SIZE]
                      for i in range(0, len(horizons), NP_GROUP_SIZE)]
            winning_group = groups[0]
            if len(groups) > 1:
                representatives = {group[len(group) // 2]: group for group in groups}
                results = sweep_horizons("horizon_coarse", representatives)
                coarse_best = best_eligible(results)
                if coarse_best is None:
                    raise SystemExit("No eligible coarse candidate. Specify a narrower --np-range "
                                     f"or explicit --np values. See {file.name}")
                representative = coarse_best["np"]
                winning_group = representatives[representative]
                fine_horizons = [np_ for np_ in winning_group if np_ != representative]
                print(f"Selected Np group {winning_group[0]}–{winning_group[-1]} "
                      f"from representative Np={representative}, Nc={coarse_best['nc']}", flush=True)
        results.extend(sweep_horizons("horizon_fine", fine_horizons))
        horizon_best = best_eligible(results)
        if horizon_best is None:
            raise SystemExit(f"No complete, zero-QP-failure horizon candidate. See {file.name}")
        for factors in product(scales, repeat=len(WEIGHTS)):
            if factors != (1, 1, 1):
                results.append(evaluate("weights", horizon_best["np"], horizon_best["nc"],
                                        [weight * scale for weight, scale in zip(weights, factors)]))
    best = min((r for r in results if r["eligible"]), key=rank)
    fixed, rl = write_training_configs(base, frozen, args.distance, best)
    (root / "sweep_selection.json").write_text(
        json.dumps({"scenario": scenario, "distance": args.distance, "coarse_best": coarse_best,
                    "winning_np_group": winning_group, "horizon_best": horizon_best,
                    "best": best, "fixed_config": str(fixed), "rl_config": str(rl)}, indent=2),
        encoding="utf-8")
    print(f"Fixed MPC: {fixed}\nDRL: {rl}\nSweep metrics: {root / 'sweep_metrics.csv'}")


if __name__ == "__main__":
    main()
