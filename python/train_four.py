"""Train DDPG, TD3 and TQC sequentially and compare them with fixed MPC."""
import argparse
import configparser
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from algorithms import ALGORITHMS
from compare_four import compare_four, comparison_limits
from evaluate import DEFAULT_CONTROLLER, DEFAULT_FIXED_CONFIG
from report import render_training_summary
from scenario import read_scenario, resolve_simfile, snapshot_config, snapshot_scenario


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--eval-frequency", type=int)
    parser.add_argument("--parity-blocks", type=int, default=10)
    parser.add_argument("--config", default="configs/carsim_python_rl.ini")
    parser.add_argument("--fixed-config", help="Fixed MPC baseline, such as the one written by sweep.py")
    parser.add_argument("--train-config", default="configs/rl_training.ini")
    parser.add_argument("--simfile")
    parser.add_argument("--bridge", default="out/python-rl-build/Release/MPC_RL_Bridge.exe")
    parser.add_argument("--controller", default=DEFAULT_CONTROLLER)
    parser.add_argument("--segments", nargs="+", type=float, default=[],
                        help="Increasing X positions that split the comparison into segments")
    parser.add_argument("--out", default="runs")
    args = parser.parse_args()
    training = configparser.ConfigParser(interpolation=None)
    with Path(args.train_config).open(encoding="utf-8-sig") as file:
        training.read_file(file)
    for key in ("episodes", "seed", "eval_frequency"):
        if getattr(args, key) is None:
            setattr(args, key, training.getint("training", key))
    if min(args.episodes, args.eval_frequency, args.parity_blocks) < 1:
        parser.error("Episode count, evaluation frequency and parity blocks must be positive")
    config = Path(args.config).resolve(strict=True)
    train_config = Path(args.train_config).resolve(strict=True)
    bridge = Path(args.bridge).resolve(strict=True)
    controller = Path(args.controller).resolve(strict=True)
    simfile = resolve_simfile(config, args.simfile)
    scenario = read_scenario(simfile)
    fixed_source = Path(args.fixed_config) if args.fixed_config else config.with_name("default.ini")
    if args.fixed_config is None and not fixed_source.is_file():
        fixed_source = DEFAULT_FIXED_CONFIG
    comparison_limits(scenario, config, fixed_source, args.segments)
    root = Path(args.out).resolve() / (
        f"four_{scenario['speed_token']}_{args.episodes}ep_seed{args.seed}_{datetime.now():%Y%m%d_%H%M%S}")
    root.mkdir(parents=True, exist_ok=False)
    frozen = snapshot_scenario(scenario, root / "case")
    frozen_config = snapshot_config(config, root / "configs")
    fixed_config = snapshot_config(fixed_source, root / "fixed_configs")
    frozen_train_config = root / "rl_training.ini"
    shutil.copy2(train_config, frozen_train_config)
    info = {"scenario": scenario, "seed": args.seed, "episodes_per_algorithm": args.episodes,
            "model_selection": "final_models",
            "segment_cuts_m": args.segments,
            "fixed_config": str(fixed_config),
            "source_config": str(config), "frozen_config": str(frozen_config),
            "source_training_config": str(train_config), "frozen_training_config": str(frozen_train_config),
            "snapshot_simfile": str(frozen), "status": "training", "models": {}}
    manifest = root / "suite_manifest.json"
    def save():
        manifest.write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
    save()
    print(f"Suite directory: {root}", flush=True)
    monitors = {}
    try:
        for algorithm in ALGORITHMS:
            output = root / ("train_" + algorithm)
            output.mkdir()
            command = [sys.executable, "-u", str(Path(__file__).with_name("train.py")),
                       "--algorithm", algorithm, "--episodes", str(args.episodes),
                       "--seed", str(args.seed), "--eval-frequency", str(args.eval_frequency),
                       "--parity-blocks", str(args.parity_blocks), "--bridge", str(bridge),
                       "--controller", str(controller), "--config", str(frozen_config),
                       "--train-config", str(frozen_train_config),
                       "--simfile", str(frozen), "--out", str(output), "--training-only"]
            print(f"Starting {algorithm.upper()}: {args.episodes} episodes. Log: {output / 'training.log'}", flush=True)
            environment = os.environ.copy()
            environment["PYTHONIOENCODING"] = "utf-8"
            with (output / "training.log").open("w", encoding="utf-8") as log:
                completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                                           cwd=Path(__file__).resolve().parent.parent,
                                           env=environment)
            if completed.returncode:
                raise RuntimeError(f"{algorithm.upper()} training failed; see {output / 'training.log'}")
            runs = [p for p in output.iterdir() if p.is_dir() and (p / "final_model.zip").is_file()]
            if len(runs) != 1:
                raise RuntimeError(f"Expected exactly one completed {algorithm.upper()} run in {output}")
            info["models"][algorithm] = str(runs[0] / "final_model.zip")
            monitors[algorithm] = runs[0] / "train_monitor.csv"
            save()
            print(f"{algorithm.upper()} complete: {info['models'][algorithm]}", flush=True)
        info["training_summary"] = str(render_training_summary(root, monitors, args.episodes, args.seed))
        info["status"] = "evaluating"
        save()
        report = compare_four(info["models"], bridge, frozen_config, frozen,
                              root / "comparison", args.seed, controller, fixed_config,
                              segments=args.segments)
        info["status"] = "complete"
        info["comparison_report"] = str(report)
        save()
        print(f"Four-controller report: {report}", flush=True)
    except Exception as exc:
        info["status"] = "failed"
        info["error"] = str(exc)
        save()
        raise


if __name__ == "__main__":
    main()
