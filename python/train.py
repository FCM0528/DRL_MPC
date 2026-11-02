"""Train DDPG, TD3 or TQC against the existing C++ CarSim/MPC bridge."""
import argparse
import configparser
import json
import math
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, StopTrainingOnMaxEpisodes
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise
from algorithms import ConfiguredDDPG, ConfiguredTD3, ConfiguredTQC

from envs.carsim_mpc_env import CarSimMPCEnv
from evaluate import evaluate_models
from parity import check_neutral_parity
from scenario import read_scenario, resolve_simfile, snapshot_scenario


class TrainingEnv(CarSimMPCEnv):
    """Keep only the current solver episode; evaluation preserves its traces."""
    def reset(self, *, seed=None, options=None):
        previous = self.output_root / f"episode_{self._episode_id:06d}"
        log = previous.with_suffix(".log")
        result = super().reset(seed=seed, options=options)
        if previous.is_dir():
            shutil.rmtree(previous)
        if log.is_file():
            log.unlink()
        return result


def load_training_settings(path) -> dict:
    parser = configparser.ConfigParser(interpolation=None)
    with Path(path).open(encoding="utf-8-sig") as file:
        parser.read_file(file)
    schema = {
        "training": {"episodes": int, "seed": int, "eval_frequency": int,
                     "buffer_size": int, "learning_starts": int,
                     "batch_size": int, "train_freq": int, "gradient_steps": int},
        "ddpg": {"actor_learning_rate": float, "critic_learning_rate": float,
                 "tau": float, "actor_hidden_layers": int, "actor_hidden_width": int,
                 "critic_hidden_layers": int, "critic_hidden_width": int,
                 "exploration_noise_sigma": float},
        "td3": {"actor_learning_rate": float, "critic_learning_rate": float,
                "tau": float, "actor_hidden_layers": int, "actor_hidden_width": int,
                "critic_hidden_layers": int, "critic_hidden_width": int,
                "n_critics": int, "policy_delay": int, "target_policy_noise": float,
                "target_noise_clip": float, "exploration_noise_sigma": float},
        "tqc": {"profile": str, "actor_learning_rate": float,
                "critic_learning_rate": float, "entropy_learning_rate": float,
                "tau": float, "hidden_layers": int, "hidden_width": int,
                "n_critics": int, "n_quantiles": int, "drop": int,
                "target_update_interval": int, "entropy_coefficient": str,
                "target_entropy": str},
    }
    if set(parser.sections()) != set(schema):
        raise ValueError("Training config needs [training], [ddpg], [td3], [tqc]")
    result = {}
    for section, fields in schema.items():
        if set(parser[section]) != set(fields):
            raise ValueError(f"Unknown or missing setting in [{section}]")
        for key, kind in fields.items():
            result[key if section == "training" else f"{section}_{key}"] = kind(parser[section][key])
    if (result["episodes"] < 1 or result["seed"] < 0 or
        result["eval_frequency"] < 1 or
        result["buffer_size"] < 1 or not 0 <= result["learning_starts"] < result["buffer_size"] or
        result["batch_size"] < 1 or result["train_freq"] < 1 or
        result["gradient_steps"] < 1 or
        result["tqc_profile"] not in ("practical", "paper") or
        not 0 <= result["tqc_drop"] < result["tqc_n_quantiles"] or
        any(not math.isfinite(value) for value in result.values() if isinstance(value, float))):
        raise ValueError("Invalid Python training setting")
    for algorithm in ("ddpg", "td3", "tqc"):
        if (result[f"{algorithm}_actor_learning_rate"] <= 0 or
            result[f"{algorithm}_critic_learning_rate"] <= 0 or
            not 0 < result[f"{algorithm}_tau"] <= 1):
            raise ValueError(f"Invalid {algorithm} learning rates or tau")
    for algorithm in ("ddpg", "td3"):
        if (min(result[f"{algorithm}_{key}"] for key in
                ("actor_hidden_layers", "actor_hidden_width", "critic_hidden_layers",
                 "critic_hidden_width")) < 1 or
            result[f"{algorithm}_exploration_noise_sigma"] < 0):
            raise ValueError(f"Invalid {algorithm} network or noise setting")
    if (result["td3_n_critics"] < 2 or result["td3_policy_delay"] < 1 or
        result["td3_target_policy_noise"] < 0 or result["td3_target_noise_clip"] < 0 or
        min(result[f"tqc_{key}"] for key in ("hidden_layers", "hidden_width", "n_critics",
                                                "n_quantiles", "target_update_interval")) < 1 or
        result["tqc_n_quantiles"] < 2 or result["tqc_entropy_learning_rate"] <= 0):
        raise ValueError("Invalid TD3 or TQC architecture and update settings")
    for key in ("tqc_entropy_coefficient", "tqc_target_entropy"):
        if result[key] != "auto":
            value = float(result[key])
            if not math.isfinite(value) or (key.endswith("coefficient") and value <= 0):
                raise ValueError(f"Invalid {key}")
    return result


def common_design(settings: dict) -> dict:
    return {key: settings[key] for key in ("buffer_size", "learning_starts",
                                           "batch_size", "train_freq", "gradient_steps")}


def deterministic_design(algorithm: str, settings: dict) -> dict:
    prefix = algorithm + "_"
    design = dict(**common_design(settings),
                  actor_learning_rate=settings[prefix + "actor_learning_rate"],
                  critic_learning_rate=settings[prefix + "critic_learning_rate"],
                  tau=settings[prefix + "tau"],
                  policy_kwargs=dict(net_arch={
                      "pi": [settings[prefix + "actor_hidden_width"]] * settings[prefix + "actor_hidden_layers"],
                      "qf": [settings[prefix + "critic_hidden_width"]] * settings[prefix + "critic_hidden_layers"],
                  }))
    if algorithm == "td3":
        design.update(policy_delay=settings["td3_policy_delay"],
                      target_policy_noise=settings["td3_target_policy_noise"],
                      target_noise_clip=settings["td3_target_noise_clip"])
        design["policy_kwargs"]["n_critics"] = settings["td3_n_critics"]
    return design


def tqc_design(profile: str, drop: int, settings: dict) -> dict:
    """Paper TQC: pooled quantile critics, upper-tail truncation, SAC entropy."""
    n_critics = settings["tqc_n_critics"]
    net_arch = [settings["tqc_hidden_width"]] * settings["tqc_hidden_layers"]
    if profile == "paper":
        n_critics, net_arch = 5, [512, 512, 512]
    n_quantiles = settings["tqc_n_quantiles"]
    if not 0 <= drop < n_quantiles:
        raise ValueError("TQC drop count must be below n_quantiles")
    return dict(
        **common_design(settings),
        actor_learning_rate=settings["tqc_actor_learning_rate"],
        critic_learning_rate=settings["tqc_critic_learning_rate"],
        entropy_learning_rate=settings["tqc_entropy_learning_rate"],
        tau=settings["tqc_tau"],
        ent_coef=("auto" if settings["tqc_entropy_coefficient"] == "auto" else
                  float(settings["tqc_entropy_coefficient"])),
        target_entropy=("auto" if settings["tqc_target_entropy"] == "auto" else
                        float(settings["tqc_target_entropy"])),
        target_update_interval=settings["tqc_target_update_interval"],
        top_quantiles_to_drop_per_net=drop,
        policy_kwargs=dict(net_arch=net_arch, n_critics=n_critics,
                           n_quantiles=n_quantiles),
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--algorithm", choices=("ddpg", "td3", "tqc"), required=True)
    p.add_argument("--bridge", default="out/python-rl-build/Release/MPC_RL_Bridge.exe")
    p.add_argument("--controller", default="out/python-rl-build/Release/MPC_DoubleLaneChange.exe")
    p.add_argument("--config", default="configs/carsim_python_rl.ini")
    p.add_argument("--train-config", default="configs/rl_training.ini")
    p.add_argument("--simfile", help="CarSim Simfile; defaults to carsim.simfile from config")
    p.add_argument("--episodes", type=int,
                   help="Number of completed training episodes in this run")
    p.add_argument("--seed", type=int)
    p.add_argument("--eval-frequency", type=int)
    p.add_argument("--out", default="runs/python_rl")
    p.add_argument("--resume", help="Continue a Python SB3 model of the same algorithm")
    p.add_argument("--parity-blocks", type=int, default=10)
    p.add_argument("--tqc-profile", choices=("practical", "paper"),
                   help="Practical CPU profile: 3 critics x 25 quantiles, 2x256; paper: 5 critics, 3x512")
    p.add_argument("--tqc-drop", type=int,
                   help="Highest target quantiles removed per critic (paper tests 0..5)")
    args = p.parse_args()
    settings = load_training_settings(args.train_config)
    for arg, key in (("episodes", "episodes"), ("seed", "seed"),
                     ("eval_frequency", "eval_frequency"),
                     ("tqc_profile", "tqc_profile"), ("tqc_drop", "tqc_drop")):
        if getattr(args, arg) is None:
            setattr(args, arg, settings[key])
        settings[key] = getattr(args, arg)
    if args.episodes < 1 or args.eval_frequency < 1 or args.parity_blocks < 1:
        p.error("--episodes, --eval-frequency and --parity-blocks must be positive")
    if not 0 <= args.tqc_drop < settings["tqc_n_quantiles"]:
        p.error("--tqc-drop must be below tqc.n_quantiles")
    args.simfile = str(resolve_simfile(args.config, args.simfile))
    scenario = read_scenario(args.simfile)
    run_name = (f"{args.algorithm}_{scenario['speed_token']}_{args.episodes}episodes_"
                f"seed{args.seed}_{datetime.now():%Y%m%d_%H%M%S}")
    root = Path(args.out).resolve() / run_name
    root.mkdir(parents=True, exist_ok=False)
    frozen_simfile = snapshot_scenario(scenario, root / "source_case")
    source_config = Path(args.config).resolve(strict=True)
    config_snapshot = root / "source_configs"
    config_snapshot.mkdir()
    for file in source_config.parent.iterdir():
        if file.is_file() and file.suffix.lower() in (".ini", ".csv"):
            shutil.copy2(file, config_snapshot / file.name)
    frozen_config = config_snapshot / source_config.name
    manifest = {**vars(args), "training_settings": settings, "scenario": scenario,
                "bridge": str(Path(args.bridge).resolve()),
                "config": str(source_config), "frozen_config": str(frozen_config),
                "simfile": str(Path(args.simfile).resolve()),
                "snapshot_simfile": str(frozen_simfile)}
    manifest["algorithm_design"] = (tqc_design(args.tqc_profile, args.tqc_drop, settings)
                                    if args.algorithm == "tqc" else
                                    deterministic_design(args.algorithm, settings))
    manifest["reward_scheme"] = "negative_log1p_v3"
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    shutil.copy2(source_config, root / "source_config.ini")
    shutil.copy2(args.train_config, root / "source_training.ini")
    args.simfile = str(frozen_simfile)
    args.config = str(frozen_config)
    parity = check_neutral_parity(args.bridge, args.controller, args.config,
                                  args.simfile, root / "neutral_parity", args.parity_blocks)
    manifest["neutral_parity"] = parity
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    train_env = Monitor(TrainingEnv(args.bridge, args.config, args.simfile, root / "solver_train"),
                        filename=str(root / "train_monitor.csv"))
    validation = Monitor(CarSimMPCEnv(args.bridge, args.config, args.simfile, root / "solver_validation"))
    cls = {"ddpg": ConfiguredDDPG, "td3": ConfiguredTD3,
           "tqc": ConfiguredTQC}[args.algorithm]
    try:
        if args.resume:
            model_path = Path(args.resume).resolve()
            previous_manifest = next((parent / "manifest.json" for parent in model_path.parents
                                      if (parent / "manifest.json").is_file()), None)
            if previous_manifest is None:
                raise ValueError("Resume requires a run manifest with the current reward scheme")
            previous = json.loads(previous_manifest.read_text(encoding="utf-8"))
            if previous.get("reward_scheme") != manifest["reward_scheme"] or \
               previous.get("algorithm_design") != manifest["algorithm_design"]:
                raise ValueError("Resumed model design or reward differs; start a fresh run")
            model = cls.load(args.resume, env=train_env, tensorboard_log=str(root / "tensorboard"))
            if abs(model.gamma - parity["gamma_per_decision"]) > 1e-12:
                raise ValueError("Resumed model gamma differs from C++ decision discount")
        else:
            kwargs = dict(seed=args.seed, verbose=1, tensorboard_log=str(root / "tensorboard"),
                          gamma=parity["gamma_per_decision"], **manifest["algorithm_design"])
            if args.algorithm in ("ddpg", "td3"):
                sigma = settings[f"{args.algorithm}_exploration_noise_sigma"]
                kwargs["action_noise"] = NormalActionNoise(np.zeros(4), np.full(4, sigma))
            model = cls("MlpPolicy", train_env, **kwargs)
        manifest["training_start_timesteps"] = int(model.num_timesteps) if args.resume else 0
        (root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        callbacks = [CheckpointCallback(save_freq=args.eval_frequency,
                                        save_path=str(root / "checkpoints")),
                     EvalCallback(validation, best_model_save_path=str(root / "best"),
                                  log_path=str(root / "validation"),
                                  eval_freq=args.eval_frequency, n_eval_episodes=1,
                                  deterministic=True),
                     StopTrainingOnMaxEpisodes(max_episodes=args.episodes, verbose=1)]
        model.learn(total_timesteps=2_147_483_647, callback=callbacks,
                    reset_num_timesteps=not bool(args.resume))
        model.save(root / "final_model")
        shutil.copy2(root / "final_model.zip", root / f"{run_name}_final_model.zip")
        best_model = root / "best" / "best_model.zip"
        if best_model.is_file():
            shutil.copy2(best_model, root / f"{run_name}_best_validation_model.zip")
    finally:
        train_env.close()
        validation.close()
    evaluate_models(args.algorithm, [root / "final_model.zip"], args.bridge, args.config,
                    args.simfile, [args.seed], root / "evaluation", root)
    if best_model.is_file():
        evaluate_models(args.algorithm, [best_model], args.bridge, args.config,
                        args.simfile, [args.seed], root / "best_evaluation", root)
    (root / "文件说明.txt").write_text(
        f"训练：{args.algorithm.upper()}，目标 {args.episodes} 回合，种子 {args.seed}\n"
        f"CarSim 目标车速：{scenario['target_speed_kmh'] if scenario['target_speed_kmh'] is not None else '非单一恒定值'} km/h（实际车速见报告）\n"
        f"工况参数：{scenario['parsfile']}\nSHA-256：{scenario['parsfile_sha256']}\n"
        f"{run_name}_final_model.zip：最终模型\n"
        f"{run_name}_best_validation_model.zip：验证回报最高模型（有验证时）\n"
        "final_model.zip、best/best_model.zip：兼容原有命令的模型文件\n"
        "train_monitor.csv：每回合回报与长度；validation/evaluations.npz：验证数据\n"
        "evaluation/training_and_control_report.html：训练与控制对照汇总\n"
        "best_evaluation/training_and_control_report.html：验证最佳模型与中性 MPC 对照（有验证时）\n"
        "evaluation/seed_*_*/episode_000001/report.html：C++ 完整控制图与指标\n"
        "checkpoints/：按决策步数保存的中间检查点；tensorboard/：训练日志\n",
        encoding="utf-8")
    print(f"训练与控制汇总报告: {root / 'evaluation' / 'training_and_control_report.html'}")


if __name__ == "__main__":
    main()
