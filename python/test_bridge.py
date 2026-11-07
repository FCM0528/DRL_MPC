"""Test the C++ bridge without any RL agent: neutral 4D action or adaptive-Np sweep.

  python python/test_bridge.py --exe out/build/x64-Release/Release/MPC_RL_Bridge.exe \\
      --config configs/carsim_python_rl.ini \\
      --simfile policies/fixed_speed/v60/source/simfile.sim
"""
import argparse
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))
from envs.carsim_mpc_env import CarSimMPCEnv
from parity import bridge_params


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--exe", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--simfile", required=True)
    p.add_argument("--output-root", default="runs/carsim_bridge_smoke")
    p.add_argument("--exercise-np", action="store_true",
                   help="Cycle Np action over -1, 0, +1 with fixed neutral Q/R")
    p.add_argument("--max-blocks", type=int, default=0,
                   help="Stop after this many RL blocks (0 means full episode)")
    args = p.parse_args()
    if not args.exercise_np:
        nominal_np = bridge_params(args.exe, args.config, args.simfile, args.output_root)["np"]
    env = CarSimMPCEnv(args.exe, args.config, args.simfile, args.output_root)
    try:
        obs, info = env.reset(seed=42)
        print("RESET OK", "observation.shape=", obs.shape, info)
        total = 0.0
        count = 0
        seen_nps = set()
        while True:
            # Each np setting is maintained for 4 consecutive decision blocks.
            np_action = (-1.0, 0.0, 1.0)[(count // 4) % 3] if args.exercise_np else 0.0
            obs, reward, terminated, truncated, info = env.step(
                [np_action, 0.0, 0.0, 0.0])
            seen_nps.add(info["np"])
            total += reward
            count += 1
            if count % 20 == 0 or terminated or truncated:
                print(f"step={count} x={info['x_m']:.2f} m, "
                      f"ey={info['ey_m']:.4f} m, Np={info['np']}, reward={reward:.4f}, "
                      f"qp_fails={info['qp_failures']}")
            if terminated or truncated:
                print("EPISODE END", "terminated", terminated, "truncated", truncated,
                      "failed", info["failed"], "blocks", count, "reward", total)
                break
            if args.max_blocks and count >= args.max_blocks:
                print("STOPPED at requested max-blocks; this is not a completed CarSim episode")
                break
        if args.exercise_np:
            if len(seen_nps) < 3:
                raise RuntimeError(f"Np adaptivity failed; observed only {sorted(seen_nps)}")
            print(f"NP ADAPTIVITY PASS; observed executed integer horizons: {sorted(seen_nps)}")
        else:
            if seen_nps != {nominal_np}:
                raise RuntimeError(f"Neutral action expected Np={nominal_np}, got {sorted(seen_nps)}")
            if not (terminated or truncated) and args.max_blocks == 0:
                raise RuntimeError("Bridge run stopped before terminal event")
            if terminated or truncated:
                if not terminated or info["failed"]:
                    raise RuntimeError("Baseline bridge run did not finish the lane change safely")
            print("BRIDGE NEUTRAL PASS; check recorded trace parity with existing fixed MPC")
    finally:
        env.close()


if __name__ == "__main__":
    main()
