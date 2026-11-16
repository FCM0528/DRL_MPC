"""CarSim-MPC Gym environment with adaptive integer Np and three Q/R weights."""
from __future__ import annotations

from pathlib import Path
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from bridge.client import BridgeClient, BridgeError
from scenario import resolve_config_value
from action_mapping import action_contract, bridge_action
from scenario_sets import SAMPLING_RULE


class CarSimMPCEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, executable: str, config: str, simfile: str,
                 output_root: str = "runs/carsim_bridge"):
        super().__init__()
        self.executable = Path(executable).resolve()
        self.config = Path(config).resolve()
        self.simfile = Path(simfile).resolve()
        self.output_root = Path(output_root).resolve()
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.policy_action_names, self.action_mapping_version = action_contract(self.config)
        self.policy_action_dims = len(self.policy_action_names)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(self.policy_action_dims,), dtype=np.float32)
        self.observation_profile = resolve_config_value(
            self.config, "rl", "observation_profile", "road_preview_v2"
        ).lower()
        dimensions = {"road_preview": 12, "road_preview_v2": 14, "road_preview_joint": 20}
        if self.observation_profile not in dimensions:
            raise ValueError(f"Unknown observation profile: {self.observation_profile}")
        self._observation_size = dimensions[self.observation_profile]
        self.observation_space = spaces.Box(
            -np.inf, np.inf, shape=(self._observation_size,), dtype=np.float32
        )
        self._bridge = None
        self._episode_id = 0
        self._done = False
        self.scenario_pool = [self.simfile]
        self.scenario_decision_counts = {}
        self.scenario_exposure = {}
        self._episode_decisions = 0
        self._episode_return = 0.0
        self._episode_qp = 0
        self._episode_recorded = True
        self._case_index = 0
        for file in (self.executable, self.config, self.simfile):
            if not file.is_file():
                raise FileNotFoundError(file)

    def _observation(self, fields):
        if len(fields) != self._observation_size:
            raise BridgeError(f"{self.observation_profile} expects {self._observation_size} "
                              f"observation scalars, got {len(fields)}")
        state = np.asarray(fields, dtype=np.float32)
        if not np.all(np.isfinite(state)):
            raise BridgeError("CarSim produced nonfinite observations")
        return state

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.close()
        self._episode_id += 1
        selected = int(self.np_random.integers(len(self.scenario_pool)))
        self._case_index = selected
        self.simfile = Path(self.scenario_pool[selected]).resolve(strict=True)
        import json
        with (self.output_root / "episode_scenarios.jsonl").open("a", encoding="utf-8") as log:
            log.write(json.dumps(dict(episode=self._episode_id, case_index=selected,
                                      simfile=str(self.simfile), seed=seed,
                                      sampling_rule=SAMPLING_RULE,pool_size=len(self.scenario_pool))) + "\n")
        base = self.output_root / f"episode_{self._episode_id:06d}"
        self._bridge = BridgeClient(
            self.executable, self.config, self.simfile,
            base, self.output_root / f"episode_{self._episode_id:06d}.log")
        self._done = False
        self._episode_decisions = 0
        self._episode_return = 0.0
        self._episode_qp = 0
        self._episode_recorded = False
        try:
            fields = self._bridge.exchange("RESET", "BRIDGE_RESET")
            obs = self._observation(fields)
        except Exception:
            self.close()
            raise
        # Seed controls case sampling; the source's physical content stays fixed.
        return obs, {"episode_id": self._episode_id, "seed": seed,
                     "case_index":selected,"simfile":str(self.simfile)}

    def step(self, action):
        if self._bridge is None or self._done:
            raise RuntimeError("Call reset() before step(), or after episode termination")
        a = np.asarray(action, dtype=np.float64)
        if a.shape != (self.policy_action_dims,) or not np.all(np.isfinite(a)):
            raise ValueError(f"Action must be {self.policy_action_dims} finite numbers")
        if np.any(a < -1.0) or np.any(a > 1.0):
            raise ValueError("Action is outside [-1, 1]")
        mapped = bridge_action(a, self.policy_action_names)
        data = self._bridge.exchange("STEP " + " ".join(format(x, ".17g") for x in mapped),
                                     "BRIDGE_STEP")
        n = self._observation_size
        if len(data) != n + 12:
            raise BridgeError(f"Expected {n + 12} STEP values, got {len(data)}")
        obs = self._observation(data[:n])
        reward, terminated, truncated, x, ey, epsi, steer, qp, _, steps, failed, np_horizon = data[n:]
        reward = float(reward)
        terminated = bool(int(terminated))
        truncated = bool(int(truncated))
        self._done = terminated or truncated
        info = {
            "x_m": float(x),
            "ey_m": float(ey),
            "epsi_rad": float(epsi),
            "steer_rad": float(steer),
            "qp_failures": int(qp),
            "mpc_steps": int(steps),
            "failed": bool(int(failed)),
            "np": int(np_horizon),
        }
        if info["np"] <= 0:
            raise BridgeError("C++ bridge returned nonpositive prediction horizon")
        if not np.isfinite(reward):
            raise BridgeError("Nonfinite reward")
        path=str(self.simfile)
        self.scenario_decision_counts[path]=self.scenario_decision_counts.get(path,0)+1
        counts=self.scenario_exposure.setdefault(path,dict(episodes=0,completed=0,decisions=0,reward_sum=0.,qp_failures=0))
        if self._episode_decisions == 0: counts['episodes'] += 1
        counts['decisions'] += 1
        counts['reward_sum'] += reward
        # C++ emits failures within this decision block, not a cumulative counter.
        counts['qp_failures'] += info['qp_failures']
        self._episode_decisions += 1
        self._episode_return += reward
        self._episode_qp += info['qp_failures']
        if self._done:
            counts['completed'] += int(terminated and not truncated and not info['failed'])
            self._record_episode(terminated and not truncated and not info['failed'])
        info.update(case_index=self._case_index,simfile=path)
        return obs, reward, terminated, truncated, info

    def _record_episode(self, completed=False):
        if self._episode_recorded or not self._episode_decisions:
            return
        import json
        with (self.output_root/'episode_outcomes.jsonl').open('a',encoding='utf-8') as log:
            log.write(json.dumps(dict(episode=self._episode_id,simfile=str(self.simfile),
                case_index=self._case_index,decisions=self._episode_decisions,
                return_=self._episode_return,qp_failures=self._episode_qp,
                completed=bool(completed),sampling_rule=SAMPLING_RULE))+'\n')
        self._episode_recorded = True

    def close(self):
        self._record_episode()
        if self._bridge is not None:
            self._bridge.close()
            self._bridge = None

    def export_report(self, label):
        if self._bridge is None:
            raise RuntimeError("No active bridge episode")
        self._bridge.exchange("REPORT " + label, "BRIDGE_REPORT")
