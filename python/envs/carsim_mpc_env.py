"""CarSim-MPC Gym environment with adaptive integer Np and three Q/R weights."""
from __future__ import annotations

from pathlib import Path
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from bridge.client import BridgeClient, BridgeError


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
        self.action_space = spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32)
        # Observations are the 17 feature scalings from src/rl.cpp::makeRlObservation.
        self.observation_space = spaces.Box(
            -np.inf, np.inf, shape=(17,), dtype=np.float32
        )
        self._bridge = None
        self._episode_id = 0
        self._done = False
        for file in (self.executable, self.config, self.simfile):
            if not file.is_file():
                raise FileNotFoundError(file)

    @staticmethod
    def _observation(fields):
        if len(fields) != 17:
            raise BridgeError(f"Expected 17 observation scalars, got {len(fields)}")
        state = np.asarray(fields, dtype=np.float32)
        if not np.all(np.isfinite(state)):
            raise BridgeError("CarSim produced nonfinite observations")
        return state

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.close()
        self._episode_id += 1
        base = self.output_root / f"episode_{self._episode_id:06d}"
        self._bridge = BridgeClient(
            self.executable, self.config, self.simfile,
            base, self.output_root / f"episode_{self._episode_id:06d}.log")
        self._done = False
        try:
            fields = self._bridge.exchange("RESET", "BRIDGE_RESET")
            obs = self._observation(fields)
        except Exception:
            self.close()
            raise
        # The seed does not randomize the selected CarSim source.
        return obs, {"episode_id": self._episode_id, "seed": seed}

    def step(self, action):
        if self._bridge is None or self._done:
            raise RuntimeError("Call reset() before step(), or after episode termination")
        a = np.asarray(action, dtype=np.float64)
        if a.shape != (4,) or not np.all(np.isfinite(a)):
            raise ValueError("Action must be 4 finite numbers: np, q_y, q_psi, r_delta")
        if np.any(a < -1.0) or np.any(a > 1.0):
            raise ValueError("Action is outside [-1, 1]")
        data = self._bridge.exchange("STEP " + " ".join(format(x, ".17g") for x in a),
                                     "BRIDGE_STEP")
        if len(data) != 29:
            raise BridgeError(f"Expected 29 STEP values, got {len(data)}")
        obs = self._observation(data[:17])
        reward = float(data[17])
        terminated = bool(int(data[18]))
        truncated = bool(int(data[19]))
        self._done = terminated or truncated
        info = {
            "x_m": float(data[20]),
            "ey_m": float(data[21]),
            "epsi_rad": float(data[22]),
            "steer_rad": float(data[23]),
            "qp_failures": int(data[24]),
            "mpc_steps": int(data[26]),
            "failed": bool(int(data[27])),
            "np": int(data[28]),
        }
        # 17 obs + reward/terminated/truncated + 8 diagnostic fields + executed Np = 29.
        if info["np"] <= 0:
            raise BridgeError("C++ bridge returned nonpositive prediction horizon")
        if not np.isfinite(reward):
            raise BridgeError("Nonfinite reward")
        return obs, reward, terminated, truncated, info

    def close(self):
        if self._bridge is not None:
            self._bridge.close()
            self._bridge = None

    def export_report(self, label):
        if self._bridge is None:
            raise RuntimeError("No active bridge episode")
        self._bridge.exchange("REPORT " + label, "BRIDGE_REPORT")
