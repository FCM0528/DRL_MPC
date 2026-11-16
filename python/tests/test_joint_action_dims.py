import numpy as np
import pytest

from envs.carsim_mpc_env import CarSimMPCEnv


@pytest.mark.parametrize("dims", [2, 3, 4])
def test_policy_actions_pad_to_four_bridge_actions(tmp_path, dims):
    exe, simfile, config = (tmp_path / name for name in ("bridge.exe", "case.sim", "case.ini"))
    exe.write_bytes(b"test")
    simfile.write_bytes(b"test")
    config.write_text(f"[rl]\npolicy_action_dims = {dims}\nobservation_profile = road_preview_joint\n")
    env = CarSimMPCEnv(str(exe), str(config), str(simfile), str(tmp_path / "out"))
    class Bridge:
        def exchange(self, command, tag):
            assert tag == "BRIDGE_STEP"
            assert len(command.split()) == 5
            assert command.split()[1 + dims:] == ["0"] * (4 - dims)
            return [0.0] * 20 + [0.0, 0, 0, 1.0, 0.0, 0.0, 0.0, 0, 0, 1, 0, 40]
        def close(self):
            pass
    env._bridge = Bridge()
    obs, *_ = env.step(np.zeros(dims))
    assert obs.shape == (20,)
    env.close()
