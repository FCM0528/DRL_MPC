import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bridge.client import BridgeClient


def test_client_mock_protocol():
    if sys.platform == "win32":
        # This fixture uses a POSIX shell launcher; Windows integration is
        # exercised by python/test_bridge.py with the actual C++ executable.
        return
    with tempfile.TemporaryDirectory() as tmp:
        parent = Path(tmp)
        (parent / "configs").mkdir()
        cfg = parent / "configs" / "mock.ini"
        sim = parent / "mock.sim"
        cfg.write_text("mock", encoding="utf-8")
        sim.write_text("mock", encoding="utf-8")
        mock = Path(__file__).with_name("mock_bridge_server.py")
        launcher = parent / "mock_bridge"
        launcher.write_text(f"#!/bin/sh\nexec '{sys.executable}' '{mock}' \"$@\"\n")
        launcher.chmod(0o755)
        client = BridgeClient(launcher, cfg, sim, parent / "out", parent / "log.txt")
        try:
            reset = client.exchange("RESET", "BRIDGE_RESET")
            assert len(reset) == 17
            step = client.exchange("STEP 0 0 0 0", "BRIDGE_STEP")
            assert len(step) == 29
            assert float(step[17]) == -0.4
            assert int(step[18]) == 1
            assert int(step[28]) == 25
        finally:
            client.close()
