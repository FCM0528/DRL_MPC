"""A one-episode-per-subprocess bridge to CarSim's stateful VS Solver DLL."""
from __future__ import annotations

import subprocess
from pathlib import Path
import numpy as np


class BridgeError(RuntimeError):
    pass


class BridgeClient:
    def __init__(self, executable: Path, config: Path, simfile: Path,
                 episode_dir: Path, log_file: Path):
        self.process = None
        self._stderr = None
        self.episode_dir = Path(episode_dir)
        self.episode_dir.parent.mkdir(parents=True, exist_ok=True)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        self._stderr = open(log_file, "w", encoding="utf-8", errors="replace")
        self.process = subprocess.Popen(
            [str(executable.resolve()), str(config.resolve()),
             str(simfile.resolve()), str(self.episode_dir.resolve())],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self._stderr,
            text=True, encoding="utf-8", errors="replace", bufsize=1,
            cwd=str(Path(config).resolve().parent.parent),
        )

    def exchange(self, command: str, response_tag: str) -> list[str]:
        if self.process is None or self.process.poll() is not None:
            raise BridgeError(f"Bridge process has exited. See log in {self.episode_dir.parent}")
        assert self.process.stdin and self.process.stdout
        self.process.stdin.write(command + "\n")
        self.process.stdin.flush()
        # Solver DLL may print diagnostics; accept only framed response lines.
        for _ in range(5000):
            line = self.process.stdout.readline()
            if not line:
                raise BridgeError(f"C++ bridge exited without a reply. See stderr log near {self.episode_dir}")
            fields = line.strip().split()
            if not fields or not fields[0].startswith("BRIDGE_"):
                continue
            if fields[0] == "BRIDGE_ERROR":
                raise BridgeError("C++: " + " ".join(fields[1:]))
            if fields[0] != response_tag:
                raise BridgeError(f"Unexpected bridge reply: {line.strip()}")
            return fields[1:]
        raise BridgeError("No framed reply after 5000 diagnostic lines")

    def close(self):
        p = self.process
        if p:
            if p.poll() is None:
                try:
                    self.exchange("CLOSE", "BRIDGE_CLOSE")
                    p.wait(timeout=5)
                except (BridgeError, BrokenPipeError, OSError, TimeoutError, subprocess.TimeoutExpired):
                    p.terminate()
                    try:
                        p.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        p.kill()
                        p.wait()
            if p.stdin:
                try:
                    p.stdin.close()
                except OSError:
                    pass
            if p.stdout:
                try:
                    p.stdout.close()
                except OSError:
                    pass
            self.process = None
        if self._stderr:
            self._stderr.close()
            self._stderr = None
