"""Protocol-only fake: does NOT simulate CarSim or MPC."""
import sys
for line in sys.stdin:
    key = line.split()[0]
    if key == "RESET":
        print("BRIDGE_RESET " + " ".join("0" for _ in range(17)), flush=True)
    elif key == "STEP":
        # 17 observations + reward + terminated + truncated + 8 diagnostic fields + executed Np
        result = ["0"] * 17 + ["-0.4", "1", "0", "201", "0.02", "0.001", "0.004", "0", "0", "4", "0", "25"]
        print("BRIDGE_STEP " + " ".join(result), flush=True)
    elif key == "CLOSE":
        print("BRIDGE_CLOSE", flush=True)
        break
    else:
        print("BRIDGE_ERROR Bad command", flush=True)
        sys.exit(1)
