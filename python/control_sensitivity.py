"""One-factor-at-a-time fixed MPC sensitivity on the frozen CarSim case."""
import argparse
import csv
from pathlib import Path

import numpy as np
from evaluate import fixed_episode
from report import control_trace


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--controller", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--simfile", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    factors = {"mpc.np": (28, 40, 52), "weights.q_y": (8050, 11500, 14950),
               "weights.q_psi": (1960, 2800, 3640),
               "weights.r_delta": (7560, 10800, 14040)}
    rows = []
    for factor, values in factors.items():
        for value in values:
            label = factor.replace(".", "_") + "_" + str(value)
            result = fixed_episode(args.controller, args.simfile, out / label, args.config,
                                   overrides={factor: value}, require_complete=False)
            trace = control_trace(result["trace"])
            command = trace.get("steer_rad", np.array([np.nan]))
            actual = trace.get("actual_steer_rad", np.array([np.nan]))
            rows.append(dict(factor=factor, value=value, completed=result["terminated"],
                             rmse_y_m=result["rmse_y_m"], peak_y_m=result["peak_y_m"],
                             rmse_heading_deg=result["rmse_heading_deg"],
                             peak_heading_deg=float(np.rad2deg(np.nanmax(np.abs(trace["epsi_rad"])))),
                             qp_failures=result["qp_failures"],
                             peak_command_rad=float(np.nanmax(np.abs(command))),
                             peak_actual_rad=float(np.nanmax(np.abs(actual))),
                             command_rms_rad=float(np.sqrt(np.nanmean(command**2))),
                             actual_rms_rad=float(np.sqrt(np.nanmean(actual**2)))))
    with (out / "sensitivity.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (out / "assessment.txt").open("w", encoding="utf-8") as file:
        for factor in factors:
            subset = [row for row in rows if row["factor"] == factor]
            commands = [row["command_rms_rad"] for row in subset]
            lateral = [row["rmse_y_m"] for row in subset]
            heading = [row["rmse_heading_deg"] for row in subset]
            weak = (max(commands) - min(commands) < 1e-4 and
                    max(lateral) - min(lateral) < 1e-3 and
                    max(heading) - min(heading) < 0.01)
            file.write(f"{factor}: weak_response={weak}; command_rms_span_rad={max(commands)-min(commands):.6g}; "
                       f"lateral_rmse_span_m={max(lateral)-min(lateral):.6g}; "
                       f"heading_rmse_span_deg={max(heading)-min(heading):.6g}\n")


if __name__ == "__main__":
    main()
