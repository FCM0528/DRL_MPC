"""Validation-only checkpoint selection; never consumes a held-out test case."""
import csv
import json
import shutil
from pathlib import Path

from evaluate import _episode, fixed_episode
from scenario import read_scenario


def rank_candidate(row, baseline):
    complete = row["terminated"] and not row["truncated"] and not row["failed"]
    eligible = complete and row["qp_failures"] == 0
    iy = 1 - row["rmse_y_m"] / baseline["rmse_y_m"]
    ih = 1 - row["rmse_heading_deg"] / baseline["rmse_heading_deg"]
    return eligible, eligible and iy > 0 and ih > 0, min(iy, ih), iy, ih


def select_checkpoints(root, cls, bridge, controller, config, simfile, fixed_config,
                       seed, config_hashes):
    root = Path(root)
    baseline = fixed_episode(controller, simfile, root / "joint_validation_fixed", fixed_config)
    if baseline["qp_failures"] or not baseline["terminated"]:
        raise ValueError("Fixed validation baseline must complete without QP failures")
    checkpoints = sorted((root / "checkpoints").glob("*.zip"))
    if not checkpoints:
        raise ValueError("Joint selection requires at least one periodic checkpoint")
    candidates = checkpoints + [root / "final_model.zip"]
    rows = []
    for i, path in enumerate(candidates):
        model = cls.load(path, device="cpu")
        result = _episode(bridge, config, simfile, root / f"joint_validation_{i:04d}",
                          model, seed, path.stem, export_report=False)
        eligible, dual, score, iy, ih = rank_candidate(result, baseline)
        diagnostic_only = path.name == "final_model.zip"
        rows.append(dict(checkpoint=str(path), diagnostic_only=diagnostic_only,
                         diagnostic_dual_improvement=dual,
                         eligible=eligible and not diagnostic_only,
                         dual_improvement=dual and not diagnostic_only,
                         min_improvement=score, improvement_y=iy, improvement_heading=ih,
                         **{k: v for k, v in result.items() if k != "trace"}))
    with (root / "joint_validation.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.scatter([baseline["rmse_y_m"]], [baseline["rmse_heading_deg"]],
               marker="x", s=90, color="black", label="fixed MPC (validation)")
    for label, selected, color in (("qualified", True, "#14856a"),
                                   ("other checkpoints", False, "#b8624f")):
        group = [row for row in rows if row["dual_improvement"] is selected]
        if group:
            ax.scatter([row["rmse_y_m"] for row in group],
                       [row["rmse_heading_deg"] for row in group],
                       color=color, label=label)
    ax.set(xlabel="Lateral RMSE (m)", ylabel="Body heading RMSE (deg)",
           title="Validation checkpoints only")
    ax.legend()
    fig.tight_layout()
    fig.savefig(root / "joint_validation_pareto.png", dpi=160)
    plt.close(fig)
    qualified = [row for row in rows if row["dual_improvement"]]
    chosen = max(qualified, key=lambda row: row["min_improvement"]) if qualified else None
    if chosen:
        shutil.copy2(chosen["checkpoint"], root / "best_joint_model.zip")
    pareto = [row["checkpoint"] for row in rows if row["eligible"] and not any(
        other["eligible"] and other["rmse_y_m"] <= row["rmse_y_m"] and
        other["rmse_heading_deg"] <= row["rmse_heading_deg"] and
        (other["rmse_y_m"] < row["rmse_y_m"] or
         other["rmse_heading_deg"] < row["rmse_heading_deg"])
        for other in rows)]
    result = dict(rule="complete, zero QP failures, dual RMSE improvement; maximize minimum relative improvement",
                  qualified=bool(chosen), selected=chosen["checkpoint"] if chosen else None,
                  pareto_candidates=pareto, baseline_y_rmse_m=baseline["rmse_y_m"],
                  baseline_heading_rmse_deg=baseline["rmse_heading_deg"],
                  validation_simfile=str(simfile), validation_case_sha256={
                      key: read_scenario(simfile)[key] for key in ("simfile_sha256", "parsfile_sha256")},
                  config_sha256=config_hashes, seed=seed)
    (root / "checkpoint_selection.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
