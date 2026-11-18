import csv
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluate import sensor_metrics
from report import control_trace


def test_rotating_frame_force_balance_and_partial_interval():
    def constant(value):
        return np.full(3, value)
    data = dict(time_s=np.array([0.0, 0.04, 0.055]), vx_mps=constant(20.0),
                vy_mps=constant(2.0), yaw_rate_radps=constant(0.2),
                body_ax_mps2=constant(-0.4), body_ay_mps2=constant(4.0),
                measured_beta_rad=constant(np.arctan2(2.0, 20.0)),
                actual_steer_rad=constant(0.05), steer_rad=constant(0.04),
                front_lateral_force_n=constant(2000 / np.cos(0.05)),
                rear_lateral_force_n=constant(2000),
                driver_curvature_inv_m=constant(0), reference_curvature_inv_m=constant(0.01))
    metrics = sensor_metrics(data, dict(mass_kg=1000, iz_kgm2=2000, a_m=1, b_m=1))
    for key in ("ax_transport_rmse_mps2", "ay_transport_rmse_mps2",
                "beta_consistency_rmse_deg", "lateral_force_balance_rmse_mps2",
                "yaw_force_balance_rmse_radps2"):
        assert metrics[key] == pytest.approx(0, abs=1e-12)
    assert metrics["peak_measured_ay_mps2"] == 4.0
    assert metrics["steer_tracking_rmse_deg"] == pytest.approx(np.rad2deg(0.01))
    assert metrics["driver_target_curvature_rmse_inv_m"] == pytest.approx(0.01)
    data["time_s"][2] = data["time_s"][1]
    with pytest.raises(ValueError, match="strictly increasing"):
        sensor_metrics(data)


def test_unavailable_telemetry_is_not_reported_as_zero():
    assert all(value == "" for value in sensor_metrics({}).values())
    assert sensor_metrics({"body_ay_mps2": np.array([np.nan])})["peak_measured_ay_mps2"] == ""


def test_fixed_telemetry_is_aligned_with_post_step_state(tmp_path):
    columns = ("time_s", "vx_carsim_kmh", "vy_carsim_kmh", "X_carsim_m", "Y_carsim_m", "ey_m",
               "psi_carsim_deg", "epsi_deg", "yaw_rate_carsim_degps", "front_steer_command_deg",
               "qp_converged", "np", "q_y_applied", "q_psi_applied", "r_delta_applied",
               "front_left_wheel_steer_deg", "front_right_wheel_steer_deg", "body_ay_mps2")
    path = tmp_path / "simulation.csv"
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        for i in range(3):
            row = dict.fromkeys(columns, 0)
            row.update(time_s=0.04 * i, front_steer_command_deg=1 + 2 * i,
                       front_left_wheel_steer_deg=2 * i, front_right_wheel_steer_deg=2 * i,
                       body_ay_mps2=i + 10)
            writer.writerow(row)
    data = control_trace(path)
    assert data["body_ay_mps2"].tolist() == [11, 12]
    assert data["time_s"].tolist() == [0.04, 0.08]
    assert sensor_metrics(data)["steer_tracking_rmse_deg"] == pytest.approx(1.0)
