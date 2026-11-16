"""Use CarSim measurements to audit units, rotating-frame kinematics and tire forces."""
import numpy as np


def sensor_metrics(data, vehicle=None):
    # Empty cells distinguish unavailable measurements from measured zeros.
    result = dict.fromkeys(("peak_measured_ay_mps2", "peak_measured_beta_deg",
                           "steer_tracking_rmse_deg", "beta_consistency_rmse_deg",
                           "driver_target_curvature_rmse_inv_m", "ax_transport_rmse_mps2",
                           "ay_transport_rmse_mps2", "lateral_force_balance_rmse_mps2",
                           "yaw_force_balance_rmse_radps2"), "")

    def available(*keys):
        return all(key in data for key in keys)

    def assign(name, values, peak=False, scale=1.0):
        finite = np.asarray(values)[np.isfinite(values)]
        if finite.size:
            value = np.max(np.abs(finite)) if peak else np.sqrt(np.mean(finite ** 2))
            result[name] = float(value * scale)

    if available("body_ay_mps2"):
        assign("peak_measured_ay_mps2", data["body_ay_mps2"], peak=True)
    if available("measured_beta_rad"):
        assign("peak_measured_beta_deg", data["measured_beta_rad"], peak=True, scale=180 / np.pi)
    if available("actual_steer_rad", "steer_rad"):
        assign("steer_tracking_rmse_deg", data["actual_steer_rad"] - data["steer_rad"], scale=180 / np.pi)
    if available("measured_beta_rad", "vx_mps", "vy_mps"):
        difference = data["measured_beta_rad"] - np.arctan2(data["vy_mps"], data["vx_mps"])
        assign("beta_consistency_rmse_deg", np.arctan2(np.sin(difference), np.cos(difference)), scale=180 / np.pi)
    if available("driver_curvature_inv_m", "reference_curvature_inv_m"):
        assign("driver_target_curvature_rmse_inv_m",
               data["driver_curvature_inv_m"] - data["reference_curvature_inv_m"])

    # Compare interval velocity changes with trapezoidal averages of instantaneous
    # acceleration. This includes the rotating-frame terms and the final partial step.
    def transport(velocity, acceleration, rotation, metric):
        if not available("time_s", velocity, acceleration):
            return
        dt = np.diff(data["time_s"])
        if np.any(~np.isfinite(dt)) or np.any(dt <= 0):
            raise ValueError("Sensor diagnostics require finite, strictly increasing timestamps")
        predicted = data[acceleration] + rotation
        residual = np.diff(data[velocity]) / dt - 0.5 * (predicted[1:] + predicted[:-1])
        assign(metric, residual)

    if available("yaw_rate_radps", "vx_mps", "vy_mps"):
        transport("vx_mps", "body_ax_mps2", data["yaw_rate_radps"] * data["vy_mps"], "ax_transport_rmse_mps2")
        transport("vy_mps", "body_ay_mps2", -data["yaw_rate_radps"] * data["vx_mps"], "ay_transport_rmse_mps2")
    if vehicle and available("front_lateral_force_n", "rear_lateral_force_n", "actual_steer_rad"):
        front = data["front_lateral_force_n"] * np.cos(data["actual_steer_rad"])
        rear = data["rear_lateral_force_n"]
        if available("body_ay_mps2"):
            assign("lateral_force_balance_rmse_mps2", (front + rear) / vehicle["mass_kg"] - data["body_ay_mps2"])
        moment_rate = (vehicle["a_m"] * front - vehicle["b_m"] * rear) / vehicle["iz_kgm2"]
        if available("time_s", "yaw_rate_radps"):
            dt = np.diff(data["time_s"])
            if np.any(~np.isfinite(dt)) or np.any(dt <= 0):
                raise ValueError("Sensor diagnostics require finite, strictly increasing timestamps")
            assign("yaw_force_balance_rmse_radps2", np.diff(data["yaw_rate_radps"]) / dt -
                   0.5 * (moment_rate[1:] + moment_rate[:-1]))
    return result
