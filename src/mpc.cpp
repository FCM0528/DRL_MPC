#include "mpc.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>

LateralMpc::LateralMpc(const VehicleParameters& v,
                       const MpcParameters& p,
                       const DoubleLaneChange& r,
                       const TireTable& tires)
    : vehicle_(v), p_(p), ref_(r), tires_(tires) {
    if (p.np < 2 || p.np > 150 || p.nc < 1 || p.nc > p.np ||
        p.dt <= 0.0 || p.max_steer <= 0.0 || p.max_steer_rate <= 0.0 ||
        p.admm_rho <= 0.0 || p.admm_max_iterations <= 0 ||
        p.admm_primal_tol <= 0.0 || p.admm_dual_tol <= 0.0)
        throw std::invalid_argument("Invalid MPC settings (require 1 <= Nc <= Np <= 150)");
    const auto [front_load, rear_load] = staticWheelLoads(v);
    front_stiffness_ = 2.0 * v.road_grip_scale * tires.corneringStiffness(front_load);
    rear_stiffness_ = 2.0 * v.road_grip_scale * tires.corneringStiffness(rear_load);
    max_lateral_accel_ = 2.0 * v.road_grip_scale *
        (tires.peakLateralForce(front_load) + tires.peakLateralForce(rear_load)) / v.mass;
    operating_point_.vx = v.vx;
    model_ = discretizePredictionModel(v, tires, operating_point_, 0.0, 0.0, p.dt);
    precompute();
}

//
void LateralMpc::precompute() {
    const int np=p_.np, nc=p_.nc;
    sensitivity_.resize(np);
    std::vector<State6> current(nc, State6{});
    for (int stage=0; stage<np; ++stage) {
        std::vector<State6> next(nc, State6{});
        for (int j=0;j<nc;++j)
            next[j]=matrixTimesState(model_.a,current[j]);
        const int active=std::min(stage,nc-1);
        for (int d=0;d<6;++d) next[active][d]+=model_.b[d];
        current=std::move(next);
        sensitivity_[stage]=current;
    }
    if (last_solution_.size()!=static_cast<std::size_t>(nc))
        last_solution_.assign(nc,0.0);
    low_.assign(2*nc,0.0);
    high_.assign(2*nc,0.0);
}
//
std::vector<std::vector<double>> LateralMpc::choleskyFactor(
    const std::vector<std::vector<double>>& a) {
    const int n = static_cast<int>(a.size());
    std::vector<std::vector<double>> l(n, std::vector<double>(n, 0.0));
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j <= i; ++j) {
            double v = a[i][j];
            for (int k = 0; k < j; ++k) v -= l[i][k] * l[j][k];
            if (i == j) {
                if (!(v > 0.0)) throw std::runtime_error("QP matrix not positive definite");
                l[i][j] = std::sqrt(v);
            } else {
                l[i][j] = v / l[j][j];
            }
        }
    }
    return l;
}

//
std::vector<double> LateralMpc::solveCholesky(const std::vector<double>& rhs) const {
    const int n = p_.nc;
    std::vector<double> y(n, 0.0), x(n, 0.0);
    for (int i = 0; i < n; ++i) {
        double s = rhs[i];
        for (int j = 0; j < i; ++j) s -= cholesky_[i][j] * y[j];
        y[i] = s / cholesky_[i][i];
    }
    for (int i = n - 1; i >= 0; --i) {
        double s = y[i];
        for (int j = i + 1; j < n; ++j) s -= cholesky_[j][i] * x[j];
        x[i] = s / cholesky_[i][i];
    }
    return x;
}

//
void LateralMpc::buildBounds(double prev) {
    const int nc = p_.nc;
    const double step = p_.max_steer_rate * p_.dt;
    for (int i = 0; i < nc; ++i) {
        low_[i] = -p_.max_steer;
        high_[i] = p_.max_steer;
        low_[nc + i] = -step + (i == 0 ? prev : 0.0);
        high_[nc + i] = step + (i == 0 ? prev : 0.0);
    }
}

//
std::vector<double> LateralMpc::constraintProduct(const std::vector<double>& u) const {
    const int nc = p_.nc;
    std::vector<double> v(2 * nc, 0.0);
    for (int i = 0; i < nc; ++i) {
        v[i] = u[i];
        v[nc + i] = u[i] - (i == 0 ? 0.0 : u[i - 1]);
    }
    return v;
}

//
std::vector<double> LateralMpc::constraintTranspose(const std::vector<double>& v) const {
    const int nc = p_.nc;
    std::vector<double> u(nc, 0.0);
    for (int i = 0; i < nc; ++i)
        u[i] = v[i] + v[nc + i] - (i + 1 < nc ? v[nc + i + 1] : 0.0);
    return u;
}

//
double LateralMpc::predictedSteeringAt(int prediction_step) const {
    if (prediction_step < 0 || prediction_step >= p_.np)
        throw std::out_of_range("Prediction step is outside [0,Np)");
    return last_solution_[std::min(prediction_step, p_.nc - 1)];
}

MpcResult LateralMpc::gripLimitedControl(const VehicleState& plant,
                                        double previous_steer) const {
    const double speed = std::max(1.0, std::abs(plant.vx));
    const double lookahead = std::clamp(0.8 * speed, 10.0, 28.0);
    const auto target = ref_.at(plant.x + lookahead);
    const double dx = lookahead;
    const double dy = target.y - plant.y;
    const double distance = std::max(1.0, std::hypot(dx, dy));
    const double bearing_error = angleDifference(std::atan2(dy, dx), plant.psi);
    const double max_curvature = max_lateral_accel_ / (speed * speed);
    const double desired_curvature = std::clamp(
        2.0 * std::sin(bearing_error) / distance, -max_curvature, max_curvature);
    const double desired_yaw_rate = speed * desired_curvature;
    const double wheelbase = vehicle_.a + vehicle_.b;
    const double understeer = vehicle_.mass * speed * speed / wheelbase *
        (vehicle_.b / front_stiffness_ - vehicle_.a / rear_stiffness_);
    const double sideslip = std::atan2(plant.vy, speed);
    const double desired_steer = (wheelbase + understeer) * desired_curvature
        + 0.15 * (desired_yaw_rate - plant.r) - 0.35 * sideslip;
    const double max_step = p_.dt * p_.max_steer_rate;
    MpcResult result;
    result.steer = std::clamp(desired_steer,
        std::max(-p_.max_steer, previous_steer - max_step),
        std::min(p_.max_steer, previous_steer + max_step));
    result.grip_limited = true;
    return result;
}

MpcResult LateralMpc::enterGripLimitedMode(const VehicleState& plant,
                                          double previous_steer) {
    grip_mode_ = true;
    return gripLimitedControl(plant, previous_steer);
}

//
MpcResult LateralMpc::control(const VehicleState& plant, double prev) {
    if (!std::isfinite(plant.vx) || !std::isfinite(plant.ax))
        throw std::runtime_error("MPC requires finite vehicle speed and longitudinal acceleration");
    const auto current_ref = ref_.at(plant.x);
    const double position_error = plant.y - current_ref.y;
    const double heading_error = angleDifference(plant.psi, current_ref.psi);
    const double current_demand = plant.vx * plant.vx * std::abs(current_ref.curvature);
    if (current_demand > 1.10 * max_lateral_accel_ ||
        (std::abs(position_error) > 0.3 && std::abs(plant.vy) > 0.5))
        grip_mode_ = true;
    if (grip_mode_ && current_demand < 0.7 * max_lateral_accel_ &&
        std::abs(position_error) < 0.08 && std::abs(heading_error) < 0.05 &&
        std::abs(plant.vy) < 0.3)
        grip_mode_ = false;
    if (grip_mode_ || plant.vx < 1.0)
        return gripLimitedControl(plant, prev);
    const double ax = std::clamp(plant.ax, -6.0, 6.0);
    if (std::abs(plant.vx - operating_point_.vx) > 1.0e-3 ||
        std::abs(plant.vy - operating_point_.vy) > 1.0e-3 ||
        std::abs(plant.r - operating_point_.r) > 1.0e-4 ||
        std::abs(plant.psi - operating_point_.psi) > 1.0e-4 ||
        std::abs(ax - operating_ax_) > 1.0e-3 ||
        std::abs(prev - operating_steer_) > 1.0e-4) {
        vehicle_.vx = plant.vx;
        operating_point_ = plant;
        operating_steer_ = prev;
        operating_ax_ = ax;
        model_ = discretizePredictionModel(vehicle_, tires_, operating_point_,
                                           operating_steer_, ax, p_.dt);
        try { precompute(); }
        catch (const std::exception&) {
            grip_mode_ = true;
            return gripLimitedControl(plant, prev);
        }
    }
    const int np=p_.np, nc=p_.nc;
    const State6 initial{plant.vy,plant.vx,plant.psi,plant.r,plant.y,plant.x};
    State6 free_response=initial;
    std::vector<double> gradient(nc,0.0);
    hessian_.assign(nc,std::vector<double>(nc,0.0));
    const double q[4]={p_.q_y,p_.q_psi,p_.q_vy,p_.q_r};
    const double wheelbase=vehicle_.a+vehicle_.b;
    const double understeer=vehicle_.mass*plant.vx*plant.vx/wheelbase *
        (vehicle_.b/front_stiffness_-vehicle_.a/rear_stiffness_);
    for (int stage=0;stage<np;++stage) {
        free_response=matrixTimesState(model_.a,free_response);
        for (int d=0;d<6;++d) free_response[d]+=model_.c[d];
        const auto path=ref_.at(free_response[5]);
        const double curve_limit=max_lateral_accel_/
            std::max(1.0,free_response[1]*free_response[1]);
        const double curvature=std::clamp(path.curvature,-curve_limit,curve_limit);
        const double heading_slope=path.d2ydx2/(1.0+path.dydx*path.dydx);
        const double curvature_slope=(ref_.at(free_response[5]+0.2).curvature-
            ref_.at(free_response[5]-0.2).curvature)/0.4;
        const double error[4]={free_response[4]-path.y,
            angleDifference(free_response[2],path.psi),
            free_response[0],free_response[3]-free_response[1]*curvature};
        const double terminal=stage==np-1 ? p_.terminal_scale : 1.0;
        for (int j=0;j<nc;++j) {
            const auto& sj=sensitivity_[stage][j];
            const double ej[4]={sj[4]-path.dydx*sj[5],
                sj[2]-heading_slope*sj[5],sj[0],
                sj[3]-curvature*sj[1]-free_response[1]*curvature_slope*sj[5]};
            for (int d=0;d<4;++d)
                gradient[j]+=2.0*terminal*q[d]*ej[d]*error[d];
            for (int k=0;k<nc;++k) {
                const auto& sk=sensitivity_[stage][k];
                const double ek[4]={sk[4]-path.dydx*sk[5],
                    sk[2]-heading_slope*sk[5],sk[0],
                    sk[3]-curvature*sk[1]-free_response[1]*curvature_slope*sk[5]};
                for (int d=0;d<4;++d)
                    hessian_[j][k]+=2.0*terminal*q[d]*ej[d]*ek[d];
            }
        }
        const int active=std::min(stage,nc-1);
    const double ff=(wheelbase + dynamicPredictionWeight(plant.vx) *
                    understeer)*curvature;
        hessian_[active][active]+=2.0*p_.r_steer;
        gradient[active]-=2.0*p_.r_steer*ff;
    }
    for (int i=0;i<nc;++i) {
        hessian_[i][i]+=2.0*p_.r_delta;
        if (i>0) {
            hessian_[i-1][i-1]+=2.0*p_.r_delta;
            hessian_[i][i-1]-=2.0*p_.r_delta;
            hessian_[i-1][i]-=2.0*p_.r_delta;
        }
    }
    auto lhs=hessian_;
    for (int i=0;i<nc;++i) {
        lhs[i][i]+=p_.admm_rho*(1.0+(i==nc-1 ? 1.0 : 2.0));
        if (i>0) {
            lhs[i][i-1]-=p_.admm_rho;
            lhs[i-1][i]-=p_.admm_rho;
        }
    }
    cholesky_=choleskyFactor(lhs);
    gradient[0]-=2.0*p_.r_delta*prev;    buildBounds(prev);

    // Warm-start with shifted previous move sequence and feasible projection.
    std::vector<double> u(nc, 0.0);
    double last = prev;
    const double max_step = p_.dt * p_.max_steer_rate;
    for (int i = 0; i < nc; ++i) {
        const double candidate = last_solution_[std::min(i + 1, nc - 1)];
        u[i] = std::clamp(candidate,
                 std::max(-p_.max_steer, last - max_step),
                 std::min(p_.max_steer, last + max_step));
        last = u[i];
    }
    auto z = constraintProduct(u);
    std::vector<double> dual(2 * nc, 0.0);

	// ADMM iterations
    MpcResult status;
    for (int iteration = 0; iteration < p_.admm_max_iterations; ++iteration) {
        std::vector<double> z_minus_dual(2 * nc);
        for (int i = 0; i < 2 * nc; ++i) z_minus_dual[i] = z[i] - dual[i];
        auto rhs_correction = constraintTranspose(z_minus_dual);
        std::vector<double> rhs(nc);
        for (int i = 0; i < nc; ++i)
            rhs[i] = -gradient[i] + p_.admm_rho * rhs_correction[i];
        u = solveCholesky(rhs);

        auto au = constraintProduct(u);
        auto z_old = z;
        double primal_inf = 0.0;
        for (int i = 0; i < 2 * nc; ++i) {
            z[i] = std::clamp(au[i] + dual[i], low_[i], high_[i]);
            dual[i] += au[i] - z[i];
            primal_inf = std::max(primal_inf, std::abs(au[i] - z[i]));
        }
        for (int i = 0; i < 2 * nc; ++i) z_old[i] = z[i] - z_old[i];
        auto dual_change = constraintTranspose(z_old);
        double dual_inf = 0.0;
        for (double v : dual_change)
            dual_inf = std::max(dual_inf, std::abs(p_.admm_rho * v));
        status.iterations = iteration + 1;
        status.primal_residual = primal_inf;
        status.dual_residual = dual_inf;
        if (primal_inf < p_.admm_primal_tol && dual_inf < p_.admm_dual_tol) {
            status.converged = true;
            break;
        }
    }

    // Last-resort projection protects actuator limits even if solver did not converge.
    // Nonconvergence must be checked using status.converged in experimental results.
    last = prev;
    for (int i = 0; i < nc; ++i) {
        u[i] = std::clamp(u[i],
             std::max(-p_.max_steer, last - max_step),
             std::min(p_.max_steer, last + max_step));
        last = u[i];
    }
    last_solution_ = u;
    status.steer = u[0];
    if (!status.converged) {
        grip_mode_ = true;
        return gripLimitedControl(plant, prev);
    }
    return status;
}
