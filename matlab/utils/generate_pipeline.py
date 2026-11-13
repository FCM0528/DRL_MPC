"""Extract inspectable stage methods from the original MPC, preserving arithmetic.

The original snapshots stay byte-identical. The generated PipelineMpc exposes
the same computations as seven ordered calls for the Simulink presentation.
Run the independent original-vs-pipeline replay after regeneration.
"""
from pathlib import Path
root=Path(__file__).resolve().parents[1]
native=root/'controllers'/'native'
h=(native/'mpc.h').read_text(encoding='utf-8')
cpp=(native/'mpc.cpp').read_text(encoding='utf-8')
cls=h[h.index('class LateralMpc'):].replace('LateralMpc','PipelineMpc')
public='''
    void stageOperatingPoint(const VehicleState&, double, double);
    void stageLinearization();
    void stageReference();
    void stageCost();
    void stageConstraints();
    void stageQp();
    MpcResult stageSteering();
    double stageMetric() const { return double(nominal_states_.size()); }
'''
members='''
    VehicleState stage_plant_;
    double stage_previous_ = 0.0, stage_increment_ = 0.0;
    std::vector<ReferencePoint> stage_path_, stage_ahead_, stage_behind_;
    std::vector<double> stage_curvature_, stage_gradient_, stage_moves_;
    MpcResult stage_result_;
'''
cls=cls.replace('private:',public+'\nprivate:'+members,1)
(native/'mpc_pipeline.h').write_text('#pragma once\n#include "mpc.h"\n'+cls,encoding='utf-8')
start=cpp.index('LateralMpc::LateralMpc')
body=cpp[start:].replace('LateralMpc','PipelineMpc')
solve=cpp[cpp.index('MpcResult LateralMpc::solve'):]
cost=solve[solve.index('    const int np ='):solve.index('    auto lhs =')]
cost=cost.replace('std::vector<double> gradient(nc, 0.0);','stage_gradient_.assign(nc, 0.0);\n    auto& gradient=stage_gradient_;')
cost=cost.replace('ref_.at(nominal[5])','stage_path_[stage]').replace('ref_.at(nominal[5] + 0.2)','stage_ahead_[stage]').replace('ref_.at(nominal[5] - 0.2)','stage_behind_[stage]')
cost=cost.replace('ref_.controlCurvature(nominal[5], p_.reference_curvature_sigma_m)','stage_curvature_[stage]')
factor=solve[solve.index('    auto lhs ='):solve.index('    // Last-resort projection')]
factor=factor.replace('    buildBounds(prev);','')
factor=factor.replace('    // ADMM iterations','    // ADMM iterations')
factor+='    stage_moves_=std::move(u); stage_result_=status;\n'
projection=solve[solve.index('    // Last-resort projection'):solve.rindex('}')]
stages='''
void PipelineMpc::stageOperatingPoint(const VehicleState& s,double prev,double increment) {
    if (!std::isfinite(prev) || std::abs(prev)>p_.max_steer || !std::isfinite(s.vx) || !std::isfinite(s.ax))
        throw std::invalid_argument("Invalid staged MPC operating point");
    stage_plant_=s;stage_previous_=prev;stage_increment_=increment;
}
void PipelineMpc::stageLinearization() {
    current_steer_rate_=effectiveSteerRate(p_,stage_plant_.vx);
    precompute(stage_plant_,stage_previous_,std::clamp(stage_plant_.ax,-6.,6.));
}
void PipelineMpc::stageReference() {
    stage_path_.resize(p_.np);stage_ahead_.resize(p_.np);stage_behind_.resize(p_.np);stage_curvature_.resize(p_.np);
    for(int k=0;k<p_.np;++k) {
        const double x=nominal_states_[k+1][5];
        stage_path_[k]=ref_.at(x);stage_ahead_[k]=ref_.at(x+.2);stage_behind_[k]=ref_.at(x-.2);
        stage_curvature_[k]=ref_.controlCurvature(x,p_.reference_curvature_sigma_m);
    }
}
void PipelineMpc::stageCost() {
    const double prev=stage_previous_,previous_increment=stage_increment_;
'''+cost+'''
}
void PipelineMpc::stageConstraints() { buildBounds(stage_previous_); }
void PipelineMpc::stageQp() {
    const int nc=p_.nc; auto& gradient=stage_gradient_;
'''+factor+'''
}
MpcResult PipelineMpc::stageSteering() {
    const int nc=p_.nc;const double prev=stage_previous_;auto& u=stage_moves_;auto& status=stage_result_;
'''+projection+'''
}
'''
(native/'mpc_pipeline.cpp').write_text('#include "mpc_pipeline.h"\n#include <algorithm>\n#include <cmath>\n#include <stdexcept>\n'+body+'\n'+stages,encoding='utf-8')
print('Generated PipelineMpc with seven executable stages')
