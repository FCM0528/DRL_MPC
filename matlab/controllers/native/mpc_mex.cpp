// MATLAB adapter only: no CarSimPlant instance and no solver integration.
#include "mex.h"
#include "config.h"
#include "mpc_pipeline.h"
#include <memory>
#include <cmath>
#include <algorithm>
namespace {
struct Session {
    AppConfig cfg;
    CarSimMpcModel model;
    DoubleLaneChange ref;
    TireTable tires;
    std::unique_ptr<LateralMpc> controller;
    std::unique_ptr<PipelineMpc> pipeline;
    MpcTuningAction pipeline_tuning;
    double applied_steer=0,applied_increment=0,last_vx=std::numeric_limits<double>::quiet_NaN();
    bool stage_failed=false;
    MpcTuningAction tuning;
    Session(const char* ini,const char* dir): cfg(loadConfig(ini)),
      model(readCarSimMpcModel(dir)),
      ref(DoubleLaneChange::fromCarSimParsfile(std::filesystem::path(dir)/"Run_all.par")),
      tires(model.tire_csv) {
        cfg.distance_m=150;
        cfg.vehicle=model.vehicle;
        cfg.mpc.max_steer=readCarSimSteeringModel(dir,cfg.carsim).max_front_angle;
        tuning=neutralRlAction(cfg.rl);
    }
};
std::unique_ptr<Session> session;
std::string str(const mxArray* a){char* c=mxArrayToString(a); if(!c)throw std::runtime_error("Expected string");std::string s(c);mxFree(c);return s;}
const double* vec(const mxArray* a,size_t n){if(!mxIsDouble(a)||mxIsComplex(a)||mxGetNumberOfElements(a)!=n)throw std::runtime_error("Invalid vector dimensions/type");return mxGetPr(a);}
VehicleState state(const mxArray* a){const auto v=vec(a,20); VehicleState s;
 s.x=v[0];s.y=v[1];s.vx=v[2];s.vy=v[3];s.psi=v[4];s.r=v[5];s.ax=v[6];
 s.steer_front_left=v[8];s.steer_front_right=v[9];
 s.alpha_front=-0.5*(v[12]+v[13]);s.alpha_rear=-0.5*(v[14]+v[15]);
 s.fy_front=v[16]+v[17];s.fy_rear=v[18]+v[19];return s;}
mxArray* array(const std::vector<double>& v){auto a=mxCreateDoubleMatrix(v.size(),1,mxREAL);std::copy(v.begin(),v.end(),mxGetPr(a));return a;}
void cleanup(){session.reset();}
}
MEXFUNCTION_LINKAGE void mexFunction(int nlhs,mxArray* plhs[],int nrhs,const mxArray* prhs[]) {
 try {
    if(nrhs<1)throw std::runtime_error("Command required");
    auto cmd=str(prhs[0]);
    if(cmd=="init") {if(nrhs!=3)throw std::runtime_error("init(ini,case_dir)");session=std::make_unique<Session>(str(prhs[1]).c_str(),str(prhs[2]).c_str());mexAtExit(cleanup);return;}
    if(cmd=="close"){cleanup();return;}
    if(!session)throw std::runtime_error("Call mpc_mex('init',ini,case_dir) first");
    auto& q=*session;
    if(cmd=="reset"){q.controller.reset();q.pipeline.reset();q.tuning=neutralRlAction(q.cfg.rl);q.applied_steer=0;q.applied_increment=0;q.last_vx=std::numeric_limits<double>::quiet_NaN();return;}
    if(cmd=="info"){plhs[0]=array({q.cfg.mpc.max_steer,q.model.vehicle.mass,q.model.vehicle.iz,q.model.vehicle.a,q.model.vehicle.b,q.model.road_grip.at(0)});return;}
    if(cmd=="reference") {auto p=q.ref.at(mxGetScalar(prhs[1]));plhs[0]=array({p.y,p.psi,p.curvature,p.dydx,p.d2ydx2,p.x_min,p.x_max});return;}
    if(cmd=="map") {auto a=vec(prhs[1],4); auto t=advanceRlTuning({a[0],a[1],a[2],a[3]},q.tuning,q.cfg.rl);plhs[0]=array({double(t.np),t.q_y,t.q_psi,t.r_delta});return;}
    if(cmd.rfind("stage_",0)==0) {
      MpcResult result;result.steer=q.applied_steer;
      try {
        if(cmd=="stage_operatingpoint") {
          q.stage_failed=false;auto s=state(prhs[1]);auto t=vec(prhs[2],4);
          s.ax=std::isfinite(q.last_vx)?(s.vx-q.last_vx)/q.cfg.mpc.dt:0;q.last_vx=s.vx;
          if(!q.pipeline||q.pipeline_tuning.np!=int(t[0])||q.pipeline_tuning.q_y!=t[1]||q.pipeline_tuning.q_psi!=t[2]||q.pipeline_tuning.r_delta!=t[3]) {
            auto tuning=q.tuning;tuning.np=int(t[0]);tuning.q_y=t[1];tuning.q_psi=t[2];tuning.r_delta=t[3];
            auto p=applyRlAction(q.cfg.mpc,tuning,q.cfg.rl);auto updated=std::make_unique<PipelineMpc>(q.model.vehicle,p,q.ref,q.tires);
            updated->setRoadGripProfile([&](double x){return q.model.road_grip.at(x);});
            if(q.pipeline)updated->adoptPreviousSolution(*q.pipeline);q.pipeline=std::move(updated);q.pipeline_tuning=tuning;
          }
          q.tuning=q.pipeline_tuning;q.pipeline->stageOperatingPoint(s,q.applied_steer,q.applied_increment);
        } else if(!q.stage_failed) {
          if(cmd=="stage_linearization")q.pipeline->stageLinearization();
          else if(cmd=="stage_reference")q.pipeline->stageReference();
          else if(cmd=="stage_cost")q.pipeline->stageCost();
          else if(cmd=="stage_constraints")q.pipeline->stageConstraints();
          else if(cmd=="stage_qp")q.pipeline->stageQp();
          else if(cmd=="stage_steering")result=q.pipeline->stageSteering();
          else throw std::invalid_argument("Unknown pipeline stage");
        }
      } catch(const std::exception&) {q.stage_failed=true;q.pipeline.reset();}
      if(cmd=="stage_steering") {
        q.applied_increment=result.steer-q.applied_steer;q.applied_steer=result.steer;
        plhs[0]=array({result.steer,double(result.converged),double(result.iterations),result.primal_residual,result.dual_residual,double(q.tuning.np),q.tuning.q_y,q.tuning.q_psi,q.tuning.r_delta});
      } else plhs[0]=mxCreateDoubleScalar(q.stage_failed?-1:1);
      return;
    }
    auto s=state(prhs[1]);
    if(cmd=="prediction") {
      const double delta=mxGetScalar(prhs[2]);auto vehicle=q.model.vehicle;vehicle.road_grip_scale=q.model.road_grip.at(s.x);
      auto loads=staticWheelLoads(vehicle);const double mu=vehicle.road_grip_scale;
      const double fb=s.fy_front-2*mu*q.tires.boundedLateralForce(s.alpha_front/mu,loads.first);
      const double rb=s.fy_rear-2*mu*q.tires.boundedLateralForce(s.alpha_rear/mu,loads.second);
      auto m=discretizeErrorModel(vehicle,q.tires,s,delta,std::clamp(s.ax,-6.,6.),q.cfg.mpc.dt,q.cfg.mpc.tire_slope_floor,q.cfg.mpc.front_relaxation_m,q.cfg.mpc.rear_relaxation_m,fb,rb);
      PredictionState z{s.vy,s.vx,s.psi,s.r,s.y,s.x,s.alpha_front,s.alpha_rear};auto n=matrixTimesState(m.a,z);
      for(int k=0;k<8;++k)n[k]+=m.b[k]*delta+m.c[k];plhs[0]=array(std::vector<double>(n.begin(),n.end()));return;
    }
    if(cmd=="observe") {int np=int(mxGetScalar(prhs[2]));double prev=mxGetScalar(prhs[3]);
      auto v=makeRoadPreviewObservation(s,q.ref,[&](double x){return q.model.road_grip.at(x);},150,q.cfg.mpc,q.cfg.rl,np,prev);plhs[0]=array(v);return;}
    if(cmd=="step") {auto t=vec(prhs[2],4); double prev=mxGetScalar(prhs[3]),inc=mxGetScalar(prhs[4]);
      if(!q.controller||q.tuning.np!=int(t[0])||q.tuning.q_y!=t[1]||q.tuning.q_psi!=t[2]||q.tuning.r_delta!=t[3]){
        auto tuning=q.tuning;tuning.np=int(t[0]);tuning.q_y=t[1];tuning.q_psi=t[2];tuning.r_delta=t[3];
        auto p=applyRlAction(q.cfg.mpc,tuning,q.cfg.rl);
        auto c=std::make_unique<LateralMpc>(q.model.vehicle,p,q.ref,q.tires);
        c->setRoadGripProfile([&](double x){return q.model.road_grip.at(x);});
        if(q.controller)c->adoptPreviousSolution(*q.controller);q.controller=std::move(c);q.tuning=tuning;
      }
      auto r=q.controller->control(s,prev,inc);
      plhs[0]=array({r.steer,double(r.converged),double(r.iterations),r.primal_residual,r.dual_residual,t[0],t[1],t[2],t[3]});return;}
    if(cmd=="reward") {double steer=mxGetScalar(prhs[2]),prev=mxGetScalar(prhs[3]);auto a=vec(prhs[4],4),b=vec(prhs[5],4);
      auto ta=q.tuning,tb=q.tuning;ta.np=int(a[0]);ta.q_y=a[1];ta.q_psi=a[2];ta.r_delta=a[3];tb.np=int(b[0]);tb.q_y=b[1];tb.q_psi=b[2];tb.r_delta=b[3];
      auto p=q.ref.at(s.x);double beta=std::atan2(s.vy,s.vx);
      plhs[0]=mxCreateDoubleScalar(scalarizedRlReward(s.y-p.y,angleDifference(s.psi,p.psi),steer,ta.np,q.cfg.mpc,q.cfg.rl,prev,tb.np,ta,tb,s.vy,s.r-s.vx*p.curvature,beta,lateralErrorRate(s,p)));return;}
    throw std::runtime_error("Unknown command");
 }catch(const std::exception& e){mexErrMsgIdAndTxt("DRLMPC:Native", "%s",e.what());}
}
