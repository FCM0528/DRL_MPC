// Independently linked against ../src, never against the migration snapshots.
#include "config.h"
#ifdef PARITY_PIPELINE
#include "mpc_pipeline.h"
#endif
#include <fstream>
#include <sstream>
#include <iostream>
#include <iomanip>
#include <memory>
#include <algorithm>
#include <cmath>
int main(int argc,char** argv){try{
 if(argc!=5)throw std::runtime_error("ini case input.csv output.csv");
 auto cfg=loadConfig(argv[1]);auto model=readCarSimMpcModel(argv[2]);
 auto ref=DoubleLaneChange::fromCarSimParsfile(std::filesystem::path(argv[2])/"Run_all.par");
 TireTable tires(model.tire_csv);cfg.mpc.max_steer=readCarSimSteeringModel(argv[2],cfg.carsim).max_front_angle;
#ifdef PARITY_PIPELINE
 using Control=PipelineMpc;
#else
 using Control=LateralMpc;
#endif
 std::unique_ptr<Control> ctrl;auto tuning=neutralRlAction(cfg.rl),previous_tuning=tuning;
 std::ifstream in(argv[3]);std::ofstream out(argv[4]);out<<std::setprecision(17);
 double prev=0,inc=0,oldvx=0;bool first=true;std::string line;
 while(std::getline(in,line)){
  std::replace(line.begin(),line.end(),',',' ');std::istringstream row(line);double s[20];for(auto& x:s)if(!(row>>x))throw std::runtime_error("Invalid state row");
  previous_tuning=tuning;double np,qy,qpsi,rd;
  if(row>>np>>qy>>qpsi>>rd){tuning.np=int(np);tuning.q_y=qy;tuning.q_psi=qpsi;tuning.r_delta=rd;}
  if(!ctrl||tuning.np!=previous_tuning.np||tuning.q_y!=previous_tuning.q_y||tuning.q_psi!=previous_tuning.q_psi||tuning.r_delta!=previous_tuning.r_delta){
    auto settings=applyRlAction(cfg.mpc,tuning,cfg.rl);auto updated=std::make_unique<Control>(model.vehicle,settings,ref,tires);updated->setRoadGripProfile([&](double x){return model.road_grip.at(x);});
    if(ctrl)updated->adoptPreviousSolution(*ctrl);ctrl=std::move(updated);
  }
  VehicleState v;v.x=s[0];v.y=s[1];v.vx=s[2];v.vy=s[3];v.psi=s[4];v.r=s[5];v.ax=first?0:(s[2]-oldvx)/.04;first=false;oldvx=s[2];
  v.alpha_front=-.5*(s[12]+s[13]);v.alpha_rear=-.5*(s[14]+s[15]);v.fy_front=s[16]+s[17];v.fy_rear=s[18]+s[19];
#ifdef PARITY_PIPELINE
  ctrl->stageOperatingPoint(v,prev,inc);ctrl->stageLinearization();ctrl->stageReference();ctrl->stageCost();ctrl->stageConstraints();ctrl->stageQp();auto r=ctrl->stageSteering();
#else
  auto r=ctrl->control(v,prev,inc);
#endif
  out<<r.steer<<','<<r.converged<<','<<r.iterations<<','<<r.primal_residual<<','<<r.dual_residual;
  auto vehicle=model.vehicle;vehicle.road_grip_scale=model.road_grip.at(v.x);auto loads=staticWheelLoads(vehicle);double mu=vehicle.road_grip_scale;
  double fb=v.fy_front-2*mu*tires.boundedLateralForce(v.alpha_front/mu,loads.first),rb=v.fy_rear-2*mu*tires.boundedLateralForce(v.alpha_rear/mu,loads.second);
  auto dm=discretizeErrorModel(vehicle,tires,v,r.steer,std::clamp(v.ax,-6.,6.),cfg.mpc.dt,cfg.mpc.tire_slope_floor,cfg.mpc.front_relaxation_m,cfg.mpc.rear_relaxation_m,fb,rb);
  PredictionState z{v.vy,v.vx,v.psi,v.r,v.y,v.x,v.alpha_front,v.alpha_rear};auto next=matrixTimesState(dm.a,z);
  for(int k=0;k<8;++k)out<<','<<(next[k]+dm.b[k]*r.steer+dm.c[k]);
  auto obs=makeRoadPreviewObservation(v,ref,[&](double x){return model.road_grip.at(x);},150,cfg.mpc,cfg.rl,tuning.np,prev);for(double x:obs)out<<','<<x;
  auto p=ref.at(v.x);for(double x:{p.y,p.psi,p.curvature,p.dydx,p.d2ydx2,p.x_min,p.x_max})out<<','<<x;
  out<<','<<scalarizedRlReward(v.y-p.y,angleDifference(v.psi,p.psi),r.steer,tuning.np,cfg.mpc,cfg.rl,prev,previous_tuning.np,tuning,previous_tuning,v.vy,v.r-v.vx*p.curvature,std::atan2(v.vy,v.vx),lateralErrorRate(v,p))<<'\n';
  inc=r.steer-prev;prev=r.steer;
 }
 return 0;
}catch(const std::exception& e){std::cerr<<e.what();return 1;}}
