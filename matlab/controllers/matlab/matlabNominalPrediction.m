function [nominal,sensitivity,increments,valid]=matlabNominalPrediction(op,lastSolution,hasSolution,p)
%#codegen
% Shifted Nc warm start; affine LTV rollout and delta-steering sensitivities.
nominal=zeros(8,41);sensitivity=zeros(8,12,40);increments=zeros(12,1);steering=zeros(12,1);valid=false;
if any(~isfinite(op([1:9 12:13])))||op(2)<1||abs(op(12))>p.maxSteer,return;end
last=op(12);maxStep=p.dt*p.maxRate;
for j=1:12
    candidate=op(12);if hasSolution,candidate=lastSolution(min(j+1,12));end
    steering(j)=min(min(p.maxSteer,last+maxStep),max(max(-p.maxSteer,last-maxStep),candidate));
    increments(j)=steering(j)-last;last=steering(j);
end
nominal(:,1)=op(1:8);current=zeros(8,12);mu=matlabGrip(op(6),p);
frontBias=0;rearBias=0;
if p.frontLength>0&&isfinite(op(10)),frontBias=op(10)-2*mu*matlabTireForce(op(7)/mu,p.frontLoad,p);end
if p.rearLength>0&&isfinite(op(11)),rearBias=op(11)-2*mu*matlabTireForce(op(8)/mu,p.rearLoad,p);end
for stage=1:40
    active=min(stage,12);ax=op(9)*exp(-(stage-1)*p.dt/.5);fade=exp(-(stage-1)*p.dt/.08);
    [a,b,c,ok]=matlabVehicleModel(nominal(:,stage),steering(active),ax,frontBias*fade,rearBias*fade,p);
    if ~ok,return;end
    nominal(:,stage+1)=a*nominal(:,stage)+b*steering(active)+c;
    if any(~isfinite(nominal(:,stage+1))),return;end
    next=a*current;for j=1:active,next(:,j)=next(:,j)+b;end
    current=next;sensitivity(:,:,stage)=current;
end
valid=true;
end
