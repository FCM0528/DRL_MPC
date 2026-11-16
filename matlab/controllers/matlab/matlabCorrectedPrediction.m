function [nominal,sensitivity,increments,valid]=matlabCorrectedPrediction(op,lastSolution,hasSolution,p)
%#codegen
if ~p.correctionEnabled
    [nominal,sensitivity,increments,valid]=matlabNominalPrediction(op,lastSolution,hasSolution,p);return;
end
nominal=zeros(8,41);sensitivity=zeros(8,12,40);increments=zeros(12,1);steering=zeros(12,1);valid=false;
if any(~isfinite(op([1:9 12:16])))||op(2)<1||abs(op(12))>p.maxSteer,return;end
last=op(12);
for j=1:12
    candidate=last;if hasSolution,candidate=lastSolution(min(j+1,12));end
    steering(j)=min(min(p.maxSteer,last+p.dt*p.maxRate),max(max(-p.maxSteer,last-p.dt*p.maxRate),candidate));
    increments(j)=steering(j)-last;last=steering(j);
end
nominal(:,1)=op(1:8);current=zeros(8,12);
[front,rear]=matlabPhysicalForces(op(1:8),op(12),op(9),p);bias=zeros(2,1);
if isfinite(op(10)),bias(1)=op(10)-front;end
if isfinite(op(11)),bias(2)=op(11)-rear;end
for stage=1:40
    time=(stage-1)*p.dt;active=min(stage,12);ax=op(9)*exp(-time/.5);
    if p.speedPlanActive
        goal=matlabPlanValue(nominal(6,stage),p.speedPlan);
        planned=min(p.speedAcceleration,max(-p.speedDeceleration,(goal-nominal(2,stage))/.5));
        ax=op(9)*exp(-time/.2)+planned*(1-exp(-time/.2));
    end
    d=op(14:16)*exp(-time/p.disturbanceTime);
    [a,b,c,ok]=matlabCorrectedVehicleModel(nominal(:,stage),steering(active),ax,bias*exp(-time/p.forceBiasTime),d,p);
    if ~ok,return;end
    nominal(:,stage+1)=a*nominal(:,stage)+b*steering(active)+c;
    next=a*current;for j=1:active,next(:,j)=next(:,j)+b;end
    current=next;sensitivity(:,:,stage)=current;
end
valid=true;
end
