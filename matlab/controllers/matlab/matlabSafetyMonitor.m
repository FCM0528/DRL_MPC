function log=matlabSafetyMonitor(s,slack,p)
%#codegen
mu=matlabGrip(s(1),p);capacity=min(mu*9.80665,mu*p.baseLateralCapacity);
limit=p.safetyUtilization*sqrt(max(.1,capacity^2-s(7)^2));r=matlabReference(s(1),p);
utilization=1;if p.speedPlanActive,utilization=p.speedUtilization;end
cap=min(p.nominalSpeed,sqrt(utilization*capacity/max(abs(r(3)),1e-8)));
target=p.nominalSpeed;if p.speedPlanActive,target=matlabPlanValue(s(1),p.speedPlan);end
log=[s(8);limit;atan2(s(4),s(3));p.betaLimit;hypot(s(3),s(4))*3.6;target*3.6;cap*3.6;slack;max(0,hypot(s(3),s(4))^2*abs(r(3))/(utilization*capacity)-1)];
end
