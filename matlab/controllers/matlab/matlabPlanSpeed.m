function plan=matlabPlanSpeed(p)
% Curvature/grip ceiling with backward braking and forward acceleration passes.
x=(-10:1:200)';limit=zeros(size(x));
for k=1:numel(x)
    curve=0;
    for offset=0:.5:p.speedLookahead,r=matlabReference(x(k)+offset,p);curve=max(curve,abs(r(3)));end
    mu=matlabGrip(x(k),p);capacity=min(mu*9.80665,mu*p.baseLateralCapacity);
    limit(k)=min(p.nominalSpeed,sqrt(p.speedUtilization*capacity/max(curve,1e-8)));
end
speed=limit;
for k=numel(x)-1:-1:1,speed(k)=min(speed(k),sqrt(speed(k+1)^2+2*p.speedDeceleration*(x(k+1)-x(k))));end
for k=2:numel(x),speed(k)=min(speed(k),sqrt(speed(k-1)^2+2*p.speedAcceleration*(x(k)-x(k-1))));end
plan=[x,speed];
end
