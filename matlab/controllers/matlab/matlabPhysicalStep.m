function next=matlabPhysicalStep(z,steer,ax,forceBias,disturbance,p)
%#codegen
% Stable substep integration: RK4 body motion plus exact tire relaxation.
step=p.dt/p.integrationSubsteps;next=z;
for k=1:p.integrationSubsteps
    origin=next;[f1,instant1,rate1]=body(origin,steer,ax,forceBias,disturbance,p);
    q2=origin;q2(1:6)=origin(1:6)+.5*step*f1;q2(7:8)=instant1+(origin(7:8)-instant1).*exp(-.5*step*rate1);
    [f2,instant2,rate2]=body(q2,steer,ax,forceBias,disturbance,p);
    q3=origin;q3(1:6)=origin(1:6)+.5*step*f2;q3(7:8)=instant2+(origin(7:8)-instant2).*exp(-.5*step*rate2);
    [f3,instant3,rate3]=body(q3,steer,ax,forceBias,disturbance,p);
    q4=origin;q4(1:6)=origin(1:6)+step*f3;q4(7:8)=instant3+(origin(7:8)-instant3).*exp(-step*rate3);
    f4=body(q4,steer,ax,forceBias,disturbance,p);
    next(1:6)=origin(1:6)+step*(f1+2*f2+2*f3+f4)/6;
    next(7:8)=instant3+(origin(7:8)-instant3).*exp(-step*rate3);
end
end
function [f,instant,rate]=body(z,steer,ax,bias,d,p)
[front,rear,lf,lr]=matlabPhysicalForces(z,steer,ax,p);front=front+bias(1);rear=rear+bias(2);
steer=steer+p.steerCompliance*front;
v=z(1);u=max(1,z(2));yaw=z(3);r=z(4);c=cos(yaw);s=sin(yaw);
f=[(front*cos(steer)+rear)/p.mass-u*r+d(1);ax+d(2);r; ...
    (p.a*front*cos(steer)-p.b*rear)/p.iz+d(3);u*s+v*c;u*c-v*s];
instant=[steer-atan((v+p.a*r)/u);-atan((v-p.b*r)/u)];rate=[u/lf;u/lr];
end
