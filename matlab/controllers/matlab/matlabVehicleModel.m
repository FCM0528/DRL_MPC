function [ad,bd,cd,valid]=matlabVehicleModel(z,steer,ax,frontBias,rearBias,p)
%#codegen
% Eight-state local Jacobian and affine Tustin discretization, in MATLAB.
% z=[vy;vx;psi;r;Y;X;alpha_f;alpha_r].
ad=zeros(8);bd=zeros(8,1);cd=zeros(8,1);valid=false;
v=z(1);u=z(2);yaw=z(3);rate=z(4);dt=p.dt;
if any(~isfinite(z))||u<1||~isfinite(ax)||~isfinite(steer),return;end
mu=matlabGrip(z(6),p);if ~isfinite(mu)||mu<=0,return;end
dynamic=p.frontLength>0||p.rearLength>0;
ratioF=(v+p.a*rate)/u;ratioR=(v-p.b*rate)/u;
if dynamic,instantF=steer-atan(ratioF);instantR=-atan(ratioR);else,instantF=steer-ratioF;instantR=-ratioR;end
if p.frontLength>0,alphaF=z(7);else,alphaF=instantF;end
if p.rearLength>0,alphaR=z(8);else,alphaR=instantR;end
if dynamic,scale=mu;else,scale=1;end
h=1e-5;
cf0=matlabTireForce(p.tireSlip(2)*pi/180,p.frontLoad,p)/(p.tireSlip(2)*pi/180);
cr0=matlabTireForce(p.tireSlip(2)*pi/180,p.rearLoad,p)/(p.tireSlip(2)*pi/180);
sf=(matlabTireForce((alphaF+h)/scale,p.frontLoad,p)-matlabTireForce((alphaF-h)/scale,p.frontLoad,p))/(2*h);
sr=(matlabTireForce((alphaR+h)/scale,p.rearLoad,p)-matlabTireForce((alphaR-h)/scale,p.rearLoad,p))/(2*h);
cf=2*mu*max(sf,p.slopeFloor*cf0/scale);cr=2*mu*max(sr,p.slopeFloor*cr0/scale);
fyf=2*mu*matlabTireForce(alphaF/scale,p.frontLoad,p)+frontBias;
fyr=2*mu*matlabTireForce(alphaR/scale,p.rearLoad,p)+rearBias;
c=cos(yaw);s=sin(yaw);
if dynamic,lat=cos(steer);lon=sin(steer);else,lat=1;lon=steer;end
jac=zeros(8);input=zeros(8,1);
f=[(fyf*lat+fyr)/p.mass-u*rate;ax;rate;(p.a*fyf*lat-p.b*fyr)/p.iz;u*s+v*c;u*c-v*s;0;0];
[fg,jf,bf,ff]=axleGradient(7,p.a,p.frontLength,instantF,alphaF,cf,1,z,p,dynamic);
[rg,jr,br,fr]=axleGradient(8,-p.b,p.rearLength,instantR,alphaR,cr,0,z,p,dynamic);
jac(7,:)=jf;input(7)=bf;f(7)=ff;jac(8,:)=jr;input(8)=br;f(8)=fr;
jac(1,:)=(fg*lat+rg)/p.mass;jac(2,:)=-fg*lon/p.mass;jac(4,:)=(p.a*fg*lat-p.b*rg)/p.iz;
jac(1,2)=jac(1,2)-rate;jac(1,4)=jac(1,4)-u;jac(2,1)=jac(2,1)+rate;jac(2,4)=jac(2,4)+v;
if p.frontLength>0,forceSteer=0;else,forceSteer=cf;end
latSteer=forceSteer*lat;if dynamic,latSteer=latSteer-fyf*lon;end
input(1)=latSteer/p.mass;input(2)=-(forceSteer*lon+fyf*lat)/p.mass;input(4)=p.a*latSteer/p.iz;
jac(3,4)=1;jac(5,1)=c;jac(5,2)=s;jac(5,3)=u*c-v*s;
jac(6,1)=-s;jac(6,2)=c;jac(6,3)=-u*s-v*c;
% Pivoted elimination preserves the reference algorithm's discretization.
aug=[eye(8)-.5*dt*jac,eye(8)+.5*dt*jac,dt*input,dt*(f-jac*z-input*steer)];
for col=1:8
    pivot=col;for row=col+1:8,if abs(aug(row,col))>abs(aug(pivot,col)),pivot=row;end;end
    if ~isfinite(aug(pivot,col))||abs(aug(pivot,col))<1e-12,return;end
    if pivot~=col,temp=aug(col,:);aug(col,:)=aug(pivot,:);aug(pivot,:)=temp;end
    diagonal=aug(col,col);aug(col,col:end)=aug(col,col:end)/diagonal;
    for row=1:8
        if row~=col,factor=aug(row,col);aug(row,col:end)=aug(row,col:end)-factor*aug(col,col:end);end
    end
end
ad=aug(:,9:16);bd=aug(:,17);cd=aug(:,18);valid=all(isfinite(aug(:)));
end
function [gradient,row,input,f]=axleGradient(index,axle,length,instant,lagged,stiffness,steerDerivative,z,p,dynamic)
u=z(2);ratio=(z(1)+axle*z(4))/u;
if dynamic,slope=1/(1+ratio*ratio);else,slope=1;end
gradient=zeros(1,8);gradient(1)=-slope/u;gradient(2)=slope*ratio/u;gradient(4)=-slope*axle/u;
row=zeros(1,8);input=0;f=0;
if length>0
    effective=length/(1+(lagged/(2*pi/180))^2);rate=min(u/effective,2/p.dt);
    f=rate*(instant-lagged);row=rate*gradient;
    if rate<2/p.dt,row(2)=row(2)+(instant-lagged)/effective;end
    row(index)=-rate;input=rate*steerDerivative;gradient=zeros(1,8);gradient(index)=stiffness;
else
    gradient=gradient*stiffness;
end
end
