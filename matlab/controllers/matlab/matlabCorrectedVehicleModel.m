function [a,b,c,valid]=matlabCorrectedVehicleModel(z,steer,ax,bias,disturbance,p)
%#codegen
% Consistent discrete Jacobian of the stable physical integrator.
a=zeros(8);b=zeros(8,1);c=zeros(8,1);valid=false;
if any(~isfinite(z))||z(2)<1,return;end
next=matlabPhysicalStep(z,steer,ax,bias,disturbance,p);
if any(~isfinite(next)),return;end
a(5,5)=1;a(6,6)=1;
for j=[1 2 3 4 7 8]
    h=1e-6*max(1,abs(z(j)));perturbed=z;perturbed(j)=perturbed(j)+h;
    value=matlabPhysicalStep(perturbed,steer,ax,bias,disturbance,p);a(:,j)=(value-next)/h;
end
if size(p.gripTable,1)>1
    perturbed=z;perturbed(6)=perturbed(6)+1e-5;value=matlabPhysicalStep(perturbed,steer,ax,bias,disturbance,p);a(:,6)=(value-next)/1e-5;
end
value=matlabPhysicalStep(z,steer+1e-6,ax,bias,disturbance,p);b=(value-next)/1e-6;
c=next-a*z-b*steer;valid=all(isfinite(a(:)))&&all(isfinite(b))&&all(isfinite(c));
end
