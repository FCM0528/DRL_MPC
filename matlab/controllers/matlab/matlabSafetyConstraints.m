function [matrix,low,high]=matlabSafetyConstraints(op,nominal,sensitivity,increments,p)
%#codegen
% Hard angle/rate bounds plus normalized soft ay, beta and axle slip bounds.
matrix=zeros(345,13);low=-inf(345,1);high=inf(345,1);base=[tril(ones(12));eye(12)];
matrix(1:24,1:12)=base;[low(1:24),high(1:24)]=matlabConstraintBounds(op,p);
matrix(25,13)=1;low(25)=0;angle=op(12)+cumsum(increments);
for stage=1:40
    z=nominal(:,stage+1);active=min(stage,12);steer=angle(active);
    ax=(nominal(2,stage+1)-nominal(2,stage))/p.dt;mu=matlabGrip(z(6),p);
    bodyAx=ax-z(1)*z(4);
    capacity=min(mu*9.80665,mu*p.baseLateralCapacity);ayLimit=p.safetyUtilization*sqrt(max(.1,capacity^2-bodyAx^2));
    limits=[ayLimit;p.betaLimit;p.slipUtilization*mu*p.peakSlip];d=op(14:16)*exp(-stage*p.dt/p.disturbanceTime);
    y=matlabLateralQuantities(z,steer,ax,d,p);jac=zeros(4,8);
    for j=[1 2 4 7 8]
        step=1e-6*max(1,abs(z(j)));q=z;q(j)=q(j)+step;
        jac(:,j)=(matlabLateralQuantities(q,steer,ax,d,p)-y)/step;
    end
    input=(matlabLateralQuantities(z,steer+1e-6,ax,d,p)-y)/1e-6;
    derivative=jac*sensitivity(:,:,stage);derivative(:,1:active)=derivative(:,1:active)+input;
    derivative=derivative./limits;offset=y./limits-derivative*increments;
    index=25+(stage-1)*8;
    for j=1:4
        matrix(index+2*j-1,1:12)=derivative(j,:);matrix(index+2*j-1,13)=-1;high(index+2*j-1)=1-offset(j);
        matrix(index+2*j,1:12)=-derivative(j,:);matrix(index+2*j,13)=-1;high(index+2*j)=1+offset(j);
    end
end
end
