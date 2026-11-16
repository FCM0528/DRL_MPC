function y=matlabLateralQuantities(z,steer,ax,d,p)
%#codegen
[front,rear]=matlabPhysicalForces(z,steer,ax,p);effective=steer+p.steerCompliance*front;
y=[(front*cos(effective)+rear)/p.mass+d(1);atan2(z(1),z(2));z(7);z(8)];
end
