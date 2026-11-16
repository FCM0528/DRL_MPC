function f=matlabTireForce(alpha,load,p)
%#codegen
% Bounded, odd symmetric bilinear interpolation of the CarSim tire table.
angle=min(abs(alpha)*180/pi,p.tireSlip(end));i=1;j=1;
while i<size(p.tireSlip,1)-1&&p.tireSlip(i+1)<=angle,i=i+1;end
while j<numel(p.tireLoads)-1&&p.tireLoads(j+1)<=load,j=j+1;end
ta=(angle-p.tireSlip(i))/(p.tireSlip(i+1)-p.tireSlip(i));
tf=(load-p.tireLoads(j))/(p.tireLoads(j+1)-p.tireLoads(j));
low=p.tireForce(i,j)+tf*(p.tireForce(i,j+1)-p.tireForce(i,j));
high=p.tireForce(i+1,j)+tf*(p.tireForce(i+1,j+1)-p.tireForce(i+1,j));
f=max(0,low+ta*(high-low));if alpha<0,f=-f;end
end
