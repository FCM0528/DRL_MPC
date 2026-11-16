function preview=matlabReferencePreview(s,p)
%#codegen
% Actual CarSim target plus 50 points of distance-based preview, 157 values.
preview=zeros(157,1);preview(1:7)=matlabReference(s(1),p);xd=max(0,s(3)*cos(s(5))-s(4)*sin(s(5)));
for k=1:50
    point=matlabReference(s(1)+k*p.dt*xd,p);preview(8+(k-1)*3:10+(k-1)*3)=point(1:3);
end
end
