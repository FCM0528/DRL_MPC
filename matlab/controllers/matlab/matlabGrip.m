function mu=matlabGrip(x,p)
%#codegen
% CarSim STEP station table, with gain, offset and lateral constant.
station=(x-p.gripStart)/p.gripScale;i=1;
while i<size(p.gripTable,1)&&p.gripTable(i+1,1)<=station,i=i+1;end
mu=p.gripGain*p.gripTable(i,2)+p.gripOffset;
end
