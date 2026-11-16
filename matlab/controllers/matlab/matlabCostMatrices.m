function [h,g]=matlabCostMatrices(op,nominal,sensitivity,increments,targets,tuning,p)
%#codegen
% Condensed quadratic cost in Nc steering increments; Q_i=Q exp(lambda*i/Np).
h=zeros(12);g=zeros(12,1);q=[tuning(2);tuning(3);p.qVy;p.qR];
wheelbase=p.a+p.b;cf=matlabTireForce(p.tireSlip(2)*pi/180,p.frontLoad,p)/(p.tireSlip(2)*pi/180);
cr=matlabTireForce(p.tireSlip(2)*pi/180,p.rearLoad,p)/(p.tireSlip(2)*pi/180);
if p.correctionEnabled&&p.consistentFeedforward
    cf=cf*p.forceGain(1);cr=cr*p.forceGain(2);
end
frontPeak=0;rearPeak=0;
for k=1:numel(p.tireSlip)
    frontPeak=max(frontPeak,matlabTireForce(p.tireSlip(k)*pi/180,p.frontLoad,p));
    rearPeak=max(rearPeak,matlabTireForce(p.tireSlip(k)*pi/180,p.rearLoad,p));
end
peakAccel=2*(frontPeak+rearPeak)/p.mass;understeer=p.mass/(2*wheelbase)*(p.b/cf-p.a/cr);
for stage=1:40
    z=nominal(:,stage+1);r=targets(:,stage);s=sensitivity(:,:,stage);
    c=cos(z(3));sn=sin(z(3));xd=z(2)*c-z(1)*sn;yd=z(2)*sn+z(1)*c;
    e=[z(5)-r(1);atan2(sin(z(3)-r(2)),cos(z(3)-r(2)));yd-r(4)*xd;z(4)-r(6)*xd];
    es=zeros(4,12);
    for j=1:12
        xds=c*s(2,j)-sn*s(1,j)-yd*s(3,j);yds=sn*s(2,j)+c*s(1,j)+xd*s(3,j);
        es(:,j)=[s(5,j)-r(4)*s(6,j);s(3,j)-r(6)*s(6,j); ...
            yds-r(4)*xds-r(5)*xd*s(6,j);s(4,j)-r(6)*xds-r(7)*xd*s(6,j)];
    end
    e=e-es*increments;terminal=exp(tuning(5)*stage/40);if stage==40,terminal=terminal*p.terminal;end
    g=g+2*terminal*(es'*(q.*e));h=h+2*terminal*(es'*(q.*es));
    active=min(stage,12);ff=0;
    if p.feedforward&&p.rSteer>0
        vx2=z(2)^2;curveLimit=r(9)*peakAccel/max(1,vx2);stiffnessScale=r(9);
        if p.frontLength>0||p.rearLength>0,stiffnessScale=1;end
        ff=min(p.maxSteer,max(-p.maxSteer,(wheelbase+understeer*vx2/stiffnessScale)*min(curveLimit,max(-curveLimit,r(8)))));
    end
    g(1:active)=g(1:active)+2*p.rSteer*(op(12)-ff);
    h(1:active,1:active)=h(1:active,1:active)+2*p.rSteer;
end
h=h+(2*tuning(4)+1e-9)*eye(12);change=2*p.rateChange;
g(1)=g(1)-change*op(13);h(1,1)=h(1,1)+change;
for i=2:12
    h(i,i)=h(i,i)+change;h(i-1,i-1)=h(i-1,i-1)+change;
    h(i,i-1)=h(i,i-1)-change;h(i-1,i)=h(i-1,i)-change;
end
h(12,12)=h(12,12)+change;
end
