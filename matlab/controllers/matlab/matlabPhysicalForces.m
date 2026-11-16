function [front,rear,lengthF,lengthR]=matlabPhysicalForces(z,steer,ax,p)
%#codegen
% CarSim-derived force/relaxation tables, optional quasi-static load transfer.
mu=matlabGrip(z(6),p);loadF=p.frontLoad;loadR=p.rearLoad;
if p.loadTransfer
    transfer=p.mass*p.cgHeight*ax/(2*(p.a+p.b));loadF=loadF-transfer;loadR=loadR+transfer;
end
loadF=min(p.tireLoads(end),max(p.tireLoads(2)*.1,loadF));loadR=min(p.tireLoads(end),max(p.tireLoads(2)*.1,loadR));
front=2*mu*p.forceGain(1)*matlabTireForce(z(7)/mu,loadF,p);
rear=2*mu*p.forceGain(2)*matlabTireForce(z(8)/mu,loadR,p);
if p.loadTransfer
    ay=max(-mu*9.80665,min(mu*9.80665,z(2)*z(4)));
    df=p.mass*p.cgHeight*ay/p.trackFront*p.b/(p.a+p.b);dr=p.mass*p.cgHeight*ay/p.trackRear*p.a/(p.a+p.b);
    front=0;rear=0;
    for side=[-1 1]
        lf=min(p.tireLoads(end),max(p.tireLoads(2)*.1,loadF+side*df));lr=min(p.tireLoads(end),max(p.tireLoads(2)*.1,loadR+side*dr));
        instant=steer-atan((z(1)+p.a*z(4))/max(1,z(2)));
        instSide=steer-atan((z(1)+p.a*z(4))/max(1,z(2)+side*p.trackFront*z(4)/2));
        ar=-atan((z(1)-p.b*z(4))/max(1,z(2)));arSide=-atan((z(1)-p.b*z(4))/max(1,z(2)+side*p.trackRear*z(4)/2));
        front=front+mu*p.forceGain(1)*matlabTireForce((z(7)+instSide-instant)/mu,lf,p);
        rear=rear+mu*p.forceGain(2)*matlabTireForce((z(8)+arSide-ar)/mu,lr,p);
    end
end
lengthF=max(.005,p.relaxationScale(1)*p.relaxationGain*matlabSplineValue(z(7)*180/pi,p.relaxation)*matlabSplineValue(loadF/p.relaxationLoadScale,p.relaxationLoad));
lengthR=max(.005,p.relaxationScale(2)*p.relaxationGain*matlabSplineValue(z(8)*180/pi,p.relaxation)*matlabSplineValue(loadR/p.relaxationLoadScale,p.relaxationLoad));
end
