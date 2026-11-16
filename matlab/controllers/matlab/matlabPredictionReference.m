function targets=matlabPredictionReference(nominal,p)
%#codegen
% Reference, local heading curvature gradient and smoothed feedforward kappa.
targets=zeros(9,40);
for stage=1:40
    x=nominal(6,stage+1);r=matlabReference(x,p);ahead=matlabReference(x+.2,p);behind=matlabReference(x-.2,p);
    slope=r(5)/(1+r(4)^2);
    slopeX=(ahead(5)/(1+ahead(4)^2)-behind(5)/(1+behind(4)^2))/.4;
    curve=r(3);
    if p.curvatureSigma>0
        curve=0;
        for k=1:73
            point=matlabReference(x+p.curvatureSigma*p.smoothOffsets(k),p);
            curve=curve+p.smoothWeights(k)*point(3);
        end
    end
    targets(:,stage)=[r(1:5);slope;slopeX;curve;matlabGrip(x,p)];
end
end
