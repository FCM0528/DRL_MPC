function [moves,status,valid]=matlabAdmmQp(h,g,warm,low,high,p,predictionValid)
%#codegen
% MATLAB implementation of the constrained ADMM QP; no external solver bridge.
moves=zeros(12,1);status=zeros(4,1);valid=false;
if ~predictionValid||any(~isfinite(h(:)))||any(~isfinite(g)),return;end
matrix=[tril(ones(12));eye(12)];lhs=h+p.rho*(matrix'*matrix);
% Explicit Cholesky and substitutions mirror the reference algorithm.
l=zeros(12);
for i=1:12
    for j=1:i
        value=lhs(i,j);for k=1:j-1,value=value-l(i,k)*l(j,k);end
        if i==j
            if ~(value>0)||~isfinite(value),return;end
            l(i,j)=sqrt(value);
        else
            l(i,j)=value/l(j,j);
        end
    end
end
u=warm;z=matrix*u;dual=zeros(24,1);
for iteration=1:p.maxIterations
    rhs=-g+p.rho*(matrix'*(z-dual));y=zeros(12,1);
    for i=1:12
        value=rhs(i);for j=1:i-1,value=value-l(i,j)*y(j);end;y(i)=value/l(i,i);
    end
    for i=12:-1:1
        value=y(i);for j=i+1:12,value=value-l(j,i)*u(j);end;u(i)=value/l(i,i);
    end
    if any(~isfinite(u)),return;end
    au=matrix*u;old=z;z=min(high,max(low,au+dual));dual=dual+au-z;
    primal=max(abs(au-z));dualResidual=max(abs(p.rho*(matrix'*(z-old))));
    status=[0;iteration;primal;dualResidual];
    if primal<p.primalTol&&dualResidual<p.dualTol,status(1)=1;break;end
end
moves=u;valid=true;
end
