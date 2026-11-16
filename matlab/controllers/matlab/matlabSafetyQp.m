function [moves,status,valid,slack]=matlabSafetyQp(h,g,warm,lowAct,highAct,op,nominal,sensitivity,predictionValid,p)
%#codegen
% Thirteen-variable ADMM: twelve steering moves plus one dimensionless slack.
if ~p.safetyEnabled
    [moves,status,valid]=matlabAdmmQp(h,g,warm,lowAct,highAct,p,predictionValid);slack=0;return;
end
moves=zeros(12,1);status=zeros(4,1);valid=false;slack=0;if ~predictionValid,return;end
[matrix,low,high]=matlabSafetyConstraints(op,nominal,sensitivity,warm,p);
H=zeros(13);H(1:12,1:12)=h;H(13,13)=2*p.slackQuadratic;gradient=[g;p.slackLinear];
lhs=H+p.rho*(matrix'*matrix);lhs=(lhs+lhs')/2;l=zeros(13);
for i=1:13
    for j=1:i
        value=lhs(i,j);for k=1:j-1,value=value-l(i,k)*l(j,k);end
        if i==j
            if ~(value>0)||~isfinite(value),return;end
            l(i,j)=sqrt(value);
        else,l(i,j)=value/l(j,j);
        end
    end
end
u=[warm;0];violation=matrix*u-high;u(13)=max(0,max(violation(26:end)));
z=min(high,max(low,matrix*u));dual=zeros(345,1);
for iteration=1:p.maxIterations
    rhs=-gradient+p.rho*(matrix'*(z-dual));y=zeros(13,1);
    for i=1:13
        value=rhs(i);for j=1:i-1,value=value-l(i,j)*y(j);end;y(i)=value/l(i,i);
    end
    for i=13:-1:1
        value=y(i);for j=i+1:13,value=value-l(j,i)*u(j);end;u(i)=value/l(i,i);
    end
    if any(~isfinite(u)),return;end
    au=matrix*u;old=z;z=min(high,max(low,au+dual));dual=dual+au-z;
    primal=max(abs(au-z));dr=max(abs(p.rho*(matrix'*(z-old))));status=[0;iteration;primal;dr];
    if primal<p.primalTol&&dr<p.dualTol,status(1)=1;break;end
end
moves=u(1:12);slack=max(0,u(13));valid=true;
end
