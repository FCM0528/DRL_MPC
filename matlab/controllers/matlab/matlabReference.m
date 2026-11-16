function r=matlabReference(x,p)
%#codegen
% Natural cubic LTARG spline, endpoint-flat. [Y;psi;kappa;Yx;Yxx;Xmin;Xmax].
t=p.reference;n=size(t,1);r=zeros(7,1);r(6)=t(1,1);r(7)=t(n,1);
if x<=t(1,1),r(1)=t(1,2);return;end
if x>=t(n,1),r(1)=t(n,2);return;end
lo=1;hi=n;
while hi-lo>1
    mid=floor((hi+lo)/2);if t(mid,1)<=x,lo=mid;else,hi=mid;end
end
h=t(hi,1)-t(lo,1);a=(t(hi,1)-x)/h;b=(x-t(lo,1))/h;
r(1)=a*t(lo,2)+b*t(hi,2)+((a^3-a)*t(lo,3)+(b^3-b)*t(hi,3))*h*h/6;
r(4)=(t(hi,2)-t(lo,2))/h+(-(3*a*a-1)*t(lo,3)+(3*b*b-1)*t(hi,3))*h/6;
r(5)=a*t(lo,3)+b*t(hi,3);r(2)=atan(r(4));r(3)=r(5)/(1+r(4)^2)^1.5;
end
