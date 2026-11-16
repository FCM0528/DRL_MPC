function value=matlabSplineValue(x,table)
%#codegen
% Natural cubic interpolation with flat extrapolation, for physical tables.
n=size(table,1);
if x<=table(1,1),value=table(1,2);return;end
if x>=table(n,1),value=table(n,2);return;end
lo=1;hi=n;
while hi-lo>1,mid=floor((lo+hi)/2);if table(mid,1)<=x,lo=mid;else,hi=mid;end;end
h=table(hi,1)-table(lo,1);a=(table(hi,1)-x)/h;b=(x-table(lo,1))/h;
value=a*table(lo,2)+b*table(hi,2)+((a^3-a)*table(lo,3)+(b^3-b)*table(hi,3))*h*h/6;
end
