function op=matlabOperatingPoint(s,previous,previousVx)
%#codegen
% [8-state plant; body-vx derivative; observed axle Fy; previous steer/delta].
ax=0;if isfinite(previousVx),ax=(s(3)-previousVx)/.04;end
op=[s(4);s(3);s(5);s(6);s(2);s(1);-.5*(s(13)+s(14));-.5*(s(15)+s(16)); ...
    min(6,max(-6,ax));s(17)+s(18);s(19)+s(20);previous(1);previous(2)];
end
