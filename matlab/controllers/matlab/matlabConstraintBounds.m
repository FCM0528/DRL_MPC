function [low,high]=matlabConstraintBounds(op,p)
%#codegen
% G=[tril(ones(Nc));eye(Nc)], actuator angle and increment constraints.
low=[repmat(-p.maxSteer-op(12),12,1);repmat(-p.maxRate*p.dt,12,1)];
high=[repmat(p.maxSteer-op(12),12,1);repmat(p.maxRate*p.dt,12,1)];
end
