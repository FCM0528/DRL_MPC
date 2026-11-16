function [d,innovation]=matlabDisturbanceObserver(s,memory,p)
%#codegen
% Causal bounded innovation observer. Memory contains last one-step forecast.
d=zeros(3,1);innovation=zeros(3,1);
if ~p.observerEnabled||memory(12)<.5||any(~isfinite(s)),return;end
actual=[s(4);s(3);s(6)];innovation=actual-memory([1 2 4]);
d=min(p.observerBound,max(-p.observerBound,memory(9:11)+p.observerGain.*innovation/p.dt));
end
