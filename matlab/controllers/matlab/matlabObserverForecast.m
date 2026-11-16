function memory=matlabObserverForecast(op,control,p)
%#codegen
memory=zeros(12,1);if ~p.observerEnabled,return;end
[front,rear]=matlabPhysicalForces(op(1:8),control(1),op(9),p);bias=[op(10)-front;op(11)-rear];
next=matlabPhysicalStep(op(1:8),control(1),op(9),bias,op(14:16),p);
if all(isfinite(next)),memory=[next;op(14:16);1];end
end
