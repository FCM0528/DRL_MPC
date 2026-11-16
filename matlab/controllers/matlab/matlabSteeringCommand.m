function [control,solution,memory,hasSolution]=matlabSteeringCommand(op,moves,status,valid,tuning,p)
%#codegen
% Roll out constrained steering moves; hold the previous command on failure.
solution=zeros(12,1);last=op(12);maxStep=p.maxRate*p.dt;
if valid
    for i=1:12
        last=min(min(p.maxSteer,last+maxStep),max(max(-p.maxSteer,last-maxStep),last+moves(i)));
        solution(i)=last;
    end
else
    solution(:)=last;status(:)=0;
end
steer=solution(1);control=[steer;status;40;tuning(2:4)];memory=[steer;steer-op(12)];hasSolution=valid;
end
