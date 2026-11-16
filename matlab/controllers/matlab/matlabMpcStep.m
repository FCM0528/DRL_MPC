function [control,next,debug]=matlabMpcStep(s,tuning,memory,p)
% Readable MATLAB entry used for offline replay of the same Simulink stages.
arguments
    s (20,1) double
    tuning (5,1) double
    memory struct
    p struct
end
op=matlabOperatingPoint(s,memory.previous,memory.vx);
[nominal,sensitivity,increments,valid]=matlabNominalPrediction(op,memory.solution,memory.hasSolution,p);
targets=matlabPredictionReference(nominal,p);
[h,g]=matlabCostMatrices(op,nominal,sensitivity,increments,targets,tuning,p);
[low,high]=matlabConstraintBounds(op,p);
[moves,status,ok]=matlabAdmmQp(h,g,increments,low,high,p,valid);
[control,solution,previous,hasSolution]=matlabSteeringCommand(op,moves,status,ok,tuning,p);
next=struct('previous',previous,'vx',s(3),'solution',solution,'hasSolution',hasSolution);
debug=struct('op',op,'nominal',nominal,'sensitivity',sensitivity,'hessian',h,'gradient',g,'low',low,'high',high,'moves',moves);
end
