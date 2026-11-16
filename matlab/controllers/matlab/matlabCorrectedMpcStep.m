function [control,next,debug]=matlabCorrectedMpcStep(s,tuning,memory,p)
% Offline entry identical to the corrected Simulink stages.
[d,innovation]=matlabDisturbanceObserver(s,memory.observer,p);
op=[matlabOperatingPoint(s,memory.previous,memory.vx);d];
[nominal,sensitivity,increments,valid]=matlabCorrectedPrediction(op,memory.solution,memory.hasSolution,p);
targets=matlabPredictionReference(nominal,p);[h,g]=matlabCostMatrices(op,nominal,sensitivity,increments,targets,tuning,p);
[low,high]=matlabConstraintBounds(op,p);
[moves,status,ok,slack]=matlabSafetyQp(h,g,increments,low,high,op,nominal,sensitivity,valid,p);
[control,solution,previous,hasSolution]=matlabSteeringCommand(op,moves,status,ok,tuning,p);
observer=matlabObserverForecast(op,control,p);
next=struct('previous',previous,'vx',s(3),'solution',solution,'hasSolution',hasSolution,'observer',observer);
debug=struct('nominal',nominal,'sensitivity',sensitivity,'increments',increments,'op',op,'slack',slack, ...
    'innovation',innovation,'safety',matlabSafetyMonitor(s,slack,p),'hessian',h,'gradient',g);
end
