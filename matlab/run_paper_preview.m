function out=run_paper_preview(controller)
% Compatibility entry for the MATLAB/Simulink implementation.
if nargin<1,controller='fixed';end
out=run_matlab_preview(controller);
end
