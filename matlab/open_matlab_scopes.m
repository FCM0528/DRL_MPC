function open_matlab_scopes(name)
% Open an actual Scope connected to the MATLAB/Simulink controller.
arguments
    name (1,:) char {mustBeMember(name,{'Tracking_Errors','Steering','Weights','Lambda','Reward','QP_Status','Lateral_Capacity','Sideslip','Speed_Curvature','Constraint_Slack'})}='Tracking_Errors'
end
m='DRL_MPC_MATLAB';if ~bdIsLoaded(m),open_matlab_model();end
p=find_system(m,'LookUnderMasks','all','FollowLinks','off','BlockType','Scope','Name',['Scope_' name]);
assert(numel(p)==1);open_system(p{1});
end
