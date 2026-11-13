function [id,path]=migration_id(name,scope)
% Resolve actual paths; never reconstruct paths from display text.
if nargin<2,scope='DRL_MPC';end
if startsWith(scope,'blk_'),scope=Simulink.ID.getFullName(['DRL_MPC:' extractAfter(scope,'blk_')]);end
paths=find_system(scope,'SearchDepth',1,'LookUnderMasks','all','FollowLinks','off','Name',name);
paths=paths(~strcmp(paths,scope));assert(numel(paths)==1,'Expected one block named %s',name);
path=paths{1};id=['blk_' extractAfter(Simulink.ID.getSID(path),':')];
end
