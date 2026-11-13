function [agent,stats]=train_ddpg(episodes,seeds)
if nargin<1,episodes=[];end
if nargin<2,seeds=[];end
[agent,stats]=train_controller('ddpg',episodes,seeds);
end
