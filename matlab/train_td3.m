function [agent,stats]=train_td3(episodes,seeds)
if nargin<1,episodes=[];end
if nargin<2,seeds=[];end
[agent,stats]=train_controller('td3',episodes,seeds);
end
