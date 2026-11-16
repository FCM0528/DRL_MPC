function [agent,metadata]=load_paper_agent(file)
% Accept final_agent.mat and RL Toolbox EpisodeFrequency checkpoints.
assert(isfile(file),'Agent file does not exist: %s',file);
contents=whos('-file',file);index=find(contains({contents.class},'rlTD3Agent'));
assert(numel(index)==1,'Expected exactly one compatible TD3 agent in %s.',file);
variable=contents(index).name;loaded=load(file,variable);agent=loaded.(variable);
metadata=struct('file',file,'variable',variable);
if any(strcmp({contents.name},'settings')),s=load(file,'settings');metadata.settings=s.settings;end
agent.UseExplorationPolicy=false;
end
