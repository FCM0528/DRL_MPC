function [names,positions]=route_monitor_signals()
% Route long logging and reward return wires through explicitly scoped tags.
model='DRL_MPC';
sourceNames={'Reference_Trajectory_Preview','Tracking_Error_Calculation','Previous_Control_Result','Reward_Termination','Reward_Termination'};
sourcePorts=[1 1 1 1 2];tags={'RefPreview','TrackingErrors','PreviousMpcResult','DecisionReward','EpisodeDone'};
gotoNames={'Reference_Monitor_Bus','Error_Monitor_Bus','Previous_MPC_Result','Decision_Reward_Signal','Episode_Done_Signal'};
if isempty(find_system(model,'SearchDepth',1,'Name',gotoNames{1}))
    ops={};
    for k=1:5
        ops{end+1}=struct('op','add_block','type','Goto','name',gotoNames{k},'ref',['g' num2str(k)],'params',struct('GotoTag',tags{k},'TagVisibility','scoped','ShowName','off'));
        ops{end+1}=struct('op','add_block','type','GotoTagVisibility','name',[gotoNames{k} '_Scope'],'params',struct('GotoTag',tags{k},'ShowName','off'));
        ops{end+1}=struct('op','connect','target',sprintf('%s.y%d -> #g%d.u1',migration_id(sourceNames{k}),sourcePorts(k),k));
    end
    fromNames={'Ref_For_Monitor','Error_For_Monitor','Previous_For_DRL','Previous_For_Reward','Reward_For_DRL','Reward_For_Monitor','Done_For_DRL','Done_For_Stop'};
    tagIndex=[1 2 3 3 4 4 5 5];targets={'Monitoring_Evaluation','Monitoring_Evaluation','DRL_Adaptive_Tuning','Reward_Termination','DRL_Adaptive_Tuning','Monitoring_Evaluation','DRL_Adaptive_Tuning','Episode_Complete'};ports=[3 4 2 2 3 5 4 1];
    for k=1:numel(fromNames)
        ops{end+1}=struct('op','add_block','type','From','name',fromNames{k},'ref',['f' num2str(k)],'params',struct('GotoTag',tags{tagIndex(k)},'ShowName','off','FontSize','9'));
        ops{end+1}=struct('op','connect','target',sprintf('#f%d.y1 -> %s.u%d',k,migration_id(targets{k}),ports(k)));
    end
    result=model_edit(model,'root',jsonencode(ops),'incremental');assert(~contains(result,'status: partial'),result);
end
ops={};
for k=1:5,ops{end+1}=struct('op','connect','target',sprintf('%s.y%d -> %s.u1',migration_id(sourceNames{k}),sourcePorts(k),migration_id(gotoNames{k})));end
fromNames={'Ref_For_Monitor','Error_For_Monitor','Previous_For_DRL','Previous_For_Reward','Reward_For_DRL','Reward_For_Monitor','Done_For_DRL','Done_For_Stop'};
targets={'Monitoring_Evaluation','Monitoring_Evaluation','DRL_Adaptive_Tuning','Reward_Termination','DRL_Adaptive_Tuning','Monitoring_Evaluation','DRL_Adaptive_Tuning','Episode_Complete'};ports=[3 4 2 2 3 5 4 1];
for k=1:8,ops{end+1}=struct('op','connect','target',sprintf('%s.y1 -> %s.u%d',migration_id(fromNames{k}),migration_id(targets{k}),ports(k)));end
result=model_edit(model,'root',jsonencode(ops),'incremental');assert(~contains(result,'status: partial'),result);
names=[gotoNames,cellfun(@(n)[n '_Scope'],gotoNames,'UniformOutput',false),{'Ref_For_Monitor','Error_For_Monitor','Previous_For_DRL','Previous_For_Reward','Reward_For_DRL','Reward_For_Monitor','Done_For_DRL','Done_For_Stop'}];
positions=[975 180 1075 210;1345 130 1450 160;800 660 930 690;1350 500 1460 524;1350 535 1460 559; ...
    710 20 920 40;1080 20 1300 40;710 725 930 745;1080 600 1210 620;1080 710 1210 730; ...
    1530 550 1615 574;1530 579 1630 603;555 445 670 475;920 527 1035 557;585 475 685 505;1530 608 1630 632;600 505 685 535;1530 740 1615 764];
end
