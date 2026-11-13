function refresh_120m_plots()
% Regenerate current plots without rerunning simulations or rewriting raw data.
root=fileparts(fileparts(mfilename('fullpath')));
labels={'fixed_np40_nc12','ddpg_np40_nc12','td3_np40_nc12', ...
    'training_ddpg_np40_nc12_ep300_seed42_evaluation','training_td3_np40_nc12_ep300_seed42_evaluation'};
for k=1:numel(labels)
    loaded=load(fullfile(root,'results',labels{k},'raw.mat'),'out');
    export_run(loaded.out,labels{k},true);
end
folder=fullfile(root,'results','comparison_np40_nc12');
f=openfig(fullfile(folder,'comparison.fig'),'invisible');
loaded=load(fullfile(root,'results',labels{1},'raw.mat'),'out');
endTime=interp1(loaded.out.state_log.Data(:,1),loaded.out.state_log.Time,plot_station_limit());
for ax=findall(f,'Type','axes')'
    if contains(string(ax.XLabel.String),'X (m)'),xlim(ax,[0 plot_station_limit()]);
    else,xlim(ax,[0 endTime]);end
end
savefig(f,fullfile(folder,'comparison.fig'));
exportgraphics(f,fullfile(folder,'comparison.png'),'Resolution',180);close(f);
plot_current_comparison();
end
