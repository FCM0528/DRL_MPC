function refresh_matlab_metrics()
% Recalculate the exact 0<=X<=120 domain without rerunning training or plant.
root=fileparts(fileparts(mfilename('fullpath')));folder=fullfile(root,'results','matlab_comparison');
audit=fullfile(root,'results','cleanup');if ~isfolder(audit),mkdir(audit);end
prior=fullfile(audit,'metrics_before_range_fix.csv');if ~isfile(prior),copyfile(fullfile(folder,'metrics_120m.csv'),prior);end
labels={'matched_fixed','native_td3_final','same_final_policy_lambda_zero','native_td3_reward_selected'};rows=table();
for k=1:numel(labels)
    file=fullfile(folder,[labels{k} '_raw.mat']);v=load(file,'out');out=v.out;
    metrics=metrics_to_distance(out,labels{k},120);show=out.state_log.Data(:,1)>=0&out.state_log.Data(:,1)<=120;
    metrics.lateral_mae_m=mean(abs(out.error_log.Data(show,1)));metrics.heading_mae_rad=mean(abs(out.error_log.Data(show,2)));
    metrics.max_abs_front_command_deg=rad2deg(max(abs(out.control_log.Data(show,1))));
    rows=[rows;struct2table(metrics)];save(file,'out','metrics','-v7.3'); %#ok<AGROW>
end
writetable(rows,fullfile(folder,'metrics_120m.csv'));
file=fullfile(folder,'checkpoint_selection.json');selection=jsondecode(fileread(file));selection.metrics=table2struct(rows(4,:));
fid=fopen(file,'w');fprintf(fid,'%s\n',jsonencode(selection,'PrettyPrint',true));fclose(fid);
disp(rows);fprintf('MATLAB_METRICS_RANGE_FIXED\n');
end
