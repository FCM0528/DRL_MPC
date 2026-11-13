function run_training_campaign()
% One real 300-episode run per algorithm, then compare deterministic policies.
root=setup();s=read_mpc_settings();c=read_training_config();
assert(s.np==40 && s.nc==12 && c.training.episodes==300);
statusFile=fullfile(root,'results','training_campaign_status.json');
status=struct('configuration',s.key,'episodes_per_algorithm',300,'seed',42, ...
    'started_at',char(datetime('now','Format','yyyy-MM-dd HH:mm:ss')), ...
    'phase','starting','completed',false);
writeStatus();
try
    status.phase='training_ddpg';writeStatus();train_ddpg(300,42);
    status.phase='training_td3';writeStatus();train_td3(300,42);
    status.phase='comparing';writeStatus();compare_controllers();
    status.phase='exporting';writeStatus();plot_current_comparison();
    archive_training_summary(sprintf('training_*_%s_ep300_seed42',s.key));write_current_configuration_report();
    status.phase='complete';status.completed=true;
    status.finished_at=char(datetime('now','Format','yyyy-MM-dd HH:mm:ss'));writeStatus();
    fprintf('TRAINING_CAMPAIGN_COMPLETE: DDPG=300, TD3=300, seed=42\n');
catch ME
    status.phase='failed';status.error=getReport(ME,'extended','hyperlinks','off');writeStatus();rethrow(ME);
end
    function writeStatus()
        fid=fopen(statusFile,'w','n','UTF-8');
        fprintf(fid,'%s',jsonencode(status,'PrettyPrint',true));fclose(fid);
    end
end
