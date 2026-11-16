function finalize_matlab_delivery()
% Run only after the independent native training and comparison have finished.
root=setup_matlab_mpc();folder=fullfile(root,'results','matlab_native_validation');
summary=jsondecode(fileread(fullfile(root,'results','matlab_td3_ep300_seed42','summary.json')));
assert(summary.episodes==300&&strcmp(summary.model,'DRL_MPC_MATLAB')&&~summary.controller_mex_loaded);
assert(isfile(fullfile(root,'results','matlab_comparison','metrics_120m.csv')));
% Update only the private CarSim case's model link, preserving physical data.
parsFile=fullfile(root,'carsim','case','Run_all.par');pars=fileread(parsFile);
modelFile=fullfile(root,'models','DRL_MPC_MATLAB.slx');
oldLink=regexp(pars,'(?m)^SIMULINK_MODEL_FILE [^\r\n]*','match','once');assert(~isempty(oldLink));
updated=strrep(pars,oldLink,['SIMULINK_MODEL_FILE ' modelFile]);
fid=fopen(parsFile,'w','n','UTF-8');fprintf(fid,'%s',updated);fclose(fid);
review_matlab_native();
% Verify the public compatibility entry in a fresh process, using new training.
out=run_paper_preview('selected');
v=load(fullfile(root,'results','matlab_comparison','native_td3_reward_selected_raw.mat'),'out');
stateDifference=max(abs(out.state_log.Data-v.out.state_log.Data),[],'all');
steerDifference=max(abs(out.control_log.Data(:,1)-v.out.control_log.Data(:,1)));
assert(stateDifference<1e-6&&steerDifference<1e-8);
entryText=fileread(fullfile(root,'open_paper_model.m'));
assert(contains(entryText,'open_matlab_model')&&~contains(entryText,'setup_paper'));
m='DRL_MPC_MATLAB';scope=Simulink.ID.getFullName([m ':76']);
print(['-s' m],'-dpng',fullfile(folder,'model_overview.png'));
print(['-s' scope],'-dpng',fullfile(folder,'matlab_controller.png'));
print(['-s' Simulink.ID.getFullName([m ':106'])],'-dpng',fullfile(folder,'scopes.png'));
[files,products]=matlab.codetools.requiredFilesAndProducts(fullfile(root,'controllers','matlab','matlabMpcStep.m'));
assert(~any(contains(files,{'mpc_paper_mex','mpc_mex.mex'})));
[~,mexfiles]=inmem('-completenames');assert(~any(contains(mexfiles,{'mpc_paper_mex','mpc_mex'})));
data=struct('passed',true,'public_entry_model',m,'controller_mex_loaded',false, ...
    'selected_preview_state_difference',stateDifference,'selected_preview_steer_difference',steerDifference, ...
    'matlab_controller_dependencies',{files},'products',{products}, ...
    'private_carsim_model_link',modelFile,'training_episodes',summary.episodes);
parameters=loadMatlabMpcParameters(root);save(fullfile(folder,'native_parameters.mat'),'parameters');
fid=fopen(fullfile(folder,'delivery_audit.json'),'w');fprintf(fid,'%s\n',jsonencode(data,'PrettyPrint',true));fclose(fid);
fprintf('MATLAB_NATIVE_DELIVERY_PASS\n');
end
