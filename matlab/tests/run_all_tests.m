function run_all_tests()
root=setup();open_system(fullfile(root,'models','DRL_MPC.slx'));
tests={};
settings=read_mpc_settings();
check=model_check('DRL_MPC','root');assert(contains(check,'status: healthy'),check);tests{end+1}='Model structural connectivity';
assert(isequal(mpc_mex('map',zeros(4,1)),settings.baseline));
assert(isequal(mpc_mex('map',-ones(4,1)),[settings.min_np;4600;1400;6750]));
assert(isequal(mpc_mex('map',ones(4,1)),[settings.max_np;13800;5600;27000]));tests{end+1}='Neutral and action endpoints';
test_steering_interface;tests{end+1}='Real steering authority: zero/positive/negative';
test_native_parity;tests{end+1}='Original executable vs MATLAB MEX numeric parity';
test_mex_pipeline;test_mex_pipeline(true);tests{end+1}='Seven MEX stages: online Np/Q/R and forced QP failure';
in=Simulink.SimulationInput('DRL_MPC');in=in.setVariable('controller_mode',0);in=in.setVariable('steering_interface_test',0);
first=sim(in);second=sim(in);
reset_error=max(abs(first.state_log.Data-second.state_log.Data),[],'all');assert(reset_error<1e-9);
assert(all(first.control_log.Data(:,2)==1)&&first.state_log.Data(end,1)>=150);
tests{end+1}='Full fixed-loop completion, QP status and repeat Reset';
data=load(fullfile(root,'results',settings.fixed_label,'raw.mat'),'out');
pipeline_error=max(abs(first.control_log.Data(:,1)-data.out.control_log.Data(:,1)));
save(fullfile(root,'results','reset_validation.mat'),'reset_error','pipeline_error');
fid=fopen(fullfile(root,'results','test_results.txt'),'w');
for k=1:numel(tests),fprintf(fid,'PASS %s\n',tests{k});end
fprintf(fid,'Reset maximum state error: %.17g\n',reset_error);
fprintf(fid,'Pipeline vs previous fixed control maximum error: %.17g\n',pipeline_error);fclose(fid);
disp(tests');
end
