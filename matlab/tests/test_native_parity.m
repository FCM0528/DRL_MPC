function test_native_parity()
root=setup();settings=read_mpc_settings();x=load(fullfile(root,'results',settings.fixed_label,'raw.mat'),'out');
s=x.out.state_log.Data;writematrix(s,fullfile(root,'results','parity_states.csv'));
cmd=sprintf('"%s" "%s" "%s" "%s" "%s"',fullfile(root,'build','parity_original.exe'),fullfile(root,'configs','carsim_python_rl.ini'),fullfile(root,'carsim','case'),fullfile(root,'results','parity_states.csv'),fullfile(root,'results','parity_original.csv'));
[status,msg]=system(cmd);assert(status==0,msg);expected=readmatrix(fullfile(root,'results','parity_original.csv'));
mpc_mex('reset');actual=zeros(size(s,1),35);prev=0;inc=0;tuning=settings.baseline;
for k=1:size(s,1)
    state=s(k,:)';if k==1,state(7)=0;else,state(7)=(s(k,3)-s(k-1,3))/.04;end
    d=mpc_mex('step',state,tuning,prev,inc);
    actual(k,:)=[d(1:5);mpc_mex('prediction',state,d(1));mpc_mex('observe',state,tuning(1),prev);mpc_mex('reference',state(1));mpc_mex('reward',state,d(1),prev,tuning,tuning)]';
    inc=d(1)-prev;prev=d(1);
end
error=max(abs(actual-expected),[],1);disp(error);
% CSV round trip and independent compilation perturb final ADMM residuals.
assert(error(1)<1e-10 && all(error(2:3)==0) && all(error(4:5)<1e-7));
assert(all(error(6:13)<1e-9) && all(error(14:27)<1e-9) && all(error(28:34)<1e-12) && error(35)<1e-8);
info=mpc_mex('info');assert(all(abs(actual(:,1))<=info(1)+1e-12));
save(fullfile(root,'results','native_parity.mat'),'actual','expected','error');
summary=[error(1:5),max(error(6:13)),max(error(14:27)),max(error(28:34)),error(35)];
writetable(array2table(summary,'VariableNames',{'steer_rad','converged','iterations','primal','dual','predicted_state','observation','reference','reward'}),fullfile(root,'results','native_parity.csv'));
end
