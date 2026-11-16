function evaluate_prediction_candidates()
root=fileparts(fileparts(mfilename('fullpath')));addpath(fullfile(root,'controllers','matlab'));
p=matlabCorrectionParameters(loadMatlabMpcParameters(root),root);folder=fullfile(root,'results','prediction_correction');if ~isfolder(folder),mkdir(folder);end
labels={'matched_fixed','native_td3_final','native_td3_reward_selected'};data=cell(1,3);
for k=1:3,v=load(fullfile(root,'results','matlab_comparison',[labels{k} '_raw.mat']),'out');data{k}=v.out;end
s=data{1}.state_log.Data;u=data{1}.control_log.Data(:,1);z=[s(:,4),s(:,3),s(:,5),s(:,6),s(:,2),s(:,1),-.5*(s(:,13)+s(:,14)),-.5*(s(:,15)+s(:,16))];
stats=struct('steer_current_error_deg',rad2deg(max(abs(mean(s(:,9:10),2)-u))), ...
    'steer_previous_error_deg',rad2deg(max(abs(mean(s(2:end,9:10),2)-u(1:end-1)))));
for axle=1:2
    x=zeros(size(s,1),1);y=sum(s(:,15+2*axle:16+2*axle),2);
    for k=1:numel(x),if axle==1,loadN=p.frontLoad;else,loadN=p.rearLoad;end;x(k)=2*matlabGrip(s(k,1),p)*matlabTireForce(z(k,6+axle)/matlabGrip(s(k,1),p),loadN,p);end
    training=s(:,1)>=0&s(:,1)<=80;p.forceGain(axle)=min(1.4,max(.6,(x(training)'*y(training))/(x(training)'*x(training))));
end
save(fullfile(folder,'physical_parameters.mat'),'p','stats');disp(p.forceGain);disp(stats);
results=table();
for variant=1:4
    candidate=p;candidate.loadTransfer=variant>=3;if ismember(variant,[1 3]),candidate.forceGain=ones(2,1);end
    for record=1:3
        out=data{record};s=out.state_log.Data;control=out.control_log.Data(:,1);errors=zeros(0,3);
        for start=6:10:size(s,1)-40
            if s(start,1)<0||s(start+40,1)>120,continue;end
            op=matlabOperatingPoint(s(start,:)',zeros(2,1),s(start-1,3));state=op(1:8);
            [ff,fr]=matlabPhysicalForces(state,control(start),op(9),candidate);bias=[op(10)-ff;op(11)-fr];
            for h=1:40
                state=matlabPhysicalStep(state,control(start+h-1),op(9)*exp(-(h-1)*p.dt/.5),bias*exp(-(h-1)*p.dt/p.forceBiasTime),zeros(3,1),candidate);
                if ismember(h,[1 10 20 40])
                    a=matlabReference(state(6),candidate);b=matlabReference(s(start+h,1),candidate);
                    errors(end+1,:)=[h,state(5)-s(start+h,2),(state(5)-a(1))-(s(start+h,2)-b(1))]; %#ok<AGROW>
                end
            end
        end
        for horizon=[1 10 20 40]
            e=errors(errors(:,1)==horizon,:);row=struct('variant',variant,'record',string(labels{record}),'horizon_s',horizon*p.dt, ...
                'Y_rmse_m',sqrt(mean(e(:,2).^2)),'tracking_prediction_rmse_m',sqrt(mean(e(:,3).^2)));
            results=[results;struct2table(row)]; %#ok<AGROW>
        end
    end
end
writetable(results,fullfile(folder,'candidate_prediction.csv'));disp(results);fprintf('PREDICTION_CANDIDATES_COMPLETE\n');
end
