function calibrate_prediction_model()
root=fileparts(fileparts(mfilename('fullpath')));addpath(fullfile(root,'controllers','matlab'));
folder=fullfile(root,'results','prediction_correction');v=load(fullfile(folder,'physical_parameters.mat'),'p');p=v.p;
p.relaxationScale=ones(2,1);p.loadTransfer=true;p.integrationSubsteps=5;
v=load(fullfile(root,'results','matlab_comparison','matched_fixed_raw.mat'),'out');s=v.out.state_log.Data;u=v.out.control_log.Data(:,1);
starts=6:10:size(s,1)-40;starts=starts(s(starts,1)>=0&s(starts+40,1)<=80);
mask=s(2:end,1)>=0&s(2:end,1)<=80;force=sum(s(2:end,17:18),2);actual=mean(s(2:end,9:10),2);
steerError=actual-u(1:end-1);p.steerCompliance=(force(mask)'*steerError(mask))/(force(mask)'*force(mask));
fprintf('Measured steering compliance %.9g rad/N; residual rms %.6g deg\n',p.steerCompliance,rad2deg(sqrt(mean((steerError(mask)-p.steerCompliance*force(mask)).^2))));
initial=[p.forceGain;1;1];options=optimset('Display','iter','MaxIter',100,'MaxFunEvals',200,'TolX',1e-4,'TolFun',1e-5);
initialLoss=objective(initial);[parameters,loss]=fminsearch(@objective,initial,options);
assert(all(parameters(1:2)>.8&parameters(1:2)<1.2)&&all(parameters(3:4)>.4&parameters(3:4)<1.6));
p.forceGain=parameters(1:2);p.relaxationScale=parameters(3:4);p.integrationSubsteps=10;
metadata=struct('training_record','matched_fixed','training_domain_m',[0 80],'training_windows',numel(starts), ...
    'parameters',parameters,'initial_loss',initialLoss,'final_loss',loss,'validation_records_not_used_in_optimizer',true);
save(fullfile(folder,'calibrated_parameters.mat'),'p','metadata');
fid=fopen(fullfile(folder,'calibration.json'),'w');fprintf(fid,'%s\n',jsonencode(metadata,'PrettyPrint',true));fclose(fid);
disp(metadata);fprintf('MODEL_CALIBRATION_COMPLETE\n');
    function loss=objective(theta)
        if any(theta(1:2)<.8|theta(1:2)>1.2)||any(theta(3:4)<.4|theta(3:4)>1.6),loss=1e6+sum(theta.^2);return;end
        q=p;q.forceGain=theta(1:2);q.relaxationScale=theta(3:4);residual=0;count=0;
        for start=starts
            op=matlabOperatingPoint(s(start,:)',zeros(2,1),s(start-1,3));state=op(1:8);
            [ff,fr]=matlabPhysicalForces(state,u(start),op(9),q);bias=[op(10)-ff;op(11)-fr];
            for h=1:40
                state=matlabPhysicalStep(state,u(start+h-1),op(9)*exp(-(h-1)*q.dt/.5),bias*exp(-(h-1)*q.dt/q.forceBiasTime),zeros(3,1),q);
                if any(~isfinite(state)),loss=1e8;return;end
                if ismember(h,[10 20 40])
                    a=matlabReference(state(6),q);b=matlabReference(s(start+h,1),q);
                    ey=(state(5)-a(1))-(s(start+h,2)-b(1));ep=atan2(sin(state(3)-s(start+h,5)),cos(state(3)-s(start+h,5)));
                    truth=s(start+h,:);alpha=[-.5*(truth(13)+truth(14));-.5*(truth(15)+truth(16))];
                    residual=residual+(ey/.1)^2+(ep/.05)^2+ ...
                        .2*((state(1)-truth(4))/.2)^2+.2*((state(4)-truth(6))/.05)^2+.1*sum(((state(7:8)-alpha)/.01).^2);count=count+1;
                end
            end
        end
        loss=residual/count+.001*sum(((theta-initial)./[.1;.1;.4;.4]).^2);
    end
end
