function validate_corrected_prediction()
root=fileparts(fileparts(mfilename('fullpath')));addpath(fullfile(root,'controllers','matlab'));
folder=fullfile(root,'results','prediction_correction');results=table();
labels={'matched_fixed','native_td3_final','native_td3_reward_selected','same_final_policy_lambda_zero'};
for record=1:4
    v=load(fullfile(root,'results','matlab_comparison',[labels{record} '_raw.mat']),'out');s=v.out.state_log.Data;u=v.out.control_log.Data(:,1);
    for variant=0:2
        p=matlabMpcProfile(root,min(2,variant+1),false);p.observerEnabled=variant==2;
        mem=zeros(12,1);disturbances=zeros(3,size(s,1));
        for k=1:size(s,1)-1
            [d,~]=matlabDisturbanceObserver(s(k,:)',mem,p);previousVx=nan;if k>1,previousVx=s(k-1,3);end
            op=[matlabOperatingPoint(s(k,:)',zeros(2,1),previousVx);d];mem=matlabObserverForecast(op,[u(k);zeros(8,1)],p);disturbances(:,k)=d;
        end
        errors=zeros(0,5);
        for start=6:5:size(s,1)-40
            if s(start,1)<0||s(start+40,1)>120,continue;end
            op=matlabOperatingPoint(s(start,:)',zeros(2,1),s(start-1,3));z=op(1:8);
            [f,r]=matlabPhysicalForces(z,u(start),op(9),p);bias=[op(10)-f;op(11)-r];
            if variant==0
                mu=matlabGrip(op(6),p);bias=[op(10)-2*mu*matlabTireForce(z(7)/mu,p.frontLoad,p);op(11)-2*mu*matlabTireForce(z(8)/mu,p.rearLoad,p)];
            end
            for h=1:40
                ax=op(9)*exp(-(h-1)*p.dt/.5);decay=exp(-(h-1)*p.dt/p.forceBiasTime);
                if variant==0
                    [a,b,c,ok]=matlabVehicleModel(z,u(start+h-1),ax,bias(1)*decay,bias(2)*decay,p);assert(ok);z=a*z+b*u(start+h-1)+c;
                else
                    d=disturbances(:,start)*exp(-(h-1)*p.dt/p.disturbanceTime);
                    z=matlabPhysicalStep(z,u(start+h-1),ax,bias*decay,d,p);
                end
                if ismember(h,[1 10 20 40])
                    a=matlabReference(z(6),p);b=matlabReference(s(start+h,1),p);
                    errors(end+1,:)=[h,z(5)-s(start+h,2),(z(5)-a(1))-(s(start+h,2)-b(1)),s(start,1),s(start+h,1)]; %#ok<AGROW>
                end
            end
        end
        for horizon=[1 10 20 40]
            e=errors(errors(:,1)==horizon,:);row=struct('record',string(labels{record}),'variant',variant,'horizon_s',horizon*p.dt,'windows',height(e), ...
                'Y_rmse_m',sqrt(mean(e(:,2).^2)),'tracking_prediction_rmse_m',sqrt(mean(e(:,3).^2)));
            results=[results;struct2table(row)]; %#ok<AGROW>
        end
    end
end
writetable(results,fullfile(folder,'prediction_validation.csv'));disp(results);fprintf('CORRECTED_PREDICTION_VALIDATION_COMPLETE\n');
end
