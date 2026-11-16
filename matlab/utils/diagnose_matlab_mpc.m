function diagnose_matlab_mpc()
% Independently check QP optimality, model prediction and path feasibility.
root=fileparts(fileparts(mfilename('fullpath')));addpath(fullfile(root,'controllers','matlab'));p=loadMatlabMpcParameters(root);
folder=fullfile(root,'results','mpc_diagnosis');if ~isfolder(folder),mkdir(folder);end
v=load(fullfile(root,'results','matlab_comparison','matched_fixed_raw.mat'),'out');out=v.out;
show=out.state_log.Data(:,1)<=120;s=out.state_log.Data(show,:);commands=out.control_log.Data(show,1);n=size(s,1);
memory=struct('previous',zeros(2,1),'vx',nan,'solution',zeros(12,1),'hasSolution',false);
hasQuadprog=exist('quadprog','file')==2&&license('test','Optimization_Toolbox');
if hasQuadprog
    options=optimoptions('quadprog','Algorithm','interior-point-convex','Display','off', ...
        'OptimalityTolerance',1e-12,'ConstraintTolerance',1e-10,'MaxIterations',500);
end
qp=table();prediction=table();costs=table();G=[tril(ones(12));eye(12)];
for k=1:n
    before=memory;[d,memory,debug]=matlabMpcStep(s(k,:)',[40;9200;2800;13500;0],memory,p);
    if s(k,1)>=0&&mod(k-1,10)==0
        H=debug.hessian;g=debug.gradient;u=debug.moves;scale=max(abs(diag(H)));
        A=[G;-G];b=[debug.high;-debug.low];finite=isfinite(b);A=A(finite,:);b=b(finite);
        exact=-H\g;unconstrainedFeasible=all(A*exact<=b+1e-9);
        if hasQuadprog
            [independent,~,flag]=quadprog((H+H')/(2*scale),g/scale,A,b,[],[],[],[],[],options);
        elseif unconstrainedFeasible
            independent=exact;flag=1;
        else,error('Independent constrained QP solver is unavailable.');end
        assert(flag>0,'Independent QP solve failed.');
        objective=@(x).5*x'*H*x+g'*x;
        q=struct('sample',k,'X_m',s(k,1),'hessian_condition',cond(H), ...
            'hessian_relative_asymmetry',norm(H-H','inf')/max(1,norm(H,'inf')), ...
            'min_hessian_eigenvalue',min(eig((H+H')/2)), ...
            'max_move_difference_rad',max(abs(u-independent)), ...
            'first_move_difference_rad',abs(u(1)-independent(1)), ...
            'relative_objective_gap',abs(objective(u)-objective(independent))/max(1,abs(objective(independent))), ...
            'max_constraint_violation_rad',max([0;A*u-b]),'admm_iterations',d(3), ...
            'unconstrained_optimum_feasible',unconstrainedFeasible,'independent_exitflag',flag);
        qp=[qp;struct2table(q)]; %#ok<AGROW>
        [nominal,sens,inc,~]=matlabNominalPrediction(debug.op,before.solution,before.hasSolution,p);
        targets=matlabPredictionReference(nominal,p);tracking=0;angleCost=0;
        cf=matlabTireForce(p.tireSlip(2)*pi/180,p.frontLoad,p)/(p.tireSlip(2)*pi/180);
        cr=matlabTireForce(p.tireSlip(2)*pi/180,p.rearLoad,p)/(p.tireSlip(2)*pi/180);
        peakF=0;peakR=0;
        for j=1:numel(p.tireSlip),peakF=max(peakF,matlabTireForce(p.tireSlip(j)*pi/180,p.frontLoad,p));peakR=max(peakR,matlabTireForce(p.tireSlip(j)*pi/180,p.rearLoad,p));end
        under=p.mass/(2*(p.a+p.b))*(p.b/cf-p.a/cr);peakAcceleration=2*(peakF+peakR)/p.mass;
        for stage=1:40
            z=nominal(:,stage+1);r=targets(:,stage);delta=u-inc;
            ey=z(5)-r(1)+(sens(5,:,stage)-r(4)*sens(6,:,stage))*delta;
            ep=atan2(sin(z(3)-r(2)),cos(z(3)-r(2)))+(sens(3,:,stage)-r(6)*sens(6,:,stage))*delta;
            tracking=tracking+9200*ey^2+2800*ep^2;
            ff=0;if p.feedforward
                limit=r(9)*peakAcceleration/max(1,z(2)^2);
                ff=min(p.maxSteer,max(-p.maxSteer,(p.a+p.b+under*z(2)^2)*min(limit,max(-limit,r(8)))));
            end
            angle=debug.op(12)+sum(u(1:min(stage,12)));angleCost=angleCost+p.rSteer*(angle-ff)^2;
        end
        rateCost=13500*sum(u.^2);jerkCost=p.rateChange*(sum(diff([debug.op(13);u]).^2)+u(end)^2);
        row=struct('sample',k,'X_m',s(k,1),'tracking_cost',tracking,'angle_feedforward_cost',angleCost, ...
            'increment_cost',rateCost,'increment_change_cost',jerkCost);
        costs=[costs;struct2table(row)]; %#ok<AGROW>
    end
    if s(k,1)>=0&&mod(k-1,5)==0
        op=debug.op;mu=matlabGrip(op(6),p);fb=op(10)-2*mu*matlabTireForce(op(7)/mu,p.frontLoad,p);
        rb=op(11)-2*mu*matlabTireForce(op(8)/mu,p.rearLoad,p);z=op(1:8);
        for h=1:min(40,n-k)
            [A,B,C,valid]=matlabVehicleModel(z,commands(k+h-1),op(9)*exp(-(h-1)*p.dt/.5), ...
                fb*exp(-(h-1)*p.dt/.08),rb*exp(-(h-1)*p.dt/.08),p);
            if ~valid,break;end
            z=A*z+B*commands(k+h-1)+C;
            if ismember(h,[1 5 10 20 40])
                truth=s(k+h,:);ep=atan2(sin(z(3)-truth(5)),cos(z(3)-truth(5)));
                predRef=matlabReference(z(6),p);trueRef=matlabReference(truth(1),p);
                trackingError=(z(5)-predRef(1))-(truth(2)-trueRef(1));
                row=struct('start_X_m',s(k,1),'horizon_steps',h,'horizon_s',h*p.dt, ...
                    'Y_error_m',z(5)-truth(2),'X_error_m',z(6)-truth(1),'yaw_error_rad',ep, ...
                    'vy_error_mps',z(1)-truth(4),'r_error_radps',z(4)-truth(6), ...
                    'lateral_tracking_error_prediction_m',trackingError);
                prediction=[prediction;struct2table(row)]; %#ok<AGROW>
            end
        end
    end
end
writetable(qp,fullfile(folder,'qp_independent_check.csv'));writetable(prediction,fullfile(folder,'prediction_errors.csv'));
writetable(costs,fullfile(folder,'cost_components.csv'));
predictionSummary=table();
for h=[1 5 10 20 40]
    a=prediction(prediction.horizon_steps==h,:);
    row=struct('horizon_steps',h,'horizon_s',h*p.dt,'samples',height(a), ...
        'Y_rmse_m',sqrt(mean(a.Y_error_m.^2)),'Y_peak_abs_m',max(abs(a.Y_error_m)), ...
        'X_rmse_m',sqrt(mean(a.X_error_m.^2)), ...
        'lateral_tracking_prediction_rmse_m',sqrt(mean(a.lateral_tracking_error_prediction_m.^2)), ...
        'yaw_rmse_rad',sqrt(mean(a.yaw_error_rad.^2)),'vy_rmse_mps',sqrt(mean(a.vy_error_mps.^2)), ...
        'r_rmse_radps',sqrt(mean(a.r_error_radps.^2)));
    predictionSummary=[predictionSummary;struct2table(row)]; %#ok<AGROW>
end
writetable(predictionSummary,fullfile(folder,'prediction_summary.csv'));
x=(0:.1:120)';curvature=zeros(size(x));speed=interp1(s(:,1),hypot(s(:,3),s(:,4)),x,'linear','extrap');mu=zeros(size(x));
for k=1:numel(x),r=matlabReference(x(k),p);curvature(k)=r(3);mu(k)=matlabGrip(x(k),p);end
demand=speed.^2.*abs(curvature);friction=mu*9.80665;tireCapacity=mu*peakAcceleration;
feasibility=table(x,curvature,speed,demand,friction,tireCapacity,'VariableNames',{'X_m','curvature_per_m','speed_mps','reference_normal_accel_mps2','mu_g_mps2','static_tire_table_peak_mps2'});
writetable(feasibility,fullfile(folder,'reference_feasibility.csv'));
summary=struct('quadprog_used',hasQuadprog,'qp_samples',height(qp), ...
    'max_first_move_difference_rad',max(qp.first_move_difference_rad), ...
    'max_relative_objective_gap',max(qp.relative_objective_gap), ...
    'max_constraint_violation_rad',max(qp.max_constraint_violation_rad), ...
    'min_hessian_eigenvalue',min(qp.min_hessian_eigenvalue), ...
    'max_hessian_condition',max(qp.hessian_condition), ...
    'reference_peak_normal_accel_mps2',max(demand),'reference_mu_g_exceed_fraction',mean(demand>friction), ...
    'static_tire_peak_accel_mps2',max(tireCapacity),'reference_static_tire_exceed_fraction',mean(demand>tireCapacity), ...
    'measured_peak_body_ay_mps2',max(abs(s(:,8))),'max_rate_radps',p.maxRate, ...
    'steering_rate_limit_enabled',isfinite(p.maxRate));
fid=fopen(fullfile(folder,'summary.json'),'w');fprintf(fid,'%s\n',jsonencode(summary,'PrettyPrint',true));fclose(fid);
f=figure('Visible','off','Color','w','Position',[100 100 1200 450]);tiledlayout(1,2);
nexttile;plot(predictionSummary.horizon_s,predictionSummary.Y_rmse_m,'-o','LineWidth',1.6);hold on;plot(predictionSummary.horizon_s,predictionSummary.lateral_tracking_prediction_rmse_m,'-s','LineWidth',1.6);xlabel('预测时间 / s');ylabel('开环预测RMSE / m');legend('全局Y位置','横向跟踪误差');grid on;
nexttile;plot(x,demand,'LineWidth',1.5);hold on;plot(x,friction,'--','LineWidth',1.5);plot(x,tireCapacity,':','LineWidth',1.5);xlim([0 120]);xlabel('X / m');ylabel('法向加速度 / m/s²');legend('参考曲率与实际车速需求','μg参考线','静载轮胎表峰值估计');grid on;
set(findall(f,'-property','FontName'),'FontName','宋体');exportgraphics(f,fullfile(folder,'model_diagnosis.png'),'Resolution',160);close(f);
disp(summary);disp(predictionSummary);fprintf('MPC_DIAGNOSIS_COMPLETE\n');
end
