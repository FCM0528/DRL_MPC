function verify_correction_math()
% Independent QP reference and finite-difference directional validation.
root=setup_matlab_mpc();p=matlabMpcProfile(root,3,false);
v=load(fullfile(root,'results','matlab_comparison','matched_fixed_raw.mat'),'out');s=v.out.state_log.Data;u=v.out.control_log.Data(:,1);rows=table();
for k=[40 80 110]
    op=[matlabOperatingPoint(s(k,:)',[u(k-1);u(k-1)-u(k-2)],s(k-1,3));zeros(3,1)];
    [nominal,S,warm,ok]=matlabCorrectedPrediction(op,repmat(u(k-1),12,1),true,p);assert(ok);
    target=matlabPredictionReference(nominal,p);[h,g]=matlabCostMatrices(op,nominal,S,warm,target,[40;9200;2800;13500;0],p);
    [low,high]=matlabConstraintBounds(op,p);[moves,status,valid,slack]=matlabSafetyQp(h,g,warm,low,high,op,nominal,S,ok,p);
    [A,lb,ub]=matlabSafetyConstraints(op,nominal,S,warm,p);H=blkdiag(h,2*p.slackQuadratic);H=(H+H')/2;f=[g;p.slackLinear];
    finiteUpper=isfinite(ub);finiteLower=isfinite(lb);
    options=optimoptions('quadprog','Display','off','OptimalityTolerance',1e-9,'ConstraintTolerance',1e-9,'MaxIterations',1000);
    [ref,obj,flag]=quadprog(H,f,[A(finiteUpper,:);-A(finiteLower,:)], [ub(finiteUpper);-lb(finiteLower)],[],[],[],[],[],options);assert(flag>0);
    solution=[moves;slack];violation=max([0;A*solution-ub;lb-A*solution]);
    row=struct('sample',k,'x_m',s(k,1),'valid',valid,'converged',status(1),'iterations',status(2), ...
        'max_bound_violation',violation,'first_move_difference_rad',abs(moves(1)-ref(1)), ...
        'relative_objective_difference',abs((.5*solution'*H*solution+f'*solution)-obj)/max(1,abs(obj)), ...
        'slack',slack,'reference_slack',ref(13));rows=[rows;struct2table(row)]; %#ok<AGROW>
end
writetable(rows,fullfile(root,'results','nominal_speed_correction','qp_independent_check.csv'));disp(rows);
fprintf('CORRECTION_MATH_CHECK_COMPLETE\n');
end
