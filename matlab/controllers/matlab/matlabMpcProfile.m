function p=matlabMpcProfile(root,profile,coordinated)
% Construct baseline/corrected ablation parameters without changing files.
arguments
    root (1,:) char
    profile (1,1) double {mustBeMember(profile,0:3)}=0
    coordinated (1,1) logical=false
end
p=loadMatlabMpcParameters(root);p.speedPlanActive=coordinated;
if coordinated,p.speedPlan=matlabPlanSpeed(p);end
if profile>0
    v=load(fullfile(root,'configs','prediction_calibration.mat'),'p');
    p.forceGain=v.p.forceGain;p.relaxationScale=v.p.relaxationScale;p.steerCompliance=v.p.steerCompliance;
    % Profile 3 retains measured-force residual compensation. The additional
    % integral disturbance observer worsened this case's matched-speed test.
    p.loadTransfer=true;p.correctionEnabled=true;p.observerEnabled=profile==2;p.safetyEnabled=profile>=3;
    if p.safetyEnabled,p.maxRate=p.designSteerRate;end
    tuningFile=fullfile(root,'configs','nominal_speed_tuning.json');
    if ~coordinated&&isfile(tuningFile)
        tuning=jsondecode(fileread(tuningFile));
        p.rateChange=tuning.rate_change_weight;p.safetyUtilization=tuning.safety_utilization;
        p.consistentFeedforward=tuning.consistent_feedforward;p.maxIterations=tuning.max_iterations;
    end
end
end
