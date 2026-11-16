function review_matlab_native()
% Static MATLAB review and meaningful solver/failure-mode checks, no plant run.
root=fileparts(fileparts(mfilename('fullpath')));source=fullfile(root,'controllers','matlab');addpath(source);
p=loadMatlabMpcParameters(root);files=dir(fullfile(source,'*.m'));issues=struct('file',{},'line',{},'message',{});
for file=files'
    name=fullfile(file.folder,file.name);info=checkcode(name,'-struct');
    for k=1:numel(info),issues(end+1)=struct('file',file.name,'line',info(k).line,'message',info(k).message);end %#ok<AGROW>
end
v=load(fullfile(root,'tests','fixtures','controller_reference.mat'),'fixtures');states=v.fixtures(1).state;
memory=struct('previous',zeros(2,1),'vx',nan,'solution',zeros(12,1),'hasSolution',false);tuning=[40;9200;2800;13500;0];
for k=1:79,[~,memory]=matlabMpcStep(states(k,:)',tuning,memory,p);end
s=states(80,:)';
forced=p;forced.maxIterations=1;[d,~,debug]=matlabMpcStep(s,tuning,memory,forced);
assert(d(2)==0&&all(isfinite(d))&&abs(d(1))<=p.maxSteer);
assert(norm(debug.hessian-debug.hessian','inf')<1e-12*max(1,norm(debug.hessian,'inf')));
[~,positiveDefinite]=chol((debug.hessian+debug.hessian')/2);assert(positiveDefinite==0);
invalid=s;invalid(3)=.5;[held,next]=matlabMpcStep(invalid,tuning,memory,p);
assert(held(1)==memory.previous(1)&&held(2)==0&&~next.hasSolution&&next.previous(2)==0);
missing=s;missing(17:20)=nan;[~,next]=matlabMpcStep(missing,tuning,memory,p);assert(next.hasSolution);
limited=p;limited.maxRate=.3;[d]=matlabMpcStep(s,tuning,memory,limited);
assert(abs(d(1)-memory.previous(1))<=limited.maxRate*limited.dt+1e-12);
lambdaSteer=zeros(3,1);
for k=1:3,tuning(5)=k-2;d=matlabMpcStep(s,tuning,memory,p);lambdaSteer(k)=d(1);end
assert(max(lambdaSteer)-min(lambdaSteer)>1e-6,'Lambda must change the actual control calculation.');
% A scalar trace of runtime dependency names is also inspectable without MEX.
for file=files'
    text=fileread(fullfile(file.folder,file.name));
    assert(~contains(text,{'mpc_paper_mex','mpc_mex(','coder.extrinsic','coder.ceval','loadlibrary(','calllib('}));
end
data=struct('files_reviewed',numel(files),'code_analyzer_messages',{issues}, ...
    'forced_nonconvergence_projection_passed',true,'invalid_prediction_hold_passed',true, ...
    'missing_force_bias_fallback_passed',true,'finite_steering_rate_bound_passed',true, ...
    'lambda_control_values',lambdaSteer,'controller_bridge_reference_count',0);
folder=fullfile(root,'results','matlab_native_validation');fid=fopen(fullfile(folder,'code_review.json'),'w','n','UTF-8');
fprintf(fid,'%s\n',jsonencode(data,'PrettyPrint',true));fclose(fid);
fprintf('MATLAB_NATIVE_REVIEW_PASS: %d files, %d analyzer messages\n',numel(files),numel(issues));
end
