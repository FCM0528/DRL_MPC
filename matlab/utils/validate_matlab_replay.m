function validate_matlab_replay()
root=fileparts(fileparts(mfilename('fullpath')));addpath(fullfile(root,'controllers','matlab'));p=loadMatlabMpcParameters(root);
folder=fullfile(root,'results','matlab_native_validation');if ~isfolder(folder),mkdir(folder);end
data=load(fullfile(root,'tests','fixtures','controller_reference.mat'),'fixtures');fixtures=data.fixtures;
results=table();
for item=1:numel(fixtures)
    fixture=fixtures(item);
    memory=struct('previous',zeros(2,1),'vx',nan,'solution',zeros(12,1),'hasSolution',false);
    n=size(fixture.state,1);actual=zeros(n,9);ref=zeros(n,7);
    for k=1:n
        s=fixture.state(k,:)';t=[fixture.control(k,6:9)';fixture.lambda(k)];
        [d,memory]=matlabMpcStep(s,t,memory,p);actual(k,:)=d';ref(k,:)=matlabReference(s(1),p)';
    end
    difference=actual-fixture.control;
    r=struct('fixture',string(fixture.label),'samples',n,'max_steer_difference_rad',max(abs(difference(:,1))), ...
        'convergence_mismatch',nnz(difference(:,2)),'max_iteration_difference',max(abs(difference(:,3))), ...
        'max_reference_difference',max(abs(ref-fixture.reference),[],'all'));
    results=[results;struct2table(r)]; %#ok<AGROW>
    disp(r);
end
writetable(results,fullfile(folder,'replay.csv'));
assert(all(results.max_steer_difference_rad<1e-6)&&all(results.convergence_mismatch==0));
assert(all(results.max_reference_difference<1e-10));
fprintf('MATLAB_NATIVE_REPLAY_PASS\n');
end
