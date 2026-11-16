function create_coordinated_case()
% Create a private longitudinal speed scenario; retain both steering imports.
root=fileparts(fileparts(mfilename('fullpath')));addpath(fullfile(root,'controllers','matlab'));
p=matlabCorrectionParameters(loadMatlabMpcParameters(root),root);plan=matlabPlanSpeed(p);
folder=fullfile(root,'carsim','coordinated_case');if ~isfolder(folder),mkdir(folder);end
source=fullfile(root,'carsim','case');names={'LastRun_echo.par','carsim_tire_fy.csv'};
for k=1:numel(names),copyfile(fullfile(source,names{k}),fullfile(folder,names{k}));end
pars=fileread(fullfile(source,'Run_all.par'));
assert(contains(pars,'SPEED_TARGET_COMBINE ADD')&&contains(pars,'SPEED_TARGET_CONSTANT 70'));
% The original driver's default OPT_BK_SC=0 only permits coast-down.
% Give the private coordinated scenario braking authority explicitly.
pars=regexprep(pars,'(?m)^OPT_SC 3[^\r\n]*',sprintf('OPT_SC 3\nOPT_BK_SC 1'),'once');
text=sprintf('SET_UNITS SPEED_TARGET_S_TABLE km/h\nSPEED_TARGET_S_TABLE LINEAR_FLAT\n');
for k=1:size(plan,1),text=[text,sprintf('%.6f, %.9f\n',plan(k,1),plan(k,2)*3.6-70)];end %#ok<AGROW>
text=[text,sprintf('ENDTABLE')];pars=strrep(pars,'SPEED_TARGET_S_CONSTANT 0',text);
fid=fopen(fullfile(folder,'Run_all.par'),'w','n','UTF-8');fprintf(fid,'%s',pars);fclose(fid);
sim=fileread(fullfile(source,'simfile.sim'));sim=strrep(sim,source,folder);
fid=fopen(fullfile(folder,'simfile.sim'),'w','n','UTF-8');fprintf(fid,'%s',sim);fclose(fid);
writetable(array2table([plan(:,1),plan(:,2)*3.6],'VariableNames',{'station_m','target_kmh'}),fullfile(folder,'speed_plan.csv'));
metadata=struct('source_case',source,'new_case',folder,'LTARG_unchanged',true,'imports_unchanged',true, ...
    'mechanism','CarSim SPEED_TARGET time component 70 km/h plus station component', ...
    'design_speed_utilization',p.speedUtilization,'max_deceleration_mps2',p.speedDeceleration, ...
    'max_acceleration_mps2',p.speedAcceleration,'minimum_target_kmh',min(plan(:,2))*3.6);
metadata.driver_braking_enabled=true;metadata.curvature_preview_m=p.speedLookahead;
fid=fopen(fullfile(folder,'manifest.json'),'w');fprintf(fid,'%s\n',jsonencode(metadata,'PrettyPrint',true));fclose(fid);
disp(metadata);fprintf('COORDINATED_CASE_READY\n');
end
