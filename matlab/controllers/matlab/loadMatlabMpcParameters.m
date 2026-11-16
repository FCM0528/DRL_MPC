function p=loadMatlabMpcParameters(root)
% Read the current CarSim data and MPC configuration without a controller DLL.
arguments
    root (1,:) char
end
cfg=readIni(fullfile(root,'configs','paper_mpc.ini'));
echo=fileread(fullfile(root,'carsim','case','LastRun_echo.par'));
pars=fileread(fullfile(root,'carsim','case','Run_all.par'));
p.mass=value('M_TL');p.iz=value('IZZ_TL');
p.a=(value('LX_CG_TL')-value('LX_AXLE(1)'))/1000;
p.b=(value('LX_AXLE(2)')-value('LX_CG_TL'))/1000;
p.dt=number('mpc.dt_s');p.np=number('mpc.np');p.nc=number('mpc.nc');
assert(p.np==40&&p.nc==12,'This implementation uses fixed Np=40, Nc=12.');
p.frontLength=number('mpc.front_relaxation_m');p.rearLength=number('mpc.rear_relaxation_m');
p.slopeFloor=number('mpc.tire_slope_floor');p.curvatureSigma=number('mpc.reference_curvature_sigma_m');
p.rSteer=number('weights.r_steer');p.rateChange=number('weights.r_delta_change');
p.terminal=number('weights.terminal_scale');p.feedforward=strcmp(cfg('weights.steer_feedforward'),'true');
p.qVy=number('weights.q_vy');p.qR=number('weights.q_r');
assert(number('weights.q_tire_slip')==0&&strcmp(cfg('weights.track_course'),'false'), ...
    'The paper configuration uses body heading and no tire-slip tracking penalty.');
p.rho=number('solver.rho');p.maxIterations=number('solver.max_iterations');
p.primalTol=number('solver.primal_tol');p.dualTol=number('solver.dual_tol');
rate=cfg('constraints.max_steer_rate_deg_s');
if strcmp(rate,'none'),p.maxRate=inf;else,p.maxRate=str2double(rate)*pi/180;end
left=tableData(echo,'RACK_KIN_TABLE(1,1)');right=tableData(echo,'RACK_KIN_TABLE(1,2)');
left(:,2)=value('RACK_KIN_GAIN(1,1)')*left(:,2)+value('RACK_KIN_OFFSET(1,1)');
right(:,2)=value('RACK_KIN_GAIN(1,2)')*right(:,2)+value('RACK_KIN_OFFSET(1,2)');
bounds=[max(left(1,1),right(1,1)),min(left(end,1),right(end,1))];
lv=interp1(left(:,1),left(:,2),bounds);rv=interp1(right(:,1),right(:,2),bounds);
p.maxSteer=min([-min(lv),max(lv),-min(rv),max(rv)])*pi/180;
p.frontLoad=p.mass*9.80665*p.b/(2*(p.a+p.b));p.rearLoad=p.mass*9.80665*p.a/(2*(p.a+p.b));
tire=readmatrix(fullfile(root,'carsim','case','carsim_tire_fy.csv'));
% readmatrix treats the mixed text/numeric header differently across releases.
header=regexp(fileread(fullfile(root,'carsim','case','carsim_tire_fy.csv')),'^[^\r\n]+','match','once');
parts=split(string(header),',');p.tireLoads=str2double(parts(2:end))';
if isnan(tire(1,1)),tire=tire(2:end,:);end
p.tireSlip=[0;tire(:,1)];p.tireForce=[zeros(1,numel(p.tireLoads));tire(:,2:end)];
assert(all(diff(p.tireSlip)>0)&&all(diff(p.tireLoads)>0));
assert(p.frontLoad>=p.tireLoads(1)&&p.frontLoad<=p.tireLoads(end));
assert(p.rearLoad>=p.tireLoads(1)&&p.rearLoad<=p.tireLoads(end));
assert(contains(pars,'set_description path_id Straight East'),'Reference station must equal global X.');
ref=tableData(pars,'LTARG_TABLE');n=size(ref,1);lower=zeros(n,1);diag=ones(n,1);upper=zeros(n,1);rhs=zeros(n,1);
for k=2:n-1
    h0=ref(k,1)-ref(k-1,1);h1=ref(k+1,1)-ref(k,1);
    lower(k)=h0;diag(k)=2*(h0+h1);upper(k)=h1;
    rhs(k)=6*((ref(k+1,2)-ref(k,2))/h1-(ref(k,2)-ref(k-1,2))/h0);
end
for k=2:n,t=lower(k)/diag(k-1);diag(k)=diag(k)-t*upper(k-1);rhs(k)=rhs(k)-t*rhs(k-1);end
d2=zeros(n,1);d2(n)=rhs(n)/diag(n);
for k=n-1:-1:1,d2(k)=(rhs(k)-upper(k)*d2(k+1))/diag(k);end
p.reference=[ref,d2];p.smoothOffsets=(-36:36)/12;
p.smoothWeights=exp(-.5*p.smoothOffsets.^2);p.smoothWeights=p.smoothWeights/sum(p.smoothWeights);
p.gripGain=value('MU_ROAD_GAIN(1)');p.gripOffset=value('MU_ROAD_OFFSET(1)')+value('MU_ROAD_L_CONSTANT(1)');
if contains(echo,'MU_ROAD_TABLE(1) STEP')
    p.gripTable=tableData(echo,'MU_ROAD_TABLE(1)');p.gripStart=value('SSTART_MU_ROAD(1)');p.gripScale=value('SSCALE_MU_ROAD(1)');
else
    p.gripTable=[0,value('MU_ROAD_CONSTANT(1)')];p.gripStart=0;p.gripScale=1;
end
assert(all([p.mass,p.iz,p.a,p.b,p.maxSteer]>0)&&p.dt==.04);
assert(p.rho>0&&p.maxIterations>=1&&p.maxIterations==floor(p.maxIterations));
assert(p.primalTol>0&&p.dualTol>0&&p.rSteer>=0&&p.rateChange>=0&&p.terminal>0);
assert(p.frontLength>=0&&p.rearLength>=0&&p.slopeFloor>=0&&p.slopeFloor<=1);
assert(p.maxRate>0&&p.curvatureSigma>=0&&size(p.tireForce,2)==numel(p.tireLoads));
p=matlabCorrectionParameters(p,root);
    function v=value(key)
        escaped=regexptranslate('escape',key);
        token=regexp(echo,['(?m)^\s*!?\s*' escaped '\s+([-+0-9.eE]+)(?:\s|$)'],'tokens');
        assert(isscalar(token),'Missing or ambiguous CarSim parameter: %s',key);v=str2double(token{1}{1});assert(isfinite(v));
    end
    function v=number(key),v=str2double(cfg(key));assert(isfinite(v),'Invalid INI number: %s',key);end
end
function data=tableData(text,key)
escaped=regexptranslate('escape',key);
token=regexp(text,['(?ms)^\s*' escaped '[^\r\n]*\r?\n(.*?)^\s*ENDTABLE'],'tokens');
assert(isscalar(token),'Missing or ambiguous table: %s',key);
lines=splitlines(string(token{1}{1}));data=zeros(numel(lines),2);count=0;
for k=1:numel(lines)
    line=strtrim(lines(k));if strlength(line)==0||startsWith(line,{'!','#'}),continue;end
    row=sscanf(char(line),'%f , %f');assert(numel(row)==2,'Malformed row in %s',key);count=count+1;data(count,:)=row';
end
data=data(1:count,:);assert(size(data,1)>=1&&all(diff(data(:,1))>0));
end
function cfg=readIni(file)
lines=splitlines(string(fileread(file)));cfg=containers.Map('KeyType','char','ValueType','char');section='';
for k=1:numel(lines)
    line=strtrim(char(lines(k)));if isempty(line)||ismember(line(1),'#;'),continue;end
    if line(1)=='[',section=line(2:end-1);continue;end
    index=strfind(line,'=');if isempty(index),continue;end
    cfg([section '.' strtrim(line(1:index(1)-1))])=strtrim(line(index(1)+1:end));
end
if isKey(cfg,'inherit.base')
    inherited=readIni(fullfile(fileparts(file),cfg('inherit.base')));keys=cfg.keys;
    for k=1:numel(keys),inherited(keys{k})=cfg(keys{k});end;cfg=inherited;
end
end
