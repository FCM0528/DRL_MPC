function build_prediction_correction()
% Edit in an isolated MATLAB process using the proven native API path.
root=setup_matlab_mpc();sldiagviewer.diary(fullfile(root,'results','prediction_correction','compiler_diagnostics.txt'));m='DRL_MPC_MATLAB';open_system(fullfile(root,'models',[m '.slx']));
scope=Simulink.ID.getFullName([m ':76']);rt=sfroot;
all=rt.find('-isa','Stateflow.EMChart');
for ch=all'
    if startsWith(ch.Path,[scope '/'])
        data=ch.find('-isa','Stateflow.Data','Name','op');if ~isempty(data),data.Props.Array.Size='[16 1]';end
    end
end
ch=chart(421);ch.Script=sprintf('function [op,vx]=fcn(s,memory,previousVx,disturbance)\nop=[matlabOperatingPoint(s,memory,previousVx);disturbance];vx=s(3);\nend');
input(ch,'disturbance',4,[3 1],'double');output(ch,'op',1,[16 1],'double');drawnow;
ch=chart(422);ch.Script=sprintf('function [nominal,sensitivity,increments,valid]=fcn(op,solution,hasSolution)\n[nominal,sensitivity,increments,valid]=matlabCorrectedPrediction(op,solution,hasSolution,matlab_mpc_parameters);\nend');drawnow;
input(ch,'op',1,[16 1],'double');output(ch,'nominal',1,[8 41],'double');output(ch,'sensitivity',2,[8 12 40],'double');output(ch,'increments',3,[12 1],'double');output(ch,'valid',4,1,'boolean');
parameter(ch);
ch=chart(426);ch.Script=sprintf(['function [moves,status,valid,slack]=fcn(h,g,warm,low,high,predictionValid,op,nominal,sensitivity)\n' ...
    '[moves,status,valid,slack]=matlabSafetyQp(h,g,warm,low,high,op,nominal,sensitivity,predictionValid,matlab_mpc_parameters);\nend']);
input(ch,'op',7,[16 1],'double');input(ch,'nominal',8,[8 41],'double');input(ch,'sensitivity',9,[8 12 40],'double');
output(ch,'slack',4,1,'double');drawnow;
parameter(ch);
wire(scope,path(421),1,path(426),7);wire(scope,path(422),1,path(426),8);wire(scope,path(422),2,path(426),9);
if isempty(find_system(scope,'SearchDepth',1,'Name','Mismatch_Observer'))
    observer=fn(scope,'Mismatch_Observer',sprintf('function [d,innovation]=fcn(s,memory)\n[d,innovation]=matlabDisturbanceObserver(s,memory,matlab_mpc_parameters);\nend'),[350 780 590 880]);
    ch=rt.find('-isa','Stateflow.EMChart','Path',getfullname(observer));input(ch,'s',1,[20 1],'double');input(ch,'memory',2,[12 1],'double');output(ch,'d',1,[3 1],'double');output(ch,'innovation',2,[3 1],'double');
    forecast=fn(scope,'Observer_Forecast',sprintf('function memory=fcn(op,control)\nmemory=matlabObserverForecast(op,control,matlab_mpc_parameters);\nend'),[2390 760 2640 850]);
    ch=rt.find('-isa','Stateflow.EMChart','Path',getfullname(forecast));input(ch,'op',1,[16 1],'double');input(ch,'control',2,[9 1],'double');output(ch,'memory',1,[12 1],'double');
    delay=add_block('simulink/Discrete/Unit Delay',[scope '/Observer_Memory'],'InitialCondition','zeros(12,1)','SampleTime','.04','Position',[1120 820 1320 870]);
    wire(scope,path(245),1,observer,1);wire(scope,delay,1,observer,2);wire(scope,observer,1,path(421),4);
    wire(scope,path(421),1,forecast,1);wire(scope,path(427),1,forecast,2);wire(scope,forecast,1,delay,1);
    log=add_block('simulink/Sinks/To Workspace',[scope '/Innovation_Log'],'VariableName','model_innovation_log','SaveFormat','Timeseries','Position',[700 910 950 940]);
    wire(scope,observer,2,log,1);
    monitor=fn(scope,'Lateral_Monitor',sprintf('function log=fcn(s,slack)\nlog=matlabSafetyMonitor(s,slack,matlab_mpc_parameters);\nend'),[1770 790 2040 870]);
    ch=rt.find('-isa','Stateflow.EMChart','Path',getfullname(monitor));input(ch,'s',1,[20 1],'double');input(ch,'slack',2,1,'double');output(ch,'log',1,[9 1],'double');
    format=add_block('simulink/Math Operations/Reshape',[scope '/Safety_Vector'],'OutputDimensionality','1-D array','Position',[2100 810 2130 840]);
    tag=add_block('simulink/Signal Routing/Goto',[scope '/Safety_Tag'],'GotoTag','NativeSafety','TagVisibility','global','Position',[2180 810 2340 840]);
    wire(scope,path(245),1,monitor,1);wire(scope,path(426),4,monitor,2);wire(scope,monitor,1,format,1);wire(scope,format,1,tag,1);
    note(scope,'有界失配观测器：仅使用当前测量与上一周期预测',[350 720 950 760]);
    note(scope,'下一步预测经 Unit Delay 返回，避免代数环',[1120 910 1750 950]);
end
monitorScope=path(106);
if isempty(find_system(monitorScope,'SearchDepth',1,'Name','Safety_From'))
    from=add_block('simulink/Signal Routing/From',[monitorScope '/Safety_From'],'GotoTag','NativeSafety','Position',[30 1000 170 1030]);
    log=add_block('simulink/Sinks/To Workspace',[monitorScope '/Safety_Log'],'VariableName','lateral_safety_log','SaveFormat','Timeseries','Position',[220 1000 400 1030]);
    split=add_block('simulink/Signal Routing/Demux',[monitorScope '/Safety_Fields'],'Outputs','9','Position',[450 1030 455 1450]);
    wire(monitorScope,from,1,log,1);wire(monitorScope,from,1,split,1);
    names={'Lateral_Capacity','Sideslip','Speed_Curvature','Constraint_Slack'};ports={[1 2],[3 4],[5 6 7],[8 9]};
    titles={'实测侧向加速度 / 允许值（m/s²）','质心侧偏角 / 允许值（deg）','实际车速 / 计划目标 / 曲率上限（km/h）','QP 松弛量 / 参考可行性超限比'};
    for k=1:4
        y=1060+(k-1)*150;mux=add_block('simulink/Signal Routing/Mux',[monitorScope '/Safety_Mux_' num2str(k)],'Inputs',num2str(numel(ports{k})),'Position',[600 y-30 605 y+30]);
        for j=1:numel(ports{k}),wire(monitorScope,split,ports{k}(j),mux,j);end
        source=mux;
        if k==2,source=add_block('simulink/Math Operations/Gain',[monitorScope '/Beta_deg'],'Gain','180/pi','Position',[650 y-15 710 y+15]);wire(monitorScope,mux,1,source,1);end
        b=add_block('simulink/Sinks/Scope',[monitorScope '/Scope_' names{k}],'Position',[780 y-25 970 y+25]);wire(monitorScope,source,1,b,1);
        config=get_param(b,'ScopeConfiguration');config.OpenAtSimulationStart=false;config.TimeSpan='10';
        note(monitorScope,titles{k},[780 y-60 1250 y-30]);
    end
end
try
    set_param(m,'SimulationCommand','update');
catch problem
    disp(getReport(problem,'extended','hyperlinks','off'));
    diagnostics=sllasterror;disp(diagnostics);
    for q=1:numel(diagnostics),disp(diagnostics(q).Message);end
    rethrow(problem);
end
save_system(m);
Simulink.BlockDiagram.routeLine(find_system(scope,'FindAll','on','SearchDepth',1,'Type','Line'));
Simulink.BlockDiagram.routeLine(find_system(monitorScope,'FindAll','on','SearchDepth',1,'Type','Line'));save_system(m);
folder=fullfile(root,'results','prediction_correction');print(['-s' scope],'-dpng',fullfile(folder,'controller.png'));print(['-s' monitorScope],'-dpng',fullfile(folder,'scopes.png'));
fprintf('PREDICTION_CORRECTION_MODEL_BUILD_PASS\n');
    function p=path(sid),p=Simulink.ID.getFullName([m ':' num2str(sid)]);end
    function ch=chart(sid),ch=rt.find('-isa','Stateflow.EMChart','Path',path(sid));end
end
function b=fn(parent,name,script,pos)
b=add_block('simulink/User-Defined Functions/MATLAB Function',[parent '/' name],'Position',pos,'FontName','宋体');rt=sfroot;ch=rt.find('-isa','Stateflow.EMChart','Path',getfullname(b));ch.Script=script;
parameter(ch);drawnow;
end
function input(ch,name,port,size,type),data=datum(ch,name);data.Scope='Input';data.Port=port;typed(data,size,type);end
function output(ch,name,port,size,type),data=datum(ch,name);data.Scope='Output';data.Port=port;typed(data,size,type);end
function data=datum(ch,name),data=ch.find('-isa','Stateflow.Data','Name',name);if isempty(data),data=Stateflow.Data(ch);data.Name=name;end;end
function parameter(ch)
data=datum(ch,'matlab_mpc_parameters');data.Scope='Parameter';data.Tunable=false;
code=ch.Script;lineEnd=find(code==newline,1);if isempty(lineEnd),lineEnd=numel(code)+1;end
if ~contains(code(1:lineEnd-1),'matlab_mpc_parameters')
    closing=find(code(1:lineEnd-1)==')',1,'last');
    ch.Script=[code(1:closing-1) ',matlab_mpc_parameters' code(closing:end)];
end
end
function typed(data,size,type),data.Props.Array.Size=mat2str(size);data.Props.Type.Method='Built-in';data.Props.Type.Primitive=type;end
function wire(parent,a,ap,b,bp)
source=get_param(a,'PortHandles');dest=get_param(b,'PortHandles');old=get_param(dest.Inport(bp),'Line');if old~=-1,delete_line(old);end
add_line(parent,source.Outport(ap),dest.Inport(bp),'autorouting','on');
end
function note(parent,text,pos),a=Simulink.Annotation(parent,text);a.FontName='宋体';a.FontSize=12;a.Position=pos;end
