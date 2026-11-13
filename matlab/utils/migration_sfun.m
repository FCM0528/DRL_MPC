function migration_sfun(block)
% Normal-mode MATLAB adapter around the unchanged C++ prediction/QP core.
block.NumDialogPrms=1;
mode=block.DialogPrm(1).Data;
switch mode
    case 'controller', widths=[20 4]; outs=9; ts=.04;
    case 'reference', widths=20; outs=157; ts=.04;
    case 'errors', widths=[20 157]; outs=2; ts=.04;
    case 'observation', widths=[20 9]; outs=14; ts=.08;
    case 'reward', widths=[20 9]; outs=[1 1]; ts=.04;
    case 'action', widths=4; outs=4; ts=.08;
    case 'operatingpoint',widths=[20 4];outs=1;ts=.04;
    case {'linearization','reference_stage','cost','constraints','qp'},widths=1;outs=1;ts=.04;
    case 'steering',widths=1;outs=9;ts=.04;
end
block.NumInputPorts=numel(widths); block.NumOutputPorts=numel(outs);
block.SetPreCompInpPortInfoToDynamic; block.SetPreCompOutPortInfoToDynamic;
for k=1:numel(widths)
    block.InputPort(k).Dimensions=widths(k); block.InputPort(k).DatatypeID=0;
    block.InputPort(k).DirectFeedthrough=true;
end
for k=1:numel(outs)
    block.OutputPort(k).Dimensions=outs(k); block.OutputPort(k).DatatypeID=0;
end
block.SampleTimes=[ts 0]; block.SimStateCompliance='DefaultSimState';
block.RegBlockMethod('Start',@start);
block.RegBlockMethod('Outputs',@outputs);
end
function start(b)
u=struct('time',-inf,'value',[],'prev',0,'increment',0,'vx',nan, ...
    'tuning',mpc_mex('map',zeros(4,1)),'reward',0,'count',0,'npfail',0);
set_param(b.BlockHandle,'UserData',u);
if any(strcmp(b.DialogPrm(1).Data,{'controller','operatingpoint'})),mpc_mex('reset');end
end
function outputs(b)
u=get_param(b.BlockHandle,'UserData');
if b.CurrentTime==u.time
    for k=1:b.NumOutputPorts,b.OutputPort(k).Data=u.value{k};end
    return
end
s=b.InputPort(1).Data; mode=b.DialogPrm(1).Data;
switch mode
    case 'controller'
        if isnan(u.vx),s(7)=0;else,s(7)=(s(3)-u.vx)/.04;end
        d=mpc_mex('step',s,b.InputPort(2).Data,u.prev,u.increment);
        u.increment=d(1)-u.prev;u.prev=d(1);u.vx=s(3); v={d};
    case 'reference'
        ref=mpc_mex('reference',s(1));p=zeros(3,50);
        xd=max(0,s(3)*cos(s(5))-s(4)*sin(s(5)));
        for k=1:50
            f=mpc_mex('reference',s(1)+k*.04*xd);p(:,k)=f(1:3);
        end
        v={[ref;p(:)]};
    case 'errors'
        r=b.InputPort(2).Data;v={[s(2)-r(1);atan2(sin(s(5)-r(2)),cos(s(5)-r(2)))]};
    case 'observation'
        d=b.InputPort(2).Data;v={mpc_mex('observe',s,d(6),d(1))};
    case 'action'
        v={mpc_mex('map',s)};
    case 'operatingpoint'
        v={mpc_mex('stage_operatingpoint',s,b.InputPort(2).Data)};
    case 'reference_stage'
        v={mpc_mex('stage_reference',s)};
    case {'linearization','cost','constraints','qp','steering'}
        v={mpc_mex(['stage_' mode],s)};
    case 'reward'
        d=b.InputPort(2).Data;
        if b.CurrentTime>0
            u.reward=u.reward+mpc_mex('reward',s,d(1),u.prev,d(6:9),u.tuning);
            u.count=u.count+1;u.npfail=u.npfail+(d(2)==0);
        end
        failed=any(~isfinite(s))||abs(s(2))>20;
        done=(failed||s(1)>=150||b.CurrentTime>=19.999) && mod(round(b.CurrentTime/.04),2)==0;
        value=u.reward/(max(1,u.count)*10)-u.npfail*25;
        if failed,value=value-100;elseif done&&s(1)<150,value=value-100;end
        if mod(round(b.CurrentTime/.04),2)==0
            u.reward=0;u.count=0;u.npfail=0;
        end
        u.prev=d(1);u.tuning=d(6:9);v={value,double(done)};
end
u.time=b.CurrentTime;u.value=v;set_param(b.BlockHandle,'UserData',u);
for k=1:numel(v),b.OutputPort(k).Data=v{k};end
end
