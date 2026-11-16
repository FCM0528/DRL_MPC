function p=matlabCorrectionParameters(p,root)
% Physical data and explicitly configurable correction/safety design choices.
echo=fileread(fullfile(root,'carsim','case','LastRun_echo.par'));
p.correctionEnabled=false;p.safetyEnabled=false;p.observerEnabled=false;
p.cgHeight=number('H_CG_TL')/1000;p.trackFront=number('L_TRACK(1)');p.trackRear=number('L_TRACK(2)');
if p.trackFront>10,p.trackFront=p.trackFront/1000;p.trackRear=p.trackRear/1000;end
p.relaxation=splineTable('L_RELAX_Y_TABLE(1,1)');p.relaxationLoad=splineTable('L_RELAX_Y_FZ_TABLE(1,1)');
p.relaxationGain=number('L_RELAX_Y_GAIN(1,1)');p.relaxationLoadScale=number('FZ_SCALE_L_RELAX_Y(1,1)');
p.integrationSubsteps=10;p.forceGain=[1;1];p.relaxationScale=[1;1];p.loadTransfer=false;p.forceBiasTime=.08;
p.steerCompliance=0;
p.consistentFeedforward=false;
p.observerGain=[.03;0;.03];p.observerBound=[2;0;.6];p.disturbanceTime=.08;
p.safetyUtilization=.95;p.speedUtilization=.85;p.betaLimit=5*pi/180;p.slipUtilization=.95;
p.designSteerRate=120*pi/180;p.slackQuadratic=1e7;p.slackLinear=1e5;
p.speedDeceleration=2;p.speedAcceleration=1;p.speedLookahead=20;p.nominalSpeed=70/3.6;
p.speedPlan=[(-10:1:200)',repmat(p.nominalSpeed,211,1)];
p.speedPlanActive=false;
peakF=0;peakR=0;angleF=0;angleR=0;
for k=1:numel(p.tireSlip)
    f=matlabTireForce(p.tireSlip(k)*pi/180,p.frontLoad,p);r=matlabTireForce(p.tireSlip(k)*pi/180,p.rearLoad,p);
    if f>peakF,peakF=f;angleF=p.tireSlip(k)*pi/180;end
    if r>peakR,peakR=r;angleR=p.tireSlip(k)*pi/180;end
end
p.baseLateralCapacity=2*(peakF+peakR)/p.mass;p.peakSlip=[angleF;angleR];
    function value=number(key)
        token=regexp(echo,['(?m)^\s*!?\s*' regexptranslate('escape',key) '\s+([-+0-9.eE]+)(?:\s|$)'],'tokens');
        assert(isscalar(token),'Missing physical parameter: %s',key);value=str2double(token{1}{1});
    end
    function table=splineTable(key)
        token=regexp(echo,['(?ms)^\s*' regexptranslate('escape',key) '[^\r\n]*\r?\n(.*?)^\s*ENDTABLE'],'tokens');
        assert(isscalar(token));lines=splitlines(string(token{1}{1}));xy=zeros(numel(lines),2);count=0;
        for j=1:numel(lines),row=sscanf(char(lines(j)),'%f , %f');if numel(row)==2,count=count+1;xy(count,:)=row';end;end
        xy=xy(1:count,:);n=count;a=zeros(n);b=zeros(n,1);a(1,1)=1;a(n,n)=1;
        for j=2:n-1,h0=xy(j,1)-xy(j-1,1);h1=xy(j+1,1)-xy(j,1);a(j,j-1:j+1)=[h0,2*(h0+h1),h1];b(j)=6*((xy(j+1,2)-xy(j,2))/h1-(xy(j,2)-xy(j-1,2))/h0);end
        table=[xy,a\b];
    end
end
