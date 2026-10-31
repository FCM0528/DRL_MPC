param(
    [Parameter(Mandatory=$true)][string[]]$Runs,
    [string]$OutputRoot='policies\audit\rl_convergence_20261031'
)
$ErrorActionPreference='Stop'
Push-Location (Split-Path -Parent $PSScriptRoot)
try {
    if(!(Test-Path -LiteralPath $OutputRoot)){New-Item -ItemType Directory -Path $OutputRoot -Force|Out-Null}
    $groups=@()
    foreach($run in $Runs){
        $directory=(Resolve-Path -LiteralPath $run).Path
        $ini=Get-Content (Join-Path $directory 'effective_config.ini') -Raw
        $value={param($key) $m=[regex]::Match($ini,'(?m)^'+$key+'\s*=\s*([^\s#;]+)');if(!$m.Success){throw "Missing $key in $directory"};$m.Groups[1].Value}
        $algorithm=& $value 'algorithm';$speed=& $value 'train_speed_min_kmh'
        $selected=@(Import-Csv (Join-Path $directory 'selected_validation.csv'))
        $reward=@(Import-Csv (Join-Path $directory 'training_rewards.csv'))
        $checkpoints=@(Import-Csv (Join-Path $directory 'checkpoint_validation.csv'))
        $optimization=@(Import-Csv (Join-Path $directory 'optimization_metrics.csv'))
        $warm=@(Import-Csv (Join-Path $directory 'warm_start_validation.csv'))
        $augmentedMatch=[regex]::Match($ini,'(?m)^augmented_observation\s*=\s*([^\s#;]+)')
        $augmented=if($augmentedMatch.Success){$augmentedMatch.Groups[1].Value}else{'false'}
        if($reward.Count -ne $optimization.Count -or !$reward.Count -or $selected.Count -ne 2){throw "Incomplete records in $directory"}
        $runName=Split-Path $directory -Leaf;$cycle=Split-Path (Split-Path $directory -Parent) -Leaf
        $rel="../../iterations/$algorithm/$cycle/$runName/"
        $groups += [pscustomobject]@{algorithm=$algorithm;speed=$speed;align=& $value 'align_integer_np';direct=& $value 'direct_block_reward';augmented=$augmented;warm_rmse=$warm[0].rmse_y_m;selected=$selected;reward=$reward;checkpoints=$checkpoints;optimization=$optimization;report=($rel+'rl_report.html');metrics=($rel+'optimization_metrics.csv');validation=($rel+'checkpoint_validation.csv');run=$directory}
    }
    $json=ConvertTo-Json -InputObject $groups -Depth 8 -Compress
    $html=@'
<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>固定工况强化学习收敛自查</title><style>body{font:15px 'Microsoft YaHei',sans-serif;color:#18324b;max-width:1260px;margin:28px auto;padding:0 20px}p{line-height:1.7}.note{background:#fff4df;border-left:4px solid #d99000;padding:12px}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(520px,1fr));gap:15px}.panel{border:1px solid #d5e0eb;border-radius:8px;padding:12px}svg{width:100%}table{border-collapse:collapse;width:100%;font-size:13px}td,th{border:1px solid #d7e0ea;padding:7px;text-align:right}td:first-child,th:first-child{text-align:left}th{background:#eaf1f8}a{color:#1755a0}</style>
<h1>固定车速强化学习收敛自查</h1><p>同一车速下实验使用同一车辆、道路与固定策略作为 Actor 热启动；各组保留各自 Critic 和经验池，不自动替换既有固定策略。</p><p class="note">“原始决策块奖励”仍经过比例缩放和截断；整数 Np 对齐的 Actor 梯度是直通近似。增强观测只补充部分控制器历史。检查点在同一工况上选择，并非独立泛化测试。图中网络损失或滚动均值的变化不能单独证明闭环收敛。</p>
<h2>实验与选中结果</h2><div id="summary"></div><h2>每组无探索检查点横向 RMSE</h2><p>灰虚线为固定 MPC，绿虚线为热启动 Actor；彩色线为每 10 回合的当前 Actor。最佳策略可能仍是热启动 Actor。</p><div id="validation" class="grid"></div>
<h2>训练回报</h2><p>浅蓝为每回合含探索噪声的真实回报，深蓝为 10 回合滚动均值；滚动均值只用于观察，不代表无探索策略的表现。</p><div id="rewards" class="grid"></div>
<h2>Critic 训练损失</h2><p>每回合内训练批次的平均 Huber 损失；TD3 的两条线分别为两个 Critic。不同奖励模式的损失尺度不应直接排名。</p><div id="critic" class="grid"></div>
<h2>Actor 代理目标</h2><p>每回合内批次的平均 −Q 与动作惩罚之和；梯度还存在裁剪与 Np 直通近似，不能等同于闭环路径误差。</p><div id="actor" class="grid"></div>
<script>
const groups=__DATA__;const n=x=>Number(x),f=(x,k=4)=>Number(x).toFixed(k);const color=['#2563eb','#c02645','#0f766e','#9333ea','#ca6702','#0891b2'];
const fmtBool=x=>x==='1'||x==='true';function mode(g){return (fmtBool(g.augmented)?'增强观测 / ':'')+(fmtBool(g.direct)?'直接奖励':'配对奖励')+(fmtBool(g.align)?' / Np 对齐':'')}
document.getElementById('summary').innerHTML='<table><thead><tr><th>工况</th><th>实验</th><th>回合</th><th>选中回合</th><th>固定 MPC RMSE</th><th>选中策略 RMSE</th><th>航向 RMSE °</th><th>保护步</th><th>原始数据</th></tr></thead><tbody>'+groups.map(g=>{let b=g.selected.find(r=>r.controller==='fixed'),p=g.selected.find(r=>r.controller===g.algorithm);return `<tr><td>${g.speed} km/h ${g.algorithm.toUpperCase()}</td><td>${mode(g)}</td><td>${g.reward.length}</td><td>${p.checkpoint_episode==='-1'?'热启动 Actor':p.checkpoint_episode==='0'?'中性回退':p.checkpoint_episode}</td><td>${f(b.rmse_y_m)} m</td><td>${f(p.rmse_y_m)} m</td><td>${f(p.rmse_heading_deg,3)}</td><td>${p.grip_limited_steps}</td><td><a href="${g.report}">训练报告</a> · <a href="${g.metrics}">网络指标 CSV</a> · <a href="${g.validation}">检查点 CSV</a></td></tr>`}).join('')+'</tbody></table>';
function graph(title,series,xmax,yLabel){let all=series.flatMap(s=>s.p).filter(p=>Number.isFinite(p.y));if(!all.length)return `<div class="panel"><h3>${title}</h3><p>无记录</p></div>`;let lo=Math.min(...all.map(p=>p.y)),hi=Math.max(...all.map(p=>p.y)),pad=Math.max((hi-lo)*.09,.001);lo-=pad;hi+=pad;let px=x=>70+(x-1)/(Math.max(2,xmax)-1)*440,py=y=>235-(y-lo)/(hi-lo)*185;let grid=Array.from({length:5},(_,i)=>{let y=lo+(hi-lo)*i/4;return `<line x1="70" x2="510" y1="${py(y)}" y2="${py(y)}" stroke="#e2e8f0"/><text x="63" y="${py(y)+4}" text-anchor="end" font-size="11">${f(y,2)}</text>`}).join('');let lines=series.map(s=>{let p=s.p.filter(x=>Number.isFinite(x.y));return `<polyline fill="none" stroke="${s.c}" stroke-width="${s.w||2}" ${s.dash?'stroke-dasharray="5 4"':''} points="${p.map(x=>`${px(x.x)},${py(x.y)}`).join(' ')}"/>`}).join('');let xs=[1,Math.round(xmax/2),xmax].map(x=>`<text x="${px(x)}" y="254" text-anchor="middle" font-size="11">${x}</text>`).join('');return `<div class="panel"><h3>${title}</h3><svg viewBox="0 0 540 280" role="img" aria-label="${title}">${grid}${lines}${xs}<text x="290" y="275" text-anchor="middle" font-size="12">回合</text><text transform="translate(15,145) rotate(-90)" text-anchor="middle" font-size="12">${yLabel}</text></svg></div>`}
groups.forEach((g,i)=>{let title=`${g.speed} km/h ${g.algorithm.toUpperCase()} · ${mode(g)}`,max=g.reward.length,b=n(g.selected.find(r=>r.controller==='fixed').rmse_y_m),warm=n(g.warm_rmse);let check=g.checkpoints.filter(r=>+r.episode>0).map(r=>({x:n(r.episode),y:n(r.rmse_y_m)}));let series=[{p:check,c:color[i%color.length]},{p:[{x:1,y:b},{x:max,y:b}],c:'#64748b',dash:true}];if(Number.isFinite(warm))series.push({p:[{x:1,y:warm},{x:max,y:warm}],c:'#0f766e',dash:true});document.getElementById('validation').innerHTML+=graph(title,series,max,'横向 RMSE (m)');let rewardVals=k=>g.reward.filter(r=>r[k]!==''&&r[k]!=null).map(r=>({x:n(r.episode),y:n(r[k])}));document.getElementById('rewards').innerHTML+=graph(title,[{p:rewardVals('episode_return'),c:'#93c5fd',w:1},{p:rewardVals('episode_return_10'),c:'#1d4ed8'}],max,'每回合累计回报');let vals=k=>g.optimization.filter(r=>r[k]!==''&&r[k]!=null).map(r=>({x:n(r.episode),y:n(r[k])}));let cs=[{p:vals('critic1_huber_loss'),c:'#2563eb'}];if(g.algorithm==='td3')cs.push({p:vals('critic2_huber_loss'),c:'#c02645'});document.getElementById('critic').innerHTML+=graph(title,cs,max,'Huber 损失');document.getElementById('actor').innerHTML+=graph(title,[{p:vals('actor_surrogate_objective'),c:'#0f766e'}],max,'代理目标')});
</script></html>
'@
    $html=$html.Replace('__DATA__',$json)
    [IO.File]::WriteAllText([IO.Path]::GetFullPath((Join-Path $OutputRoot 'report.html')),$html,[Text.UTF8Encoding]::new($false))
    Write-Host "Audit report: $(Join-Path $OutputRoot 'report.html')"
} finally {Pop-Location}
