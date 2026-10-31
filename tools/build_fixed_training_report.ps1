param([string]$Output='policies\fixed_speed\training_curves.html')
$ErrorActionPreference='Stop'
Push-Location (Split-Path -Parent $PSScriptRoot)
try {
    $groups=@()
    foreach($speed in @(60,80)){foreach($algorithm in @('ddpg','td3')){
        $base="policies\fixed_speed\v$speed\$algorithm"
        $meta=Import-Csv (Join-Path $base 'pinned_policy.csv')|Select-Object -First 1
        $episodes=@(Import-Csv (Join-Path $meta.training_run 'training_rewards.csv'))
        $checks=@(Import-Csv (Join-Path $meta.training_run 'checkpoint_summary.csv'))
        $selected=@(Import-Csv (Join-Path $base 'selected_validation.csv'))
        if($episodes.Count -ne 200 -or $episodes[0].episode -ne '1' -or $episodes[-1].episode -ne '200'){throw "Incomplete training log: $speed $algorithm"}
        if(@($episodes|Where-Object {[math]::Abs([double]$_.speed_kmh-$speed) -gt 1e-6}).Count){throw "Unexpected speed: $speed $algorithm"}
        $policy=Join-Path $base ($algorithm+'_policy.txt')
        if((Get-FileHash $policy -Algorithm SHA256).Hash -ne $meta.sha256){throw "Policy checksum mismatch: $speed $algorithm"}
        $sourcePolicy=Join-Path $meta.training_run ($algorithm+'_policy.txt')
        if((Get-FileHash $sourcePolicy -Algorithm SHA256).Hash -ne $meta.sha256){throw "Source policy mismatch: $speed $algorithm"}
        $selectedName=if($meta.checkpoint_episode -eq '-1'){'warm_start_rebased_policy.txt'}elseif($meta.checkpoint_episode -eq '0'){'best_validation_policy.txt'}else{'checkpoint_'+([int]$meta.checkpoint_episode).ToString('D4')+'_policy.txt'}
        if((Get-FileHash (Join-Path $meta.training_run $selectedName) -Algorithm SHA256).Hash -ne $meta.sha256){throw "Selected checkpoint mismatch: $speed $algorithm"}
        $runName=Split-Path $meta.training_run -Leaf
        $cycle=Split-Path (Split-Path $meta.training_run -Parent) -Leaf
        $rel="../iterations/$algorithm/$cycle/$runName/"
        $groups += [pscustomobject]@{speed=$speed;algorithm=$algorithm;checkpoint=$meta.checkpoint_episode;status=$meta.status;sha256=$meta.sha256;episodes=$episodes;checks=$checks;selected=$selected;report=($rel+'rl_report.html');csv=($rel+'training_rewards.csv');checkpoint_csv=($rel+'checkpoint_summary.csv')}
    }}
    $json=ConvertTo-Json -InputObject $groups -Depth 8 -Compress
    $html=@'
<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>固定车速 DDPG/TD3 训练曲线核查</title><style>body{font:15px 'Microsoft YaHei',sans-serif;color:#18324b;max-width:1260px;margin:28px auto;padding:0 20px}p{line-height:1.7}h1{font-size:26px}.note{background:#fff5df;border-left:4px solid #d99000;padding:12px}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(530px,1fr));gap:16px}.panel{border:1px solid #d5e0eb;border-radius:8px;padding:12px}svg{width:100%}table{border-collapse:collapse;width:100%;font-size:13px}th,td{border:1px solid #d7e0ea;padding:8px;text-align:right}th:first-child,td:first-child{text-align:left}th{background:#eaf1f8}a{color:#1755a0}.muted{color:#64748b}</style>
<h1>60 与 80 km/h：DDPG / TD3 训练记录核查</h1><p>四次训练均为 200 回合；每次训练固定目标车速、车辆与道路快照。策略每 0.2 s 调参，动作包括 Np、Qy、Qψ、RΔ。下表及曲线来自各次原始 CSV，并校验了固定策略与训练所选策略的 SHA-256。</p>
<p class="note">“继承 Actor”表示本轮所有检查点均未超过热启动初始策略；它不是本轮 200 回合新训练出的最优检查点。每组只含一个训练种子；同一确定性场景上的奖励曲线无法证明泛化或稳定收敛，也不能认为每个指标均更好。不同车速的奖励尺度不同，应分别看图。</p>
<h2>结果与原始文件</h2><div id="summary"></div><h2>每步训练奖励</h2><p>浅线为各回合带探索噪声的平均每步奖励，深线为最近 10 回合的算术平均；越高越好。图表没有人为单调化，逐回合原始数值可从 CSV 查看。</p><div id="reward" class="grid"></div>
<h2>每 10 回合的无探索验证</h2><p>横轴是检查点回合，纵轴是验证平均每步奖励相对于固定 MPC 的差值，虚线零表示固定 MPC；绿点为当时被选中的检查点，红点未选中。选中还须满足横向、航向、附着保护和 QP 门槛；初始继承 Actor 不对应训练回合，故不画成第 0 回合的学习结果。</p><div id="validation" class="grid"></div>
<p class="note">本批原始程序没有记录 Actor/Critic 网络损失，因此此处不绘制损失曲线，也不能从这些文件补算真实网络损失。可以从每组训练报告查看更多曲线及配置。</p>
<script>
const groups=__DATA__;
const n=x=>Number(x),f=(x,k=3)=>Number(x).toFixed(k),color=a=>a==='ddpg'?'#c02645':'#7040c7';
const selectedName=g=>g.status==='inherited_actor'?'继承 Actor':g.status==='neutral_fallback'?'中性回退':'第 '+g.checkpoint+' 回合';
function rel(g){const fixed=g.selected.find(x=>x.controller==='fixed'),agent=g.selected.find(x=>x.controller===g.algorithm);return {fixed,agent}}
document.getElementById('summary').innerHTML='<table><thead><tr><th>工况</th><th>记录回合</th><th>固定策略来源</th><th>被选中后续检查点</th><th>固定 MPC RMSE</th><th>策略 RMSE</th><th>航向 RMSE °</th><th>附着保护步</th><th>原始文件</th></tr></thead><tbody>'+groups.map(g=>{let x=rel(g),accept=g.checks.filter(c=>c.accepted==='1').map(c=>c.episode).join('、')||'无';return `<tr><td>${g.speed} km/h ${g.algorithm.toUpperCase()}</td><td>${g.episodes.length}</td><td>${selectedName(g)}</td><td>${accept}</td><td>${f(x.fixed.rmse_y_m,4)} m</td><td>${f(x.agent.rmse_y_m,4)} m</td><td>${f(x.agent.rmse_heading_deg,3)}</td><td>${x.agent.grip_limited_steps}</td><td><a href="${g.report}">原训练报告</a> · <a href="${g.csv}">逐回合 CSV</a> · <a href="${g.checkpoint_csv}">检查点 CSV</a></td></tr>`}).join('')+'</tbody></table>';
function graph(title,series,zero=false){let pts=series.flatMap(s=>s.points).filter(p=>Number.isFinite(p.y));let lo=Math.min(...pts.map(p=>p.y)),hi=Math.max(...pts.map(p=>p.y));if(zero){lo=Math.min(lo,0);hi=Math.max(hi,0)}let pad=Math.max((hi-lo)*.09,.001);lo-=pad;hi+=pad;const px=x=>70+(x-1)/199*470,py=y=>230-(y-lo)/(hi-lo)*180;let grid=Array.from({length:5},(_,i)=>{let y=lo+(hi-lo)*i/4;return `<line x1="70" x2="540" y1="${py(y)}" y2="${py(y)}" stroke="#e2e8f0"/><text x="63" y="${py(y)+4}" text-anchor="end" font-size="11">${f(y,2)}</text>`}).join('');let content=series.map(s=>{let points=s.points.filter(p=>Number.isFinite(p.y));let line=s.line?`<polyline fill="none" stroke="${s.color}" stroke-width="${s.width||2}" opacity="${s.opacity||1}" points="${points.map(p=>`${px(p.x)},${py(p.y)}`).join(' ')}"/>`:'';let circles=s.dots?points.map(p=>`<circle cx="${px(p.x)}" cy="${py(p.y)}" r="4" fill="${p.accepted?'#059669':'#dc2626'}"><title>第 ${p.x} 回合：奖励差 ${f(p.y,4)}；${p.accepted?'选中':'未选中'}</title></circle>`).join(''):'';return line+circles}).join('');let z=zero?`<line x1="70" x2="540" y1="${py(0)}" y2="${py(0)}" stroke="#666" stroke-dasharray="4 4"/>`:'';let xs=[1,50,100,150,200].map(x=>`<text x="${px(x)}" y="250" text-anchor="middle" font-size="11">${x}</text>`).join('');return `<div class="panel"><h3>${title}</h3><svg viewBox="0 0 565 275" role="img" aria-label="${title}">${grid}${z}${content}${xs}<text x="305" y="270" text-anchor="middle" font-size="12">回合</text></svg></div>`}
for(const g of groups){let c=color(g.algorithm),ep=g.episodes.map(r=>({x:n(r.episode),y:n(r.mean_step_reward)}));let avg=ep.map((p,i)=>({x:p.x,y:ep.slice(Math.max(0,i-9),i+1).reduce((a,b)=>a+b.y,0)/Math.min(10,i+1)}));document.getElementById('reward').innerHTML+=graph(`${g.speed} km/h · ${g.algorithm.toUpperCase()} · 平均每步奖励`,[{points:ep,line:true,color:c,width:1,opacity:.27},{points:avg,line:true,color:c,width:2.5}]);let check=g.checks.map(r=>({x:n(r.episode),y:n(r.mean_reward_advantage_per_step),accepted:r.accepted==='1'}));document.getElementById('validation').innerHTML+=graph(`${g.speed} km/h · ${g.algorithm.toUpperCase()} · 验证奖励差`,[{points:check,line:true,dots:true,color:c,width:1.5}],true)}
</script></html>
'@
    $html=$html.Replace('__DATA__',$json)
    $target=[IO.Path]::GetFullPath($Output)
    [IO.File]::WriteAllText($target,$html,[Text.UTF8Encoding]::new($false))
    Write-Host "Training curves: $target"
} finally {Pop-Location}
