param([Parameter(Mandatory=$true)][string]$OutputRoot, [string[]]$InteractionRoots = @())
$ErrorActionPreference='Stop'
Push-Location (Split-Path -Parent $PSScriptRoot)
try {
    $files=@(Get-ChildItem -LiteralPath $OutputRoot -Directory | ForEach-Object { Join-Path $_.FullName 'metrics.csv' } | Where-Object {Test-Path -LiteralPath $_ -PathType Leaf})
    if($files.Count -lt 1){throw 'No scan metrics found.'}
    $rows=@(foreach($file in $files){Import-Csv -LiteralPath $file})
    $rows|Export-Csv -LiteralPath (Join-Path $OutputRoot 'sensitivity_metrics.csv') -NoTypeInformation -Encoding UTF8
    $json=ConvertTo-Json -InputObject $rows -Depth 5 -Compress
    $interactionRows=@(foreach($root in $InteractionRoots){
        foreach($directory in Get-ChildItem -LiteralPath $root -Directory){
            $file=Join-Path $directory.FullName 'metrics.csv'
            if(Test-Path -LiteralPath $file -PathType Leaf){Import-Csv -LiteralPath $file}
        }
    })
    if($interactionRows.Count -gt 0){$interactionRows|Export-Csv -LiteralPath (Join-Path $OutputRoot 'interaction_metrics.csv') -NoTypeInformation -Encoding UTF8}
    $interactionJson=ConvertTo-Json -InputObject $interactionRows -Depth 5 -Compress
    $html=@'
<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>MPC 参数敏感性扫描</title>
<style>body{font:15px 'Microsoft YaHei',sans-serif;color:#18324b;max-width:1250px;margin:28px auto;padding:0 20px}h1{font-size:27px}p{line-height:1.7}.note{background:#fff5df;border-left:4px solid #d99000;padding:12px}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(520px,1fr));gap:15px}.panel{border:1px solid #d5e0eb;border-radius:8px;padding:12px}svg{width:100%}table{border-collapse:collapse;width:100%;font-size:13px}th,td{border:1px solid #d7e0ea;padding:7px;text-align:right}th:first-child,td:first-child{text-align:left}th{background:#eaf1f8}label{margin-right:15px}a{color:#1755a0}</style>
<h1>MPC 参数敏感性扫描</h1><p>同一 CarSim 道路与车辆快照；每次仅改变一个固定 MPC 参数。横向误差统计包括整段路径，最大误差和求解状态请同时查看下表。</p>
<p class="note">基准：Np=25、Qy=100、Qψ=350、RΔ=350；采样周期 0.05 s、Nc=8。其余参数按 configs/carsim_focus_fast_guarded.ini。此扫描不是强化学习训练，也不能证明这些参数在其他附着、道路或车速下有效。车速是目标设定值。</p>
<p>道路附着表：0 m 起 0.9、100 m 起 0.6、200 m 起 0.8。可点击图例筛选车速；鼠标悬停在点上查看数值。<a href="sensitivity_metrics.csv">下载完整 CSV</a></p><div id="legend"></div><div id="charts" class="grid"></div>
<h2>每个速度的基准与单变量最优结果</h2><p>“最优”仅指本扫描取值中横向 RMSE 最小，不代表最大误差、航向误差或其他工况同时最优。</p><div id="table"></div>
<h2>Np、Qy 与 RΔ 的组合扫描</h2><p>Np ∈ {22, 30, 40}，Qy ∈ {60, 100}，RΔ ∈ {350, 500}，Qψ 固定为 350。可查看每个速度的组合最优与完整数据。<a href="interaction_metrics.csv">下载组合扫描 CSV</a></p><div id="combo_best"></div><h3>全部组合</h3><div id="combos"></div>
<h2>全部扫描记录</h2><div id="all"></div>
<script>
const rows=__DATA__;
const combos=__INTERACTION_DATA__;
const specs=[['np','预测时域 Np',25],['q_y','横向误差权重 Qy',100],['q_psi','航向误差权重 Qψ',350],['r_delta','转角增量权重 RΔ',350]];
const speeds=[...new Set(rows.map(r=>+r.speed_kmh))].sort((a,b)=>a-b),colors=['#2563eb','#be123c','#0f766e','#9333ea','#ca6702'];
const active=new Set(speeds);const fmt=(x,n=4)=>Number(x).toFixed(n);const sname=x=>Number(x).toFixed(2).replace(/\.00$/,'');
function points(speed,key,base){let a=rows.filter(r=>+r.speed_kmh===speed&&(r.parameter===key||r.parameter==='baseline')).map(r=>({...r,x:r.parameter==='baseline'?base:+r.value}));return a.sort((a,b)=>a.x-b.x)}
function draw(){document.getElementById('legend').innerHTML=speeds.map((s,i)=>`<label><input type="checkbox" data-s="${s}" ${active.has(s)?'checked':''}><span style="color:${colors[i]}">● ${sname(s)} km/h</span></label>`).join('');document.querySelectorAll('[data-s]').forEach(e=>e.onchange=()=>{e.checked?active.add(+e.dataset.s):active.delete(+e.dataset.s);draw()});
document.getElementById('charts').innerHTML=specs.map(([key,title,base])=>{const series=speeds.filter(s=>active.has(s)).map(s=>({s,p:points(s,key,base)}));let all=series.flatMap(a=>a.p);if(!all.length)return `<div class="panel">${title}：请选择速度</div>`;let xmin=Math.min(...all.map(r=>r.x)),xmax=Math.max(...all.map(r=>r.x)),ymax=Math.max(...all.map(r=>+r.rmse_y_m))*1.1;let px=x=>70+(x-xmin)/(xmax-xmin||1)*440,py=y=>245-y/ymax*200;let grid=Array.from({length:5},(_,i)=>{let y=i*ymax/4;return `<line x1="70" x2="510" y1="${py(y)}" y2="${py(y)}" stroke="#e2e8f0"/><text x="64" y="${py(y)+4}" text-anchor="end" font-size="11">${fmt(y,2)}</text>`}).join('');let lines=series.map(({s,p})=>{let color=colors[speeds.indexOf(s)],line=p.map(r=>`${px(r.x)},${py(+r.rmse_y_m)}`).join(' ');return `<polyline fill="none" stroke="${color}" stroke-width="2.5" points="${line}"/>`+p.map(r=>`<circle cx="${px(r.x)}" cy="${py(+r.rmse_y_m)}" r="4" fill="${color}"><title>${sname(s)} km/h，${title}=${fmt(r.x,2)}，RMSE=${fmt(r.rmse_y_m)} m，最大误差=${fmt(r.max_abs_y_m)} m</title></circle>`).join('')}).join('');let ticks=[0,.25,.5,.75,1].map(f=>{let x=xmin+f*(xmax-xmin);return `<text x="${px(x)}" y="268" text-anchor="middle" font-size="11">${fmt(x,1)}</text>`}).join('');return `<div class="panel"><h3>${title}</h3><svg viewBox="0 0 540 295" role="img" aria-label="${title} 与横向 RMSE">${grid}<line x1="70" x2="510" y1="245" y2="245" stroke="#789"/><line x1="${px(base)}" x2="${px(base)}" y1="45" y2="245" stroke="#999" stroke-dasharray="4 4"/>${lines}${ticks}<text x="285" y="290" text-anchor="middle" font-size="12">${title}（虚线为基准）</text><text transform="translate(15,160) rotate(-90)" text-anchor="middle" font-size="12">横向 RMSE (m)</text></svg></div>`}).join('');}
function table(rs){return '<table><thead><tr><th>速度 km/h</th><th>参数</th><th>取值</th><th>横向 RMSE m</th><th>最大横向误差 m</th><th>航向 RMSE °</th><th>QP 未收敛步</th><th>保护步</th><th>完成</th></tr></thead><tbody>'+rs.map(r=>`<tr><td>${sname(r.speed_kmh)}</td><td>${r.parameter}</td><td>${r.parameter==='baseline'?'基准':fmt(r.value,2)}</td><td>${fmt(r.rmse_y_m)}</td><td>${fmt(r.max_abs_y_m)}</td><td>${fmt(r.rmse_heading_deg)}</td><td>${r.qp_nonconverged_steps}</td><td>${r.grip_limited_steps}</td><td>${r.completed}</td></tr>`).join('')+'</tbody></table>'}
const best=[];for(const s of speeds){best.push(rows.find(r=>+r.speed_kmh===s&&r.parameter==='baseline'));for(const [key,,base] of specs){const candidates=points(s,key,base).filter(r=>r.completed==='True'&&+r.qp_nonconverged_steps===0);if(candidates.length){const winner=candidates.reduce((a,b)=>+a.rmse_y_m<+b.rmse_y_m?a:b);best.push({...winner,parameter:key,value:winner.x})}}}document.getElementById('table').innerHTML=table(best);document.getElementById('all').innerHTML=table(rows);draw();
function comboTable(rs){return '<table><thead><tr><th>速度 km/h</th><th>Np</th><th>Qy</th><th>RΔ</th><th>横向 RMSE m</th><th>最大横向误差 m</th><th>航向 RMSE °</th><th>QP 未收敛步</th><th>保护步</th></tr></thead><tbody>'+rs.map(r=>`<tr><td>${sname(r.speed_kmh)}</td><td>${r.np}</td><td>${r.q_y}</td><td>${r.r_delta}</td><td>${fmt(r.rmse_y_m)}</td><td>${fmt(r.max_abs_y_m)}</td><td>${fmt(r.rmse_heading_deg)}</td><td>${r.qp_nonconverged_steps}</td><td>${r.grip_limited_steps}</td></tr>`).join('')+'</tbody></table>'}
combos.sort((a,b)=>+a.speed_kmh-+b.speed_kmh||+a.np-+b.np||+a.q_y-+b.q_y||+a.r_delta-+b.r_delta);const bestCombos=speeds.map(s=>combos.filter(r=>+r.speed_kmh===s&&r.completed==='True'&&+r.qp_nonconverged_steps===0)).filter(a=>a.length).map(a=>a.reduce((x,y)=>+x.rmse_y_m<+y.rmse_y_m?x:y));document.getElementById('combo_best').innerHTML=comboTable(bestCombos);document.getElementById('combos').innerHTML=comboTable(combos);
</script></html>
'@
    $html=$html.Replace('__DATA__',$json)
    $html=$html.Replace('__INTERACTION_DATA__',$interactionJson)
    [IO.File]::WriteAllText([IO.Path]::GetFullPath((Join-Path $OutputRoot 'report.html')),$html,[Text.UTF8Encoding]::new($false))
    Write-Host "Report: $(Join-Path $OutputRoot 'report.html')"
} finally {Pop-Location}
