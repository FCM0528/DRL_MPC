param(
    [Parameter(Mandatory=$true)][string]$DdpgPolicy,
    [Parameter(Mandatory=$true)][string]$Td3Policy,
    [Parameter(Mandatory=$true)][string]$OutputRoot,
    [string]$Config = 'configs\default.ini',
    [switch]$AllowModelMismatch,
    [string]$SourceDirectory = 'policies\experiments\variable_road_80_fixture',
    [switch]$FinalHoldout
)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Push-Location $projectRoot
try {
    if (Test-Path -LiteralPath $OutputRoot) { throw "Output already exists: $OutputRoot" }
    foreach ($file in @($DdpgPolicy, $Td3Policy, (Join-Path $SourceDirectory 'simfile.sim'), (Join-Path $SourceDirectory 'Run_all.par'))) {
        if (-not (Test-Path -LiteralPath $file -PathType Leaf)) { throw "Required file missing: $file" }
    }
    $sourcePar = @(Get-Content -LiteralPath (Join-Path $SourceDirectory 'Run_all.par'))
    if (@($sourcePar | Where-Object { $_ -match '^\s*SPEED_TARGET_CONSTANT\s' }).Count -ne 1 -or
        @($sourcePar | Where-Object { $_ -match '^\s*MU_ROAD_TABLE\s+STEP\s*$' }).Count -ne 1) {
        throw 'Expected exactly one constant speed target and one STEP road friction table.'
    }
    $profiles = @(
        @{ Name='original'; Code='a'; Rows=@('0, 0.9','100, 0.6','200, 0.8') },
        @{ Name='lower_early_drop'; Code='b'; Rows=@('0, 0.75','85, 0.5','160, 0.8') },
        @{ Name='low_to_high'; Code='c'; Rows=@('0, 0.5','110, 0.8','190, 0.65') }
    )
    if ($FinalHoldout) {
        $profiles += @(
            @{ Name='holdout_rise'; Code='d'; Rows=@('0, 0.65','95, 0.85','175, 0.7') },
            @{ Name='holdout_late_drop'; Code='e'; Rows=@('0, 0.85','120, 0.55','205, 0.75') }
        )
    }
    New-Item -ItemType Directory -Path $OutputRoot | Out-Null
    $root = (Resolve-Path -LiteralPath $OutputRoot).Path
    $rows = @()
    foreach ($profile in $profiles) {
        foreach ($speed in @(60,80)) {
            $case = $profile.Code + $speed
            $fixture = Join-Path $root (Join-Path 'f' $case)
            New-Item -ItemType Directory -Force -Path $fixture | Out-Null
            $newPar = [System.Collections.Generic.List[string]]::new()
            $inTable = $false; $ended = $false
            foreach ($line in $sourcePar) {
                if ($line -match '^\s*MU_ROAD_TABLE\s+STEP\s*$') {
                    $newPar.Add($line); foreach ($item in $profile.Rows) { $newPar.Add($item) }
                    $inTable = $true; continue
                }
                if ($inTable) {
                    if ($line -match '^\s*ENDTABLE\s*$') { $newPar.Add('ENDTABLE'); $inTable=$false; $ended=$true }
                    continue
                }
                if ($line -match '^\s*SPEED_TARGET_CONSTANT\s') { $newPar.Add("SPEED_TARGET_CONSTANT $speed") }
                else { $newPar.Add($line) }
            }
            if ($inTable -or -not $ended) { throw 'Unterminated road table.' }
            $parPath = Join-Path $fixture 'Run_all.par'
            [System.IO.File]::WriteAllLines($parPath, $newPar, [System.Text.Encoding]::ASCII)
            $simPath = Join-Path $fixture 'simfile.sim'
            $simLines = @(Get-Content -LiteralPath (Join-Path $SourceDirectory 'simfile.sim') | ForEach-Object {
                if ($_ -match '^INPUT\s') { 'INPUT ' + $parPath } else { $_ }
            })
            [System.IO.File]::WriteAllLines($simPath, [string[]]$simLines, [System.Text.Encoding]::ASCII)
            $caseOut = Join-Path $root (Join-Path 'r' $case)
            $compareArgs=@{ DdpgPolicy=$DdpgPolicy; Td3Policy=$Td3Policy; Simfile=$simPath; OutputRoot=$caseOut; Config=$Config; NoOpen=$true }
            if($AllowModelMismatch){$compareArgs.AllowModelMismatch=$true}
            & (Join-Path $PSScriptRoot 'compare_three_controllers.ps1') @compareArgs | Out-Host
            $run = @(Get-ChildItem -LiteralPath $caseOut -Directory)
            if ($run.Count -ne 1) { throw "Expected one comparison run for $case" }
            foreach ($metric in @(Import-Csv -LiteralPath (Join-Path $run[0].FullName 'comparison_metrics.csv'))) {
                $rows += [pscustomobject]@{ scenario=$profile.Name; speed_kmh=$speed; controller=$metric.controller;
                    completed=$metric.completed; rmse_y_m=$metric.rmse_y_m; max_abs_y_m=$metric.max_abs_y_m;
                    report=('r/' + $case + '/' + $run[0].Name + '/comparison_report.html') }
            }
        }
    }
    $rows | Export-Csv -LiteralPath (Join-Path $root 'target_scenario_metrics.csv') -NoTypeInformation -Encoding UTF8
    $enc = [System.Net.WebUtility]
    $html = @('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>目标工况对比</title>',
      '<style>body{font:16px Arial;max-width:1100px;margin:32px auto}table{border-collapse:collapse;width:100%}th,td{border:1px solid #bbb;padding:8px}th{background:#eef4fa}</style>',
      '<h1>60 / 80 km/h 与不同附着道路</h1><p>每组链接包含三控制器完整对比图。original 为原道路，另外两组道路改变了附着及切换位置；是否独立于训练应核对训练工况。每组使用同一车辆和轨迹。</p>',
      '<table><tr><th>道路</th><th>车速 km/h</th><th>控制器</th><th>完成</th><th>横向 RMSE m</th><th>最大横向误差 m</th><th>报告</th></tr>')
    foreach ($row in $rows) {
        $html += '<tr><td>'+$enc::HtmlEncode($row.scenario)+'</td><td>'+$row.speed_kmh+'</td><td>'+$enc::HtmlEncode($row.controller)+'</td><td>'+$row.completed+'</td><td>'+$row.rmse_y_m+'</td><td>'+$row.max_abs_y_m+'</td><td><a href="'+$enc::HtmlEncode($row.report)+'">查看对比图</a></td></tr>'
    }
    $html += '</table></html>'
    [System.IO.File]::WriteAllLines((Join-Path $root 'target_scenario_report.html'), [string[]]$html, [System.Text.UTF8Encoding]::new($false))
    Write-Host "Target scenario report: $(Join-Path $root 'target_scenario_report.html')"
} finally { Pop-Location }
