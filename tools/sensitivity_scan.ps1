param(
    [Parameter(Mandatory=$true)][string]$SpeedKmh,
    [Parameter(Mandatory=$true)][string]$OutputRoot,
    [string]$SourcePar = 'policies\iterations\ddpg\cycle_20261030-193544-607\ddpg_train_carsim_NpNominal25_Nc8_20261030-193544\training_source_Run_all.par',
    [string]$SourceSimfile = 'policies\audit\focus60_80_source\v60\simfile.sim',
    [string]$Config = 'configs\carsim_focus_fast_guarded.ini',
    [switch]$Interactions
)
$ErrorActionPreference='Stop'
Push-Location (Split-Path -Parent $PSScriptRoot)
try {
    $culture=[cultureinfo]::InvariantCulture
    $speed=[double]::Parse($SpeedKmh,$culture)
    $speedId=$SpeedKmh.Replace('.','_')
    $speedRoot=Join-Path $OutputRoot "v$speedId"
    if (Test-Path -LiteralPath $speedRoot) { throw "Output already exists: $speedRoot" }
    $fixture=Join-Path $speedRoot 'source'
    New-Item -ItemType Directory -Path $fixture -Force | Out-Null
    $par=[IO.File]::ReadAllText((Resolve-Path -LiteralPath $SourcePar))
    if (([regex]::Matches($par,'(?m)^SPEED_TARGET_CONSTANT\s+\S+')).Count -ne 1) { throw 'Expected one constant speed target.' }
    $par=[regex]::Replace($par,'(?m)^SPEED_TARGET_CONSTANT\s+\S+',"SPEED_TARGET_CONSTANT $SpeedKmh")
    $parPath=[IO.Path]::GetFullPath((Join-Path $fixture 'Run_all.par'))
    [IO.File]::WriteAllText($parPath,$par,[Text.UTF8Encoding]::new($false))
    $sim=[IO.File]::ReadAllText((Resolve-Path -LiteralPath $SourceSimfile))
    $sim=[regex]::Replace($sim,'(?m)^INPUT [^\r\n]+',('INPUT '+$parPath))
    $simPath=[IO.Path]::GetFullPath((Join-Path $fixture 'simfile.sim'))
    [IO.File]::WriteAllText($simPath,$sim,[Text.UTF8Encoding]::new($false))
    $cases=@([pscustomobject]@{parameter='baseline';value=0;case='baseline'})
    foreach($item in @(
        @('np',@(18,22,25,30,35,40)),
        @('q_y',@(60,80,100,130,166.66666666666666)),
        @('q_psi',@(210,280,350,450,583.3333333333334)),
        @('r_delta',@(245,300,350,420,500))
    )) {
        foreach($value in $item[1]) {
            if (($item[0] -eq 'np' -and $value -eq 25) -or
                ($item[0] -eq 'q_y' -and $value -eq 100) -or
                ($item[0] -eq 'q_psi' -and $value -eq 350) -or
                ($item[0] -eq 'r_delta' -and $value -eq 350)) { continue }
            $v=([double]$value).ToString('G17',$culture)
            $cases += [pscustomobject]@{parameter=$item[0];value=$v;case=($item[0]+'_'+$v.Replace('.','_'))}
        }
    }
    if ($Interactions) {
        $cases=@()
        foreach($np in @(22,30,40)) { foreach($qy in @(60,100)) { foreach($rd in @(350,500)) {
            $cases += [pscustomobject]@{parameter='combined';value='';case="np${np}_qy${qy}_rd${rd}";np=$np;qy=$qy;rd=$rd}
        } } }
    }
    $results=@()
    foreach($c in $cases) {
        $out=Join-Path $speedRoot $c.case
        $argsList=@('--config',$Config,'--rl-mode','fixed','--set',"carsim.simfile=$simPath",'--set','carsim.sync_browser_result=false','--no-open','--out',$out)
        if($c.parameter -eq 'combined') {$argsList += @('--set',"mpc.np=$($c.np)",'--set',"weights.q_y=$($c.qy)",'--set',"weights.r_delta=$($c.rd)")}
        elseif($c.parameter -eq 'np') {$argsList += @('--set',"mpc.np=$($c.value)")}
        elseif($c.parameter -ne 'baseline') {$argsList += @('--set',"weights.$($c.parameter)=$($c.value)")}
        & .\out\build\x64-Release\Release\MPC_DoubleLaneChange.exe @argsList | Out-Null
        if($LASTEXITCODE -ne 0){throw "Run failed: $SpeedKmh $($c.case)"}
        $run=@(Get-ChildItem -LiteralPath $out -Directory)
        if($run.Count -ne 1){throw "Unexpected result count in $out"}
        $csv=@(Import-Csv -LiteralPath (Join-Path $run[0].FullName 'simulation.csv'))
        if($csv.Count -eq 0){throw "Empty simulation: $out"}
        $sq=0.0;$hsq=0.0;$peak=0.0;$qp=0;$fallback=0;$grip=0;$steer=0.0
        foreach($r in $csv){
            $e=[double]::Parse($r.ey_m,$culture);$h=[double]::Parse($r.epsi_deg,$culture)
            $sq+=$e*$e;$hsq+=$h*$h;$peak=[math]::Max($peak,[math]::Abs($e));$steer=[math]::Max($steer,[math]::Abs([double]::Parse($r.front_steer_command_deg,$culture)))
            if($r.is_command -eq '1'){
                if($r.fallback_control -eq '1'){$fallback++}
                if($r.fallback_control -eq '1' -or ([int]$r.qp_iterations -gt 0 -and $r.qp_converged -eq '0')){$qp++}
                if($r.grip_limited_control -eq '1'){$grip++}
            }
        }
        $distance=[double]::Parse($csv[-1].X_carsim_m,$culture)
        $results += [pscustomobject]@{speed_kmh=$speed;parameter=$c.parameter;value=$c.value;np=$c.np;q_y=$c.qy;r_delta=$c.rd;rmse_y_m=[math]::Sqrt($sq/$csv.Count);max_abs_y_m=$peak;rmse_heading_deg=[math]::Sqrt($hsq/$csv.Count);max_steer_deg=$steer;qp_nonconverged_steps=$qp;fallback_steps=$fallback;grip_limited_steps=$grip;completed=($distance -ge 180);distance_reached_m=$distance;run=$run[0].FullName}
        $results | Export-Csv -LiteralPath (Join-Path $speedRoot 'metrics.csv') -NoTypeInformation
        Write-Host "$SpeedKmh $($c.case) RMSE=$($results[-1].rmse_y_m) max=$peak QP=$qp"
    }
} finally {Pop-Location}
