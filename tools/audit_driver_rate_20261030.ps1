param([string]$OutputRoot = 'policies\audit\driver_rate_schedule_20261030')
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
Push-Location $root
try {
    if (Test-Path -LiteralPath $OutputRoot) { throw 'Output root already exists.' }
    $exe = '.\out\build\x64-Release\Release\MPC_DoubleLaneChange.exe'
    $rows = @()
    foreach ($speed in @(80,85,90,95,100,105,110,120)) {
        foreach ($variant in @('old','end90','end95','end100')) {
            $sim = "policies\audit\focus60_80_source\v$speed\simfile.sim"
            $out = Join-Path $OutputRoot "v${speed}_$variant"
            $argsList = @('--config','configs\carsim_focus_60_80.ini','--rl-mode','fixed',
                '--set',"carsim.simfile=$sim",'--set','carsim.sync_browser_result=false','--no-open','--out',$out)
            if ($variant -ne 'old') {
                $end = $variant.Substring(3)
                $argsList += @('--set','constraints.max_steer_rate_deg_s=65','--set',"mpc.speed_rate_end_kmh=$end")
            }
            & $exe @argsList | Out-Null
            if ($LASTEXITCODE -ne 0) { throw "Simulation failed: $speed $variant" }
            $run = Get-ChildItem -LiteralPath $out -Directory | Select-Object -First 1
            $csv = @(Import-Csv (Join-Path $run.FullName 'simulation.csv'))
            $sum=0.0; $peak=0.0; $qp=0; $grip=0
            foreach ($row in $csv) {
                $e=[double]$row.ey_m; $sum += $e*$e; $peak=[Math]::Max($peak,[Math]::Abs($e))
                if ($row.fallback_control -eq '1') { $qp++ }
                if ($row.grip_limited_control -eq '1') { $grip++ }
            }
            $rows += [pscustomobject]@{speed_kmh=$speed;variant=$variant;rmse_m=[Math]::Sqrt($sum/$csv.Count);max_error_m=$peak;fallback_steps=$qp;grip_steps=$grip;run=$run.FullName}
            Write-Host "$speed $variant RMSE=$($rows[-1].rmse_m) max=$peak fallback=$qp grip=$grip"
        }
    }
    $rows | Export-Csv -LiteralPath (Join-Path $OutputRoot 'metrics.csv') -NoTypeInformation
} finally { Pop-Location }
