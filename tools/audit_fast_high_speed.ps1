$ErrorActionPreference = 'Stop'
Push-Location (Split-Path -Parent $PSScriptRoot)
try {
    $root='policies\audit\fast_high_speed_20261030'
    if (Test-Path -LiteralPath $root) { throw 'Output already exists.' }
    $rows=@()
    foreach($speed in @(88,92,98,102,104,116)) {
        $out=Join-Path $root "v$speed"
        & .\out\build\x64-Release\Release\MPC_DoubleLaneChange.exe --config configs\carsim_focus_fast.ini --rl-mode fixed --set "carsim.simfile=policies\audit\focus60_80_source\v$speed\simfile.sim" --set carsim.sync_browser_result=false --no-open --out $out | Out-Null
        if ($LASTEXITCODE -ne 0) { throw "Run failed at $speed km/h" }
        $run=Get-ChildItem -LiteralPath $out -Directory | Select-Object -First 1
        $csv=@(Import-Csv (Join-Path $run.FullName 'simulation.csv'))
        $sum=0.0; $peak=0.0; $failure=0
        foreach($row in $csv){$e=[double]$row.ey_m;$sum+=$e*$e;$peak=[Math]::Max($peak,[Math]::Abs($e));if($row.fallback_control -eq '1'){$failure++}}
        $rows += [pscustomobject]@{speed_kmh=$speed;rmse_m=[Math]::Sqrt($sum/$csv.Count);max_error_m=$peak;fallback_steps=$failure;run=$run.FullName}
        Write-Host "$speed km/h RMSE=$($rows[-1].rmse_m), max=$peak, fallback=$failure"
    }
    $rows|Export-Csv (Join-Path $root 'metrics.csv') -NoTypeInformation
} finally {Pop-Location}
