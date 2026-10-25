param(
    [string]$OutputRoot = 'results/absolute_v15_five_seeds',
    [int[]]$Seeds = @(101, 102, 103, 104, 105)
)

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$executable = Join-Path $projectRoot 'out/build/x64-Debug/MPC_DoubleLaneChange.exe'
$root = [System.IO.Path]::GetFullPath((Join-Path $projectRoot $OutputRoot))
$sourceRun = Get-ChildItem -LiteralPath (Join-Path $projectRoot 'results/paper_model_five_seeds_frozen_source') -Directory |
    Sort-Object LastWriteTime -Descending | Select-Object -First 1
if (-not $sourceRun) { throw 'Frozen CarSim source is missing' }
$sourceSimfile = Join-Path $sourceRun.FullName 'carsim_native/simfile.sim'
if (-not (Test-Path -LiteralPath $executable -PathType Leaf) -or
    -not (Test-Path -LiteralPath $sourceSimfile -PathType Leaf)) {
    throw 'Executable or frozen CarSim Simfile is missing'
}
$manifest = Join-Path $root 'heldout_runs.csv'
"seed,exit_code,run_directory" | Set-Content -LiteralPath $manifest -Encoding utf8
$jobs = @()
foreach ($seed in $Seeds) {
    $policies = @(Get-ChildItem -LiteralPath (Join-Path $root "seed_$seed") -Directory |
        ForEach-Object { Join-Path $_.FullName 'ddpg_policy.txt' } |
        Where-Object { Test-Path -LiteralPath $_ -PathType Leaf })
    if ($policies.Count -ne 1) { throw "Expected one trained policy for seed $seed" }
    $evalRoot = Join-Path $root "v15_$seed"
    New-Item -ItemType Directory -Path $evalRoot -Force | Out-Null
    $log = Join-Path $evalRoot 'evaluation_console.log'
    Write-Output "Starting independent evaluation for seed $seed"
    $jobs += Start-Job -Name "Heldout_$seed" -ArgumentList $executable, $seed, $evalRoot, $sourceSimfile, $policies[0], $log -ScriptBlock {
        param($exe, $runSeed, $runRoot, $frozenSimfile, $policyFile, $runLog)
        & $exe --rl-mode eval_grid --rl-model $policyFile --set "rl.seed=$runSeed" `
            --set "carsim.simfile=$frozenSimfile" --no-open --out $runRoot `
            2>&1 | Out-File -LiteralPath $runLog -Encoding utf8
        [pscustomobject]@{ Seed = $runSeed; ExitCode = $LASTEXITCODE; Root = $runRoot; Log = $runLog }
    }
}
$failures = @()
while ($jobs.Count -gt 0) {
    $finished = Wait-Job -Job $jobs -Any
    try {
        $result = Receive-Job -Job $finished -ErrorAction Stop
        $run = Get-ChildItem -LiteralPath $result.Root -Directory |
            Sort-Object LastWriteTime -Descending | Select-Object -First 1
        $runPath = if ($run) { $run.FullName } else { '' }
        "$($result.Seed),$($result.ExitCode),`"${runPath}`"" |
            Add-Content -LiteralPath $manifest -Encoding utf8
        Write-Output "Finished independent evaluation for seed $($result.Seed); exit $($result.ExitCode)"
        if ($result.ExitCode -ne 0) { $failures += "$($result.Seed): $($result.Log)" }
    } catch {
        $failures += "$($finished.Name): $($_.Exception.Message)"
    }
    $jobs = @($jobs | Where-Object Id -NE $finished.Id)
    Remove-Job -Job $finished
}
if ($failures.Count -gt 0) { throw "Independent evaluation failures: $($failures -join '; ')" }
