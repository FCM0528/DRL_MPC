param(
    [string]$OutputRoot = 'results/absolute_v15_five_seeds',
    [int[]]$Seeds = @(101, 102, 103, 104, 105),
    [string]$FrozenSimfile = ''
)

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$executable = Join-Path $projectRoot 'out/build/x64-Debug/MPC_DoubleLaneChange.exe'
if (-not (Test-Path -LiteralPath $executable -PathType Leaf)) {
    throw "Build the executable first: $executable"
}
if (-not $FrozenSimfile) {
    $sourceRoot = Join-Path $projectRoot 'results/paper_model_five_seeds_frozen_source'
    $sourceRun = Get-ChildItem -LiteralPath $sourceRoot -Directory |
        Sort-Object LastWriteTime -Descending | Select-Object -First 1
    if (-not $sourceRun) { throw "Frozen CarSim source missing in $sourceRoot" }
    $FrozenSimfile = Join-Path $sourceRun.FullName 'carsim_native/simfile.sim'
}
$FrozenSimfile = [System.IO.Path]::GetFullPath($FrozenSimfile)
if (-not (Test-Path -LiteralPath $FrozenSimfile -PathType Leaf)) {
    throw "Frozen Simfile missing: $FrozenSimfile"
}
$outputPath = [System.IO.Path]::GetFullPath((Join-Path $projectRoot $OutputRoot))
New-Item -ItemType Directory -Path $outputPath -Force | Out-Null
$manifest = Join-Path $outputPath 'training_runs.csv'
"seed,episodes,exit_code,run_directory" | Set-Content -LiteralPath $manifest -Encoding utf8
"Frozen Simfile: $FrozenSimfile" | Set-Content -LiteralPath (Join-Path $outputPath 'source_snapshot.txt') -Encoding utf8

$jobs = @()
foreach ($seed in $Seeds) {
    $seedRoot = Join-Path $outputPath "seed_$seed"
    New-Item -ItemType Directory -Path $seedRoot -Force | Out-Null
    $log = Join-Path $seedRoot 'training_console.log'
    Write-Output "Starting DDPG seed $seed, 200 episodes"
    $jobs += Start-Job -Name "DDPG_$seed" -ArgumentList $executable, $seed, $seedRoot, $FrozenSimfile, $log -ScriptBlock {
        param($exe, $runSeed, $runRoot, $sourceSimfile, $runLog)
        & $exe --rl-mode train --rl-episodes 200 --set "rl.seed=$runSeed" `
            --set "carsim.simfile=$sourceSimfile" `
            --no-open --out $runRoot 2>&1 | Out-File -LiteralPath $runLog -Encoding utf8
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
        "$($result.Seed),200,$($result.ExitCode),`"${runPath}`"" |
            Add-Content -LiteralPath $manifest -Encoding utf8
        Write-Output "Finished seed $($result.Seed); exit code $($result.ExitCode); run $runPath"
        if ($result.ExitCode -ne 0) { $failures += "$($result.Seed): $($result.Log)" }
    } catch {
        $failures += "$($finished.Name): $($_.Exception.Message)"
    }
    $jobs = @($jobs | Where-Object Id -NE $finished.Id)
    Remove-Job -Job $finished
}
if ($failures.Count -gt 0) { throw "Training failures: $($failures -join '; ')" }
