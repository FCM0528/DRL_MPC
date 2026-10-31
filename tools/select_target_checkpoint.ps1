param([Parameter(Mandatory=$true)][string]$RunDirectory)
$ErrorActionPreference='Stop'
$root=Split-Path -Parent $PSScriptRoot
Push-Location $root
try {
    if (Test-Path -LiteralPath (Join-Path $RunDirectory 'invalid_training.txt') -PathType Leaf) {
        throw 'Training run is marked invalid and cannot be selected.'
    }
    $data=@(Import-Csv -LiteralPath (Join-Path $RunDirectory 'checkpoint_validation.csv'))
    $base=@($data | Where-Object episode -eq '0')
    if ($base.Count -lt 3) { throw 'Checkpoint reference grid is missing.' }
    $lookup=@{}
    foreach($r in $base) { $lookup["$($r.speed_kmh)|$($r.mu)"]=$r }
    $meanBase=($base | Measure-Object rmse_y_m -Average).Average
    $targetMu=[double]$base[0].mu
    $results=@()
    $episodes=@($data | Where-Object { [int]$_.episode -gt 0 } | Select-Object -ExpandProperty episode -Unique)
    foreach($episode in $episodes) {
        $cases=@($data | Where-Object episode -eq $episode)
        if ($cases.Count -ne $base.Count) { throw "Incomplete checkpoint: $episode" }
        $target=@($cases | Where-Object { ([Math]::Abs([double]$_.speed_kmh-60) -lt 1e-5 -or [Math]::Abs([double]$_.speed_kmh-80) -lt 1e-5) -and [Math]::Abs([double]$_.mu-$targetMu) -lt 1e-5 })
        if ($target.Count -ne 2) { throw 'Target 60/80 km/h checks are missing or ambiguous.' }
        $targetOkay=$true
        foreach($r in $target) {
            $b=$lookup["$($r.speed_kmh)|$($r.mu)"]
            if (-not $b -or $r.completed -ne '1' -or [double]$r.rmse_y_m -gt 1.10*[double]$b.rmse_y_m+0.02 -or
                [double]$r.max_abs_y_m -gt 1.10*[double]$b.max_abs_y_m+0.05) { $targetOkay=$false }
        }
        $completed=@($cases | Where-Object completed -eq '1').Count
        $mean=($cases | Measure-Object rmse_y_m -Average).Average
        $eligible=$targetOkay -and $completed -ge @($base | Where-Object completed -eq '1').Count -and $mean -le 1.10*$meanBase+0.02
        $results += [pscustomobject]@{episode=[int]$episode;target_mu=$targetMu;
            target_rmse=($target | Measure-Object rmse_y_m -Average).Average;grid_rmse=$mean;
            baseline_grid_rmse=$meanBase;completed=$completed;eligible=$eligible;
            policy=(Join-Path $RunDirectory ('checkpoint_{0:d4}_policy.txt' -f [int]$episode))}
    }
    $results | Export-Csv -LiteralPath (Join-Path $RunDirectory 'target_checkpoint_selection.csv') -NoTypeInformation -Encoding UTF8
    $best=@($results | Where-Object eligible | Sort-Object target_rmse,episode | Select-Object -First 1)
    if ($best.Count -eq 0) { Write-Host 'No checkpoint met the target and grid criteria.'; exit 2 }
    Write-Host "Selected for independent testing: episode $($best[0].episode), policy $($best[0].policy)"
    $best[0] | Format-List
} finally { Pop-Location }
