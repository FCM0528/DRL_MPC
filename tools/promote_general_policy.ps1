param(
    [Parameter(Mandatory = $true)][ValidateSet('ddpg', 'td3')][string]$Algorithm,
    [Parameter(Mandatory = $true)][string]$CandidateRun,
    [Parameter(Mandatory = $true)][string]$CandidateEvaluation,
    [string]$RegistryPath = 'policies\active_registry.csv',
    [string]$HistoryDirectory = 'policies\registry_history'
)

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Push-Location $projectRoot
$registryLock = [System.Threading.Mutex]::new($false, 'Local\MPC_GeneralPolicyRegistry_v1')
$lockHeld = $false
try {
    if (-not $registryLock.WaitOne([TimeSpan]::FromSeconds(30))) {
        throw 'Timed out waiting for the policy registry lock.'
    }
    $lockHeld = $true
    $registryPath = $RegistryPath
    $registry = @(Import-Csv -LiteralPath $registryPath)
    $entries = @($registry | Where-Object { $_.algorithm -eq $Algorithm })
    if ($entries.Count -ne 1) { throw "Expected one active $Algorithm policy." }
    $active = $entries[0]
    $modelName = if ($Algorithm -eq 'ddpg') { 'ddpg_policy.txt' } else { 'td3_policy.txt' }
    $candidatePolicy = Join-Path $CandidateRun $modelName
    $candidateCsv = Join-Path $CandidateEvaluation 'generalization_validation.csv'
    $referenceCsv = Join-Path $active.evaluation_dir 'generalization_validation.csv'
    foreach ($path in @($candidatePolicy, $candidateCsv, $referenceCsv,
        (Join-Path $CandidateRun 'selected_validation.csv'),
        (Join-Path $CandidateEvaluation 'policy_used.txt'),
        (Join-Path $active.evaluation_dir 'policy_used.txt'),
        (Join-Path $CandidateEvaluation 'evaluation_source_Run_all.par'),
        (Join-Path $active.evaluation_dir 'evaluation_source_Run_all.par'))) {
        if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { throw "Required evidence missing: $path" }
    }
    $candidateHash = (Get-FileHash -LiteralPath $candidatePolicy -Algorithm SHA256).Hash
    if ($candidateHash -ne (Get-FileHash -LiteralPath (Join-Path $CandidateEvaluation 'policy_used.txt') -Algorithm SHA256).Hash) {
        throw 'Candidate evaluation used a different policy file.'
    }
    if ($active.sha256 -ne (Get-FileHash -LiteralPath (Join-Path $active.evaluation_dir 'policy_used.txt') -Algorithm SHA256).Hash) {
        throw 'Active policy evaluation does not match the registry.'
    }
    function Comparable-Source([string]$path) {
        $lines = @(Get-Content -LiteralPath $path | ForEach-Object {
            if ($_ -match '^\s*(SPEED_TARGET_CONSTANT|MU_ROAD_CONSTANT|TSTOP)\s') {
                $Matches[1] + ' <variable>'
            } else { $_ }
        })
        return [string]::Join([Environment]::NewLine, [string[]]$lines)
    }
    if ((Comparable-Source (Join-Path $CandidateEvaluation 'evaluation_source_Run_all.par')) -ne
        (Comparable-Source (Join-Path $active.evaluation_dir 'evaluation_source_Run_all.par'))) {
        Write-Host 'Candidate not promoted: the active evaluation uses a different vehicle or road source.'
        return [pscustomobject]@{ promoted = $false; reason = 'different_evaluation_source' }
    }
    $selected = Import-Csv -LiteralPath (Join-Path $CandidateRun 'selected_validation.csv') | Select-Object -First 1
    if (-not $selected -or $selected.checkpoint_episode -eq '0') {
        Write-Host 'Candidate not promoted: selected policy is the neutral fallback.'
        return [pscustomobject]@{ promoted = $false; reason = 'neutral_fallback' }
    }
    $candidateRows = @(Import-Csv -LiteralPath $candidateCsv)
    $referenceRows = @(Import-Csv -LiteralPath $referenceCsv)
    $candidate = @($candidateRows | Where-Object { $_.controller -eq $Algorithm })
    $reference = @($referenceRows | Where-Object { $_.controller -eq $Algorithm })
    if ($candidate.Count -ne 36 -or $reference.Count -ne 36) {
        throw 'Promotion requires the full 36-case generalization grid.'
    }
    $culture = [System.Globalization.CultureInfo]::InvariantCulture
    function Number([string]$value) { return [double]::Parse($value, $culture) }
    function Key($row) { return $row.mu + '|' + $row.speed_kmh }
    $oldByCase = @{}
    $baseByCase = @{}
    $oldBaseByCase = @{}
    foreach ($row in $reference) { $oldByCase[(Key $row)] = $row }
    foreach ($row in $candidateRows | Where-Object { $_.controller -eq 'fixed_mpc' }) { $baseByCase[(Key $row)] = $row }
    foreach ($row in $referenceRows | Where-Object { $_.controller -eq 'fixed_mpc' }) { $oldBaseByCase[(Key $row)] = $row }
    $violations = @()
    $candidateSum = 0.0
    $referenceSum = 0.0
    foreach ($row in $candidate) {
        $key = Key $row
        if (-not $oldByCase.ContainsKey($key) -or -not $baseByCase.ContainsKey($key) -or -not $oldBaseByCase.ContainsKey($key)) {
            throw "Mismatched evaluation case: $key"
        }
        $old = $oldByCase[$key]
        $base = $baseByCase[$key]
        $oldBase = $oldBaseByCase[$key]
        $actual = Number $row.rmse_y_m
        $prior = Number $old.rmse_y_m
        $fixed = Number $base.rmse_y_m
        if ([math]::Abs($fixed - (Number $oldBase.rmse_y_m)) -gt 1e-6) {
            throw "Fixed MPC baseline changed in case $key; re-evaluate the active policy."
        }
        if (($old.completed -eq '1' -or $base.completed -eq '1') -and $row.completed -ne '1') {
            $violations += "$key incomplete"
        }
        if ($actual -gt ($prior * 1.10 + 0.02) -or $actual -gt ($fixed * 1.10 + 0.02)) {
            $violations += "$key RMSE"
        }
        $maxError = Number $row.max_abs_y_m
        if ($maxError -gt ((Number $old.max_abs_y_m) * 1.10 + 0.05) -or
            $maxError -gt ((Number $base.max_abs_y_m) * 1.10 + 0.05)) {
            $violations += "$key maximum error"
        }
        $candidateSum += $actual
        $referenceSum += $prior
    }
    foreach ($grip in ($candidate.mu | Select-Object -Unique)) {
        $newGroup = @($candidate | Where-Object { $_.mu -eq $grip })
        $oldGroup = @($reference | Where-Object { $_.mu -eq $grip })
        $newMean = ($newGroup | ForEach-Object { Number $_.rmse_y_m } | Measure-Object -Average).Average
        $oldMean = ($oldGroup | ForEach-Object { Number $_.rmse_y_m } | Measure-Object -Average).Average
        if ($newMean -gt ($oldMean * 1.05 + 0.005)) { $violations += "mu=$grip mean RMSE" }
    }
    $candidateMean = $candidateSum / $candidate.Count
    $referenceMean = $referenceSum / $reference.Count
    if ($candidateMean -ge $referenceMean - 0.001) { $violations += 'mean RMSE did not improve by 0.001 m' }
    if ($violations.Count -gt 0) {
        Write-Host "Candidate not promoted; current mean RMSE=$candidateMean, active mean=$referenceMean."
        Write-Host "Reasons: $(($violations | Select-Object -First 12) -join ', ')"
        return [pscustomobject]@{ promoted = $false; reason = 'evaluation_gate'; violations = $violations.Count }
    }

    $history = $HistoryDirectory
    New-Item -ItemType Directory -Force -Path $history | Out-Null
    $backup = Join-Path (Resolve-Path -LiteralPath $history).Path ('active_registry_' + (Get-Date -Format 'yyyyMMdd-HHmmss-fff') + '.csv')
    $active.policy_file = (Resolve-Path -LiteralPath $candidatePolicy).Path
    $active.evaluation_dir = (Resolve-Path -LiteralPath $CandidateEvaluation).Path
    $active.sha256 = $candidateHash
    $active.status = 'relative_non_regression'
    $temporary = $registryPath + '.next'
    $registry | Export-Csv -LiteralPath $temporary -NoTypeInformation -Encoding UTF8
    [System.IO.File]::Replace((Resolve-Path -LiteralPath $temporary).Path,
                              (Resolve-Path -LiteralPath $registryPath).Path, $backup)
    Write-Host "Promoted $Algorithm policy: mean RMSE $referenceMean -> $candidateMean."
    return [pscustomobject]@{ promoted = $true; reason = 'passed_evaluation_gate'; violations = 0 }
}
finally {
    if ($lockHeld) { $registryLock.ReleaseMutex() }
    $registryLock.Dispose()
    Pop-Location
}
