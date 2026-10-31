$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Push-Location $projectRoot
try {
    $models = @()
    foreach ($root in @('policies\archive', 'policies\iterations', 'policies\candidates', 'policies\tgt\rw')) {
        if (Test-Path -LiteralPath $root -PathType Container) {
            $models += @(Get-ChildItem -LiteralPath $root -Recurse -File |
                Where-Object { $_.Name -in @('ddpg_policy.txt', 'td3_policy.txt') })
        }
    }
    $rows = foreach ($model in $models) {
        $fields = ((Get-Content -LiteralPath $model.FullName -TotalCount 1) -split '\s+' |
            Where-Object { $_ -ne '' })
        if ($fields.Count -lt 19) { continue }
        $selectedFile = Join-Path $model.DirectoryName 'selected_validation.csv'
        $selected = if (Test-Path -LiteralPath $selectedFile -PathType Leaf) {
            Import-Csv -LiteralPath $selectedFile | Select-Object -First 1
        } else { $null }
        $checkpoint = if ($selected) { $selected.checkpoint_episode } else { '' }
        $status = if (Test-Path -LiteralPath (Join-Path $model.DirectoryName 'invalid_training.txt') -PathType Leaf) {
            'invalid_training'
        } elseif (-not $selected -and (Test-Path -LiteralPath (Join-Path $model.DirectoryName 'candidate_manifest.txt') -PathType Leaf)) {
            'unvalidated_actor'
        } elseif (-not $selected) { 'incomplete' } elseif ($checkpoint -eq '0') {
            'neutral_fallback'
        } elseif ($checkpoint -eq '-1') { 'inherited' } else { 'learned_checkpoint' }
        [pscustomobject]@{
            algorithm = if ($model.Name -eq 'ddpg_policy.txt') { 'ddpg' } else { 'td3' }
            policy = $model.FullName
            format = $fields[0]
            mu_min = $fields[$fields.Count - 2]
            mu_max = $fields[$fields.Count - 1]
            checkpoint_episode = $checkpoint
            status = $status
            training_state_available = Test-Path -LiteralPath (Join-Path $model.DirectoryName 'best_training_state.txt') -PathType Leaf
            sha256 = (Get-FileHash -LiteralPath $model.FullName -Algorithm SHA256).Hash
        }
    }
    if ($rows.Count -eq 0) { throw 'No policy files found.' }
    $catalog = 'policies\catalog.csv'
    $rows | Sort-Object algorithm, policy | Export-Csv -LiteralPath $catalog -NoTypeInformation -Encoding UTF8
    Write-Host "Policy catalog: $((Resolve-Path -LiteralPath $catalog).Path)"
    $rows | Select-Object algorithm, format, mu_min, mu_max, checkpoint_episode, status |
        Format-Table -AutoSize
}
finally {
    Pop-Location
}
