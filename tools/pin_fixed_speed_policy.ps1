param(
    [Parameter(Mandatory=$true)][ValidateSet(60,80)][int]$Speed,
    [Parameter(Mandatory=$true)][ValidateSet('ddpg','td3')][string]$Algorithm,
    [Parameter(Mandatory=$true)][string]$TrainingRun
)
$ErrorActionPreference='Stop'
Push-Location (Split-Path -Parent $PSScriptRoot)
try {
    $run=(Resolve-Path -LiteralPath $TrainingRun).Path
    $name=$Algorithm+'_policy.txt'
    foreach($file in @($name,'effective_config.ini','controller_revision.txt','selected_validation.csv','rl_report.html','tire_data.csv','training_source_Run_all.par')){
        if(-not (Test-Path -LiteralPath (Join-Path $run $file) -PathType Leaf)){throw "Training result missing $file"}
    }
    if(Test-Path -LiteralPath (Join-Path $run 'invalid_training.txt')){throw 'Training run is marked invalid.'}
    $config=Get-Content -LiteralPath (Join-Path $run 'effective_config.ini') -Raw
    $algorithmLine=[regex]::Match($config,'(?m)^algorithm\s*=\s*([^\s#;]+)')
    $header=Get-Content -LiteralPath (Join-Path $run $name) -TotalCount 1
    if(-not $algorithmLine.Success -or $algorithmLine.Groups[1].Value -ne $Algorithm -or
       $header -notmatch ('^'+$Algorithm.ToUpperInvariant()+'_MPC_V18_COMPACT_')){throw 'Policy algorithm or action profile does not match.'}
    foreach($key in @('train_speed_min_kmh','train_speed_max_kmh')){
        $m=[regex]::Match($config, '(?m)^'+$key+'\s*=\s*([^\s#;]+)')
        if(-not $m.Success -or [math]::Abs([double]::Parse($m.Groups[1].Value,[cultureinfo]::InvariantCulture)-$Speed) -gt 1e-8){throw "Policy $key does not match $Speed km/h."}
    }
    $row=Import-Csv -LiteralPath (Join-Path $run 'selected_validation.csv')|Select-Object -First 1
    if(-not $row){throw 'Selected checkpoint provenance missing.'}
    $target=Join-Path (Join-Path 'policies\fixed_speed' ("v$Speed")) $Algorithm
    if(Test-Path -LiteralPath $target){throw "Pinned policy already exists: $target. Use a new archive location before replacing it."}
    New-Item -ItemType Directory -Path $target -Force|Out-Null
    foreach($file in @($name,'effective_config.ini','controller_revision.txt','selected_validation.csv','tire_data.csv','training_source_Run_all.par')){
        Copy-Item -LiteralPath (Join-Path $run $file) -Destination (Join-Path $target $file)
    }
    $pinned=Join-Path $target $name
    $status=if($row.checkpoint_episode -eq '0'){'neutral_fallback'}elseif($row.checkpoint_episode -eq '-1'){'inherited_actor'}else{'learned_checkpoint'}
    [pscustomobject]@{speed_kmh=$Speed;algorithm=$Algorithm;checkpoint_episode=$row.checkpoint_episode;status=$status;sha256=(Get-FileHash -LiteralPath $pinned -Algorithm SHA256).Hash;source_sha256=(Get-FileHash -LiteralPath (Join-Path $target 'training_source_Run_all.par') -Algorithm SHA256).Hash;config_sha256=(Get-FileHash -LiteralPath (Join-Path $target 'effective_config.ini') -Algorithm SHA256).Hash;tire_sha256=(Get-FileHash -LiteralPath (Join-Path $target 'tire_data.csv') -Algorithm SHA256).Hash;training_run=$run} |
        Export-Csv -LiteralPath (Join-Path $target 'pinned_policy.csv') -NoTypeInformation -Encoding UTF8
    if($status -eq 'neutral_fallback'){Write-Warning "$Algorithm at $Speed km/h is neutral fallback, not an improved learned policy."}
    if($status -eq 'inherited_actor'){Write-Warning "$Algorithm at $Speed km/h selected its inherited Actor; later training did not beat it."}
    Write-Host "Pinned $Algorithm $Speed km/h ($status): $pinned"
} finally {Pop-Location}
