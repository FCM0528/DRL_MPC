param(
    [ValidateSet('ddpg', 'td3')][string]$Algorithm = 'ddpg',
    [string]$Policy = '',
    [double]$SpeedMinKmh = 30,
    [double]$SpeedMaxKmh = 120,
    [double]$MuMin = 0.3,
    [double]$MuMax = 0.9,
    [string]$OutputRoot = '',
    [string]$Simfile = '',
    [string]$Config = 'configs\carsim_compact_training.ini',
    [switch]$AllowModelMismatch,
    [switch]$NoOpen
)

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Push-Location $projectRoot
try {
    if ((Split-Path -Leaf $Config) -in @('carsim_focus_60_80.ini','carsim_focus_60_80_tracking.ini','carsim_focus_fast.ini','carsim_focus_fast_guarded.ini')) {
        if (-not $PSBoundParameters.ContainsKey('SpeedMinKmh')) { $SpeedMinKmh = 55 }
        if (-not $PSBoundParameters.ContainsKey('SpeedMaxKmh')) { $SpeedMaxKmh = 85 }
    }
    $exe = '.\out\build\x64-Release\Release\MPC_DoubleLaneChange.exe'
    if (-not (Test-Path -LiteralPath $exe -PathType Leaf)) { throw "Build the Release executable first: $exe" }
    function ConfigLines([string]$ini, [string[]]$chain = @()) {
        $path = (Resolve-Path -LiteralPath $ini).Path
        if ($chain -contains $path) { throw "Circular configuration inheritance: $path" }
        $lines = @(Get-Content -LiteralPath $path)
        $section = ''
        foreach ($line in $lines) {
            if ($line -match '^\s*\[([^]]+)\]') { $section = $Matches[1].ToLowerInvariant(); continue }
            if ($section -eq 'inherit' -and $line -match '^\s*base\s*=\s*([^#;]+)') {
                $base = $Matches[1].Trim()
                ConfigLines (Join-Path (Split-Path -Parent $path) $base) ($chain + $path)
            }
        }
        $lines
    }
    function ModelSignature([string]$ini) {
        $settings=@{nonlinear_tire='0';tire_slope_floor='0.15';grip_switch_enabled='1';grip_state_only='0';grip_preview_s='0.8';speed_rate_enabled='0';speed_rate_start_kmh='60';speed_rate_end_kmh='120';speed_rate_end_deg_s='5';rho='20000';max_iterations='2500';primal_tol='1e-5';dual_tol='0.002';low_speed_blend='0';blend_start_kmh='60';blend_end_kmh='80';train_speed_min_kmh='30';train_speed_max_kmh='120';decision_steps='5';max_steer_deg='30';max_steer_rate_deg_s='45';high_speed_blend='0';high_blend_start_kmh='85';high_blend_end_kmh='95';augmented_observation='0';grip_demand_clip='3'}
        $section=''
        foreach($line in ConfigLines $ini){
            if($line -match '^\s*\[([^]]+)\]'){$section=$Matches[1].ToLowerInvariant();continue}
            if($section -in @('mpc','solver','rl','constraints') -and $line -match '^\s*([a-zA-Z_]+)\s*=\s*([^#;\s]+)'){
                $key=$Matches[1].ToLowerInvariant();if($settings.ContainsKey($key)){$settings[$key]=$Matches[2].ToLowerInvariant()}
            }
        }
        foreach($key in @('nonlinear_tire','grip_switch_enabled','grip_state_only','speed_rate_enabled','low_speed_blend','high_speed_blend','augmented_observation')){
            if($settings[$key] -in @('true','yes','1')){$settings[$key]='1'}
            elseif($settings[$key] -in @('false','no','0')){$settings[$key]='0'}
            else {throw "Invalid $key in $ini"}
        }
        foreach($key in @('tire_slope_floor','grip_preview_s','speed_rate_start_kmh','speed_rate_end_kmh','speed_rate_end_deg_s','rho','max_iterations','primal_tol','dual_tol','blend_start_kmh','blend_end_kmh','train_speed_min_kmh','train_speed_max_kmh','decision_steps','max_steer_deg','max_steer_rate_deg_s','high_blend_start_kmh','high_blend_end_kmh','grip_demand_clip')){
            $settings[$key]=([double]::Parse($settings[$key],[cultureinfo]::InvariantCulture)).ToString('G12',[cultureinfo]::InvariantCulture)
        }
        return ($settings.nonlinear_tire,$settings.tire_slope_floor,$settings.grip_switch_enabled,$settings.grip_state_only,$settings.grip_preview_s,$settings.speed_rate_enabled,$settings.speed_rate_start_kmh,$settings.speed_rate_end_kmh,$settings.speed_rate_end_deg_s,$settings.rho,$settings.max_iterations,$settings.primal_tol,$settings.dual_tol,$settings.low_speed_blend,$settings.blend_start_kmh,$settings.blend_end_kmh, $settings.train_speed_min_kmh, $settings.train_speed_max_kmh, $settings.decision_steps, $settings.max_steer_deg, $settings.max_steer_rate_deg_s, $settings.high_speed_blend, $settings.high_blend_start_kmh, $settings.high_blend_end_kmh, $settings.augmented_observation, $settings.grip_demand_clip) -join '|'
    }
    $selectionModel = ModelSignature $Config
    $configLeaf = Split-Path -Leaf $Config
    if (-not $Policy -and $configLeaf -in @('carsim_np_only.ini','carsim_compact_training.ini','carsim_compact_relative.ini','carsim_compact_guarded.ini','carsim_compact_blend.ini','carsim_focus_60_80.ini','carsim_focus_60_80_tracking.ini','carsim_focus_fast.ini','carsim_focus_fast_guarded.ini')) {
        $policyTag = if ($configLeaf -eq 'carsim_np_only.ini') { 'V19_NP_ONLY' } else { 'V18_COMPACT' }
        $root = Join-Path 'policies\iterations' $Algorithm
        $candidate = @(Get-ChildItem -LiteralPath $root -Recurse -File -Filter ($Algorithm + '_policy.txt') -ErrorAction SilentlyContinue |
            Where-Object {
                (Test-Path -LiteralPath (Join-Path $_.DirectoryName 'rl_report.html') -PathType Leaf) -and
                -not (Test-Path -LiteralPath (Join-Path $_.DirectoryName 'invalid_training.txt') -PathType Leaf) -and
                (Test-Path -LiteralPath (Join-Path $_.DirectoryName 'controller_revision.txt') -PathType Leaf) -and
                ((Get-Content -LiteralPath (Join-Path $_.DirectoryName 'controller_revision.txt') -Raw).Trim() -eq 'ltv-warm-sequence-v1') -and
                (Test-Path -LiteralPath (Join-Path $_.DirectoryName 'effective_config.ini') -PathType Leaf) -and
                ((ModelSignature (Join-Path $_.DirectoryName 'effective_config.ini')) -eq $selectionModel) -and
                ((Get-Content -LiteralPath $_.FullName -TotalCount 1) -match ('^' + $Algorithm.ToUpperInvariant() + '_MPC_' + $policyTag + '_'))
            } | Sort-Object LastWriteTime -Descending | Select-Object -First 1)
        if ($candidate.Count -eq 0) { throw "No completed $policyTag $Algorithm policy found." }
        $Policy = $candidate[0].FullName
        Write-Host "Using latest completed $policyTag policy: $Policy"
    }
    if (-not $Policy) {
        $rows = @(Import-Csv -LiteralPath 'policies\active_registry.csv' |
            Where-Object { $_.algorithm -eq $Algorithm })
        if ($rows.Count -ne 1) { throw "Expected one registered $Algorithm policy." }
        $entry = $rows[0]
        $Policy = $entry.policy_file
        if (-not (Test-Path -LiteralPath $Policy -PathType Leaf) -or
            (Get-FileHash -LiteralPath $Policy -Algorithm SHA256).Hash -ne $entry.sha256) {
            throw "Registered $Algorithm policy is missing or its checksum changed."
        }
        $culture = [System.Globalization.CultureInfo]::InvariantCulture
        if (-not $PSBoundParameters.ContainsKey('MuMin')) { $MuMin = [double]::Parse($entry.mu_min, $culture) }
        if (-not $PSBoundParameters.ContainsKey('MuMax')) { $MuMax = [double]::Parse($entry.mu_max, $culture) }
        if (-not $PSBoundParameters.ContainsKey('SpeedMinKmh')) { $SpeedMinKmh = [double]::Parse($entry.speed_min_kmh, $culture) }
        if (-not $PSBoundParameters.ContainsKey('SpeedMaxKmh')) { $SpeedMaxKmh = [double]::Parse($entry.speed_max_kmh, $culture) }
    }
    if (-not (Test-Path -LiteralPath $Policy -PathType Leaf)) { throw "Policy missing: $Policy" }
    $invalidMarker = Join-Path (Split-Path -Parent (Resolve-Path -LiteralPath $Policy).Path) 'invalid_training.txt'
    if (Test-Path -LiteralPath $invalidMarker -PathType Leaf) { throw "Training run is marked invalid: $invalidMarker" }
    $trainingConfig=Join-Path (Split-Path -Parent (Resolve-Path -LiteralPath $Policy).Path) 'effective_config.ini'
    $revisionPath=Join-Path (Split-Path -Parent $trainingConfig) 'controller_revision.txt'
    if (-not (Test-Path -LiteralPath $revisionPath -PathType Leaf) -or
        (Get-Content -LiteralPath $revisionPath -Raw).Trim() -ne 'ltv-warm-sequence-v1') {
        if (-not $AllowModelMismatch) { throw 'Policy was trained under an older or unknown MPC controller revision.' }
        Write-Warning 'Policy was trained under an older or unknown MPC controller revision.'
    }
    if (Test-Path -LiteralPath $trainingConfig -PathType Leaf) {
        $minLine = @(ConfigLines $trainingConfig | Where-Object { $_ -match '^\s*train_speed_min_kmh\s*=' } | Select-Object -Last 1)
        $maxLine = @(ConfigLines $trainingConfig | Where-Object { $_ -match '^\s*train_speed_max_kmh\s*=' } | Select-Object -Last 1)
        if ($minLine.Count -and $maxLine.Count -and
            (([double]::Parse(($minLine[0] -split '=')[1].Trim(), [cultureinfo]::InvariantCulture) -ne $SpeedMinKmh) -or
             ([double]::Parse(($maxLine[0] -split '=')[1].Trim(), [cultureinfo]::InvariantCulture) -ne $SpeedMaxKmh))) {
            if (-not $AllowModelMismatch) { throw 'Policy training speed bounds differ from requested bounds.' }
            Write-Warning 'Policy training speed bounds differ from requested bounds.'
        }
    }
    if(-not (Test-Path -LiteralPath $trainingConfig -PathType Leaf) -or (ModelSignature $trainingConfig) -ne (ModelSignature $Config)){
        $message='Policy training model differs from or is not recorded for the evaluation configuration.'
        if(-not $AllowModelMismatch){throw "$message Use -AllowModelMismatch only for diagnostics."}
        Write-Warning $message
    }
    if (-not $OutputRoot) {
        $OutputRoot = Join-Path (Join-Path 'policies\evaluations\manual' $Algorithm) (Get-Date -Format 'yyyyMMdd-HHmmss-fff')
    }
    if (Test-Path -LiteralPath $OutputRoot) { throw "Output already exists: $OutputRoot" }
    $culture = [System.Globalization.CultureInfo]::InvariantCulture
    $programArgs = @(
        '--config', $Config, '--rl-mode', 'eval_general', '--rl-model', $Policy,
        '--set', "rl.algorithm=$Algorithm",
        '--set', ('rl.train_speed_min_kmh=' + $SpeedMinKmh.ToString($culture)),
        '--set', ('rl.train_speed_max_kmh=' + $SpeedMaxKmh.ToString($culture)),
        '--set', ('rl.train_mu_min=' + $MuMin.ToString($culture)),
        '--set', ('rl.train_mu_max=' + $MuMax.ToString($culture)),
        '--set', 'carsim.sync_browser_result=false', '--out', $OutputRoot
    )
    if ($Simfile) { $programArgs += @('--set', "carsim.simfile=$Simfile") }
    if ($NoOpen) { $programArgs += '--no-open' } else { $programArgs += '--open' }
    & $exe @programArgs
    if ($LASTEXITCODE -ne 0) { throw "Generalization evaluation failed (exit code $LASTEXITCODE)." }
    $run = Get-ChildItem -LiteralPath $OutputRoot -Directory | Select-Object -First 1
    if (-not $run) { throw 'Evaluation result directory is missing.' }
    Write-Host "Generalization report: $(Join-Path $run.FullName 'generalization_report.html')"
}
finally {
    Pop-Location
}
