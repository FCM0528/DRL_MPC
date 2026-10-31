param(
    [ValidateSet('ddpg', 'td3')][string]$Algorithm = 'ddpg',
    [int]$Episodes = 300,
    [int]$Seed = -1,
    [double]$SpeedMinKmh = 30,
    [double]$SpeedMaxKmh = 120,
    [double]$MuMin = 0.3,
    [double]$MuMax = 0.9,
    [string]$WarmStart = '',
    [string]$ResumeState = '',
    [string]$Executable = '',
    [string]$Simfile = '',
    [string]$Config = 'configs\carsim_compact_training.ini',
    [switch]$FocusTargetSpeeds,
    [switch]$Curriculum,
    [switch]$VaryRoadTransition,
    [switch]$Fresh,
    [switch]$NoEvaluation,
    [switch]$NoPromotion = $true
)

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Push-Location $projectRoot
try {
    if ((Split-Path -Leaf $Config) -eq 'carsim_fixed60_np_focus.ini') {
        if (-not $PSBoundParameters.ContainsKey('SpeedMinKmh')) { $SpeedMinKmh = 60 }
        if (-not $PSBoundParameters.ContainsKey('SpeedMaxKmh')) { $SpeedMaxKmh = 60 }
    }
    if ((Split-Path -Leaf $Config) -eq 'carsim_fixed80_np_focus.ini') {
        if (-not $PSBoundParameters.ContainsKey('SpeedMinKmh')) { $SpeedMinKmh = 80 }
        if (-not $PSBoundParameters.ContainsKey('SpeedMaxKmh')) { $SpeedMaxKmh = 80 }
    }
    if ((Split-Path -Leaf $Config) -match '^carsim_fixed(60|80)_audit_(paired|direct|augmented|calibrated|fixedscale|slow|grip|slowgrip)\.ini$') {
        $auditSpeed = [int]$Matches[1]
        if (-not $PSBoundParameters.ContainsKey('SpeedMinKmh')) { $SpeedMinKmh = $auditSpeed }
        if (-not $PSBoundParameters.ContainsKey('SpeedMaxKmh')) { $SpeedMaxKmh = $auditSpeed }
    }
    if ((Split-Path -Leaf $Config) -in @('carsim_focus_60_80.ini','carsim_focus_60_80_tracking.ini','carsim_focus_fast.ini','carsim_focus_fast_guarded.ini')) {
        if (-not $PSBoundParameters.ContainsKey('SpeedMinKmh')) { $SpeedMinKmh = 55 }
        if (-not $PSBoundParameters.ContainsKey('SpeedMaxKmh')) { $SpeedMaxKmh = 85 }
    }
    $exe = if ($Executable) { $Executable } else { '.\out\build\x64-Release\Release\MPC_DoubleLaneChange.exe' }
    if (-not (Test-Path -LiteralPath $exe -PathType Leaf)) { throw "Build the Release executable first: $exe" }
    if ($Episodes -lt 1) { throw 'Episodes must be positive.' }
    if ($Simfile -and -not (Test-Path -LiteralPath $Simfile -PathType Leaf)) { throw "Simfile missing: $Simfile" }
    if ($WarmStart -and $ResumeState) { throw 'Choose either WarmStart or ResumeState.' }
    function GetActionProfile([string]$path, [string[]]$seen = @()) {
        $resolved = (Resolve-Path -LiteralPath $path).Path
        if ($seen -contains $resolved) { throw "Circular config inheritance: $resolved" }
        $lines = @(Get-Content -LiteralPath $resolved)
        $section = ''; $base = ''; $profile = ''
        foreach ($line in $lines) {
            if ($line -match '^\s*\[([^]]+)\]') { $section = $Matches[1].ToLowerInvariant(); continue }
            if ($section -eq 'inherit' -and $line -match '^\s*base\s*=\s*([^#;]+)') { $base = $Matches[1].Trim() }
            if ($section -eq 'rl' -and $line -match '^\s*action_profile\s*=\s*([^#;\s]+)') { $profile = $Matches[1].ToLowerInvariant() }
        }
        if ($profile) { return $profile }
        if ($base) { return GetActionProfile (Join-Path (Split-Path -Parent $resolved) $base) ($seen + $resolved) }
        return 'full'
    }
    $desiredProfile = GetActionProfile $Config
    function GetRelativeAdvantage([string]$path, [string[]]$seen = @()) {
        $resolved = (Resolve-Path -LiteralPath $path).Path
        if ($seen -contains $resolved) { throw "Circular config inheritance: $resolved" }
        $section = ''; $base = ''; $setting = ''
        foreach ($line in Get-Content -LiteralPath $resolved) {
            if ($line -match '^\s*\[([^]]+)\]') { $section = $Matches[1].ToLowerInvariant(); continue }
            if ($section -eq 'inherit' -and $line -match '^\s*base\s*=\s*([^#;]+)') { $base = $Matches[1].Trim() }
            if ($section -eq 'rl' -and $line -match '^\s*relative_advantage\s*=\s*([^#;\s]+)') { $setting = $Matches[1].ToLowerInvariant() }
        }
        if ($setting -in @('true','yes','1')) { return $true }
        if ($setting -in @('false','no','0')) { return $false }
        if ($setting) { throw "Invalid relative_advantage in $resolved" }
        if ($base) { return GetRelativeAdvantage (Join-Path (Split-Path -Parent $resolved) $base) ($seen + $resolved) }
        return $false
    }
    $desiredRelativeAdvantage = GetRelativeAdvantage $Config
    function GetQpFailurePenalty([string]$path, [string[]]$seen = @()) {
        $resolved = (Resolve-Path -LiteralPath $path).Path
        if ($seen -contains $resolved) { throw "Circular config inheritance: $resolved" }
        $section = ''; $base = ''; $setting = ''
        foreach ($line in Get-Content -LiteralPath $resolved) {
            if ($line -match '^\s*\[([^]]+)\]') { $section = $Matches[1].ToLowerInvariant(); continue }
            if ($section -eq 'inherit' -and $line -match '^\s*base\s*=\s*([^#;]+)') { $base = $Matches[1].Trim() }
            if ($section -eq 'rl' -and $line -match '^\s*penalize_qp_failure\s*=\s*([^#;\s]+)') { $setting = $Matches[1].ToLowerInvariant() }
        }
        if ($setting -in @('true','yes','1')) { return $true }
        if ($setting -in @('false','no','0')) { return $false }
        if ($setting) { throw "Invalid penalize_qp_failure in $resolved" }
        if ($base) { return GetQpFailurePenalty (Join-Path (Split-Path -Parent $resolved) $base) ($seen + $resolved) }
        return $false
    }
    $desiredQpPenalty = GetQpFailurePenalty $Config
    function GetLowSpeedBlend([string]$path, [string[]]$seen = @()) {
        $resolved = (Resolve-Path -LiteralPath $path).Path
        if ($seen -contains $resolved) { throw "Circular config inheritance: $resolved" }
        $section = ''; $base = ''; $setting = ''
        foreach ($line in Get-Content -LiteralPath $resolved) {
            if ($line -match '^\s*\[([^]]+)\]') { $section = $Matches[1].ToLowerInvariant(); continue }
            if ($section -eq 'inherit' -and $line -match '^\s*base\s*=\s*([^#;]+)') { $base = $Matches[1].Trim() }
            if ($section -eq 'rl' -and $line -match '^\s*low_speed_blend\s*=\s*([^#;\s]+)') { $setting = $Matches[1].ToLowerInvariant() }
        }
        if ($setting -in @('true','yes','1')) { return $true }
        if ($setting -in @('false','no','0')) { return $false }
        if ($setting) { throw "Invalid low_speed_blend in $resolved" }
        if ($base) { return GetLowSpeedBlend (Join-Path (Split-Path -Parent $resolved) $base) ($seen + $resolved) }
        return $false
    }
    $desiredLowSpeedBlend = GetLowSpeedBlend $Config
    function GetIniSetting([string]$path, [string]$wantedSection, [string]$wantedKey, [string[]]$seen = @()) {
        $resolved = (Resolve-Path -LiteralPath $path).Path
        if ($seen -contains $resolved) { throw "Circular config inheritance: $resolved" }
        $section = ''; $base = ''; $value = ''
        foreach ($line in Get-Content -LiteralPath $resolved) {
            if ($line -match '^\s*\[([^]]+)\]') { $section = $Matches[1].ToLowerInvariant(); continue }
            if ($section -eq 'inherit' -and $line -match '^\s*base\s*=\s*([^#;]+)') { $base = $Matches[1].Trim() }
            if ($section -eq $wantedSection -and $line -match '^\s*([a-zA-Z_]+)\s*=\s*([^#;\s]+)' -and $Matches[1].ToLowerInvariant() -eq $wantedKey) { $value = $Matches[2] }
        }
        if ($value) { return $value }
        if ($base) { return GetIniSetting (Join-Path (Split-Path -Parent $resolved) $base) $wantedSection $wantedKey ($seen + $resolved) }
        return ''
    }
    $rewardKeys = @('reward_scale','ey_scale_m','epsi_scale_rad','w_y','w_peak','peak_threshold_m','w_psi','w_steer','w_steer_rate','w_horizon','w_np_change','w_vy','w_yaw_error','w_weight_change','w_weight_deviation','w_grip_limited','failure_penalty','incomplete_episode_penalty')
    function MatchesTrainingDistribution([string]$path) {
        foreach ($pair in @(@('rl','train_speed_min_kmh',"$SpeedMinKmh"),@('rl','train_speed_max_kmh',"$SpeedMaxKmh"))) {
            $actual = GetIniSetting $path $pair[0] $pair[1]
            if (-not $actual -or [double]::Parse($actual,[cultureinfo]::InvariantCulture) -ne [double]$pair[2]) { return $false }
        }
        foreach ($key in $rewardKeys) {
            $expected = GetIniSetting $Config 'rl_reward' $key
            $actual = GetIniSetting $path 'rl_reward' $key
            if ($expected -and $actual -and [double]::Parse($expected,[cultureinfo]::InvariantCulture) -ne [double]::Parse($actual,[cultureinfo]::InvariantCulture)) { return $false }
        }
        foreach ($key in @('min_np','max_np','nominal_np','q_y_min','q_y_max','q_psi_min','q_psi_max','q_vy_min','q_vy_max','q_r_min','q_r_max','r_steer_min','r_steer_max','r_delta_min','r_delta_max','terminal_scale','curvature_preview_s')) {
            $expected = GetIniSetting $Config 'rl' $key
            $actual = GetIniSetting $path 'rl' $key
            if ($expected -and $actual -and [double]::Parse($expected,[cultureinfo]::InvariantCulture) -ne [double]::Parse($actual,[cultureinfo]::InvariantCulture)) { return $false }
        }
        foreach ($pair in @(@('decision_steps','5'), @('discount_gamma','0.98'), @('np_exploration_scale','1'))) {
            $expected = GetIniSetting $Config 'rl' $pair[0]
            $actual = GetIniSetting $path 'rl' $pair[0]
            if (-not $expected) { $expected = $pair[1] }
            if (-not $actual) { $actual = $pair[1] }
            if ([double]::Parse($expected,[cultureinfo]::InvariantCulture) -ne [double]::Parse($actual,[cultureinfo]::InvariantCulture)) { return $false }
        }
        foreach ($pair in @(@('high_blend_start_kmh','85'), @('high_blend_end_kmh','95'))) {
            $expected = GetIniSetting $Config 'rl' $pair[0]
            $actual = GetIniSetting $path 'rl' $pair[0]
            if (-not $expected) { $expected = $pair[1] }
            if (-not $actual) { $actual = $pair[1] }
            if ([double]::Parse($expected,[cultureinfo]::InvariantCulture) -ne [double]::Parse($actual,[cultureinfo]::InvariantCulture)) { return $false }
        }
        $expectedHigh = (GetIniSetting $Config 'rl' 'high_speed_blend') -in @('true','yes','1')
        $actualHigh = (GetIniSetting $path 'rl' 'high_speed_blend') -in @('true','yes','1')
        if ($expectedHigh -ne $actualHigh) { return $false }
        $expectedStrict = (GetIniSetting $Config 'rl' 'strict_tracking_selection') -in @('true','yes','1')
        $actualStrict = (GetIniSetting $path 'rl' 'strict_tracking_selection') -in @('true','yes','1')
        if ($expectedStrict -ne $actualStrict) { return $false }
        foreach ($key in @('align_integer_np','direct_block_reward','augmented_observation')) {
            $expectedFlag = (GetIniSetting $Config 'rl' $key) -in @('true','yes','1')
            $actualFlag = (GetIniSetting $path 'rl' $key) -in @('true','yes','1')
            if ($expectedFlag -ne $actualFlag) { return $false }
        }
        foreach ($pair in @(@('actor_learning_rate','0.0001'), @('grip_demand_clip','3'))) {
            $expected = GetIniSetting $Config 'rl' $pair[0]; $actual = GetIniSetting $path 'rl' $pair[0]
            if (-not $expected) { $expected = $pair[1] }; if (-not $actual) { $actual = $pair[1] }
            if ([double]::Parse($expected,[cultureinfo]::InvariantCulture) -ne [double]::Parse($actual,[cultureinfo]::InvariantCulture)) { return $false }
        }
        return $true
    }
    $iterationRoot = Join-Path 'policies\iterations' $Algorithm
    $previous = @()
    if (Test-Path -LiteralPath $iterationRoot -PathType Container) {
        $previous = @(Get-ChildItem -LiteralPath $iterationRoot -Recurse -File -Filter 'last_training_state.txt' |
            Where-Object {
                (Test-Path -LiteralPath (Join-Path $_.DirectoryName 'rl_report.html') -PathType Leaf) -and
                -not (Test-Path -LiteralPath (Join-Path $_.DirectoryName 'invalid_training.txt') -PathType Leaf) -and
                (Test-Path -LiteralPath (Join-Path $_.DirectoryName 'training_state_context.txt') -PathType Leaf) -and
                (Test-Path -LiteralPath (Join-Path $_.DirectoryName 'controller_revision.txt') -PathType Leaf) -and
                ((Get-Content -LiteralPath (Join-Path $_.DirectoryName 'controller_revision.txt') -Raw).Trim() -eq 'ltv-warm-sequence-v1') -and
                (Test-Path -LiteralPath (Join-Path $_.DirectoryName 'effective_config.ini') -PathType Leaf) -and
                ((GetActionProfile (Join-Path $_.DirectoryName 'effective_config.ini')) -eq $desiredProfile) -and
                ((GetRelativeAdvantage (Join-Path $_.DirectoryName 'effective_config.ini')) -eq $desiredRelativeAdvantage) -and
                ((GetQpFailurePenalty (Join-Path $_.DirectoryName 'effective_config.ini')) -eq $desiredQpPenalty) -and
                ((GetLowSpeedBlend (Join-Path $_.DirectoryName 'effective_config.ini')) -eq $desiredLowSpeedBlend) -and
                (MatchesTrainingDistribution (Join-Path $_.DirectoryName 'effective_config.ini'))
            } | Sort-Object LastWriteTime -Descending)
    }
    if ($Fresh -and ($WarmStart -or $ResumeState)) { throw 'Fresh cannot be combined with WarmStart or ResumeState.' }
    if (-not $Fresh -and -not $WarmStart -and -not $ResumeState) {
        if ($previous.Count -gt 0) {
            $ResumeState = $previous[0].FullName
        }
    }
    if ($Seed -lt 0) { $Seed = 42 + $previous.Count }
    $modelName = if ($Algorithm -eq 'ddpg') { 'ddpg_policy.txt' } else { 'td3_policy.txt' }
    $cycle = Get-Date -Format 'yyyyMMdd-HHmmss-fff'
    $cycleRoot = Join-Path $iterationRoot ("cycle_" + $cycle)
    if (Test-Path -LiteralPath $cycleRoot) { throw "Output already exists: $cycleRoot" }
    $culture = [System.Globalization.CultureInfo]::InvariantCulture
    $programArgs = @(
        '--config', $Config, '--rl-mode', 'train', '--rl-episodes', "$Episodes",
        '--set', "rl.algorithm=$Algorithm",
        '--set', "rl.seed=$Seed",
        '--set', ('rl.train_speed_min_kmh=' + $SpeedMinKmh.ToString($culture)),
        '--set', ('rl.train_speed_max_kmh=' + $SpeedMaxKmh.ToString($culture)),
        '--set', ('rl.train_mu_min=' + $MuMin.ToString($culture)),
        '--set', ('rl.train_mu_max=' + $MuMax.ToString($culture)),
        '--set', "rl.model_file=$modelName",
        '--set', 'carsim.sync_browser_result=false', '--no-open', '--out', $cycleRoot
    )
    if ($Simfile) { $programArgs += @('--set', "carsim.simfile=$Simfile") }
    if ($Algorithm -eq 'ddpg' -or $Curriculum) { $programArgs += @('--set', 'rl.curriculum=true') }
    if ($FocusTargetSpeeds) { $programArgs += @('--set', 'rl.focus_target_speeds=true') }
    if ($VaryRoadTransition) { $programArgs += @('--set', 'rl.vary_road_transition=true') }
    if ($ResumeState) {
        if (-not (Test-Path -LiteralPath $ResumeState -PathType Leaf)) { throw "State missing: $ResumeState" }
        if (Test-Path -LiteralPath (Join-Path (Split-Path -Parent (Resolve-Path -LiteralPath $ResumeState).Path) 'invalid_training.txt') -PathType Leaf) {
            throw 'Cannot resume a training run marked invalid.'
        }
        $programArgs += @('--set', "rl.resume_state_file=$ResumeState")
        if (-not $PSBoundParameters.ContainsKey('ResumeState')) {
            $programArgs += @('--set', 'rl.resume_actor_fallback=true')
        }
        Write-Host "Continuing complete training state: $ResumeState"
    } elseif ($WarmStart) {
        if (-not (Test-Path -LiteralPath $WarmStart -PathType Leaf)) { throw "Policy missing: $WarmStart" }
        if (Test-Path -LiteralPath (Join-Path (Split-Path -Parent (Resolve-Path -LiteralPath $WarmStart).Path) 'invalid_training.txt') -PathType Leaf) {
            throw 'Cannot warm-start from a training run marked invalid.'
        }
        $programArgs += @('--set', "rl.warm_start_file=$WarmStart")
        Write-Host "Initializing from actor only: $WarmStart"
    } else {
        Write-Host 'Training a new policy from scratch.'
    }
    & $exe @programArgs
    if ($LASTEXITCODE -ne 0) { throw "Training failed (exit code $LASTEXITCODE)." }
    $run = Get-ChildItem -LiteralPath $cycleRoot -Directory | Select-Object -First 1
    if (-not $run) { throw 'Training result directory is missing.' }
    $model = Join-Path $run.FullName $modelName
    Write-Host "Training report: $(Join-Path $run.FullName 'rl_report.html')"
    Write-Host "Selected policy: $model"
    Write-Host "Resume from last training state: $(Join-Path $run.FullName 'last_training_state.txt')"
    Write-Host "Resume from selected checkpoint: $(Join-Path $run.FullName 'best_training_state.txt')"
    & (Join-Path $PSScriptRoot 'update_policy_catalog.ps1')
    if (-not $NoEvaluation -and $SpeedMinKmh -eq $SpeedMaxKmh) {
        Write-Host "Fixed-speed training has no independent speed grid; inspect the selected $SpeedMinKmh km/h checkpoint and run a same-condition comparison."
    }
    if (-not $NoEvaluation -and $SpeedMinKmh -ne $SpeedMaxKmh) {
        $evalRoot = Join-Path (Join-Path 'policies\evaluations\iterations' $Algorithm) ("cycle_" + $cycle)
        $evalArgs = @(
            '--config', $Config, '--rl-mode', 'eval_general', '--rl-model', $model,
            '--set', "rl.algorithm=$Algorithm",
            '--set', ('rl.train_speed_min_kmh=' + $SpeedMinKmh.ToString($culture)),
            '--set', ('rl.train_speed_max_kmh=' + $SpeedMaxKmh.ToString($culture)),
            '--set', ('rl.train_mu_min=' + $MuMin.ToString($culture)),
            '--set', ('rl.train_mu_max=' + $MuMax.ToString($culture)),
            '--set', 'carsim.sync_browser_result=false', '--no-open', '--out', $evalRoot
        )
        if ($Simfile) { $evalArgs += @('--set', "carsim.simfile=$Simfile") }
        & $exe @evalArgs
        if ($LASTEXITCODE -ne 0) { throw "Generalization evaluation failed (exit code $LASTEXITCODE)." }
        $eval = Get-ChildItem -LiteralPath $evalRoot -Directory | Select-Object -First 1
        Write-Host "Generalization report: $(Join-Path $eval.FullName 'generalization_report.html')"
        if (-not $NoPromotion) {
            $decision = & (Join-Path $PSScriptRoot 'promote_general_policy.ps1') -Algorithm $Algorithm -CandidateRun $run.FullName -CandidateEvaluation $eval.FullName
            Write-Host "Promotion result: $($decision.reason)"
        }
    }
}
finally {
    Pop-Location
}
