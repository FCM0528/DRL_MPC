param(
    [string]$DdpgPolicy = '',
    [string]$Td3Policy = '',
    [double]$DdpgMuMin = 0.3,
    [double]$DdpgMuMax = 0.9,
    [double]$Td3MuMin = 0.3,
    [double]$Td3MuMax = 0.9,
    [double]$TrainSpeedMinKmh = 30,
    [double]$TrainSpeedMaxKmh = 120,
    [string]$OutputRoot = '',
    [string]$Simfile = '',
    [string]$Executable = '',
    [string]$Config = 'configs\carsim_compact_training.ini',
    [switch]$AllowModelMismatch,
    [switch]$NoOpen
)

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Push-Location $projectRoot
try {
    $fixedConfig = Split-Path -Leaf $Config
    if ($fixedConfig -in @('carsim_fixed60_np_focus.ini','carsim_fixed80_np_focus.ini')) {
        $fixedSpeed = if ($fixedConfig -eq 'carsim_fixed60_np_focus.ini') { 60 } else { 80 }
        if (-not $PSBoundParameters.ContainsKey('TrainSpeedMinKmh')) { $TrainSpeedMinKmh = $fixedSpeed }
        if (-not $PSBoundParameters.ContainsKey('TrainSpeedMaxKmh')) { $TrainSpeedMaxKmh = $fixedSpeed }
    }
    if ((Split-Path -Leaf $Config) -in @('carsim_focus_60_80.ini','carsim_focus_60_80_tracking.ini','carsim_focus_fast.ini','carsim_focus_fast_guarded.ini')) {
        if (-not $PSBoundParameters.ContainsKey('TrainSpeedMinKmh')) { $TrainSpeedMinKmh = 55 }
        if (-not $PSBoundParameters.ContainsKey('TrainSpeedMaxKmh')) { $TrainSpeedMaxKmh = 85 }
    }
    $exe = if ($Executable) { $Executable } else { '.\out\build\x64-Release\Release\MPC_DoubleLaneChange.exe' }
    $useRegisteredDdpg = -not $DdpgPolicy
    $useRegisteredTd3 = -not $Td3Policy
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
        $settings = @{ nonlinear_tire='0'; tire_slope_floor='0.15'; grip_switch_enabled='1'; grip_state_only='0'; grip_preview_s='0.8'; speed_rate_enabled='0'; speed_rate_start_kmh='60'; speed_rate_end_kmh='120'; speed_rate_end_deg_s='5'; rho='20000'; max_iterations='2500'; primal_tol='1e-5'; dual_tol='0.002'; low_speed_blend='0'; blend_start_kmh='60'; blend_end_kmh='80'; train_speed_min_kmh='30'; train_speed_max_kmh='120'; decision_steps='5'; max_steer_deg='30'; max_steer_rate_deg_s='45'; high_speed_blend='0'; high_blend_start_kmh='85'; high_blend_end_kmh='95'; augmented_observation='0'; grip_demand_clip='3' }
        $section=''
        foreach($line in ConfigLines $ini) {
            if($line -match '^\s*\[([^]]+)\]') { $section=$Matches[1].ToLowerInvariant(); continue }
            if($section -in @('mpc','solver','rl','constraints') -and $line -match '^\s*([a-zA-Z_]+)\s*=\s*([^#;\s]+)') {
                $key=$Matches[1].ToLowerInvariant(); if($settings.ContainsKey($key)) { $settings[$key]=$Matches[2].ToLowerInvariant() }
            }
        }
        foreach($key in @('nonlinear_tire','grip_switch_enabled','grip_state_only','speed_rate_enabled','low_speed_blend','high_speed_blend','augmented_observation')) {
            if($settings[$key] -in @('true','yes','1')){$settings[$key]='1'}
            elseif($settings[$key] -in @('false','no','0')){$settings[$key]='0'}
            else { throw "Invalid $key in $ini" }
        }
        foreach($key in @('tire_slope_floor','grip_preview_s','speed_rate_start_kmh','speed_rate_end_kmh','speed_rate_end_deg_s','rho','max_iterations','primal_tol','dual_tol','blend_start_kmh','blend_end_kmh','train_speed_min_kmh','train_speed_max_kmh','decision_steps','max_steer_deg','max_steer_rate_deg_s','high_blend_start_kmh','high_blend_end_kmh','grip_demand_clip')) {
            $settings[$key]=([double]::Parse($settings[$key],[cultureinfo]::InvariantCulture)).ToString('G12',[cultureinfo]::InvariantCulture)
        }
        return ($settings.nonlinear_tire, $settings.tire_slope_floor, $settings.grip_switch_enabled, $settings.grip_state_only, $settings.grip_preview_s, $settings.speed_rate_enabled, $settings.speed_rate_start_kmh, $settings.speed_rate_end_kmh, $settings.speed_rate_end_deg_s, $settings.rho, $settings.max_iterations, $settings.primal_tol, $settings.dual_tol, $settings.low_speed_blend, $settings.blend_start_kmh, $settings.blend_end_kmh, $settings.train_speed_min_kmh, $settings.train_speed_max_kmh, $settings.decision_steps, $settings.max_steer_deg, $settings.max_steer_rate_deg_s, $settings.high_speed_blend, $settings.high_blend_start_kmh, $settings.high_blend_end_kmh, $settings.augmented_observation, $settings.grip_demand_clip) -join '|'
    }
    $selectionModel = ModelSignature $Config
    $configLeaf = Split-Path -Leaf $Config
    if ($configLeaf -in @('carsim_fixed60_np_focus.ini','carsim_fixed80_np_focus.ini')) {
        $speed = if ($configLeaf -eq 'carsim_fixed60_np_focus.ini') { 60 } else { 80 }
        foreach ($algorithm in @('ddpg','td3')) {
            if (($algorithm -eq 'ddpg' -and $DdpgPolicy) -or ($algorithm -eq 'td3' -and $Td3Policy)) { continue }
            $dir=Join-Path (Join-Path 'policies\fixed_speed' "v$speed") $algorithm
            $policy=Join-Path $dir ($algorithm+'_policy.txt')
            $manifest=Join-Path $dir 'pinned_policy.csv'
            if (!(Test-Path -LiteralPath $policy -PathType Leaf) -or !(Test-Path -LiteralPath $manifest -PathType Leaf)) { throw "Pinned $algorithm policy is missing for $speed km/h." }
            $record=Import-Csv -LiteralPath $manifest | Select-Object -First 1
            if ($record.algorithm -ne $algorithm -or [int]$record.speed_kmh -ne $speed -or
                (Get-FileHash -LiteralPath $policy -Algorithm SHA256).Hash -ne $record.sha256) { throw "Pinned $algorithm policy checksum or metadata mismatch." }
            if ($algorithm -eq 'ddpg') { $DdpgPolicy=$policy; $useRegisteredDdpg=$false }
            else { $Td3Policy=$policy; $useRegisteredTd3=$false }
            Write-Host "Using pinned $algorithm policy for $speed km/h: $policy"
        }
    }
    if ($configLeaf -in @('carsim_np_only.ini','carsim_compact_training.ini','carsim_compact_relative.ini','carsim_compact_guarded.ini','carsim_compact_blend.ini','carsim_focus_60_80.ini','carsim_focus_60_80_tracking.ini','carsim_focus_fast.ini','carsim_focus_fast_guarded.ini','carsim_fixed60_np_focus.ini','carsim_fixed80_np_focus.ini')) {
        $policyTag = if ($configLeaf -eq 'carsim_np_only.ini') { 'V19_NP_ONLY' } else { 'V18_COMPACT' }
        foreach ($algorithm in @('ddpg', 'td3')) {
            if (($algorithm -eq 'ddpg' -and $DdpgPolicy) -or ($algorithm -eq 'td3' -and $Td3Policy)) { continue }
            $root = Join-Path 'policies\iterations' $algorithm
            $candidate = @(Get-ChildItem -LiteralPath $root -Recurse -File -Filter ($algorithm + '_policy.txt') -ErrorAction SilentlyContinue |
                Where-Object {
                    (Test-Path -LiteralPath (Join-Path $_.DirectoryName 'rl_report.html') -PathType Leaf) -and
                    -not (Test-Path -LiteralPath (Join-Path $_.DirectoryName 'invalid_training.txt') -PathType Leaf) -and
                    (Test-Path -LiteralPath (Join-Path $_.DirectoryName 'controller_revision.txt') -PathType Leaf) -and
                    ((Get-Content -LiteralPath (Join-Path $_.DirectoryName 'controller_revision.txt') -Raw).Trim() -eq 'ltv-warm-sequence-v1') -and
                (Test-Path -LiteralPath (Join-Path $_.DirectoryName 'effective_config.ini') -PathType Leaf) -and
                ((ModelSignature (Join-Path $_.DirectoryName 'effective_config.ini')) -eq $selectionModel) -and
                    ((Get-Content -LiteralPath $_.FullName -TotalCount 1) -match ('^' + $algorithm.ToUpperInvariant() + '_MPC_' + $policyTag + '_'))
                } | Sort-Object LastWriteTime -Descending | Select-Object -First 1)
            if ($candidate.Count -eq 0) { throw "No completed $policyTag $algorithm policy found; train it first or specify policy paths." }
            if ($algorithm -eq 'ddpg') { $DdpgPolicy = $candidate[0].FullName; $useRegisteredDdpg = $false }
            else { $Td3Policy = $candidate[0].FullName; $useRegisteredTd3 = $false }
            Write-Host "Using latest completed $policyTag $algorithm policy: $($candidate[0].FullName)"
        }
    }
    if (-not $DdpgPolicy -or -not $Td3Policy) {
        $registry = 'policies\active_registry.csv'
        if (-not (Test-Path -LiteralPath $registry -PathType Leaf)) {
            throw "Active policy registry not found: $registry"
        }
        $registered = @(Import-Csv -LiteralPath $registry)
        foreach ($algorithm in @('ddpg', 'td3')) {
            $needed = ($algorithm -eq 'ddpg' -and -not $DdpgPolicy) -or
                      ($algorithm -eq 'td3' -and -not $Td3Policy)
            if (-not $needed) { continue }
            $entry = @($registered | Where-Object { $_.algorithm -eq $algorithm })
            if ($entry.Count -ne 1) { throw "Expected one registered $algorithm policy." }
            $record = $entry[0]
            if (-not (Test-Path -LiteralPath $record.policy_file -PathType Leaf) -or
                (Get-FileHash -LiteralPath $record.policy_file -Algorithm SHA256).Hash -ne $record.sha256) {
                throw "Registered $algorithm policy is missing or its checksum changed."
            }
            if ($algorithm -eq 'ddpg') {
                $DdpgPolicy = $record.policy_file
                if (-not $PSBoundParameters.ContainsKey('DdpgMuMin')) { $DdpgMuMin = [double]::Parse($record.mu_min, [cultureinfo]::InvariantCulture) }
                if (-not $PSBoundParameters.ContainsKey('DdpgMuMax')) { $DdpgMuMax = [double]::Parse($record.mu_max, [cultureinfo]::InvariantCulture) }
            } else {
                $Td3Policy = $record.policy_file
                if (-not $PSBoundParameters.ContainsKey('Td3MuMin')) { $Td3MuMin = [double]::Parse($record.mu_min, [cultureinfo]::InvariantCulture) }
                if (-not $PSBoundParameters.ContainsKey('Td3MuMax')) { $Td3MuMax = [double]::Parse($record.mu_max, [cultureinfo]::InvariantCulture) }
            }
            Write-Host "Using registered $algorithm policy ($($record.status)): $($record.policy_file)"
        }
    }
    foreach ($path in @($exe, $DdpgPolicy, $Td3Policy, $Config)) {
        if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
            throw "Required file does not exist: $path"
        }
    }
    $runtimeModel = ModelSignature $Config
    $policyNotices = @()
    foreach($entry in @(@('DDPG',$DdpgPolicy),@('TD3',$Td3Policy))) {
        $sourceConfig = Join-Path (Split-Path -Parent (Resolve-Path -LiteralPath $entry[1]).Path) 'effective_config.ini'
        $invalidMarker = Join-Path (Split-Path -Parent $sourceConfig) 'invalid_training.txt'
        if ((Test-Path -LiteralPath $invalidMarker -PathType Leaf) -and -not $AllowModelMismatch) {
            throw "$($entry[0]) policy comes from a training run marked invalid: $invalidMarker"
        }
        $known = Test-Path -LiteralPath $sourceConfig -PathType Leaf
        $independentStatus = Join-Path (Split-Path -Parent $sourceConfig) 'independent_validation_status.txt'
        if (Test-Path -LiteralPath $independentStatus -PathType Leaf) {
            $status = (Get-Content -LiteralPath $independentStatus -Raw).Trim()
            if ($status -match '^FAILED') {
                $notice = "$($entry[0]): independent validation failed: $status"
                Write-Warning $notice
                $policyNotices += $notice
            }
        }
        $selection = Join-Path (Split-Path -Parent $sourceConfig) 'selected_validation.csv'
        $policyName = Split-Path -Leaf $entry[1]
        if ($policyName -match '^checkpoint_\d+_policy\.txt$' -or $policyName -eq 'last_unvalidated_policy.txt') {
            $notice = "$($entry[0]): raw checkpoint, not the selected validated policy."
            Write-Warning $notice
            $policyNotices += $notice
        } elseif (Test-Path -LiteralPath $selection -PathType Leaf) {
            $first = Import-Csv -LiteralPath $selection | Select-Object -First 1
            if ($first -and $first.checkpoint_episode -eq '0') {
                $notice = "$($entry[0]): neutral fallback; no learned checkpoint passed selection."
                Write-Warning $notice
                $policyNotices += $notice
            }
            elseif ($first -and $first.checkpoint_episode -eq '-1') {
                $notice = "$($entry[0]): selected inherited Actor; no later checkpoint was selected."
                Write-Warning $notice
                $policyNotices += $notice
            }
        }
        $different = -not $known -or (ModelSignature $sourceConfig) -ne $runtimeModel
        $revisionPath = Join-Path (Split-Path -Parent $sourceConfig) 'controller_revision.txt'
        if (-not (Test-Path -LiteralPath $revisionPath -PathType Leaf) -or
            (Get-Content -LiteralPath $revisionPath -Raw).Trim() -ne 'ltv-warm-sequence-v1') { $different = $true }
        if ($known) {
            $speedMinLine = @(ConfigLines $sourceConfig | Where-Object { $_ -match '^\s*train_speed_min_kmh\s*=' } | Select-Object -Last 1)
            $speedMaxLine = @(ConfigLines $sourceConfig | Where-Object { $_ -match '^\s*train_speed_max_kmh\s*=' } | Select-Object -Last 1)
            if ($speedMinLine.Count -and $speedMaxLine.Count) {
                $sourceMin = [double]::Parse(($speedMinLine[0] -split '=')[1].Trim(), [cultureinfo]::InvariantCulture)
                $sourceMax = [double]::Parse(($speedMaxLine[0] -split '=')[1].Trim(), [cultureinfo]::InvariantCulture)
                if ($sourceMin -ne $TrainSpeedMinKmh -or $sourceMax -ne $TrainSpeedMaxKmh) { $different = $true }
            }
        }
        if($different) {
            $message = "$($entry[0]) policy training model differs from or is not recorded for $Config; retrain or explicitly use -AllowModelMismatch for diagnostic evaluation."
            if(-not $AllowModelMismatch){ throw $message }
            Write-Warning $message
            $policyNotices += $message
        }
    }
    if (-not $OutputRoot) {
        $OutputRoot = Join-Path 'results' ("three_controller_compare_" + (Get-Date -Format 'yyyyMMdd-HHmmss-fff'))
    }
    if (Test-Path -LiteralPath $OutputRoot) {
        throw "Output directory already exists: $OutputRoot"
    }
    $culture = [System.Globalization.CultureInfo]::InvariantCulture
    $programArgs = @('--config', $Config, '--rl-mode', 'compare_three', '--rl-model', $DdpgPolicy, '--td3-model', $Td3Policy, '--set', ('rl.train_speed_min_kmh=' + $TrainSpeedMinKmh.ToString($culture)), '--set', ('rl.train_speed_max_kmh=' + $TrainSpeedMaxKmh.ToString($culture)), '--set', ('rl.train_mu_min=' + $DdpgMuMin.ToString($culture)), '--set', ('rl.train_mu_max=' + $DdpgMuMax.ToString($culture)), '--td3-mu-min', $Td3MuMin.ToString($culture), '--td3-mu-max', $Td3MuMax.ToString($culture), '--set', 'carsim.sync_browser_result=false', '--out', $OutputRoot)
    if ($Simfile) {
        if (-not (Test-Path -LiteralPath $Simfile -PathType Leaf)) { throw "Simfile missing: $Simfile" }
        $programArgs += @('--set', "carsim.simfile=$Simfile")
    }
    $routeRegistry = 'policies\region_registry.csv'
    $legacyModel = '0|0.15|1|0|0.8|0|60|120|5|20000|2500|1E-05|0.002|0|60|80'
    if ((Test-Path -LiteralPath $routeRegistry -PathType Leaf) -and $runtimeModel -ne $legacyModel) {
        Write-Host 'Regional expert routing disabled because the MPC model or protection settings differ from its validation.'
    }
    if ((Test-Path -LiteralPath $routeRegistry -PathType Leaf) -and $runtimeModel -eq $legacyModel) {
        $routes = @(Import-Csv -LiteralPath $routeRegistry)
        $sharedDomain = ''
        foreach ($route in $routes) {
            $useRoute = ($route.algorithm -eq 'ddpg' -and $useRegisteredDdpg) -or
                        ($route.algorithm -eq 'td3' -and $useRegisteredTd3)
            if (-not $useRoute -or $route.status -ne 'validated_against_registered_general') { continue }
            $generalPolicy = if ($route.algorithm -eq 'ddpg') { $DdpgPolicy } else { $Td3Policy }
            if ((Get-FileHash -LiteralPath $generalPolicy -Algorithm SHA256).Hash -ne $route.reference_general_sha256) {
                Write-Host "Skipping $($route.algorithm) region expert: registered general policy changed and requires re-evaluation."
                continue
            }
            if (-not (Test-Path -LiteralPath $route.policy_file -PathType Leaf) -or
                (Get-FileHash -LiteralPath $route.policy_file -Algorithm SHA256).Hash -ne $route.sha256) {
                throw "Registered region expert is missing or changed: $($route.policy_file)"
            }
            $domain = $route.mu + '|' + $route.speed_min_kmh + '|' + $route.speed_max_kmh
            if ($sharedDomain -and $sharedDomain -ne $domain) {
                throw 'The current comparison supports one shared expert domain per run.'
            }
            if (-not $sharedDomain) {
                $programArgs += @('--expert-mu', $route.mu, '--expert-speed-min', $route.speed_min_kmh, '--expert-speed-max', $route.speed_max_kmh)
                $sharedDomain = $domain
            }
            if ($route.algorithm -eq 'ddpg') {
                $programArgs += @('--ddpg-expert', $route.policy_file)
            } else {
                $programArgs += @('--td3-expert', $route.policy_file)
            }
            Write-Host "Enabled $($route.algorithm) expert at mu=$($route.mu), $($route.speed_min_kmh)-$($route.speed_max_kmh) km/h."
        }
    }
    $programArgs += '--no-open'
    & $exe @programArgs
    if ($LASTEXITCODE -ne 0) { throw "Three-controller comparison failed (exit code $LASTEXITCODE)." }
    $run = Get-ChildItem -LiteralPath $OutputRoot -Directory | Select-Object -First 1
    if (-not $run) { throw 'Comparison result directory is missing.' }
    if ($policyNotices.Count -gt 0) {
        $reportPath = Join-Path $run.FullName 'comparison_report.html'
        $page = [IO.File]::ReadAllText($reportPath)
        $noticeHtml = '<p style="color:#a00;font-weight:bold">' +
            (($policyNotices | ForEach-Object { [System.Net.WebUtility]::HtmlEncode($_) }) -join '<br>') + '</p>'
        $page = $page.Replace('</h1>', '</h1>' + $noticeHtml)
        [IO.File]::WriteAllText($reportPath, $page, [Text.UTF8Encoding]::new($false))
    }
    if (-not $NoOpen) { Invoke-Item -LiteralPath (Join-Path $run.FullName 'comparison_report.html') }
    $summary = Join-Path $run.FullName 'comparison_metrics.csv'
    $rows = @(Import-Csv -LiteralPath $summary)
    if ($rows.Count -ne 3) { throw 'Expected one result for each of fixed, DDPG and TD3.' }
    $rows | Format-Table controller, completed, rmse_y_m, max_abs_y_m, rmse_heading_deg -AutoSize
    Write-Host "Three-controller metrics: $summary"
    Write-Host "Three-controller report: $(Join-Path $run.FullName 'comparison_report.html')"
}
finally {
    Pop-Location
}
