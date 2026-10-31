param(
    [string]$Config = 'configs\carsim_fixed60_np_focus.ini',
    [string]$Simfile = 'policies\fixed_speed\v60\source\simfile.sim',
    [string]$Policy = '',
    [string]$OutputRoot = '',
    [switch]$NoOpen
)
$ErrorActionPreference='Stop'
Push-Location (Split-Path -Parent $PSScriptRoot)
try {
    function ConfigLines([string]$ini, [string[]]$chain=@()) {
        $path=(Resolve-Path -LiteralPath $ini).Path
        if($chain -contains $path){throw "Circular inheritance: $path"}
        $lines=@(Get-Content -LiteralPath $path);$section=''
        foreach($line in $lines){
            if($line -match '^\s*\[([^]]+)\]'){$section=$Matches[1].ToLowerInvariant();continue}
            if($section -eq 'inherit' -and $line -match '^\s*base\s*=\s*([^#;]+)'){
                ConfigLines (Join-Path (Split-Path -Parent $path) $Matches[1].Trim()) ($chain+$path)
            }
        }
        $lines
    }
    function RuntimeSettings([string]$ini) {
        $settings=@{};$section=''
        $rlKeys=@('action_profile','min_np','max_np','nominal_np','decision_steps','augmented_observation','grip_demand_clip','discount_gamma','np_exploration_scale','strict_tracking_selection','q_y_min','q_y_max','q_psi_min','q_psi_max','q_vy_min','q_vy_max','q_r_min','q_r_max','r_steer_min','r_steer_max','r_delta_min','r_delta_max','terminal_scale','curvature_preview_s','low_speed_blend','blend_start_kmh','blend_end_kmh','high_speed_blend','high_blend_start_kmh','high_blend_end_kmh','train_speed_min_kmh','train_speed_max_kmh','train_mu_min','train_mu_max')
        foreach($line in ConfigLines $ini){
            if($line -match '^\s*\[([^]]+)\]'){$section=$Matches[1].ToLowerInvariant();continue}
            if($line -match '^\s*([a-zA-Z_]+)\s*=\s*([^#;\s]+)'){
                $key=$Matches[1].ToLowerInvariant();$value=$Matches[2].ToLowerInvariant()
                if($section -in @('mpc','constraints','solver','weights') -or ($section -eq 'rl' -and $key -in $rlKeys)){
                    if($value -in @('true','yes','1')){$value='1'}elseif($value -in @('false','no','0')){$value='0'}
                    else {$num=0.0;if([double]::TryParse($value,[Globalization.NumberStyles]::Float,[cultureinfo]::InvariantCulture,[ref]$num)){$value=$num.ToString('G12',[cultureinfo]::InvariantCulture)}}
                    $settings[$section+'.'+$key]=$value
                }
            }
        }
        return $settings
    }
    $wanted=RuntimeSettings $Config
    if($wanted['rl.train_speed_min_kmh'] -ne '60' -or $wanted['rl.train_speed_max_kmh'] -ne '60') {throw 'This command requires a 60 km/h fixed-speed configuration.'}
    function Compatible([string]$file){
        $dir=Split-Path -Parent $file;$effective=Join-Path $dir 'effective_config.ini';$revision=Join-Path $dir 'controller_revision.txt'
        if(!(Test-Path -LiteralPath $effective -PathType Leaf) -or !(Test-Path -LiteralPath $revision -PathType Leaf)){return $false}
        if((Get-Content -LiteralPath $revision -Raw).Trim() -ne 'ltv-warm-sequence-v1'){return $false}
        $actual=RuntimeSettings $effective
        foreach($key in $wanted.Keys){if(!$actual.ContainsKey($key) -or $actual[$key] -ne $wanted[$key]){return $false}}
        return $true
    }
    if(-not $Policy){
        $Policy='policies\fixed_speed\v60\ddpg\ddpg_policy.txt'
        $manifest='policies\fixed_speed\v60\ddpg\pinned_policy.csv'
        if(!(Test-Path -LiteralPath $Policy -PathType Leaf) -or !(Test-Path -LiteralPath $manifest -PathType Leaf)){throw 'Pinned fixed-60 DDPG policy is missing.'}
        $record=Import-Csv -LiteralPath $manifest|Select-Object -First 1
        if($record.algorithm -ne 'ddpg' -or [int]$record.speed_kmh -ne 60 -or (Get-FileHash -LiteralPath $Policy -Algorithm SHA256).Hash -ne $record.sha256){throw 'Pinned fixed-60 DDPG policy checksum mismatch.'}
    }
    if(!(Test-Path -LiteralPath $Policy -PathType Leaf) -or !(Compatible (Resolve-Path -LiteralPath $Policy).Path)){throw 'Policy settings differ from the requested controller configuration.'}
    $isRaw = (Split-Path -Leaf $Policy) -match '^checkpoint_\d+_policy\.txt$'
    $selection=Join-Path (Split-Path -Parent (Resolve-Path -LiteralPath $Policy).Path) 'selected_validation.csv'
    $notice=''
    if($isRaw){$notice='Raw checkpoint is for diagnostics and was not necessarily selected.'}
    elseif(Test-Path -LiteralPath $selection){
        $row=Import-Csv $selection|Select-Object -First 1
        if($row.checkpoint_episode -eq '0'){$notice='Selected policy is neutral fallback.'}
        elseif($row.checkpoint_episode -eq '-1'){$notice='Selected policy is the inherited warm-start Actor; no later checkpoint was selected.'}
    }
    if($notice){Write-Warning $notice}
    if(!$OutputRoot){$OutputRoot=Join-Path 'policies\comparisons' ('fixed60_ddpg_'+(Get-Date -Format 'yyyyMMdd-HHmmss-fff'))}
    if(Test-Path -LiteralPath $OutputRoot){throw "Output exists: $OutputRoot"}
    $argsList=@('--config',$Config,'--rl-mode','compare','--rl-model',$Policy,'--set',"carsim.simfile=$Simfile",'--set','carsim.sync_browser_result=false','--out',$OutputRoot)
    $argsList+='--no-open'
    Write-Host "Using fixed-60 DDPG policy: $Policy"
    & .\out\build\x64-Release\Release\MPC_DoubleLaneChange.exe @argsList
    if($LASTEXITCODE -ne 0){throw "Comparison failed: $LASTEXITCODE"}
    $run=Get-ChildItem -LiteralPath $OutputRoot -Directory|Select-Object -First 1
    $reportPath=Join-Path $run.FullName 'comparison_report.html'
    if($notice){
        $page=[IO.File]::ReadAllText($reportPath)
        $page=$page.Replace('</h1>','</h1><p style="color:#a00;font-weight:bold">'+[Net.WebUtility]::HtmlEncode($notice)+'</p>')
        [IO.File]::WriteAllText($reportPath,$page,[Text.UTF8Encoding]::new($false))
    }
    if(-not $NoOpen){Invoke-Item -LiteralPath $reportPath}
    Write-Host "Report: $reportPath"
} finally {Pop-Location}
