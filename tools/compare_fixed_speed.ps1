param(
    [Parameter(Mandatory=$true)][ValidateSet(60,80)][int]$Speed,
    [string]$OutputRoot='',
    [switch]$NoOpen
)
$ErrorActionPreference='Stop'
Push-Location (Split-Path -Parent $PSScriptRoot)
try {
    $base=Join-Path 'policies\fixed_speed' ("v$Speed")
    $files=@{}
    foreach($algorithm in @('ddpg','td3')){
        $dir=Join-Path $base $algorithm
        $manifest=Join-Path $dir 'pinned_policy.csv'
        $file=Join-Path $dir ($algorithm+'_policy.txt')
        if(!(Test-Path -LiteralPath $manifest -PathType Leaf) -or !(Test-Path -LiteralPath $file -PathType Leaf)){throw "Pinned $algorithm policy missing for $Speed km/h."}
        $metadata=Import-Csv -LiteralPath $manifest | Select-Object -First 1
        if($metadata.algorithm -ne $algorithm -or [int]$metadata.speed_kmh -ne $Speed -or
           (Get-FileHash -LiteralPath $file -Algorithm SHA256).Hash -ne $metadata.sha256){throw "Pinned $algorithm policy metadata or checksum mismatch."}
        foreach($entry in @(@('training_source_Run_all.par','source_sha256'),@('effective_config.ini','config_sha256'),@('tire_data.csv','tire_sha256'))){
            $path=Join-Path $dir $entry[0]
            if(!(Test-Path -LiteralPath $path -PathType Leaf) -or (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash -ne $metadata.($entry[1])){throw "Pinned $algorithm $($entry[0]) checksum mismatch."}
        }
        $source=Join-Path (Join-Path $base 'source') 'Run_all.par'
        if(!(Test-Path -LiteralPath $source -PathType Leaf) -or (Get-FileHash -LiteralPath $source -Algorithm SHA256).Hash -ne $metadata.source_sha256){throw "Fixed $Speed km/h snapshot differs from the $algorithm training source."}
        if($metadata.status -eq 'neutral_fallback'){Write-Warning "$algorithm $Speed km/h is neutral fallback."}
        elseif($metadata.status -eq 'inherited_actor'){Write-Warning "$algorithm $Speed km/h selected its inherited Actor."}
        $files[$algorithm]=$file
    }
    $config="configs\carsim_fixed${Speed}_np_focus.ini"
    $sim=Join-Path (Join-Path $base 'source') 'simfile.sim'
    if(!$OutputRoot){$OutputRoot=Join-Path 'policies\comparisons' ("fixed_v${Speed}_"+(Get-Date -Format 'yyyyMMdd-HHmmss-fff'))}
    $params=@{Config=$config;Simfile=$sim;DdpgPolicy=$files.ddpg;Td3Policy=$files.td3;OutputRoot=$OutputRoot}
    if($NoOpen){$params.NoOpen=$true}
    & (Join-Path $PSScriptRoot 'compare_three_controllers.ps1') @params
} finally {Pop-Location}
