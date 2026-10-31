param([Parameter(Mandatory=$true)][ValidateSet(60,80)][int]$Speed)
$ErrorActionPreference='Stop'
Push-Location (Split-Path -Parent $PSScriptRoot)
try {
    $base=Join-Path 'policies\fixed_speed' "v$Speed"
    $ddpg=Join-Path (Join-Path $base 'ddpg') 'training_source_Run_all.par'
    $td3=Join-Path (Join-Path $base 'td3') 'training_source_Run_all.par'
    if(!(Test-Path -LiteralPath $ddpg -PathType Leaf) -or !(Test-Path -LiteralPath $td3 -PathType Leaf)){throw 'Both pinned training sources are required.'}
    if((Get-FileHash -LiteralPath $ddpg).Hash -ne (Get-FileHash -LiteralPath $td3).Hash){throw 'DDPG and TD3 were trained on different source snapshots.'}
    $source=Join-Path $base 'source'
    if(Test-Path -LiteralPath $source){throw "Snapshot already exists: $source"}
    New-Item -ItemType Directory -Path $source -Force|Out-Null
    $par=[IO.Path]::GetFullPath((Join-Path $source 'Run_all.par'))
    Copy-Item -LiteralPath $ddpg -Destination $par
    $template="policies\audit\focus60_80_source\v$Speed\simfile.sim"
    $sim=[IO.File]::ReadAllText((Resolve-Path -LiteralPath $template))
    if(([regex]::Matches($sim,'(?m)^INPUT [^\r\n]+')).Count -ne 1){throw 'Expected a single INPUT entry in Simfile.'}
    $sim=[regex]::Replace($sim,'(?m)^INPUT [^\r\n]+',('INPUT '+$par))
    [IO.File]::WriteAllText([IO.Path]::GetFullPath((Join-Path $source 'simfile.sim')),$sim,[Text.UTF8Encoding]::new($false))
    Write-Host "Pinned $Speed km/h source snapshot: $source"
} finally {Pop-Location}
