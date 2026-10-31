param([string]$Simfile, [string]$OutputRoot, [string]$Executable)
$ErrorActionPreference='Stop'
if (Test-Path -LiteralPath $OutputRoot) { throw 'Output root already exists' }
New-Item -ItemType Directory -Path $OutputRoot -Force | Out-Null
$all=@()
foreach($np in @(13,20,30)) {
 foreach($qy in @(100,800)) {
  foreach($preview in @(0.4,0.5,0.6,0.8,1.0,1.2)) {
   $case="n${np}q${qy}p${preview}"; $out=Join-Path $OutputRoot $case
   $args=@('--rl-mode','fixed','--np',"$np",'--set',"weights.q_y=$qy",'--set',"mpc.grip_preview_s=$preview",'--set',"carsim.simfile=$Simfile",'--set','carsim.sync_browser_result=false','--no-open','--out',$out)
   & $Executable @args | Out-Null
   if ($LASTEXITCODE -ne 0) { throw "Sweep failed: $case" }
   $run=@(Get-ChildItem -LiteralPath $out -Directory); if($run.Count -ne 1){throw "Ambiguous run: $case"}
   $csv=@(Import-Csv -LiteralPath (Join-Path $run[0].FullName 'simulation.csv'))
   $sum=0.0; $peak=0.0; $limited=0; $failure=0
   foreach($row in $csv) {$e=[double]$row.ey_m;$sum+=$e*$e;$peak=[Math]::Max($peak,[Math]::Abs($e));if($row.grip_limited_control -eq '1'){$limited++};if($row.fallback_control -eq '1'){$failure++}}
   $all += [pscustomobject]@{case=$case;np=$np;qy=$qy;preview=$preview;rmse=[Math]::Sqrt($sum/$csv.Count);peak=$peak;limited=$limited;fallback=$failure;run=$run[0].FullName}
   Write-Host "$case RMSE=$($all[-1].rmse) max=$peak"
  }
 }
}
$all | Sort-Object rmse | Export-Csv -LiteralPath (Join-Path $OutputRoot 'sweep.csv') -NoTypeInformation
$all | Sort-Object rmse | Select-Object -First 8 | Format-Table
