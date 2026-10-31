param([string]$Simfile, [string]$OutputRoot, [string]$Executable)
$ErrorActionPreference='Stop'
if (Test-Path -LiteralPath $OutputRoot) { throw 'Output root already exists' }
New-Item -ItemType Directory -Path $OutputRoot -Force | Out-Null
$all=@()
foreach($qpsi in @(350,1000,2500)) {
 foreach($qvy in @(0.4,4,20)) {
  foreach($qr in @(0.5,8,40)) {
   $case="p${qpsi}v${qvy}r${qr}"; $out=Join-Path $OutputRoot $case
   $args=@('--rl-mode','fixed','--np','30','--set','weights.q_y=800','--set',"weights.q_psi=$qpsi",'--set',"weights.q_vy=$qvy",'--set',"weights.q_r=$qr",'--set','mpc.grip_switch_enabled=false','--set',"carsim.simfile=$Simfile",'--set','carsim.sync_browser_result=false','--no-open','--out',$out)
   & $Executable @args | Out-Null
   if ($LASTEXITCODE -ne 0) { throw "Sweep failed: $case" }
   $run=@(Get-ChildItem -LiteralPath $out -Directory); if($run.Count -ne 1){throw "Ambiguous run: $case"}
   $csv=@(Import-Csv -LiteralPath (Join-Path $run[0].FullName 'simulation.csv'))
   $sum=0.0; $peak=0.0; $limited=0; $failure=0
   foreach($row in $csv) {$e=[double]$row.ey_m;$sum+=$e*$e;$peak=[Math]::Max($peak,[Math]::Abs($e));if($row.grip_limited_control -eq '1'){$limited++};if($row.fallback_control -eq '1'){$failure++}}
   $all += [pscustomobject]@{case=$case;qpsi=$qpsi;qvy=$qvy;qr=$qr;rmse=[Math]::Sqrt($sum/$csv.Count);peak=$peak;limited=$limited;fallback=$failure;run=$run[0].FullName}
   Write-Host "$case RMSE=$($all[-1].rmse) max=$peak"
  }
 }
}
$all | Sort-Object rmse | Export-Csv -LiteralPath (Join-Path $OutputRoot 'sweep.csv') -NoTypeInformation
$all | Sort-Object rmse | Select-Object -First 8 | Format-Table
