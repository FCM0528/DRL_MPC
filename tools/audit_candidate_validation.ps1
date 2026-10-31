param([string]$FixtureRoot='policies\tgt\b2\f', [string]$OutputRoot='policies\audit\val', [string]$Executable='.\out\build\x64-CompareThree\Release\MPC_DoubleLaneChange.exe', [switch]$KeepSwitch, [switch]$StateSwitch)
$ErrorActionPreference='Stop'
if(Test-Path -LiteralPath $OutputRoot){throw 'Output exists'}
New-Item -ItemType Directory -Path $OutputRoot -Force | Out-Null
$results=@()
foreach($case in @('a60','a80','b60','b80','c60','c80','d60','d80','e60','e80')){
 foreach($variant in @(@('n20q100',20,100,350),@('n30q100',30,100,350),@('n30q300',30,300,1000))){
  $tag="$case-$($variant[0])";$out=Join-Path $OutputRoot $tag
  $args=@('--rl-mode','fixed','--np',"$($variant[1])",'--set',"weights.q_y=$($variant[2])",'--set',"weights.r_delta=$($variant[3])",'--set','mpc.nonlinear_tire=true','--set',"carsim.simfile=$(Join-Path $FixtureRoot "$case\simfile.sim")",'--set','carsim.sync_browser_result=false','--no-open','--out',$out)
  if($StateSwitch){$args+=@('--set','mpc.grip_state_only=true')}
  elseif(-not $KeepSwitch){$args+=@('--set','mpc.grip_switch_enabled=false')}
  & $Executable @args | Out-Null
  if($LASTEXITCODE -ne 0){throw "Run failed $tag"}
  $run=@(Get-ChildItem -LiteralPath $out -Directory);if($run.Count -ne 1){throw "Ambiguous $tag"}
  $rows=@(Import-Csv (Join-Path $run[0].FullName 'simulation.csv'));$sum=0.0;$peak=0.0;$limited=0;$fallback=0
  foreach($r in $rows){$e=[double]$r.ey_m;$sum+=$e*$e;$peak=[Math]::Max($peak,[Math]::Abs($e));if($r.grip_limited_control -eq '1'){$limited++};if($r.fallback_control -eq '1'){$fallback++}}
  $result=[pscustomobject]@{case=$case;variant=$variant[0];rmse=[Math]::Sqrt($sum/$rows.Count);peak=$peak;completed=([double]$rows[-1].X_carsim_m -ge 180);limited=$limited;fallback=$fallback;report=(Join-Path $run[0].FullName 'report.html')}
  $results+=$result;Write-Host "$tag RMSE=$($result.rmse) max=$peak completed=$($result.completed)"
 }
}
$results|Export-Csv -LiteralPath (Join-Path $OutputRoot 'candidate_validation.csv') -NoTypeInformation
$results|Format-Table case,variant,rmse,peak,completed,limited,fallback
