param([Parameter(Mandatory=$true)][string]$Code)
# Fallback when MCP's previously attached MATLAB process has exited.
$matlabLink = New-Object -ComObject Matlab.Application
$result = $matlabLink.Execute("try, $Code; catch ME, fprintf('DRLMPC_EXECUTION_FAILED\n'); disp(getReport(ME,'extended','hyperlinks','off')); end;")
Write-Output $result
$executionLog = Join-Path (Split-Path $PSScriptRoot -Parent) 'results\execution.log'
Add-Content -LiteralPath $executionLog -Value $result -Encoding utf8
if ($result -match 'DRLMPC_EXECUTION_FAILED') { exit 1 }
