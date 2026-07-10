param(
    [Parameter(Mandatory = $true)]
    [string]$Image,
    [switch]$SkipLayout,
    [switch]$Full,
    [string]$WorkRoot = ".\workflow",
    [string]$DoclayoutDevice = "cpu"
)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$python = Join-Path $root ".conda\messtoclean\python.exe"
Set-Location $root

$argsList = @(
  "-m", "agent.workflow",
  "--image", $Image,
  "--work-root", $WorkRoot,
  "--doclayout-device", $DoclayoutDevice
)

if ($SkipLayout) {
    $argsList += "--skip-layout"
}

if ($Full) {
    $argsList += @("--with-patcher", "--with-fig", "--use-crop-qno")
}

& $python @argsList
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
