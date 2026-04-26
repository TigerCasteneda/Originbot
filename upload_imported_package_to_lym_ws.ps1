param(
    [string]$HostName = "hz-4.matpool.com",
    [int]$Port = 29611,
    [string]$User = "root",
    [string]$RemoteDir = "/root/lym_ws/imported",
    [switch]$KeepBundle
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

function Require-Command {
    param([string]$Name)
    if (-not (Get-Command $Name -ErrorAction SilentlyContinue)) {
        throw "Missing required command: $Name"
    }
}

Require-Command "ssh"
Require-Command "scp"
Require-Command "tar"

$repoRoot = $PSScriptRoot
if (-not $repoRoot) {
    $repoRoot = (Get-Location).Path
}
Set-Location $repoRoot

$sourceDir = "imported_package_20260425"
if (-not (Test-Path -LiteralPath $sourceDir)) {
    throw "Missing source directory: $sourceDir"
}

$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$bundleName = "imported_package_20260425_$timestamp.tgz"
$bundlePath = Join-Path $repoRoot $bundleName
$remote = "$User@$HostName"

Write-Host "[1/4] Creating bundle: $bundlePath"
& tar -czf $bundlePath $sourceDir

Write-Host "[2/4] Ensuring remote dir: $RemoteDir"
& ssh -p $Port $remote "mkdir -p '$RemoteDir'"

Write-Host "[3/4] Uploading bundle to remote /tmp"
& scp -P $Port $bundlePath "${remote}:/tmp/$bundleName"

Write-Host "[4/4] Extracting bundle on remote"
$extractCmd = @"
mkdir -p '$RemoteDir' && \
tar -xzf /tmp/$bundleName -C '$RemoteDir' && \
rm -f /tmp/$bundleName
"@
& ssh -p $Port $remote $extractCmd

if (-not $KeepBundle) {
    Remove-Item -LiteralPath $bundlePath -Force
}

Write-Host ""
Write-Host "Upload complete."
Write-Host "Remote path: $RemoteDir/imported_package_20260425"
Write-Host "Host: ${remote}:$Port"
Write-Host ""
Write-Host "Next (remote):"
Write-Host "  cd $RemoteDir/imported_package_20260425"
Write-Host "  ls -lah"
