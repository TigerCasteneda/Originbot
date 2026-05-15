param(
    [string]$HostName = "140.207.205.81",
    [int]$Port = 32222,
    [string]$User = "root+vm-B46pSyUOwTzQlMW3",
    [string]$RemoteDir = "/data/originbot",
    [switch]$SkipImageDataset,
    [switch]$SkipDataset0424,
    [switch]$SkipWeights,
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

$items = @(
    "README.md",
    "originbot_train.py",
    "originbot_onnx.py",
    "prepare_horizon_mapper.py",
    "horizon_preprocess.py",
    "run_horizon_convert.sh",
    "setup_jupyter_autostart.sh",
    "02_preprocess.sh",
    "03_build.sh",
    "resnet18_224x224_nv12.yaml",
    "01 train.ipynb",
    "02 model convert.ipynb"
)

if (-not $SkipImageDataset) {
    $items += "image_dataset"
}

if (-not $SkipDataset0424) {
    $items += "image_dataset_0424"
}

if ((-not $SkipWeights) -and (Test-Path -LiteralPath "best_line_follower_model_xy.pth")) {
    $items += "best_line_follower_model_xy.pth"
}

$missing = @()
foreach ($item in $items) {
    if (-not (Test-Path -LiteralPath $item)) {
        $missing += $item
    }
}
if ($missing.Count -gt 0) {
    throw ("Missing required paths:`n - " + ($missing -join "`n - "))
}

$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$bundleName = "originbot_upload_$timestamp.tgz"
$bundlePath = Join-Path $repoRoot $bundleName
$remote = "$User@$HostName"

Write-Host "[1/4] Creating bundle: $bundlePath"
& tar -czf $bundlePath @items
if ($LASTEXITCODE -ne 0) {
    throw "Failed to create bundle."
}

Write-Host "[2/4] Ensuring remote dir: $RemoteDir"
& ssh -p $Port $remote "mkdir -p '$RemoteDir'"
if ($LASTEXITCODE -ne 0) {
    throw "Failed to create remote directory: $RemoteDir"
}

Write-Host "[3/4] Uploading bundle to remote /tmp"
& scp -P $Port $bundlePath "${remote}:/tmp/$bundleName"
if ($LASTEXITCODE -ne 0) {
    throw "Failed to upload bundle."
}

Write-Host "[4/4] Extracting bundle on remote and setting execute bits"
$extractCmd = @"
mkdir -p '$RemoteDir' && \
tar -xzf /tmp/$bundleName -C '$RemoteDir' && \
chmod +x '$RemoteDir/run_horizon_convert.sh' '$RemoteDir/setup_jupyter_autostart.sh' '$RemoteDir/02_preprocess.sh' '$RemoteDir/03_build.sh' && \
rm -f /tmp/$bundleName
"@
& ssh -p $Port $remote $extractCmd
if ($LASTEXITCODE -ne 0) {
    throw "Failed to extract bundle on remote host."
}

if (-not $KeepBundle) {
    Remove-Item -LiteralPath $bundlePath -Force
}

Write-Host ""
Write-Host "Upload complete."
Write-Host "Remote path: $RemoteDir"
Write-Host "Host: ${remote}:$Port"
Write-Host ""
Write-Host "Next (remote):"
Write-Host "  cd $RemoteDir"
Write-Host "  ls -lah"
Write-Host "  python originbot_train.py --dataset-dir ./image_dataset --dataset-dir ./image_dataset_0424 --epochs 3 --batch-size 128 --num-workers 4 --amp"
