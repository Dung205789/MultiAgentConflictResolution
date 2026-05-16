param(
    [string]$KernelDir = "kaggle\\kernels\\projectmem_full",
    [string]$NotebookPath = "kaggle\\kaggle_runner_main_full.ipynb",
    [string]$KernelRef = "",
    [string]$DownloadDir = "kaggle_outputs/projectmem-main-full-qwen2p5-1p5b",
    [int]$PollSeconds = 60,
    [int]$MaxPolls = 60
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$kernelPath = Join-Path $repoRoot $KernelDir
$sourceNotebook = Join-Path $repoRoot $NotebookPath
$destNotebook = Join-Path $kernelPath (Split-Path $NotebookPath -Leaf)
$downloadPath = Join-Path $repoRoot $DownloadDir
$kaggleConfigDir = Join-Path $repoRoot "kaggle"
$metadataPath = Join-Path $kernelPath "kernel-metadata.json"

if (-not (Test-Path $kaggleConfigDir)) {
    throw "Kaggle config folder not found: $kaggleConfigDir"
}

if (-not (Test-Path (Join-Path $kaggleConfigDir "kaggle.json"))) {
    throw "Kaggle credentials not found: $(Join-Path $kaggleConfigDir 'kaggle.json')"
}

$env:KAGGLE_CONFIG_DIR = $kaggleConfigDir
$env:PYTHONWARNINGS = "ignore"

if (-not (Test-Path $sourceNotebook)) {
    throw "Notebook not found: $sourceNotebook"
}

if (-not (Test-Path $kernelPath)) {
    throw "Kernel folder not found: $kernelPath"
}

if (-not (Test-Path $metadataPath)) {
    throw "Kernel metadata not found: $metadataPath"
}

$metadata = Get-Content -LiteralPath $metadataPath -Raw | ConvertFrom-Json
if ([string]::IsNullOrWhiteSpace($KernelRef)) {
    $KernelRef = $metadata.id
}

if ([string]::IsNullOrWhiteSpace($KernelRef)) {
    throw "KernelRef is empty and kernel-metadata.json does not define an id."
}

Copy-Item -LiteralPath $sourceNotebook -Destination $destNotebook -Force
Write-Host "Notebook synced to $destNotebook"
Write-Host "Using KAGGLE_CONFIG_DIR=$env:KAGGLE_CONFIG_DIR"
Write-Host "Using KernelRef=$KernelRef"

$currentStatus = kaggle kernels status $KernelRef 2>&1
$currentStatusText = ($currentStatus | Out-String).Trim()
if ($currentStatusText -match "RUNNING") {
    Write-Host "Kernel is currently RUNNING. Pushing a new version will replace the active notebook revision."
}

kaggle kernels push -p $kernelPath

for ($i = 0; $i -lt $MaxPolls; $i++) {
    Start-Sleep -Seconds $PollSeconds
    $statusOutput = kaggle kernels status $KernelRef 2>&1
    $statusText = ($statusOutput | Out-String).Trim()
    Write-Host $statusText

    if ($statusText -match "complete" -or $statusText -match "succeeded") {
        break
    }

    if ($statusText -match "error" -or $statusText -match "failed") {
        throw "Kaggle kernel run failed: $statusText"
    }
}

New-Item -ItemType Directory -Force -Path $downloadPath | Out-Null
kaggle kernels output $KernelRef -p $downloadPath -o
Write-Host "Outputs downloaded to $downloadPath"
