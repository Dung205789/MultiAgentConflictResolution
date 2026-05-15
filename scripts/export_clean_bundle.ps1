param(
    [string]$OutputDir = "dist"
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$outputRoot = Join-Path $repoRoot $OutputDir
$stageDir = Join-Path $outputRoot "projectmem_colab_bundle"
$timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
$zipPath = Join-Path $outputRoot "ProjectMem-colab-bundle-$timestamp.zip"

$rootFiles = @(
    ".dockerignore",
    ".env.example",
    ".gitignore",
    "Dockerfile",
    "README.md",
    "colab_runner.ipynb",
    "docker-compose.yml",
    "docker-entrypoint.sh",
    "main.py",
    "requirements.txt"
)

$rootDirs = @(
    "configs",
    "reports",
    "scripts",
    "src"
)

New-Item -ItemType Directory -Force -Path $outputRoot | Out-Null

if (Test-Path $stageDir) {
    Remove-Item -LiteralPath $stageDir -Recurse -Force
}

New-Item -ItemType Directory -Force -Path $stageDir | Out-Null

foreach ($relativePath in $rootFiles) {
    $sourcePath = Join-Path $repoRoot $relativePath
    if (Test-Path $sourcePath) {
        Copy-Item -LiteralPath $sourcePath -Destination $stageDir -Force
    }
}

foreach ($relativePath in $rootDirs) {
    $sourcePath = Join-Path $repoRoot $relativePath
    if (-not (Test-Path $sourcePath)) {
        continue
    }

    $destinationPath = Join-Path $stageDir $relativePath
    New-Item -ItemType Directory -Force -Path $destinationPath | Out-Null
    Copy-Item -LiteralPath (Join-Path $sourcePath "*") -Destination $destinationPath -Recurse -Force
}

Get-ChildItem -LiteralPath $stageDir -Recurse -Directory |
    Where-Object { $_.Name -in @("__pycache__", ".ipynb_checkpoints") } |
    Remove-Item -Recurse -Force

Get-ChildItem -LiteralPath $stageDir -Recurse -File |
    Where-Object { $_.Extension -in @(".pyc", ".pyo") } |
    Remove-Item -Force

$stageReports = Join-Path $stageDir "reports"
if (Test-Path $stageReports) {
    Get-ChildItem -LiteralPath $stageReports -Force |
        Where-Object { $_.Name -ne ".gitkeep" } |
        Remove-Item -Recurse -Force
}

if (Test-Path $zipPath) {
    Remove-Item -LiteralPath $zipPath -Force
}

Compress-Archive -Path (Join-Path $stageDir "*") -DestinationPath $zipPath
Remove-Item -LiteralPath $stageDir -Recurse -Force
Write-Host "Created clean bundle: $zipPath"
