# Build script for PaddleOCR-rs Android targets (Windows)
# Usage: .\build-android.ps1 [-Target <target>] [-Release] [-Features <features>]
#
# Note: ort-sys only provides prebuilt binaries for aarch64-linux-android.
# Other Android targets (armv7, x86_64) are NOT supported by ort-sys.

param(
    [ValidateSet("aarch64-linux-android")]
    [string]$Target = "aarch64-linux-android",
    
    [switch]$Release,
    
    [string]$Features = "ffi"
)

$ErrorActionPreference = "Stop"

# Check for Android NDK
if (-not $env:ANDROID_NDK_HOME -and -not $env:NDK_HOME) {
    Write-Error "ANDROID_NDK_HOME or NDK_HOME environment variable not set"
    Write-Host "Please install Android NDK and set the environment variable"
    exit 1
}

$NDKDir = if ($env:ANDROID_NDK_HOME) { $env:ANDROID_NDK_HOME } else { $env:NDK_HOME }

Write-Host "Building for Android target: $Target"
Write-Host "NDK directory: $NDKDir"
Write-Host "Features: $Features"
Write-Host "Mode: $(if ($Release) { 'release' } else { 'debug' })"
Write-Host ""
Write-Host "Note: Only aarch64-linux-android is supported by ort-sys."
Write-Host "      armv7-linux-androideabi and x86_64-linux-android are NOT supported."

# Build
$cargoArgs = @("build")
if ($Release) { $cargoArgs += "--release" }
$cargoArgs += "--target", $Target
$cargoArgs += "--features", $Features

& cargo @cargoArgs

# Find output file
$OutputDir = if ($Release) { "target\$Target\release" } else { "target\$Target\debug" }
$OutputFile = "paddleocr_rs_onnx.dll"
$OutputPath = "$OutputDir\$OutputFile"

if (Test-Path $OutputPath) {
    Write-Host ""
    Write-Host "Build successful!"
    Write-Host "Output: $OutputPath"
    Write-Host "Size: $((Get-Item $OutputPath).Length / 1MB) MB"
} else {
    Write-Host ""
    Write-Host "Build completed but output file not found at: $OutputPath"
    Write-Host "Available files in $OutputDir:"
    if (Test-Path $OutputDir) {
        Get-ChildItem $OutputDir | Format-Table Name, Length
    } else {
        Write-Host "  (directory not found)"
    }
}