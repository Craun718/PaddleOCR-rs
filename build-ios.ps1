# Build script for PaddleOCR-rs iOS targets (Windows with cross-compilation)
# Usage: .\build-ios.ps1 [-Target <target>] [-Release] [-Features <features>]
#
# Targets:
#   aarch64-apple-ios      (ARM64 device, default)
#   aarch64-apple-ios-sim  (ARM64 simulator)
#   x86_64-apple-ios-sim   (x86_64 simulator, Intel Mac)
#
# Note: iOS builds typically require macOS with Xcode.
# This script is for cross-compilation setup verification.

param(
    [ValidateSet("aarch64-apple-ios", "aarch64-apple-ios-sim", "x86_64-apple-ios-sim")]
    [string]$Target = "aarch64-apple-ios",
    
    [switch]$Release,
    
    [string]$Features = "ffi"
)

$ErrorActionPreference = "Stop"

Write-Host "Building for iOS target: $Target"
Write-Host "Features: $Features"
Write-Host "Mode: $(if ($Release) { 'release' } else { 'debug' })"
Write-Host ""
Write-Host "Note: iOS builds require macOS with Xcode installed."
Write-Host "This script will attempt to build, but may fail on Windows."

# Build
$cargoArgs = @("build")
if ($Release) { $cargoArgs += "--release" }
$cargoArgs += "--target", $Target
$cargoArgs += "--features", $Features

& cargo @cargoArgs

# Find output file
$OutputDir = if ($Release) { "target\$Target\release" } else { "target\$Target\debug" }
$OutputFile = "libpaddleocr_rs_onnx.a"
$OutputPath = "$OutputDir\$OutputFile"

if (Test-Path $OutputPath) {
    Write-Host ""
    Write-Host "Build successful!"
    Write-Host "Output: $OutputPath"
    Write-Host "Size: $((Get-Item $OutputPath).Length / 1KB) KB"
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