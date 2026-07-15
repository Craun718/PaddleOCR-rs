#!/bin/bash
# Build script for PaddleOCR-rs Android targets
# Usage: ./build-android.sh [target] [--release]
#
# Note: ort-sys only provides prebuilt binaries for aarch64-linux-android.
# Other Android targets (armv7, x86_64) are NOT supported by ort-sys.

set -e

# Default values
TARGET="aarch64-linux-android"
RELEASE=""
FEATURES="ffi"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        aarch64-linux-android)
            TARGET="$1"
            shift
            ;;
        --release)
            RELEASE="--release"
            shift
            ;;
        --features)
            FEATURES="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [target] [--release] [--features features]"
            echo ""
            echo "Supported targets:"
            echo "  aarch64-linux-android  ARM64 (default)"
            echo ""
            echo "Note: armv7-linux-androideabi and x86_64-linux-android"
            echo "      are NOT supported by ort-sys prebuilt binaries."
            echo ""
            echo "Options:"
            echo "  --release    Build in release mode"
            echo "  --features   Comma-separated list of features to enable"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Only aarch64-linux-android is supported by ort-sys."
            exit 1
            ;;
    esac
done

# Check for Android NDK
if [ -z "$ANDROID_NDK_HOME" ] && [ -z "$NDK_HOME" ]; then
    echo "Error: ANDROID_NDK_HOME or NDK_HOME environment variable not set"
    echo "Please install Android NDK and set the environment variable"
    exit 1
fi

NDK_DIR="${ANDROID_NDK_HOME:-$NDK_HOME}"

echo "Building for Android target: $TARGET"
echo "NDK directory: $NDK_DIR"
echo "Features: $FEATURES"
echo "Mode: ${RELEASE:+release}${RELEASE:-debug}"

# Build
cargo build $RELEASE --target "$TARGET" --features "$FEATURES"

# Find output file
if [ -n "$RELEASE" ]; then
    OUTPUT_DIR="target/$TARGET/release"
else
    OUTPUT_DIR="target/$TARGET/debug"
fi

OUTPUT_FILE="libpaddleocr_rs_onnx.so"
OUTPUT_PATH="$OUTPUT_DIR/$OUTPUT_FILE"

if [ -f "$OUTPUT_PATH" ]; then
    echo ""
    echo "Build successful!"
    echo "Output: $OUTPUT_PATH"
    echo "Size: $(du -h "$OUTPUT_PATH" | cut -f1)"
else
    echo ""
    echo "Build completed but output file not found at: $OUTPUT_PATH"
    echo "Available files in $OUTPUT_DIR:"
    ls -la "$OUTPUT_DIR/" 2>/dev/null || echo "  (directory not found)"
fi