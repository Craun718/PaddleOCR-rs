#!/bin/bash
# Build script for PaddleOCR-rs iOS targets
# Usage: ./build-ios.sh [target] [--release]
#
# Targets:
#   aarch64-apple-ios      (ARM64 device, default)
#   aarch64-apple-ios-sim  (ARM64 simulator)
#   x86_64-apple-ios-sim   (x86_64 simulator, Intel Mac)

set -e

# Default values
TARGET="aarch64-apple-ios"
RELEASE=""
FEATURES="ffi"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        aarch64-apple-ios|aarch64-apple-ios-sim|x86_64-apple-ios-sim)
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
            echo "Targets:"
            echo "  aarch64-apple-ios      ARM64 device (default)"
            echo "  aarch64-apple-ios-sim  ARM64 simulator (Apple Silicon Mac)"
            echo "  x86_64-apple-ios-sim   x86_64 simulator (Intel Mac)"
            echo ""
            echo "Options:"
            echo "  --release    Build in release mode"
            echo "  --features   Comma-separated list of features to enable"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Check for Xcode
if ! command -v xcrun &> /dev/null; then
    echo "Error: xcrun not found. Please install Xcode command line tools."
    echo "Run: xcode-select --install"
    exit 1
fi

echo "Building for iOS target: $TARGET"
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

OUTPUT_FILE="libpaddleocr_rs_onnx.a"
OUTPUT_PATH="$OUTPUT_DIR/$OUTPUT_FILE"

if [ -f "$OUTPUT_PATH" ]; then
    echo ""
    echo "Build successful!"
    echo "Output: $OUTPUT_PATH"
    echo "Size: $(du -h "$OUTPUT_PATH" | cut -f1)"
    
    # Show architectures if lipo is available
    if command -v lipo &> /dev/null; then
        echo "Architectures:"
        lipo -info "$OUTPUT_PATH" 2>/dev/null || echo "  (single architecture)"
    fi
else
    echo ""
    echo "Build completed but output file not found at: $OUTPUT_PATH"
    echo "Available files in $OUTPUT_DIR:"
    ls -la "$OUTPUT_DIR/" 2>/dev/null || echo "  (directory not found)"
fi