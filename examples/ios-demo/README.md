# PaddleOCR-rs iOS Example

## Overview

This example demonstrates how to use PaddleOCR-rs in an iOS application via Swift/C interop.

## Prerequisites

1. Xcode 14 or later
2. Rust toolchain with iOS targets:
   ```bash
   rustup target add aarch64-apple-ios
   rustup target add aarch64-apple-ios-sim
   ```
3. Apple Developer account (for device testing)

## Building the Native Library

### Using the build script:

```bash
# Build for ARM64 device
./build-ios.sh aarch64-apple-ios --release

# Build for ARM64 simulator (Apple Silicon Mac)
./build-ios.sh aarch64-apple-ios-sim --release

# Build for x86_64 simulator (Intel Mac)
./build-ios.sh x86_64-apple-ios-sim --release
```

### Using the PowerShell script (Windows):

```powershell
# Note: iOS builds require macOS
.\build-ios.ps1 -Target aarch64-apple-ios -Release
```

## Project Structure

```
ios-demo/
├── PaddleOCRDemo/
│   ├── PaddleOCRDemo.xcodeproj
│   ├── PaddleOCRDemo/
│   │   ├── AppDelegate.swift
│   │   ├── ViewController.swift
│   │   ├── PaddleOcrBridge.swift
│   │   ├── Info.plist
│   │   ├── Assets.xcassets
│   │   ├── Models/
│   │   │   ├── ch_PP-OCRv4_det_infer.onnx
│   │   │   ├── ch_PP-OCRv4_rec_infer.onnx
│   │   │   └── ppocr_keys_v1.txt
│   │   └── Libs/
│   │       ├── arm64/
│   │       │   └── libpaddleocr_rs_onnx.a
│   │       └── arm64-sim/
│   │           └── libpaddleocr_rs_onnx.a
│   └── PaddleOCRDemoTests/
└── README.md
```

## Swift/C Interop Bridge

### PaddleOcrBridge.h (Bridging Header)

```c
#ifndef PaddleOcrBridge_h
#define PaddleOcrBridge_h

#include <stdint.h>

// Opaque handle types
typedef struct OcrEngineHandle OcrEngineHandle;
typedef struct OcrClassifierHandle OcrClassifierHandle;

// Result structure
typedef struct {
    char* text;
    float confidence;
    float x;
    float y;
    float width;
    float height;
} OcrResultC;

// Error structure
typedef struct {
    int code;
    char* message;
} OcrErrorC;

// OCR Engine functions
OcrEngineHandle* paddle_ocr_create(
    const uint8_t* det_model, size_t det_model_len,
    const uint8_t* rec_model, size_t rec_model_len,
    const uint8_t* keys_data, size_t keys_data_len,
    OcrErrorC* error
);

OcrEngineHandle* paddle_ocr_create_with_device(
    const uint8_t* det_model, size_t det_model_len,
    const uint8_t* rec_model, size_t rec_model_len,
    const uint8_t* keys_data, size_t keys_data_len,
    uint32_t device,
    OcrErrorC* error
);

void paddle_ocr_destroy(OcrEngineHandle* handle);

int paddle_ocr_recognize(
    const OcrEngineHandle* handle,
    const uint8_t* image_data, size_t image_len,
    uint32_t order,
    OcrResultC** results, size_t* results_len,
    OcrErrorC* error
);

void paddle_ocr_free_results(OcrResultC* results, size_t len);
void paddle_ocr_free_string(char* s);
void paddle_ocr_free_buffer(uint8_t* data, size_t len);

// Classifier functions
OcrClassifierHandle* paddle_ocr_classifier_create(
    const uint8_t* model_data, size_t model_len,
    OcrErrorC* error
);

void paddle_ocr_classifier_destroy(OcrClassifierHandle* handle);

int paddle_ocr_classifier_classify(
    const OcrClassifierHandle* handle,
    const uint8_t* image_data, size_t image_len,
    uint32_t* orientation, float* confidence,
    OcrErrorC* error
);

#endif /* PaddleOcrBridge_h */
```

### PaddleOcrBridge.swift

```swift
import Foundation

class PaddleOcrBridge {
    
    // MARK: - Types
    
    struct OcrResult {
        let text: String
        let confidence: Float
        let x: Float
        let y: Float
        let width: Float
        let height: Float
    }
    
    enum OcrError: Error, LocalizedError {
        case invalidArgument(String)
        case modelError(String)
        case inferenceError(String)
        case generalError(String)
        
        var errorDescription: String? {
            switch self {
            case .invalidArgument(let msg): return "Invalid argument: \(msg)"
            case .modelError(let msg): return "Model error: \(msg)"
            case .inferenceError(let msg): return "Inference error: \(msg)"
            case .generalError(let msg): return "General error: \(msg)"
            }
        }
    }
    
    enum AccelerationDevice: UInt32 {
        case cpu = 0
        case directML = 1
        case cuda = 2
        case openVINO = 3
        case nnapi = 4
        case coreML = 5
        case cann = 6
    }
    
    enum OrderBy: UInt32 {
        case horizontal = 0
        case vertical = 1
        case score = 2
    }
    
    // MARK: - Properties
    
    private let handle: OcrEngineHandle
    
    // MARK: - Initialization
    
    init(detModel: Data, recModel: Data, keys: Data, device: AccelerationDevice = .cpu) throws {
        var error = OcrErrorC()
        
        let result = detModel.withUnsafeBytes { detPtr in
            recModel.withUnsafeBytes { recPtr in
                keys.withUnsafeBytes { keysPtr in
                    paddle_ocr_create_with_device(
                        detPtr.baseAddress?.assumingMemoryBound(to: UInt8.self),
                        detModel.count,
                        recPtr.baseAddress?.assumingMemoryBound(to: UInt8.self),
                        recModel.count,
                        keysPtr.baseAddress?.assumingMemoryBound(to: UInt8.self),
                        keys.count,
                        device.rawValue,
                        &error
                    )
                }
            }
        }
        
        guard let handle = result else {
            let message = error.message.map { String(cString: $0) } ?? "Unknown error"
            paddle_ocr_free_error(error)
            throw OcrError.modelError(message)
        }
        
        self.handle = handle
    }
    
    deinit {
        paddle_ocr_destroy(handle)
    }
    
    // MARK: - Recognition
    
    func recognize(image: Data, order: OrderBy = .horizontal) throws -> [OcrResult] {
        var error = OcrErrorC()
        var results: UnsafeMutablePointer<OcrResultC>?
        var resultsLen = 0
        
        let status = image.withUnsafeBytes { imagePtr in
            paddle_ocr_recognize(
                handle,
                imagePtr.baseAddress?.assumingMemoryBound(to: UInt8.self),
                image.count,
                order.rawValue,
                &results,
                &resultsLen,
                &error
            )
        }
        
        guard status == 0 else {
            let message = error.message.map { String(cString: $0) } ?? "Unknown error"
            paddle_ocr_free_error(error)
            throw OcrError.inferenceError(message)
        }
        
        guard let results = results, resultsLen > 0 else {
            return []
        }
        
        var ocrResults: [OcrResult] = []
        
        for i in 0..<resultsLen {
            let result = results[i]
            let text = result.text.map { String(cString: $0) } ?? ""
            
            ocrResults.append(OcrResult(
                text: text,
                confidence: result.confidence,
                x: result.x,
                y: result.y,
                width: result.width,
                height: result.height
            ))
        }
        
        paddle_ocr_free_results(results, resultsLen)
        
        return ocrResults
    }
}

// MARK: - Helper Extensions

extension PaddleOcrBridge {
    
    /// Load OCR engine from bundle resources
    static func loadFromBundle(bundle: Bundle = .main, device: AccelerationDevice = .cpu) throws -> PaddleOcrBridge {
        guard let detModelURL = bundle.url(forResource: "ch_PP-OCRv4_det_infer", withExtension: "onnx"),
              let recModelURL = bundle.url(forResource: "ch_PP-OCRv4_rec_infer", withExtension: "onnx"),
              let keysURL = bundle.url(forResource: "ppocr_keys_v1", withExtension: "txt") else {
            throw OcrError.invalidArgument("Model files not found in bundle")
        }
        
        let detModel = try Data(contentsOf: detModelURL)
        let recModel = try Data(contentsOf: recModelURL)
        let keys = try Data(contentsOf: keysURL)
        
        return try PaddleOcrBridge(detModel: detModel, recModel: recModel, keys: keys, device: device)
    }
    
    /// Recognize text from UIImage
    @available(iOS 13.0, *)
    func recognize(uiImage: UIImage, order: OrderBy = .horizontal) throws -> [OcrResult] {
        guard let imageData = uiImage.jpegData(compressionQuality: 0.8) else {
            throw OcrError.invalidArgument("Failed to convert image to JPEG")
        }
        
        return try recognize(image: imageData, order: order)
    }
}
```

## Usage Example (Swift)

```swift
import UIKit

class ViewController: UIViewController {
    
    private var ocrEngine: PaddleOcrBridge?
    
    override func viewDidLoad() {
        super.viewDidLoad()
        
        do {
            // Load OCR engine from bundle
            ocrEngine = try PaddleOcrBridge.loadFromBundle(device: .cpu)
            print("OCR engine loaded successfully")
        } catch {
            print("Failed to load OCR engine: \(error)")
        }
    }
    
    @IBAction func recognizeImage(_ sender: UIButton) {
        guard let engine = ocrEngine else {
            print("OCR engine not initialized")
            return
        }
        
        // Load image from bundle or camera
        guard let image = UIImage(named: "test_image") else {
            print("Image not found")
            return
        }
        
        do {
            let results = try engine.recognize(uiImage: image)
            
            for result in results {
                print("Text: \(result.text)")
                print("Confidence: \(result.confidence)")
                print("Position: (\(result.x), \(result.y)) \(result.width)x\(result.height)")
                print("---")
            }
        } catch {
            print("OCR failed: \(error)")
        }
    }
}
```

## Model Files

Place the following model files in `PaddleOCRDemo/Models/`:

1. `ch_PP-OCRv4_det_infer.onnx` - Text detection model
2. `ch_PP-OCRv4_rec_infer.onnx` - Text recognition model
3. `ppocr_keys_v1.txt` - Character dictionary

Download from: https://paddleocr.bj.bcebos.com/PP-OCRv4/chinese/ch_PP-OCRv4_infer.tar

## Xcode Configuration

### Build Settings

1. Add the library search path:
   ```
   LIBRARY_SEARCH_PATHS = $(PROJECT_DIR)/Libs/$(ARCHS)
   ```

2. Add the bridging header:
   ```
   SWIFT_OBJC_BRIDGING_HEADER = PaddleOCRDemo/PaddleOcrBridge.h
   ```

3. Link the library:
   ```
   OTHER_LDFLAGS = -lpaddleocr_rs_onnx
   ```

### Fat Binary (Optional)

To create a universal binary for both device and simulator:

```bash
lipo -create \
    Libs/arm64/libpaddleocr_rs_onnx.a \
    Libs/arm64-sim/libpaddleocr_rs_onnx.a \
    -output Libs/universal/libpaddleocr_rs_onnx.a
```

## CoreML Acceleration

To use CoreML acceleration:

```swift
let engine = try PaddleOcrBridge(
    detModel: detModelData,
    recModel: recModelData,
    keys: keysData,
    device: .coreML
)
```

Note: CoreML acceleration requires iOS 14 or later and is automatically available on devices with Neural Engine.

## Performance Tips

1. Use ARM64 (aarch64-apple-ios) for best performance on devices
2. CoreML can provide significant speedup on devices with Neural Engine
3. Consider using quantized models for mobile deployment
4. Cache the engine instance - creating it is expensive
5. Use background threads for OCR operations
6. For real-time OCR, consider processing frames at lower resolution

## Troubleshooting

### "Library not found" error
- Ensure the `.a` file is in the correct directory
- Check that the architecture matches your target (device vs simulator)
- Verify `LIBRARY_SEARCH_PATHS` is set correctly

### "Model load failed" error
- Verify model files are added to the Xcode project
- Check that files are included in the app bundle
- Verify file permissions

### CoreML fallback to CPU
- Check iOS version (CoreML requires iOS 14+)
- Verify device has Neural Engine (A12 or later)
- Check console for CoreML-related warnings

### Simulator vs Device
- Use `aarch64-apple-ios-sim` for Apple Silicon Mac simulators
- Use `x86_64-apple-ios-sim` for Intel Mac simulators
- Use `aarch64-apple-ios` for physical devices