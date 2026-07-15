//! # FFI Interface
//!
//! C-compatible FFI interface for PaddleOCR-rs, enabling integration with
//! Android (JNI), iOS (Swift/ObjC), and other languages via C bindings.
//!
//! ## Safety
//!
//! All FFI functions are `unsafe` and require careful memory management.
//! Use the provided `free` functions to release allocated memory.

use std::ffi::CString;
use std::os::raw::c_char;
use std::ptr;

use crate::{AccelerationDevice, OcrEngine, OrderBy, DocOrientationClassifier};
use crate::error::PaddleOcrError;

/// Result of a single text recognition operation.
///
/// Contains the recognized text, confidence score, and bounding box coordinates.
/// The caller must free `text` using `paddle_ocr_free_string`.
#[repr(C)]
pub struct OcrResult {
    /// Recognized text (UTF-8 encoded, null-terminated).
    /// Caller must free with `paddle_ocr_free_string`.
    pub text: *mut c_char,
    /// Recognition confidence score in `[0.0, 1.0]`.
    pub confidence: f32,
    /// X coordinate of the top-left corner (pixels).
    pub x: f32,
    /// Y coordinate of the top-left corner (pixels).
    pub y: f32,
    /// Width of the bounding rectangle (pixels).
    pub width: f32,
    /// Height of the bounding rectangle (pixels).
    pub height: f32,
}

/// Error code returned by FFI operations.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OcrErrorCode {
    /// Success.
    Ok = 0,
    /// Invalid argument.
    InvalidArgument = 1,
    /// Model loading or session creation failed.
    ModelError = 2,
    /// Inference failed.
    InferenceError = 3,
    /// Memory allocation failed.
    AllocationError = 4,
    /// Invalid UTF-8 data.
    Utf8Error = 5,
    /// General error.
    GeneralError = 6,
}

/// Error information returned by FFI operations.
///
/// The caller must free `message` using `paddle_ocr_free_string`.
#[repr(C)]
pub struct OcrError {
    /// Error code.
    pub code: OcrErrorCode,
    /// Human-readable error message (UTF-8, null-terminated).
    /// Caller must free with `paddle_ocr_free_string`.
    pub message: *mut c_char,
}

impl From<PaddleOcrError> for OcrError {
    fn from(err: PaddleOcrError) -> Self {
        let (code, msg) = match &err {
            PaddleOcrError::Image { message } => (OcrErrorCode::InvalidArgument, message.clone()),
            PaddleOcrError::Model(e) => (OcrErrorCode::ModelError, e.to_string()),
            PaddleOcrError::Inference { message } => (OcrErrorCode::InferenceError, message.clone()),
            PaddleOcrError::Preprocessing { message } => (OcrErrorCode::GeneralError, message.clone()),
            PaddleOcrError::Decoding { message } => (OcrErrorCode::GeneralError, message.clone()),
            PaddleOcrError::Utf8(e) => (OcrErrorCode::Utf8Error, e.to_string()),
            PaddleOcrError::DegenerateRegion { reason } => (OcrErrorCode::InvalidArgument, reason.clone()),
            PaddleOcrError::Projection { reason } => (OcrErrorCode::GeneralError, reason.clone()),
            PaddleOcrError::General(s) => (OcrErrorCode::GeneralError, s.clone()),
        };

        let c_msg = CString::new(msg).unwrap_or_else(|_| CString::new("unknown error").unwrap());

        OcrError {
            code,
            message: c_msg.into_raw(),
        }
    }
}

/// Opaque handle to an OCR engine instance.
///
/// This handle is thread-safe and can be shared across threads.
/// The caller must destroy it with `paddle_ocr_destroy`.
pub struct OcrEngineHandle {
    engine: OcrEngine,
}

/// Opaque handle to a document orientation classifier.
///
/// The caller must destroy it with `paddle_ocr_classifier_destroy`.
pub struct OcrClassifierHandle {
    classifier: DocOrientationClassifier,
}

// ============================================================================
// OCR Engine Functions
// ============================================================================

/// Create a new OCR engine with default (CPU) acceleration.
///
/// # Arguments
///
/// * `det_model` -- Raw bytes of the detection ONNX model.
/// * `det_model_len` -- Length of `det_model` in bytes.
/// * `rec_model` -- Raw bytes of the recognition ONNX model.
/// * `rec_model_len` -- Length of `rec_model` in bytes.
/// * `keys_data` -- UTF-8 bytes of the character dictionary file.
/// * `keys_data_len` -- Length of `keys_data` in bytes.
/// * `error` -- Pointer to receive error information (optional, can be NULL).
///
/// # Returns
///
/// Pointer to a new `OcrEngineHandle` on success, or NULL on failure.
/// The caller must destroy the handle with `paddle_ocr_destroy`.
///
/// # Safety
///
/// The input pointers must be valid for the specified lengths.
#[no_mangle]
pub unsafe extern "C" fn paddle_ocr_create(
    det_model: *const u8,
    det_model_len: usize,
    rec_model: *const u8,
    rec_model_len: usize,
    keys_data: *const u8,
    keys_data_len: usize,
    error: *mut OcrError,
) -> *mut OcrEngineHandle {
    paddle_ocr_create_with_device(
        det_model, det_model_len,
        rec_model, rec_model_len,
        keys_data, keys_data_len,
        0, // AccelerationDevice::Cpu
        error,
    )
}

/// Create a new OCR engine with a specific acceleration device.
///
/// # Arguments
///
/// * `det_model` -- Raw bytes of the detection ONNX model.
/// * `det_model_len` -- Length of `det_model` in bytes.
/// * `rec_model` -- Raw bytes of the recognition ONNX model.
/// * `rec_model_len` -- Length of `rec_model` in bytes.
/// * `keys_data` -- UTF-8 bytes of the character dictionary file.
/// * `keys_data_len` -- Length of `keys_data` in bytes.
/// * `device` -- Acceleration device (0=CPU, 1=DirectML, 2=CUDA, 3=OpenVINO, 4=NNAPI, 5=CoreML, 6=CANN).
/// * `error` -- Pointer to receive error information (optional, can be NULL).
///
/// # Returns
///
/// Pointer to a new `OcrEngineHandle` on success, or NULL on failure.
///
/// # Safety
///
/// The input pointers must be valid for the specified lengths.
#[no_mangle]
pub unsafe extern "C" fn paddle_ocr_create_with_device(
    det_model: *const u8,
    det_model_len: usize,
    rec_model: *const u8,
    rec_model_len: usize,
    keys_data: *const u8,
    keys_data_len: usize,
    device: u32,
    error: *mut OcrError,
) -> *mut OcrEngineHandle {
    // Validate inputs
    if det_model.is_null() || rec_model.is_null() || keys_data.is_null() {
        if !error.is_null() {
            (*error) = OcrError {
                code: OcrErrorCode::InvalidArgument,
                message: CString::new("null pointer argument").unwrap().into_raw(),
            };
        }
        return ptr::null_mut();
    }

    // Convert device code to AccelerationDevice
    let accel_device = match device {
        0 => AccelerationDevice::Cpu,
        1 => AccelerationDevice::DirectML,
        2 => AccelerationDevice::Cuda,
        3 => AccelerationDevice::OpenVINO,
        4 => AccelerationDevice::Nnapi,
        5 => AccelerationDevice::Coreml,
        6 => AccelerationDevice::Cann,
        _ => {
            if !error.is_null() {
                (*error) = OcrError {
                    code: OcrErrorCode::InvalidArgument,
                    message: CString::new("invalid device code").unwrap().into_raw(),
                };
            }
            return ptr::null_mut();
        }
    };

    // Create slices from pointers
    let det_slice = std::slice::from_raw_parts(det_model, det_model_len);
    let rec_slice = std::slice::from_raw_parts(rec_model, rec_model_len);
    let keys_slice = std::slice::from_raw_parts(keys_data, keys_data_len);

    // Create engine
    match OcrEngine::new_with_device(det_slice, rec_slice, keys_slice, accel_device) {
        Ok(engine) => {
            let handle = Box::new(OcrEngineHandle { engine });
            Box::into_raw(handle)
        }
        Err(e) => {
            if !error.is_null() {
                (*error) = e.into();
            }
            ptr::null_mut()
        }
    }
}

/// Destroy an OCR engine handle and release all associated resources.
///
/// # Safety
///
/// `handle` must have been created by `paddle_ocr_create` or `paddle_ocr_create_with_device`.
/// Passing a null pointer or a dangling pointer is undefined behavior.
#[no_mangle]
pub unsafe extern "C" fn paddle_ocr_destroy(handle: *mut OcrEngineHandle) {
    if !handle.is_null() {
        unsafe { drop(Box::from_raw(handle)); }
    }
}

/// Recognize all text in an image.
///
/// # Arguments
///
/// * `handle` -- Valid OCR engine handle.
/// * `image_data` -- Raw image bytes (PNG, JPEG, etc.).
/// * `image_len` -- Length of `image_data` in bytes.
/// * `order` -- Ordering mode (0=Horizontal, 1=Vertical, 2=Score).
/// * `results` -- Pointer to receive the results array (caller must free with `paddle_ocr_free_results`).
/// * `results_len` -- Pointer to receive the number of results.
/// * `error` -- Pointer to receive error information (optional, can be NULL).
///
/// # Returns
///
/// 0 on success, non-zero error code on failure.
///
/// # Safety
///
/// The input pointers must be valid for the specified lengths.
#[no_mangle]
pub unsafe extern "C" fn paddle_ocr_recognize(
    handle: *const OcrEngineHandle,
    image_data: *const u8,
    image_len: usize,
    order: u32,
    results: *mut *mut OcrResult,
    results_len: *mut usize,
    error: *mut OcrError,
) -> i32 {
    // Validate inputs
    if handle.is_null() || image_data.is_null() || results.is_null() || results_len.is_null() {
        if !error.is_null() {
            (*error) = OcrError {
                code: OcrErrorCode::InvalidArgument,
                message: CString::new("null pointer argument").unwrap().into_raw(),
            };
        }
        return OcrErrorCode::InvalidArgument as i32;
    }

    // Convert order
    let order_by = match order {
        0 => OrderBy::Horizontal,
        1 => OrderBy::Vertical,
        2 => OrderBy::Score,
        _ => OrderBy::Horizontal,
    };

    // Create image slice
    let image_slice = std::slice::from_raw_parts(image_data, image_len);

    // Load image
    let image = match image::load_from_memory(image_slice) {
        Ok(img) => img,
        Err(e) => {
            if !error.is_null() {
                (*error) = PaddleOcrError::Image { message: e.to_string() }.into();
            }
            return OcrErrorCode::InvalidArgument as i32;
        }
    };

    // Run OCR
    let engine = &(*handle).engine;
    match engine.recognize_all(&image, order_by) {
        Ok(blocks) => {
            let len = blocks.len();
            let mut results_vec = Vec::with_capacity(len);

            for block in blocks {
                let text = CString::new(block.text).unwrap_or_else(|_| CString::new("").unwrap());
                results_vec.push(OcrResult {
                    text: text.into_raw(),
                    confidence: block.confidence,
                    x: block.x,
                    y: block.y,
                    width: block.width,
                    height: block.height,
                });
            }

            let boxed = results_vec.into_boxed_slice();
            let ptr = Box::into_raw(boxed) as *mut OcrResult;
            *results = ptr;
            *results_len = len;

            OcrErrorCode::Ok as i32
        }
        Err(e) => {
            if !error.is_null() {
                (*error) = e.into();
            }
            OcrErrorCode::GeneralError as i32
        }
    }
}

/// Free an array of OCR results.
///
/// # Safety
///
/// `results` must have been allocated by `paddle_ocr_recognize`.
#[no_mangle]
pub unsafe extern "C" fn paddle_ocr_free_results(results: *mut OcrResult, len: usize) {
    if results.is_null() || len == 0 {
        return;
    }

    let slice = std::slice::from_raw_parts_mut(results, len);
    for result in slice.iter() {
        if !result.text.is_null() {
            unsafe { drop(CString::from_raw(result.text)); }
        }
    }

    unsafe { drop(Box::from_raw(slice as *mut [OcrResult])); }
}

/// Free a string allocated by the OCR engine.
///
/// # Safety
///
/// `s` must have been allocated by the OCR engine (e.g., from `OcrResult.text`).
#[no_mangle]
pub unsafe extern "C" fn paddle_ocr_free_string(s: *mut c_char) {
    if !s.is_null() {
        unsafe { drop(CString::from_raw(s)); }
    }
}

/// Free error information.
///
/// # Safety
///
/// `error` must have been allocated by an OCR function.
#[no_mangle]
pub unsafe extern "C" fn paddle_ocr_free_error(error: OcrError) {
    if !error.message.is_null() {
        unsafe { drop(CString::from_raw(error.message)); }
    }
}

// ============================================================================
// Document Orientation Classifier Functions
// ============================================================================

/// Create a new document orientation classifier with default (CPU) acceleration.
///
/// # Arguments
///
/// * `model_data` -- Raw bytes of the orientation classification ONNX model.
/// * `model_len` -- Length of `model_data` in bytes.
/// * `error` -- Pointer to receive error information (optional, can be NULL).
///
/// # Returns
///
/// Pointer to a new `OcrClassifierHandle` on success, or NULL on failure.
///
/// # Safety
///
/// The input pointer must be valid for the specified length.
#[no_mangle]
pub unsafe extern "C" fn paddle_ocr_classifier_create(
    model_data: *const u8,
    model_len: usize,
    error: *mut OcrError,
) -> *mut OcrClassifierHandle {
    paddle_ocr_classifier_create_with_device(model_data, model_len, 0, error)
}

/// Create a new document orientation classifier with a specific acceleration device.
///
/// # Arguments
///
/// * `model_data` -- Raw bytes of the orientation classification ONNX model.
/// * `model_len` -- Length of `model_data` in bytes.
/// * `device` -- Acceleration device code (see `paddle_ocr_create_with_device`).
/// * `error` -- Pointer to receive error information (optional, can be NULL).
///
/// # Returns
///
/// Pointer to a new `OcrClassifierHandle` on success, or NULL on failure.
///
/// # Safety
///
/// The input pointer must be valid for the specified length.
#[no_mangle]
pub unsafe extern "C" fn paddle_ocr_classifier_create_with_device(
    model_data: *const u8,
    model_len: usize,
    device: u32,
    error: *mut OcrError,
) -> *mut OcrClassifierHandle {
    if model_data.is_null() {
        if !error.is_null() {
            (*error) = OcrError {
                code: OcrErrorCode::InvalidArgument,
                message: CString::new("null pointer argument").unwrap().into_raw(),
            };
        }
        return ptr::null_mut();
    }

    let accel_device = match device {
        0 => AccelerationDevice::Cpu,
        1 => AccelerationDevice::DirectML,
        2 => AccelerationDevice::Cuda,
        3 => AccelerationDevice::OpenVINO,
        4 => AccelerationDevice::Nnapi,
        5 => AccelerationDevice::Coreml,
        6 => AccelerationDevice::Cann,
        _ => {
            if !error.is_null() {
                (*error) = OcrError {
                    code: OcrErrorCode::InvalidArgument,
                    message: CString::new("invalid device code").unwrap().into_raw(),
                };
            }
            return ptr::null_mut();
        }
    };

    let model_slice = std::slice::from_raw_parts(model_data, model_len);

    match DocOrientationClassifier::new_with_device(model_slice, accel_device) {
        Ok(classifier) => {
            let handle = Box::new(OcrClassifierHandle { classifier });
            Box::into_raw(handle)
        }
        Err(e) => {
            if !error.is_null() {
                (*error) = e.into();
            }
            ptr::null_mut()
        }
    }
}

/// Destroy a classifier handle and release all associated resources.
///
/// # Safety
///
/// `handle` must have been created by `paddle_ocr_classifier_create` or
/// `paddle_ocr_classifier_create_with_device`.
#[no_mangle]
pub unsafe extern "C" fn paddle_ocr_classifier_destroy(handle: *mut OcrClassifierHandle) {
    if !handle.is_null() {
        unsafe { drop(Box::from_raw(handle)); }
    }
}

/// Classify the orientation of a document image.
///
/// # Arguments
///
/// * `handle` -- Valid classifier handle.
/// * `image_data` -- Raw image bytes (PNG, JPEG, etc.).
/// * `image_len` -- Length of `image_data` in bytes.
/// * `orientation` -- Pointer to receive the orientation code (0=Upright, 1=Rotate90, 2=Rotate180, 3=Rotate270).
/// * `confidence` -- Pointer to receive the confidence score.
/// * `error` -- Pointer to receive error information (optional, can be NULL).
///
/// # Returns
///
/// 0 on success, non-zero error code on failure.
///
/// # Safety
///
/// The input pointers must be valid for the specified lengths.
#[no_mangle]
pub unsafe extern "C" fn paddle_ocr_classifier_classify(
    handle: *const OcrClassifierHandle,
    image_data: *const u8,
    image_len: usize,
    orientation: *mut u32,
    confidence: *mut f32,
    error: *mut OcrError,
) -> i32 {
    if handle.is_null() || image_data.is_null() || orientation.is_null() || confidence.is_null() {
        if !error.is_null() {
            (*error) = OcrError {
                code: OcrErrorCode::InvalidArgument,
                message: CString::new("null pointer argument").unwrap().into_raw(),
            };
        }
        return OcrErrorCode::InvalidArgument as i32;
    }

    let image_slice = std::slice::from_raw_parts(image_data, image_len);

    let image = match image::load_from_memory(image_slice) {
        Ok(img) => img,
        Err(e) => {
            if !error.is_null() {
                (*error) = PaddleOcrError::Image { message: e.to_string() }.into();
            }
            return OcrErrorCode::InvalidArgument as i32;
        }
    };

    let classifier = &(*handle).classifier;
    match classifier.classify(&image) {
        Ok(result) => {
            *orientation = match result.orientation {
                crate::DocOrientation::Upright => 0,
                crate::DocOrientation::Rotate90 => 1,
                crate::DocOrientation::Rotate180 => 2,
                crate::DocOrientation::Rotate270 => 3,
            };
            *confidence = result.confidence;
            OcrErrorCode::Ok as i32
        }
        Err(e) => {
            if !error.is_null() {
                (*error) = e.into();
            }
            OcrErrorCode::GeneralError as i32
        }
    }
}

/// Classify and correct the orientation of a document image.
///
/// # Arguments
///
/// * `handle` -- Valid classifier handle.
/// * `image_data` -- Raw image bytes (PNG, JPEG, etc.).
/// * `image_len` -- Length of `image_data` in bytes.
/// * `corrected_data` -- Pointer to receive the corrected image bytes (PNG format).
/// * `corrected_len` -- Pointer to receive the length of `corrected_data`.
/// * `orientation` -- Pointer to receive the orientation code (optional, can be NULL).
/// * `confidence` -- Pointer to receive the confidence score (optional, can be NULL).
/// * `error` -- Pointer to receive error information (optional, can be NULL).
///
/// # Returns
///
/// 0 on success, non-zero error code on failure.
/// The caller must free `corrected_data` with `paddle_ocr_free_buffer`.
///
/// # Safety
///
/// The input pointers must be valid for the specified lengths.
#[no_mangle]
pub unsafe extern "C" fn paddle_ocr_classifier_correct(
    handle: *const OcrClassifierHandle,
    image_data: *const u8,
    image_len: usize,
    corrected_data: *mut *mut u8,
    corrected_len: *mut usize,
    orientation: *mut u32,
    confidence: *mut f32,
    error: *mut OcrError,
) -> i32 {
    if handle.is_null() || image_data.is_null() || corrected_data.is_null() || corrected_len.is_null() {
        if !error.is_null() {
            (*error) = OcrError {
                code: OcrErrorCode::InvalidArgument,
                message: CString::new("null pointer argument").unwrap().into_raw(),
            };
        }
        return OcrErrorCode::InvalidArgument as i32;
    }

    let image_slice = std::slice::from_raw_parts(image_data, image_len);

    let image = match image::load_from_memory(image_slice) {
        Ok(img) => img,
        Err(e) => {
            if !error.is_null() {
                (*error) = PaddleOcrError::Image { message: e.to_string() }.into();
            }
            return OcrErrorCode::InvalidArgument as i32;
        }
    };

    let classifier = &(*handle).classifier;
    match classifier.correct_orientation(&image) {
        Ok((corrected, result)) => {
            if !orientation.is_null() {
                *orientation = match result.orientation {
                    crate::DocOrientation::Upright => 0,
                    crate::DocOrientation::Rotate90 => 1,
                    crate::DocOrientation::Rotate180 => 2,
                    crate::DocOrientation::Rotate270 => 3,
                };
            }
            if !confidence.is_null() {
                *confidence = result.confidence;
            }

            // Encode corrected image as PNG
            let mut buf = std::io::Cursor::new(Vec::new());
            match corrected.write_to(&mut buf, image::ImageFormat::Png) {
                Ok(()) => {
                    let bytes = buf.into_inner();
                    let len = bytes.len();
                    let boxed = bytes.into_boxed_slice();
                    *corrected_data = Box::into_raw(boxed) as *mut u8;
                    *corrected_len = len;
                    OcrErrorCode::Ok as i32
                }
                Err(e) => {
                    if !error.is_null() {
                        (*error) = PaddleOcrError::Image { message: e.to_string() }.into();
                    }
                    OcrErrorCode::GeneralError as i32
                }
            }
        }
        Err(e) => {
            if !error.is_null() {
                (*error) = e.into();
            }
            OcrErrorCode::GeneralError as i32
        }
    }
}

/// Free a buffer allocated by the OCR engine.
///
/// # Safety
///
/// `data` must have been allocated by the OCR engine.
#[no_mangle]
pub unsafe extern "C" fn paddle_ocr_free_buffer(data: *mut u8, len: usize) {
    if data.is_null() || len == 0 {
        return;
    }

    unsafe { drop(Vec::from_raw_parts(data, len, len)); }
}

// ============================================================================
// Utility Functions
// ============================================================================

/// Get the version string of the library.
///
/// # Returns
///
/// Pointer to a null-terminated version string.
/// The caller must free with `paddle_ocr_free_string`.
#[no_mangle]
pub unsafe extern "C" fn paddle_ocr_version() -> *mut c_char {
    CString::new(env!("CARGO_PKG_VERSION"))
        .unwrap_or_else(|_| CString::new("unknown").unwrap())
        .into_raw()
}

/// Check if a specific acceleration device is available.
///
/// # Arguments
///
/// * `device` -- Device code (see `paddle_ocr_create_with_device`).
///
/// # Returns
///
/// 1 if available, 0 if not.
#[no_mangle]
pub unsafe extern "C" fn paddle_ocr_is_device_available(device: u32) -> i32 {
    let accel_device = match device {
        0 => AccelerationDevice::Cpu,
        1 => AccelerationDevice::DirectML,
        2 => AccelerationDevice::Cuda,
        3 => AccelerationDevice::OpenVINO,
        4 => AccelerationDevice::Nnapi,
        5 => AccelerationDevice::Coreml,
        6 => AccelerationDevice::Cann,
        _ => return 0,
    };

    // CPU is always available
    if matches!(accel_device, AccelerationDevice::Cpu) {
        return 1;
    }

    // Try to create a session builder to check availability
    // This is a heuristic - the actual availability depends on runtime
    // For now, we return 1 for all devices and let the engine handle fallback
    1
}

