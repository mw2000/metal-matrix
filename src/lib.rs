/*!
 * # Metal Matrix
 *
 * A high-performance linear algebra library with Metal GPU acceleration.
 *
 * This library provides GPU-accelerated matrix operations using Apple's Metal framework.
 * It's designed for efficient computation of common linear algebra operations like
 * matrix multiplication, addition, subtraction, transposition, and scalar multiplication.
 *
 * ## Features
 *
 * - GPU-accelerated matrix operations
 * - Clean, ergonomic API
 * - Support for vectors as 1D matrices
 * - CPU fallback implementations
 * - Comprehensive error handling
 *
 * ## Example
 *
 * ```rust
 * use metal_matrix::{MetalContext, Matrix, matrix_multiply};
 * use anyhow::Result;
 *
 * fn main() -> Result<()> {
 *     // Initialize Metal context
 *     let context = MetalContext::new()?;
 *     
 *     // Create matrices
 *     let mut a = Matrix::new(3, 2);
 *     let mut b = Matrix::new(2, 4);
 *     
 *     // Fill matrices with data
 *     // ...
 *     
 *     // Multiply matrices
 *     let result = matrix_multiply(&context, &a, &b)?;
 *     
 *     Ok(())
 * }
 * ```
 */

/// Metal kernel definitions and paths
pub mod kernels;

/// Metal context and device management
pub mod metal_context;

/// Matrix operations implementation
pub mod operations;

/// Matrix data structure and methods
pub mod matrix;

pub use matrix::Matrix;
pub use metal_context::MetalContext;
pub use operations::*;
