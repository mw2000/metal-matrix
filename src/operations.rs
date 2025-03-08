/*!
 * # Matrix Operations
 *
 * This module provides GPU-accelerated matrix operations using Metal.
 *
 * All operations are implemented as functions that take a `MetalContext` and
 * input matrices, and return a new matrix with the result of the operation.
 *
 * ## Available Operations
 *
 * - Matrix multiplication (`matrix_multiply`)
 * - Matrix addition (`matrix_add`)
 * - Matrix subtraction (`matrix_subtract`)
 * - Matrix transpose (`matrix_transpose`)
 * - Scalar multiplication (`matrix_scalar_multiply`)
 * - Dot product (`dot_product`)
 *
 * Each operation validates the input dimensions and returns appropriate errors
 * if the inputs are incompatible.
 */

use crate::kernels;
use crate::matrix::Matrix;
use crate::MetalContext;
use anyhow::Result;
use metal::*;

/// Performs matrix multiplication on the GPU: C = A * B
///
/// Computes the matrix product of two matrices using the GPU.
///
/// # Arguments
///
/// * `context` - The Metal context for GPU computation
/// * `a` - The first matrix (m × k)
/// * `b` - The second matrix (k × n)
///
/// # Returns
///
/// A `Result` containing the product matrix (m × n) or an error.
///
/// # Errors
///
/// Returns an error if the matrices have incompatible dimensions (a.cols != b.rows).
///
/// # Example
///
/// ```
/// use metal_matrix::{MetalContext, Matrix, matrix_multiply};
///
/// let context = MetalContext::new().unwrap();
/// let a = Matrix::with_data(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
/// let b = Matrix::with_data(3, 2, vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0]).unwrap();
///
/// let result = matrix_multiply(&context, &a, &b).unwrap();
/// ```
pub fn matrix_multiply(context: &MetalContext, a: &Matrix, b: &Matrix) -> Result<Matrix> {
    // Validate input
    if a.cols != b.rows {
        anyhow::bail!("Matrix dimensions incompatible for multiplication");
    }

    let m = a.rows;
    let n = b.cols;
    let k = a.cols;

    // Load kernel
    let pipeline =
        context.load_kernel(kernels::paths::MATRIX_MUL, kernels::functions::MATRIX_MUL)?;

    // Create buffers
    let buffer_a = context.new_buffer_with_data(&a.data);
    let buffer_b = context.new_buffer_with_data(&b.data);
    let buffer_result = context.new_buffer::<f32>(m * n);

    // Create dimension buffers
    let m_val = m as u32;
    let n_val = n as u32;
    let k_val = k as u32;

    let buffer_m = context.new_buffer_with_data(&[m_val]);
    let buffer_n = context.new_buffer_with_data(&[n_val]);
    let buffer_k = context.new_buffer_with_data(&[k_val]);

    // Execute computation
    context.execute_compute(|encoder| {
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_buffer(0, Some(&buffer_a), 0);
        encoder.set_buffer(1, Some(&buffer_b), 0);
        encoder.set_buffer(2, Some(&buffer_result), 0);
        encoder.set_buffer(3, Some(&buffer_m), 0);
        encoder.set_buffer(4, Some(&buffer_n), 0);
        encoder.set_buffer(5, Some(&buffer_k), 0);

        let grid_size = MTLSize::new(n as u64, m as u64, 1);

        // Calculate optimal threadgroup size
        let max_threads = pipeline.max_total_threads_per_threadgroup();
        let width = (n as u64).min(16);
        let height = (max_threads as u64 / width).min(m as u64).max(1);

        let threadgroup_size = MTLSize::new(width, height, 1);
        encoder.dispatch_threads(grid_size, threadgroup_size);
    })?;

    // Read results
    let result_ptr = buffer_result.contents() as *const f32;
    let mut result_data = vec![0.0f32; m * n];

    unsafe {
        std::ptr::copy_nonoverlapping(result_ptr, result_data.as_mut_ptr(), m * n);
    }

    Ok(Matrix::with_data(m, n, result_data)?)
}

/// Performs matrix addition on the GPU: C = A + B
///
/// Computes the element-wise sum of two matrices using the GPU.
///
/// # Arguments
///
/// * `context` - The Metal context for GPU computation
/// * `a` - The first matrix (m × n)
/// * `b` - The second matrix (m × n)
///
/// # Returns
///
/// A `Result` containing the sum matrix (m × n) or an error.
///
/// # Errors
///
/// Returns an error if the matrices have different dimensions.
///
/// # Example
///
/// ```
/// use metal_matrix::{MetalContext, Matrix, matrix_add};
///
/// let context = MetalContext::new().unwrap();
/// let a = Matrix::with_data(2, 2, vec![1.0, 2.0, 3.0, 4.0]).unwrap();
/// let b = Matrix::with_data(2, 2, vec![5.0, 6.0, 7.0, 8.0]).unwrap();
///
/// let result = matrix_add(&context, &a, &b).unwrap();
/// ```
pub fn matrix_add(context: &MetalContext, a: &Matrix, b: &Matrix) -> Result<Matrix> {
    // Validate input
    if a.rows != b.rows || a.cols != b.cols {
        anyhow::bail!("Matrix dimensions must match for addition");
    }

    let rows = a.rows;
    let cols = a.cols;
    let size = rows * cols;

    // Load kernel
    let pipeline =
        context.load_kernel(kernels::paths::MATRIX_ADD, kernels::functions::MATRIX_ADD)?;

    // Create buffers
    let buffer_a = context.new_buffer_with_data(&a.data);
    let buffer_b = context.new_buffer_with_data(&b.data);
    let buffer_result = context.new_buffer::<f32>(size);

    // Execute computation
    context.execute_compute(|encoder| {
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_buffer(0, Some(&buffer_a), 0);
        encoder.set_buffer(1, Some(&buffer_b), 0);
        encoder.set_buffer(2, Some(&buffer_result), 0);

        let grid_size = MTLSize::new(size as u64, 1, 1);
        let threadgroup_size = MTLSize::new(
            pipeline.max_total_threads_per_threadgroup().min(256) as u64,
            1,
            1,
        );
        encoder.dispatch_threads(grid_size, threadgroup_size);
    })?;

    // Read results
    let result_ptr = buffer_result.contents() as *const f32;
    let mut result_data = vec![0.0f32; size];

    unsafe {
        std::ptr::copy_nonoverlapping(result_ptr, result_data.as_mut_ptr(), size);
    }

    Ok(Matrix::with_data(rows, cols, result_data)?)
}

/// Performs matrix subtraction on the GPU: C = A - B
///
/// Computes the element-wise difference of two matrices using the GPU.
///
/// # Arguments
///
/// * `context` - The Metal context for GPU computation
/// * `a` - The first matrix (m × n)
/// * `b` - The second matrix (m × n)
///
/// # Returns
///
/// A `Result` containing the difference matrix (m × n) or an error.
///
/// # Errors
///
/// Returns an error if the matrices have different dimensions.
///
/// # Example
///
/// ```
/// use metal_matrix::{MetalContext, Matrix, matrix_subtract};
///
/// let context = MetalContext::new().unwrap();
/// let a = Matrix::with_data(2, 2, vec![5.0, 6.0, 7.0, 8.0]).unwrap();
/// let b = Matrix::with_data(2, 2, vec![1.0, 2.0, 3.0, 4.0]).unwrap();
///
/// let result = matrix_subtract(&context, &a, &b).unwrap();
/// ```
pub fn matrix_subtract(context: &MetalContext, a: &Matrix, b: &Matrix) -> Result<Matrix> {
    // Validate input
    if a.rows != b.rows || a.cols != b.cols {
        anyhow::bail!("Matrix dimensions must match for subtraction");
    }

    let rows = a.rows;
    let cols = a.cols;
    let size = rows * cols;

    // Load kernel
    let pipeline =
        context.load_kernel(kernels::paths::MATRIX_SUB, kernels::functions::MATRIX_SUB)?;

    // Create buffers
    let buffer_a = context.new_buffer_with_data(&a.data);
    let buffer_b = context.new_buffer_with_data(&b.data);
    let buffer_result = context.new_buffer::<f32>(size);

    // Execute computation
    context.execute_compute(|encoder| {
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_buffer(0, Some(&buffer_a), 0);
        encoder.set_buffer(1, Some(&buffer_b), 0);
        encoder.set_buffer(2, Some(&buffer_result), 0);

        let grid_size = MTLSize::new(size as u64, 1, 1);
        let threadgroup_size = MTLSize::new(
            pipeline.max_total_threads_per_threadgroup().min(256) as u64,
            1,
            1,
        );
        encoder.dispatch_threads(grid_size, threadgroup_size);
    })?;

    // Read results
    let result_ptr = buffer_result.contents() as *const f32;
    let mut result_data = vec![0.0f32; size];

    unsafe {
        std::ptr::copy_nonoverlapping(result_ptr, result_data.as_mut_ptr(), size);
    }

    Ok(Matrix::with_data(rows, cols, result_data)?)
}

/// Performs matrix transpose on the GPU: B = A^T
///
/// Computes the transpose of a matrix using the GPU.
///
/// # Arguments
///
/// * `context` - The Metal context for GPU computation
/// * `a` - The input matrix (m × n)
///
/// # Returns
///
/// A `Result` containing the transposed matrix (n × m) or an error.
///
/// # Example
///
/// ```
/// use metal_matrix::{MetalContext, Matrix, matrix_transpose};
///
/// let context = MetalContext::new().unwrap();
/// let a = Matrix::with_data(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
///
/// let result = matrix_transpose(&context, &a).unwrap();
/// assert_eq!(result.rows, 3);
/// assert_eq!(result.cols, 2);
/// ```
pub fn matrix_transpose(context: &MetalContext, a: &Matrix) -> Result<Matrix> {
    let rows = a.rows;
    let cols = a.cols;

    // Load kernel
    let pipeline = context.load_kernel(
        kernels::paths::MATRIX_TRANSPOSE,
        kernels::functions::MATRIX_TRANSPOSE,
    )?;

    // Create buffers
    let buffer_a = context.new_buffer_with_data(&a.data);
    let buffer_result = context.new_buffer::<f32>(rows * cols);

    // Create dimension buffers
    let rows_val = rows as u32;
    let cols_val = cols as u32;

    let buffer_rows = context.new_buffer_with_data(&[rows_val]);
    let buffer_cols = context.new_buffer_with_data(&[cols_val]);

    // Execute computation
    context.execute_compute(|encoder| {
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_buffer(0, Some(&buffer_a), 0);
        encoder.set_buffer(1, Some(&buffer_result), 0);
        encoder.set_buffer(2, Some(&buffer_rows), 0);
        encoder.set_buffer(3, Some(&buffer_cols), 0);

        let grid_size = MTLSize::new(cols as u64, rows as u64, 1);

        // Calculate optimal threadgroup size
        let max_threads = pipeline.max_total_threads_per_threadgroup();
        let width = (cols as u64).min(16);
        let height = (max_threads as u64 / width).min(rows as u64).max(1);

        let threadgroup_size = MTLSize::new(width, height, 1);
        encoder.dispatch_threads(grid_size, threadgroup_size);
    })?;

    // Read results
    let result_ptr = buffer_result.contents() as *const f32;
    let mut result_data = vec![0.0f32; rows * cols];

    unsafe {
        std::ptr::copy_nonoverlapping(result_ptr, result_data.as_mut_ptr(), rows * cols);
    }

    Ok(Matrix::with_data(cols, rows, result_data)?)
}

/// Performs scalar multiplication on the GPU: B = scalar * A
///
/// Multiplies each element of a matrix by a scalar value using the GPU.
///
/// # Arguments
///
/// * `context` - The Metal context for GPU computation
/// * `scalar` - The scalar value to multiply by
/// * `a` - The input matrix (m × n)
///
/// # Returns
///
/// A `Result` containing the scaled matrix (m × n) or an error.
///
/// # Example
///
/// ```
/// use metal_matrix::{MetalContext, Matrix, matrix_scalar_multiply};
///
/// let context = MetalContext::new().unwrap();
/// let a = Matrix::with_data(2, 2, vec![1.0, 2.0, 3.0, 4.0]).unwrap();
///
/// let result = matrix_scalar_multiply(&context, 2.5, &a).unwrap();
/// ```
pub fn matrix_scalar_multiply(context: &MetalContext, scalar: f32, a: &Matrix) -> Result<Matrix> {
    let rows = a.rows;
    let cols = a.cols;
    let size = rows * cols;

    // Load kernel
    let pipeline = context.load_kernel(
        kernels::paths::MATRIX_SCALAR_MUL,
        kernels::functions::MATRIX_SCALAR_MUL,
    )?;

    // Create buffers
    let buffer_a = context.new_buffer_with_data(&a.data);
    let buffer_scalar = context.new_buffer_with_data(&[scalar]);
    let buffer_result = context.new_buffer::<f32>(size);

    // Execute computation
    context.execute_compute(|encoder| {
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_buffer(0, Some(&buffer_a), 0);
        encoder.set_buffer(1, Some(&buffer_scalar), 0);
        encoder.set_buffer(2, Some(&buffer_result), 0);

        let grid_size = MTLSize::new(size as u64, 1, 1);
        let threadgroup_size = MTLSize::new(
            pipeline.max_total_threads_per_threadgroup().min(256) as u64,
            1,
            1,
        );
        encoder.dispatch_threads(grid_size, threadgroup_size);
    })?;

    // Read results
    let result_ptr = buffer_result.contents() as *const f32;
    let mut result_data = vec![0.0f32; size];

    unsafe {
        std::ptr::copy_nonoverlapping(result_ptr, result_data.as_mut_ptr(), size);
    }

    Ok(Matrix::with_data(rows, cols, result_data)?)
}
