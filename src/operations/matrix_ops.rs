use anyhow::Result;
use metal::*;
use crate::kernels;
use crate::MetalContext;
use crate::matrix::Matrix;
use std::ops::{Add, Sub, Mul};


/// Performs matrix multiplication on the GPU: C = A * B
pub fn matrix_multiply(context: &MetalContext, a: &Matrix, b: &Matrix) -> Result<Matrix> {
    // Validate input
    if a.cols != b.rows {
        anyhow::bail!("Matrix dimensions incompatible for multiplication");
    }
    
    let m = a.rows;
    let n = b.cols;
    let k = a.cols;
    
    // Load kernel
    let pipeline = context.load_kernel(
        kernels::paths::MATRIX_MUL,
        kernels::functions::MATRIX_MUL
    )?;
    
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
pub fn matrix_add(context: &MetalContext, a: &Matrix, b: &Matrix) -> Result<Matrix> {
    // Validate input
    if a.rows != b.rows || a.cols != b.cols {
        anyhow::bail!("Matrix dimensions must match for addition");
    }
    
    let rows = a.rows;
    let cols = a.cols;
    let size = rows * cols;
    
    // Load kernel
    let pipeline = context.load_kernel(
        kernels::paths::MATRIX_ADD,
        kernels::functions::MATRIX_ADD
    )?;
    
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
            1
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
pub fn matrix_subtract(context: &MetalContext, a: &Matrix, b: &Matrix) -> Result<Matrix> {
    // Validate input
    if a.rows != b.rows || a.cols != b.cols {
        anyhow::bail!("Matrix dimensions must match for subtraction");
    }
    
    let rows = a.rows;
    let cols = a.cols;
    let size = rows * cols;
    
    // Load kernel
    let pipeline = context.load_kernel(
        kernels::paths::MATRIX_SUB,
        kernels::functions::MATRIX_SUB
    )?;
    
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
            1
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
pub fn matrix_transpose(context: &MetalContext, a: &Matrix) -> Result<Matrix> {
    let rows = a.rows;
    let cols = a.cols;
    
    // Load kernel
    let pipeline = context.load_kernel(
        kernels::paths::MATRIX_TRANSPOSE,
        kernels::functions::MATRIX_TRANSPOSE
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
pub fn matrix_scalar_multiply(context: &MetalContext, scalar: f32, a: &Matrix) -> Result<Matrix> {
    let rows = a.rows;
    let cols = a.cols;
    let size = rows * cols;
    
    // Load kernel
    let pipeline = context.load_kernel(
        kernels::paths::MATRIX_SCALAR_MUL,
        kernels::functions::MATRIX_SCALAR_MUL
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
            1
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
