use anyhow::Result;
use metal::*;
use crate::kernels;
use crate::MetalContext;

/// Represents a 2D matrix with dimensions and data
pub struct Matrix {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<f32>,
}

impl Matrix {
    /// Create a new matrix with given dimensions
    pub fn new(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            data: vec![0.0; rows * cols],
        }
    }
    
    /// Create a new matrix with given dimensions and data
    pub fn with_data(rows: usize, cols: usize, data: Vec<f32>) -> Result<Self> {
        if data.len() != rows * cols {
            anyhow::bail!("Data length does not match matrix dimensions");
        }
        
        Ok(Self {
            rows,
            cols,
            data,
        })
    }
    
    /// Get element at position (row, col)
    pub fn get(&self, row: usize, col: usize) -> f32 {
        self.data[row * self.cols + col]
    }
    
    /// Set element at position (row, col)
    pub fn set(&mut self, row: usize, col: usize, value: f32) {
        self.data[row * self.cols + col] = value;
    }
}

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