use anyhow::Result;
use metal::*;
use crate::kernels;
use crate::MetalContext;
use std::ops::{Add, Sub, Mul};

/// Represents a 2D matrix with dimensions and data
#[derive(Clone, Debug)]
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
    
    /// Create an identity matrix of size n×n
    pub fn identity(n: usize) -> Self {
        let mut matrix = Self::new(n, n);
        for i in 0..n {
            matrix.set(i, i, 1.0);
        }
        matrix
    }
    
    /// Get element at position (row, col)
    pub fn get(&self, row: usize, col: usize) -> f32 {
        self.data[row * self.cols + col]
    }
    
    /// Set element at position (row, col)
    pub fn set(&mut self, row: usize, col: usize, value: f32) {
        self.data[row * self.cols + col] = value;
    }
    
    /// Transpose the matrix (CPU implementation)
    pub fn transpose_cpu(&self) -> Self {
        let mut result = Self::new(self.cols, self.rows);
        for i in 0..self.rows {
            for j in 0..self.cols {
                result.set(j, i, self.get(i, j));
            }
        }
        result
    }
    
    /// Scale the matrix by a scalar (CPU implementation)
    pub fn scale_cpu(&self, scalar: f32) -> Self {
        let mut result = self.clone();
        for i in 0..self.data.len() {
            result.data[i] *= scalar;
        }
        result
    }
    
    /// Extract a row as a Vector
    pub fn row(&self, row: usize) -> Vector {
        let mut data = Vec::with_capacity(self.cols);
        for col in 0..self.cols {
            data.push(self.get(row, col));
        }
        Vector { data }
    }
    
    /// Extract a column as a Vector
    pub fn column(&self, col: usize) -> Vector {
        let mut data = Vec::with_capacity(self.rows);
        for row in 0..self.rows {
            data.push(self.get(row, col));
        }
        Vector { data }
    }
}

/// Represents a vector with data
#[derive(Clone, Debug)]
pub struct Vector {
    pub data: Vec<f32>,
}

impl Vector {
    /// Create a new vector with given size
    pub fn new(size: usize) -> Self {
        Self {
            data: vec![0.0; size],
        }
    }
    
    /// Create a new vector with given data
    pub fn with_data(data: Vec<f32>) -> Self {
        Self { data }
    }
    
    /// Get the size of the vector
    pub fn size(&self) -> usize {
        self.data.len()
    }
    
    /// Get element at position
    pub fn get(&self, index: usize) -> f32 {
        self.data[index]
    }
    
    /// Set element at position
    pub fn set(&mut self, index: usize, value: f32) {
        self.data[index] = value;
    }
    
    /// Compute dot product with another vector (CPU implementation)
    pub fn dot_cpu(&self, other: &Vector) -> Result<f32> {
        if self.size() != other.size() {
            anyhow::bail!("Vector dimensions must match for dot product");
        }
        
        let mut result = 0.0;
        for i in 0..self.size() {
            result += self.get(i) * other.get(i);
        }
        
        Ok(result)
    }
    
    /// Scale the vector by a scalar (CPU implementation)
    pub fn scale_cpu(&self, scalar: f32) -> Self {
        let mut result = self.clone();
        for i in 0..self.data.len() {
            result.data[i] *= scalar;
        }
        result
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

/// Computes the dot product of two vectors on the GPU
pub fn vector_dot_product(context: &MetalContext, a: &Vector, b: &Vector) -> Result<f32> {
    // Validate input
    if a.size() != b.size() {
        anyhow::bail!("Vector dimensions must match for dot product");
    }
    
    let size = a.size();
    
    // Load kernel
    let pipeline = context.load_kernel(
        kernels::paths::VECTOR_DOT,
        kernels::functions::VECTOR_DOT
    )?;
    
    // Create buffers
    let buffer_a = context.new_buffer_with_data(&a.data);
    let buffer_b = context.new_buffer_with_data(&b.data);
    let buffer_result = context.new_buffer::<f32>(1);
    let buffer_size = context.new_buffer_with_data(&[size as u32]);
    
    // Execute computation
    context.execute_compute(|encoder| {
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_buffer(0, Some(&buffer_a), 0);
        encoder.set_buffer(1, Some(&buffer_b), 0);
        encoder.set_buffer(2, Some(&buffer_result), 0);
        encoder.set_buffer(3, Some(&buffer_size), 0);
        
        // For dot product, we'll use a reduction approach
        let threadgroup_size = pipeline.max_total_threads_per_threadgroup().min(256);
        let grid_size = MTLSize::new(1, 1, 1);
        let threadgroup_size = MTLSize::new(threadgroup_size as u64, 1, 1);
        
        // We need to set threadgroup memory size for the reduction
        let threadgroup_mem_length = threadgroup_size.width * std::mem::size_of::<f32>() as u64;
        encoder.set_threadgroup_memory_length(0, threadgroup_mem_length);
        
        encoder.dispatch_threads(grid_size, threadgroup_size);
    })?;
    
    // Read result
    let result_ptr = buffer_result.contents() as *const f32;
    let result = unsafe { *result_ptr };
    
    Ok(result)
} 