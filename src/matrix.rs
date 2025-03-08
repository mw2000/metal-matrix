use anyhow::Result;

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
    
    /// Create a vector (1D matrix) with given data
    pub fn vector(data: Vec<f32>) -> Self {
        Self {
            rows: data.len(),
            cols: 1,
            data,
        }
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
    
    /// Check if this matrix is a vector (1D matrix)
    pub fn is_vector(&self) -> bool {
        self.cols == 1 || self.rows == 1
    }
    
    /// Get the size of the matrix if it's a vector
    pub fn vector_size(&self) -> usize {
        if self.cols == 1 {
            self.rows
        } else if self.rows == 1 {
            self.cols
        } else {
            0 // Not a vector
        }
    }
    
    /// Get vector element at index (for 1D matrices)
    pub fn vector_get(&self, index: usize) -> Result<f32> {
        if !self.is_vector() {
            anyhow::bail!("Not a vector");
        }
        
        if self.cols == 1 {
            Ok(self.get(index, 0))
        } else {
            Ok(self.get(0, index))
        }
    }
    
    /// Extract a row as a new Matrix
    pub fn row(&self, row: usize) -> Self {
        let mut data = Vec::with_capacity(self.cols);
        for col in 0..self.cols {
            data.push(self.get(row, col));
        }
        Self {
            rows: 1,
            cols: self.cols,
            data,
        }
    }
    
    /// Extract a column as a new Matrix
    pub fn column(&self, col: usize) -> Self {
        let mut data = Vec::with_capacity(self.rows);
        for row in 0..self.rows {
            data.push(self.get(row, col));
        }
        Self {
            rows: self.rows,
            cols: 1,
            data,
        }
    }
}