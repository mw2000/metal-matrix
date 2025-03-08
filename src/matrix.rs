/*!
 * # Matrix
 *
 * This module provides the core `Matrix` data structure for linear algebra operations.
 *
 * The `Matrix` struct represents a 2D matrix with floating-point elements.
 * It supports both regular matrices and vectors (as 1D matrices).
 */

use anyhow::Result;

/// Represents a 2D matrix with dimensions and data.
///
/// This is the core data structure for all linear algebra operations in the library.
/// It can represent both regular matrices and vectors (as 1D matrices with either
/// one row or one column).
///
/// # Examples
///
/// Creating a new matrix:
/// ```
/// use metal_matrix::Matrix;
///
/// // Create a 3x3 zero matrix
/// let mut matrix = Matrix::new(3, 3);
///
/// // Set some values
/// matrix.set(0, 0, 1.0);
/// matrix.set(1, 1, 2.0);
/// matrix.set(2, 2, 3.0);
/// ```
///
/// Creating a vector:
/// ```
/// use metal_matrix::Matrix;
///
/// // Create a column vector
/// let vector = Matrix::vector(vec![1.0, 2.0, 3.0]);
/// assert_eq!(vector.rows, 3);
/// assert_eq!(vector.cols, 1);
/// ```
#[derive(Clone, Debug)]
pub struct Matrix {
    /// Number of rows in the matrix
    pub rows: usize,

    /// Number of columns in the matrix
    pub cols: usize,

    /// Matrix data in row-major order
    pub data: Vec<f32>,
}

impl Matrix {
    /// Create a new matrix with given dimensions, initialized with zeros.
    ///
    /// # Arguments
    ///
    /// * `rows` - Number of rows
    /// * `cols` - Number of columns
    ///
    /// # Returns
    ///
    /// A new matrix of the specified dimensions, filled with zeros.
    pub fn new(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            data: vec![0.0; rows * cols],
        }
    }

    /// Create a new matrix with given dimensions and data.
    ///
    /// # Arguments
    ///
    /// * `rows` - Number of rows
    /// * `cols` - Number of columns
    /// * `data` - Vector of data in row-major order
    ///
    /// # Returns
    ///
    /// A `Result` containing the new matrix or an error if the data length doesn't match dimensions.
    ///
    /// # Errors
    ///
    /// Returns an error if `data.len() != rows * cols`.
    pub fn with_data(rows: usize, cols: usize, data: Vec<f32>) -> Result<Self> {
        if data.len() != rows * cols {
            anyhow::bail!("Data length does not match matrix dimensions");
        }

        Ok(Self { rows, cols, data })
    }

    /// Create a column vector (1D matrix) with given data.
    ///
    /// # Arguments
    ///
    /// * `data` - Vector of data
    ///
    /// # Returns
    ///
    /// A new matrix with dimensions `(data.len(), 1)`.
    pub fn vector(data: Vec<f32>) -> Self {
        Self {
            rows: data.len(),
            cols: 1,
            data,
        }
    }

    /// Create an identity matrix of size n×n.
    ///
    /// # Arguments
    ///
    /// * `n` - Size of the square matrix
    ///
    /// # Returns
    ///
    /// A new n×n identity matrix (ones on the diagonal, zeros elsewhere).
    pub fn identity(n: usize) -> Self {
        let mut matrix = Self::new(n, n);
        for i in 0..n {
            matrix.set(i, i, 1.0);
        }
        matrix
    }

    /// Get element at position (row, col).
    ///
    /// # Arguments
    ///
    /// * `row` - Row index (0-based)
    /// * `col` - Column index (0-based)
    ///
    /// # Returns
    ///
    /// The value at the specified position.
    ///
    /// # Panics
    ///
    /// Panics if the indices are out of bounds.
    pub fn get(&self, row: usize, col: usize) -> f32 {
        self.data[row * self.cols + col]
    }

    /// Set element at position (row, col).
    ///
    /// # Arguments
    ///
    /// * `row` - Row index (0-based)
    /// * `col` - Column index (0-based)
    /// * `value` - Value to set
    ///
    /// # Panics
    ///
    /// Panics if the indices are out of bounds.
    pub fn set(&mut self, row: usize, col: usize, value: f32) {
        self.data[row * self.cols + col] = value;
    }

    /// Check if this matrix is a vector (1D matrix).
    ///
    /// # Returns
    ///
    /// `true` if the matrix has either one row or one column, `false` otherwise.
    pub fn is_vector(&self) -> bool {
        self.cols == 1 || self.rows == 1
    }

    /// Get the size of the matrix if it's a vector.
    ///
    /// # Returns
    ///
    /// The number of elements if the matrix is a vector, or 0 if it's not a vector.
    pub fn vector_size(&self) -> usize {
        if self.cols == 1 {
            self.rows
        } else if self.rows == 1 {
            self.cols
        } else {
            0 // Not a vector
        }
    }

    /// Get vector element at index (for 1D matrices).
    ///
    /// # Arguments
    ///
    /// * `index` - Element index (0-based)
    ///
    /// # Returns
    ///
    /// A `Result` containing the value at the specified index or an error if the matrix is not a vector.
    ///
    /// # Errors
    ///
    /// Returns an error if the matrix is not a vector.
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

    /// Extract a row as a new Matrix.
    ///
    /// # Arguments
    ///
    /// * `row` - Row index (0-based)
    ///
    /// # Returns
    ///
    /// A new 1×n matrix containing the specified row.
    ///
    /// # Panics
    ///
    /// Panics if the row index is out of bounds.
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

    /// Extract a column as a new Matrix.
    ///
    /// # Arguments
    ///
    /// * `col` - Column index (0-based)
    ///
    /// # Returns
    ///
    /// A new m×1 matrix containing the specified column.
    ///
    /// # Panics
    ///
    /// Panics if the column index is out of bounds.
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
