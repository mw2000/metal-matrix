/*!
 * # Metal Kernels
 * 
 * This module contains paths and function names for all Metal kernel files used in the library.
 * 
 * The kernels are organized into two submodules:
 * - `paths`: Contains the file paths to the Metal kernel files
 * - `functions`: Contains the function names within those kernel files
 * 
 * This organization makes it easy to load and use the kernels throughout the library.
 */

/// Paths to Metal kernel files
pub mod paths {
    /// Path to the matrix multiplication kernel
    pub const MATRIX_MUL: &str = "src/kernels/matrix_mul.metal";
    
    /// Path to the matrix addition kernel
    pub const MATRIX_ADD: &str = "src/kernels/matrix_add.metal";
    
    /// Path to the matrix subtraction kernel
    pub const MATRIX_SUB: &str = "src/kernels/matrix_sub.metal";
    
    /// Path to the matrix transpose kernel
    pub const MATRIX_TRANSPOSE: &str = "src/kernels/matrix_transpose.metal";
    
    /// Path to the matrix scalar multiplication kernel
    pub const MATRIX_SCALAR_MUL: &str = "src/kernels/matrix_scalar_mul.metal";
}

/// Names of kernel functions
pub mod functions {
    /// Matrix multiplication kernel function name
    pub const MATRIX_MUL: &str = "matrix_multiply";
    
    /// Matrix addition kernel function name
    pub const MATRIX_ADD: &str = "matrix_add";
    
    /// Matrix subtraction kernel function name
    pub const MATRIX_SUB: &str = "matrix_subtract";
    
    /// Matrix transpose kernel function name
    pub const MATRIX_TRANSPOSE: &str = "matrix_transpose";
    
    /// Matrix scalar multiplication kernel function name
    pub const MATRIX_SCALAR_MUL: &str = "matrix_scalar_multiply";
}
 