/// Paths to Metal kernel files
pub mod paths {
    /// Path to the vector addition kernel
    pub const VECTOR_ADD: &str = "src/kernels/vector_add.metal";
    
    /// Path to the matrix multiplication kernel (to be implemented)
    pub const MATRIX_MUL: &str = "src/kernels/matrix_mul.metal";
}

/// Names of kernel functions
pub mod functions {
    /// Vector addition kernel function name
    pub const VECTOR_ADD: &str = "add_arrays";
    
    /// Matrix multiplication kernel function name (to be implemented)
    pub const MATRIX_MUL: &str = "matrix_multiply";
} 