pub mod kernels;
pub mod metal_context;
pub mod operations;

pub use metal_context::MetalContext;
pub use operations::*;
pub use operations::matrix_ops::{Matrix, Vector}; 