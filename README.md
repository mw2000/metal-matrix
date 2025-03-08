# Metal Matrix

A high-performance linear algebra library with Metal GPU acceleration for macOS and iOS.

## Overview

Metal Matrix is a Rust library that provides GPU-accelerated matrix operations using Apple's Metal framework. It's designed for efficient computation of common linear algebra operations like matrix multiplication, addition, subtraction, transposition, and scalar multiplication.

## Features

- **GPU Acceleration**: Leverages Apple's Metal framework for high-performance matrix operations
- **Clean API**: Simple, ergonomic interface for matrix operations
- **Flexible Matrix Type**: Supports both regular matrices and vectors (as 1D matrices)
- **Comprehensive Error Handling**: Clear error messages for dimension mismatches and other issues
- **Well-Documented**: Extensive documentation for all types and functions

## Installation

Add this to your `Cargo.toml`:

```toml
[dependencies]
metal-matrix = "0.1.0"
```

## Requirements

- macOS or iOS device with Metal support
- Rust 1.56 or later

## Usage

### Basic Example

```rust
use metal_matrix::{MetalContext, Matrix, matrix_multiply};
use anyhow::Result;

fn main() -> Result<()> {
    // Initialize Metal context
    let context = MetalContext::new()?;
    
    // Create matrices
    let a = Matrix::with_data(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])?;
    let b = Matrix::with_data(3, 2, vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0])?;
    
    // Multiply matrices
    let result = matrix_multiply(&context, &a, &b)?;
    
    // Print result
    for i in 0..result.rows {
        for j in 0..result.cols {
            print!("{:.1} ", result.get(i, j));
        }
        println!();
    }
    
    Ok(())
}
```

### Available Operations

- **Matrix Multiplication**: `matrix_multiply(context, &a, &b)`
- **Matrix Addition**: `matrix_add(context, &a, &b)`
- **Matrix Subtraction**: `matrix_subtract(context, &a, &b)`
- **Matrix Transpose**: `matrix_transpose(context, &a)`
- **Scalar Multiplication**: `matrix_scalar_multiply(context, scalar, &a)`

### Working with Vectors

Vectors are represented as 1D matrices (either a single row or a single column):

```rust
// Create a column vector
let vec_a = Matrix::vector(vec![1.0, 2.0, 3.0, 4.0, 5.0]);

// Check if a matrix is a vector
if vec_a.is_vector() {
    println!("Vector size: {}", vec_a.vector_size());
}

// Get a specific element from a vector
let value = vec_a.vector_get(2)?; // Gets the third element
```

## Performance Considerations

- The library automatically selects appropriate threadgroup sizes for different operations
- For very small matrices, the overhead of GPU operations might outweigh the benefits

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. 