use anyhow::Result;
use kernel_benches::{MetalContext, vector_add, Matrix, matrix_multiply};

fn main() -> Result<()> {
    // Initialize logger
    env_logger::init();
    
    // Setup Metal context
    let context = MetalContext::new()?;
    
    // Test vector addition
    test_vector_addition(&context)?;
    
    // Test matrix multiplication
    test_matrix_multiplication(&context)?;
    
    Ok(())
}

fn test_vector_addition(context: &MetalContext) -> Result<()> {
    println!("Testing vector addition...");
    
    // Create test data
    let size = 1024;
    let a: Vec<f32> = (0..size).map(|x| x as f32).collect();
    let b: Vec<f32> = (0..size).map(|x| (x * 2) as f32).collect();
    
    // Perform vector addition
    let result = vector_add(context, &a, &b)?;
    
    // Verify results
    println!("Vector addition result[0] = {}", result[0]); // Should be 0.0
    println!("Vector addition result[1] = {}", result[1]); // Should be 3.0
    
    // Verify a few more elements
    for i in 0..5 {
        let expected = a[i] + b[i];
        let actual = result[i];
        println!("result[{}] = {} (expected: {})", i, actual, expected);
        assert!((actual - expected).abs() < 1e-6);
    }
    
    println!("Vector addition test completed successfully!\n");
    Ok(())
}

fn test_matrix_multiplication(context: &MetalContext) -> Result<()> {
    println!("Testing matrix multiplication...");
    
    // Create test matrices
    let m = 4; // rows of A
    let k = 3; // cols of A, rows of B
    let n = 2; // cols of B
    
    // Matrix A: m x k
    let mut matrix_a = Matrix::new(m, k);
    for i in 0..m {
        for j in 0..k {
            matrix_a.set(i, j, (i * k + j) as f32);
        }
    }
    
    // Matrix B: k x n
    let mut matrix_b = Matrix::new(k, n);
    for i in 0..k {
        for j in 0..n {
            matrix_b.set(i, j, (i * n + j) as f32);
        }
    }
    
    // Perform matrix multiplication
    let result = matrix_multiply(context, &matrix_a, &matrix_b)?;
    
    // Print matrices for verification
    println!("Matrix A ({}x{}):", m, k);
    print_matrix(&matrix_a);
    
    println!("Matrix B ({}x{}):", k, n);
    print_matrix(&matrix_b);
    
    println!("Result Matrix ({}x{}):", m, n);
    print_matrix(&result);
    
    println!("Matrix multiplication test completed!\n");
    Ok(())
}

fn print_matrix(matrix: &Matrix) {
    for i in 0..matrix.rows {
        for j in 0..matrix.cols {
            print!("{:.1} ", matrix.get(i, j));
        }
        println!();
    }
    println!();
}