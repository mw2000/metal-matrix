use anyhow::Result;
use kernel_benches::{
    MetalContext, Matrix, Vector,
    vector_add, matrix_multiply, matrix_add, matrix_subtract,
    matrix_transpose, matrix_scalar_multiply, vector_dot_product
};

fn main() -> Result<()> {
    // Initialize logger
    env_logger::init();
    
    // Setup Metal context
    let context = MetalContext::new()?;
    
    // Test vector addition
    test_vector_addition(&context)?;
    
    // Test matrix operations
    test_matrix_operations(&context)?;
    
    // Test vector dot product
    test_vector_dot_product(&context)?;
    
    Ok(())
}

fn test_vector_addition(context: &MetalContext) -> Result<()> {
    println!("\n=== Testing Vector Addition ===");
    
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
    
    println!("Vector addition test completed successfully!");
    Ok(())
}

fn test_matrix_operations(context: &MetalContext) -> Result<()> {
    println!("\n=== Testing Matrix Operations ===");
    
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
    
    // Matrix C: m x k (same shape as A)
    let mut matrix_c = Matrix::new(m, k);
    for i in 0..m {
        for j in 0..k {
            matrix_c.set(i, j, (i + j) as f32);
        }
    }
    
    // 1. Test matrix multiplication
    println!("\n--- Matrix Multiplication ---");
    println!("Matrix A ({}x{}):", m, k);
    print_matrix(&matrix_a);
    
    println!("Matrix B ({}x{}):", k, n);
    print_matrix(&matrix_b);
    
    let mul_result = matrix_multiply(context, &matrix_a, &matrix_b)?;
    println!("A * B ({}x{}):", m, n);
    print_matrix(&mul_result);
    
    // 2. Test matrix addition
    println!("\n--- Matrix Addition ---");
    println!("Matrix A ({}x{}):", m, k);
    print_matrix(&matrix_a);
    
    println!("Matrix C ({}x{}):", m, k);
    print_matrix(&matrix_c);
    
    let add_result = matrix_add(context, &matrix_a, &matrix_c)?;
    println!("A + C ({}x{}):", m, k);
    print_matrix(&add_result);
    
    // 3. Test matrix subtraction
    println!("\n--- Matrix Subtraction ---");
    let sub_result = matrix_subtract(context, &matrix_a, &matrix_c)?;
    println!("A - C ({}x{}):", m, k);
    print_matrix(&sub_result);
    
    // 4. Test matrix transpose
    println!("\n--- Matrix Transpose ---");
    println!("Matrix A ({}x{}):", m, k);
    print_matrix(&matrix_a);
    
    let transpose_result = matrix_transpose(context, &matrix_a)?;
    println!("A^T ({}x{}):", k, m);
    print_matrix(&transpose_result);
    
    // 5. Test scalar multiplication
    println!("\n--- Scalar Multiplication ---");
    let scalar = 2.5;
    println!("Matrix A ({}x{}):", m, k);
    print_matrix(&matrix_a);
    
    let scalar_result = matrix_scalar_multiply(context, scalar, &matrix_a)?;
    println!("{} * A ({}x{}):", scalar, m, k);
    print_matrix(&scalar_result);
    
    println!("Matrix operations tests completed successfully!");
    Ok(())
}

fn test_vector_dot_product(context: &MetalContext) -> Result<()> {
    println!("\n=== Testing Vector Dot Product ===");
    
    // Create test vectors
    let vec_a = Vector::with_data(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    let vec_b = Vector::with_data(vec![5.0, 4.0, 3.0, 2.0, 1.0]);
    
    // Compute dot product
    let dot_result = vector_dot_product(context, &vec_a, &vec_b)?;
    
    // Verify result
    let expected = vec_a.dot_cpu(&vec_b)?;
    println!("Dot product: {} (expected: {})", dot_result, expected);
    assert!((dot_result - expected).abs() < 1e-6);
    
    println!("Vector dot product test completed successfully!");
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