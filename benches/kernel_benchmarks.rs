use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use kernel_benches::{matrix_add, matrix_multiply, matrix_scalar_multiply, matrix_subtract, matrix_transpose, Matrix, MetalContext};

fn bench_matrix_multiply(c: &mut Criterion) {
    let context = MetalContext::new().unwrap();
    let mut group = c.benchmark_group("matrix_operations");
    
    // Test different matrix sizes
    for size in [32, 64, 128, 256, 512].iter() {
        group.bench_with_input(BenchmarkId::new("matrix_multiply", size), size, |b, &size| {
            // Create square matrices of the given size
            let mut matrix_a = Matrix::new(size, size);
            let mut matrix_b = Matrix::new(size, size);
            
            // Initialize with some data
            for i in 0..size {
                for j in 0..size {
                    matrix_a.set(i, j, (i * size + j) as f32 * 0.01);
                    matrix_b.set(i, j, (j * size + i) as f32 * 0.01);
                }
            }
            
            b.iter(|| {
                black_box(matrix_multiply(&context, &matrix_a, &matrix_b).unwrap());
            });
        });
    }
    
    group.finish();
}

fn bench_matrix_add(c: &mut Criterion) {
    let context = MetalContext::new().unwrap();
    let mut group = c.benchmark_group("matrix_operations");
    
    // Test different matrix sizes
    for size in [32, 64, 128, 256, 512].iter() {
        group.bench_with_input(BenchmarkId::new("matrix_add", size), size, |b, &size| {
            // Create square matrices of the given size
            let mut matrix_a = Matrix::new(size, size);
            let mut matrix_b = Matrix::new(size, size);
            
            // Initialize with some data
            for i in 0..size {
                for j in 0..size {
                    matrix_a.set(i, j, (i * size + j) as f32 * 0.01);
                    matrix_b.set(i, j, (j * size + i) as f32 * 0.01);
                }
            }
            
            b.iter(|| {
                black_box(matrix_add(&context, &matrix_a, &matrix_b).unwrap());
            });
        });
    }

    group.finish();
}

fn bench_matrix_subtract(c: &mut Criterion) {
    let context = MetalContext::new().unwrap();
    let mut group = c.benchmark_group("matrix_operations");
    
    // Test different matrix sizes  
    for size in [32, 64, 128, 256, 512].iter() {
        group.bench_with_input(BenchmarkId::new("matrix_subtract", size), size, |b, &size| {
            // Create square matrices of the given size
            let mut matrix_a = Matrix::new(size, size);
            let mut matrix_b = Matrix::new(size, size);
            
            // Initialize with some data
            for i in 0..size {
                for j in 0..size {
                    matrix_a.set(i, j, (i * size + j) as f32 * 0.01);
                    matrix_b.set(i, j, (j * size + i) as f32 * 0.01);
                }
            }
            
            b.iter(|| {
                black_box(matrix_subtract(&context, &matrix_a, &matrix_b).unwrap());
            });
        });
    }   

    group.finish();
}

fn bench_matrix_transpose(c: &mut Criterion) {
    let context = MetalContext::new().unwrap();
    let mut group = c.benchmark_group("matrix_operations");
    
    // Test different matrix sizes
    for size in [32, 64, 128, 256, 512].iter() {
        group.bench_with_input(BenchmarkId::new("matrix_transpose", size), size, |b, &size| {
            // Create square matrices of the given size
            let mut matrix = Matrix::new(size, size);
            
            // Initialize with some data
            for i in 0..size {
                for j in 0..size {
                    matrix.set(i, j, (i * size + j) as f32 * 0.01);
                }
            }
            
            b.iter(|| {
                black_box(matrix_transpose(&context, &matrix).unwrap());
            });
        });
    }

    group.finish();
}   

fn bench_matrix_scalar_multiply(c: &mut Criterion) {
    let context = MetalContext::new().unwrap();
    let mut group = c.benchmark_group("matrix_operations");
    
    // Test different matrix sizes
    for size in [32, 64, 128, 256, 512].iter() {
        group.bench_with_input(BenchmarkId::new("matrix_scalar_multiply", size), size, |b, &size| {
            // Create square matrices of the given size
            let mut matrix = Matrix::new(size, size);
            
            // Initialize with some data
            for i in 0..size {
                for j in 0..size {
                    matrix.set(i, j, (i * size + j) as f32 * 0.01);
                }
            }
            
            b.iter(|| {
                black_box(matrix_scalar_multiply(&context, 2.0, &matrix).unwrap());
            });
        });
    }
    
    group.finish();
}
    

criterion_group!(benches, bench_matrix_multiply, bench_matrix_add, bench_matrix_subtract, bench_matrix_transpose, bench_matrix_scalar_multiply);
criterion_main!(benches); 