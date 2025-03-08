use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use kernel_benches::{MetalContext, vector_add, Matrix, matrix_multiply};

fn bench_vector_add(c: &mut Criterion) {
    let context = MetalContext::new().unwrap();
    let mut group = c.benchmark_group("vector_operations");
    
    for size in [1024, 10_000, 100_000, 1_000_000].iter() {
        group.bench_with_input(BenchmarkId::new("vector_add", size), size, |b, &size| {
            // Create test data
            let vec_a: Vec<f32> = (0..size).map(|x| x as f32).collect();
            let vec_b: Vec<f32> = (0..size).map(|x| (x * 2) as f32).collect();
            
            b.iter(|| {
                black_box(vector_add(&context, &vec_a, &vec_b).unwrap());
            });
        });
    }
    
    group.finish();
}

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

criterion_group!(benches, bench_vector_add, bench_matrix_multiply);
criterion_main!(benches); 