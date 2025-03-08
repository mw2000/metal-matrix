use anyhow::Result;
use metal::*;
use crate::kernels;
use crate::MetalContext;

/// Performs vector addition on the GPU: result = a + b
pub fn vector_add(context: &MetalContext, a: &[f32], b: &[f32]) -> Result<Vec<f32>> {
    // Validate input
    if a.len() != b.len() {
        anyhow::bail!("Vector dimensions must match for addition");
    }
    
    let size = a.len();
    
    // Load kernel
    let pipeline = context.load_kernel(
        kernels::paths::VECTOR_ADD,
        kernels::functions::VECTOR_ADD
    )?;
    
    // Create buffers
    let buffer_a = context.new_buffer_with_data(a);
    let buffer_b = context.new_buffer_with_data(b);
    let buffer_result = context.new_buffer::<f32>(size);
    
    // Execute computation
    context.execute_compute(|encoder| {
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_buffer(0, Some(&buffer_a), 0);
        encoder.set_buffer(1, Some(&buffer_b), 0);
        encoder.set_buffer(2, Some(&buffer_result), 0);
        
        let grid_size = MTLSize::new(size as u64, 1, 1);
        let threadgroup_size = MTLSize::new(
            pipeline.max_total_threads_per_threadgroup().min(64) as u64, 
            1, 
            1
        );
        encoder.dispatch_threads(grid_size, threadgroup_size);
    })?;
    
    // Read results
    let result_ptr = buffer_result.contents() as *const f32;
    let mut result = vec![0.0f32; size];
    
    unsafe {
        std::ptr::copy_nonoverlapping(result_ptr, result.as_mut_ptr(), size);
    }
    
    Ok(result)
} 