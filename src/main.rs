use metal::*;
use std::fs;

fn main() {
    // Setup Metal
    let device = Device::system_default().expect("No device found");
    let command_queue = device.new_command_queue();

    // Load kernel
    let source = fs::read_to_string("src/kernels/kernel.metal").unwrap();
    let library = device
        .new_library_with_source(&source, &CompileOptions::new())
        .unwrap();
    let kernel = library.get_function("add_arrays", None).unwrap();
    let pipeline = device
        .new_compute_pipeline_state_with_function(&kernel)
        .unwrap();

    // Create buffers
    let size = 1024;
    let a: Vec<f32> = (0..size).map(|x| x as f32).collect();
    let b: Vec<f32> = (0..size).map(|x| (x * 2) as f32).collect();
    let buffer_a = device.new_buffer_with_data(
        unsafe { std::mem::transmute(a.as_ptr()) },
        (size * std::mem::size_of::<f32>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let buffer_b = device.new_buffer_with_data(
        unsafe { std::mem::transmute(b.as_ptr()) },
        (size * std::mem::size_of::<f32>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let buffer_result = device.new_buffer(
        (size * std::mem::size_of::<f32>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    // Encode and dispatch
    let command_buffer = command_queue.new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipeline);
    encoder.set_buffer(0, Some(&buffer_a), 0);
    encoder.set_buffer(1, Some(&buffer_b), 0);
    encoder.set_buffer(2, Some(&buffer_result), 0);
    let grid_size = MTLSize::new(size as u64, 1, 1);
    let threadgroup_size = MTLSize::new(64, 1, 1); // Tune this based on device
    encoder.dispatch_threads(grid_size, threadgroup_size);
    encoder.end_encoding();

    command_buffer.commit();
    command_buffer.wait_until_completed();

    // Check results
    let result = unsafe { *(buffer_result.contents() as *const f32) };
    println!("Result[0] = {}", result); // Should be 0.0
}