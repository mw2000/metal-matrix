use anyhow::{Context, Result};
use metal::*;
use std::fs;

/// Manages the Metal context including device and command queue
pub struct MetalContext {
    pub device: Device,
    pub command_queue: CommandQueue,
}

impl MetalContext {
    /// Create a new Metal context with the default system device
    pub fn new() -> Result<Self> {
        let device = Device::system_default().context("No Metal device found")?;
        let command_queue = device.new_command_queue();
        
        Ok(Self {
            device,
            command_queue,
        })
    }
    
    /// Load a Metal kernel from a file
    pub fn load_kernel(&self, file_path: &str, function_name: &str) -> Result<ComputePipelineState> {
        let source = fs::read_to_string(file_path)
            .context(format!("Failed to read kernel file: {}", file_path))?;
            
        let library = self.device
            .new_library_with_source(&source, &CompileOptions::new())
            .map_err(|e| anyhow::anyhow!("Failed to create library from source: {} - {}", file_path, e))?;
            
        let kernel = library
            .get_function(function_name, None)
            .map_err(|e| anyhow::anyhow!("Failed to get function: {} - {}", function_name, e))?;
            
        let pipeline = self.device
            .new_compute_pipeline_state_with_function(&kernel)
            .map_err(|e| anyhow::anyhow!("Failed to create compute pipeline: {} - {}", function_name, e))?;
            
        Ok(pipeline)
    }
    
    /// Create a new buffer with data
    pub fn new_buffer_with_data<T: Copy>(&self, data: &[T]) -> Buffer {
        let size = (data.len() * std::mem::size_of::<T>()) as u64;
        self.device.new_buffer_with_data(
            unsafe { std::mem::transmute(data.as_ptr()) },
            size,
            MTLResourceOptions::StorageModeShared,
        )
    }
    
    /// Create a new empty buffer
    pub fn new_buffer<T>(&self, count: usize) -> Buffer {
        let size = (count * std::mem::size_of::<T>()) as u64;
        self.device.new_buffer(
            size,
            MTLResourceOptions::StorageModeShared,
        )
    }
    
    /// Execute a compute operation and wait for completion
    pub fn execute_compute<F>(&self, encoder_setup: F) -> Result<()> 
    where
        F: FnOnce(&ComputeCommandEncoderRef),
    {
        let command_buffer = self.command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        
        encoder_setup(&encoder);
        
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();
        
        Ok(())
    }
} 