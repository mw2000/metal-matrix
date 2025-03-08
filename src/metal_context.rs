/*!
 * # Metal Context
 *
 * This module provides a wrapper around the Metal API for GPU computation.
 *
 * The `MetalContext` struct encapsulates the Metal device and command queue,
 * and provides methods for loading kernels, creating buffers, and executing
 * compute operations.
 */

use anyhow::{Context, Result};
use metal::*;
use std::fs;

/// Manages the Metal context including device and command queue.
///
/// This struct is the central point for interacting with the Metal GPU.
/// It handles device initialization, kernel loading, buffer creation,
/// and command execution.
pub struct MetalContext {
    /// The Metal device (GPU) to use for computation
    pub device: Device,

    /// The command queue for submitting work to the GPU
    pub command_queue: CommandQueue,
}

impl MetalContext {
    /// Create a new Metal context with the default system device.
    ///
    /// # Returns
    ///
    /// A `Result` containing the new `MetalContext` or an error if no Metal device is found.
    ///
    /// # Example
    ///
    /// ```
    /// use metal_matrix::MetalContext;
    ///
    /// let context = MetalContext::new().expect("Failed to create Metal context");
    /// ```
    pub fn new() -> Result<Self> {
        let device = Device::system_default().context("No Metal device found")?;
        let command_queue = device.new_command_queue();

        Ok(Self {
            device,
            command_queue,
        })
    }

    /// Load a Metal kernel from a file.
    ///
    /// This method reads a Metal shader file, compiles it, and creates a compute pipeline.
    ///
    /// # Arguments
    ///
    /// * `file_path` - Path to the Metal shader file
    /// * `function_name` - Name of the kernel function to load
    ///
    /// # Returns
    ///
    /// A `Result` containing the compute pipeline state or an error if loading fails.
    pub fn load_kernel(
        &self,
        file_path: &str,
        function_name: &str,
    ) -> Result<ComputePipelineState> {
        let source = fs::read_to_string(file_path)
            .context(format!("Failed to read kernel file: {}", file_path))?;

        let library = self
            .device
            .new_library_with_source(&source, &CompileOptions::new())
            .map_err(|e| {
                anyhow::anyhow!(
                    "Failed to create library from source: {} - {}",
                    file_path,
                    e
                )
            })?;

        let kernel = library
            .get_function(function_name, None)
            .map_err(|e| anyhow::anyhow!("Failed to get function: {} - {}", function_name, e))?;

        let pipeline = self
            .device
            .new_compute_pipeline_state_with_function(&kernel)
            .map_err(|e| {
                anyhow::anyhow!(
                    "Failed to create compute pipeline: {} - {}",
                    function_name,
                    e
                )
            })?;

        Ok(pipeline)
    }

    /// Create a new buffer with data.
    ///
    /// # Arguments
    ///
    /// * `data` - Slice of data to copy into the buffer
    ///
    /// # Returns
    ///
    /// A Metal buffer containing the data.
    ///
    /// # Type Parameters
    ///
    /// * `T` - Type of data to store in the buffer (must be `Copy`)
    pub fn new_buffer_with_data<T: Copy>(&self, data: &[T]) -> Buffer {
        let size = std::mem::size_of_val(data) as u64;
        self.device.new_buffer_with_data(
            unsafe { std::mem::transmute::<*const T, *const std::ffi::c_void>(data.as_ptr()) },
            size,
            MTLResourceOptions::StorageModeShared,
        )
    }

    /// Create a new empty buffer.
    ///
    /// # Arguments
    ///
    /// * `count` - Number of elements to allocate space for
    ///
    /// # Returns
    ///
    /// An empty Metal buffer with space for `count` elements of type `T`.
    ///
    /// # Type Parameters
    ///
    /// * `T` - Type of data the buffer will store
    pub fn new_buffer<T>(&self, count: usize) -> Buffer {
        let size = (count * std::mem::size_of::<T>()) as u64;
        self.device
            .new_buffer(size, MTLResourceOptions::StorageModeShared)
    }

    /// Execute a compute operation and wait for completion.
    ///
    /// This method creates a command buffer and encoder, calls the provided setup function,
    /// and then commits and waits for the command to complete.
    ///
    /// # Arguments
    ///
    /// * `encoder_setup` - Function that sets up the compute command encoder
    ///
    /// # Returns
    ///
    /// A `Result` indicating success or failure.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use metal_matrix::MetalContext;
    ///
    /// let context = MetalContext::new().unwrap();
    /// let pipeline = context.load_kernel("path/to/kernel.metal", "kernel_function").unwrap();
    ///
    /// context.execute_compute(|encoder| {
    ///     encoder.set_compute_pipeline_state(&pipeline);
    ///     // Set buffers, dispatch threads, etc.
    /// }).unwrap();
    /// ```
    pub fn execute_compute<F>(&self, encoder_setup: F) -> Result<()>
    where
        F: FnOnce(&ComputeCommandEncoderRef),
    {
        let command_buffer = self.command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        encoder_setup(encoder);

        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        Ok(())
    }
}
