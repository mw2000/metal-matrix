#include <metal_stdlib>
using namespace metal;

kernel void vector_dot_product(device const float* A,
                              device const float* B,
                              device float* result,
                              constant uint& size,
                              uint thread_id [[thread_position_in_threadgroup]],
                              uint threads_per_threadgroup [[threads_per_threadgroup]],
                              threadgroup float* shared_memory [[threadgroup(0)]])
{
    // First, each thread computes partial dot products
    float dot = 0.0;
    for (uint i = thread_id; i < size; i += threads_per_threadgroup) {
        dot += A[i] * B[i];
    }
    
    // Store in shared memory
    shared_memory[thread_id] = dot;
    
    // Synchronize threads
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Reduction in shared memory
    for (uint stride = threads_per_threadgroup / 2; stride > 0; stride >>= 1) {
        if (thread_id < stride) {
            shared_memory[thread_id] += shared_memory[thread_id + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    // Write the result
    if (thread_id == 0) {
        *result = shared_memory[0];
    }
} 