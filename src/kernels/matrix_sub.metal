#include <metal_stdlib>
using namespace metal;

kernel void matrix_subtract(device const float* A,
                           device const float* B,
                           device float* C,
                           uint index [[thread_position_in_grid]])
{
    C[index] = A[index] - B[index];
} 