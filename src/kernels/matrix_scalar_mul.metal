#include <metal_stdlib>
using namespace metal;

kernel void matrix_scalar_multiply(device const float* A,
                                  constant float& scalar,
                                  device float* B,
                                  uint index [[thread_position_in_grid]])
{
    B[index] = scalar * A[index];
} 