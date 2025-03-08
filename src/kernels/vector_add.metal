#include <metal_stdlib>
using namespace metal;

kernel void add_arrays(device const float* a,
                       device const float* b,
                       device float* result,
                       uint index [[thread_position_in_grid]])
{
    result[index] = a[index] + b[index];
} 