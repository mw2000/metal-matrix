//
// Matrix Addition Kernel
//
// This kernel performs element-wise addition of two matrices.
// Each thread processes one element of the matrices.
//
// Parameters:
// - A: First input matrix
// - B: Second input matrix
// - C: Output matrix (result of A + B)
// - index: Thread position in the grid (one thread per matrix element)
//

#include <metal_stdlib>
using namespace metal;

kernel void matrix_add(device const float* A,
                      device const float* B,
                      device float* C,
                      uint index [[thread_position_in_grid]])
{
    C[index] = A[index] + B[index];
} 