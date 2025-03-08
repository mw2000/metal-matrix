//
// Matrix Scalar Multiplication Kernel
//
// This kernel performs scalar multiplication of a matrix.
// Each thread processes one element of the matrix.
//
// Parameters:
// - A: Input matrix
// - scalar: Scalar value to multiply by
// - B: Output matrix (result of scalar * A)
// - index: Thread position in the grid (one thread per matrix element)
//

#include <metal_stdlib>
using namespace metal;

kernel void matrix_scalar_multiply(device const float* A,
                                  constant float& scalar,
                                  device float* B,
                                  uint index [[thread_position_in_grid]])
{
    B[index] = scalar * A[index];
} 