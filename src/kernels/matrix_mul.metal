//
// Matrix Multiplication Kernel
//
// This kernel performs matrix multiplication: C = A * B
// Each thread computes one element of the output matrix.
//
// Parameters:
// - A: First input matrix (M × K)
// - B: Second input matrix (K × N)
// - C: Output matrix (M × N)
// - M: Number of rows in A
// - N: Number of columns in B
// - K: Number of columns in A / rows in B
// - position: 2D thread position in the grid
//

#include <metal_stdlib>
using namespace metal;

// Matrix multiplication kernel
// A: M x K matrix
// B: K x N matrix
// C: M x N matrix (result)
kernel void matrix_multiply(device const float* A,
                           device const float* B,
                           device float* C,
                           constant uint& M,
                           constant uint& N,
                           constant uint& K,
                           uint2 position [[thread_position_in_grid]])
{
    // Get global row and column position
    uint row = position.y;
    uint col = position.x;
    
    // Ensure we're within bounds
    if (row < M && col < N) {
        // Compute the dot product of row of A and column of B
        float sum = 0.0f;
        for (uint i = 0; i < K; i++) {
            sum += A[row * K + i] * B[i * N + col];
        }
        
        // Store the result
        C[row * N + col] = sum;
    }
} 