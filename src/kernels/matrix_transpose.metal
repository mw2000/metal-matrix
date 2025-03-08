//
// Matrix Transpose Kernel
//
// This kernel performs matrix transposition.
// Each thread processes one element of the matrix.
//
// Parameters:
// - A: Input matrix (rows × cols)
// - B: Output matrix (cols × rows)
// - rows: Number of rows in the input matrix
// - cols: Number of columns in the input matrix
// - position: 2D thread position in the grid
//

#include <metal_stdlib>
using namespace metal;

kernel void matrix_transpose(device const float* A,
                            device float* B,
                            constant uint& rows,
                            constant uint& cols,
                            uint2 position [[thread_position_in_grid]])
{
    uint row = position.y;
    uint col = position.x;
    
    // Ensure we're within bounds
    if (row < rows && col < cols) {
        // B[col, row] = A[row, col]
        B[col * rows + row] = A[row * cols + col];
    }
} 